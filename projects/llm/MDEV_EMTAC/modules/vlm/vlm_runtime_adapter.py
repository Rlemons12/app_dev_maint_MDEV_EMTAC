"""
vlm_runtime_adapter.py

DROP-IN REPLACEMENT — Dual Logger (App logger + GPU logger)

Uses BOTH:
- modules.configuration.logging_config (app/orchestrator-facing, request_id aware)
- app.config.gpu_logger (GPU/runtime-facing, request_id aware)

Design:
- App logger (info_id/debug_id/...) for high-level lifecycle + service-facing messages
- GPU logger (gpu_info/gpu_debug/...) for VRAM snapshots, device_map, performance, CUDA/OOM

Also:
- Lazy-loads VLM on first use
- Registers model+processor in GPU_MODELS (so eviction/unload works)
- Uses GPU_MODELS guarded_generate to avoid unload overlap
- Supports image->markdown, pdf->markdown, describe_image
"""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForImageTextToText

# Optional PDF support
try:
    import fitz  # PyMuPDF
    _HAVE_PYMUPDF = True
except Exception:
    fitz = None  # type: ignore[assignment]
    _HAVE_PYMUPDF = False

# App logger (Flask-safe request_id, rotation, unicode safe)
from modules.configuration.log_config import (
    info_id,
    debug_id,
    warning_id,
    error_id,
    log_timed_operation,
)

# GPU logger (FastAPI-safe request_id, dashboard forwarder)
from app.config.gpu_logger import (
    gpu_info,
    gpu_debug,
    gpu_warning,
    gpu_error,
    gpu_snapshot,
    gpu_phase,
    gpu_oom,
    gpu_shard_info,
)

from app.models.model_manager import GPU_MODELS


# ---------------------------------------------------------
# Helpers
# ---------------------------------------------------------
def _bytes_to_gb(x: int) -> float:
    return float(x) / (1024 ** 3)


def _infer_devices_from_hf_map(hf_device_map: Any) -> List[str]:
    devices: set[str] = set()
    if isinstance(hf_device_map, dict):
        for v in hf_device_map.values():
            if isinstance(v, int):
                devices.add(f"cuda:{v}")
            elif isinstance(v, str):
                s = v.strip().lower()
                if s.isdigit():
                    devices.add(f"cuda:{int(s)}")
                else:
                    devices.add(s)
    return sorted(devices)


def _primary_device_from_hf_map(hf_device_map: Any) -> torch.device:
    if not torch.cuda.is_available():
        return torch.device("cpu")

    devices = _infer_devices_from_hf_map(hf_device_map)
    for d in devices:
        if d.startswith("cuda:"):
            return torch.device(d)

    return torch.device("cuda:0")


def _safe_to_device(x: Any, device: torch.device):
    try:
        if isinstance(x, torch.Tensor):
            return x.to(device)
    except Exception:
        pass
    return x


def _log_vram(prefix: str, request_id=None):
    """
    Log VRAM to GPU logger (primary) and a light summary to app logger (secondary).
    """
    if not torch.cuda.is_available():
        gpu_debug(f"{prefix} | cpu-only", request_id)
        debug_id(f"[VLM][VRAM] {prefix} | cpu-only", request_id)
        return

    parts: List[str] = []
    for i in range(torch.cuda.device_count()):
        free_b, total_b = torch.cuda.mem_get_info(i)
        alloc = torch.cuda.memory_allocated(i)
        reserved = torch.cuda.memory_reserved(i)

        parts.append(
            f"cuda:{i} free={_bytes_to_gb(free_b):.2f}G "
            f"alloc={_bytes_to_gb(alloc):.2f}G resv={_bytes_to_gb(reserved):.2f}G "
            f"total={_bytes_to_gb(total_b):.2f}G"
        )

    # GPU logger gets full detail
    gpu_info(f"{prefix} | VRAM | " + " | ".join(parts), request_id)

    # App logger gets a compact single-line marker (keeps app.log readable)
    info_id(f"[VLM][VRAM] {prefix} | gpus={torch.cuda.device_count()}", request_id)


# ---------------------------------------------------------
# Adapter
# ---------------------------------------------------------
class VLMRuntimeAdapter:
    """
    Runtime-only adapter:
    - Loads model/processor lazily
    - Registers inside GPU_MODELS
    - Uses dual logging
    """

    def __init__(
        self,
        model_path: str,
        *,
        model_name: str = "nu_markdown",
        dtype: torch.dtype = torch.bfloat16,
        max_new_tokens: int = 384,
        offline: bool = True,
    ):
        self.model_name = (model_name or "nu_markdown").lower().strip()
        self.model_path = (model_path or "").strip()
        self.max_new_tokens = int(max_new_tokens)
        self.dtype = dtype
        self.offline = bool(offline)

        if not self.model_path:
            raise RuntimeError("model_path cannot be empty")

        if not Path(self.model_path).exists():
            raise RuntimeError(f"Model path does not exist: {self.model_path}")

        if self.offline:
            os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
            os.environ.setdefault("HF_HUB_OFFLINE", "1")

        # App-level lifecycle
        info_id(
            f"[VLM] Adapter initialized | name={self.model_name} path={self.model_path} "
            f"max_new_tokens={self.max_new_tokens}"
        )

        # GPU-level lifecycle
        gpu_info(
            f"VLM adapter init | name={self.model_name} path={self.model_path} "
            f"max_new_tokens={self.max_new_tokens} dtype={self.dtype}"
        )

    # ---------------------------------------------------------
    # Lazy load
    # ---------------------------------------------------------
    def _ensure_loaded(self, request_id=None) -> Tuple[Any, Any]:
        name = self.model_name

        if name in GPU_MODELS.models and name in GPU_MODELS.tokenizers:
            return GPU_MODELS.models[name], GPU_MODELS.tokenizers[name]

        info_id(f"[VLM] Loading model | name={name}", request_id)
        gpu_info(f"Loading VLM | name={name}", request_id)

        _log_vram("before-vlm-load", request_id)
        gpu_snapshot("before-vlm-load", request_id)

        # Respect GPUModelManager locks if present
        critical_lock = getattr(GPU_MODELS, "_critical_lock", None)
        load_lock = getattr(GPU_MODELS, "_load_lock", None)

        class _Noop:
            def __enter__(self):  # noqa
                return self
            def __exit__(self, exc_type, exc, tb):  # noqa
                return False

        crit_ctx = critical_lock if critical_lock is not None else _Noop()
        load_ctx = load_lock if load_lock is not None else _Noop()

        with crit_ctx:
            with load_ctx:
                # re-check after lock
                if name in GPU_MODELS.models and name in GPU_MODELS.tokenizers:
                    return GPU_MODELS.models[name], GPU_MODELS.tokenizers[name]

                with log_timed_operation("vlm_model_load", request_id):
                    t0 = time.time()

                    try:
                        processor = AutoProcessor.from_pretrained(
                            self.model_path,
                            local_files_only=True,
                            trust_remote_code=True,
                        )

                        model = AutoModelForImageTextToText.from_pretrained(
                            self.model_path,
                            device_map="auto",
                            torch_dtype=self.dtype if torch.cuda.is_available() else torch.float32,
                            low_cpu_mem_usage=True,
                            local_files_only=True,
                            trust_remote_code=True,
                        )
                        model.eval()

                        hf_map = getattr(model, "hf_device_map", {}) if torch.cuda.is_available() else {}
                        devices = _infer_devices_from_hf_map(hf_map)
                        sharded = len([d for d in devices if d.startswith("cuda:")]) > 1

                        # Register for eviction/unload/status
                        GPU_MODELS.models[name] = model
                        GPU_MODELS.tokenizers[name] = processor
                        GPU_MODELS.model_meta[name] = {
                            "kind": "vision",
                            "type": "image_text_to_text",
                            "device": "sharded" if sharded else (devices[0] if devices else "cpu"),
                            "devices": devices if devices else (["cpu"] if not torch.cuda.is_available() else ["cuda:0"]),
                            "sharded": sharded,
                            "hf_device_map": hf_map,
                            "path": self.model_path,
                        }

                        if devices:
                            gpu_shard_info(name, devices, request_id)

                        dt = time.time() - t0
                        gpu_phase("vlm_load", dt, request_id)

                        info_id(f"[VLM] Model loaded | name={name} sharded={sharded}", request_id)
                        gpu_info(f"VLM loaded | name={name} sharded={sharded} devices={devices}", request_id)

                        _log_vram("after-vlm-load", request_id)
                        gpu_snapshot("after-vlm-load", request_id)

                        # Optional manager introspection
                        try:
                            GPU_MODELS.log_loaded_models()
                        except Exception:
                            pass

                        return model, processor

                    except RuntimeError as e:
                        # OOM often comes as RuntimeError
                        msg = str(e)
                        gpu_error(f"VLM load failed | name={name} err={msg}", request_id)
                        error_id(f"[VLM] Load failed | name={name} err={msg}", request_id)
                        if "out of memory" in msg.lower():
                            gpu_oom(model=name, device="auto", context="vlm_load", request_id=request_id)
                        raise

                    except Exception as e:
                        gpu_error(f"VLM load failed | name={name} err={e}", request_id)
                        error_id(f"[VLM] Load failed | name={name} err={e}", request_id)
                        raise

    # ---------------------------------------------------------
    # Generate (guarded)
    # ---------------------------------------------------------
    def _generate(
        self,
        *,
        prompt: str,
        images: Optional[List[Image.Image]] = None,
        max_tokens: Optional[int] = None,
        request_id=None,
    ) -> str:
        images = images or []
        max_tokens = int(max_tokens or self.max_new_tokens)

        model, processor = self._ensure_loaded(request_id=request_id)

        hf_map = getattr(model, "hf_device_map", {}) if torch.cuda.is_available() else {}
        primary_device = _primary_device_from_hf_map(hf_map)

        # Build a Qwen-VL style message; if template not present, fallback
        content: List[Dict[str, Any]] = [{"type": "image"} for _ in images]
        content.append({"type": "text", "text": prompt})
        messages = [{"role": "user", "content": content}]

        try:
            text = processor.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        except Exception:
            text = prompt

        inputs = processor(
            text=[text],
            images=images if images else None,
            padding=True,
            return_tensors="pt",
        )
        inputs = {k: _safe_to_device(v, primary_device) for k, v in inputs.items()}

        def _run() -> str:
            with log_timed_operation("vlm_generate", request_id):
                t0 = time.time()

                _log_vram("before-generate", request_id)
                gpu_snapshot("before-generate", request_id)
                gpu_info(
                    f"VLM generate start | name={self.model_name} "
                    f"max_new_tokens={max_tokens} device={primary_device}",
                    request_id,
                )

                try:
                    with torch.no_grad():
                        output_ids = model.generate(
                            **inputs,
                            max_new_tokens=max_tokens,
                            do_sample=False,
                            use_cache=True,
                        )

                    out_text = processor.batch_decode(output_ids, skip_special_tokens=True)[0]

                    dt = time.time() - t0

                    # Token metrics (best-effort)
                    new_tokens = 0
                    try:
                        in_len = int(inputs["input_ids"].shape[-1])  # type: ignore[index]
                        out_len = int(output_ids.shape[-1])
                        new_tokens = max(out_len - in_len, 0)
                    except Exception:
                        pass

                    tps = (new_tokens / dt) if (dt > 0 and new_tokens > 0) else 0.0

                    gpu_phase("vlm_generate", dt, request_id)
                    gpu_info(
                        f"VLM generate done | seconds={dt:.2f} new_tokens={new_tokens} tps={tps:.2f} chars={len(out_text)}",
                        request_id,
                    )

                    info_id(
                        f"[VLM] Generation complete | seconds={dt:.2f} chars={len(out_text)}",
                        request_id,
                    )

                    _log_vram("after-generate", request_id)
                    gpu_snapshot("after-generate", request_id)

                    return out_text

                except RuntimeError as e:
                    msg = str(e)
                    gpu_error(f"VLM generate failed | err={msg}", request_id)
                    error_id(f"[VLM] Generate failed | err={msg}", request_id)
                    if "out of memory" in msg.lower():
                        gpu_oom(model=self.model_name, device=str(primary_device), context="vlm_generate", request_id=request_id)
                    raise

        # Guard against unload overlap, if manager supports it
        guarded = getattr(GPU_MODELS, "guarded_generate", None)
        if callable(guarded):
            return guarded(self.model_name, _run)

        # Fallback (should not happen in your system)
        warning_id("[VLM] GPU_MODELS.guarded_generate not found; running unguarded", request_id)
        gpu_warning("guarded_generate missing; running unguarded", request_id)
        return _run()

    # ---------------------------------------------------------
    # Public API
    # ---------------------------------------------------------
    def describe_image(self, image_path: str, request_id=None) -> str:
        img = Image.open(image_path).convert("RGB")
        return self._generate(
            prompt="Describe the image in detail.",
            images=[img],
            request_id=request_id,
        )

    def image_to_markdown(
        self,
        *,
        image_path: str,
        request_id=None,
        max_side: int = 1024,
        prompt: str = "Extract the full document text as clean markdown.",
        max_tokens: Optional[int] = None,
    ) -> str:
        img = Image.open(image_path).convert("RGB")

        # Resize safety (keeps inference stable)
        w, h = img.size
        if max(w, h) > max_side:
            scale = max_side / float(max(w, h))
            img = img.resize((int(w * scale), int(h * scale)), Image.BILINEAR)

        return self._generate(
            prompt=prompt,
            images=[img],
            max_tokens=max_tokens,
            request_id=request_id,
        )

    def pdf_to_markdown(
        self,
        *,
        pdf_path: str,
        max_pages: Optional[int] = None,
        dpi: int = 120,
        max_side: int = 1024,
        prompt: str = "Extract the full document text as clean markdown.",
        max_tokens: Optional[int] = None,
        request_id=None,
    ) -> str:
        if not _HAVE_PYMUPDF:
            raise RuntimeError("PyMuPDF (fitz) required for PDF support. Install: pip install pymupdf")

        doc = fitz.open(pdf_path)  # type: ignore[union-attr]
        total_pages = len(doc)
        pages_to_process = min(max_pages, total_pages) if max_pages else total_pages

        info_id(f"[VLM][PDF] Start | pages={pages_to_process}/{total_pages} path={pdf_path}", request_id)
        gpu_info(f"VLM PDF start | pages={pages_to_process}/{total_pages} path={pdf_path}", request_id)

        parts: List[str] = []

        try:
            for i in range(pages_to_process):
                page_no = i + 1
                t0 = time.time()

                gpu_info(f"PDF render | page={page_no}/{pages_to_process}", request_id)
                page = doc.load_page(i)

                zoom = dpi / 72.0
                mat = fitz.Matrix(zoom, zoom)  # type: ignore[union-attr]
                pix = page.get_pixmap(matrix=mat)

                img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)

                w, h = img.size
                if max(w, h) > max_side:
                    scale = max_side / float(max(w, h))
                    img = img.resize((int(w * scale), int(h * scale)), Image.BILINEAR)

                md = self._generate(
                    prompt=prompt,
                    images=[img],
                    max_tokens=max_tokens,
                    request_id=request_id,
                )

                dt = time.time() - t0
                gpu_phase("vlm_pdf_page", dt, request_id)
                info_id(f"[VLM][PDF] Page done | page={page_no}/{pages_to_process} seconds={dt:.2f}", request_id)
                gpu_info(f"PDF page done | page={page_no}/{pages_to_process} seconds={dt:.2f}", request_id)

                parts.append(md)

        finally:
            try:
                doc.close()
            except Exception:
                pass

        info_id("[VLM][PDF] Complete", request_id)
        gpu_info("VLM PDF complete", request_id)

        return "\n\n".join(parts)

    def status(self) -> Dict[str, Any]:
        meta = GPU_MODELS.model_meta.get(self.model_name, {})
        return {
            "name": self.model_name,
            "loaded": self.model_name in GPU_MODELS.models,
            "model_path": self.model_path,
            "meta": meta,
        }