from __future__ import annotations

import gc
import time
from dataclasses import dataclass
from typing import List, Optional, TYPE_CHECKING

import torch

from app.config.gpu_logger import (
    gpu_info,
    gpu_debug,
    gpu_warning,
    gpu_phase,
    gpu_snapshot,
)

from app.config.gpu_service_config import GPU_SERVICE_CONFIG
from app.models.model_manager import GPU_MODELS
from app.services.vlm_service import VLM_SERVICE


# ============================================================
# GPU Contracts
# ============================================================

@dataclass(frozen=True)
class GPUPageResult:
    page_number: int
    markdown: str


# ============================================================
# GPU Orchestrator
# ============================================================

class GPUOrchestrator:
    """
    GPU-side orchestration layer.

    Responsibilities:
      - Ensure VLM model is ready
      - Run inference over page render images
      - Keep VRAM stable
      - Provide observability hooks

    Ownership rule:
      - GPUModelManager is the sole lifecycle owner of loaded model objects
      - GPUOrchestrator must NOT retain strong references to loaded models
    """

    PROMPT: str = "Transcribe this page into Markdown."

    def __init__(self):
        self._loaded_model_name: Optional[str] = None

    # ------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------

    def _clear_local_runtime_refs(self) -> None:
        try:
            gc.collect()
        except Exception:
            pass

        if torch.cuda.is_available():
            try:
                torch.cuda.empty_cache()
            except Exception:
                pass

    def _get_loaded_vision_pair(self, model_name: str, *, rid: str):
        model_name = (model_name or "").lower().strip()

        if model_name not in GPU_MODELS.models:
            raise RuntimeError(f"Vision model not loaded: {model_name}")

        if model_name not in GPU_MODELS.tokenizers:
            raise RuntimeError(f"Vision processor not loaded: {model_name}")

        model = GPU_MODELS.models[model_name]
        processor = GPU_MODELS.tokenizers[model_name]

        meta = GPU_MODELS.model_meta.get(model_name, {}) or {}

        gpu_debug(
            f"GPU live model fetched | model={model_name} "
            f"device={meta.get('device')} sharded={meta.get('sharded')} "
            f"devices={meta.get('devices')}",
            rid,
        )

        return model, processor

    # ------------------------------------------------------------
    # Ensure Model Ready
    # ------------------------------------------------------------

    def ensure_model_loaded(self, model_name: str, *, rid: str) -> None:
        model_name = (model_name or "").lower().strip()
        if not model_name:
            raise RuntimeError("Vision model name cannot be empty")

        if (
            self._loaded_model_name == model_name
            and model_name in GPU_MODELS.model_meta
            and model_name in GPU_MODELS.models
            and model_name in GPU_MODELS.tokenizers
        ):
            gpu_debug(
                f"GPU model already ready | model={model_name}",
                rid,
            )
            return

        t0 = time.time()

        gpu_info(
            f"GPU ensure model ready | requested={model_name}",
            rid,
        )

        GPU_MODELS.ensure_vision_model_loaded(model_name)

        self._loaded_model_name = model_name

        meta = GPU_MODELS.model_meta.get(model_name, {}) or {}

        gpu_debug(
            f"GPU model ready | model={model_name} "
            f"device={meta.get('device')} sharded={meta.get('sharded')} "
            f"devices={meta.get('devices')}",
            rid,
        )

        gpu_phase("GPU_ENSURE_MODEL", time.time() - t0, rid)

    # ------------------------------------------------------------
    # Inference Window
    # ------------------------------------------------------------

    def infer_window(
        self,
        *,
        window: "CPUPageWindow",
        model: str,
        max_new_tokens: int,
        rid: str,
    ) -> List[GPUPageResult]:

        t0 = time.time()

        if not window.page_items:
            gpu_warning("GPU infer_window | empty window.page_items", rid)
            return []

        self.ensure_model_loaded(model, rid=rid)

        vlm, processor = self._get_loaded_vision_pair(model, rid=rid)

        try:
            cfg_limit = int(getattr(GPU_SERVICE_CONFIG, "vlm_max_new_tokens", 0) or 0)
        except Exception:
            cfg_limit = 0

        effective_max_new_tokens = int(max_new_tokens)

        if cfg_limit > 0:
            effective_max_new_tokens = min(effective_max_new_tokens, cfg_limit)

        page_items = window.page_items
        first_page = window.page_numbers[0]
        last_page = window.page_numbers[-1]

        gpu_info(
            "GPU infer_window start | "
            f"pages={first_page}..{last_page} "
            f"count={len(page_items)} model={model} "
            f"max_new_tokens={effective_max_new_tokens}",
            rid,
        )

        enable_snapshots = bool(
            getattr(GPU_SERVICE_CONFIG, "enable_gpu_snapshots", True)
        )

        if enable_snapshots:
            gpu_snapshot("before_vlm", rid)

        results: List[GPUPageResult] = []

        use_batch = bool(getattr(GPU_SERVICE_CONFIG, "vlm_use_batch_infer", False))
        batch_size = int(getattr(GPU_SERVICE_CONFIG, "vlm_batch_pages", 1) or 1)

        if use_batch and batch_size > 1:
            image_paths = [str(item.render_path) for item in page_items]
            page_numbers = [item.page_number for item in page_items]

            t_batch = time.time()

            def _run_batch():
                return VLM_SERVICE.images_to_text(
                    model=vlm,
                    processor=processor,
                    image_paths=image_paths,
                    prompt=self.PROMPT,
                    max_new_tokens=effective_max_new_tokens,
                    request_id=rid,
                    max_batch_size=batch_size,
                )

            md_list = GPU_MODELS.guarded_generate(model, _run_batch)

            gpu_phase("GPU_BATCH_INFER", time.time() - t_batch, rid)

            for pn, md in zip(page_numbers, md_list):
                md = md or ""

                gpu_debug(
                    f"GPU page infer complete | page={pn} chars={len(md)}",
                    rid,
                )

                results.append(
                    GPUPageResult(
                        page_number=pn,
                        markdown=md,
                    )
                )

            self._clear_local_runtime_refs()

        else:
            for item in page_items:
                t_page = time.time()

                def _run_single():
                    return VLM_SERVICE.image_to_text(
                        model=vlm,
                        processor=processor,
                        image_path=str(item.render_path),
                        prompt=self.PROMPT,
                        max_new_tokens=effective_max_new_tokens,
                        request_id=rid,
                    )

                markdown = GPU_MODELS.guarded_generate(model, _run_single)
                markdown = markdown or ""

                gpu_debug(
                    f"GPU page infer complete | page={item.page_number} chars={len(markdown)}",
                    rid,
                )

                results.append(
                    GPUPageResult(
                        page_number=item.page_number,
                        markdown=markdown,
                    )
                )

                gpu_phase("GPU_PAGE_INFER", time.time() - t_page, rid)
                self._clear_local_runtime_refs()

        if enable_snapshots:
            gpu_snapshot("after_vlm", rid)

        gpu_phase("GPU_INFER_WINDOW_INNER", time.time() - t0, rid)

        gpu_info(f"GPU infer_window complete | pages={len(results)}", rid)

        return results


if TYPE_CHECKING:
    from app.orchestrators.cpu_orchestrator import CPUPageWindow