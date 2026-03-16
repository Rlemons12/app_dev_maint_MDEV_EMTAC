from __future__ import annotations

import gc
import os
import time
import traceback

import torch
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from app.config.gpu_logger import (
    get_request_id,
    gpu_debug,
    gpu_error,
    gpu_info,
    gpu_warning,
)
from app.models.model_manager import GPU_MODELS

router = APIRouter()


class GenerateRequest(BaseModel):
    model: str
    prompt: str
    max_tokens: int = 128
    temperature: float = 0.7
    top_p: float = 0.95
    num_beams: int = 1


# ---------------------------------------------------------
# Best-effort cleanup after OOM
# ---------------------------------------------------------
def _safe_cuda_cleanup(rid: str | None = None):
    try:
        gc.collect()
    except Exception:
        pass

    if torch.cuda.is_available():
        try:
            torch.cuda.empty_cache()
            gpu_warning("CUDA cache cleared", rid)
        except Exception:
            pass


def _safe_model_meta(model_name: str) -> dict:
    return GPU_MODELS.model_meta.get((model_name or "").lower().strip(), {}) or {}


@router.post("/generate")
def generate_text(req: GenerateRequest):
    rid = get_request_id()
    t_start = time.time()

    model_name = (req.model or "").lower().strip()
    if not model_name:
        raise HTTPException(
            status_code=400,
            detail={
                "error": "Model name is required",
                "request_id": rid,
            },
        )

    max_input_tokens = int(os.getenv("GPU_MAX_INPUT_TOKENS", "1536"))

    gpu_info(
        f"Generate request received | model={model_name} "
        f"prompt_chars={len(req.prompt)} max_new_tokens={req.max_tokens} "
        f"beams={req.num_beams} temp={req.temperature} top_p={req.top_p}",
        rid,
    )

    try:
        # -------------------------------------------------
        # Ensure model ready via unified manager contract
        # -------------------------------------------------
        t0 = time.time()
        model, tokenizer = GPU_MODELS.ensure_generation_model_loaded(model_name)
        meta = _safe_model_meta(model_name)

        is_sharded = bool(meta.get("sharded", False))
        devices = meta.get("devices", []) or []
        device_label = str(meta.get("device", "cpu"))

        gpu_debug(
            f"Model ready | model={model_name} sharded={is_sharded} "
            f"device={device_label} devices={devices} "
            f"ready_s={time.time() - t0:.3f}",
            rid,
        )

        # -------------------------------------------------
        # Tokenize on CPU
        # -------------------------------------------------
        t0 = time.time()
        inputs = tokenizer(
            req.prompt,
            return_tensors="pt",
            truncation=True,
            max_length=max_input_tokens,
        )
        input_len = int(inputs["input_ids"].shape[-1])

        gpu_debug(
            f"Tokenized | input_tokens={input_len} tokenization_s={time.time() - t0:.3f}",
            rid,
        )

        # -------------------------------------------------
        # Move tensors only for non-sharded GPU models
        # -------------------------------------------------
        if not is_sharded and device_label.startswith("cuda"):
            try:
                inputs = {k: v.to(device_label) for k, v in inputs.items()}
            except Exception as e:
                gpu_warning(
                    f"Failed moving inputs to device {device_label}: {e}",
                    rid,
                )
                raise
            response_device = device_label
        elif not is_sharded:
            response_device = device_label
        else:
            response_device = "sharded"

        # -------------------------------------------------
        # Generation mode
        # -------------------------------------------------
        num_beams = max(1, int(req.num_beams))

        if num_beams > 1:
            do_sample = False
            temperature = None
            top_p = None
            gpu_warning(
                f"Beam search enabled (beams={num_beams}) -> sampling disabled",
                rid,
            )
        else:
            do_sample = float(req.temperature) > 0.0
            temperature = float(req.temperature) if do_sample else None
            top_p = float(req.top_p) if do_sample else None

        # -------------------------------------------------
        # Generate (guarded)
        # -------------------------------------------------
        t0 = time.time()

        def _do_generate():
            gen_kwargs = {
                "max_new_tokens": int(req.max_tokens),
                "num_beams": num_beams,
                "do_sample": do_sample,
                "use_cache": True,
            }

            if do_sample:
                if temperature is not None:
                    gen_kwargs["temperature"] = temperature
                if top_p is not None:
                    gen_kwargs["top_p"] = top_p

            if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
                gen_kwargs["pad_token_id"] = tokenizer.eos_token_id

            return model.generate(**inputs, **gen_kwargs)

        with torch.no_grad():
            outputs = GPU_MODELS.guarded_generate(model_name, _do_generate)

        gen_s = time.time() - t0
        out_tokens = int(outputs.shape[-1])

        gpu_info(
            f"Generation completed | model={model_name} device={response_device} "
            f"in_tokens={input_len} out_tokens={out_tokens} gen_s={gen_s:.3f}",
            rid,
        )

        # -------------------------------------------------
        # Decode
        # -------------------------------------------------
        t0 = time.time()
        text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        decode_s = time.time() - t0

        gpu_debug(
            f"Decode complete | chars={len(text)} decode_s={decode_s:.3f}",
            rid,
        )

        total_s = time.time() - t_start
        gpu_info(
            f"Generate request finished | model={model_name} total_s={total_s:.3f}",
            rid,
        )

        return {
            "text": text,
            "model": model_name,
            "device": response_device,
            "request_id": rid,
            "timing": {
                "total_s": round(total_s, 3),
                "gen_s": round(gen_s, 3),
                "decode_s": round(decode_s, 3),
            },
            "input_tokens": input_len,
            "output_tokens": out_tokens,
            "mode": {
                "num_beams": num_beams,
                "do_sample": do_sample,
            },
            "sharded": is_sharded,
        }

    except torch.cuda.OutOfMemoryError as e:
        gpu_error(f"CUDA OOM | model={model_name} error={e}", rid)

        _safe_cuda_cleanup(rid)

        try:
            GPU_MODELS.evict_model(model_name)
            gpu_warning(f"Evicted model after OOM | model={model_name}", rid)
        except Exception as evict_err:
            gpu_warning(
                f"Post-OOM eviction failed | model={model_name} err={evict_err}",
                rid,
            )

        raise HTTPException(
            status_code=503,
            detail={
                "error": "CUDA out of memory during generation",
                "model": model_name,
                "request_id": rid,
                "hint": (
                    "Reduce max_tokens, set num_beams=1, "
                    "reduce prompt/context length, or load a smaller model."
                ),
            },
        )

    except Exception as e:
        tb = traceback.format_exc()
        gpu_error(f"Generation FAILED | model={model_name} error={e}", rid)
        gpu_error(tb, rid)

        raise HTTPException(
            status_code=500,
            detail={
                "error": "GPU generation failed",
                "model": model_name,
                "request_id": rid,
            },
        )