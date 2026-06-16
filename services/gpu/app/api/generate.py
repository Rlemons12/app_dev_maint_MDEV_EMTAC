from __future__ import annotations

import gc
import os
import time
import traceback
from typing import Optional

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

    # Existing client compatibility
    max_tokens: int = 128

    # Preferred HuggingFace-style name.
    # If provided, this takes priority over max_tokens.
    max_new_tokens: Optional[int] = None

    temperature: float = 0.7
    top_p: float = 0.95
    num_beams: int = 1

    # These are sent by the EMTAC adapter.
    # The service will honor return_full_text=False by returning only new tokens.
    return_full_text: bool = False
    echo: bool = False
    strip_prompt: bool = True


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


def _resolve_max_new_tokens(req: GenerateRequest) -> int:
    """
    Resolve generation length.

    max_new_tokens is preferred. max_tokens is kept for backward compatibility.
    """

    raw_value = req.max_new_tokens if req.max_new_tokens is not None else req.max_tokens

    try:
        value = int(raw_value)
    except Exception:
        value = 128

    if value <= 0:
        value = 128

    return value


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

    prompt = str(req.prompt or "")
    max_input_tokens = int(os.getenv("GPU_MAX_INPUT_TOKENS", "1536"))
    max_new_tokens = _resolve_max_new_tokens(req)

    gpu_info(
        f"Generate request received | model={model_name} "
        f"prompt_chars={len(prompt)} max_new_tokens={max_new_tokens} "
        f"beams={req.num_beams} temp={req.temperature} top_p={req.top_p} "
        f"return_full_text={req.return_full_text} echo={req.echo} strip_prompt={req.strip_prompt}",
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
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=max_input_tokens,
        )

        input_len = int(inputs["input_ids"].shape[-1])

        gpu_debug(
            f"Tokenized | input_tokens={input_len} "
            f"max_input_tokens={max_input_tokens} "
            f"tokenization_s={time.time() - t0:.3f}",
            rid,
        )

        if input_len >= max_input_tokens:
            gpu_warning(
                f"Prompt was truncated by tokenizer | input_tokens={input_len} "
                f"max_input_tokens={max_input_tokens} prompt_chars={len(prompt)}",
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
        # Generate guarded
        # -------------------------------------------------
        t0 = time.time()

        def _do_generate():
            gen_kwargs = {
                "max_new_tokens": max_new_tokens,
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

        if outputs is None:
            raise RuntimeError("model.generate returned None")

        if not hasattr(outputs, "shape"):
            raise RuntimeError(
                f"model.generate returned unsupported output type: {type(outputs).__name__}"
            )

        total_output_tokens = int(outputs.shape[-1])

        # -------------------------------------------------
        # Decode only newly generated tokens
        # -------------------------------------------------
        t0 = time.time()

        output_ids = outputs[0]

        if output_ids.shape[-1] > input_len:
            generated_ids = output_ids[input_len:]
        else:
            generated_ids = output_ids.new_empty((0,), dtype=output_ids.dtype)

        generated_token_count = int(generated_ids.shape[-1])

        if req.return_full_text or req.echo:
            # Backward-compatible mode if a caller explicitly wants the full text.
            text = tokenizer.decode(output_ids, skip_special_tokens=True).strip()
            decode_mode = "full_text"
        else:
            # Correct chatbot mode: return only the answer/newly generated text.
            text = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
            decode_mode = "generated_only"

        decode_s = time.time() - t0

        gpu_debug(
            f"Decode complete | mode={decode_mode} chars={len(text)} "
            f"input_tokens={input_len} total_output_tokens={total_output_tokens} "
            f"generated_tokens={generated_token_count} decode_s={decode_s:.3f}",
            rid,
        )

        if not text:
            gpu_warning(
                f"Generated text is empty | model={model_name} "
                f"input_tokens={input_len} total_output_tokens={total_output_tokens} "
                f"generated_tokens={generated_token_count} max_new_tokens={max_new_tokens}",
                rid,
            )

        total_s = time.time() - t_start

        gpu_info(
            f"Generate request finished | model={model_name} total_s={total_s:.3f}",
            rid,
        )

        return {
            "text": text,
            "generated_text": text,
            "model": model_name,
            "device": response_device,
            "request_id": rid,
            "timing": {
                "total_s": round(total_s, 3),
                "gen_s": round(gen_s, 3),
                "decode_s": round(decode_s, 3),
            },
            "input_tokens": input_len,
            "output_tokens": generated_token_count,
            "generated_tokens": generated_token_count,
            "total_output_tokens": total_output_tokens,
            "decode_mode": decode_mode,
            "mode": {
                "num_beams": num_beams,
                "do_sample": do_sample,
                "return_full_text": bool(req.return_full_text),
                "echo": bool(req.echo),
                "strip_prompt": bool(req.strip_prompt),
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