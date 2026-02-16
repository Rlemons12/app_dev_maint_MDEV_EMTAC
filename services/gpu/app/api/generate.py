# app/api/generate.py

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
import time
import traceback
import torch

from app.models.model_manager import GPU_MODELS
from app.config.gpu_logger import (
    gpu_info,
    gpu_debug,
    gpu_warning,
    gpu_error,
    get_request_id,
)

router = APIRouter()


# ---------------------------------------------------------
# Request model
# ---------------------------------------------------------
class GenerateRequest(BaseModel):
    model: str
    prompt: str
    max_tokens: int = 128
    temperature: float = 0.7
    top_p: float = 0.95
    num_beams: int = 4


# ---------------------------------------------------------
# Generate endpoint
# ---------------------------------------------------------
@router.post("/generate")
def generate_text(req: GenerateRequest):
    rid = get_request_id()
    start_ts = time.time()

    model_name = req.model.lower().strip()

    gpu_info(
        f"Generate request received | model={model_name} "
        f"prompt_len={len(req.prompt)} "
        f"max_tokens={req.max_tokens}",
        rid,
    )

    try:
        # -------------------------------------------------
        # Load model
        # -------------------------------------------------
        t0 = time.time()
        model, tokenizer = GPU_MODELS.get_generation_model(model_name)
        gpu_debug(
            f"Model loaded | model={model_name} device={model.device} "
            f"load_time={time.time() - t0:.3f}s",
            rid,
        )

        # -------------------------------------------------
        # Tokenization
        # -------------------------------------------------
        t0 = time.time()
        inputs = tokenizer(
            req.prompt,
            return_tensors="pt",
            truncation=True,
        ).to(model.device)

        gpu_debug(
            f"Tokenization complete | input_ids={inputs['input_ids'].shape} "
            f"time={time.time() - t0:.3f}s",
            rid,
        )

        # -------------------------------------------------
        # Generation (GUARDED)
        # -------------------------------------------------
        t0 = time.time()
        with torch.no_grad():
            outputs = GPU_MODELS.guarded_generate(
                model_name,
                lambda: model.generate(
                    **inputs,
                    max_new_tokens=req.max_tokens,
                    temperature=req.temperature,
                    top_p=req.top_p,
                    num_beams=req.num_beams,
                ),
            )

        gen_time = time.time() - t0
        gpu_info(
            f"Generation completed | model={model_name} "
            f"tokens_out={outputs.shape[-1]} "
            f"time={gen_time:.3f}s",
            rid,
        )

        # -------------------------------------------------
        # Decode
        # -------------------------------------------------
        t0 = time.time()
        text = tokenizer.decode(outputs[0], skip_special_tokens=True)

        gpu_debug(
            f"Decode complete | chars={len(text)} "
            f"time={time.time() - t0:.3f}s",
            rid,
        )

        total_time = time.time() - start_ts
        gpu_info(
            f"Generate request finished | model={model_name} "
            f"total_time={total_time:.3f}s",
            rid,
        )

        return {
            "text": text,
            "model": model_name,
            "device": str(model.device),
            "request_id": rid,
            "timing": {
                "total_s": round(total_time, 3),
                "generation_s": round(gen_time, 3),
            },
        }

    except Exception as e:
        # -------------------------------------------------
        # HARD SYNC CUDA ERRORS (CRITICAL)
        # -------------------------------------------------
        if torch.cuda.is_available():
            try:
                torch.cuda.synchronize()
            except Exception:
                pass

        tb = traceback.format_exc()

        gpu_error(
            f"Generation FAILED | model={model_name} "
            f"error={e.__class__.__name__}: {e}",
            rid,
        )

        gpu_error(tb, rid)

        raise HTTPException(
            status_code=500,
            detail={
                "error": "GPU generation failed",
                "model": model_name,
                "request_id": rid,
            },
        )
