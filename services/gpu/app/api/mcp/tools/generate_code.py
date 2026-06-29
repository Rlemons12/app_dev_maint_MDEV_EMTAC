from __future__ import annotations

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from app.config.gpu_logger import gpu_error, gpu_info, gpu_warning
from app.models.model_manager import GPU_MODELS

router = APIRouter()


class MCPGenerateCodeRequest(BaseModel):
    prompt: str = Field(..., min_length=1)
    model: str = "qwen"
    max_tokens: int = 512
    temperature: float = 0.2
    top_p: float = 0.95
    system_prompt: str | None = "You are a helpful coding assistant."


@router.post("/mcp/tools/generate_code")
def generate_code(request: MCPGenerateCodeRequest):
    try:
        model_name = request.model.strip().lower()

        gpu_info(
            f"[GENERATE_CODE] Request received | model={model_name} "
            f"max_tokens={request.max_tokens} temperature={request.temperature} top_p={request.top_p} "
            f"prompt_len={len(request.prompt)}"
        )

        model, tokenizer = GPU_MODELS.ensure_generation_model_loaded(model_name)

        def _run():
            gpu_info(f"[GENERATE_CODE] _run start | model={model_name}")

            messages = []

            if request.system_prompt:
                messages.append(
                    {"role": "system", "content": request.system_prompt}
                )

            messages.append(
                {"role": "user", "content": request.prompt}
            )

            gpu_info(f"[GENERATE_CODE] Building chat template | model={model_name}")

            try:
                prompt_text = tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                )
            except Exception as exc:
                gpu_error(f"[GENERATE_CODE] apply_chat_template failed | model={model_name} | error={exc}")
                raise

            gpu_info(
                f"[GENERATE_CODE] Chat template built | model={model_name} "
                f"chars={len(prompt_text)} preview={prompt_text[:300]!r}"
            )

            try:
                inputs = tokenizer(prompt_text, return_tensors="pt")
            except Exception as exc:
                gpu_error(f"[GENERATE_CODE] Tokenization failed | model={model_name} | error={exc}")
                raise

            meta = GPU_MODELS.model_meta.get(model_name, {})
            device = meta.get("device", "cpu")
            sharded = bool(meta.get("sharded", False))

            gpu_info(
                f"[GENERATE_CODE] Tokenization complete | model={model_name} "
                f"device={device} sharded={sharded} "
                f"input_shape={tuple(inputs['input_ids'].shape)}"
            )

            if tokenizer.eos_token_id is None:
                gpu_warning(f"[GENERATE_CODE] eos_token_id is None | model={model_name}")

            if isinstance(device, str) and device.startswith("cuda:") and not sharded:
                gpu_info(f"[GENERATE_CODE] Moving inputs to device | model={model_name} device={device}")
                inputs = {k: v.to(device) for k, v in inputs.items()}

            gpu_info(f"[GENERATE_CODE] Starting model.generate | model={model_name}")

            try:
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=request.max_tokens,
                    temperature=request.temperature,
                    top_p=request.top_p,
                    do_sample=request.temperature > 0,
                    pad_token_id=tokenizer.eos_token_id,
                )
            except Exception as exc:
                gpu_error(f"[GENERATE_CODE] model.generate failed | model={model_name} | error={exc}")
                raise

            gpu_info(
                f"[GENERATE_CODE] model.generate complete | model={model_name} "
                f"output_shape={tuple(outputs.shape) if hasattr(outputs, 'shape') else 'unknown'}"
            )

            try:
                generated_ids = outputs[0][inputs["input_ids"].shape[1]:]
                text = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
            except Exception as exc:
                gpu_error(f"[GENERATE_CODE] decode failed | model={model_name} | error={exc}")
                raise

            gpu_info(
                f"[GENERATE_CODE] decode complete | model={model_name} "
                f"text_len={len(text)} preview={text[:300]!r}"
            )
            return text

        text = GPU_MODELS.guarded_generate(model_name, _run)

        gpu_info(
            f"[GENERATE_CODE] Success | model={model_name} text_len={len(text)}"
        )

        return {
            "tool": "generate_code",
            "model": model_name,
            "text": text,
            "finish_reason": "stop",
        }

    except Exception as exc:
        gpu_error(f"[GENERATE_CODE] Request failed | error={exc}")
        raise HTTPException(status_code=500, detail=str(exc))