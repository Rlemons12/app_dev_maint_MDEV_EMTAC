from __future__ import annotations

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

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
        model, tokenizer = GPU_MODELS.ensure_generation_model_loaded(model_name)

        def _run():
            messages = []

            if request.system_prompt:
                messages.append(
                    {"role": "system", "content": request.system_prompt}
                )

            messages.append(
                {"role": "user", "content": request.prompt}
            )

            prompt_text = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )

            inputs = tokenizer(prompt_text, return_tensors="pt")

            meta = GPU_MODELS.model_meta.get(model_name, {})
            device = meta.get("device", "cpu")
            sharded = bool(meta.get("sharded", False))

            if isinstance(device, str) and device.startswith("cuda:") and not sharded:
                inputs = {k: v.to(device) for k, v in inputs.items()}

            outputs = model.generate(
                **inputs,
                max_new_tokens=request.max_tokens,
                temperature=request.temperature,
                top_p=request.top_p,
                do_sample=request.temperature > 0,
                pad_token_id=tokenizer.eos_token_id,
            )

            generated_ids = outputs[0][inputs["input_ids"].shape[1]:]
            text = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
            return text

        text = GPU_MODELS.guarded_generate(model_name, _run)

        return {
            "tool": "generate_code",
            "model": model_name,
            "text": text,
            "finish_reason": "stop",
        }

    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))