from typing import Optional

from modules.runtime.gpu_service_adapter_emtac import GPUServerAdapter
from modules.configuration.log_config import (
    debug_id,
    info_id,
    warning_id,
    error_id,
    with_request_id,
)
from modules.ai.config.models_config import ModelsConfig


class AIModelsService:
    """
    High-level facade for AI text generation.

    Responsibilities:
    - Read active AI model from DB
    - Route generation through GPU service
    - Build prompts via DB-backed PromptProvider
    - Enforce clean, user-only output
    """

    _model_cache = {}
    _current_model_name: Optional[str] = None
    _gpu_adapter: Optional[GPUServerAdapter] = None

    # ---------------------------------------------------------
    # GPU Adapter
    # ---------------------------------------------------------
    @classmethod
    def _get_gpu_adapter(cls) -> Optional[GPUServerAdapter]:
        if cls._gpu_adapter is None:
            cls._gpu_adapter = GPUServerAdapter()
        return cls._gpu_adapter if cls._gpu_adapter.is_available() else None

    # ---------------------------------------------------------
    # Model Name Handling
    # ---------------------------------------------------------
    @classmethod
    @with_request_id
    def get_current_model_name(cls, request_id: Optional[str] = None) -> str:
        model_name = ModelsConfig.get_config_value(
            "ai", "CURRENT_MODEL", default="NoAIModel"
        )
        debug_id(
            f"[AIModelsService] Current AI model from DB: {model_name}",
            request_id,
        )
        return model_name

    @classmethod
    @with_request_id
    def set_current_model(
        cls,
        model_name: str,
        request_id: Optional[str] = None,
    ) -> bool:
        ok = ModelsConfig.set_current_ai_model(model_name)
        if ok:
            info_id(
                f"[AIModelsService] Updated CURRENT_MODEL (ai) to '{model_name}'",
                request_id,
            )
            cls._current_model_name = None
            cls._model_cache.clear()
        else:
            error_id(
                f"[AIModelsService] Failed to update CURRENT_MODEL (ai) → '{model_name}'",
                request_id,
            )
        return ok

    # ---------------------------------------------------------
    # Public RAG API
    # ---------------------------------------------------------
    @classmethod
    @with_request_id
    def answer(
        cls,
        question: str,
        context: str = "",
        request_id: Optional[str] = None,
    ) -> str:
        """
        Generate an answer using:
        - DB-selected prompt
        - DB-selected model
        - GPU service backend

        GUARANTEES:
        - No system prompt leakage
        - No context echo
        - Final-answer-only output
        """

        backend = ModelsConfig.get_execution_backend("ai")
        from modules.emtac_ai.prompts.prompt_provider import PromptProvider
        # -------------------------------------------------
        # Build DB-driven prompt
        # -------------------------------------------------
        prompt = PromptProvider.build_prompt(
            purpose="default",  # change to "rag" later if desired
            question=question,
            context=context,
            request_id=request_id,
        )

        if backend != "gpu_service":
            error_id(
                f"[AIModelsService] Unsupported AI execution backend: '{backend}'",
                request_id,
            )
            return "AI service unavailable."

        gpu = cls._get_gpu_adapter()
        if not gpu:
            error_id("[AIModelsService] GPU service unavailable", request_id)
            return "AI service unavailable."

        model_name = cls.get_current_model_name(request_id=request_id)

        debug_id(
            f"[AIModelsService] GPU generate "
            f"(model={model_name}, prompt_len={len(prompt)})",
            request_id,
        )

        try:
            raw = gpu.generate(
                prompt=prompt,
                model=model_name,
            )

            # -------------------------------------------------
            # HARD OUTPUT BOUNDARY (CRITICAL)
            # -------------------------------------------------
            # This guarantees ONLY the final answer is returned,
            # even if the model echoes the prompt or context.
            if not raw:
                warning_id(
                    "[AIModelsService] Empty response from GPU service",
                    request_id,
                )
                return "No answer generated."

            if "ANSWER:" in raw:
                return raw.split("ANSWER:", 1)[-1].strip()

            if "FINAL ANSWER" in raw.upper():
                return raw.split("FINAL ANSWER", 1)[-1].strip(":\n ")

            return raw.strip()

        except Exception as e:
            error_id(
                f"[AIModelsService] GPU generation failed: {e}",
                request_id,
            )
            return "Error generating answer."
