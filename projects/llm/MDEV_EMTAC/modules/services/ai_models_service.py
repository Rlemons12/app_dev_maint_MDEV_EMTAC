from typing import Optional
import re

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

    @classmethod
    def _get_gpu_adapter(cls) -> Optional[GPUServerAdapter]:
        if cls._gpu_adapter is None:
            cls._gpu_adapter = GPUServerAdapter()
        return cls._gpu_adapter if cls._gpu_adapter.is_available() else None

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

    @classmethod
    @with_request_id
    def answer(
        cls,
        question: str,
        context: str = "",
        request_id: Optional[str] = None,
    ) -> str:
        backend = ModelsConfig.get_execution_backend("ai")

        from modules.ai.prompts.prompt_provider import PromptProvider

        prompt = PromptProvider.build_prompt(
            purpose="default",
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
        debug_id(
            "[AIModelsService] FINAL PROMPT START\n"
            + prompt[-2500:]
            + "\n[AIModelsService] FINAL PROMPT END",
            request_id,
        )

        try:
            raw = gpu.generate(
                prompt=prompt,
                model=model_name,
            )
            debug_id(
                "[AIModelsService] RAW MODEL OUTPUT START\n"
                + str(raw)[:4000]
                + "\n[AIModelsService] RAW MODEL OUTPUT END",
                request_id,
            )
            if not raw:
                warning_id(
                    "[AIModelsService] Empty response from GPU service",
                    request_id,
                )
                return "No answer generated."

            cleaned = cls._clean_generated_answer(raw)

            if not cleaned:
                warning_id(
                    "[AIModelsService] Cleaned model response was empty",
                    request_id,
                )
                return "No answer generated."

            return cleaned

        except Exception as e:
            error_id(
                f"[AIModelsService] GPU generation failed: {e}",
                request_id,
            )
            return "Error generating answer."

    @staticmethod
    def _clean_generated_answer(raw: str) -> str:
        if not raw:
            return ""

        text = str(raw).strip()
        text = text.replace("\r\n", "\n").replace("\r", "\n")

        # -------------------------------------------------
        # 1. Prefer hard final-answer boundary
        # -------------------------------------------------
        marker_patterns = [
            r"FINAL\s+ANSWER\s*:",
            r"ANSWER\s*:",
        ]

        for pattern in marker_patterns:
            matches = list(re.finditer(pattern, text, flags=re.IGNORECASE))
            if matches:
                last = matches[-1]
                text = text[last.end():].strip()
                return AIModelsService._final_trim_answer(text)

        # -------------------------------------------------
        # 2. If no marker, remove echoed prompt through context end
        # -------------------------------------------------
        context_end_matches = list(
            re.finditer(
                r"---\s*CONTEXT\s+END\s*---",
                text,
                flags=re.IGNORECASE,
            )
        )

        if context_end_matches:
            last_context_end = context_end_matches[-1]
            text = text[last_context_end.end():].strip()

        # -------------------------------------------------
        # 3. If model only echoed prompt and gave no answer
        # -------------------------------------------------
        if not text:
            return ""

        # If the remainder still starts with prompt policy, reject it.
        prompt_leak_starts = [
            "You are an EMTAC maintenance and engineering assistant",
            "GENERAL RULES",
            "USE OF CONTEXT",
            "AUTHORITATIVE INFORMATION",
            "OUTPUT FORMAT",
        ]

        lowered = text.lower()
        for leak in prompt_leak_starts:
            if lowered.startswith(leak.lower()):
                return ""

        return AIModelsService._final_trim_answer(text)

    @staticmethod
    def _final_trim_answer(text: str) -> str:
        if not text:
            return ""

        stop_patterns = [
            r"\n\s*USER\s*:",
            r"\n\s*ASSISTANT\s*:",
            r"\n\s*SYSTEM\s*:",
            r"\n\s*QUESTION\s*:",
            r"\n\s*---\s*CONTEXT\s+START\s*---",
            r"\n\s*GENERAL\s+RULES\b",
            r"\n\s*USE\s+OF\s+CONTEXT\b",
            r"\n\s*AUTHORITATIVE\s+INFORMATION\b",
            r"\n\s*WHEN\s+CONTEXT\s+IS\s+INCOMPLETE\b",
            r"\n\s*OUTPUT\s+FORMAT\b",
        ]

        cut_positions = []
        for pattern in stop_patterns:
            match = re.search(pattern, text, flags=re.IGNORECASE)
            if match:
                cut_positions.append(match.start())

        if cut_positions:
            text = text[:min(cut_positions)].strip()

        text = re.sub(
            r"^(USER|ASSISTANT|SYSTEM|QUESTION|ANSWER|FINAL\s+ANSWER)\s*:\s*",
            "",
            text,
            flags=re.IGNORECASE,
        ).strip()

        text = re.sub(r"\n{3,}", "\n\n", text).strip()

        return text