# modules/rag_core/answer_generator.py

from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional

from modules.configuration.log_config import (
    info_id,
    error_id,
    warning_id,
    debug_id,
    with_request_id,
)
from modules.services.ai_models_service import AIModelsService



# ============================================================
# Base Interface
# ============================================================

class BaseAnswerGenerator(ABC):
    """Interface for RAG answer generation components."""

    @abstractmethod
    def generate_answer(
            self,
            question: str,
            context: str,
            request_id: Optional[str] = None,
            *,
            system_instruction: Optional[str] = None,
            **kwargs: Any,
    ) -> Dict[str, Any]:
        """
        Generate a user-facing answer using an LLM.

        Args:
            question:
                The user's natural-language question.

            context:
                Retrieved RAG context. May be empty or unrelated.

            request_id:
                Optional request identifier for logging and tracing.

            system_instruction:
                Optional instruction that defines assistant behavior.
                Implementations MUST NOT override core EMTAC response policy.

        Returns:
            Dict with at least:
                - 'answer': str
                - 'model_name': str
        """
        raise NotImplementedError


# ============================================================
# DB-Configured Answer Generator (Primary)
# ============================================================

class DBConfiguredAnswerGenerator(BaseAnswerGenerator):
    """
    Uses AIModelsService for all LLM access.

    - The active model is read from the DB by AIModelsService
    - No direct loading of models occurs here
    - No fallbacks
    - No OpenAI support
    """

    def __init__(self):
        current = AIModelsService.get_current_model_name()
        info_id(f"[AnswerGen] Using AI model (DB): {current}")

    @with_request_id
    def generate_answer(
            self,
            question: str,
            context: str,
            request_id: Optional[str] = None,
            **kwargs,
    ) -> Dict[str, Any]:

        debug_id("[AnswerGen] Calling AIModelsService.answer()", request_id)

        try:
            # Delegate entirely to AIModelsService
            # Assistant behavior is defined ONLY in _build_rag_prompt()
            answer = AIModelsService.answer(
                question=question,
                context=context,
                request_id=request_id,
                **kwargs,
            )

            return {
                "answer": answer,
                "model_name": AIModelsService.get_current_model_name(
                    request_id=request_id
                ),
            }

        except Exception as e:
            error_id(f"[AnswerGen] Error generating answer: {e}", request_id)

            return {
                "answer": "An error occurred while generating the answer.",
                "model_name": AIModelsService.get_current_model_name(
                    request_id=request_id
                ),
            }


# ============================================================
# OPTIONAL: Local HuggingFace generator (not used by RAG)
# ============================================================

class LocalHFAnswerGenerator(BaseAnswerGenerator):
    """
    Optional manual-only mechanism for testing local HF models directly.

    This is NOT used by RAG now that AIModelsService controls all LLM usage.
    """

    def __init__(self, model_path: str):
        from transformers import AutoTokenizer, AutoModelForCausalLM
        import torch

        info_id(f"[LocalHFAnswerGenerator] Loading HF model: {model_path}")

        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            device_map="auto",
        )
        self.model_path = model_path

    def generate_answer(
        self,
        question: str,
        context: str,
        request_id: Optional[str] = None,
        **kwargs,
    ) -> Dict[str, Any]:

        prompt = (
            "You are an EMTAC maintenance assistant.\n"
            "Use ONLY the provided context.\n\n"
            f"CONTEXT:\n{context}\n\n"
            f"QUESTION:\n{question}\n"
            "ANSWER:\n"
        )

        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=256,
            temperature=0.3,
        )

        decoded = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

        if "ANSWER:" in decoded:
            decoded = decoded.split("ANSWER:")[-1].strip()

        return {"answer": decoded}


# ============================================================
# Exports
# ============================================================

__all__ = [
    "BaseAnswerGenerator",
    "DBConfiguredAnswerGenerator",
    "LocalHFAnswerGenerator",
]
