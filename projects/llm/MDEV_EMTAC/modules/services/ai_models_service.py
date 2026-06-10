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

        safe_question = (question or "").strip()
        safe_context = (context or "").strip()

        prompt = PromptProvider.build_prompt(
            purpose="default",
            question=safe_question,
            context=safe_context,
            request_id=request_id,
        )

        prompt = cls._reinforce_memory_prompt_if_needed(
            prompt=prompt,
            question=safe_question,
            context=safe_context,
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
            + prompt[-3500:]
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

            debug_id(
                "[AIModelsService] CLEANED MODEL OUTPUT START\n"
                + str(cleaned)[:4000]
                + "\n[AIModelsService] CLEANED MODEL OUTPUT END",
                request_id,
            )

            if not cleaned:
                warning_id(
                    "[AIModelsService] Cleaned model response was empty",
                    request_id,
                )
                return "No answer generated."

            cleaned_question = safe_question.strip().lower()
            cleaned_answer = str(cleaned or "").strip()

            if (
                cleaned_answer.strip().lower() == cleaned_question
                and str(raw or "").strip().lower() != cleaned_question
            ):
                recovered = cls._recover_answer_from_raw_output(
                    raw=raw,
                    question=safe_question,
                )

                debug_id(
                    "[AIModelsService] RECOVERED MODEL OUTPUT START\n"
                    + str(recovered)[:4000]
                    + "\n[AIModelsService] RECOVERED MODEL OUTPUT END",
                    request_id,
                )

                if recovered and recovered.strip().lower() != cleaned_question:
                    return recovered.strip()

                warning_id(
                    "[AIModelsService] Cleaned answer matched the user question; "
                    "raw output contained additional text but recovery failed.",
                    request_id,
                )

            return cleaned_answer

        except Exception as e:
            error_id(
                f"[AIModelsService] GPU generation failed: {e}",
                request_id,
                exc_info=True,
            )
            return "Error generating answer."

    @classmethod
    @with_request_id
    def generate_raw(
            cls,
            *,
            prompt: str,
            model: Optional[str] = None,
            max_new_tokens: int = 256,
            request_id: Optional[str] = None,
    ) -> str:
        """
        Low-level raw model generation.

        This bypasses PromptProvider and the default answer-generation prompt.
        Use this for classifiers, routers, and JSON-only internal tasks.
        """

        safe_prompt = (prompt or "").strip()

        if not safe_prompt:
            return ""

        backend = ModelsConfig.get_execution_backend("ai")

        if backend != "gpu_service":
            error_id(
                f"[AIModelsService] Unsupported AI execution backend for raw generation: {backend}",
                request_id,
            )
            return ""

        gpu = cls._get_gpu_adapter()

        if not gpu:
            error_id("[AIModelsService] GPU service unavailable for raw generation", request_id)
            return ""

        model_name = model or cls.get_current_model_name(request_id=request_id)

        debug_id(
            f"[AIModelsService] RAW GPU generate model={model_name} "
            f"prompt_len={len(safe_prompt)} max_new_tokens={max_new_tokens}",
            request_id,
        )

        try:
            raw = gpu.generate(
                prompt=safe_prompt,
                model=model_name,
                max_new_tokens=max_new_tokens,
            )

            return str(raw or "").strip()

        except TypeError:
            raw = gpu.generate(
                prompt=safe_prompt,
                model=model_name,
            )

            return str(raw or "").strip()

        except Exception as exc:
            error_id(
                f"[AIModelsService] Raw GPU generation failed: {exc}",
                request_id,
                exc_info=True,
            )
            return ""

    @staticmethod
    def _reinforce_memory_prompt_if_needed(
        *,
        prompt: str,
        question: str,
        context: str,
    ) -> str:
        """
        Add a final high-priority memory instruction near the end of the prompt.

        Why:
            The DB prompt still says "answer from provided context."
            For recall questions, conversation memory is valid provided context,
            but the model may over-focus on retrieved document chunks.

        This keeps memory guidance close to FINAL ANSWER.
        """

        safe_prompt = str(prompt or "")
        safe_question = (question or "").strip()
        safe_context = str(context or "")

        if not safe_context:
            return safe_prompt

        has_memory = (
            "CONVERSATION MEMORY MODE ACTIVE" in safe_context
            or "Conversation memory evidence:" in safe_context
            or "Relevant long-term conversation memories:" in safe_context
            or "Rolling conversation summary:" in safe_context
            or "Recent conversation messages:" in safe_context
        )

        if not has_memory:
            return safe_prompt

        recall_question = bool(
            re.search(
                r"\b("
                r"what\s+(is|was)\s+my|"
                r"what\s+did\s+i\s+(say|tell|mention)|"
                r"what\s+was\s+my\s+test|"
                r"remember|"
                r"previously\s+said|"
                r"prior\s+conversation|"
                r"test\s+memory\s+phrase"
                r")\b",
                safe_question,
                flags=re.IGNORECASE,
            )
        )

        if not recall_question:
            return safe_prompt

        reinforcement = (
            "\n\nMEMORY RECALL OVERRIDE:\n"
            "The user is asking a conversation-memory recall question.\n"
            "Conversation memory inside the provided context is valid evidence.\n"
            "Answer from conversation memory first.\n"
            "Do not answer from unrelated retrieved document chunks.\n"
            "Do not say the answer is missing from context if it appears in conversation memory.\n"
            "For this question, give the remembered value directly and briefly.\n"
        )

        final_answer_marker = "FINAL ANSWER:"

        if final_answer_marker in safe_prompt:
            return safe_prompt.replace(
                final_answer_marker,
                reinforcement + "\n" + final_answer_marker,
                1,
            )

        return safe_prompt + reinforcement

    @staticmethod
    def _clean_generated_answer(raw: str) -> str:
        if not raw:
            return ""

        text = str(raw).strip()
        text = text.replace("\r\n", "\n").replace("\r", "\n")

        # Best case: model produced assistant turn. Keep the LAST assistant answer.
        assistant_matches = list(
            re.finditer(
                r"(^|\n)\s*assistant\s*:\s*",
                text,
                flags=re.IGNORECASE,
            )
        )

        if assistant_matches:
            last = assistant_matches[-1]
            candidate = text[last.end():].strip()
            candidate = AIModelsService._final_trim_answer(candidate)
            if candidate:
                return candidate

        # Only use FINAL ANSWER as a boundary. Do NOT use generic ANSWER:
        # because it can match prompt text and destroy assistant output.
        final_matches = list(
            re.finditer(
                r"(^|\n)\s*FINAL\s+ANSWER\s*:\s*",
                text,
                flags=re.IGNORECASE,
            )
        )

        if final_matches:
            last = final_matches[-1]
            candidate = text[last.end():].strip()
            candidate = AIModelsService._final_trim_answer(candidate)
            if candidate:
                return candidate

        # If prompt echo appears, remove everything through the last context end.
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

            # After context end, prefer assistant again.
            assistant_matches = list(
                re.finditer(
                    r"(^|\n)\s*assistant\s*:\s*",
                    text,
                    flags=re.IGNORECASE,
                )
            )

            if assistant_matches:
                last = assistant_matches[-1]
                candidate = text[last.end():].strip()
                candidate = AIModelsService._final_trim_answer(candidate)
                if candidate:
                    return candidate

            # Remove question/instruction blocks if they were echoed.
            text = re.sub(
                r"^\s*QUESTION\s*:\s*.*?(?=\n\s*(INSTRUCTIONS|FINAL\s+ANSWER|assistant\s*:|$))",
                "",
                text,
                flags=re.IGNORECASE | re.DOTALL,
            ).strip()

            text = re.sub(
                r"^\s*INSTRUCTIONS\s*:\s*.*?(?=\n\s*(FINAL\s+ANSWER|assistant\s*:|$))",
                "",
                text,
                flags=re.IGNORECASE | re.DOTALL,
            ).strip()

        if not text:
            return ""

        prompt_leak_starts = [
            "You are an EMTAC maintenance and engineering assistant",
            "GENERAL RULES",
            "USE OF CONTEXT",
            "AUTHORITATIVE INFORMATION",
            "OUTPUT FORMAT",
            "DOCUMENT MODE ACTIVE",
            "Document Mode rules",
            "Selected document profile",
            "Selected document chunk context",
            "Retrieved context",
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

        text = str(text).strip()
        text = text.replace("\r\n", "\n").replace("\r", "\n")

        # Remove leading role/label only.
        text = re.sub(
            r"^\s*(ASSISTANT|ANSWER|FINAL\s+ANSWER)\s*:\s*",
            "",
            text,
            flags=re.IGNORECASE,
        ).strip()

        # If a user/question block appears AFTER the answer, cut there.
        # Do NOT cut on assistant:, because assistant: is often where the answer starts.
        stop_patterns = [
            r"\n\s*USER\s*:",
            r"\n\s*SYSTEM\s*:",
            r"\n\s*QUESTION\s*:",
            r"\n\s*---\s*CONTEXT\s+START\s*---",
            r"\n\s*---\s*CONTEXT\s+END\s*---",
            r"\n\s*GENERAL\s+RULES\b",
            r"\n\s*USE\s+OF\s+CONTEXT\b",
            r"\n\s*AUTHORITATIVE\s+INFORMATION\b",
            r"\n\s*WHEN\s+CONTEXT\s+IS\s+INCOMPLETE\b",
            r"\n\s*OUTPUT\s+FORMAT\b",
            r"\n\s*DOCUMENT\s+MODE\s+ACTIVE\b",
            r"\n\s*Document\s+Mode\s+rules\s*:",
            r"\n\s*Selected\s+document\s+profile\s*:",
            r"\n\s*Selected\s+document\s+chunk\s+context\s*:",
            r"\n\s*Retrieved\s+context\s*:",
        ]

        cut_positions = []

        for pattern in stop_patterns:
            match = re.search(pattern, text, flags=re.IGNORECASE)
            if match:
                cut_positions.append(match.start())

        if cut_positions:
            text = text[:min(cut_positions)].strip()

        # Remove leftover leading user/question labels only if they are the first line.
        text = re.sub(
            r"^\s*(USER|SYSTEM|QUESTION)\s*:\s*.*?\n",
            "",
            text,
            flags=re.IGNORECASE | re.DOTALL,
        ).strip()

        text = re.sub(r"\n{3,}", "\n\n", text).strip()

        return text


    @staticmethod
    def _recover_answer_from_raw_output(
        *,
        raw: str,
        question: str = "",
    ) -> str:
        """
        Recovery path for cases where downstream cleanup accidentally returns
        the original user question instead of the generated answer.

        Common raw shape from local models:

            echoed profile / context lines
            user: Can you explain this?
            assistant: The selected document says ...

        In that case, the real answer is after the last assistant marker.
        """

        text = str(raw or "").strip()

        if not text:
            return ""

        text = text.replace("\r\n", "\n").replace("\r", "\n")
        safe_question = str(question or "").strip().lower()

        assistant_matches = list(
            re.finditer(
                r"(^|\n)\s*assistant\s*:\s*",
                text,
                flags=re.IGNORECASE,
            )
        )

        if assistant_matches:
            last = assistant_matches[-1]
            candidate = text[last.end():].strip()
            candidate = AIModelsService._final_trim_answer(candidate)

            if candidate and candidate.strip().lower() != safe_question:
                return candidate

        final_answer_matches = list(
            re.finditer(
                r"FINAL\s+ANSWER\s*:",
                text,
                flags=re.IGNORECASE,
            )
        )

        if final_answer_matches:
            last = final_answer_matches[-1]
            candidate = text[last.end():].strip()
            candidate = AIModelsService._final_trim_answer(candidate)

            if candidate and candidate.strip().lower() != safe_question:
                return candidate

        context_end_matches = list(
            re.finditer(
                r"---\s*CONTEXT\s+END\s*---",
                text,
                flags=re.IGNORECASE,
            )
        )

        if context_end_matches:
            last = context_end_matches[-1]
            candidate = text[last.end():].strip()

            candidate = re.sub(
                r"^\s*QUESTION\s*:\s*.*?(?=\n|$)",
                "",
                candidate,
                flags=re.IGNORECASE | re.DOTALL,
            ).strip()

            candidate = re.sub(
                r"^\s*INSTRUCTIONS\s*:\s*.*?(FINAL\s+ANSWER\s*:)?",
                "",
                candidate,
                flags=re.IGNORECASE | re.DOTALL,
            ).strip()

            candidate = AIModelsService._final_trim_answer(candidate)

            if candidate and candidate.strip().lower() != safe_question:
                return candidate

        return AIModelsService._final_trim_answer(text)