from __future__ import annotations

import os
import re
import time
from pathlib import Path
from typing import List, Dict, Any, Optional

import requests
from dotenv import load_dotenv

from modules.configuration.log_config import (
    debug_id,
    info_id,
    warning_id,
    error_id,
    get_request_id,
)


# ---------------------------------------------------------
# Load shared EMTAC environment
# ---------------------------------------------------------
DEFAULT_ENV_PATH = Path(r"E:\emtac\dev_env\.env")
ENV_PATH = Path(os.getenv("EMTAC_ENV_PATH", str(DEFAULT_ENV_PATH)))

if ENV_PATH.exists():
    load_dotenv(ENV_PATH, override=False)


class GPUServerAdapter:
    """
    Adapter for EMTAC GPU Service.

    Responsibilities:
        - Route embedding and generation requests to GPU service
        - Handle retries, timeouts, and fallback behavior
        - Keep main app free of model-loading concerns
        - Defensively clean GPU generation responses so prompt echo is not
          stored as the final chatbot answer

    Important:
        The best fix is inside the GPU service /generate route:
            generated_ids = outputs[0][input_token_count:]

        This adapter also sends:
            return_full_text=False
            max_new_tokens=<value>

        and then strips prompt echo as a safety net.
    """

    # ---------------------------------------------------------
    # Lifecycle
    # ---------------------------------------------------------

    def __init__(
        self,
        base_url: Optional[str] = None,
        timeout: int = 300,
        max_retries: int = 2,
        enabled: Optional[bool] = None,
    ):
        resolved_base_url = (
            base_url
            or os.getenv("SERVICE_GPU_BASE_URL")
            or os.getenv("GPU_SERVICE_URL")
            or "http://127.0.0.1:5051"
        )

        self.base_url = resolved_base_url.rstrip("/")
        self.timeout = timeout
        self.max_retries = max_retries
        self.enabled = enabled if enabled is not None else self._detect_service()

        rid = get_request_id()
        info_id(
            f"[GPU-ADAPTER] Initialized | enabled={self.enabled} | url={self.base_url}",
            rid,
        )

    # ---------------------------------------------------------
    # Internal helpers
    # ---------------------------------------------------------

    def _detect_service(self) -> bool:
        """Check if GPU service is reachable."""
        try:
            response = requests.get(f"{self.base_url}/health", timeout=5)
            return response.status_code == 200
        except Exception:
            return False

    def _post(self, endpoint: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        if not self.enabled:
            raise RuntimeError("GPU service is disabled or unreachable")

        url = f"{self.base_url}{endpoint}"
        rid = get_request_id()

        last_err: Optional[Exception] = None

        for attempt in range(1, self.max_retries + 1):
            try:
                debug_id(
                    f"[GPU-ADAPTER] POST {endpoint} attempt={attempt}",
                    rid,
                )

                response = requests.post(
                    url,
                    json=payload,
                    timeout=self.timeout,
                )

                response.raise_for_status()

                data = response.json()

                if not isinstance(data, dict):
                    raise RuntimeError(
                        f"GPU service returned non-dict JSON response: {type(data).__name__}"
                    )

                return data

            except Exception as exc:
                last_err = exc
                warning_id(
                    f"[GPU-ADAPTER] Attempt {attempt} failed: {exc}",
                    rid,
                )
                time.sleep(1)

        raise RuntimeError(f"GPU service request failed: {last_err}")

    # ---------------------------------------------------------
    # Public API
    # ---------------------------------------------------------

    def embed(
        self,
        texts: List[str],
        *,
        gpu_model: str,
        batch_size: int = 32,
        normalize: bool = True,
    ) -> List[List[float]]:
        """
        Call GPU embedding service.

        NOTE:
            - This is only called when backend == 'gpu_service'
            - gpu_model must already be resolved by ModelsConfig
        """

        if not self.enabled:
            raise RuntimeError("GPU service is disabled or unreachable")

        rid = get_request_id()

        payload = {
            "texts": texts,
            "model": gpu_model,
            "batch_size": batch_size,
            "normalize": normalize,
        }

        debug_id(
            f"[GPU-ADAPTER] POST /embed model={gpu_model} texts={len(texts)}",
            rid,
        )

        data = self._post("/embed", payload)

        embeddings = data.get("embeddings")

        if not embeddings:
            raise RuntimeError("GPU service returned no embeddings")

        return embeddings

    def generate(
        self,
        prompt: str,
        model: str = "qwen",
        max_new_tokens: int = 512,
        temperature: float = 0.2,
        top_p: float = 0.95,
    ) -> str:
        """
        Request text generation from GPU service.

        This method expects the GPU service to return only newly generated text.

        If the GPU service returns the full prompt anyway, this method strips
        the prompt and keeps only the final answer content.

        If the GPU service returns only a truncated prompt and no generated
        answer, this method raises. That prevents EMTAC from storing a bad
        prompt echo as the assistant answer.
        """

        rid = get_request_id()

        prompt = str(prompt or "")

        MODEL_MAP = {
            "TinyLlamaModel": "tinyllama",
            "QwenModel": "qwen",
            "MistralModel": "mistral",
            "GemmaModel": "gemma",
            None: "qwen",
        }

        gpu_model = MODEL_MAP.get(model, model)

        payload = {
            "prompt": prompt,
            "model": gpu_model,

            # Backward compatibility with the existing GPU service.
            "max_tokens": max_new_tokens,

            # Preferred HuggingFace-style generation parameter.
            # The GPU service should use this as max_new_tokens, not max_length.
            "max_new_tokens": max_new_tokens,

            "temperature": temperature,
            "top_p": top_p,

            # Tell the GPU service not to return the prompt.
            # If unsupported, the adapter still cleans the response below.
            "return_full_text": False,
            "echo": False,
            "strip_prompt": True,
        }

        debug_id(
            f"[GPU-ADAPTER] POST /generate "
            f"model={gpu_model} max_new_tokens={max_new_tokens} prompt_len={len(prompt)}",
            rid,
        )

        try:
            data = self._post("/generate", payload)

            raw_text = self._extract_generation_text(data)

            if not raw_text:
                raise RuntimeError(
                    f"GPU service returned empty generation result. keys={list(data.keys())}"
                )

            cleaned_text = self._clean_generation_text(
                raw_text=raw_text,
                prompt=prompt,
                request_id=rid,
            )

            if not cleaned_text:
                raise RuntimeError(
                    "GPU service returned prompt echo/truncated prompt but no generated answer. "
                    "Fix the GPU service /generate route to decode only generated tokens, "
                    "using outputs[0][input_token_count:] and max_new_tokens."
                )

            debug_id(
                f"[GPU-ADAPTER] Generation cleaned "
                f"raw_len={len(raw_text)} cleaned_len={len(cleaned_text)} "
                f"prompt_echo_detected={self._looks_like_prompt_echo(raw_text, prompt)}",
                rid,
            )

            return cleaned_text

        except Exception as exc:
            error_id(
                f"[GPU-ADAPTER] Generation failed (model={gpu_model}): {exc}",
                rid,
                exc_info=True,
            )
            raise

    def is_available(self) -> bool:
        self.enabled = self._detect_service()
        return self.enabled

    # ---------------------------------------------------------
    # Generation response cleanup
    # ---------------------------------------------------------

    @staticmethod
    def _extract_generation_text(data: Dict[str, Any]) -> str:
        """
        Extract generated text from common GPU service response shapes.

        Supported:
            {"text": "..."}
            {"answer": "..."}
            {"generated_text": "..."}
            {"response": "..."}
            {"output": "..."}
            {"outputs": [{"generated_text": "..."}]}
            {"outputs": ["..."]}
        """

        if not isinstance(data, dict):
            return ""

        for key in (
            "text",
            "answer",
            "generated_text",
            "response",
            "output",
            "completion",
        ):
            value = data.get(key)

            if isinstance(value, str) and value.strip():
                return value.strip()

        outputs = data.get("outputs")

        if isinstance(outputs, list) and outputs:
            first = outputs[0]

            if isinstance(first, str):
                return first.strip()

            if isinstance(first, dict):
                for key in (
                    "text",
                    "answer",
                    "generated_text",
                    "response",
                    "output",
                    "completion",
                ):
                    value = first.get(key)

                    if isinstance(value, str) and value.strip():
                        return value.strip()

        choices = data.get("choices")

        if isinstance(choices, list) and choices:
            first = choices[0]

            if isinstance(first, dict):
                message = first.get("message")

                if isinstance(message, dict):
                    content = message.get("content")

                    if isinstance(content, str) and content.strip():
                        return content.strip()

                text = first.get("text")

                if isinstance(text, str) and text.strip():
                    return text.strip()

        return ""

    @classmethod
    def _clean_generation_text(
        cls,
        *,
        raw_text: str,
        prompt: str,
        request_id: Optional[str],
    ) -> str:
        """
        Clean GPU generation result.

        Cases handled:
            1. GPU service correctly returns only the generated answer.
            2. GPU service returns prompt + answer.
            3. GPU service returns prompt ending at FINAL ANSWER: + answer.
            4. GPU service returns prompt echo only; return empty so caller raises.
        """

        text = str(raw_text or "").strip()
        prompt = str(prompt or "").strip()

        if not text:
            return ""

        original_text = text

        # -----------------------------------------------------
        # Exact full-prompt prefix removal
        # -----------------------------------------------------
        if prompt and text.startswith(prompt):
            text = text[len(prompt):].strip()

            debug_id(
                "[GPU-ADAPTER] Removed exact prompt prefix from generation response.",
                request_id,
            )

        # -----------------------------------------------------
        # If response contains the final-answer marker, prefer
        # content after the last marker.
        # -----------------------------------------------------
        final_answer_markers = [
            "FINAL ANSWER:",
            "Final Answer:",
            "Final answer:",
            "ANSWER:",
            "Answer:",
        ]

        for marker in final_answer_markers:
            if marker in text:
                candidate = text.rsplit(marker, 1)[-1].strip()

                if candidate:
                    text = candidate
                    break

        # -----------------------------------------------------
        # Remove common echoed prompt wrappers.
        # -----------------------------------------------------
        text = cls._strip_prompt_headers(text)

        # -----------------------------------------------------
        # If the cleaned text still looks like the original prompt,
        # do not return it as an answer.
        # -----------------------------------------------------
        if cls._looks_like_prompt_echo(text, prompt):
            warning_id(
                "[GPU-ADAPTER] Cleaned generation still looks like prompt echo. "
                f"raw_len={len(original_text)} cleaned_len={len(text)}",
                request_id,
            )
            return ""

        # -----------------------------------------------------
        # If the model only returned instruction headers with no answer,
        # do not store them as the answer.
        # -----------------------------------------------------
        if cls._looks_like_instruction_only_output(text):
            warning_id(
                "[GPU-ADAPTER] Generation output looks like instruction-only prompt echo.",
                request_id,
            )
            return ""

        return text.strip()

    @staticmethod
    def _strip_prompt_headers(text: str) -> str:
        """
        Remove obvious prompt scaffolding if it remains at the start.
        """

        value = str(text or "").strip()

        if not value:
            return ""

        # Remove everything before and including a final answer marker if present.
        marker_patterns = [
            r"(?is).*?\bFINAL\s+ANSWER\s*:\s*",
            r"(?is).*?\bFinal\s+Answer\s*:\s*",
        ]

        for pattern in marker_patterns:
            new_value = re.sub(pattern, "", value).strip()

            if new_value != value:
                value = new_value
                break

        # If the response starts with common prompt headers, remove leading
        # scaffold lines until a non-scaffold line is found.
        scaffold_prefixes = (
            "You are an EMTAC maintenance",
            "GENERAL RULES",
            "USE OF CONTEXT",
            "AUTHORITATIVE INFORMATION",
            "WHEN CONTEXT IS INCOMPLETE",
            "WHEN INFORMATION IS UNKNOWN",
            "CLARIFICATION",
            "OUTPUT FORMAT",
            "--- CONTEXT START ---",
            "--- CONTEXT END ---",
            "DOCUMENT MODE ACTIVE",
            "Selected document:",
            "Document Mode rules:",
            "Conversation memory:",
            "Selected document context:",
            "QUESTION:",
            "INSTRUCTIONS:",
        )

        lines = value.splitlines()

        if lines and any(lines[0].strip().startswith(prefix) for prefix in scaffold_prefixes):
            cleaned_lines: List[str] = []
            skipping = True

            for line in lines:
                stripped = line.strip()

                if skipping:
                    if not stripped:
                        continue

                    is_scaffold = (
                        any(stripped.startswith(prefix) for prefix in scaffold_prefixes)
                        or stripped.startswith("- ")
                        or stripped.startswith("If ")
                        or stripped.startswith("Do NOT ")
                        or stripped.startswith("Do not ")
                        or stripped.startswith("Then:")
                    )

                    if is_scaffold:
                        continue

                    skipping = False
                    cleaned_lines.append(line)
                else:
                    cleaned_lines.append(line)

            cleaned = "\n".join(cleaned_lines).strip()

            if cleaned:
                value = cleaned

        return value.strip()

    @staticmethod
    def _looks_like_prompt_echo(text: str, prompt: str) -> bool:
        """
        Detect whether text is mostly the original prompt or prompt scaffold.
        """

        value = str(text or "").strip()
        prompt_value = str(prompt or "").strip()

        if not value:
            return False

        if prompt_value and value == prompt_value:
            return True

        if prompt_value and len(value) > 300:
            prompt_prefix = prompt_value[:300]

            if value.startswith(prompt_prefix):
                return True

        prompt_markers = [
            "You are an EMTAC maintenance and engineering assistant.",
            "GENERAL RULES",
            "USE OF CONTEXT",
            "AUTHORITATIVE INFORMATION",
            "WHEN CONTEXT IS INCOMPLETE",
            "OUTPUT FORMAT",
            "--- CONTEXT START ---",
            "DOCUMENT MODE ACTIVE",
            "Selected document context:",
            "QUESTION:",
            "INSTRUCTIONS:",
            "FINAL ANSWER:",
        ]

        marker_count = sum(1 for marker in prompt_markers if marker in value)

        # Several prompt markers means this is likely not a clean answer.
        if marker_count >= 4:
            return True

        return False

    @staticmethod
    def _looks_like_instruction_only_output(text: str) -> bool:
        """
        Detect output that is only prompt/instruction scaffolding, not an answer.
        """

        value = str(text or "").strip()

        if not value:
            return True

        lower_value = value.lower()

        instruction_phrases = [
            "answer only from the provided context",
            "do not repeat the question",
            "final answer:",
            "instructions:",
            "document mode rules:",
            "selected document context:",
            "you are an emtac maintenance",
        ]

        phrase_count = sum(
            1 for phrase in instruction_phrases
            if phrase in lower_value
        )

        if phrase_count >= 2:
            return True

        # If it is very short and just a label, reject it.
        label_only = {
            "final answer:",
            "answer:",
            "question:",
            "instructions:",
        }

        if lower_value in label_only:
            return True

        return False