from __future__ import annotations

import os
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
    - Route embedding & generation requests to GPU service
    - Handle retries, timeouts, and fallback behavior
    - Keep main app free of model-loading concerns
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
                return response.json()

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
        - This is ONLY called when backend == 'gpu_service'
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
        """
        rid = get_request_id()

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
            "max_tokens": max_new_tokens,
            "temperature": temperature,
            "top_p": top_p,
        }

        debug_id(
            f"[GPU-ADAPTER] POST /generate "
            f"model={gpu_model} max_tokens={max_new_tokens} prompt_len={len(prompt)}",
            rid,
        )

        try:
            data = self._post("/generate", payload)

            text = data.get("text")
            if not text:
                raise RuntimeError("GPU service returned empty generation result")

            return text

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