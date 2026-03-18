from __future__ import annotations

import logging
import os
import time
from typing import Any, Dict, List, Optional

import requests

logger = logging.getLogger(__name__)


class GPUServiceClient:
    """
    Thin client used by the PyCharm MCP gateway to talk to the GPU service.

    Responsibilities:
    - Health checks
    - Retries / timeout handling
    - Request formatting for MCP tool APIs
    - Response validation
    """

    def __init__(
        self,
        base_url: Optional[str] = None,
        timeout: int = 300,
        max_retries: int = 2,
    ) -> None:
        self.base_url = (
            base_url or os.getenv("GPU_SERVICE_URL", "http://127.0.0.1:5050")
        ).rstrip("/")
        self.timeout = timeout
        self.max_retries = max_retries

        logger.info(
            "GPUServiceClient initialized | base_url=%s | timeout=%s | retries=%s",
            self.base_url,
            self.timeout,
            self.max_retries,
        )

    def health(self) -> bool:
        try:
            response = requests.get(f"{self.base_url}/health", timeout=5)
            return response.status_code == 200
        except Exception as exc:
            logger.warning("GPU health check failed: %s", exc)
            return False

    def _post(self, endpoint: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        url = f"{self.base_url}{endpoint}"
        last_error: Optional[Exception] = None

        for attempt in range(1, self.max_retries + 1):
            try:
                logger.info("POST %s | attempt=%s", endpoint, attempt)

                response = requests.post(
                    url,
                    json=payload,
                    timeout=self.timeout,
                )
                response.raise_for_status()
                return response.json()

            except Exception as exc:
                last_error = exc
                logger.warning(
                    "POST failed | endpoint=%s | attempt=%s | error=%s",
                    endpoint,
                    attempt,
                    exc,
                )
                time.sleep(1)

        raise RuntimeError(f"GPU service request failed for {endpoint}: {last_error}")

    def generate_code(
        self,
        *,
        prompt: str,
        model: str = "qwen",
        max_new_tokens: int = 512,
        temperature: float = 0.2,
        top_p: float = 0.95,
        system_prompt: Optional[str] = None,
    ) -> str:
        payload = {
            "prompt": prompt,
            "model": model,
            "max_tokens": max_new_tokens,
            "temperature": temperature,
            "top_p": top_p,
        }

        if system_prompt:
            payload["system_prompt"] = system_prompt

        data = self._post("/mcp/tools/generate_code", payload)

        text = data.get("text")
        if not text:
            raise RuntimeError("GPU service returned empty generation result")

        return str(text)

    def embed(
        self,
        *,
        texts: List[str],
        gpu_model: str,
        batch_size: int = 32,
        normalize: bool = True,
    ) -> List[List[float]]:
        payload = {
            "texts": texts,
            "model": gpu_model,
            "batch_size": batch_size,
            "normalize": normalize,
        }

        data = self._post("/embed", payload)

        embeddings = data.get("embeddings")
        if not embeddings:
            raise RuntimeError("GPU service returned no embeddings")

        return embeddings