from __future__ import annotations

import os
import requests
import logging
from typing import List, Dict, Optional, Any, Tuple

log = logging.getLogger("gpu_adapter")


class GPUServiceError(RuntimeError):
    """Raised when the GPU service returns an error."""


class GPUAdapter:
    """
    Thin client adapter between local coordinators and the EMTAC GPU Service.

    Responsibilities:
    - NO torch / cuda
    - NO model loading
    - Pure HTTP interface
    - Stateless
    """

    def __init__(
        self,
        base_url: Optional[str] = None,
        timeout: int = 30,
    ):
        self.base_url = (base_url or os.environ.get(
            "GPU_SERVICE_URL", "http://127.0.0.1:5050"
        )).rstrip("/")

        self.timeout = timeout

        log.info(f"[GPU ADAPTER] Using GPU service at {self.base_url}")

    # ---------------------------------------------------------
    # Health
    # ---------------------------------------------------------
    def health(self) -> Dict[str, Any]:
        try:
            r = requests.get(
                f"{self.base_url}/health",
                timeout=self.timeout,
            )
            r.raise_for_status()
            data = r.json()
            if not isinstance(data, dict):
                raise GPUServiceError("Invalid health response format")
            return data
        except Exception as e:
            raise GPUServiceError(f"GPU service health check failed: {e}")

    # ---------------------------------------------------------
    # Internal helper (shared POST)
    # ---------------------------------------------------------
    def _post_json(
        self,
        path: str,
        payload: Dict[str, Any],
        *,
        timeout: Optional[int] = None,
    ) -> Dict[str, Any]:
        try:
            r = requests.post(
                f"{self.base_url}{path}",
                json=payload,
                timeout=timeout or self.timeout,
            )
            r.raise_for_status()
            data = r.json()
            if not isinstance(data, dict):
                raise GPUServiceError(
                    f"Invalid JSON response from {path}: {type(data)}"
                )
            return data
        except Exception as e:
            rid = payload.get("request_id")
            rid_info = f" [request_id={rid}]" if rid else ""
            raise GPUServiceError(
                f"GPU service request failed ({path}){rid_info}: {e}"
            )

    # ---------------------------------------------------------
    # Embeddings
    # ---------------------------------------------------------
    def embed(
        self,
        texts: List[str],
        model: Optional[str] = None,
        batch_size: int = 32,
        normalize: bool = True,
        request_id: Optional[str] = None,
    ) -> List[List[float]]:
        """
        Return embeddings only (legacy-compatible).
        """
        vecs, _ = self.embed_with_meta(
            texts=texts,
            model=model,
            batch_size=batch_size,
            normalize=normalize,
            request_id=request_id,
        )
        return vecs

    def embed_with_meta(
        self,
        texts: List[str],
        model: Optional[str] = None,
        batch_size: int = 32,
        normalize: bool = True,
        request_id: Optional[str] = None,
    ) -> Tuple[List[List[float]], Dict[str, Any]]:
        """
        Return embeddings + metadata.
        """
        if not texts:
            return [], {}

        model = model or os.getenv("GPU_EMBED_MODEL", "minilm")

        payload = {
            "model": model,
            "texts": texts,
            "batch_size": batch_size,
            "normalize": normalize,
        }

        if request_id:
            payload["request_id"] = request_id

        data = self._post_json("/embed", payload)

        if "embeddings" not in data:
            raise GPUServiceError("GPU service response missing 'embeddings'")

        return data["embeddings"], data.get("meta", {})

    # ---------------------------------------------------------
    # Text Generation (FLAN / T5 / LLMs)
    # ---------------------------------------------------------
    def generate(
        self,
        *,
        prompt: str,
        model: str,
        max_tokens: int = 128,
        temperature: float = 0.7,
        top_p: float = 0.95,
        num_beams: int = 1,
        do_sample: Optional[bool] = None,
        repetition_penalty: Optional[float] = None,
        stop: Optional[List[str]] = None,
        request_id: Optional[str] = None,
        timeout: Optional[int] = None,
    ) -> str:
        """
        Generic text generation endpoint.

        This method is intentionally flexible:
        - Supports FLAN (beam search)
        - Supports causal LLMs (sampling / deterministic)
        - Supports OpenELM repetition control
        """

        payload: Dict[str, Any] = {
            "model": model,
            "prompt": prompt,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "num_beams": num_beams,
        }

        if do_sample is not None:
            payload["do_sample"] = do_sample

        if repetition_penalty is not None:
            payload["repetition_penalty"] = repetition_penalty

        if stop:
            payload["stop"] = stop

        if request_id:
            payload["request_id"] = request_id

        data = self._post_json(
            "/generate",
            payload,
            timeout=timeout,
        )

        if "text" not in data:
            raise GPUServiceError("GPU service response missing 'text'")

        return data["text"]

    # ---------------------------------------------------------
    # Extractive QA (Roberta / QA heads)
    # ---------------------------------------------------------
    def qa(
        self,
        *,
        question: str,
        context: str,
        model: str = "roberta-squad2",
        request_id: Optional[str] = None,
    ) -> str:
        payload = {
            "model": model,
            "question": question,
            "context": context,
        }

        if request_id:
            payload["request_id"] = request_id

        data = self._post_json("/qa", payload)

        if "answer" not in data:
            raise GPUServiceError("GPU service response missing 'answer'")

        return data["answer"]
