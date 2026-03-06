from __future__ import annotations

import os
import time
import mimetypes
import requests
from typing import Optional, Dict, Any

from modules.configuration.log_config import (
    debug_id,
    info_id,
    warning_id,
    error_id,
    get_request_id,
)


class GPUVisionAdapter:
    """
    Adapter for EMTAC GPU Vision Service.

    Responsibilities:
    - Route VLM requests to GPU service
    - Handle retries, timeouts
    - Keep main app free of VLM / CUDA concerns
    """

    def __init__(
        self,
        base_url: Optional[str] = None,
        timeout: int = 300,
        max_retries: int = 2,
        enabled: Optional[bool] = None,
    ):
        self.base_url = (base_url or os.getenv("GPU_SERVICE_URL", "http://127.0.0.1:5050")).rstrip("/")
        self.timeout = (10, 7200)
        self.max_retries = int(max_retries)
        self.enabled = enabled if enabled is not None else self._detect_service()

        rid = get_request_id()
        info_id(
            f"[GPU-VISION] Initialized | enabled={self.enabled} | url={self.base_url}",
            rid,
        )

    # ---------------------------------------------------------
    # Internal
    # ---------------------------------------------------------

    def _detect_service(self) -> bool:
        try:
            r = requests.get(f"{self.base_url}/health", timeout=5)
            return r.status_code == 200
        except Exception:
            return False

    def _guess_content_type(self, file_path: str) -> str:
        ctype, _ = mimetypes.guess_type(file_path)
        return ctype or "application/octet-stream"

    def _post_file(
            self,
            endpoint: str,
            file_path: str,
            *,
            params: Optional[Dict[str, Any]] = None,
            headers: Optional[Dict[str, str]] = None,
    ) -> Dict[str, Any]:
        """
        POST a file to the GPU service and return JSON response.

        params: query parameters such as:
            model
            max_pages

        NOTE:
        Runtime parameters (dpi, batching, tokens, etc.)
        are controlled inside the GPU service.
        """

        if not self.enabled:
            raise RuntimeError("GPU service is disabled or unreachable")

        rid = get_request_id()

        url = f"{self.base_url}{endpoint}"
        params = params or {}
        headers = headers or {}

        headers.setdefault("X-Request-Id", str(rid) if rid else "")

        last_err: Optional[Exception] = None

        try:
            file_size = os.path.getsize(file_path)
        except Exception:
            file_size = -1

        for attempt in range(1, self.max_retries + 1):

            try:

                debug_id(
                    f"[GPU-VISION] POST {endpoint} "
                    f"attempt={attempt} file={file_path} size={file_size} params={params}",
                    rid,
                )

                filename = os.path.basename(file_path)
                content_type = self._guess_content_type(file_path)

                with open(file_path, "rb") as f:

                    files = {"file": (filename, f, content_type)}

                    response = requests.post(
                        url,
                        params=params,
                        files=files,
                        headers=headers,
                        timeout=self.timeout,
                    )

                if not response.ok:
                    snippet = (response.text or "")[:2000]
                    raise RuntimeError(
                        f"GPU service error {response.status_code}: {snippet}"
                    )

                return response.json()

            except requests.exceptions.ConnectTimeout as e:
                raise RuntimeError(
                    "GPU vision service connection timed out during connect phase"
                ) from e

            except requests.exceptions.ReadTimeout as e:
                raise RuntimeError(
                    "GPU vision request timed out while waiting for long-running GPU job"
                ) from e

            except Exception as e:

                last_err = e

                warning_id(
                    f"[GPU-VISION] Attempt {attempt}/{self.max_retries} failed: {e}",
                    rid,
                )

                if attempt < self.max_retries:
                    time.sleep(1)

        raise RuntimeError(f"GPU vision request failed after retries: {last_err}")

    # ---------------------------------------------------------
    # Public API
    # ---------------------------------------------------------

    def describe_image(
        self,
        image_path: str,
        *,
        model: str = "nu_markdown",
        max_new_tokens: int = 256,
    ) -> str:
        data = self._post_file(
            "/vision/describe",
            image_path,
            params={
                "model": model,
                "max_new_tokens": int(max_new_tokens),
            },
        )
        return data.get("description", "") or ""

    def markdown_image(
        self,
        image_path: str,
        *,
        model: str = "nu_markdown",
        max_new_tokens: int = 768,
    ) -> str:
        data = self._post_file(
            "/vision/markdown",
            image_path,
            params={
                "model": model,
                "max_new_tokens": int(max_new_tokens),
            },
        )
        return data.get("markdown", "") or ""

    def pdf_to_structured(
            self,
            pdf_path: str,
            *,
            model: str = "nu_markdown",
            max_pages: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Calls GPU /vision/pdf and returns full structured payload:
          {
            total_pages: int,
            chunks: [{page_number, markdown, text}, ...],
            images: [...],
            ...
          }
        """
        params: Dict[str, Any] = {
            "model": model,
        }

        if max_pages is not None:
            params["max_pages"] = int(max_pages)

        return self._post_file("/vision/pdf", pdf_path, params=params)

    def markdown_pdf(
        self,
        pdf_path: str,
        *,
        model: str = "nu_markdown",
        max_pages: Optional[int] = None,
        dpi: int = 160,
        max_new_tokens: int = 1024,
        join_with: str = "\n\n",
    ) -> str:
        """
        Backwards-compatible: returns a single markdown string by concatenating pages.
        """
        data = self.pdf_to_structured(
            pdf_path,
            model=model,
            max_pages=max_pages,
        )

        chunks = data.get("chunks") or []
        if not isinstance(chunks, list):
            return ""

        parts = []
        for ch in chunks:
            if isinstance(ch, dict):
                md = (ch.get("markdown") or "").strip()
                if md:
                    parts.append(md)

        return join_with.join(parts)

    def is_available(self) -> bool:
        if not self.enabled:
            self.enabled = self._detect_service()
        return self.enabled