from __future__ import annotations

import json
from pathlib import Path
from typing import Optional, Dict, Any, List

from modules.configuration.log_config import (
    info_id,
    error_id,
    with_request_id,warning_id)

from modules.runtime.gpu_vision_adapter import GPUVisionAdapter


class AIModelsVLMService:
    """
    Service layer wrapper for VLM usage.

    This version routes ALL execution through the GPU service.
    No local model loading.
    """

    _gpu_adapter: Optional[GPUVisionAdapter] = None

    # ------------------------------------------------
    # Adapter lifecycle
    # ------------------------------------------------
    @classmethod
    def _get_adapter(cls) -> GPUVisionAdapter:
        if cls._gpu_adapter is None:
            cls._gpu_adapter = GPUVisionAdapter()
        return cls._gpu_adapter

    # ------------------------------------------------
    # Internal JSON extractor
    # ------------------------------------------------
    @classmethod
    def _safe_json_extract(cls, raw_str: str) -> Dict[str, Any]:
        if not raw_str:
            return {"text": "", "visual_elements": []}

        raw_str = raw_str.strip()

        try:
            obj = json.loads(raw_str)
            if isinstance(obj, dict):
                return obj
        except Exception:
            pass

        start = raw_str.find("{")
        end = raw_str.rfind("}")
        if start != -1 and end != -1 and end > start:
            try:
                obj = json.loads(raw_str[start:end + 1])
                if isinstance(obj, dict):
                    return obj
            except Exception:
                pass

        return {"text": raw_str, "visual_elements": []}

    # ------------------------------------------------
    # Structured PDF Extraction (FIXED)
    # ------------------------------------------------
    @classmethod
    @with_request_id
    def extract_structured_pages_from_pdf(
            cls,
            pdf_path: str,
            request_id=None,
            *,
            max_pages: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """
        Calls GPU /vision/pdf and returns one dict per page:

        [
          {
            "page_number": int,
            "text": str,                 # plain text used for indexing/FTS/embeddings
            "markdown": str,             # raw markdown (useful for UI/rendering)
            "visual_elements": list,     # figures/tables/bboxes (if GPU provides)
            "images": list,              # image refs (if GPU provides per-page)
            "raw": dict                  # original chunk payload
          },
          ...
        ]
        """
        p = Path(pdf_path)
        if not p.exists():
            raise RuntimeError(f"PDF path does not exist: {p}")

        gpu = cls._get_adapter()

        info_id(
            f"[AIModelsVLMService] Structured PDF extraction start | pdf={p} max_pages={max_pages}",
            request_id,
        )

        try:
            # IMPORTANT:
            # Must call /vision/pdf so we get per-page chunks.
            resp: Any = gpu.pdf_to_structured(str(p), max_pages=max_pages)

            # Defensive: if adapter returned a string, attempt JSON recovery
            if isinstance(resp, str):
                resp = cls._safe_json_extract(resp)

            if not isinstance(resp, dict):
                raise RuntimeError(f"GPU vision response unexpected type: {type(resp)}")

            # GPU service schema uses "chunks" (VisionPDFResponse)
            chunks = resp.get("chunks") or resp.get("pages") or []
            if not isinstance(chunks, list):
                raise RuntimeError(
                    f"GPU vision response missing chunks list. keys={list(resp.keys())}"
                )

            # Optional: GPU may return top-level images list
            top_images = resp.get("images") or []
            if not isinstance(top_images, list):
                top_images = []

            pages_out: List[Dict[str, Any]] = []

            for idx, ch in enumerate(chunks, start=1):
                if not isinstance(ch, dict):
                    continue

                # Page number normalization
                page_num_raw = ch.get("page_number") or ch.get("page") or idx
                try:
                    page_num = int(page_num_raw)
                except Exception:
                    page_num = idx

                md = (ch.get("markdown") or "").strip()

                # Prefer GPU-provided plain text; fallback to markdown if empty
                txt = (ch.get("text") or "").strip()
                if not txt and md:
                    txt = md

                # Normalize visual elements
                ve = ch.get("visual_elements") or []
                if not isinstance(ve, list):
                    ve = []

                # Per-page images if GPU provides, otherwise empty.
                # (If you later add images per page in GPU payload, this will pick it up.)
                page_images = ch.get("images") or []
                if not isinstance(page_images, list):
                    page_images = []

                pages_out.append(
                    {
                        "page_number": page_num,
                        "text": txt,
                        "markdown": md,
                        "visual_elements": ve,
                        "images": page_images,
                        "raw": ch,
                    }
                )

            total_pages = resp.get("total_pages")
            info_id(
                f"[AIModelsVLMService] Structured PDF extraction complete | "
                f"pages={len(pages_out)} total_pages={total_pages} gpu_images={len(top_images)}",
                request_id,
            )

            # If GPU produced no usable output, make that explicit for downstream logic
            if not pages_out:
                warning_id(
                    f"[AIModelsVLMService] Structured PDF extraction returned 0 pages | pdf={p}",
                    request_id,
                )

            return pages_out

        except Exception as e:
            error_id(
                f"[AIModelsVLMService] Structured PDF extraction failed | pdf={p} | err={e}",
                request_id,
            )
            raise