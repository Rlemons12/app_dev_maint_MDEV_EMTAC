from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import Optional, Dict, Any, List
import re
import fitz  # PyMuPDF

from modules.configuration.log_config import (
    info_id,
    debug_id,
    error_id,
    with_request_id,
    warning_id,
)

from modules.runtime.gpu_vision_adapter import GPUVisionAdapter


class AIModelsVLMService:
    """
    Service layer wrapper for VLM usage.

    Primary path:
        PDF -> GPU /vision/pdf

    Fallback path:
        PDF -> rendered page images -> GPU image extraction if adapter supports it
    """

    _gpu_adapter: Optional[GPUVisionAdapter] = None

    @classmethod
    def _get_adapter(cls) -> GPUVisionAdapter:
        if cls._gpu_adapter is None:
            cls._gpu_adapter = GPUVisionAdapter()
        return cls._gpu_adapter

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

    @classmethod
    def _normalize_gpu_page_response(
            cls,
            *,
            response: Any,
            page_number: int,
    ) -> Dict[str, Any]:
        if isinstance(response, str):
            response = cls._safe_json_extract(response)

        if not isinstance(response, dict):
            response = {
                "text": str(response or ""),
                "visual_elements": [],
            }

        markdown = str(response.get("markdown") or "").strip()
        text = str(response.get("text") or "").strip()

        if not text and markdown:
            text = markdown

        text = cls._clean_vlm_text(text)
        markdown = cls._clean_vlm_text(markdown)

        visual_elements = response.get("visual_elements") or []
        if not isinstance(visual_elements, list):
            visual_elements = []

        images = response.get("images") or []
        if not isinstance(images, list):
            images = []

        return {
            "page_number": page_number,
            "text": text,
            "markdown": markdown,
            "visual_elements": visual_elements,
            "images": images,
            "raw": response,
        }

    @classmethod
    def _call_gpu_image_fallback(
        cls,
        *,
        gpu: GPUVisionAdapter,
        image_path: str,
        request_id: Optional[str] = None,
    ) -> Any:
        """
        Try common GPUVisionAdapter image methods without hard-binding
        this service to one exact adapter method name.
        """

        candidate_methods = [
            "image_to_structured",
            "image_to_text",
            "describe_image",
            "analyze_image",
            "extract_image_text",
        ]

        for method_name in candidate_methods:
            method = getattr(gpu, method_name, None)

            if not callable(method):
                continue

            debug_id(
                f"[AIModelsVLMService] Trying GPU image fallback method={method_name} image={image_path}",
                request_id,
            )

            try:
                return method(image_path)
            except TypeError:
                try:
                    return method(str(image_path))
                except Exception as exc:
                    warning_id(
                        f"[AIModelsVLMService] GPU image fallback method failed "
                        f"| method={method_name} | err={exc}",
                        request_id,
                    )
            except Exception as exc:
                warning_id(
                    f"[AIModelsVLMService] GPU image fallback method failed "
                    f"| method={method_name} | err={exc}",
                    request_id,
                )

        warning_id(
            "[AIModelsVLMService] No compatible GPU image fallback method found on GPUVisionAdapter",
            request_id,
        )
        return None

    @classmethod
    def _fallback_pdf_pages_via_rendered_images(
        cls,
        *,
        pdf_path: str,
        gpu: GPUVisionAdapter,
        request_id: Optional[str] = None,
        max_pages: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """
        Render PDF pages to PNG images and send each rendered page through
        any available GPU image extraction method.

        This is useful when:
          - native PDF text is nearly empty
          - /vision/pdf returns 0 chunks/pages
          - the PDF is scanned/image-only
        """

        pages_out: List[Dict[str, Any]] = []
        pdf = Path(pdf_path)

        info_id(
            f"[AIModelsVLMService] Rendered-image PDF fallback start | pdf={pdf}",
            request_id,
        )

        try:
            with tempfile.TemporaryDirectory(prefix="emtac_pdf_vlm_pages_") as tmp_dir:
                tmp_root = Path(tmp_dir)

                with fitz.open(str(pdf)) as doc:
                    total_pages = doc.page_count
                    limit = total_pages

                    if max_pages is not None:
                        limit = min(total_pages, max_pages)

                    for page_index in range(limit):
                        page_number = page_index + 1
                        page = doc.load_page(page_index)

                        pix = page.get_pixmap(
                            matrix=fitz.Matrix(2, 2),
                            alpha=False,
                        )

                        image_path = tmp_root / f"page_{page_number:04d}.png"
                        pix.save(str(image_path))

                        debug_id(
                            f"[AIModelsVLMService] Rendered PDF page image "
                            f"| page={page_number} | image={image_path}",
                            request_id,
                        )

                        response = cls._call_gpu_image_fallback(
                            gpu=gpu,
                            image_path=str(image_path),
                            request_id=request_id,
                        )

                        page_payload = cls._normalize_gpu_page_response(
                            response=response,
                            page_number=page_number,
                        )

                        if page_payload.get("text"):
                            pages_out.append(page_payload)
                        else:
                            warning_id(
                                f"[AIModelsVLMService] Rendered page fallback produced no text "
                                f"| pdf={pdf} | page={page_number}",
                                request_id,
                            )

            info_id(
                f"[AIModelsVLMService] Rendered-image PDF fallback complete "
                f"| pdf={pdf} | pages={len(pages_out)}",
                request_id,
            )

            return pages_out

        except Exception as exc:
            error_id(
                f"[AIModelsVLMService] Rendered-image PDF fallback failed "
                f"| pdf={pdf} | err={exc}",
                request_id,
                exc_info=True,
            )
            return []

    @classmethod
    @with_request_id
    def extract_structured_pages_from_pdf(
        cls,
        pdf_path: str,
        request_id=None,
        *,
        max_pages: Optional[int] = None,
    ) -> List[Dict[str, Any]]:

        p = Path(pdf_path)

        if not p.exists():
            raise RuntimeError(f"PDF path does not exist: {p}")

        gpu = cls._get_adapter()

        info_id(
            f"[AIModelsVLMService] Structured PDF extraction start | pdf={p} max_pages={max_pages}",
            request_id,
        )

        try:
            resp: Any = gpu.pdf_to_structured(
                str(p),
                max_pages=max_pages,
            )

            if isinstance(resp, str):
                resp = cls._safe_json_extract(resp)

            if not isinstance(resp, dict):
                raise RuntimeError(
                    f"GPU vision response unexpected type: {type(resp)}"
                )

            debug_id(
                f"[AIModelsVLMService] Raw PDF vision response keys={list(resp.keys())}",
                request_id,
            )

            debug_id(
                "[AIModelsVLMService] Raw PDF vision response preview="
                + str(resp)[:1500],
                request_id,
            )

            chunks = resp.get("chunks") or resp.get("pages") or []

            if not isinstance(chunks, list):
                raise RuntimeError(
                    f"GPU vision response missing chunks list. keys={list(resp.keys())}"
                )

            top_images = resp.get("images") or []
            if not isinstance(top_images, list):
                top_images = []

            pages_out: List[Dict[str, Any]] = []

            for idx, ch in enumerate(chunks, start=1):
                if not isinstance(ch, dict):
                    continue

                page_num_raw = ch.get("page_number") or ch.get("page") or idx

                try:
                    page_num = int(page_num_raw)
                except Exception:
                    page_num = idx

                page_payload = cls._normalize_gpu_page_response(
                    response=ch,
                    page_number=page_num,
                )

                if page_payload.get("text"):
                    pages_out.append(page_payload)
                else:
                    warning_id(
                        f"[AIModelsVLMService] GPU PDF chunk had no text "
                        f"| pdf={p} | page={page_num}",
                        request_id,
                    )

            total_pages = resp.get("total_pages")

            info_id(
                f"[AIModelsVLMService] Structured PDF extraction complete | "
                f"pages={len(pages_out)} total_pages={total_pages} gpu_images={len(top_images)}",
                request_id,
            )

            if pages_out:
                return pages_out

            warning_id(
                f"[AIModelsVLMService] Structured PDF extraction returned 0 usable pages | pdf={p}",
                request_id,
            )

            return cls._fallback_pdf_pages_via_rendered_images(
                pdf_path=str(p),
                gpu=gpu,
                request_id=request_id,
                max_pages=max_pages,
            )

        except Exception as e:
            error_id(
                f"[AIModelsVLMService] Structured PDF extraction failed | pdf={p} | err={e}",
                request_id,
                exc_info=True,
            )

            warning_id(
                f"[AIModelsVLMService] Trying rendered-image fallback after PDF extraction exception | pdf={p}",
                request_id,
            )

            return cls._fallback_pdf_pages_via_rendered_images(
                pdf_path=str(p),
                gpu=gpu,
                request_id=request_id,
                max_pages=max_pages,
            )

    @classmethod
    def _clean_vlm_text(cls, value: Any) -> str:
        """
        Remove chat transcript wrappers returned by /vision/describe.

        Example bad text:
            system
            You are a helpful assistant.
            user
            Describe the image precisely...
            assistant
            Actual useful description...

        Keeps only the assistant/document description.
        """

        if value is None:
            return ""

        text = str(value).strip()

        if not text:
            return ""

        text = text.replace("\r\n", "\n").replace("\r", "\n").strip()

        # If response contains role transcript markers, keep only content after last assistant marker.
        assistant_matches = list(
            re.finditer(
                r"(?im)^\s*assistant\s*$",
                text,
            )
        )

        if assistant_matches:
            last = assistant_matches[-1]
            text = text[last.end():].strip()

        # Remove leading role labels if they remain inline.
        text = re.sub(
            r"(?is)^\s*system\s*\n.*?\buser\s*\n.*?\bassistant\s*\n",
            "",
            text,
        ).strip()

        text = re.sub(
            r"(?i)^\s*(system|user|assistant)\s*:\s*",
            "",
            text,
        ).strip()

        # Remove common prompt instruction if it leaked.
        text = re.sub(
            r"(?is)^You are a helpful assistant\.\s*",
            "",
            text,
        ).strip()

        text = re.sub(
            r"(?is)^Describe the image precisely and succinctly\.\s*Focus on visible content only\.\s*",
            "",
            text,
        ).strip()

        text = re.sub(r"\n{3,}", "\n\n", text).strip()

        return text