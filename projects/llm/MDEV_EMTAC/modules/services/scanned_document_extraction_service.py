from __future__ import annotations

import json
import re
import tempfile
import unicodedata
from pathlib import Path
from typing import Any, Dict, List, Optional

import fitz  # PyMuPDF

from modules.configuration.log_config import (
    debug_id,
    info_id,
    warning_id,
    error_id,
    with_request_id,
)

from modules.runtime.gpu_vision_adapter import GPUVisionAdapter


class ScannedDocumentExtractionService:
    """
    Recovery service for scanned/image-heavy PDFs.

    Emits live progress events when Socket.IO is available.

    Event name:
        upload_processing_progress
    """

    DEFAULT_RENDER_ZOOM = 2.0
    DEFAULT_MAX_IMAGE_TOKENS = 1024

    OCR_PROMPT = (
        "Extract all readable text from this scanned technical document page. "
        "Do not summarize. Preserve headings, labels, warnings, numbered steps, "
        "parameter names, values, figure captions, table text, and notes. "
        "Return the text in reading order. If a table is present, format it as "
        "plain text or markdown rows. If text is unclear, include your best reading."
    )

    _gpu_adapter: Optional[GPUVisionAdapter] = None

    @classmethod
    def _get_adapter(cls) -> GPUVisionAdapter:
        if cls._gpu_adapter is None:
            cls._gpu_adapter = GPUVisionAdapter()
        return cls._gpu_adapter

    @classmethod
    def _emit_progress(
        cls,
        *,
        request_id: Optional[str],
        stage: str,
        message: str,
        page: Optional[int] = None,
        total_pages: Optional[int] = None,
        file: Optional[str] = None,
        mode: Optional[str] = None,
        extra: Optional[Dict[str, Any]] = None,
    ) -> None:
        payload: Dict[str, Any] = {
            "request_id": request_id,
            "stage": stage,
            "message": message,
            "page": page,
            "total_pages": total_pages,
            "file": file,
            "mode": mode,
        }

        if extra:
            payload.update(extra)

        try:
            socketio = None

            try:
                from ai_emtac import socketio as app_socketio
                socketio = app_socketio
            except Exception:
                try:
                    from extensions import socketio as ext_socketio
                    socketio = ext_socketio
                except Exception:
                    try:
                        from app.extensions import socketio as app_ext_socketio
                        socketio = app_ext_socketio
                    except Exception:
                        socketio = None

            if socketio is None:
                return

            socketio.emit(
                "upload_processing_progress",
                payload,
                namespace="/",
            )

        except Exception:
            return

    @classmethod
    @with_request_id
    def extract_pages(
        cls,
        pdf_path: str,
        request_id: Optional[str] = None,
        *,
        max_pages: Optional[int] = None,
        sensitive_retry: bool = True,
        render_zoom: float = DEFAULT_RENDER_ZOOM,
    ) -> List[Dict[str, Any]]:
        pdf = Path(pdf_path)

        if not pdf.exists():
            warning_id(
                f"[ScannedDocumentExtractionService] PDF path does not exist: {pdf}",
                request_id,
            )
            return []

        gpu = cls._get_adapter()

        cls._emit_progress(
            request_id=request_id,
            stage="scanned_pdf_start",
            message=f"Starting scanned PDF extraction: {pdf.name}",
            file=str(pdf),
        )

        info_id(
            f"[ScannedDocumentExtractionService] Starting scanned PDF extraction "
            f"| pdf={pdf} | max_pages={max_pages} | sensitive_retry={sensitive_retry}",
            request_id,
        )

        pages = cls._try_gpu_pdf_structured(
            gpu=gpu,
            pdf_path=str(pdf),
            request_id=request_id,
            max_pages=max_pages,
        )

        if cls._has_usable_pages(pages):
            cls._emit_progress(
                request_id=request_id,
                stage="scanned_pdf_complete",
                message=f"Structured PDF extraction recovered {len(pages)} page(s).",
                total_pages=len(pages),
                file=str(pdf),
                mode="gpu_pdf_structured",
            )
            return pages

        warning_id(
            f"[ScannedDocumentExtractionService] GPU PDF structured extraction returned no usable pages; "
            f"trying rendered image fallback | pdf={pdf}",
            request_id,
        )

        pages = cls._extract_from_rendered_pages(
            gpu=gpu,
            pdf_path=str(pdf),
            request_id=request_id,
            max_pages=max_pages,
            render_zoom=render_zoom,
            sensitive=False,
        )

        if cls._has_usable_pages(pages):
            cls._emit_progress(
                request_id=request_id,
                stage="scanned_pdf_complete",
                message=f"Rendered page extraction recovered {len(pages)} page(s).",
                total_pages=len(pages),
                file=str(pdf),
                mode="rendered_image_fallback",
            )
            return pages

        if sensitive_retry:
            warning_id(
                f"[ScannedDocumentExtractionService] Normal rendered fallback returned no usable pages; "
                f"trying sensitive OCR retry | pdf={pdf}",
                request_id,
            )

            pages = cls._extract_from_rendered_pages(
                gpu=gpu,
                pdf_path=str(pdf),
                request_id=request_id,
                max_pages=max_pages,
                render_zoom=render_zoom,
                sensitive=True,
            )

            if cls._has_usable_pages(pages):
                cls._emit_progress(
                    request_id=request_id,
                    stage="scanned_pdf_complete",
                    message=f"Sensitive OCR retry recovered {len(pages)} page(s).",
                    total_pages=len(pages),
                    file=str(pdf),
                    mode="sensitive_ocr_retry",
                )
                return pages

        cls._emit_progress(
            request_id=request_id,
            stage="scanned_pdf_failed",
            message=f"No usable scanned text recovered from {pdf.name}.",
            file=str(pdf),
        )

        warning_id(
            f"[ScannedDocumentExtractionService] No usable scanned text recovered | pdf={pdf}",
            request_id,
        )
        return []

    @classmethod
    def _try_gpu_pdf_structured(
        cls,
        *,
        gpu: GPUVisionAdapter,
        pdf_path: str,
        request_id: Optional[str] = None,
        max_pages: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        try:
            cls._emit_progress(
                request_id=request_id,
                stage="gpu_pdf_structured",
                message="Trying structured PDF extraction first.",
                file=pdf_path,
                mode="gpu_pdf_structured",
            )

            debug_id(
                f"[ScannedDocumentExtractionService] Trying GPU pdf_to_structured | pdf={pdf_path}",
                request_id,
            )

            resp = gpu.pdf_to_structured(
                pdf_path,
                max_pages=max_pages,
            )

            if isinstance(resp, str):
                resp = cls._safe_json_extract(resp)

            if not isinstance(resp, dict):
                warning_id(
                    f"[ScannedDocumentExtractionService] Unexpected pdf_to_structured response type: {type(resp)}",
                    request_id,
                )
                return []

            chunks = resp.get("chunks") or resp.get("pages") or []

            if not isinstance(chunks, list):
                warning_id(
                    "[ScannedDocumentExtractionService] pdf_to_structured response has no chunks/pages list",
                    request_id,
                )
                return []

            pages_out: List[Dict[str, Any]] = []

            for idx, chunk in enumerate(chunks, start=1):
                if not isinstance(chunk, dict):
                    continue

                page_number = cls._safe_page_number(
                    chunk.get("page_number") or chunk.get("page") or idx,
                    fallback=idx,
                )

                page_payload = cls._normalize_page_response(
                    response=chunk,
                    page_number=page_number,
                    method="gpu_pdf_structured",
                )

                if page_payload.get("text"):
                    pages_out.append(page_payload)

            info_id(
                f"[ScannedDocumentExtractionService] GPU pdf_to_structured complete "
                f"| pages={len(pages_out)} | total_pages={resp.get('total_pages')}",
                request_id,
            )

            return pages_out

        except Exception as exc:
            error_id(
                f"[ScannedDocumentExtractionService] GPU pdf_to_structured failed "
                f"| pdf={pdf_path} | err={exc}",
                request_id,
                exc_info=True,
            )
            return []

    @classmethod
    def _extract_from_rendered_pages(
        cls,
        *,
        gpu: GPUVisionAdapter,
        pdf_path: str,
        request_id: Optional[str] = None,
        max_pages: Optional[int] = None,
        render_zoom: float = DEFAULT_RENDER_ZOOM,
        sensitive: bool = False,
    ) -> List[Dict[str, Any]]:
        pages_out: List[Dict[str, Any]] = []
        pdf = Path(pdf_path)

        mode = "sensitive_ocr_retry" if sensitive else "rendered_image_fallback"

        info_id(
            f"[ScannedDocumentExtractionService] Rendered page extraction start "
            f"| mode={mode} | pdf={pdf}",
            request_id,
        )

        try:
            with tempfile.TemporaryDirectory(prefix="emtac_scanned_pdf_pages_") as tmp_dir:
                tmp_root = Path(tmp_dir)

                with fitz.open(str(pdf)) as doc:
                    total_pages = doc.page_count
                    limit = total_pages

                    if max_pages is not None:
                        limit = min(total_pages, max_pages)

                    cls._emit_progress(
                        request_id=request_id,
                        stage="rendered_pdf_pages_start",
                        message=f"Rendering scanned PDF pages for extraction. Total pages: {limit}",
                        total_pages=limit,
                        file=str(pdf),
                        mode=mode,
                    )

                    for page_index in range(limit):
                        page_number = page_index + 1
                        page = doc.load_page(page_index)

                        pix = page.get_pixmap(
                            matrix=fitz.Matrix(render_zoom, render_zoom),
                            alpha=False,
                        )

                        image_path = tmp_root / f"page_{page_number:04d}.png"
                        pix.save(str(image_path))

                        cls._emit_progress(
                            request_id=request_id,
                            stage="scanned_pdf_page_rendered",
                            message=f"Rendered page {page_number} of {limit}.",
                            page=page_number,
                            total_pages=limit,
                            file=str(image_path),
                            mode=mode,
                        )

                        debug_id(
                            f"[ScannedDocumentExtractionService] Rendered page "
                            f"| mode={mode} | page={page_number}/{limit} | image={image_path}",
                            request_id,
                        )

                        cls._emit_progress(
                            request_id=request_id,
                            stage="scanned_pdf_page_vlm",
                            message=f"Running VLM extraction on page {page_number} of {limit}.",
                            page=page_number,
                            total_pages=limit,
                            file=str(image_path),
                            mode=mode,
                            extra={
                                "image_size_bytes": image_path.stat().st_size
                                if image_path.exists()
                                else None,
                            },
                        )

                        response = cls._call_gpu_image_extraction(
                            gpu=gpu,
                            image_path=str(image_path),
                            request_id=request_id,
                            sensitive=sensitive,
                        )

                        page_payload = cls._normalize_page_response(
                            response=response,
                            page_number=page_number,
                            method=mode,
                        )

                        recovered_chars = len(str(page_payload.get("text") or ""))

                        if page_payload.get("text"):
                            pages_out.append(page_payload)
                            cls._emit_progress(
                                request_id=request_id,
                                stage="scanned_pdf_page_complete",
                                message=f"Recovered text from page {page_number} of {limit} ({recovered_chars} chars).",
                                page=page_number,
                                total_pages=limit,
                                file=str(image_path),
                                mode=mode,
                                extra={
                                    "recovered_chars": recovered_chars,
                                },
                            )
                        else:
                            cls._emit_progress(
                                request_id=request_id,
                                stage="scanned_pdf_page_no_text",
                                message=f"No text recovered from page {page_number} of {limit}.",
                                page=page_number,
                                total_pages=limit,
                                file=str(image_path),
                                mode=mode,
                            )

                            warning_id(
                                f"[ScannedDocumentExtractionService] No text from rendered page "
                                f"| mode={mode} | page={page_number} | pdf={pdf}",
                                request_id,
                            )

            cls._emit_progress(
                request_id=request_id,
                stage="rendered_pdf_pages_complete",
                message=f"Rendered page extraction complete. Recovered {len(pages_out)} page(s).",
                total_pages=len(pages_out),
                file=str(pdf),
                mode=mode,
            )

            info_id(
                f"[ScannedDocumentExtractionService] Rendered page extraction complete "
                f"| mode={mode} | pages={len(pages_out)} | pdf={pdf}",
                request_id,
            )

            return pages_out

        except Exception as exc:
            error_id(
                f"[ScannedDocumentExtractionService] Rendered page extraction failed "
                f"| mode={mode} | pdf={pdf} | err={exc}",
                request_id,
                exc_info=True,
            )

            cls._emit_progress(
                request_id=request_id,
                stage="rendered_pdf_pages_error",
                message=f"Rendered page extraction failed: {exc}",
                file=str(pdf),
                mode=mode,
            )

            return []

    @classmethod
    def _call_gpu_image_extraction(
        cls,
        *,
        gpu: GPUVisionAdapter,
        image_path: str,
        request_id: Optional[str] = None,
        sensitive: bool = False,
    ) -> Any:
        prompt = cls.OCR_PROMPT if sensitive else None
        max_tokens = cls.DEFAULT_MAX_IMAGE_TOKENS if sensitive else 512

        candidate_methods = [
            "ocr_image",
            "image_to_text",
            "extract_image_text",
            "image_to_structured",
            "analyze_image",
            "describe_image",
        ]

        for method_name in candidate_methods:
            method = getattr(gpu, method_name, None)

            if not callable(method):
                continue

            debug_id(
                f"[ScannedDocumentExtractionService] Trying GPU image method "
                f"| method={method_name} | sensitive={sensitive} | image={image_path}",
                request_id,
            )

            call_attempts = []

            if sensitive and prompt:
                call_attempts.extend(
                    [
                        lambda: method(image_path, prompt=prompt, max_new_tokens=max_tokens),
                        lambda: method(image_path, instruction=prompt, max_new_tokens=max_tokens),
                        lambda: method(image_path, question=prompt, max_new_tokens=max_tokens),
                        lambda: method(image_path, prompt=prompt),
                    ]
                )

            call_attempts.extend(
                [
                    lambda: method(image_path, max_new_tokens=max_tokens),
                    lambda: method(image_path),
                    lambda: method(str(image_path)),
                ]
            )

            for attempt in call_attempts:
                try:
                    return attempt()
                except TypeError:
                    continue
                except Exception as exc:
                    warning_id(
                        f"[ScannedDocumentExtractionService] GPU image method failed "
                        f"| method={method_name} | err={exc}",
                        request_id,
                    )
                    break

        warning_id(
            "[ScannedDocumentExtractionService] No compatible GPU image extraction method found",
            request_id,
        )
        return None

    @classmethod
    def _normalize_page_response(
        cls,
        *,
        response: Any,
        page_number: int,
        method: str,
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
            "method": method,
        }

    @classmethod
    def _clean_vlm_text(cls, value: Any) -> str:
        if value is None:
            return ""

        text = str(value).strip()

        if not text:
            return ""

        text = unicodedata.normalize("NFKC", text)
        text = text.replace("\x00", "").replace("\u0000", "")
        text = text.replace("\r\n", "\n").replace("\r", "\n").strip()

        assistant_matches = list(
            re.finditer(r"(?im)^\s*assistant\s*$", text)
        )

        if assistant_matches:
            last = assistant_matches[-1]
            text = text[last.end():].strip()

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

        text = re.sub(
            r"(?is)^Extract all readable text from this scanned technical document page\.\s*",
            "",
            text,
        ).strip()

        text = re.sub(r"\n{3,}", "\n\n", text).strip()

        return text

    @classmethod
    def _safe_json_extract(cls, raw_str: str) -> Dict[str, Any]:
        if not raw_str:
            return {"text": "", "visual_elements": []}

        raw_str = str(raw_str).strip()

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

        return {
            "text": raw_str,
            "visual_elements": [],
        }

    @classmethod
    def _safe_page_number(cls, value: Any, *, fallback: int) -> int:
        try:
            return int(value)
        except Exception:
            return fallback

    @classmethod
    def _has_usable_pages(cls, pages: List[Dict[str, Any]]) -> bool:
        if not pages:
            return False

        for page in pages:
            if not isinstance(page, dict):
                continue

            text = str(page.get("text") or "").strip()

            if len(text) >= 25:
                return True

        return False