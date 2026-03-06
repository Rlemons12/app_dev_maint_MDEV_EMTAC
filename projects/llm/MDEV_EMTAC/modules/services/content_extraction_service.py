import os
import unicodedata
from typing import Optional, Dict, Any, List

import fitz  # PyMuPDF

from modules.configuration.log_config import (
    with_request_id,
    debug_id,
    info_id,
    warning_id,
    error_id,
)

from modules.services.ai_models_vlm_service import AIModelsVLMService


class ContentExtractionService:
    """
    Extracts text from documents.

    Strategy:
        1. Try native extraction (PyMuPDF).
        2. If PDF appears scanned (very little text),
           fallback to NuMarkdown VLM model.
        3. If VLM is used, return BOTH:
            - flattened text (for legacy compatibility)
            - structured page objects (for multimodal ingestion)
    """

    # ------------------------------------------------
    # Public API
    # ------------------------------------------------

    @with_request_id
    def extract(self, file_path: str, request_id=None) -> Optional[Dict[str, Any]]:
        ext = os.path.splitext(file_path)[1].lower()

        if ext == ".pdf":
            return self._extract_pdf_with_fallback(
                file_path,
                request_id=request_id,
            )

        if ext == ".txt":
            return {
                "text": self._extract_txt(file_path),
                "pdf_path": None,
                "source_type": "txt",
                "method": "native_txt",
                "scanned": False,
                "pages": None,
            }

        warning_id(
            f"[ContentExtractionService] Unsupported file type: {ext}",
            request_id,
        )
        return None

    # ------------------------------------------------
    # PDF Extraction (Native + VLM)
    # ------------------------------------------------

    @with_request_id
    def _extract_pdf_with_fallback(self, path: str, request_id=None) -> Dict[str, Any]:

        native_text, page_count = self._extract_pdf_native(path)
        native_text = unicodedata.normalize("NFKC", native_text).strip()

        debug_id(
            f"[ContentExtractionService] Native PDF extraction "
            f"| pages={page_count} | chars={len(native_text)}",
            request_id,
        )

        # ------------------------------------------------
        # If scanned → VLM structured extraction
        # ------------------------------------------------
        if self._is_likely_scanned(native_text, page_count):

            info_id(
                f"[ContentExtractionService] PDF appears scanned "
                f"-> Falling back to VLM structured extraction | file={path}",
                request_id,
            )

            try:
                pages: List[Dict[str, Any]] = (
                    AIModelsVLMService.extract_structured_pages_from_pdf(
                        path,
                        request_id=request_id,
                    )
                )

                if not pages:
                    warning_id(
                        "[ContentExtractionService] VLM returned empty page list.",
                        request_id,
                    )
                    return {
                        "text": "",
                        "pdf_path": path,
                        "source_type": "pdf",
                        "method": "vlm_empty",
                        "scanned": True,
                        "pages": [],
                    }

                # Flatten structured pages into legacy text blob
                combined_text = "\n\n".join(
                    unicodedata.normalize("NFKC", page.get("text", "")).strip()
                    for page in pages
                ).strip()

                return {
                    "text": combined_text,
                    "pdf_path": path,
                    "source_type": "pdf",
                    "method": "vlm_structured",
                    "scanned": True,
                    "pages": pages,   # 🔥 NEW structured multimodal output
                }

            except Exception as e:
                error_id(
                    f"[ContentExtractionService] VLM structured fallback failed: {e}",
                    request_id,
                )
                return {
                    "text": "",
                    "pdf_path": path,
                    "source_type": "pdf",
                    "method": "vlm_error",
                    "scanned": True,
                    "pages": [],
                }

        # ------------------------------------------------
        # Not scanned → use native text
        # ------------------------------------------------
        return {
            "text": native_text,
            "pdf_path": path,
            "source_type": "pdf",
            "method": "native_pymupdf",
            "scanned": False,
            "pages": None,
        }

    # ------------------------------------------------
    # Native PDF Extraction
    # ------------------------------------------------

    def _extract_pdf_native(self, path: str) -> tuple[str, int]:
        text = ""
        page_count = 0

        with fitz.open(path) as doc:
            page_count = doc.page_count

            for page in doc:
                page_text = page.get_text("text")
                if page_text:
                    text += page_text + "\n"

        return text, page_count

    # ------------------------------------------------
    # TXT Extraction
    # ------------------------------------------------

    def _extract_txt(self, path: str) -> str:
        with open(path, "r", encoding="utf-8") as f:
            return f.read().strip()

    # ------------------------------------------------
    # Scanned Detection Heuristic
    # ------------------------------------------------

    def _is_likely_scanned(self, text: str, pages: int) -> bool:
        """
        Heuristic rules:

        - Very low total text characters
        - Very low average characters per page

        Tuned to avoid false positives on short PDFs.
        """

        if pages <= 0:
            return True

        total_chars = len(text.strip())

        if total_chars < 200:
            return True

        avg_chars_per_page = total_chars // max(pages, 1)

        if avg_chars_per_page < 30:
            return True

        return False