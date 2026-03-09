import csv
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

        if ext == ".csv":
            return self._extract_csv(file_path, request_id=request_id)

        if ext in (".xlsx", ".xls"):
            return self._extract_excel(file_path, request_id=request_id)

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
                    "pages": pages,
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
    # CSV Extraction
    # ------------------------------------------------

    def _extract_csv(self, path: str, request_id=None) -> Optional[Dict[str, Any]]:
        try:
            with open(path, "r", encoding="utf-8", errors="replace") as f:
                reader = csv.reader(f)
                rows = [row for row in reader if any(cell.strip() for cell in row)]

            if not rows:
                warning_id(
                    f"[ContentExtractionService] CSV has no data rows: {path}",
                    request_id,
                )
                return None

            headers = rows[0]
            chunks = []

            for row in rows[1:]:
                pairs = [
                    f"{h.strip()}: {v.strip()}"
                    for h, v in zip(headers, row)
                    if h.strip() and v.strip()
                ]
                if pairs:
                    chunks.append(" | ".join(pairs))

            if not chunks:
                warning_id(
                    f"[ContentExtractionService] CSV produced no text chunks: {path}",
                    request_id,
                )
                return None

            text = "\n".join(chunks)

            debug_id(
                f"[ContentExtractionService] CSV extraction | rows={len(chunks)} | chars={len(text)}",
                request_id,
            )

            return {
                "text": text,
                "pdf_path": None,
                "source_type": "csv",
                "method": "native_csv",
                "scanned": False,
                "pages": None,
            }

        except Exception as e:
            error_id(f"[ContentExtractionService] CSV extraction failed: {e}", request_id)
            return None

    # ------------------------------------------------
    # Excel Extraction
    # ------------------------------------------------

    def _extract_excel(self, path: str, request_id=None) -> Optional[Dict[str, Any]]:
        try:
            import openpyxl

            wb = openpyxl.load_workbook(path, data_only=True)
            all_chunks = []

            for sheet_name in wb.sheetnames:
                ws = wb[sheet_name]
                rows = list(ws.iter_rows(values_only=True))

                non_empty = [r for r in rows if any(c is not None for c in r)]
                if not non_empty:
                    continue

                headers = [str(c).strip() if c is not None else "" for c in non_empty[0]]

                for row in non_empty[1:]:
                    pairs = [
                        f"{h}: {str(v).strip()}"
                        for h, v in zip(headers, row)
                        if h and v is not None and str(v).strip()
                    ]
                    if pairs:
                        all_chunks.append(f"[{sheet_name}] " + " | ".join(pairs))

            if not all_chunks:
                warning_id(
                    f"[ContentExtractionService] Excel produced no text chunks: {path}",
                    request_id,
                )
                return None

            text = "\n".join(all_chunks)

            debug_id(
                f"[ContentExtractionService] Excel extraction | sheets={len(wb.sheetnames)} | rows={len(all_chunks)} | chars={len(text)}",
                request_id,
            )

            return {
                "text": text,
                "pdf_path": None,
                "source_type": "xlsx",
                "method": "native_excel",
                "scanned": False,
                "pages": None,
            }

        except Exception as e:
            error_id(f"[ContentExtractionService] Excel extraction failed: {e}", request_id)
            return None

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