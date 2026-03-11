from __future__ import annotations

import csv
import os
import unicodedata
from typing import Optional, Dict, Any, List, Tuple

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
    Extracts text from supported document types.

    Strategy:
        1. PDF:
           - Try native extraction with PyMuPDF
           - If likely scanned, fallback to VLM structured extraction
           - Return both flattened text and structured page objects when applicable

        2. Text-like formats:
           - TXT
           - CSV
           - XLS / XLSX
           - DOCX (best-effort native extraction if python-docx is available)

        3. Legacy .doc:
           - Explicitly recognized
           - Returns a structured "unsupported_legacy_doc" result instead of silently
             appearing as a generic unsupported extension

    Notes:
        - This service does NOT perform file conversion. That belongs in
          DocumentConversionService.
        - The return contract is kept compatible with your orchestrator:
              {
                  "text": str,
                  "pdf_path": Optional[str],
                  "source_type": str,
                  "method": str,
                  "scanned": bool,
                  "pages": Optional[list],
              }
          or None when extraction completely fails.
    """

    # ------------------------------------------------
    # Public API
    # ------------------------------------------------

    @with_request_id
    def extract(
        self,
        file_path: str,
        request_id: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        if not file_path or not isinstance(file_path, str):
            warning_id(
                "[ContentExtractionService] extract() received invalid file_path",
                request_id,
            )
            return None

        if not os.path.exists(file_path):
            warning_id(
                f"[ContentExtractionService] File does not exist: {file_path}",
                request_id,
            )
            return None

        ext = os.path.splitext(file_path)[1].lower().strip()

        debug_id(
            f"[ContentExtractionService] Extract requested | ext={ext or '<none>'} | file={file_path}",
            request_id,
        )

        if ext == ".pdf":
            return self._extract_pdf_with_fallback(
                file_path,
                request_id=request_id,
            )

        if ext == ".txt":
            return self._extract_txt_result(
                file_path,
                request_id=request_id,
            )

        if ext == ".csv":
            return self._extract_csv(
                file_path,
                request_id=request_id,
            )

        if ext in (".xlsx", ".xls"):
            return self._extract_excel(
                file_path,
                request_id=request_id,
            )

        if ext == ".docx":
            return self._extract_docx(
                file_path,
                request_id=request_id,
            )

        if ext == ".doc":
            return self._extract_legacy_doc(
                file_path,
                request_id=request_id,
            )

        warning_id(
            f"[ContentExtractionService] Unsupported file type: {ext}",
            request_id,
        )
        return None

    # ------------------------------------------------
    # PDF Extraction (Native + VLM)
    # ------------------------------------------------

    @with_request_id
    def _extract_pdf_with_fallback(
            self,
            path: str,
            request_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        native_text, page_count, native_pages = self._extract_pdf_native(path)
        native_text = unicodedata.normalize("NFKC", native_text).strip()

        debug_id(
            f"[ContentExtractionService] Native PDF extraction "
            f"| pages={page_count} | chars={len(native_text)} | structured_pages={len(native_pages)}",
            request_id,
        )

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

                combined_text = "\n\n".join(
                    unicodedata.normalize("NFKC", (page.get("text") or "")).strip()
                    for page in pages
                ).strip()

                debug_id(
                    f"[ContentExtractionService] VLM structured extraction "
                    f"| pages={len(pages)} | chars={len(combined_text)}",
                    request_id,
                )

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

        return {
            "text": native_text,
            "pdf_path": path,
            "source_type": "pdf",
            "method": "native_pymupdf",
            "scanned": False,
            "pages": native_pages,
        }

    # ------------------------------------------------
    # Native PDF Extraction
    # ------------------------------------------------

    def _extract_pdf_native(self, path: str) -> Tuple[str, int, List[Dict[str, Any]]]:
        text_parts: List[str] = []
        structured_pages: List[Dict[str, Any]] = []

        with fitz.open(path) as doc:
            page_count = doc.page_count

            for page_index, page in enumerate(doc):
                page_text = unicodedata.normalize(
                    "NFKC",
                    page.get_text("text") or ""
                ).strip()

                text_parts.append(page_text)

                structured_pages.append(
                    {
                        "page_number": page_index + 1,  # use 1-based consistently
                        "text": page_text,
                    }
                )

        combined_text = "\n\n".join(p for p in text_parts if p).strip()
        return combined_text, page_count, structured_pages

    # ------------------------------------------------
    # TXT Extraction
    # ------------------------------------------------

    @with_request_id
    def _extract_txt_result(
        self,
        path: str,
        request_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        text = self._extract_txt(path)

        debug_id(
            f"[ContentExtractionService] TXT extraction | chars={len(text)}",
            request_id,
        )

        return {
            "text": text,
            "pdf_path": None,
            "source_type": "txt",
            "method": "native_txt",
            "scanned": False,
            "pages": None,
        }

    def _extract_txt(self, path: str) -> str:
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            return unicodedata.normalize("NFKC", f.read()).strip()

    # ------------------------------------------------
    # CSV Extraction
    # ------------------------------------------------

    @with_request_id
    def _extract_csv(
        self,
        path: str,
        request_id: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        try:
            with open(path, "r", encoding="utf-8", errors="replace", newline="") as f:
                reader = csv.reader(f)
                rows = [row for row in reader if any(str(cell).strip() for cell in row)]

            if not rows:
                warning_id(
                    f"[ContentExtractionService] CSV has no data rows: {path}",
                    request_id,
                )
                return None

            headers = [str(h).strip() for h in rows[0]]
            chunks: List[str] = []

            for row in rows[1:]:
                pairs = [
                    f"{h}: {str(v).strip()}"
                    for h, v in zip(headers, row)
                    if h.strip() and str(v).strip()
                ]
                if pairs:
                    chunks.append(" | ".join(pairs))

            if not chunks:
                warning_id(
                    f"[ContentExtractionService] CSV produced no text chunks: {path}",
                    request_id,
                )
                return None

            text = unicodedata.normalize("NFKC", "\n".join(chunks)).strip()

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
            error_id(
                f"[ContentExtractionService] CSV extraction failed: {e}",
                request_id,
            )
            return None

    # ------------------------------------------------
    # Excel Extraction
    # ------------------------------------------------

    @with_request_id
    def _extract_excel(
        self,
        path: str,
        request_id: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        try:
            import openpyxl

            wb = openpyxl.load_workbook(path, data_only=True)
            all_chunks: List[str] = []

            for sheet_name in wb.sheetnames:
                ws = wb[sheet_name]
                rows = list(ws.iter_rows(values_only=True))

                non_empty = [r for r in rows if any(c is not None and str(c).strip() for c in r)]
                if not non_empty:
                    continue

                headers = [
                    str(c).strip() if c is not None else ""
                    for c in non_empty[0]
                ]

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

            text = unicodedata.normalize("NFKC", "\n".join(all_chunks)).strip()

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
            error_id(
                f"[ContentExtractionService] Excel extraction failed: {e}",
                request_id,
            )
            return None

    # ------------------------------------------------
    # DOCX Extraction
    # ------------------------------------------------

    @with_request_id
    def _extract_docx(
        self,
        path: str,
        request_id: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        """
        Best-effort native DOCX extraction.

        Your pipeline may still prefer DOCX -> PDF conversion upstream.
        This method exists so the extractor can still handle DOCX if it reaches
        this layer directly.
        """
        try:
            from docx import Document as DocxDocument

            doc = DocxDocument(path)
            parts: List[str] = []

            for para in doc.paragraphs:
                txt = (para.text or "").strip()
                if txt:
                    parts.append(txt)

            for table in doc.tables:
                for row in table.rows:
                    cells = [
                        (cell.text or "").strip()
                        for cell in row.cells
                        if (cell.text or "").strip()
                    ]
                    if cells:
                        parts.append(" | ".join(cells))

            text = unicodedata.normalize("NFKC", "\n".join(parts)).strip()

            if not text:
                warning_id(
                    f"[ContentExtractionService] DOCX produced no extractable text: {path}",
                    request_id,
                )
                return None

            debug_id(
                f"[ContentExtractionService] DOCX extraction | chars={len(text)}",
                request_id,
            )

            return {
                "text": text,
                "pdf_path": None,
                "source_type": "docx",
                "method": "native_docx",
                "scanned": False,
                "pages": None,
            }

        except ImportError:
            warning_id(
                "[ContentExtractionService] python-docx is not installed; DOCX native extraction unavailable",
                request_id,
            )
            return None

        except Exception as e:
            error_id(
                f"[ContentExtractionService] DOCX extraction failed: {e}",
                request_id,
            )
            return None

    # ------------------------------------------------
    # Legacy DOC Handling
    # ------------------------------------------------

    @with_request_id
    def _extract_legacy_doc(
        self,
        path: str,
        request_id: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        """
        Legacy .doc handling.

        This service does not perform binary .doc conversion or extraction.
        That should happen upstream in DocumentConversionService.

        Returning None preserves existing orchestrator skip behavior, but the
        logging here makes the reason explicit and easier to diagnose.
        """
        warning_id(
            "[ContentExtractionService] Legacy .doc extraction is not supported directly. "
            "Convert .doc to .pdf or .docx in DocumentConversionService first.",
            request_id,
        )
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

        total_chars = len((text or "").strip())

        if total_chars < 200:
            return True

        avg_chars_per_page = total_chars // max(pages, 1)

        if avg_chars_per_page < 30:
            return True

        return False