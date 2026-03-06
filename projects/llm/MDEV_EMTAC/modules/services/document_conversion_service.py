from __future__ import annotations

import os
import shutil
import tempfile
import subprocess
from dataclasses import dataclass
from typing import Optional, Tuple

from modules.configuration.log_config import (
    debug_id,
    info_id,
    warning_id,
    error_id,
    with_request_id,
    get_request_id,
)


@dataclass
class ConversionResult:
    pdf_path: Optional[str]
    docx_path: Optional[str] = None
    temp_dir: Optional[str] = None
    source_type: str = "unknown"


class DocumentConversionService:
    """
    Converts DOC/DOCX -> PDF so downstream can:
      - extract page-aware text
      - extract images from PDFs

    HARD RULES:
    - No DB
    - No sessions
    - Returns temp paths so orchestrator can cleanup
    """

    @with_request_id
    def ensure_pdf(
        self,
        file_path: str,
        *,
        request_id: Optional[str] = None,
    ) -> ConversionResult:
        rid = request_id or get_request_id()

        ext = os.path.splitext(file_path)[1].lower()

        if ext == ".pdf":
            return ConversionResult(
                pdf_path=file_path,
                source_type="pdf",
            )

        if ext == ".docx":
            pdf_path, temp_dir = self._docx_to_pdf(file_path, rid)
            return ConversionResult(
                pdf_path=pdf_path,
                docx_path=file_path,
                temp_dir=temp_dir,
                source_type="docx->pdf",
            )

        if ext == ".doc":
            docx_path, temp_dir1 = self._doc_to_docx(file_path, rid)
            if not docx_path:
                return ConversionResult(pdf_path=None, source_type="doc->docx->pdf")

            pdf_path, temp_dir2 = self._docx_to_pdf(docx_path, rid)

            # Prefer final pdf temp dir, but keep both for cleanup
            # We'll put docx in temp_dir1 and pdf in temp_dir2.
            # Orchestrator can cleanup both.
            # If temp_dir2 is None, fall back to temp_dir1.
            return ConversionResult(
                pdf_path=pdf_path,
                docx_path=docx_path,
                temp_dir=temp_dir2 or temp_dir1,
                source_type="doc->docx->pdf",
            )

        warning_id(f"Unsupported extension for conversion: {ext}", rid)
        return ConversionResult(pdf_path=None, source_type="unsupported")

    # ---------------------------------------------------------
    # DOCX -> PDF (Windows docx2pdf)
    # ---------------------------------------------------------

    def _docx_to_pdf(self, docx_path: str, rid: str) -> Tuple[Optional[str], Optional[str]]:
        try:
            from docx2pdf import convert  # type: ignore
            import pythoncom  # type: ignore

            temp_dir = tempfile.mkdtemp(prefix="emtac_docx_pdf_")
            base = os.path.splitext(os.path.basename(docx_path))[0]
            pdf_path = os.path.join(temp_dir, f"{base}.pdf")

            debug_id(f"[DOCX->PDF] docx={docx_path} pdf={pdf_path}", rid)

            pythoncom.CoInitialize()
            try:
                convert(docx_path, pdf_path)
            finally:
                try:
                    pythoncom.CoUninitialize()
                except Exception:
                    pass

            if os.path.exists(pdf_path):
                info_id(f"[DOCX->PDF] converted ok: {pdf_path}", rid)
                return pdf_path, temp_dir

            warning_id("[DOCX->PDF] conversion produced no output file", rid)
            self._safe_rmtree(temp_dir, rid)
            return None, None

        except ImportError:
            error_id("docx2pdf/pythoncom not installed for DOCX->PDF conversion", rid)
            return None, None
        except Exception as e:
            error_id(f"[DOCX->PDF] conversion failed: {e}", rid)
            return None, None

    # ---------------------------------------------------------
    # DOC -> DOCX (LibreOffice)
    # ---------------------------------------------------------

    def _doc_to_docx(self, doc_path: str, rid: str) -> Tuple[Optional[str], Optional[str]]:
        try:
            temp_dir = tempfile.mkdtemp(prefix="emtac_doc_docx_")
            base = os.path.splitext(os.path.basename(doc_path))[0]
            out_docx = os.path.join(temp_dir, f"{base}.docx")

            debug_id(f"[DOC->DOCX] doc={doc_path} outdir={temp_dir}", rid)

            subprocess.run(
                [
                    "soffice",
                    "--headless",
                    "--convert-to",
                    "docx",
                    doc_path,
                    "--outdir",
                    temp_dir,
                ],
                check=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )

            if os.path.exists(out_docx):
                info_id(f"[DOC->DOCX] converted ok: {out_docx}", rid)
                return out_docx, temp_dir

            warning_id("[DOC->DOCX] conversion produced no output file", rid)
            self._safe_rmtree(temp_dir, rid)
            return None, None

        except FileNotFoundError:
            error_id("LibreOffice 'soffice' not found in PATH for DOC->DOCX conversion", rid)
            return None, None
        except subprocess.CalledProcessError as e:
            error_id(f"[DOC->DOCX] LibreOffice conversion failed: {e}", rid)
            return None, None
        except Exception as e:
            error_id(f"[DOC->DOCX] conversion failed: {e}", rid)
            return None, None

    # ---------------------------------------------------------
    # Cleanup helper
    # ---------------------------------------------------------

    def _safe_rmtree(self, path: str, rid: str) -> None:
        try:
            if path and os.path.exists(path):
                shutil.rmtree(path, ignore_errors=True)
                debug_id(f"[CLEANUP] removed temp dir: {path}", rid)
        except Exception as e:
            warning_id(f"[CLEANUP] failed removing temp dir {path}: {e}", rid)