from __future__ import annotations

import os
import shutil
import tempfile
import subprocess
import platform
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

    PLATFORM RULES:
    - PDF: passthrough
    - DOCX:
        - Windows -> docx2pdf
        - non-Windows -> LibreOffice (soffice)
    - DOC:
        - Windows -> not converted here unless a Windows-native converter is added
        - non-Windows -> LibreOffice DOC -> DOCX -> PDF
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
            if self._is_windows():
                warning_id(
                    "[DOC->PDF] Windows .doc conversion is not enabled in this service; "
                    "LibreOffice is optional on Windows and will not be required here",
                    rid,
                )
                return ConversionResult(
                    pdf_path=None,
                    docx_path=None,
                    temp_dir=None,
                    source_type="doc-windows-skipped",
                )

            docx_path, temp_dir1 = self._doc_to_docx(file_path, rid)
            if not docx_path:
                return ConversionResult(
                    pdf_path=None,
                    docx_path=None,
                    temp_dir=temp_dir1,
                    source_type="doc->docx-failed",
                )

            pdf_path, temp_dir2 = self._docx_to_pdf(docx_path, rid)

            return ConversionResult(
                pdf_path=pdf_path,
                docx_path=docx_path,
                temp_dir=temp_dir2 or temp_dir1,
                source_type="doc->docx->pdf",
            )

        warning_id(f"Unsupported extension for conversion: {ext}", rid)
        return ConversionResult(pdf_path=None, source_type="unsupported")

    # ---------------------------------------------------------
    # Platform helpers
    # ---------------------------------------------------------

    def _is_windows(self) -> bool:
        return platform.system().lower() == "windows"

    def _is_non_windows(self) -> bool:
        return not self._is_windows()

    # ---------------------------------------------------------
    # DOCX -> PDF
    # ---------------------------------------------------------

    def _docx_to_pdf(self, docx_path: str, rid: str) -> Tuple[Optional[str], Optional[str]]:
        temp_dir = tempfile.mkdtemp(prefix="emtac_docx_pdf_")
        base = os.path.splitext(os.path.basename(docx_path))[0]
        pdf_path = os.path.join(temp_dir, f"{base}.pdf")

        debug_id(f"[DOCX->PDF] docx={docx_path} pdf={pdf_path}", rid)

        if self._is_non_windows():
            return self._docx_to_pdf_libreoffice(docx_path, pdf_path, temp_dir, rid)

        return self._docx_to_pdf_windows(docx_path, pdf_path, temp_dir, rid)

    def _docx_to_pdf_libreoffice(
        self,
        docx_path: str,
        pdf_path: str,
        temp_dir: str,
        rid: str,
    ) -> Tuple[Optional[str], Optional[str]]:
        try:
            subprocess.run(
                [
                    "soffice",
                    "--headless",
                    "--convert-to",
                    "pdf",
                    docx_path,
                    "--outdir",
                    temp_dir,
                ],
                check=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )

            if os.path.exists(pdf_path):
                info_id(f"[DOCX->PDF] LibreOffice converted ok: {pdf_path}", rid)
                return pdf_path, temp_dir

            warning_id("[DOCX->PDF] LibreOffice produced no output file", rid)
            self._safe_rmtree(temp_dir, rid)
            return None, None

        except FileNotFoundError:
            error_id("LibreOffice 'soffice' not found in PATH for DOCX->PDF conversion", rid)
            self._safe_rmtree(temp_dir, rid)
            return None, None
        except subprocess.CalledProcessError as e:
            error_id(f"[DOCX->PDF] LibreOffice conversion failed: {e}", rid)
            self._safe_rmtree(temp_dir, rid)
            return None, None
        except Exception as e:
            error_id(f"[DOCX->PDF] conversion failed: {e}", rid)
            self._safe_rmtree(temp_dir, rid)
            return None, None

    def _docx_to_pdf_windows(
        self,
        docx_path: str,
        pdf_path: str,
        temp_dir: str,
        rid: str,
    ) -> Tuple[Optional[str], Optional[str]]:
        try:
            from docx2pdf import convert  # type: ignore
            import pythoncom  # type: ignore

            pythoncom.CoInitialize()
            try:
                convert(docx_path, pdf_path)
            finally:
                try:
                    pythoncom.CoUninitialize()
                except Exception:
                    pass

            if os.path.exists(pdf_path):
                info_id(f"[DOCX->PDF] docx2pdf converted ok: {pdf_path}", rid)
                return pdf_path, temp_dir

            warning_id("[DOCX->PDF] docx2pdf produced no output file", rid)
            self._safe_rmtree(temp_dir, rid)
            return None, None

        except ImportError:
            error_id("docx2pdf/pythoncom not installed for DOCX->PDF conversion", rid)
            self._safe_rmtree(temp_dir, rid)
            return None, None
        except Exception as e:
            error_id(f"[DOCX->PDF] docx2pdf conversion failed: {e}", rid)
            self._safe_rmtree(temp_dir, rid)
            return None, None

    # ---------------------------------------------------------
    # DOC -> DOCX
    # ---------------------------------------------------------

    def _doc_to_docx(self, doc_path: str, rid: str) -> Tuple[Optional[str], Optional[str]]:
        """
        DOC -> DOCX conversion is only handled here for non-Windows systems
        using LibreOffice.

        On Windows, this service intentionally does not require LibreOffice.
        """
        if self._is_windows():
            warning_id(
                "[DOC->DOCX] Windows path intentionally disabled in this service; "
                "LibreOffice will not be required on Windows",
                rid,
            )
            return None, None

        temp_dir = tempfile.mkdtemp(prefix="emtac_doc_docx_")
        base = os.path.splitext(os.path.basename(doc_path))[0]
        out_docx = os.path.join(temp_dir, f"{base}.docx")

        debug_id(f"[DOC->DOCX] doc={doc_path} outdir={temp_dir}", rid)

        try:
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
            self._safe_rmtree(temp_dir, rid)
            return None, None
        except subprocess.CalledProcessError as e:
            error_id(f"[DOC->DOCX] LibreOffice conversion failed: {e}", rid)
            self._safe_rmtree(temp_dir, rid)
            return None, None
        except Exception as e:
            error_id(f"[DOC->DOCX] conversion failed: {e}", rid)
            self._safe_rmtree(temp_dir, rid)
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