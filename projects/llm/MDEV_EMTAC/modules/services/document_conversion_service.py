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
    Converts office-style files into PDF for downstream extraction.

    Supported behavior:
      - PDF  -> passthrough
      - DOCX -> PDF
      - DOC  -> DOCX -> PDF

    Platform behavior:
      - Windows:
          * DOCX -> PDF via docx2pdf / Word COM
          * DOC  -> DOCX via Word COM, then DOCX -> PDF
      - Non-Windows:
          * DOCX -> PDF via LibreOffice
          * DOC  -> DOCX via LibreOffice, then DOCX -> PDF

    HARD RULES:
      - No DB
      - No sessions
      - No commit / rollback
      - Return temp paths so orchestrators can clean up
    """

    # ---------------------------------------------------------
    # Public API
    # ---------------------------------------------------------

    @with_request_id
    def ensure_pdf(
        self,
        file_path: str,
        *,
        request_id: Optional[str] = None,
    ) -> ConversionResult:
        rid = request_id or get_request_id()

        if not file_path:
            warning_id("[DocumentConversionService] No file_path provided", rid)
            return ConversionResult(pdf_path=None, source_type="missing_path")

        if not os.path.exists(file_path):
            warning_id(
                f"[DocumentConversionService] File does not exist: {file_path}",
                rid,
            )
            return ConversionResult(pdf_path=None, source_type="missing_file")

        ext = os.path.splitext(file_path)[1].lower()

        debug_id(
            f"[DocumentConversionService] ensure_pdf requested | ext={ext} | file={file_path}",
            rid,
        )

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
                source_type="docx->pdf" if pdf_path else "docx->pdf-failed",
            )

        if ext == ".doc":
            docx_path, temp_dir_1 = self._doc_to_docx(file_path, rid)
            if not docx_path:
                return ConversionResult(
                    pdf_path=None,
                    docx_path=None,
                    temp_dir=temp_dir_1,
                    source_type="doc->docx-failed",
                )

            pdf_path, temp_dir_2 = self._docx_to_pdf(docx_path, rid)

            # Keep the temp dir that actually contains the generated artifacts.
            final_temp_dir = temp_dir_2 or temp_dir_1

            return ConversionResult(
                pdf_path=pdf_path,
                docx_path=docx_path,
                temp_dir=final_temp_dir,
                source_type="doc->docx->pdf" if pdf_path else "doc->pdf-failed",
            )

        warning_id(
            f"[DocumentConversionService] Unsupported extension for conversion: {ext}",
            rid,
        )
        return ConversionResult(
            pdf_path=None,
            source_type="unsupported",
        )

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

    def _docx_to_pdf(
        self,
        docx_path: str,
        rid: str,
    ) -> Tuple[Optional[str], Optional[str]]:
        temp_dir = tempfile.mkdtemp(prefix="emtac_docx_pdf_")
        base = os.path.splitext(os.path.basename(docx_path))[0]
        pdf_path = os.path.join(temp_dir, f"{base}.pdf")

        debug_id(f"[DOCX->PDF] docx={docx_path} pdf={pdf_path}", rid)

        if self._is_windows():
            return self._docx_to_pdf_windows(docx_path, pdf_path, temp_dir, rid)

        return self._docx_to_pdf_libreoffice(docx_path, pdf_path, temp_dir, rid)

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
            error_id(
                "[DOCX->PDF] LibreOffice 'soffice' not found in PATH",
                rid,
            )
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
        """
        Uses docx2pdf, which relies on installed Microsoft Word on Windows.
        """
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

        except ImportError as e:
            error_id(
                f"[DOCX->PDF] docx2pdf/pythoncom not installed: {e}",
                rid,
            )
            self._safe_rmtree(temp_dir, rid)
            return None, None
        except Exception as e:
            error_id(f"[DOCX->PDF] docx2pdf conversion failed: {e}", rid)
            self._safe_rmtree(temp_dir, rid)
            return None, None

    # ---------------------------------------------------------
    # DOC -> DOCX
    # ---------------------------------------------------------

    def _doc_to_docx(
        self,
        doc_path: str,
        rid: str,
    ) -> Tuple[Optional[str], Optional[str]]:
        if self._is_windows():
            return self._doc_to_docx_windows(doc_path, rid)

        return self._doc_to_docx_libreoffice(doc_path, rid)

    def _doc_to_docx_windows(
        self,
        doc_path: str,
        rid: str,
    ) -> Tuple[Optional[str], Optional[str]]:
        """
        Converts legacy .doc -> .docx using Microsoft Word COM automation.

        Requirements:
          - pywin32 installed
          - Microsoft Word installed
        """
        temp_dir = tempfile.mkdtemp(prefix="emtac_doc_docx_")
        base = os.path.splitext(os.path.basename(doc_path))[0]
        out_docx = os.path.join(temp_dir, f"{base}.docx")

        debug_id(f"[DOC->DOCX] Windows Word COM | doc={doc_path} out={out_docx}", rid)

        word = None
        doc = None

        try:
            import pythoncom  # type: ignore
            import win32com.client  # type: ignore

            pythoncom.CoInitialize()

            # Word constants
            wdFormatXMLDocument = 16

            word = win32com.client.DispatchEx("Word.Application")
            word.Visible = False
            word.DisplayAlerts = 0

            doc = word.Documents.Open(doc_path, ReadOnly=True)
            doc.SaveAs(out_docx, FileFormat=wdFormatXMLDocument)
            doc.Close(False)
            doc = None

            word.Quit()
            word = None

            try:
                pythoncom.CoUninitialize()
            except Exception:
                pass

            if os.path.exists(out_docx):
                info_id(f"[DOC->DOCX] Word COM converted ok: {out_docx}", rid)
                return out_docx, temp_dir

            warning_id("[DOC->DOCX] Word COM produced no output file", rid)
            self._safe_rmtree(temp_dir, rid)
            return None, None

        except ImportError as e:
            error_id(
                f"[DOC->DOCX] pywin32/pythoncom not installed for Word COM conversion: {e}",
                rid,
            )
            self._safe_word_cleanup(word, doc, rid)
            self._safe_rmtree(temp_dir, rid)
            return None, None

        except Exception as e:
            error_id(f"[DOC->DOCX] Word COM conversion failed: {e}", rid)
            self._safe_word_cleanup(word, doc, rid)
            self._safe_rmtree(temp_dir, rid)
            return None, None

    def _doc_to_docx_libreoffice(
        self,
        doc_path: str,
        rid: str,
    ) -> Tuple[Optional[str], Optional[str]]:
        temp_dir = tempfile.mkdtemp(prefix="emtac_doc_docx_")
        base = os.path.splitext(os.path.basename(doc_path))[0]
        out_docx = os.path.join(temp_dir, f"{base}.docx")

        debug_id(f"[DOC->DOCX] LibreOffice | doc={doc_path} outdir={temp_dir}", rid)

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
                info_id(f"[DOC->DOCX] LibreOffice converted ok: {out_docx}", rid)
                return out_docx, temp_dir

            warning_id("[DOC->DOCX] LibreOffice produced no output file", rid)
            self._safe_rmtree(temp_dir, rid)
            return None, None

        except FileNotFoundError:
            error_id(
                "[DOC->DOCX] LibreOffice 'soffice' not found in PATH",
                rid,
            )
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
    # Cleanup helpers
    # ---------------------------------------------------------

    def _safe_word_cleanup(self, word, doc, rid: str) -> None:
        try:
            if doc is not None:
                try:
                    doc.Close(False)
                except Exception:
                    pass
        except Exception:
            pass

        try:
            if word is not None:
                try:
                    word.Quit()
                except Exception:
                    pass
        except Exception:
            pass

        try:
            import pythoncom  # type: ignore

            try:
                pythoncom.CoUninitialize()
            except Exception:
                pass
        except Exception:
            pass

        debug_id("[DOC->DOCX] Word cleanup completed", rid)

    def _safe_rmtree(self, path: str, rid: str) -> None:
        try:
            if path and os.path.exists(path):
                shutil.rmtree(path, ignore_errors=True)
                debug_id(f"[CLEANUP] removed temp dir: {path}", rid)
        except Exception as e:
            warning_id(f"[CLEANUP] failed removing temp dir {path}: {e}", rid)