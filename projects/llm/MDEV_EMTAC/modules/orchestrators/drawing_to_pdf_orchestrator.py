import os
import time
from pathlib import Path
from typing import Optional

from modules.configuration.config import DATABASE_DRAWING
from services.drawing_to_pdf_service import DrawingToPDFService
from services.pdf_postprocess_service import PdfPostprocessService


class DrawingToPDFOrchestrator:
    """
    Orchestrates drawing-to-PDF conversion.

    Uses DATABASE_DRAWING from modules.configuration.config as the drawing root.

    Expected .env value:

        DATABASE_DRAWING=E:\\emtac\\Database\\DB_DRAWINNG

    Relative paths are resolved inside DATABASE_DRAWING.

    Examples:
        source_file="my_drawing.dwg"
        resolves to:
        E:\\emtac\\Database\\DB_DRAWINNG\\my_drawing.dwg

        output_file="my_drawing.pdf"
        resolves to:
        E:\\emtac\\Database\\DB_DRAWINNG\\my_drawing.pdf
    """

    def __init__(
        self,
        service: Optional[DrawingToPDFService] = None,
        postprocess_service: Optional[PdfPostprocessService] = None,
        drawing_root: Optional[str] = None,
    ) -> None:
        self.service = service or DrawingToPDFService()
        self.postprocess_service = postprocess_service or PdfPostprocessService()
        self.drawing_root = self._init_drawing_root(drawing_root)

    # ------------------------------------------------------------------
    # PATH HELPERS
    # ------------------------------------------------------------------

    @staticmethod
    def _init_drawing_root(drawing_root: Optional[str] = None) -> str:
        """
        Initialize and validate the drawing root folder.

        Priority:
            1. Explicit drawing_root argument
            2. DATABASE_DRAWING from config.py
        """
        configured_root = drawing_root or DATABASE_DRAWING

        if not configured_root or not str(configured_root).strip():
            raise ValueError(
                "DATABASE_DRAWING is not configured. "
                "Add this to your .env file: "
                "DATABASE_DRAWING=E:\\emtac\\Database\\DB_DRAWINNG"
            )

        root_path = os.path.abspath(os.path.expanduser(str(configured_root)))
        os.makedirs(root_path, exist_ok=True)

        return root_path

    def resolve_drawing_path(self, file_path: str) -> str:
        """
        Resolve a file path against DATABASE_DRAWING.

        If file_path is absolute:
            return normalized absolute path

        If file_path is relative:
            return DATABASE_DRAWING/file_path
        """
        if not file_path or not str(file_path).strip():
            raise ValueError("file_path is required.")

        expanded_path = os.path.expanduser(str(file_path))

        if os.path.isabs(expanded_path):
            return os.path.abspath(expanded_path)

        return os.path.abspath(os.path.join(self.drawing_root, expanded_path))

    def default_single_output_path(
        self,
        source_file: str,
        output_file: Optional[str]
    ) -> str:
        """
        Determine the output PDF path for a single drawing conversion.

        If output_file is provided:
            - absolute output_file stays absolute
            - relative output_file is placed under DATABASE_DRAWING

        If output_file is not provided:
            - output PDF is created next to the resolved source file
        """
        if output_file:
            return self.resolve_drawing_path(output_file)

        resolved_source = self.resolve_drawing_path(source_file)
        return os.path.splitext(resolved_source)[0] + ".pdf"

    def default_folder_output_path(
        self,
        output_folder: Optional[str]
    ) -> str:
        """
        Determine the folder output path.

        If output_folder is provided:
            - absolute output_folder stays absolute
            - relative output_folder is placed under DATABASE_DRAWING

        If output_folder is not provided:
            - output folder defaults to DATABASE_DRAWING
        """
        if output_folder:
            return self.resolve_drawing_path(output_folder)

        return self.drawing_root

    # ------------------------------------------------------------------
    # SINGLE FILE CONVERSION
    # ------------------------------------------------------------------

    def run_single(
        self,
        source_file: str,
        output_file: Optional[str],
        quality: int,
        timeout: int,
        visible: bool,
        background_mode: str = "white",
    ) -> dict[str, object]:
        """
        Convert one drawing file to PDF.
        """
        source_path = self.resolve_drawing_path(source_file)
        source_path = self.service.normalize_path(source_path)

        if not os.path.isfile(source_path):
            raise ValueError(f"Input file does not exist: {source_path}")

        self.service.validate_supported_file(source_path)

        output_path = self.default_single_output_path(source_path, output_file)
        output_path = self.service.normalize_path(output_path)

        output_dir = os.path.dirname(output_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)

        command = self.service.build_single_command(
            source_file=source_path,
            output_file=output_path,
            quality=quality,
            timeout=timeout,
            visible=visible,
            background_mode=background_mode,
        )

        result = self.service.run_command(command, timeout=None)

        extension = Path(source_path).suffix.lower()
        version_mismatch = bool(extension != ".dwg" and result.returncode == 2)

        payload = {
            "mode": "single",
            "drawing_root": self.drawing_root,
            "input_file": source_path,
            "output_pdf": output_path,
            "returncode": result.returncode,
            "success": result.success,
            "version_mismatch": version_mismatch,
            "elapsed_s": result.elapsed_s,
            "command": result.command,
            "stdout_tail": result.stdout_tail,
            "stderr_tail": result.stderr_tail,
        }

        if result.success and background_mode == "auto":
            applied, decision_black = (
                self.postprocess_service.auto_apply_black_background_if_needed(output_path)
            )
            payload["background_mode"] = "black" if decision_black else "white"
            payload["background_auto_applied"] = applied
        else:
            payload["background_mode"] = background_mode
            payload["background_auto_applied"] = False

        return payload

    # ------------------------------------------------------------------
    # FOLDER CONVERSION
    # ------------------------------------------------------------------

    def run_folder(
        self,
        input_folder: str,
        output_folder: Optional[str],
        recursive: bool,
        quality: int,
        timeout: int,
        visible: bool,
        keep_going: bool,
        max_files: Optional[int],
        all_files: bool,
        dry_run: bool,
        sldprocmon_check_interval: int,
        cpu_target_percent: float,
        cpu_sample_seconds: float,
        cpu_throttle_max_wait: float,
        background_mode: str = "white",
    ) -> dict[str, object]:
        """
        Convert a folder of drawings to PDFs.

        Relative input_folder and output_folder values are resolved inside DATABASE_DRAWING.
        """
        input_path = self.resolve_drawing_path(input_folder)
        input_path = self.service.normalize_path(input_path)

        output_path = self.default_folder_output_path(output_folder)
        output_path = self.service.normalize_path(output_path)

        if not os.path.isdir(input_path):
            raise ValueError(f"Input folder does not exist: {input_path}")

        os.makedirs(output_path, exist_ok=True)

        started_at = time.time()

        command = self.service.build_folder_command(
            input_folder=input_path,
            output_folder=output_path,
            recursive=recursive,
            quality=quality,
            timeout=timeout,
            visible=visible,
            keep_going=keep_going,
            max_files=max_files,
            all_files=all_files,
            dry_run=dry_run,
            sldprocmon_check_interval=sldprocmon_check_interval,
            cpu_target_percent=cpu_target_percent,
            cpu_sample_seconds=cpu_sample_seconds,
            cpu_throttle_max_wait=cpu_throttle_max_wait,
            background_mode=background_mode,
        )

        result = self.service.run_command(command, timeout=None)

        payload = {
            "mode": "folder",
            "drawing_root": self.drawing_root,
            "input_folder": input_path,
            "output_folder": output_path,
            "returncode": result.returncode,
            "success": result.success,
            "elapsed_s": result.elapsed_s,
            "command": result.command,
            "stdout_tail": result.stdout_tail,
            "stderr_tail": result.stderr_tail,
        }

        if background_mode == "auto" and not dry_run:
            total_checked = 0
            total_black = 0
            total_applied = 0

            for root, _, files in os.walk(output_path):
                for filename in files:
                    if not filename.lower().endswith(".pdf"):
                        continue

                    pdf_path = os.path.join(root, filename)

                    try:
                        if os.path.getmtime(pdf_path) < (started_at - 1.0):
                            continue
                    except OSError:
                        continue

                    total_checked += 1

                    applied, decision_black = (
                        self.postprocess_service.auto_apply_black_background_if_needed(pdf_path)
                    )

                    if decision_black:
                        total_black += 1

                    if applied:
                        total_applied += 1

            payload["background_mode"] = "auto"
            payload["background_auto_checked"] = total_checked
            payload["background_auto_decision_black"] = total_black
            payload["background_auto_applied"] = total_applied
        else:
            payload["background_mode"] = background_mode
            payload["background_auto_checked"] = 0
            payload["background_auto_decision_black"] = 0
            payload["background_auto_applied"] = 0

        return payload