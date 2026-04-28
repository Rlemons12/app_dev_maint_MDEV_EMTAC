import os
import time
from pathlib import Path
from typing import Optional

from services.drawing_to_pdf_service import DrawingToPDFService
from services.pdf_postprocess_service import PdfPostprocessService


class DrawingToPDFOrchestrator:
    def __init__(
        self,
        service: Optional[DrawingToPDFService] = None,
        postprocess_service: Optional[PdfPostprocessService] = None,
    ) -> None:
        self.service = service or DrawingToPDFService()
        self.postprocess_service = postprocess_service or PdfPostprocessService()

    @staticmethod
    def default_single_output_path(source_file: str, output_file: Optional[str]) -> str:
        if output_file:
            return os.path.abspath(os.path.expanduser(output_file))
        return os.path.splitext(os.path.abspath(os.path.expanduser(source_file)))[0] + ".pdf"

    def run_single(
        self,
        source_file: str,
        output_file: Optional[str],
        quality: int,
        timeout: int,
        visible: bool,
        background_mode: str = "white",
    ) -> dict[str, object]:
        source_path = self.service.normalize_path(source_file)
        if not os.path.isfile(source_path):
            raise ValueError(f"Input file does not exist: {source_path}")

        self.service.validate_supported_file(source_path)

        output_path = self.default_single_output_path(source_path, output_file)
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
            applied, decision_black = self.postprocess_service.auto_apply_black_background_if_needed(output_path)
            payload["background_mode"] = "black" if decision_black else "white"
            payload["background_auto_applied"] = applied
        else:
            payload["background_mode"] = background_mode
            payload["background_auto_applied"] = False
        return payload

    def run_folder(
        self,
        input_folder: str,
        output_folder: str,
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
        input_path = self.service.normalize_path(input_folder)
        output_path = self.service.normalize_path(output_folder)

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
                    applied, decision_black = self.postprocess_service.auto_apply_black_background_if_needed(pdf_path)
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
