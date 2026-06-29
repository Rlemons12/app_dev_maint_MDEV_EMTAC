import os
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


SUPPORTED_EXTENSIONS = {".dwg", ".slddrw", ".sldprt", ".sldasm"}
NATIVE_EXTENSIONS = {".slddrw", ".sldprt", ".sldasm"}


@dataclass
class CommandResult:
    command: list[str]
    returncode: int
    elapsed_s: float
    stdout_tail: str
    stderr_tail: str

    @property
    def success(self) -> bool:
        return self.returncode == 0


class DrawingToPDFService:
    def __init__(self, python_executable: Optional[str] = None) -> None:
        self.python_executable = python_executable or sys.executable
        self.app_dir = Path(__file__).resolve().parent

    @staticmethod
    def normalize_path(path_value: str) -> str:
        return os.path.abspath(os.path.expanduser(path_value))

    @staticmethod
    def extension_for(source_file: str) -> str:
        return Path(source_file).suffix.lower()

    def validate_supported_file(self, source_file: str) -> None:
        extension = self.extension_for(source_file)
        if extension not in SUPPORTED_EXTENSIONS:
            raise ValueError(f"Unsupported extension: {extension}")

    def build_single_command(
        self,
        source_file: str,
        output_file: str,
        quality: int,
        timeout: int,
        visible: bool,
        background_mode: str = "white",
    ) -> list[str]:
        extension = self.extension_for(source_file)
        if extension == ".dwg":
            script = self.app_dir / "solidwk_prt.py"
            command = [self.python_executable, str(script), "single", source_file, output_file]
            if background_mode == "black":
                command.append("--black-background")
            return command

        if extension in NATIVE_EXTENSIONS:
            script = self.app_dir / "solidworks_pdf.py"
            command = [
                self.python_executable,
                str(script),
                "single",
                source_file,
                output_file,
                "--quality",
                str(quality),
                "--timeout",
                str(timeout),
            ]
            if visible:
                command.append("--visible")
            if background_mode == "black":
                command.append("--black-background")
            return command

        raise ValueError(f"Unsupported extension: {extension}")

    def build_folder_command(
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
    ) -> list[str]:
        script = self.app_dir / "solidworks_batch_router.py"
        command = [
            self.python_executable,
            str(script),
            input_folder,
            output_folder,
            "--quality",
            str(quality),
            "--timeout",
            str(timeout),
            "--sldprocmon-check-interval",
            str(sldprocmon_check_interval),
            "--cpu-target-percent",
            str(cpu_target_percent),
            "--cpu-sample-seconds",
            str(cpu_sample_seconds),
            "--cpu-throttle-max-wait",
            str(cpu_throttle_max_wait),
        ]
        if not recursive:
            command.append("--non-recursive")
        if visible:
            command.append("--visible")
        if keep_going:
            command.append("--keep-going")
        if max_files is not None:
            command.extend(["--max-files", str(max_files)])
        if all_files:
            command.append("--all-files")
        if dry_run:
            command.append("--dry-run")
        if background_mode == "black":
            command.append("--black-background")
        return command

    @staticmethod
    def run_command(command: list[str], timeout: Optional[int] = None) -> CommandResult:
        start = time.monotonic()
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        stdout, stderr = process.communicate(timeout=timeout)
        elapsed = round(time.monotonic() - start, 2)
        return CommandResult(
            command=command,
            returncode=process.returncode,
            elapsed_s=elapsed,
            stdout_tail=stdout[-4000:],
            stderr_tail=stderr[-4000:],
        )
