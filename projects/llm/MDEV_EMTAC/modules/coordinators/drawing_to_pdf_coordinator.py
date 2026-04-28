from typing import Optional

from orchestrators.drawing_to_pdf_orchestrator import DrawingToPDFOrchestrator


class DrawingToPDFCoordinator:
    def __init__(self, orchestrator: Optional[DrawingToPDFOrchestrator] = None) -> None:
        self.orchestrator = orchestrator or DrawingToPDFOrchestrator()

    @staticmethod
    def _validate_common(quality: int, timeout: int) -> None:
        if quality not in (1, 2, 3):
            raise ValueError("quality must be 1, 2, or 3")
        if timeout < 1:
            raise ValueError("timeout must be at least 1 second")

    @staticmethod
    def _validate_background_mode(background_mode: str) -> None:
        if background_mode not in {"white", "black", "auto"}:
            raise ValueError("background_mode must be one of: white, black, auto")

    @staticmethod
    def _validate_folder(
        max_files: Optional[int],
        sldprocmon_check_interval: int,
        cpu_target_percent: float,
        cpu_sample_seconds: float,
        cpu_throttle_max_wait: float,
        background_mode: str,
    ) -> None:
        if max_files is not None and max_files < 1:
            raise ValueError("max_files must be at least 1")
        if sldprocmon_check_interval < 0:
            raise ValueError("sldprocmon_check_interval must be 0 or greater")
        if cpu_target_percent < 0 or cpu_target_percent >= 100:
            raise ValueError("cpu_target_percent must be in the range 0 to less than 100")
        if cpu_sample_seconds <= 0:
            raise ValueError("cpu_sample_seconds must be greater than 0")
        if cpu_throttle_max_wait < 0:
            raise ValueError("cpu_throttle_max_wait must be 0 or greater")
        if background_mode not in {"white", "black", "auto"}:
            raise ValueError("background_mode must be one of: white, black, auto")

    def run_single(
        self,
        source_file: str,
        output_file: Optional[str] = None,
        quality: int = 2,
        timeout: int = 180,
        visible: bool = False,
        background_mode: str = "white",
    ) -> dict[str, object]:
        self._validate_common(quality=quality, timeout=timeout)
        self._validate_background_mode(background_mode)
        return self.orchestrator.run_single(
            source_file=source_file,
            output_file=output_file,
            quality=quality,
            timeout=timeout,
            visible=visible,
            background_mode=background_mode,
        )

    def run_folder(
        self,
        input_folder: str,
        output_folder: str,
        recursive: bool = True,
        quality: int = 2,
        timeout: int = 180,
        visible: bool = False,
        keep_going: bool = True,
        max_files: Optional[int] = None,
        all_files: bool = False,
        dry_run: bool = False,
        sldprocmon_check_interval: int = 5,
        cpu_target_percent: float = 85.0,
        cpu_sample_seconds: float = 0.7,
        cpu_throttle_max_wait: float = 20.0,
        background_mode: str = "white",
    ) -> dict[str, object]:
        self._validate_common(quality=quality, timeout=timeout)
        self._validate_background_mode(background_mode)
        self._validate_folder(
            max_files=max_files,
            sldprocmon_check_interval=sldprocmon_check_interval,
            cpu_target_percent=cpu_target_percent,
            cpu_sample_seconds=cpu_sample_seconds,
            cpu_throttle_max_wait=cpu_throttle_max_wait,
            background_mode=background_mode,
        )
        return self.orchestrator.run_folder(
            input_folder=input_folder,
            output_folder=output_folder,
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
