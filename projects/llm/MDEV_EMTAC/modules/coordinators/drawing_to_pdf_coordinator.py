from typing import Optional

from orchestrators.drawing_to_pdf_orchestrator import DrawingToPDFOrchestrator


class DrawingToPDFCoordinator:
    """
    Coordinator for drawing-to-PDF conversion.

    This layer validates user/application inputs before passing work to the
    DrawingToPDFOrchestrator.

    The orchestrator is responsible for resolving relative drawing paths against
    the configured drawing root:

        DATABASE_DRAWING=E:\\emtac\\Database\\DB_DRAWINNG

    Usage examples:

        coordinator = DrawingToPDFCoordinator()

        # Looks inside DATABASE_DRAWING
        coordinator.run_single(
            source_file="my_drawing.dwg"
        )

        # Converts folder inside DATABASE_DRAWING and outputs into DATABASE_DRAWING
        coordinator.run_folder(
            input_folder="incoming_drawings"
        )

        # Absolute paths still work
        coordinator.run_single(
            source_file="E:\\emtac\\Database\\DB_DRAWINNG\\my_drawing.dwg"
        )
    """

    ALLOWED_QUALITIES = {1, 2, 3}
    ALLOWED_BACKGROUND_MODES = {"white", "black", "auto"}

    def __init__(
        self,
        orchestrator: Optional[DrawingToPDFOrchestrator] = None,
        drawing_root: Optional[str] = None,
    ) -> None:
        """
        Initialize the coordinator.

        Args:
            orchestrator:
                Optional custom DrawingToPDFOrchestrator instance.

            drawing_root:
                Optional override for the drawing root folder.
                If omitted, the orchestrator will use DATABASE_DRAWING from config.py.
        """
        if orchestrator is not None:
            self.orchestrator = orchestrator
        else:
            self.orchestrator = DrawingToPDFOrchestrator(
                drawing_root=drawing_root
            )

    # ------------------------------------------------------------------
    # VALIDATION HELPERS
    # ------------------------------------------------------------------

    @staticmethod
    def _validate_required_text(value: str, field_name: str) -> None:
        """
        Validate a required string field.
        """
        if value is None or not str(value).strip():
            raise ValueError(f"{field_name} is required")

    @classmethod
    def _validate_common(cls, quality: int, timeout: int) -> None:
        """
        Validate options shared by single-file and folder conversion.
        """
        if quality not in cls.ALLOWED_QUALITIES:
            raise ValueError("quality must be 1, 2, or 3")

        if timeout < 1:
            raise ValueError("timeout must be at least 1 second")

    @classmethod
    def _validate_background_mode(cls, background_mode: str) -> None:
        """
        Validate background conversion mode.
        """
        if background_mode not in cls.ALLOWED_BACKGROUND_MODES:
            raise ValueError("background_mode must be one of: white, black, auto")

    @classmethod
    def _validate_folder(
        cls,
        max_files: Optional[int],
        sldprocmon_check_interval: int,
        cpu_target_percent: float,
        cpu_sample_seconds: float,
        cpu_throttle_max_wait: float,
        background_mode: str,
    ) -> None:
        """
        Validate folder conversion options.
        """
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

        cls._validate_background_mode(background_mode)

    # ------------------------------------------------------------------
    # SINGLE FILE CONVERSION
    # ------------------------------------------------------------------

    def run_single(
        self,
        source_file: str,
        output_file: Optional[str] = None,
        quality: int = 2,
        timeout: int = 180,
        visible: bool = False,
        background_mode: str = "white",
    ) -> dict[str, object]:
        """
        Convert a single drawing file to PDF.

        Relative source_file values are resolved by the orchestrator using
        DATABASE_DRAWING.

        Example:
            source_file="machine_print.dwg"

        Resolves to:
            E:\\emtac\\Database\\DB_DRAWINNG\\machine_print.dwg

        If output_file is None, the PDF will be created beside the source file.

        If output_file is relative, it will also be resolved inside DATABASE_DRAWING.
        """
        self._validate_required_text(source_file, "source_file")
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

    # ------------------------------------------------------------------
    # FOLDER CONVERSION
    # ------------------------------------------------------------------

    def run_folder(
        self,
        input_folder: str,
        output_folder: Optional[str] = None,
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
        """
        Convert a folder of drawing files to PDFs.

        Relative input_folder and output_folder values are resolved by the
        orchestrator using DATABASE_DRAWING.

        If output_folder is None, output defaults to DATABASE_DRAWING.

        Example:
            input_folder="incoming_drawings"
            output_folder="converted_pdfs"

        Resolves to:
            E:\\emtac\\Database\\DB_DRAWINNG\\incoming_drawings
            E:\\emtac\\Database\\DB_DRAWINNG\\converted_pdfs
        """
        self._validate_required_text(input_folder, "input_folder")
        self._validate_common(quality=quality, timeout=timeout)
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