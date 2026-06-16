from __future__ import annotations

from typing import Any, Optional

from modules.database_manager.orchestrators.image_artifact_cleanup_orchestrator import (
    ImageArtifactCleanupOrchestrator,
)
from modules.database_manager.services.image_artifact_cleanup_service import (
    ImageArtifactCleanupPolicy,
)


class ImageArtifactCleanupCoordinator:
    """
    Coordinator for image artifact cleanup.

    Responsibilities:
      - Validate high-level user/runner input
      - Build ImageArtifactCleanupPolicy
      - Call ImageArtifactCleanupOrchestrator
      - Return normalized result

    Rules:
      - Does NOT open sessions
      - Does NOT commit
      - Does NOT rollback
      - Does NOT use ORM directly
      - Does NOT contain SQL
    """

    def __init__(
        self,
        *,
        orchestrator: Optional[ImageArtifactCleanupOrchestrator] = None,
    ) -> None:
        self.orchestrator = orchestrator or ImageArtifactCleanupOrchestrator()

    def run(
        self,
        *,
        dry_run: bool = True,
        batch_size: int = 500,
        max_images: int | None = None,
        min_file_bytes: int = 5_000,
        min_width: int = 80,
        min_height: int = 80,
        min_area: int = 10_000,
        include_missing_files: bool = False,
        allow_delete_protected: bool = False,
        protect_document_images: bool = False,
        quarantine_files: bool = False,
        delete_files: bool = False,
        quarantine_dir: str | None = None,
        report_path: str | None = None,
        request_id: str | None = None,
    ) -> dict[str, Any]:
        self._validate_options(
            batch_size=batch_size,
            max_images=max_images,
            min_file_bytes=min_file_bytes,
            min_width=min_width,
            min_height=min_height,
            min_area=min_area,
            quarantine_files=quarantine_files,
            delete_files=delete_files,
        )

        policy = ImageArtifactCleanupPolicy(
            min_file_bytes=min_file_bytes,
            min_width=min_width,
            min_height=min_height,
            min_area=min_area,
            include_missing_files=include_missing_files,
            allow_delete_protected=allow_delete_protected,
            protect_document_images=protect_document_images,
        )

        return self.orchestrator.run_cleanup(
            policy=policy,
            dry_run=dry_run,
            batch_size=batch_size,
            max_images=max_images,
            report_path=report_path,
            delete_files=delete_files,
            quarantine_files=quarantine_files,
            quarantine_dir=quarantine_dir,
            request_id=request_id,
        )

    @staticmethod
    def _validate_options(
        *,
        batch_size: int,
        max_images: int | None,
        min_file_bytes: int,
        min_width: int,
        min_height: int,
        min_area: int,
        quarantine_files: bool,
        delete_files: bool,
    ) -> None:
        if batch_size <= 0:
            raise ValueError("batch_size must be greater than 0.")

        if max_images is not None and max_images <= 0:
            raise ValueError("max_images must be greater than 0 when provided.")

        if min_file_bytes < 0:
            raise ValueError("min_file_bytes cannot be negative.")

        if min_width < 0:
            raise ValueError("min_width cannot be negative.")

        if min_height < 0:
            raise ValueError("min_height cannot be negative.")

        if min_area < 0:
            raise ValueError("min_area cannot be negative.")

        if quarantine_files and delete_files:
            raise ValueError("Use either quarantine_files or delete_files, not both.")