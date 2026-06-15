from __future__ import annotations

import json
import shutil
import uuid
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from modules.configuration.config_env import DatabaseConfig

try:
    from modules.configuration.log_config import info_id, warning_id, error_id
except Exception:
    def info_id(message: str, request_id: Optional[str] = None) -> None:
        print(f"[INFO] [{request_id}] {message}")

    def warning_id(message: str, request_id: Optional[str] = None) -> None:
        print(f"[WARNING] [{request_id}] {message}")

    def error_id(message: str, request_id: Optional[str] = None) -> None:
        print(f"[ERROR] [{request_id}] {message}")


from modules.database_manager.services.image_artifact_cleanup_service import (
    ImageArtifactCandidate,
    ImageArtifactCleanupPolicy,
    ImageArtifactCleanupService,
)


@dataclass
class ImageArtifactCleanupResult:
    request_id: str
    dry_run: bool
    scanned_images: int
    candidate_count: int
    deleted_database_rows: int
    file_action: str
    file_action_success_count: int
    file_action_error_count: int
    report_path: str
    candidates: List[Dict[str, Any]]
    file_action_errors: List[Dict[str, Any]]


class ImageArtifactCleanupOrchestrator:
    """
    Orchestrator for image artifact database cleanup.

    Responsibilities:
      - Own DB session lifecycle
      - Own commit / rollback
      - Coordinate cleanup service
      - Write cleanup report
      - Perform file actions after successful DB commit

    Rules:
      - No SQL here
      - No ORM table manipulation here
      - Service receives the session
      - Service does not commit/rollback
    """

    def __init__(
        self,
        *,
        db: Optional[DatabaseConfig] = None,
        cleanup_service: Optional[ImageArtifactCleanupService] = None,
    ) -> None:
        self.db = db or DatabaseConfig()
        self.cleanup_service = cleanup_service or ImageArtifactCleanupService()

    def run_cleanup(
        self,
        *,
        policy: Optional[ImageArtifactCleanupPolicy] = None,
        dry_run: bool = True,
        batch_size: int = 500,
        max_images: Optional[int] = None,
        report_path: Optional[str] = None,
        delete_files: bool = False,
        quarantine_files: bool = False,
        quarantine_dir: Optional[str] = None,
        request_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        request_id = request_id or f"image-cleanup-{uuid.uuid4()}"
        policy = policy or ImageArtifactCleanupPolicy()

        self._validate_run_options(
            batch_size=batch_size,
            max_images=max_images,
            delete_files=delete_files,
            quarantine_files=quarantine_files,
        )

        if dry_run and (delete_files or quarantine_files):
            warning_id(
                "dry_run=True; file actions will be skipped.",
                request_id=request_id,
            )

        resolved_report_path = self._resolve_report_path(report_path)

        info_id(
            (
                "Starting image artifact cleanup. "
                f"dry_run={dry_run}, "
                f"batch_size={batch_size}, "
                f"max_images={max_images}, "
                f"delete_files={delete_files}, "
                f"quarantine_files={quarantine_files}"
            ),
            request_id=request_id,
        )

        candidates: List[ImageArtifactCandidate] = []
        deleted_candidates: List[ImageArtifactCandidate] = []

        deleted_database_rows = 0
        scanned_images = 0
        last_image_id = 0

        with self.db.main_session() as session:
            try:
                while True:
                    if max_images is not None and scanned_images >= max_images:
                        break

                    current_batch_size = batch_size

                    if max_images is not None:
                        remaining = max_images - scanned_images
                        current_batch_size = min(current_batch_size, remaining)

                    image_rows = self.cleanup_service.get_next_images(
                        session,
                        last_image_id=last_image_id,
                        batch_size=current_batch_size,
                    )

                    if not image_rows:
                        break

                    for image_row in image_rows:
                        image_id = self._get_image_id(image_row)

                        last_image_id = max(last_image_id, image_id)
                        scanned_images += 1

                        candidate = self.cleanup_service.analyze_image(
                            session,
                            image_row=image_row,
                            policy=policy,
                        )

                        if candidate is None:
                            continue

                        candidates.append(candidate)

                        if not dry_run:
                            deleted = self._delete_candidate_database_rows(
                                session,
                                candidate=candidate,
                            )

                            if deleted:
                                deleted_database_rows += 1
                                deleted_candidates.append(candidate)

                    info_id(
                        (
                            "Image artifact cleanup progress. "
                            f"scanned={scanned_images}, "
                            f"candidates={len(candidates)}, "
                            f"deleted_db_rows={deleted_database_rows}"
                        ),
                        request_id=request_id,
                    )

                if dry_run:
                    session.rollback()
                else:
                    session.commit()

            except Exception as exc:
                session.rollback()
                error_id(
                    (
                        "Image artifact cleanup failed and transaction was rolled back: "
                        f"{exc}"
                    ),
                    request_id=request_id,
                )
                raise

        file_action = "none"
        file_action_success_count = 0
        file_action_errors: List[Dict[str, Any]] = []

        if not dry_run:
            file_action_result = self._perform_file_actions(
                candidates=deleted_candidates,
                delete_files=delete_files,
                quarantine_files=quarantine_files,
                quarantine_dir=quarantine_dir,
                request_id=request_id,
            )

            file_action = file_action_result["file_action"]
            file_action_success_count = file_action_result["success_count"]
            file_action_errors = file_action_result["errors"]

        result = ImageArtifactCleanupResult(
            request_id=request_id,
            dry_run=dry_run,
            scanned_images=scanned_images,
            candidate_count=len(candidates),
            deleted_database_rows=deleted_database_rows,
            file_action=file_action,
            file_action_success_count=file_action_success_count,
            file_action_error_count=len(file_action_errors),
            report_path=resolved_report_path,
            candidates=[
                self.cleanup_service.candidate_to_dict(candidate)
                for candidate in candidates
            ],
            file_action_errors=file_action_errors,
        )

        self._write_report(
            result=result,
            report_path=resolved_report_path,
        )

        info_id(
            (
                "Image artifact cleanup complete. "
                f"dry_run={dry_run}, "
                f"scanned={scanned_images}, "
                f"candidates={len(candidates)}, "
                f"deleted_db_rows={deleted_database_rows}, "
                f"file_action={file_action}, "
                f"file_action_success={file_action_success_count}, "
                f"file_action_errors={len(file_action_errors)}, "
                f"report={resolved_report_path}"
            ),
            request_id=request_id,
        )

        return self._result_to_dict(result)

    # ------------------------------------------------------------
    # Validation helpers
    # ------------------------------------------------------------

    @staticmethod
    def _validate_run_options(
        *,
        batch_size: int,
        max_images: Optional[int],
        delete_files: bool,
        quarantine_files: bool,
    ) -> None:
        if batch_size <= 0:
            raise ValueError("batch_size must be greater than 0.")

        if max_images is not None and max_images <= 0:
            raise ValueError("max_images must be greater than 0 when provided.")

        if delete_files and quarantine_files:
            raise ValueError("Use either delete_files or quarantine_files, not both.")

    # ------------------------------------------------------------
    # Database delete helper
    # ------------------------------------------------------------

    def _delete_candidate_database_rows(
        self,
        session,
        *,
        candidate: ImageArtifactCandidate,
    ) -> bool:
        """
        Delete candidate DB rows through the service.

        Supports both method names:
          - delete_image_database_rows
          - delete_image_graph_for_cleanup

        This keeps the orchestrator compatible with either cleanup service version.
        """

        if hasattr(self.cleanup_service, "delete_image_database_rows"):
            return bool(
                self.cleanup_service.delete_image_database_rows(
                    session,
                    image_id=candidate.image_id,
                )
            )

        if hasattr(self.cleanup_service, "delete_image_graph_for_cleanup"):
            return bool(
                self.cleanup_service.delete_image_graph_for_cleanup(
                    session,
                    image_id=candidate.image_id,
                )
            )

        raise AttributeError(
            "ImageArtifactCleanupService must provide either "
            "delete_image_database_rows() or delete_image_graph_for_cleanup()."
        )

    # ------------------------------------------------------------
    # File handling after DB commit
    # ------------------------------------------------------------

    def _perform_file_actions(
        self,
        *,
        candidates: List[ImageArtifactCandidate],
        delete_files: bool,
        quarantine_files: bool,
        quarantine_dir: Optional[str],
        request_id: str,
    ) -> Dict[str, Any]:
        """
        File actions happen only after the DB transaction has committed.

        This prevents the worst failure mode:
          - file deleted/moved
          - DB rollback happens
          - DB still points to missing file

        Files are only acted on for candidates that were successfully deleted
        from the database.
        """

        if not delete_files and not quarantine_files:
            return {
                "file_action": "none",
                "success_count": 0,
                "errors": [],
            }

        action = "delete_files" if delete_files else "quarantine_files"

        errors: List[Dict[str, Any]] = []
        success_count = 0
        processed_paths: set[str] = set()

        quarantine_root: Optional[Path] = None

        if quarantine_files:
            quarantine_root = Path(
                quarantine_dir
                or Path("logs") / "image_artifact_quarantine"
            )
            quarantine_root.mkdir(parents=True, exist_ok=True)

        for candidate in candidates:
            path_text = self._get_candidate_file_path(candidate)

            if not path_text:
                continue

            source_path = Path(path_text)

            try:
                normalized_source = str(source_path.resolve())
            except Exception:
                normalized_source = str(source_path)

            if normalized_source in processed_paths:
                continue

            processed_paths.add(normalized_source)

            if not source_path.exists():
                continue

            try:
                if delete_files:
                    source_path.unlink()
                    success_count += 1
                    continue

                assert quarantine_root is not None

                destination = quarantine_root / (
                    f"image_{candidate.image_id}_{source_path.name}"
                )

                if destination.exists():
                    destination = quarantine_root / (
                        f"image_{candidate.image_id}_{uuid.uuid4()}_{source_path.name}"
                    )

                shutil.move(str(source_path), str(destination))
                success_count += 1

            except Exception as exc:
                error_payload = {
                    "image_id": candidate.image_id,
                    "file_path": str(source_path),
                    "error": str(exc),
                }

                errors.append(error_payload)

                warning_id(
                    (
                        "Image artifact file action failed. "
                        f"image_id={candidate.image_id}, "
                        f"path={source_path}, "
                        f"error={exc}"
                    ),
                    request_id=request_id,
                )

        return {
            "file_action": action,
            "success_count": success_count,
            "errors": errors,
        }

    @staticmethod
    def _get_candidate_file_path(
        candidate: ImageArtifactCandidate,
    ) -> Optional[str]:
        """
        Supports both candidate field names:
          - resolved_file_path
          - absolute_file_path
        """

        resolved_file_path = getattr(candidate, "resolved_file_path", None)

        if resolved_file_path:
            return str(resolved_file_path)

        absolute_file_path = getattr(candidate, "absolute_file_path", None)

        if absolute_file_path:
            return str(absolute_file_path)

        return None

    # ------------------------------------------------------------
    # Report helpers
    # ------------------------------------------------------------

    def _resolve_report_path(
        self,
        report_path: Optional[str],
    ) -> str:
        if report_path:
            resolved = Path(report_path)
        else:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            resolved = Path("logs") / f"image_artifact_cleanup_{timestamp}.json"

        resolved.parent.mkdir(parents=True, exist_ok=True)
        return str(resolved)

    def _write_report(
        self,
        *,
        result: ImageArtifactCleanupResult,
        report_path: str,
    ) -> None:
        path = Path(report_path)
        path.parent.mkdir(parents=True, exist_ok=True)

        with path.open("w", encoding="utf-8") as f:
            json.dump(
                self._result_to_dict(result),
                f,
                indent=2,
                ensure_ascii=False,
                default=str,
            )

    def _result_to_dict(
        self,
        result: ImageArtifactCleanupResult,
    ) -> Dict[str, Any]:
        return {
            "request_id": result.request_id,
            "dry_run": result.dry_run,
            "scanned_images": result.scanned_images,
            "candidate_count": result.candidate_count,
            "deleted_database_rows": result.deleted_database_rows,
            "file_action": result.file_action,
            "file_action_success_count": result.file_action_success_count,
            "file_action_error_count": result.file_action_error_count,
            "report_path": result.report_path,
            "candidates": result.candidates,
            "file_action_errors": result.file_action_errors,
        }

    # ------------------------------------------------------------
    # Small compatibility helpers
    # ------------------------------------------------------------

    @staticmethod
    def _get_image_id(
        image_row: Any,
    ) -> int:
        if hasattr(image_row, "id"):
            return int(image_row.id)

        if hasattr(image_row, "image_id"):
            return int(image_row.image_id)

        raise AttributeError("Image row must have either 'id' or 'image_id'.")