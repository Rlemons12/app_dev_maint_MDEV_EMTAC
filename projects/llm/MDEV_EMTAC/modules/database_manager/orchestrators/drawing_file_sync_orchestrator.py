"""
modules/database_manager/orchestrators/drawing_file_sync_orchestrator.py

Orchestrator for folder-first Drawing.file_path synchronization.

This sits above:

    modules/database_manager/services/drawing_file_sync_service.py

Main responsibility:
    - validate run options
    - construct the DrawingFileSyncService
    - run dry-run or apply mode
    - optionally write a JSON report
    - return a clean dictionary payload for routes, scripts, coordinators, or tests

Folder-first means:

    physical drawing file -> match database Drawing row -> update Drawing.file_path
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, Sequence

from modules.configuration.config_env import DatabaseConfig
from modules.configuration.log_config import error_id, info_id, with_request_id
from modules.database_manager.services.drawing_file_sync_service import (
    DrawingFileSyncService,
    DrawingFileSyncSummary,
    write_json_report,
)


class DrawingFileSyncOrchestrator:
    """
    Orchestrates folder-first Drawing.file_path synchronization.

    Service layer does the actual folder scanning, database matching, and update work.
    Orchestrator layer validates workflow options and controls the run mode.

    Typical use:

        orchestrator = DrawingFileSyncOrchestrator()

        result = orchestrator.run_dry_run(
            include_results=True
        )

        result = orchestrator.apply_updates(
            report_json="E:\\emtac\\logs\\drawing_file_sync_report.json"
        )
    """

    DEFAULT_COMMIT_EVERY = 100

    def __init__(
        self,
        db_config: Optional[DatabaseConfig] = None,
        service: Optional[DrawingFileSyncService] = None,
    ) -> None:
        """
        Args:
            db_config:
                Optional database config dependency.

            service:
                Optional pre-built DrawingFileSyncService.
                Usually leave this as None so the orchestrator can create a service
                using the requested drawing_root/extensions/recursive options.
        """
        self.db_config = db_config or DatabaseConfig()
        self.service = service

    # ------------------------------------------------------------------
    # VALIDATION
    # ------------------------------------------------------------------

    @staticmethod
    def _validate_limit(limit: Optional[int]) -> None:
        if limit is not None and limit < 1:
            raise ValueError("limit must be at least 1 when provided.")

    @staticmethod
    def _validate_commit_every(commit_every: int) -> None:
        if commit_every < 1:
            raise ValueError("commit_every must be at least 1.")

    @staticmethod
    def _validate_report_json(report_json: Optional[str]) -> None:
        if report_json is None:
            return

        if not str(report_json).strip():
            raise ValueError("report_json cannot be blank when provided.")

    @staticmethod
    def _validate_extensions(extensions: Optional[Sequence[str]]) -> None:
        if extensions is None:
            return

        if not isinstance(extensions, (list, tuple, set)):
            raise ValueError("allowed_extensions must be a list, tuple, or set when provided.")

        for ext in extensions:
            if ext is None or not str(ext).strip():
                raise ValueError("allowed_extensions cannot contain blank values.")

    @staticmethod
    def _validate_drawing_root(drawing_root: Optional[str]) -> None:
        if drawing_root is None:
            return

        if not str(drawing_root).strip():
            raise ValueError("drawing_root cannot be blank when provided.")

    @staticmethod
    def _validate_default_drw_type(default_drw_type: str) -> None:
        if default_drw_type is None or not str(default_drw_type).strip():
            raise ValueError("default_drw_type cannot be blank.")

    def _validate_run_options(
        self,
        drawing_root: Optional[str],
        limit: Optional[int],
        commit_every: int,
        allowed_extensions: Optional[Sequence[str]],
        report_json: Optional[str],
        default_drw_type: str,
    ) -> None:
        self._validate_drawing_root(drawing_root)
        self._validate_limit(limit)
        self._validate_commit_every(commit_every)
        self._validate_extensions(allowed_extensions)
        self._validate_report_json(report_json)
        self._validate_default_drw_type(default_drw_type)

    # ------------------------------------------------------------------
    # SERVICE FACTORY
    # ------------------------------------------------------------------

    def _build_service(
        self,
        drawing_root: Optional[str],
        allowed_extensions: Optional[Sequence[str]],
        recursive: bool,
        request_id: Optional[str],
    ) -> DrawingFileSyncService:
        """
        Build the sync service.

        If a service was injected into the orchestrator, use it.
        Otherwise create a new one with the requested options.
        """
        if self.service is not None:
            return self.service

        return DrawingFileSyncService(
            db_config=self.db_config,
            drawing_root=drawing_root,
            allowed_extensions=allowed_extensions,
            recursive=recursive,
            request_id=request_id,
        )

    # ------------------------------------------------------------------
    # PAYLOAD FORMATTING
    # ------------------------------------------------------------------

    @staticmethod
    def _summary_to_payload(
        summary: DrawingFileSyncSummary,
        include_results: bool = True,
    ) -> Dict[str, Any]:
        """
        Convert the service summary dataclass into a clean dictionary payload.
        """
        return summary.to_dict(include_results=include_results)

    @staticmethod
    def _write_report_if_requested(
        summary: DrawingFileSyncSummary,
        report_json: Optional[str],
        include_results: bool,
    ) -> Optional[str]:
        """
        Write a JSON report if report_json was provided.
        """
        if not report_json:
            return None

        output_path = Path(report_json).expanduser().resolve()

        write_json_report(
            summary=summary,
            report_path=str(output_path),
            include_results=include_results,
        )

        return str(output_path)

    # ------------------------------------------------------------------
    # MAIN RUN METHOD
    # ------------------------------------------------------------------

    @with_request_id
    def run(
        self,
        dry_run: bool = True,
        drawing_root: Optional[str] = None,
        recursive: bool = True,
        limit: Optional[int] = None,
        commit_every: int = DEFAULT_COMMIT_EVERY,
        allowed_extensions: Optional[Sequence[str]] = None,
        use_compact_match: bool = True,
        create_missing: bool = False,
        prefer_first_on_ambiguous: bool = False,
        default_drw_type: str = "Other",
        report_json: Optional[str] = None,
        include_results: bool = True,
        request_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Run folder-first drawing file synchronization.

        Args:
            dry_run:
                True = scan and report only.
                False = update Drawing.file_path values.

            drawing_root:
                Optional override for the drawing folder.
                If None, service uses DATABASE_DRAWING from config/.env.

            recursive:
                True = walk all subfolders.
                False = only scan the top-level drawing folder.

            limit:
                Optional limit on how many physical drawing files to process.

            commit_every:
                Commit interval when dry_run=False.

            allowed_extensions:
                Optional allowed file extensions.
                Example:
                    [".dwg", ".dxf", ".slddrw", ".pdf"]

            use_compact_match:
                Enables loose matching by ignoring spaces, dashes, underscores,
                and punctuation.

            create_missing:
                If True, unmatched physical files can create new Drawing rows.
                Default False is safer.

            prefer_first_on_ambiguous:
                If True, uses the first DB match when multiple rows match a file.
                Default False is safer because it reports ambiguous records.

            default_drw_type:
                Drawing type used when create_missing=True.

            report_json:
                Optional path to write a JSON report.

            include_results:
                Include per-file results in the returned payload/report.

        Returns:
            Dictionary payload with summary counts and optional detailed results.
        """
        self._validate_run_options(
            drawing_root=drawing_root,
            limit=limit,
            commit_every=commit_every,
            allowed_extensions=allowed_extensions,
            report_json=report_json,
            default_drw_type=default_drw_type,
        )

        info_id(
            (
                "Starting DrawingFileSyncOrchestrator.run "
                f"dry_run={dry_run}, "
                f"drawing_root={drawing_root}, "
                f"recursive={recursive}, "
                f"limit={limit}, "
                f"create_missing={create_missing}"
            ),
            request_id,
        )

        service = self._build_service(
            drawing_root=drawing_root,
            allowed_extensions=allowed_extensions,
            recursive=recursive,
            request_id=request_id,
        )

        try:
            summary = service.sync_folder_to_database(
                dry_run=dry_run,
                limit=limit,
                commit_every=commit_every,
                use_compact_match=use_compact_match,
                create_missing=create_missing,
                prefer_first_on_ambiguous=prefer_first_on_ambiguous,
                default_drw_type=default_drw_type,
                request_id=request_id,
            )

            report_path = self._write_report_if_requested(
                summary=summary,
                report_json=report_json,
                include_results=include_results,
            )

            payload = self._summary_to_payload(
                summary=summary,
                include_results=include_results,
            )

            payload["mode"] = "dry_run" if dry_run else "apply"
            payload["report_json"] = report_path
            payload["success"] = summary.errors == 0
            payload["has_warnings"] = bool(summary.ambiguous or summary.no_db_match)

            info_id(
                (
                    "DrawingFileSyncOrchestrator.run complete. "
                    f"mode={payload['mode']}, "
                    f"success={payload['success']}, "
                    f"updated={summary.updated}, "
                    f"would_update={summary.would_update}, "
                    f"created={summary.created}, "
                    f"would_create={summary.would_create}, "
                    f"no_db_match={summary.no_db_match}, "
                    f"ambiguous={summary.ambiguous}, "
                    f"errors={summary.errors}"
                ),
                request_id,
            )

            return payload

        except Exception as exc:
            error_id(
                f"DrawingFileSyncOrchestrator.run failed: {exc}",
                request_id,
            )
            raise

    # ------------------------------------------------------------------
    # CONVENIENCE METHODS
    # ------------------------------------------------------------------

    def run_dry_run(
        self,
        drawing_root: Optional[str] = None,
        recursive: bool = True,
        limit: Optional[int] = None,
        allowed_extensions: Optional[Sequence[str]] = None,
        use_compact_match: bool = True,
        create_missing: bool = False,
        prefer_first_on_ambiguous: bool = False,
        default_drw_type: str = "Other",
        report_json: Optional[str] = None,
        include_results: bool = True,
        request_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Scan physical drawing files and database Drawing rows without updating the database.
        """
        return self.run(
            dry_run=True,
            drawing_root=drawing_root,
            recursive=recursive,
            limit=limit,
            commit_every=self.DEFAULT_COMMIT_EVERY,
            allowed_extensions=allowed_extensions,
            use_compact_match=use_compact_match,
            create_missing=create_missing,
            prefer_first_on_ambiguous=prefer_first_on_ambiguous,
            default_drw_type=default_drw_type,
            report_json=report_json,
            include_results=include_results,
            request_id=request_id,
        )

    def apply_updates(
        self,
        drawing_root: Optional[str] = None,
        recursive: bool = True,
        limit: Optional[int] = None,
        commit_every: int = DEFAULT_COMMIT_EVERY,
        allowed_extensions: Optional[Sequence[str]] = None,
        use_compact_match: bool = True,
        create_missing: bool = False,
        prefer_first_on_ambiguous: bool = False,
        default_drw_type: str = "Other",
        report_json: Optional[str] = None,
        include_results: bool = True,
        request_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Scan physical drawing files and update Drawing.file_path values when a safe match is found.
        """
        return self.run(
            dry_run=False,
            drawing_root=drawing_root,
            recursive=recursive,
            limit=limit,
            commit_every=commit_every,
            allowed_extensions=allowed_extensions,
            use_compact_match=use_compact_match,
            create_missing=create_missing,
            prefer_first_on_ambiguous=prefer_first_on_ambiguous,
            default_drw_type=default_drw_type,
            report_json=report_json,
            include_results=include_results,
            request_id=request_id,
        )

    def scan_only(
        self,
        drawing_root: Optional[str] = None,
        recursive: bool = True,
        limit: Optional[int] = None,
        allowed_extensions: Optional[Sequence[str]] = None,
        request_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Scan the drawing folder only and return file count/path info.

        This does not query or update Drawing records.
        """
        self._validate_drawing_root(drawing_root)
        self._validate_limit(limit)
        self._validate_extensions(allowed_extensions)

        service = self._build_service(
            drawing_root=drawing_root,
            allowed_extensions=allowed_extensions,
            recursive=recursive,
            request_id=request_id,
        )

        files = service.scan_drawing_folder(limit=limit)

        return {
            "success": True,
            "mode": "scan_only",
            "drawing_root": str(service.drawing_root),
            "recursive": recursive,
            "total_files_scanned": len(files),
            "files": [str(path) for path in files],
        }