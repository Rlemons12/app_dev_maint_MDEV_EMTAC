from __future__ import annotations

from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import mimetypes
import traceback

from werkzeug.datastructures import FileStorage

from modules.configuration.log_config import (
    with_request_id,
    get_request_id,
    info_id,
    warning_id,
    error_id,
)

from modules.coordinators.file_processing_coordinator import (
    FileProcessingCoordinator,
)
from modules.coordinators.image_processing_coordinator import (
    ImageProcessingCoordinator,
)

from modules.decorators import trace_entrypoint


@dataclass
class BatchFileResult:
    file_path: str
    file_name: str
    file_type: str
    status: str
    http_status: int
    message: str
    started_at: str
    completed_at: str
    duration_ms: int
    response: Optional[Dict[str, Any]] = None


class BatchProcessingCoordinator:
    """
    Folder-based batch processor for mixed files.

    Scope:
      - validate folder path
      - recursively collect supported files
      - classify each file
      - route documents to FileProcessingCoordinator
      - route images to ImageProcessingCoordinator
      - aggregate per-file results
      - no Flask route coupling
    """

    DOCUMENT_EXTENSIONS = {
        ".pdf",
        ".doc",
        ".docx",
        ".txt",
        ".rtf",
        ".csv",
        ".xls",
        ".xlsx",
        ".md",
        ".json",
        ".xml",
    }

    IMAGE_EXTENSIONS = {
        ".jpg",
        ".jpeg",
        ".png",
        ".gif",
        ".bmp",
        ".tif",
        ".tiff",
        ".webp",
    }

    SUPPORTED_EXTENSIONS = DOCUMENT_EXTENSIONS | IMAGE_EXTENSIONS

    def __init__(
        self,
        *,
        file_processing_coordinator: Optional[FileProcessingCoordinator] = None,
        image_processing_coordinator: Optional[ImageProcessingCoordinator] = None,
    ) -> None:
        self.file_processing_coordinator = (
            file_processing_coordinator or FileProcessingCoordinator()
        )
        self.image_processing_coordinator = (
            image_processing_coordinator or ImageProcessingCoordinator()
        )

    @with_request_id
    @trace_entrypoint(
        deep_profile=True,
        capture_args=True,
        capture_return=True,
    )
    def process_folder(
        self,
        *,
        folder_path: str,
        metadata: Dict[str, Any],
        include_subfolders: bool = True,
        concurrent: bool = False,
        max_workers: int = 4,
        request_id: Optional[str] = None,
    ) -> Tuple[bool, Dict[str, Any], int]:
        rid = request_id or get_request_id()

        normalized_folder_path = (folder_path or "").strip()
        normalized_metadata = self._normalize_metadata(metadata)

        if not normalized_folder_path:
            warning_id("No folder path provided to batch coordinator", rid)
            return False, {
                "status": "validation_error",
                "error": "No folder path provided",
                "folder_path": normalized_folder_path,
            }, 400

        path_obj = Path(normalized_folder_path).expanduser().resolve()

        if not path_obj.exists():
            warning_id(f"Folder path does not exist: {path_obj}", rid)
            return False, {
                "status": "validation_error",
                "error": "Folder path does not exist",
                "folder_path": str(path_obj),
            }, 404

        if not path_obj.is_dir():
            warning_id(f"Path is not a directory: {path_obj}", rid)
            return False, {
                "status": "validation_error",
                "error": "Path is not a directory",
                "folder_path": str(path_obj),
            }, 400

        files = self._collect_supported_files(
            folder_path=path_obj,
            include_subfolders=include_subfolders,
        )

        if not files:
            warning_id(f"No supported files found in {path_obj}", rid)
            return False, {
                "status": "validation_error",
                "error": "No supported files found",
                "folder_path": str(path_obj),
                "total_files_found": 0,
            }, 400

        info_id(
            f"Batch coordinator found {len(files)} supported file(s) "
            f"| folder={path_obj} | concurrent={concurrent}",
            rid,
        )

        started_at = datetime.now()
        results: List[BatchFileResult] = []

        for file_path in files:
            file_type = self._classify_file(file_path)

            if file_type == "document":
                result = self._process_document_file(
                    file_path=file_path,
                    metadata=normalized_metadata,
                    concurrent=concurrent,
                    max_workers=max_workers,
                    request_id=rid,
                )
            elif file_type == "image":
                result = self._process_image_file(
                    file_path=file_path,
                    metadata=normalized_metadata,
                    concurrent=concurrent,
                    max_workers=max_workers,
                    request_id=rid,
                )
            else:
                result = self._build_skipped_result(
                    file_path=file_path,
                    message="Unsupported file type",
                )

            results.append(result)

        completed_at = datetime.now()

        processed = sum(1 for r in results if r.status == "success")
        failed = sum(1 for r in results if r.status == "failed")
        skipped = sum(1 for r in results if r.status == "skipped")

        overall_success = failed == 0
        overall_status = "success" if overall_success else "partial_success"

        response = {
            "status": overall_status,
            "message": (
                "Batch processing completed successfully"
                if overall_success
                else "Batch processing completed with failures"
            ),
            "folder_path": str(path_obj),
            "started_at": started_at.isoformat(),
            "completed_at": completed_at.isoformat(),
            "duration_ms": int((completed_at - started_at).total_seconds() * 1000),
            "total_files_found": len(files),
            "processed": processed,
            "failed": failed,
            "skipped": skipped,
            "results": [asdict(r) for r in results],
        }

        return overall_success, response, 200 if overall_success else 207

    def _collect_supported_files(
        self,
        *,
        folder_path: Path,
        include_subfolders: bool,
    ) -> List[Path]:
        iterator = folder_path.rglob("*") if include_subfolders else folder_path.glob("*")

        files: List[Path] = []
        for path in iterator:
            if path.is_file() and path.suffix.lower() in self.SUPPORTED_EXTENSIONS:
                files.append(path)

        files.sort(key=lambda p: str(p).lower())
        return files

    def _classify_file(self, file_path: Path) -> str:
        suffix = file_path.suffix.lower()

        if suffix in self.DOCUMENT_EXTENSIONS:
            return "document"
        if suffix in self.IMAGE_EXTENSIONS:
            return "image"
        return "unknown"

    def _process_document_file(
            self,
            *,
            file_path: Path,
            metadata: Dict[str, Any],
            concurrent: bool,
            max_workers: int,
            request_id: Optional[str],
    ) -> BatchFileResult:
        rid = request_id or get_request_id()
        file_started_at = datetime.now()

        try:
            file_metadata = dict(metadata or {})

            incoming_title = file_metadata.get("title")
            if incoming_title is None or not str(incoming_title).strip():
                resolved_title = self._derive_title(file_path)
            else:
                resolved_title = str(incoming_title).strip()

            file_metadata["title"] = resolved_title

            content_type = self._guess_content_type(file_path)

            info_id(
                f"Batch routing document to FileProcessingCoordinator | "
                f"file={file_path} | content_type={content_type}",
                rid,
            )

            info_id(
                f"Batch document metadata resolved | "
                f"file={file_path.name} | title='{resolved_title}'",
                rid,
            )

            with open(file_path, "rb") as fh:
                file_obj = FileStorage(
                    stream=fh,
                    filename=file_path.name,
                    name="files",
                    content_type=content_type,
                )

                success, response, http_status = (
                    self.file_processing_coordinator.process_upload(
                        files=[file_obj],
                        metadata=file_metadata,
                        concurrent=concurrent,
                        max_workers=max_workers,
                        request_id=rid,
                    )
                )

            file_completed_at = datetime.now()
            duration_ms = int(
                (file_completed_at - file_started_at).total_seconds() * 1000
            )

            message = self._extract_message(response, success)

            return BatchFileResult(
                file_path=str(file_path),
                file_name=file_path.name,
                file_type="document",
                status="success" if success else "failed",
                http_status=http_status,
                message=message,
                started_at=file_started_at.isoformat(),
                completed_at=file_completed_at.isoformat(),
                duration_ms=duration_ms,
                response=(
                    response
                    if isinstance(response, dict)
                    else {"raw_response": str(response)}
                ),
            )

        except Exception as exc:
            return self._build_failed_result(
                file_path=file_path,
                file_type="document",
                exc=exc,
                started_at=file_started_at,
                request_id=rid,
            )

    def _process_image_file(
            self,
            *,
            file_path: Path,
            metadata: Dict[str, Any],
            concurrent: bool,
            max_workers: int,
            request_id: Optional[str],
    ) -> BatchFileResult:
        rid = request_id or get_request_id()
        file_started_at = datetime.now()

        try:
            file_metadata = dict(metadata or {})

            incoming_title = file_metadata.get("title")
            if incoming_title is None or not str(incoming_title).strip():
                resolved_title = self._derive_title(file_path)
            else:
                resolved_title = str(incoming_title).strip()

            incoming_description = file_metadata.get("description")
            if incoming_description is None or not str(incoming_description).strip():
                resolved_description = self._derive_title(file_path)
            else:
                resolved_description = str(incoming_description).strip()

            file_metadata["title"] = resolved_title
            file_metadata["description"] = resolved_description

            content_type = self._guess_content_type(file_path)

            info_id(
                f"Batch routing image to ImageProcessingCoordinator | "
                f"file={file_path} | content_type={content_type}",
                rid,
            )

            info_id(
                f"Batch image metadata resolved | "
                f"file={file_path.name} | title='{resolved_title}' | "
                f"description='{resolved_description}'",
                rid,
            )

            with open(file_path, "rb") as fh:
                file_obj = FileStorage(
                    stream=fh,
                    filename=file_path.name,
                    name="files",
                    content_type=content_type,
                )

                success, response, http_status = (
                    self.image_processing_coordinator.process_upload(
                        files=[file_obj],
                        metadata=file_metadata,
                        concurrent=concurrent,
                        max_workers=max_workers,
                        request_id=rid,
                    )
                )

            file_completed_at = datetime.now()
            duration_ms = int(
                (file_completed_at - file_started_at).total_seconds() * 1000
            )

            message = self._extract_message(response, success)

            return BatchFileResult(
                file_path=str(file_path),
                file_name=file_path.name,
                file_type="image",
                status="success" if success else "failed",
                http_status=http_status,
                message=message,
                started_at=file_started_at.isoformat(),
                completed_at=file_completed_at.isoformat(),
                duration_ms=duration_ms,
                response=(
                    response
                    if isinstance(response, dict)
                    else {"raw_response": str(response)}
                ),
            )

        except Exception as exc:
            return self._build_failed_result(
                file_path=file_path,
                file_type="image",
                exc=exc,
                started_at=file_started_at,
                request_id=rid,
            )

    def _build_failed_result(
        self,
        *,
        file_path: Path,
        file_type: str,
        exc: Exception,
        started_at: datetime,
        request_id: Optional[str],
    ) -> BatchFileResult:
        rid = request_id or get_request_id()
        completed_at = datetime.now()
        duration_ms = int((completed_at - started_at).total_seconds() * 1000)

        error_id(
            f"Batch coordinator failed for file {file_path}: {exc}",
            rid,
            exc_info=True,
        )

        return BatchFileResult(
            file_path=str(file_path),
            file_name=file_path.name,
            file_type=file_type,
            status="failed",
            http_status=500,
            message=str(exc),
            started_at=started_at.isoformat(),
            completed_at=completed_at.isoformat(),
            duration_ms=duration_ms,
            response={"traceback": traceback.format_exc()},
        )

    def _build_skipped_result(
        self,
        *,
        file_path: Path,
        message: str,
    ) -> BatchFileResult:
        now = datetime.now()
        return BatchFileResult(
            file_path=str(file_path),
            file_name=file_path.name,
            file_type="unknown",
            status="skipped",
            http_status=400,
            message=message,
            started_at=now.isoformat(),
            completed_at=now.isoformat(),
            duration_ms=0,
            response=None,
        )

    def _extract_message(self, response: Any, success: bool) -> str:
        if isinstance(response, dict):
            return response.get("message") or response.get("error") or ("Processed" if success else "Failed")
        return "Processed" if success else "Failed"

    def _normalize_metadata(self, metadata: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        metadata = metadata or {}

        def clean_str(key: str, default: str = "") -> str:
            value = metadata.get(key, default)
            if value is None:
                return default
            return str(value).strip()

        def clean_int(key: str) -> Optional[int]:
            value = metadata.get(key)
            if value in (None, "", "None"):
                return None
            try:
                return int(value)
            except (TypeError, ValueError):
                return None

        normalized = {
            "title": clean_str("title"),
            "description": clean_str("description"),
            "source": clean_str("source"),
            "document_type": clean_str("document_type"),
            "tags": clean_str("tags"),
            "priority": clean_str("priority", "normal"),
            "area": clean_str("area"),
            "equipment_group": clean_str("equipment_group"),
            "model": clean_str("model"),
            "asset_number": clean_str("asset_number"),
            "location": clean_str("location"),
            "site_location": clean_str("site_location"),
            "room_number": clean_str("room_number", "Unknown"),
            "department": clean_str("department"),

            # preserve FK ids for downstream position resolution
            "area_id": clean_int("area_id"),
            "equipment_group_id": clean_int("equipment_group_id"),
            "model_id": clean_int("model_id"),
            "asset_number_id": clean_int("asset_number_id"),
            "location_id": clean_int("location_id"),
            "site_location_id": clean_int("site_location_id"),
        }

        return normalized

    @staticmethod
    def _derive_title(file_path: Path) -> str:
        return file_path.stem.replace("_", " ").replace("-", " ").strip()

    @staticmethod
    def _guess_content_type(file_path: Path) -> str:
        guessed, _ = mimetypes.guess_type(str(file_path))
        return guessed or "application/octet-stream"