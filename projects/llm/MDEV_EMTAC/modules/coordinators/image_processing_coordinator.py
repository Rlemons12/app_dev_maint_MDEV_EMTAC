from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from modules.configuration.log_config import (
    with_request_id,
    get_request_id,
    info_id,
    warning_id,
    error_id,
)

from modules.decorators import trace_entrypoint
from modules.orchestrators.image_orchestrator import ImageOrchestrator


class ImageProcessingCoordinator:
    """
    Transport-agnostic workflow router for image ingestion.

    Responsibilities:
      - Validate high-level image inputs
      - Normalize metadata
      - Route to ImageOrchestrator upload workflow
      - Normalize response into HTTP contract
      - NEVER open DB sessions
      - NEVER persist directly
    """

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

    def __init__(self) -> None:
        self.image_orchestrator = ImageOrchestrator()

    @with_request_id
    @trace_entrypoint(
        deep_profile=True,
        capture_args=True,
        capture_return=True,
    )
    def process_upload(
        self,
        *,
        files: List[Any],
        metadata: Dict[str, Any],
        concurrent: bool = False,
        max_workers: int = 4,
        request_id: Optional[str] = None,
    ) -> Tuple[bool, Dict[str, Any], int]:
        """
        Legacy-compatible wrapper for image ingestion.

        Returns:
            (success, response_dict, http_status_code)
        """
        rid = request_id or get_request_id()

        if not files:
            warning_id("No image files provided to coordinator", rid)
            return False, {"status": "validation_error", "error": "No image files provided"}, 400

        valid_files = [
            f for f in files
            if getattr(f, "filename", "").strip()
        ]

        if not valid_files:
            warning_id("No valid image files provided to coordinator", rid)
            return False, {"status": "validation_error", "error": "No valid image files provided"}, 400

        unsupported = []
        for file_obj in valid_files:
            filename = getattr(file_obj, "filename", "").lower().strip()
            if not any(filename.endswith(ext) for ext in self.IMAGE_EXTENSIONS):
                unsupported.append(filename)

        if unsupported:
            warning_id(f"Unsupported image file type(s): {unsupported}", rid)
            return False, {
                "status": "validation_error",
                "error": "Unsupported image file type(s)",
                "files": unsupported,
            }, 400

        normalized_metadata = self._normalize_metadata(metadata or {})

        info_id(
            f"Coordinator routing image upload | files={len(valid_files)} "
            f"| concurrent={concurrent} | max_workers={max_workers}",
            rid,
        )

        try:
            if concurrent and len(valid_files) > 1:
                result = self.image_orchestrator.process_upload_concurrent(
                    files=valid_files,
                    metadata=normalized_metadata,
                    max_workers=max_workers,
                    request_id=rid,
                )
            else:
                result = self.image_orchestrator.process_upload(
                    files=valid_files,
                    metadata=normalized_metadata,
                    request_id=rid,
                )

            if not isinstance(result, dict):
                error_id(
                    f"Image orchestrator returned non-dict result: {type(result)}",
                    rid,
                )
                return False, {"status": "processing_error", "error": "Invalid orchestrator response"}, 500

            status = result.get("status", "success")

            if status in ("success", "partial_success"):
                return True, result, 200

            if status == "validation_error":
                return False, result, 400

            if status == "processing_error":
                return False, result, 500

            warning_id(f"Unexpected image orchestrator status: {status}", rid)
            return False, result, 500

        except ValueError as e:
            warning_id(f"Image upload rejected: {e}", rid)
            return False, {"status": "validation_error", "error": str(e)}, 400

        except Exception as e:
            error_id(f"Image upload failed: {e}", rid, exc_info=True)
            return False, {
                "status": "processing_error",
                "error": "Internal server error",
                "detail": str(e),
            }, 500

    def _normalize_metadata(self, metadata: Dict[str, Any]) -> Dict[str, str]:
        def clean(key: str, default: str = "") -> str:
            value = metadata.get(key, default)
            if value is None:
                return default
            return str(value).strip()

        return {
            "title": clean("title"),
            "description": clean("description"),
            "area": clean("area"),
            "equipment_group": clean("equipment_group"),
            "model": clean("model"),
            "asset_number": clean("asset_number"),
            "location": clean("location"),
            "site_location": clean("site_location"),
            "room_number": clean("room_number", "Unknown"),
            "department": clean("department"),
            "tags": clean("tags"),
            "priority": clean("priority", "normal"),
        }