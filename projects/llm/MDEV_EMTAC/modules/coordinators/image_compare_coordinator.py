# modules/coordinators/image_compare_coordinator.py

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

from modules.configuration.log_config import (
    with_request_id,
    get_request_id,
    info_id,
    warning_id,
    error_id,
)

from modules.decorators import trace_entrypoint
from modules.orchestrators.image_orchestrator import ImageOrchestrator


class ImageCompareCoordinator:
    """
    Transport-facing coordinator for image similarity comparison.

    Responsibilities:
      - Validate the incoming request boundary
      - Extract the uploaded query image
      - Route to ImageOrchestrator.compare_uploaded_image()
      - Normalize the response into the existing frontend contract
      - NEVER open DB sessions
      - NEVER persist directly
      - NEVER generate embeddings directly
      - NEVER call Image.search_images directly
    """

    def __init__(self) -> None:
        self.image_orchestrator = ImageOrchestrator()

    @with_request_id
    @trace_entrypoint(
        deep_profile=True,
        capture_args=True,
        capture_return=True,
    )
    def compare_uploaded_image(
        self,
        *,
        files: Any,
        field_name: str = "query_image",
        similarity_threshold: float = 0.3,
        limit: int = 10,
        cleanup_query_file: bool = True,
        request_id: Optional[str] = None,
    ) -> Tuple[bool, Dict[str, Any], int]:
        """
        Compare an uploaded query image against existing stored image embeddings.

        Returns:
            (success, response_dict, http_status_code)

        Frontend success contract:
            {
                "image_similarity_search": [...]
            }

        Frontend error contract:
            {
                "error": "...",
                "status": "...",
                "image_similarity_search": []
            }
        """
        rid = request_id or get_request_id()

        try:
            file_obj = self._extract_file(
                files=files,
                field_name=field_name,
                request_id=rid,
            )

            if file_obj is None:
                warning_id(
                    f"[ImageCompareCoordinator] Missing upload field '{field_name}'",
                    rid,
                )
                return False, {
                    "status": "validation_error",
                    "error": "No file part in the request",
                    "image_similarity_search": [],
                }, 400

            filename = getattr(file_obj, "filename", "") or ""

            if not filename.strip():
                warning_id(
                    "[ImageCompareCoordinator] Uploaded compare file has no filename",
                    rid,
                )
                return False, {
                    "status": "validation_error",
                    "error": "No selected file",
                    "image_similarity_search": [],
                }, 400

            info_id(
                f"[ImageCompareCoordinator] Routing image compare request | "
                f"file='{filename}' | threshold={similarity_threshold} | limit={limit}",
                rid,
            )

            result = self.image_orchestrator.compare_uploaded_image(
                file_obj=file_obj,
                similarity_threshold=similarity_threshold,
                limit=limit,
                cleanup_query_file=cleanup_query_file,
                request_id=rid,
            )

            if not isinstance(result, dict):
                error_id(
                    f"[ImageCompareCoordinator] Orchestrator returned non-dict result: {type(result)}",
                    rid,
                )
                return False, {
                    "status": "processing_error",
                    "error": "Invalid image comparison response",
                    "image_similarity_search": [],
                }, 500

            http_status = int(result.get("http_status", 200))
            status = result.get("status", "success")

            if status == "success":
                return True, {
                    "image_similarity_search": result.get("image_similarity_search", []),
                }, http_status

            return False, {
                "status": status,
                "error": result.get("error", "Image comparison failed."),
                "detail": result.get("detail"),
                "allowed_file_types": result.get("allowed_file_types"),
                "image_similarity_search": result.get("image_similarity_search", []),
            }, http_status

        except ValueError as exc:
            warning_id(
                f"[ImageCompareCoordinator] Validation error: {exc}",
                rid,
            )
            return False, {
                "status": "validation_error",
                "error": str(exc),
                "image_similarity_search": [],
            }, 400

        except Exception as exc:
            error_id(
                f"[ImageCompareCoordinator] Image comparison failed: {exc}",
                rid,
                exc_info=True,
            )
            return False, {
                "status": "processing_error",
                "error": "Internal server error",
                "detail": str(exc),
                "image_similarity_search": [],
            }, 500

    @staticmethod
    def _extract_file(
        *,
        files: Any,
        field_name: str,
        request_id: Optional[str] = None,
    ) -> Optional[Any]:
        """
        Extract the uploaded FileStorage object from Flask's request.files-like object.

        Supports:
          - request.files MultiDict
          - plain dict
          - direct FileStorage-like object fallback
        """
        if files is None:
            return None

        getter = getattr(files, "get", None)

        if callable(getter):
            file_obj = getter(field_name)
            if file_obj is not None:
                return file_obj

        # Defensive fallback: if a FileStorage-like object was passed directly.
        if hasattr(files, "filename") and hasattr(files, "save"):
            return files

        return None