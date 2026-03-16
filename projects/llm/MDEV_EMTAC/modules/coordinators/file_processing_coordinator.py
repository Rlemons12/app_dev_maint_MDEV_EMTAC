# modules/coordinators/file_processing_coordinator.py

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from modules.configuration.log_config import (
    with_request_id,
    get_request_id,
    info_id,
    warning_id,
    error_id,
)

from modules.orchestrators.complete_document_orchestrator import (
    CompleteDocumentOrchestrator,
)

from modules.decorators import trace_entrypoint


class FileProcessingCoordinator:
    """
    Transport-agnostic workflow router.

    Responsibilities:
      - Validate high-level inputs
      - Route to correct orchestrator method
      - Normalize response into HTTP contract
      - NEVER open DB sessions
      - NEVER persist directly
    """

    def __init__(self):
        self.complete_document_orchestrator = CompleteDocumentOrchestrator()

    # ------------------------------------------------------------
    # PUBLIC API
    # ------------------------------------------------------------
    @with_request_id
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
        Legacy-compatible wrapper used by /add_document route.

        Returns:
            (success, response_dict, http_status_code)
        """

        rid = request_id or get_request_id()

        # --------------------------------------------------------
        # 1. Coordinator-Level Validation
        # --------------------------------------------------------
        if not files:
            warning_id("No files provided to coordinator", rid)
            return False, {"error": "No files provided"}, 400

        valid_files = [
            f for f in files
            if getattr(f, "filename", "").strip() or isinstance(f, str)
        ]

        if not valid_files:
            warning_id("No valid files provided to coordinator", rid)
            return False, {"error": "No valid files provided"}, 400

        if metadata is None:
            metadata = {}

        info_id(
            f"Coordinator routing upload | files={len(valid_files)} "
            f"| concurrent={concurrent} | max_workers={max_workers}",
            rid,
        )

        try:
            # ----------------------------------------------------
            # 2. Route to appropriate orchestrator workflow
            # ----------------------------------------------------
            if concurrent and len(valid_files) > 1:
                result = self.complete_document_orchestrator.process_upload_concurrent(
                    files=valid_files,
                    metadata=metadata,
                    max_workers=max_workers,
                    request_id=rid,   # propagate for trace continuity
                )
            else:
                result = self.complete_document_orchestrator.process_upload(
                    files=valid_files,
                    metadata=metadata,
                    request_id=rid,   # propagate for trace continuity
                )

            # ----------------------------------------------------
            # 3. Defensive Normalization
            # ----------------------------------------------------
            if not isinstance(result, dict):
                error_id(
                    f"Orchestrator returned non-dict result: {type(result)}",
                    rid,
                )
                return False, {"error": "Invalid orchestrator response"}, 500

            status = result.get("status", "success")

            if status in ("success", "partial_success"):
                return True, result, 200

            if status == "validation_error":
                return False, result, 400

            if status == "processing_error":
                return False, result, 500

            # Defensive fallback
            warning_id(f"Unexpected orchestrator status: {status}", rid)
            return False, result, 500

        except ValueError as e:
            warning_id(f"Upload rejected: {e}", rid)
            return False, {"error": str(e)}, 400

        except Exception as e:
            error_id(f"Upload failed: {e}", rid, exc_info=True)
            return False, {
                "error": "Internal server error",
                "detail": str(e),
            }, 500