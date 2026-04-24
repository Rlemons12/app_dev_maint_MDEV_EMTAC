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
from modules.orchestrators.parts_import_orchestrator import PartsImportOrchestrator


class PartsImportCoordinator:
    """
    Transport-agnostic coordinator for bulk part import workflows.

    Responsibilities:
      - Validate high-level import inputs
      - Normalize import options
      - Route to PartsImportOrchestrator
      - Normalize response into a stable contract
      - NEVER open DB sessions
      - NEVER commit/rollback directly
    """

    def __init__(self) -> None:
        self.parts_import_orchestrator = PartsImportOrchestrator()

    @with_request_id
    @trace_entrypoint(
        deep_profile=True,
        capture_args=True,
        capture_return=True,
    )
    def import_parts_from_excel(
        self,
        *,
        file_path: Optional[str] = None,
        create_associations: bool = True,
        create_backup: bool = False,
        request_id: Optional[str] = None,
    ) -> Tuple[bool, Dict[str, Any], int]:
        """
        Coordinator wrapper for parts Excel import.

        Returns:
            (success, response_dict, http_status_code)
        """
        rid = request_id or get_request_id()

        normalized_file_path = self._normalize_file_path(file_path)
        normalized_create_associations = self._normalize_bool(create_associations, default=True)
        normalized_create_backup = self._normalize_bool(create_backup, default=False)

        info_id(
            "Coordinator routing parts import "
            f"| file_path={normalized_file_path or '<default>'} "
            f"| create_associations={normalized_create_associations} "
            f"| create_backup={normalized_create_backup}",
            rid,
        )

        try:
            result = self.parts_import_orchestrator.import_parts_from_excel(
                file_path=normalized_file_path,
                create_associations=normalized_create_associations,
                create_backup=normalized_create_backup,
            )

            if not isinstance(result, dict):
                error_id(
                    f"Parts import orchestrator returned non-dict result: {type(result)}",
                    rid,
                )
                return False, {
                    "success": False,
                    "status": "processing_error",
                    "message": "Invalid orchestrator response",
                }, 500

            status_code = int(result.get("status_code", 500))
            success = bool(result.get("success", False))

            status = self._map_status(success=success, status_code=status_code)
            normalized_response = self._normalize_response(result=result, status=status)

            if status_code < 400 and success:
                return True, normalized_response, status_code

            if status_code == 400:
                return False, normalized_response, 400

            if status_code == 404:
                return False, normalized_response, 404

            return False, normalized_response, 500

        except ValueError as exc:
            warning_id(f"Parts import rejected: {exc}", rid)
            return False, {
                "success": False,
                "status": "validation_error",
                "message": str(exc),
            }, 400

        except Exception as exc:
            error_id(f"Parts import failed: {exc}", rid, exc_info=True)
            return False, {
                "success": False,
                "status": "processing_error",
                "message": "Internal server error",
                "detail": str(exc),
            }, 500

    def _normalize_file_path(self, file_path: Optional[Any]) -> Optional[str]:
        """
        Normalize incoming file path.
        """
        if file_path is None:
            return None

        normalized = str(file_path).strip().strip('"').strip("'")
        return normalized or None

    def _normalize_bool(self, value: Any, *, default: bool) -> bool:
        """
        Normalize common bool-like inputs.
        """
        if value is None:
            return default

        if isinstance(value, bool):
            return value

        if isinstance(value, str):
            cleaned = value.strip().lower()
            if cleaned in {"true", "1", "yes", "y", "on"}:
                return True
            if cleaned in {"false", "0", "no", "n", "off"}:
                return False

        return bool(value)

    def _map_status(self, *, success: bool, status_code: int) -> str:
        """
        Map orchestrator result to a stable coordinator status string.
        """
        if success and status_code == 200:
            return "success"

        if success and status_code == 201:
            return "success"

        if status_code == 400:
            return "validation_error"

        if status_code == 404:
            return "not_found"

        return "processing_error"

    def _normalize_response(self, *, result: Dict[str, Any], status: str) -> Dict[str, Any]:
        """
        Normalize orchestrator response into a stable coordinator contract.
        """
        normalized: Dict[str, Any] = {
            "success": bool(result.get("success", False)),
            "status": status,
            "message": result.get("message", ""),
            "status_code": int(result.get("status_code", 500)),
        }

        if "data" in result:
            normalized["data"] = result["data"]

        extra_keys = {
            key: value
            for key, value in result.items()
            if key not in {"success", "message", "status_code", "data"}
        }

        normalized.update(extra_keys)
        return normalized