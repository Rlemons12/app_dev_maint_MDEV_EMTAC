"""
modules/coordinators/drawing_search_coordinator.py

Coordinator layer for drawing search.

Responsibilities:
    - parse request args
    - validate request values
    - convert strings to bool/int/list
    - call orchestrator

Route entrypoints supported:
    - /drawings/search
    - /drawings/types
    - /drawings/search/by-type/<drawing_type>
    - /drawings/view/<drawing_id>
"""

from __future__ import annotations

from typing import Any, Dict, Optional

from werkzeug.datastructures import MultiDict

from modules.configuration.log_config import debug_id, with_request_id
from modules.orchestrators.drawing_search_orchestrator import DrawingSearchOrchestrator


class DrawingSearchCoordinator:
    """
    Coordinates drawing search request input and response output.

    The coordinator does not open database sessions.
    The orchestrator owns the session/transaction boundary.
    """

    def __init__(
        self,
        orchestrator: Optional[DrawingSearchOrchestrator] = None,
    ) -> None:
        self.orchestrator = orchestrator or DrawingSearchOrchestrator()

    # ---------------------------------------------------------
    # PARSE HELPERS
    # ---------------------------------------------------------

    @staticmethod
    def _clean_text(value: Optional[str]) -> Optional[str]:
        """
        Normalize text values coming from request args.

        Returns:
            None when the value is missing or blank.
        """
        if value is None:
            return None

        text = str(value).strip()
        return text if text else None

    @staticmethod
    def _parse_bool(value: Optional[str], default: bool = False) -> bool:
        """
        Parse common boolean request values.

        Accepts:
            true, yes, 1, on
        """
        if value is None:
            return default

        return str(value).strip().lower() in {"true", "yes", "1", "on"}

    @staticmethod
    def _parse_int(
        value: Optional[str],
        *,
        field_name: str,
        default: Optional[int] = None,
        minimum: Optional[int] = None,
        maximum: Optional[int] = None,
    ) -> tuple[Optional[int], Optional[Dict[str, Any]]]:
        """
        Parse and validate an integer request argument.

        Returns:
            parsed_value, error_payload
        """
        if value is None or not str(value).strip():
            return default, None

        try:
            parsed = int(value)
        except (TypeError, ValueError):
            return None, {
                "error": f"Invalid {field_name} parameter",
                "message": f"{field_name} must be an integer",
            }

        if minimum is not None and parsed < minimum:
            return None, {
                "error": f"Invalid {field_name} parameter",
                "message": f"{field_name} must be at least {minimum}",
            }

        if maximum is not None and parsed > maximum:
            return None, {
                "error": f"Invalid {field_name} parameter",
                "message": f"{field_name} must be no more than {maximum}",
            }

        return parsed, None

    @staticmethod
    def _parse_fields(fields_param: Optional[str]) -> Optional[list[str]]:
        """
        Parse comma-separated search fields.

        Example:
            "drw_number,drw_name,file_path"

        Returns:
            ["drw_number", "drw_name", "file_path"]
        """
        if not fields_param:
            return None

        fields = [
            field.strip()
            for field in str(fields_param).split(",")
            if field.strip()
        ]

        return fields or None

    # ---------------------------------------------------------
    # SEARCH ARG PARSING
    # ---------------------------------------------------------

    def _parse_search_args(
        self,
        request_args: MultiDict,
    ) -> tuple[Optional[Dict[str, Any]], Optional[Dict[str, Any]], int]:
        """
        Parse /drawings/search query parameters.

        Returns:
            search_params, error_payload, status_code
        """
        drawing_id, error = self._parse_int(
            request_args.get("drawing_id"),
            field_name="drawing_id",
            default=None,
            minimum=1,
        )
        if error:
            return None, error, 400

        limit, error = self._parse_int(
            request_args.get("limit"),
            field_name="limit",
            default=100,
            minimum=1,
            maximum=1000,
        )
        if error:
            return None, error, 400

        search_params = {
            "search_text": self._clean_text(request_args.get("search_text")),
            "fields": self._parse_fields(request_args.get("fields")),
            "exact_match": self._parse_bool(
                request_args.get("exact_match"),
                default=False,
            ),
            "drawing_id": drawing_id,
            "drw_equipment_name": self._clean_text(
                request_args.get("drw_equipment_name")
            ),
            "drw_number": self._clean_text(request_args.get("drw_number")),
            "drw_name": self._clean_text(request_args.get("drw_name")),
            "drw_revision": self._clean_text(request_args.get("drw_revision")),
            "drw_spare_part_number": self._clean_text(
                request_args.get("drw_spare_part_number")
            ),
            "drw_type": self._clean_text(request_args.get("drw_type")),
            "file_path": self._clean_text(request_args.get("file_path")),
            "limit": limit,
            "include_part_images": self._parse_bool(
                request_args.get("include_part_images"),
                default=False,
            ),
        }

        return search_params, None, 200

    # ---------------------------------------------------------
    # ROUTE ENTRYPOINTS
    # ---------------------------------------------------------

    @with_request_id
    def search_from_request_args(
        self,
        *,
        request_args: MultiDict,
        request_id: Optional[str] = None,
    ) -> tuple[Dict[str, Any], int]:
        """
        Entry point for /drawings/search.
        """
        search_params, error_payload, status_code = self._parse_search_args(
            request_args
        )

        if error_payload:
            return error_payload, status_code

        debug_id(
            f"[DrawingSearchCoordinator] Parsed search params: {search_params}",
            request_id,
        )

        return self.orchestrator.search_drawings(
            search_params=search_params or {},
            request_id=request_id,
        )

    @with_request_id
    def get_drawing_types(
        self,
        *,
        request_id: Optional[str] = None,
    ) -> tuple[Dict[str, Any], int]:
        """
        Entry point for /drawings/types.
        """
        return self.orchestrator.get_drawing_types(
            request_id=request_id,
        )

    @with_request_id
    def search_by_type(
        self,
        *,
        drawing_type: str,
        request_args: MultiDict,
        request_id: Optional[str] = None,
    ) -> tuple[Dict[str, Any], int]:
        """
        Entry point for /drawings/search/by-type/<drawing_type>.
        """
        clean_type = self._clean_text(drawing_type)

        if not clean_type:
            return {
                "error": "Invalid drawing type",
                "message": "drawing_type is required",
            }, 400

        limit, error = self._parse_int(
            request_args.get("limit"),
            field_name="limit",
            default=100,
            minimum=1,
            maximum=1000,
        )

        if error:
            return error, 400

        return self.orchestrator.search_by_type(
            drawing_type=clean_type,
            limit=limit or 100,
            request_id=request_id,
        )

    @with_request_id
    def get_drawing_file(
        self,
        *,
        drawing_id: int,
        request_id: Optional[str] = None,
    ) -> tuple[Dict[str, Any], int]:
        """
        Entry point for /drawings/view/<drawing_id>.

        This validates the drawing ID, then asks the orchestrator to resolve
        the physical drawing file.

        The route layer is responsible for calling Flask send_file().
        """
        if drawing_id is None:
            return {
                "error": "Invalid drawing_id",
                "message": "drawing_id is required",
            }, 400

        try:
            parsed_id = int(drawing_id)
        except (TypeError, ValueError):
            return {
                "error": "Invalid drawing_id",
                "message": "drawing_id must be an integer",
            }, 400

        if parsed_id < 1:
            return {
                "error": "Invalid drawing_id",
                "message": "drawing_id must be at least 1",
            }, 400

        return self.orchestrator.get_drawing_file(
            drawing_id=parsed_id,
            request_id=request_id,
        )