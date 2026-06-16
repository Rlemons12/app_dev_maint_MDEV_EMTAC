"""
modules/orchestrators/drawing_search_orchestrator.py

Orchestrator layer for drawing search.

Responsibilities:
    - open/close database session
    - call DrawingSearchService
    - own transaction/session boundary
    - return payload/status_code tuple to coordinator
"""

from __future__ import annotations

from typing import Any, Dict, Optional

from modules.configuration.config_env import DatabaseConfig
from modules.configuration.log_config import debug_id, error_id, info_id, log_timed_operation, with_request_id
from modules.services.drawing_search_service import DrawingSearchService


class DrawingSearchOrchestrator:
    """
    Orchestrates drawing search workflows.
    """

    def __init__(
        self,
        db_config: Optional[DatabaseConfig] = None,
        service: Optional[DrawingSearchService] = None,
    ) -> None:
        self.db_config = db_config or DatabaseConfig()
        self.service = service or DrawingSearchService()

    @with_request_id
    def search_drawings(
        self,
        *,
        search_params: Dict[str, Any],
        request_id: Optional[str] = None,
    ) -> tuple[Dict[str, Any], int]:
        """
        Run drawing search with a managed DB session.
        """
        session = None

        try:
            session = self.db_config.get_main_session()
            debug_id("[DrawingSearchOrchestrator] Created DB session for search_drawings", request_id)

            with log_timed_operation("DrawingSearchOrchestrator.search_drawings", request_id):
                payload = self.service.search_drawings(
                    session=session,
                    request_id=request_id,
                    **search_params,
                )

            return payload, 200

        except Exception as exc:
            error_id(
                f"[DrawingSearchOrchestrator] search_drawings failed: {exc}",
                request_id,
                exc_info=True,
            )
            return {
                "error": "Internal server error",
                "message": "An error occurred while processing your drawing search request",
            }, 500

        finally:
            if session is not None:
                session.close()
                debug_id("[DrawingSearchOrchestrator] Closed DB session for search_drawings", request_id)

    @with_request_id
    def get_drawing_types(
        self,
        *,
        request_id: Optional[str] = None,
    ) -> tuple[Dict[str, Any], int]:
        """
        Get available drawing types.
        """
        try:
            payload = self.service.get_drawing_types(
                request_id=request_id,
            )
            return payload, 200

        except Exception as exc:
            error_id(
                f"[DrawingSearchOrchestrator] get_drawing_types failed: {exc}",
                request_id,
                exc_info=True,
            )
            return {
                "error": "Internal server error",
                "message": "An error occurred while retrieving drawing types",
            }, 500

    @with_request_id
    def search_by_type(
        self,
        *,
        drawing_type: str,
        limit: int = 100,
        request_id: Optional[str] = None,
    ) -> tuple[Dict[str, Any], int]:
        """
        Search drawings by type with a managed DB session.
        """
        session = None

        try:
            session = self.db_config.get_main_session()
            debug_id("[DrawingSearchOrchestrator] Created DB session for search_by_type", request_id)

            with log_timed_operation("DrawingSearchOrchestrator.search_by_type", request_id):
                payload = self.service.search_by_type(
                    session=session,
                    drawing_type=drawing_type,
                    limit=limit,
                    request_id=request_id,
                )

            return payload, 200

        except Exception as exc:
            error_id(
                f"[DrawingSearchOrchestrator] search_by_type failed: {exc}",
                request_id,
                exc_info=True,
            )
            return {
                "error": "Internal server error",
                "message": "An error occurred while searching drawings by type",
            }, 500

        finally:
            if session is not None:
                session.close()
                debug_id("[DrawingSearchOrchestrator] Closed DB session for search_by_type", request_id)


    @with_request_id
    def get_drawing_file(
        self,
        *,
        drawing_id: int,
        request_id: Optional[str] = None,
    ) -> tuple[Dict[str, Any], int]:
        """
        Resolve a Drawing file for serving/viewing with a managed DB session.

        Called by:
            DrawingSearchCoordinator.get_drawing_file()

        Used by route:
            /drawings/view/<drawing_id>
        """
        session = None

        try:
            session = self.db_config.get_main_session()
            debug_id(
                "[DrawingSearchOrchestrator] Created DB session for get_drawing_file",
                request_id,
            )

            with log_timed_operation("DrawingSearchOrchestrator.get_drawing_file", request_id):
                payload, status_code = self.service.get_drawing_file_payload(
                    session=session,
                    drawing_id=drawing_id,
                    request_id=request_id,
                )

            return payload, status_code

        except Exception as exc:
            error_id(
                f"[DrawingSearchOrchestrator] get_drawing_file failed: {exc}",
                request_id,
                exc_info=True,
            )
            return {
                "error": "Internal server error",
                "message": "An error occurred while retrieving the drawing file",
            }, 500

        finally:
            if session is not None:
                session.close()
                debug_id(
                    "[DrawingSearchOrchestrator] Closed DB session for get_drawing_file",
                    request_id,
                )