import logging
from typing import Any, Dict, Optional

from flask import jsonify, request

from modules.orchestrators.tablet_edge_user_session_report_orchestrator import (
    TabletEdgeUserSessionReportOrchestrator,
)
from modules.services.tablet_edge.tablet_user_session_report_dtos import (
    TabletUserSessionReportRequest,
)


logger = logging.getLogger(__name__)


class TabletEdgeUserSessionReportCoordinator:
    """
    Handles HTTP request/response details for tablet-reported user state.
    """

    def __init__(self, db_config):
        self.orchestrator = TabletEdgeUserSessionReportOrchestrator(
            db_config=db_config,
        )

    def handle_report(self):
        payload = self._get_payload()

        tablet_uid = self._first_value(
            payload,
            "tablet_uid",
            "tabletUid",
            "device_uid",
            "deviceUid",
        )

        tablet_name = self._first_value(
            payload,
            "tablet_name",
            "tabletName",
            "device_name",
            "deviceName",
        )

        username = self._first_value(
            payload,
            "username",
            "employee_id",
            "employeeId",
            "logged_in_user",
            "loggedInUser",
        )

        display_name = self._first_value(
            payload,
            "display_name",
            "displayName",
            "user_display_name",
            "userDisplayName",
        )

        event_type = self._first_value(
            payload,
            "event_type",
            "eventType",
            default="heartbeat",
        )

        current_page_url = self._first_value(
            payload,
            "current_page_url",
            "currentPageUrl",
            "page_url",
            "pageUrl",
        )

        app_version = self._first_value(
            payload,
            "app_version",
            "appVersion",
        )

        app_version_code = self._first_int(
            payload,
            "app_version_code",
            "appVersionCode",
        )

        user_id = self._first_int(
            payload,
            "user_id",
            "userId",
        )

        report = TabletUserSessionReportRequest(
            tablet_uid=tablet_uid or "",
            tablet_name=tablet_name,
            username=username,
            display_name=display_name,
            user_id=user_id,
            event_type=event_type or "heartbeat",
            current_page_url=current_page_url,
            app_version=app_version,
            app_version_code=app_version_code,
            ip_address=request.remote_addr,
            user_agent=request.user_agent.string if request.user_agent else None,
        )

        result = self.orchestrator.report_user_session(report=report)

        status_code = 200 if result.success else 400

        return jsonify(
            {
                "success": result.success,
                "message": result.message,
                "tablet_device_id": result.tablet_device_id,
                "active_session_id": result.active_session_id,
                "tablet_uid": result.tablet_uid,
                "tablet_name": result.tablet_name,
                "username": result.username,
                "display_name": result.display_name,
                "event_type": result.event_type,
                "is_active": result.is_active,
            }
        ), status_code

    def _get_payload(self) -> Dict[str, Any]:
        json_payload = request.get_json(silent=True)

        if isinstance(json_payload, dict):
            return json_payload

        if request.form:
            return dict(request.form)

        return {}

    @staticmethod
    def _first_value(
        payload: Dict[str, Any],
        *keys: str,
        default: Optional[str] = None,
    ) -> Optional[str]:
        for key in keys:
            value = payload.get(key)

            if value is not None and str(value).strip():
                return str(value).strip()

        return default

    @staticmethod
    def _first_int(
        payload: Dict[str, Any],
        *keys: str,
    ) -> Optional[int]:
        for key in keys:
            value = payload.get(key)

            if value is None or not str(value).strip():
                continue

            try:
                return int(value)
            except ValueError:
                return None

        return None