"""
Flask blueprint for the EMTAC Tablet Edge Agent.

File:
    blueprints/tablet_edge/tablet_edge_bp.py

URL prefix:
    /tablet-edge

Responsibilities:
    - Define HTTP routes.
    - Read request JSON/query args.
    - Delegate to TabletEdgeCoordinator.
    - Return JSON responses.
    - Keep route logic thin.

Important:
    This blueprint does NOT own database sessions.
    This blueprint does NOT contain business logic.
    The coordinator/orchestrator/service layers handle the workflow.

Notes:
    The coordinator is loaded lazily so the lightweight health route can respond
    without initializing database/session dependencies.

Wi-Fi/router tracking:
    These routes can now accept Wi-Fi/router/access-point fields in register,
    heartbeat, network-events, and health-samples payloads.

    Common supported fields:
        ssid
        bssid
        router_ip
        router_name
        ip_address
        gateway_address
        dhcp_server_address
        dns_servers
        wifi_rssi
        signal_level
        frequency_mhz
        wifi_band
        link_speed_mbps

Tablet app update routes:
    The app-update routes are separated into:

        blueprints/tablet_edge/tablet_app_update_bp.py

    and registered as a child blueprint here.

    Final mounted routes:
        GET  /tablet-edge/app-update/check
        GET  /tablet-edge/app-update/download/<release_id>
        POST /tablet-edge/app-update/report
        GET  /tablet-edge/app-update/routes
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from flask import Blueprint, jsonify, make_response, request, session

from blueprints.tablet_edge.tablet_app_update_bp import tablet_app_update_bp
from modules.configuration.log_config import logger
from modules.coordinators.tablet_edge_coordinator import TabletEdgeCoordinator
from modules.coordinators.tablet_edge_user_session_report_coordinator import (
    TabletEdgeUserSessionReportCoordinator,
)
from modules.configuration.config_env import get_db_config


tablet_edge_bp = Blueprint(
    "tablet_edge_bp",
    __name__,
    url_prefix="/tablet-edge",
)

# Child blueprint:
#   /tablet-edge/app-update/...
tablet_edge_bp.register_blueprint(tablet_app_update_bp)


_tablet_edge_coordinator: TabletEdgeCoordinator | None = None


WIFI_TRACKING_FIELDS: tuple[dict[str, str], ...] = (
    {
        "name": "ssid",
        "description": "Wi-Fi network name reported by the tablet.",
        "example": "EMTAC_WIFI",
    },
    {
        "name": "bssid",
        "description": "Access point/router radio MAC address. This is the best AP identity.",
        "example": "9C:53:22:AA:10:02",
    },
    {
        "name": "router_ip",
        "description": "Router/default gateway IP address when available.",
        "example": "192.168.1.1",
    },
    {
        "name": "router_name",
        "description": "Human-readable router/AP name when known by the tablet or backend.",
        "example": "Line 2 Packaging AP",
    },
    {
        "name": "ip_address",
        "description": "Tablet local IP address on the Wi-Fi network.",
        "example": "192.168.1.47",
    },
    {
        "name": "gateway_address",
        "description": "Default gateway address reported by the tablet.",
        "example": "192.168.1.1",
    },
    {
        "name": "dhcp_server_address",
        "description": "DHCP server address when available.",
        "example": "192.168.1.1",
    },
    {
        "name": "dns_servers",
        "description": "DNS server list. Can be a comma-separated string or list from the client.",
        "example": "192.168.1.1, 8.8.8.8",
    },
    {
        "name": "wifi_rssi",
        "description": "Wi-Fi signal strength in dBm.",
        "example": "-61",
    },
    {
        "name": "signal_level",
        "description": "Signal quality level, usually 0 through 4.",
        "example": "3",
    },
    {
        "name": "frequency_mhz",
        "description": "Wi-Fi frequency in MHz.",
        "example": "5180",
    },
    {
        "name": "wifi_band",
        "description": "Wi-Fi band. Can be inferred by backend when frequency_mhz is present.",
        "example": "5GHz",
    },
    {
        "name": "link_speed_mbps",
        "description": "Reported Wi-Fi link speed.",
        "example": "72",
    },
)


def _get_tablet_edge_coordinator() -> TabletEdgeCoordinator:
    """
    Lazily create and return the TabletEdgeCoordinator.

    This avoids initializing DatabaseConfig/orchestrator dependencies during
    blueprint import and keeps /tablet-edge/health lightweight.
    """
    global _tablet_edge_coordinator

    if _tablet_edge_coordinator is None:
        logger.info("[TABLET_EDGE_BP] Initializing TabletEdgeCoordinator.")
        _tablet_edge_coordinator = TabletEdgeCoordinator()

    return _tablet_edge_coordinator


def _utc_now_iso() -> str:
    """
    Return current UTC time as an ISO string.
    """
    return datetime.now(timezone.utc).isoformat()


def _get_json_payload() -> dict[str, Any]:
    """
    Safely read a JSON request body.

    Flask request.get_json(silent=True) returns None when:
        - the request body is empty
        - the request is not JSON
        - JSON parsing fails

    The coordinator will validate required fields.
    """
    payload = request.get_json(silent=True)

    if payload is None:
        return {}

    if not isinstance(payload, dict):
        return {
            "_invalid_payload": True,
            "_payload_error": "Request JSON body must be an object.",
        }

    return payload


def _json_response(
    response_body: dict[str, Any],
    status_code: int = 200,
):
    """
    Build a JSON response with a status code.
    """
    return jsonify(response_body), status_code


def _no_cache_json_response(
    response_body: dict[str, Any],
    status_code: int = 200,
):
    """
    Build a JSON response with no-cache headers.

    Used for health checks and development helpers so the tablet/browser always
    receives the latest response.
    """
    response = make_response(jsonify(response_body), status_code)
    response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
    response.headers["Pragma"] = "no-cache"
    response.headers["Expires"] = "0"
    return response


def _invalid_payload_response(payload: dict[str, Any]):
    """
    Return a consistent error response when the request JSON body is not an object.
    """
    if payload.get("_invalid_payload") is True:
        response_body = {
            "success": False,
            "error": payload.get("_payload_error", "Invalid request payload."),
            "error_type": "invalid_json_payload",
            "server_time_utc": _utc_now_iso(),
        }
        return _json_response(response_body, 400)

    return None


@tablet_edge_bp.route("/health", methods=["GET"])
def health():
    """
    Server heartbeat route.

    Route:
        GET /tablet-edge/health

    Rules:
        - No database call.
        - No AI call.
        - No file lookup.
        - No heavy service dependency.
        - Should be fast enough for frequent tablet heartbeat checks.
    """
    response_body = {
        "success": True,
        "status": "ok",
        "service": "tablet_edge",
        "server_time_utc": _utc_now_iso(),
    }

    return _no_cache_json_response(response_body, 200)


@tablet_edge_bp.route("/register", methods=["POST"])
def register_tablet():
    """
    Register or update a tablet.

    Route:
        POST /tablet-edge/register

    This route may include Wi-Fi/router fields. If present, the coordinator and
    orchestrator can use those fields to track the current access point/router.
    """
    logger.info("[TABLET_EDGE_BP] POST /tablet-edge/register")

    payload = _get_json_payload()
    invalid_response = _invalid_payload_response(payload)

    if invalid_response is not None:
        return invalid_response

    _success, response_body, status_code = _get_tablet_edge_coordinator().register_tablet(
        payload
    )

    return _json_response(response_body, status_code)


@tablet_edge_bp.route("/heartbeat", methods=["POST"])
def heartbeat():
    """
    Update tablet last_seen_at and return server time.

    Route:
        POST /tablet-edge/heartbeat

    This route may include Wi-Fi/router fields. If present, the backend can
    record a lightweight Wi-Fi observation.
    """
    logger.info("[TABLET_EDGE_BP] POST /tablet-edge/heartbeat")

    payload = _get_json_payload()
    invalid_response = _invalid_payload_response(payload)

    if invalid_response is not None:
        return invalid_response

    _success, response_body, status_code = _get_tablet_edge_coordinator().heartbeat(
        payload
    )

    return _json_response(response_body, status_code)

@tablet_edge_bp.route("/user-session/report", methods=["POST"])
def user_session_report():
    """
    Report the current logged-in user state from the Android Tablet Edge Agent.

    Route:
        POST /tablet-edge/user-session/report

    This makes the tablet agent heartbeat/report the source of truth for:
        - tablet_uid
        - tablet_name
        - currently logged-in user
        - current page URL
        - login/logout/heartbeat user-state events
    """
    logger.info("[TABLET_EDGE_BP] POST /tablet-edge/user-session/report")

    coordinator = TabletEdgeUserSessionReportCoordinator(
        db_config=get_db_config()
    )

    return coordinator.handle_report()

@tablet_edge_bp.route("/network-events", methods=["POST"])
def network_events():
    """
    Record important network quality events from a tablet.

    Route:
        POST /tablet-edge/network-events
    """
    logger.info("[TABLET_EDGE_BP] POST /tablet-edge/network-events")

    payload = _get_json_payload()
    invalid_response = _invalid_payload_response(payload)

    if invalid_response is not None:
        return invalid_response

    _success, response_body, status_code = (
        _get_tablet_edge_coordinator().record_network_events(payload)
    )

    return _json_response(response_body, status_code)


@tablet_edge_bp.route("/health-samples", methods=["POST"])
def health_samples():
    """
    Record periodic tablet health samples.

    Route:
        POST /tablet-edge/health-samples
    """
    logger.info("[TABLET_EDGE_BP] POST /tablet-edge/health-samples")

    payload = _get_json_payload()
    invalid_response = _invalid_payload_response(payload)

    if invalid_response is not None:
        return invalid_response

    _success, response_body, status_code = (
        _get_tablet_edge_coordinator().record_health_samples(payload)
    )

    return _json_response(response_body, status_code)


@tablet_edge_bp.route("/dropdown-cache/status", methods=["GET"])
def dropdown_cache_status():
    """
    Return dropdown cache status.

    Route:
        GET /tablet-edge/dropdown-cache/status

    Optional query parameter:
        tablet_uid
    """
    logger.info("[TABLET_EDGE_BP] GET /tablet-edge/dropdown-cache/status")

    tablet_uid = request.args.get("tablet_uid")

    _success, response_body, status_code = (
        _get_tablet_edge_coordinator().get_dropdown_cache_status(
            tablet_uid=tablet_uid
        )
    )

    return _json_response(response_body, status_code)


@tablet_edge_bp.route("/dropdown-cache/full", methods=["GET"])
def dropdown_cache_full():
    """
    Return full dropdown cache payload.

    Route:
        GET /tablet-edge/dropdown-cache/full

    Optional query parameter:
        tablet_uid

    If tablet_uid is provided, the backend can update that tablet's cache
    manifest after generating the payload.
    """
    logger.info("[TABLET_EDGE_BP] GET /tablet-edge/dropdown-cache/full")

    tablet_uid = request.args.get("tablet_uid")

    _success, response_body, status_code = (
        _get_tablet_edge_coordinator().get_full_dropdown_cache(
            tablet_uid=tablet_uid
        )
    )

    return _json_response(response_body, status_code)


@tablet_edge_bp.route("/dropdown-cache/delta", methods=["GET"])
def dropdown_cache_delta():
    """
    Return delta dropdown cache payload.

    Route:
        GET /tablet-edge/dropdown-cache/delta?since=<timestamp>

    Version 1 returns a full_refresh_required response.
    """
    logger.info("[TABLET_EDGE_BP] GET /tablet-edge/dropdown-cache/delta")

    since = request.args.get("since")

    _success, response_body, status_code = (
        _get_tablet_edge_coordinator().get_delta_dropdown_cache(
            since=since
        )
    )

    return _json_response(response_body, status_code)


@tablet_edge_bp.route("/offline-events/sync", methods=["POST"])
def sync_offline_events():
    """
    Sync offline events from the tablet.

    Route:
        POST /tablet-edge/offline-events/sync
    """
    logger.info("[TABLET_EDGE_BP] POST /tablet-edge/offline-events/sync")

    payload = _get_json_payload()
    invalid_response = _invalid_payload_response(payload)

    if invalid_response is not None:
        return invalid_response

    _success, response_body, status_code = (
        _get_tablet_edge_coordinator().sync_offline_events(payload)
    )

    return _json_response(response_body, status_code)


@tablet_edge_bp.route("/app-logs", methods=["POST"])
def app_logs():
    """
    Record app logs from the Android Tablet Edge Agent.

    Route:
        POST /tablet-edge/app-logs
    """
    logger.info("[TABLET_EDGE_BP] POST /tablet-edge/app-logs")

    payload = _get_json_payload()
    invalid_response = _invalid_payload_response(payload)

    if invalid_response is not None:
        return invalid_response

    _success, response_body, status_code = _get_tablet_edge_coordinator().record_app_logs(
        payload
    )

    return _json_response(response_body, status_code)


@tablet_edge_bp.route("/wifi-fields", methods=["GET"])
def wifi_fields():
    """
    Lightweight development helper.

    Route:
        GET /tablet-edge/wifi-fields
    """
    response_body = {
        "success": True,
        "service": "tablet_edge",
        "description": "Accepted Wi-Fi/router/access-point tracking fields.",
        "identity_rule": {
            "primary_access_point_identity": "bssid",
            "human_dashboard_label": "router_name or friendly_name",
            "network_routing_fields": [
                "router_ip",
                "gateway_address",
            ],
        },
        "fields": list(WIFI_TRACKING_FIELDS),
        "server_time_utc": _utc_now_iso(),
    }

    return _no_cache_json_response(response_body, 200)

@tablet_edge_bp.route("/user-session/current-web-user", methods=["GET"])
def current_web_user():
    """
    Return the currently authenticated Flask web user for this WebView session.

    Route:
        GET /tablet-edge/user-session/current-web-user

    This is used by the Android Tablet Edge Agent WebView to identify who
    is actually logged in, instead of guessing from the login form submit.
    """
    user_id = session.get("user_id")
    employee_id = session.get("employee_id")
    first_name = session.get("first_name")
    last_name = session.get("last_name")

    if not user_id or not employee_id:
        return _json_response(
            {
                "success": True,
                "logged_in": False,
                "username": None,
                "display_name": None,
                "user_id": None,
            },
            200,
        )

    display_name = " ".join(
        part
        for part in [
            first_name,
            last_name,
        ]
        if part
    ).strip()

    if not display_name:
        display_name = str(employee_id)

    return _json_response(
        {
            "success": True,
            "logged_in": True,
            "username": str(employee_id),
            "display_name": display_name,
            "user_id": user_id,
        },
        200,
    )


@tablet_edge_bp.route("/routes", methods=["GET"])
def routes():
    """
    Lightweight route discovery helper for development/testing.

    Route:
        GET /tablet-edge/routes
    """
    response_body = {
        "success": True,
        "service": "tablet_edge",
        "routes": [
            {
                "method": "GET",
                "path": "/tablet-edge/health",
                "description": "Fast server heartbeat route.",
                "database_required": False,
            },
            {
                "method": "POST",
                "path": "/tablet-edge/register",
                "description": "Register or update a tablet. May include Wi-Fi/router fields.",
                "database_required": True,
            },
            {
                "method": "POST",
                "path": "/tablet-edge/heartbeat",
                "description": "Update tablet last_seen_at. May include Wi-Fi/router fields.",
                "database_required": True,
            },
            {
                "method": "POST",
                "path": "/tablet-edge/user-session/report",
                "description": (
                    "Report current tablet user state from the Android Tablet Edge Agent. "
                    "This is the source of truth for which user the tablet reports as logged in."
                ),
                "database_required": True,
                "example_payload": {
                    "tablet_uid": "1a45d70b-2a90-49f3-8bc7-457b9b06ae35",
                    "tablet_name": "EMTAC-ANDROID-TABLET",
                    "username": "admin",
                    "display_name": "Admin User",
                    "event_type": "heartbeat",
                    "current_page_url": "http://172.19.194.129:5000/index",
                    "app_version": "1.0.8",
                    "app_version_code": 9,
                },
            },
            {
                "method": "POST",
                "path": "/tablet-edge/network-events",
                "description": "Submit tablet network events and optional Wi-Fi/router details.",
                "database_required": True,
            },
            {
                "method": "POST",
                "path": "/tablet-edge/health-samples",
                "description": "Submit tablet health samples and optional Wi-Fi/router details.",
                "database_required": True,
            },
            {
                "method": "GET",
                "path": "/tablet-edge/dropdown-cache/status",
                "description": "Get dropdown cache status.",
                "database_required": True,
            },
            {
                "method": "GET",
                "path": "/tablet-edge/dropdown-cache/full",
                "description": "Get full dropdown cache payload.",
                "database_required": True,
            },
            {
                "method": "GET",
                "path": "/tablet-edge/dropdown-cache/delta",
                "description": "Get delta dropdown cache placeholder.",
                "database_required": True,
            },
            {
                "method": "POST",
                "path": "/tablet-edge/offline-events/sync",
                "description": "Sync offline tablet events.",
                "database_required": True,
            },
            {
                "method": "POST",
                "path": "/tablet-edge/app-logs",
                "description": "Submit tablet app logs.",
                "database_required": True,
            },
            {
                "method": "GET",
                "path": "/tablet-edge/app-update/check",
                "description": "Check whether an APK update is available.",
                "database_required": True,
            },
            {
                "method": "GET",
                "path": "/tablet-edge/app-update/download/<release_id>",
                "description": "Download an approved EMTAC Tablet Edge Agent APK.",
                "database_required": True,
            },
            {
                "method": "POST",
                "path": "/tablet-edge/app-update/report",
                "description": "Report tablet app update lifecycle events.",
                "database_required": True,
            },
            {
                "method": "GET",
                "path": "/tablet-edge/app-update/routes",
                "description": "List Tablet Edge app-update routes.",
                "database_required": False,
            },
            {
                "method": "GET",
                "path": "/tablet-edge/wifi-fields",
                "description": "List accepted Wi-Fi/router tracking fields.",
                "database_required": False,
            },
            {
                "method": "GET",
                "path": "/tablet-edge/routes",
                "description": "List Tablet Edge development routes.",
                "database_required": False,
            },
        ],
        "wifi_tracking": {
            "enabled": True,
            "primary_access_point_identity": "bssid",
            "observation_table": "tablet_edge.tablet_wifi_observation",
            "access_point_table": "tablet_edge.tablet_wifi_access_point",
        },
        "user_session_reporting": {
            "enabled": True,
            "source_of_truth": "tablet_agent_heartbeat_report",
            "route": "/tablet-edge/user-session/report",
            "session_table": "tablet_edge.tablet_user_session",
            "device_table": "tablet_edge.tablet_device",
            "supported_event_types": [
                "login",
                "logout",
                "heartbeat",
                "page_view",
                "user_changed",
            ],
            "identity_rule": {
                "tablet_identity": "tablet_uid",
                "current_user_identity": "username",
                "note": (
                    "The user is not permanently assigned to a tablet. "
                    "The tablet reports who is currently logged in."
                ),
            },
        },
        "app_updates": {
            "enabled": True,
            "release_table": "tablet_edge.tablet_app_release",
            "status_view": "tablet_edge.v_latest_tablet_app_update_status",
            "routes_prefix": "/tablet-edge/app-update",
        },
        "server_time_utc": _utc_now_iso(),
    }

    return _no_cache_json_response(response_body, 200)


__all__ = [
    "tablet_edge_bp",
]