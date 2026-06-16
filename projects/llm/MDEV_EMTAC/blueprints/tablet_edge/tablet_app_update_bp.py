"""
EMTAC Tablet Edge Agent app update routes.

File:
    blueprints/tablet_edge/tablet_app_update_bp.py

Mounted under:
    /tablet-edge/app-update

Routes:
    GET  /tablet-edge/app-update/check
    GET  /tablet-edge/app-update/download/<release_id>
    POST /tablet-edge/app-update/report
    GET  /tablet-edge/app-update/routes

Purpose:
    - Let EMTAC tablets check for available APK updates.
    - Let EMTAC tablets download approved APK releases.
    - Let EMTAC tablets report update lifecycle events.

Notes:
    This blueprint should be registered as a child blueprint under tablet_edge_bp.

    Parent blueprint:
        blueprints/tablet_edge/tablet_edge_bp.py

    Parent registration should look like:
        tablet_edge_bp.register_blueprint(tablet_app_update_bp)

    Do not also register this blueprint directly in the central
    register_blueprints(app) function, or it may create duplicate/unwanted routes.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Optional

from flask import Blueprint, jsonify, make_response, request, send_file

from modules.coordinators.tablet_edge_coordinator import TabletEdgeCoordinator


logger = logging.getLogger(__name__)


tablet_app_update_bp = Blueprint(
    "tablet_app_update_bp",
    __name__,
    url_prefix="/app-update",
)


_tablet_edge_coordinator: Optional[TabletEdgeCoordinator] = None


def _get_tablet_edge_coordinator() -> TabletEdgeCoordinator:
    """
    Lazily create the TabletEdgeCoordinator.

    Keeping this lazy prevents lightweight routes from forcing database-backed
    dependencies to initialize until an app-update route is actually called.
    """
    global _tablet_edge_coordinator

    if _tablet_edge_coordinator is None:
        logger.info("[TABLET_APP_UPDATE_BP] Initializing TabletEdgeCoordinator.")
        _tablet_edge_coordinator = TabletEdgeCoordinator()

    return _tablet_edge_coordinator


def _json_response(response_body: dict[str, Any], status_code: int):
    """
    Return a normal JSON response.
    """
    return make_response(jsonify(response_body), status_code)


def _no_cache_json_response(response_body: dict[str, Any], status_code: int):
    """
    Return a JSON response with no-cache headers.

    App-update checks should not be cached because release availability may
    change while tablets are running.
    """
    response = make_response(jsonify(response_body), status_code)
    response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
    response.headers["Pragma"] = "no-cache"
    response.headers["Expires"] = "0"
    return response


def _get_json_payload() -> dict[str, Any] | None:
    """
    Safely read a JSON request body.
    """
    payload = request.get_json(silent=True)

    if payload is None:
        return None

    if not isinstance(payload, dict):
        return None

    return payload


def _invalid_payload_response(payload: dict[str, Any] | None):
    """
    Return a standard invalid-payload response when needed.
    """
    if payload is not None:
        return None

    response_body = {
        "success": False,
        "error": "Invalid or missing JSON payload.",
        "error_type": "invalid_json_payload",
    }

    return _json_response(response_body, 400)


def _safe_query_value(*names: str, default: str | None = None) -> str | None:
    """
    Read the first non-empty query parameter from a list of possible names.

    This supports aliases like:
        app_package or package
        release_channel or channel
    """
    for name in names:
        value = request.args.get(name)

        if value is not None and str(value).strip():
            return str(value).strip()

    return default


@tablet_app_update_bp.route("/check", methods=["GET"])
def app_update_check():
    """
    Check whether an EMTAC Tablet Edge Agent APK update is available.

    Route:
        GET /tablet-edge/app-update/check

    Query parameters:
        tablet_uid       optional but recommended
        version_code     optional; Android BuildConfig.VERSION_CODE
        version_name     optional; Android BuildConfig.VERSION_NAME
        app_package      optional; default com.example.emtactablet
        package          optional alias for app_package
        release_channel  optional; default stable
        channel          optional alias for release_channel
    """
    logger.info("[TABLET_APP_UPDATE_BP] GET /tablet-edge/app-update/check")

    tablet_uid = _safe_query_value("tablet_uid")
    version_code = _safe_query_value("version_code")
    version_name = _safe_query_value("version_name")

    app_package = _safe_query_value(
        "app_package",
        "package",
        default="com.example.emtactablet",
    )

    release_channel = _safe_query_value(
        "release_channel",
        "channel",
        default="stable",
    )

    _success, response_body, status_code = (
        _get_tablet_edge_coordinator().check_app_update(
            tablet_uid=tablet_uid,
            version_code=version_code,
            version_name=version_name,
            app_package=app_package,
            release_channel=release_channel,
        )
    )

    return _no_cache_json_response(response_body, status_code)


@tablet_app_update_bp.route("/download/<int:release_id>", methods=["GET"])
def app_update_download(release_id: int):
    """
    Download an approved EMTAC Tablet Edge Agent APK release.

    Route:
        GET /tablet-edge/app-update/download/<release_id>
    """
    logger.info(
        "[TABLET_APP_UPDATE_BP] GET /tablet-edge/app-update/download/%s",
        release_id,
    )

    _success, response_body, status_code = (
        _get_tablet_edge_coordinator().get_app_update_download_file(
            release_id=release_id,
        )
    )

    if not response_body.get("success"):
        return _json_response(response_body, status_code)

    apk_file_path = response_body.get("apk_file_path")

    if not apk_file_path:
        response_body = {
            "success": False,
            "error": "APK file path was not returned by the update service.",
            "error_type": "app_update_missing_apk_file_path",
            "release_id": release_id,
        }
        return _json_response(response_body, 500)

    apk_path = Path(str(apk_file_path)).expanduser()

    if not apk_path.exists() or not apk_path.is_file():
        response_body = {
            "success": False,
            "error": f"APK file does not exist on the server: {apk_path}",
            "error_type": "app_update_apk_file_not_found",
            "release_id": release_id,
        }
        return _json_response(response_body, 404)

    apk_filename = response_body.get("apk_filename") or apk_path.name
    apk_sha256 = response_body.get("apk_sha256")
    version_name = response_body.get("version_name")
    version_code = response_body.get("version_code")

    response = send_file(
        str(apk_path),
        mimetype=response_body.get(
            "mime_type",
            "application/vnd.android.package-archive",
        ),
        as_attachment=True,
        download_name=str(apk_filename),
        max_age=0,
        conditional=False,
    )

    response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
    response.headers["Pragma"] = "no-cache"
    response.headers["Expires"] = "0"
    response.headers["X-Content-Type-Options"] = "nosniff"

    if apk_sha256:
        response.headers["X-EMTAC-APK-SHA256"] = str(apk_sha256)

    if version_name:
        response.headers["X-EMTAC-Version-Name"] = str(version_name)

    if version_code is not None:
        response.headers["X-EMTAC-Version-Code"] = str(version_code)

    return response


@tablet_app_update_bp.route("/report", methods=["POST"])
def app_update_report():
    """
    Record an app update event from the Android Tablet Edge Agent.

    Route:
        POST /tablet-edge/app-update/report

    Example event_type values:
        update_available_seen
        download_started
        download_completed
        download_failed
        install_launched
        install_completed
        install_failed
    """
    logger.info("[TABLET_APP_UPDATE_BP] POST /tablet-edge/app-update/report")

    payload = _get_json_payload()
    invalid_response = _invalid_payload_response(payload)

    if invalid_response is not None:
        return invalid_response

    _success, response_body, status_code = (
        _get_tablet_edge_coordinator().report_app_update(payload)
    )

    return _json_response(response_body, status_code)


@tablet_app_update_bp.route("/routes", methods=["GET"])
def app_update_routes():
    """
    List app-update routes.

    Route:
        GET /tablet-edge/app-update/routes
    """
    response_body = {
        "success": True,
        "service": "tablet_edge_app_update",
        "routes": [
            {
                "method": "GET",
                "path": "/tablet-edge/app-update/check",
                "description": "Check whether an APK update is available.",
                "example": (
                    "/tablet-edge/app-update/check"
                    "?version_code=1&version_name=1.0.0"
                ),
            },
            {
                "method": "GET",
                "path": "/tablet-edge/app-update/download/<release_id>",
                "description": "Download an approved EMTAC Tablet Edge Agent APK.",
                "example": "/tablet-edge/app-update/download/1",
            },
            {
                "method": "POST",
                "path": "/tablet-edge/app-update/report",
                "description": "Report tablet app update lifecycle events.",
            },
            {
                "method": "GET",
                "path": "/tablet-edge/app-update/routes",
                "description": "List tablet app update routes.",
            },
        ],
    }

    return _json_response(response_body, 200)