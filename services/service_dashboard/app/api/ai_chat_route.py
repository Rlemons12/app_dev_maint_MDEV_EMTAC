# services/service_dashboard/app/api/ai_chat_route.py

from __future__ import annotations

"""
Status A.I. Dashboard Proxy

Path:
    services/service_dashboard/app/api/ai_chat_route.py

Purpose:
    Keep the Service Dashboard as a UI and service monitor only.

    The dashboard frontend still calls:
        POST /api/ai/chat
        GET  /api/ai/health
        GET  /api/ai/tools

    But this file no longer owns:
        - AI Gateway calls
        - local model prompt/tool loop
        - tool definitions
        - service/database/Grafana tool execution

    Those responsibilities moved to:
        E:\\emtac\\services\\mcp_server\\ai_layer\\ai_rest_app.py

Flow:
    Dashboard UI
        -> service_dashboard /api/ai/*
        -> MCP AI Layer REST API :9200 /api/ai/*
        -> AI Gateway :9000
        -> GPU Service :5051
        -> MCP Coordinator :9100
        -> response back to dashboard UI

Register in main.py:
    from app.api.ai_chat_route import ai_bp, init_ai_blueprint
    init_ai_blueprint(service_manager)
    app.register_blueprint(ai_bp)
"""

import os
import time
from typing import Any

import requests as http_requests
from flask import Blueprint, jsonify, request

from app.services.service_dashboard_logger import (
    dash_debug,
    dash_error,
    dash_info,
    dash_warning,
)


ai_bp = Blueprint("ai", __name__, url_prefix="/api/ai")


# -------------------------------------------------------------------
# Injected at startup
# -------------------------------------------------------------------

_service_manager = None


def init_ai_blueprint(service_manager) -> None:
    """
    Preserve the existing dashboard startup contract.

    The old implementation needed ServiceManager for local tool execution.
    The new implementation does not execute tools here, but we keep the
    initializer so main.py does not need to change.
    """

    global _service_manager
    _service_manager = service_manager
    dash_info("ai_chat_route: ServiceManager injected into AI proxy blueprint")


# -------------------------------------------------------------------
# Config
# -------------------------------------------------------------------

MCP_AI_LAYER_BASE_URL = os.getenv(
    "SERVICE_MCP_AI_LAYER_BASE_URL",
    "http://127.0.0.1:9200",
).rstrip("/")

MCP_AI_LAYER_CHAT_URL = os.getenv(
    "SERVICE_MCP_AI_LAYER_CHAT_URL",
    f"{MCP_AI_LAYER_BASE_URL}/api/ai/chat",
)

MCP_AI_LAYER_HEALTH_URL = os.getenv(
    "SERVICE_MCP_AI_LAYER_HEALTH_URL",
    f"{MCP_AI_LAYER_BASE_URL}/api/ai/health",
)

MCP_AI_LAYER_TOOLS_URL = os.getenv(
    "SERVICE_MCP_AI_LAYER_TOOLS_URL",
    f"{MCP_AI_LAYER_BASE_URL}/api/ai/tools",
)

MCP_AI_LAYER_TIMEOUT_SECONDS = int(
    os.getenv("SERVICE_MCP_AI_LAYER_TIMEOUT_SECONDS", "180")
)


# -------------------------------------------------------------------
# Helpers
# -------------------------------------------------------------------

def _safe_json_response(response: http_requests.Response) -> dict[str, Any]:
    """
    Return a JSON body from an upstream response.

    If the upstream returns non-JSON text, wrap it in a structured payload so
    the dashboard frontend always receives JSON.
    """

    try:
        data = response.json()
        if isinstance(data, dict):
            return data

        return {
            "upstream_status_code": response.status_code,
            "upstream_body": data,
        }

    except Exception:
        return {
            "upstream_status_code": response.status_code,
            "upstream_text": response.text[:2000],
        }


def _proxy_get(url: str, *, timeout: int = 30) -> tuple[dict[str, Any], int]:
    started = time.time()

    dash_debug(f"ai_chat_route proxy GET -> {url}")

    try:
        response = http_requests.get(
            url,
            headers={"Accept": "application/json"},
            timeout=timeout,
        )

        elapsed_ms = int((time.time() - started) * 1000)
        payload = _safe_json_response(response)

        dash_info(
            "ai_chat_route proxy GET completed | "
            f"url={url} status={response.status_code} elapsed_ms={elapsed_ms}"
        )

        return payload, response.status_code

    except http_requests.exceptions.ConnectionError as exc:
        dash_error(
            "ai_chat_route proxy GET connection error | "
            f"url={url} error={exc}"
        )
        return {
            "status": "unavailable",
            "error": (
                f"MCP AI Layer is unreachable at {url}. "
                "Start the MCP AI Layer service."
            ),
            "detail": str(exc),
        }, 503

    except http_requests.exceptions.Timeout:
        dash_error(f"ai_chat_route proxy GET timeout | url={url}")
        return {
            "status": "timeout",
            "error": f"MCP AI Layer timed out calling {url}.",
        }, 504

    except Exception as exc:
        dash_error(
            "ai_chat_route proxy GET unexpected error | "
            f"url={url} error={exc}"
        )
        return {
            "status": "error",
            "error": str(exc),
        }, 500


def _proxy_post(
    url: str,
    *,
    payload: dict[str, Any],
    timeout: int,
) -> tuple[dict[str, Any], int]:
    started = time.time()

    dash_debug(
        "ai_chat_route proxy POST -> "
        f"{url} payload_keys={list(payload.keys())}"
    )

    try:
        response = http_requests.post(
            url,
            json=payload,
            headers={
                "Accept": "application/json",
                "Content-Type": "application/json",
            },
            timeout=timeout,
        )

        elapsed_ms = int((time.time() - started) * 1000)
        response_payload = _safe_json_response(response)

        dash_info(
            "ai_chat_route proxy POST completed | "
            f"url={url} status={response.status_code} elapsed_ms={elapsed_ms}"
        )

        return response_payload, response.status_code

    except http_requests.exceptions.ConnectionError as exc:
        dash_error(
            "ai_chat_route proxy POST connection error | "
            f"url={url} error={exc}"
        )
        return {
            "error": (
                f"MCP AI Layer is unreachable at {url}. "
                "Start the MCP AI Layer service."
            ),
            "detail": str(exc),
        }, 503

    except http_requests.exceptions.Timeout:
        dash_error(f"ai_chat_route proxy POST timeout | url={url}")
        return {
            "error": (
                f"MCP AI Layer timed out after "
                f"{MCP_AI_LAYER_TIMEOUT_SECONDS}s."
            )
        }, 504

    except Exception as exc:
        dash_error(
            "ai_chat_route proxy POST unexpected error | "
            f"url={url} error={exc}"
        )
        return {
            "error": str(exc),
        }, 500


def _attach_dashboard_context(payload: dict[str, Any]) -> dict[str, Any]:
    """
    Add small dashboard-side context before forwarding to mcp_server.

    This does not perform AI orchestration. It only helps the MCP AI layer know
    the request came from the service dashboard.
    """

    forwarded = dict(payload)

    forwarded.setdefault("source", "service_dashboard")
    forwarded.setdefault("dashboard_proxy", True)

    try:
        if _service_manager is not None:
            forwarded.setdefault(
                "dashboard_service_count",
                len(_service_manager.services),
            )
    except Exception as exc:
        dash_warning(
            "ai_chat_route: unable to attach dashboard service count | "
            f"error={exc}"
        )

    return forwarded


# -------------------------------------------------------------------
# POST /api/ai/chat
# -------------------------------------------------------------------

@ai_bp.route("/chat", methods=["POST"])
def ai_chat():
    """
    Thin proxy to MCP AI Layer.

    Frontend keeps calling:
        POST /api/ai/chat

    Dashboard forwards to:
        POST http://127.0.0.1:9200/api/ai/chat
    """

    body = request.get_json(force=True, silent=True) or {}

    if not isinstance(body, dict):
        return jsonify({"error": "Invalid JSON body."}), 400

    forwarded_body = _attach_dashboard_context(body)

    dash_info(
        "ai_chat_route: forwarding chat request to MCP AI Layer | "
        f"url={MCP_AI_LAYER_CHAT_URL} keys={list(forwarded_body.keys())}"
    )

    payload, status_code = _proxy_post(
        MCP_AI_LAYER_CHAT_URL,
        payload=forwarded_body,
        timeout=MCP_AI_LAYER_TIMEOUT_SECONDS,
    )

    return jsonify(payload), status_code


# -------------------------------------------------------------------
# GET /api/ai/health
# -------------------------------------------------------------------

@ai_bp.route("/health", methods=["GET"])
def ai_health():
    """
    Thin proxy to MCP AI Layer health.
    """

    payload, status_code = _proxy_get(
        MCP_AI_LAYER_HEALTH_URL,
        timeout=15,
    )

    if isinstance(payload, dict):
        payload.setdefault("dashboard_proxy", True)
        payload.setdefault("proxied_url", MCP_AI_LAYER_HEALTH_URL)

    return jsonify(payload), status_code


# -------------------------------------------------------------------
# GET /api/ai/tools
# -------------------------------------------------------------------

@ai_bp.route("/tools", methods=["GET"])
def ai_tools():
    """
    Thin proxy to MCP AI Layer tools.
    """

    payload, status_code = _proxy_get(
        MCP_AI_LAYER_TOOLS_URL,
        timeout=15,
    )

    if isinstance(payload, dict):
        payload.setdefault("dashboard_proxy", True)
        payload.setdefault("proxied_url", MCP_AI_LAYER_TOOLS_URL)

    return jsonify(payload), status_code