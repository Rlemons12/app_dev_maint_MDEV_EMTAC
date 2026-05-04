# routes/chatbot_bp.py

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, Optional

from flask import Blueprint, request, jsonify

from modules.configuration.log_config import (
    with_request_id,
    error_id,
    get_request_id,
)

from modules.coordinators.chat_coordinator import ChatCoordinator
from modules.coordinators.chat_payload_coordinator import ChatPayloadCoordinator
from modules.coordinators.feedback_coordinator import FeedbackCoordinator
from modules.coordinators.health_coordinator import HealthCoordinator
from modules.coordinators.analytics_coordinator import AnalyticsCoordinator
from modules.coordinators.dashboard_coordinator import DashboardCoordinator


chatbot_bp = Blueprint("chatbot_bp", __name__)

chat_coordinator = ChatCoordinator()
chat_payload_coordinator = ChatPayloadCoordinator()
feedback_coordinator = FeedbackCoordinator()
health_coordinator = HealthCoordinator()
analytics_coordinator = AnalyticsCoordinator()
dashboard_coordinator = DashboardCoordinator()


# ==========================================================
# SHARED HELPERS
# ==========================================================

def _empty_blocks() -> Dict[str, list]:
    return {
        "documents-container": [],
        "parts-container": [],
        "images-container": [],
        "drawings-container": [],
    }


def _empty_payload_response(
    *,
    status: str,
    payload_status: str,
    message: str,
    request_id: Optional[str],
    payload_route_request_id: Optional[str] = None,
) -> Dict[str, Any]:
    return {
        "status": status,
        "payload_status": payload_status,
        "message": message,
        "request_id": request_id,
        "payload_route_request_id": payload_route_request_id,
        "blocks": _empty_blocks(),
        "documents": [],
        "parts": [],
        "images": [],
        "drawings": [],
        "relationship_summary": None,
    }


def _resolve_route_request_id(request_id: Optional[str]) -> str:
    """
    Resolve the active request ID.

    Important:
        @with_request_id may create/bind a request ID for logging context,
        but it does not necessarily pass that value into the Flask route
        function argument.

    Therefore:
        route argument request_id may be None
        get_request_id() should still return the active logging request ID
    """

    return request_id or get_request_id()


def _json_data() -> Dict[str, Any]:
    data = request.get_json(silent=True) or {}

    if not isinstance(data, dict):
        return {}

    return data


# ==========================================================
# ASK
# ==========================================================
# Answer-first route.
#
# This route returns the chatbot text answer quickly.
# It does NOT wait for documents/images/parts/drawings payload projection.
#
# Frontend flow:
#   1. POST /chatbot/ask
#   2. Render result["answer"]
#   3. If result["payload_status"] == "pending", call /chatbot/ask/payload
#      using result["request_id"]
# ==========================================================

@chatbot_bp.route("/ask", methods=["POST"])
@with_request_id
def ask(request_id=None):
    effective_request_id = _resolve_route_request_id(request_id)

    try:
        data = _json_data()

        result = chat_coordinator.process_question(
            user_id=data.get("userId", "anonymous"),
            question=data.get("question", ""),
            client_type=data.get("clientType", "web"),
            request_id=effective_request_id,
        )

        if not isinstance(result, dict):
            result = {
                "status": "error",
                "answer": "Invalid coordinator response.",
                "payload_status": "unavailable",
                "payload_endpoint": None,
                "blocks": _empty_blocks(),
                "documents": [],
                "parts": [],
                "images": [],
                "drawings": [],
            }

        # Critical:
        # The frontend needs this ID to call /chatbot/ask/payload.
        result["request_id"] = result.get("request_id") or effective_request_id

        # Keep payload endpoint consistent for the frontend.
        if result.get("payload_status") == "pending":
            result["payload_endpoint"] = (
                result.get("payload_endpoint")
                or "/ask/payload"
            )

        if "blocks" not in result or not isinstance(result["blocks"], dict):
            result["blocks"] = _empty_blocks()

        for key, value in _empty_blocks().items():
            result["blocks"].setdefault(key, list(value))

        result.setdefault("documents", [])
        result.setdefault("parts", [])
        result.setdefault("images", [])
        result.setdefault("drawings", [])

        return jsonify(result)

    except Exception as e:
        error_id(
            f"/ask route failure: {e}",
            effective_request_id,
            exc_info=True,
        )

        return jsonify({
            "status": "error",
            "answer": "An unexpected error occurred.",
            "payload_status": "unavailable",
            "payload_endpoint": None,
            "request_id": effective_request_id,
            "blocks": _empty_blocks(),
            "documents": [],
            "parts": [],
            "images": [],
            "drawings": [],
        }), 500


# ==========================================================
# ASK PAYLOAD
# ==========================================================
# Supporting payload route.
#
# This route loads the heavier UI payload:
#   - documents
#   - parts
#   - images
#   - drawings
#
# Frontend should call this AFTER /ask returns.
#
# Expected request body:
# {
#   "requestId": "<request_id returned from /ask>",
#   "clientType": "web"
# }
#
# Optional fallback:
# {
#   "requestId": "<request_id>",
#   "payload_seed": {...},
#   "clientType": "web"
# }
# ==========================================================

@chatbot_bp.route("/ask/payload", methods=["POST"])
@with_request_id
def ask_payload(request_id=None):
    """
    Supporting payload route.

    Important:
        payload_route_request_id belongs to THIS /ask/payload request.
        original_request_id belongs to the first /ask request and is used
        to find the saved QandA raw_response seed.
    """

    payload_route_request_id = _resolve_route_request_id(request_id)
    original_request_id = None

    try:
        data = _json_data()

        original_request_id = (
            data.get("requestId")
            or data.get("request_id")
            or data.get("originalRequestId")
            or data.get("original_request_id")
        )

        payload_seed = data.get("payload_seed")

        if not original_request_id and not payload_seed:
            return jsonify(
                _empty_payload_response(
                    status="invalid_input",
                    payload_status="unavailable",
                    message="Missing requestId or payload_seed.",
                    request_id=None,
                    payload_route_request_id=payload_route_request_id,
                )
            ), 400

        result = chat_payload_coordinator.load_payload(
            request_id=original_request_id,
            payload_seed=payload_seed,
            client_type=data.get("clientType", "web"),
        )

        if not isinstance(result, dict):
            result = _empty_payload_response(
                status="error",
                payload_status="error",
                message="Invalid payload coordinator response.",
                request_id=original_request_id,
                payload_route_request_id=payload_route_request_id,
            )

        # Keep both IDs available for debugging.
        result["request_id"] = original_request_id
        result["payload_route_request_id"] = payload_route_request_id

        if "blocks" not in result or not isinstance(result["blocks"], dict):
            result["blocks"] = _empty_blocks()

        for key, value in _empty_blocks().items():
            result["blocks"].setdefault(key, list(value))

        result.setdefault("documents", [])
        result.setdefault("parts", [])
        result.setdefault("images", [])
        result.setdefault("drawings", [])
        result.setdefault("relationship_summary", None)

        return jsonify(result)

    except Exception as e:
        error_id(
            f"/ask/payload route failure: {e}",
            payload_route_request_id,
            exc_info=True,
        )

        return jsonify(
            _empty_payload_response(
                status="error",
                payload_status="error",
                message="An unexpected error occurred while loading supporting payload.",
                request_id=original_request_id,
                payload_route_request_id=payload_route_request_id,
            )
        ), 500


# ==========================================================
# UPDATE Q&A
# ==========================================================

@chatbot_bp.route("/update_qanda", methods=["POST"])
@with_request_id
def update_qanda(request_id=None):
    effective_request_id = _resolve_route_request_id(request_id)

    try:
        data = _json_data()

        result = feedback_coordinator.process_feedback(
            user_id=data.get("user_id", "anonymous"),
            question=data.get("question", ""),
            answer=data.get("answer", ""),
            rating=data.get("rating"),
            comment=data.get("comment"),
            request_id=effective_request_id,
        )

        if result.get("status") == "success":
            return jsonify({
                "status": "success",
                "message": result.get("message", "Feedback saved."),
                "request_id": effective_request_id,
            })

        return jsonify({
            "status": "error",
            "message": result.get("message", "Unable to save feedback."),
            "request_id": effective_request_id,
        }), 400

    except Exception as e:
        error_id(
            f"/update_qanda failure: {e}",
            effective_request_id,
            exc_info=True,
        )

        return jsonify({
            "status": "error",
            "message": "An unexpected error occurred",
            "request_id": effective_request_id,
        }), 500


# ==========================================================
# HEALTH
# ==========================================================

@chatbot_bp.route("/health", methods=["GET"])
@with_request_id
def health_check(request_id=None):
    effective_request_id = _resolve_route_request_id(request_id)

    try:
        result = health_coordinator.check_health(
            request_id=effective_request_id,
        )

        if isinstance(result, dict):
            result.setdefault("request_id", effective_request_id)

        return jsonify(result)

    except Exception as e:
        error_id(
            f"/health failure: {e}",
            effective_request_id,
            exc_info=True,
        )

        return jsonify({
            "status": "unhealthy",
            "error": str(e),
            "request_id": effective_request_id,
            "timestamp": datetime.utcnow().isoformat(),
        }), 500


# ==========================================================
# METRICS
# ==========================================================

@chatbot_bp.route("/metrics", methods=["GET"])
@with_request_id
def metrics(request_id=None):
    effective_request_id = _resolve_route_request_id(request_id)

    try:
        result = analytics_coordinator.get_metrics(
            request_id=effective_request_id,
        )

        if isinstance(result, dict):
            result.setdefault("request_id", effective_request_id)

        return jsonify(result)

    except Exception as e:
        error_id(
            f"/metrics failure: {e}",
            effective_request_id,
            exc_info=True,
        )

        return jsonify({
            "status": "error",
            "error": str(e),
            "request_id": effective_request_id,
            "timestamp": datetime.utcnow().isoformat(),
        }), 500


# ==========================================================
# PERFORMANCE RECOMMENDATIONS
# ==========================================================

@chatbot_bp.route("/performance/recommendations", methods=["GET"])
@with_request_id
def performance_recommendations(request_id=None):
    effective_request_id = _resolve_route_request_id(request_id)

    try:
        hours = request.args.get("hours", 24, type=int)

        result = analytics_coordinator.get_recommendations(
            hours=hours,
            request_id=effective_request_id,
        )

        if isinstance(result, dict):
            result.setdefault("request_id", effective_request_id)

        return jsonify(result)

    except Exception as e:
        error_id(
            f"/performance/recommendations failure: {e}",
            effective_request_id,
            exc_info=True,
        )

        return jsonify({
            "status": "error",
            "error": str(e),
            "request_id": effective_request_id,
            "timestamp": datetime.utcnow().isoformat(),
        }), 500


# ==========================================================
# PERFORMANCE DASHBOARD
# ==========================================================

@chatbot_bp.route("/performance/dashboard", methods=["GET"])
@with_request_id
def performance_dashboard(request_id=None):
    effective_request_id = _resolve_route_request_id(request_id)

    try:
        hours = request.args.get("hours", 24, type=int)

        result = dashboard_coordinator.get_dashboard(
            hours=hours,
            request_id=effective_request_id,
        )

        if isinstance(result, dict):
            result.setdefault("request_id", effective_request_id)

        return jsonify(result)

    except Exception as e:
        error_id(
            f"/performance/dashboard failure: {e}",
            effective_request_id,
            exc_info=True,
        )

        return jsonify({
            "status": "error",
            "error": str(e),
            "request_id": effective_request_id,
            "timestamp": datetime.utcnow().isoformat(),
        }), 500