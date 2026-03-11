# routes/chatbot_bp.py

from flask import Blueprint, request, jsonify
from datetime import datetime

from modules.configuration.log_config import (
    with_request_id,
    error_id,
)

from modules.coordinators.chat_coordinator import ChatCoordinator
from modules.coordinators.feedback_coordinator import FeedbackCoordinator
from modules.coordinators.health_coordinator import HealthCoordinator
from modules.coordinators.analytics_coordinator import AnalyticsCoordinator
from modules.coordinators.dashboard_coordinator import DashboardCoordinator
from modules.observability.models import TraceSession, TraceSpan
from modules.decorators import trace_entrypoint, integration_trace

chatbot_bp = Blueprint("chatbot_bp", __name__)

chat_coordinator = ChatCoordinator()
feedback_coordinator = FeedbackCoordinator()
health_coordinator = HealthCoordinator()
analytics_coordinator = AnalyticsCoordinator()
dashboard_coordinator = DashboardCoordinator()


# ==========================================================
# ASK
# ==========================================================

@chatbot_bp.route("/ask", methods=["POST"])
@with_request_id

def ask(request_id=None):

    try:
        data = request.json or {}

        result = chat_coordinator.process_question(
            user_id=data.get("userId", "anonymous"),
            question=data.get("question", ""),
            client_type=data.get("clientType", "web"),
            request_id=request_id,
        )

        return jsonify(result)

    except Exception as e:
        error_id(f"/ask route failure: {e}", request_id, exc_info=True)
        return jsonify({
            "status": "error",
            "answer": "An unexpected error occurred.",
        }), 500


# ==========================================================
# UPDATE Q&A
# ==========================================================

@chatbot_bp.route("/update_qanda", methods=["POST"])
@with_request_id
def update_qanda(request_id=None):

    try:
        data = request.json or {}

        result = feedback_coordinator.process_feedback(
            user_id=data.get("user_id", "anonymous"),
            question=data.get("question", ""),
            answer=data.get("answer", ""),
            rating=data.get("rating"),
            comment=data.get("comment"),
            request_id=request_id,
        )

        if result["status"] == "success":
            return jsonify({"message": result["message"]})

        return jsonify({
            "status": "error",
            "message": result["message"]
        }), 400

    except Exception as e:
        error_id(f"/update_qanda failure: {e}", request_id, exc_info=True)
        return jsonify({
            "status": "error",
            "message": "An unexpected error occurred",
        }), 500


# ==========================================================
# HEALTH
# ==========================================================

@chatbot_bp.route("/health", methods=["GET"])
@with_request_id
def health_check(request_id=None):

    try:
        result = health_coordinator.check_health(request_id=request_id)
        return jsonify(result)

    except Exception as e:
        error_id(f"/health failure: {e}", request_id, exc_info=True)
        return jsonify({
            "status": "unhealthy",
            "error": str(e),
            "timestamp": datetime.utcnow().isoformat()
        }), 500


# ==========================================================
# METRICS
# ==========================================================

@chatbot_bp.route("/metrics", methods=["GET"])
@with_request_id
def metrics(request_id=None):

    try:
        result = analytics_coordinator.get_metrics(request_id=request_id)
        return jsonify(result)

    except Exception as e:
        error_id(f"/metrics failure: {e}", request_id, exc_info=True)
        return jsonify({
            "status": "error",
            "error": str(e),
            "timestamp": datetime.utcnow().isoformat()
        }), 500


# ==========================================================
# PERFORMANCE RECOMMENDATIONS
# ==========================================================

@chatbot_bp.route("/performance/recommendations", methods=["GET"])
@with_request_id
def performance_recommendations(request_id=None):

    try:
        hours = request.args.get("hours", 24, type=int)

        result = analytics_coordinator.get_recommendations(
            hours=hours,
            request_id=request_id,
        )

        return jsonify(result)

    except Exception as e:
        error_id(f"/performance/recommendations failure: {e}", request_id, exc_info=True)
        return jsonify({
            "status": "error",
            "error": str(e),
            "timestamp": datetime.utcnow().isoformat()
        }), 500


# ==========================================================
# PERFORMANCE DASHBOARD
# ==========================================================

@chatbot_bp.route("/performance/dashboard", methods=["GET"])
@with_request_id
def performance_dashboard(request_id=None):

    try:
        hours = request.args.get("hours", 24, type=int)

        result = dashboard_coordinator.get_dashboard(
            hours=hours,
            request_id=request_id,
        )

        return jsonify(result)

    except Exception as e:
        error_id(f"/performance/dashboard failure: {e}", request_id, exc_info=True)
        return jsonify({
            "status": "error",
            "error": str(e),
            "timestamp": datetime.utcnow().isoformat()
        }), 500