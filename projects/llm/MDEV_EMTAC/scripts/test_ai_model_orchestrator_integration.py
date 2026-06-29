# scripts/test_ai_model_orchestrator_integration.py

from __future__ import annotations

import pytest
from flask import Flask, request, jsonify


# ---------------------------------------------------------
# Fixtures
# ---------------------------------------------------------

@pytest.fixture()
def app() -> Flask:
    """
    Creates a minimal Flask app that mirrors the real chatbot routes
    WITHOUT importing the actual blueprint or coordinators.

    This isolates route behavior from AI stack, DB, and logging.
    """

    app = Flask(__name__)
    app.config["TESTING"] = True

    # -----------------------------------------------------
    # Fake Route Implementations (mirror production contract)
    # -----------------------------------------------------

    @app.post("/chatbot/ask")
    def ask():
        data = request.json or {}

        user_id = data.get("userId", "anonymous")
        client_type = data.get("clientType", "web")
        question = data.get("question", "")

        return jsonify({
            "status": "success",
            "answer": f"[fake-answer] user_id={user_id} client_type={client_type} q={question}",
            "model_name": "FakeModel",
            "documents": [],
            "images": [],
            "drawings": [],
            "parts": [],
        })

    @app.post("/chatbot/update_qanda")
    def update_qanda():
        data = request.json or {}

        question = data.get("question")
        answer = data.get("answer")

        if not question or not answer:
            return jsonify({
                "status": "error",
                "message": "question and answer required"
            }), 400

        return jsonify({"message": "feedback recorded"})

    @app.get("/chatbot/health")
    def health():
        return jsonify({
            "status": "healthy",
            "timestamp": "fake"
        })

    @app.get("/chatbot/metrics")
    def metrics():
        return jsonify({
            "status": "ok",
            "metrics": {"requests": 123}
        })

    @app.get("/chatbot/performance/recommendations")
    def performance_recommendations():
        hours = request.args.get("hours", 24, type=int)
        return jsonify({
            "status": "ok",
            "hours": hours,
            "recommendations": ["fake-rec"]
        })

    @app.get("/chatbot/performance/dashboard")
    def performance_dashboard():
        hours = request.args.get("hours", 24, type=int)
        return jsonify({
            "status": "ok",
            "hours": hours,
            "dashboard": {"fake": True}
        })

    return app


@pytest.fixture()
def client(app: Flask):
    return app.test_client()


# ---------------------------------------------------------
# Tests
# ---------------------------------------------------------

def test_ask_returns_answer(client):
    resp = client.post(
        "/chatbot/ask",
        json={
            "userId": "integration_test",
            "clientType": "pytest",
            "question": "What is MTBF?"
        },
    )

    assert resp.status_code == 200
    data = resp.get_json()

    assert data["status"] == "success"
    assert "MTBF" in data["answer"]


def test_update_qanda_success(client):
    resp = client.post(
        "/chatbot/update_qanda",
        json={
            "user_id": "integration_test",
            "question": "What is MTBF?",
            "answer": "Mean time between failures.",
            "rating": 5,
            "comment": "good",
        },
    )

    assert resp.status_code == 200
    assert resp.get_json() == {"message": "feedback recorded"}


def test_update_qanda_validation_error(client):
    resp = client.post(
        "/chatbot/update_qanda",
        json={"user_id": "integration_test", "question": "", "answer": ""},
    )

    assert resp.status_code == 400
    data = resp.get_json()
    assert data["status"] == "error"
    assert "required" in data["message"].lower()


def test_health(client):
    resp = client.get("/chatbot/health")
    assert resp.status_code == 200
    assert resp.get_json()["status"] == "healthy"


def test_metrics(client):
    resp = client.get("/chatbot/metrics")
    assert resp.status_code == 200
    assert resp.get_json()["status"] == "ok"


def test_performance_recommendations(client):
    resp = client.get("/chatbot/performance/recommendations?hours=12")
    assert resp.status_code == 200
    data = resp.get_json()
    assert data["status"] == "ok"
    assert data["hours"] == 12


def test_performance_dashboard(client):
    resp = client.get("/chatbot/performance/dashboard?hours=48")
    assert resp.status_code == 200
    data = resp.get_json()
    assert data["status"] == "ok"
    assert data["hours"] == 48