# scripts/test_chatbot_endpoint_integration.py

import pytest
import json

from modules.configuration.config_env import DatabaseConfig
from ai_emtac import create_app  # adjust if your app factory is elsewhere


# ---------------------------------------------------------
# FIXTURE: Flask Test App
# ---------------------------------------------------------

@pytest.fixture(scope="module")
def app():
    """
    Boots full Flask app in testing mode.
    """
    app = create_app()
    app.config["TESTING"] = True
    return app


@pytest.fixture(scope="module")
def client(app):
    """
    Flask test client.
    """
    return app.test_client()


@pytest.fixture(scope="module")
def db_config():
    """
    Ensure DB initializes cleanly before test.
    """
    return DatabaseConfig()


# ---------------------------------------------------------
# CHATBOT ENDPOINT TEST
# ---------------------------------------------------------

def test_chatbot_ask_endpoint(client, db_config):
    """
    Full end-to-end chatbot integration test.
    """

    payload = {
        "question": "What is MTBF?",
        "context": ""
    }

    response = client.post(
        "/chatbot/ask",
        data=json.dumps(payload),
        content_type="application/json",
    )

    assert response.status_code == 200, response.data

    data = response.get_json()

    # Basic structure validation
    assert isinstance(data, dict)

    assert "answer" in data
    assert data["answer"], "Chatbot returned empty answer"

    # Optional structure validation
    if "type" in data:
        assert data["type"] in ("text", "multimodal")

    print("\nChatbot response received successfully.\n")