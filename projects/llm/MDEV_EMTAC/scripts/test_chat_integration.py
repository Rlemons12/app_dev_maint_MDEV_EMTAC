# scripts/test_chat_integration.py

import sys
import os
from datetime import datetime

# ------------------------------------------------------------
# Ensure project root is on path
# ------------------------------------------------------------
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PROJECT_ROOT)

# ------------------------------------------------------------
# Imports
# ------------------------------------------------------------
from modules.application.chat_coordinator import ChatCoordinator
from modules.configuration.config_env import DatabaseConfig
from modules.configuration.log_config import logger

from modules.emtacdb.emtacdb_fts import QandA
from sqlalchemy.orm import sessionmaker


def main():

    print("\n==============================")
    print("🚀 CHAT INTEGRATION TEST START")
    print("==============================\n")

    # --------------------------------------------------------
    # Initialize Coordinator
    # --------------------------------------------------------
    chat_coordinator = ChatCoordinator()

    # --------------------------------------------------------
    # Test Input
    # --------------------------------------------------------
    test_question = "How do I reset the SMART UPS?"

    print(f"📝 Test Question: {test_question}")

    # --------------------------------------------------------
    # Execute Chat Flow
    # --------------------------------------------------------
    result = chat_coordinator.process_question(
        user_id="integration_test_user",
        question=test_question,
        client_type="debug",
    )

    print("\n✅ Chat Response Received\n")

    # --------------------------------------------------------
    # Validate Response Structure
    # --------------------------------------------------------
    assert isinstance(result, dict), "Result is not a dictionary"

    required_fields = [
        "status",
        "answer",
        "method",
        "strategy",
        "blocks",
        "performance",
        "response_time",
        "request_id",
    ]

    for field in required_fields:
        assert field in result, f"Missing field: {field}"

    assert isinstance(result["blocks"], dict), "Blocks must be dictionary"

    block_keys = [
        "documents-container",
        "parts-container",
        "images-container",
        "drawings-container",
    ]

    for key in block_keys:
        assert key in result["blocks"], f"Missing block: {key}"

    print("✔ Response structure valid")

    # --------------------------------------------------------
    # Validate Model Name
    # --------------------------------------------------------
    model_name = result.get("model_name")
    print(f"🤖 Model Name: {model_name}")

    # --------------------------------------------------------
    # Validate DB Persistence
    # --------------------------------------------------------
    db_config = DatabaseConfig()
    SessionLocal = sessionmaker(bind=db_config.engine)
    session = SessionLocal()

    latest_entry = (
        session.query(QandA)
        .order_by(QandA.id.desc())
        .first()
    )

    assert latest_entry is not None, "No QandA entry created"
    assert latest_entry.question == test_question, "Persisted question mismatch"

    print("✔ QandA persisted successfully")
    print(f"📦 Persisted Model Name: {latest_entry.model_name}")

    # --------------------------------------------------------
    # Print Performance
    # --------------------------------------------------------
    perf = result.get("performance", {})
    print("\n⏱ Performance:")
    print(perf)

    print("\n==============================")
    print("🎉 CHAT INTEGRATION TEST PASSED")
    print("==============================\n")

    session.close()


if __name__ == "__main__":
    main()