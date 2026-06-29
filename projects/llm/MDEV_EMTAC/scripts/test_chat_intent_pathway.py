from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import json
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional

from modules.configuration.config_env import DatabaseConfig
from modules.configuration.log_config import info_id, warning_id, error_id

from modules.coordinators.chat_intent_coordinator import ChatIntentCoordinator
from modules.emtacdb.emtacdb_fts import ChatSession


REQUEST_ID = "test-chat-intent"


def _utc_iso() -> str:
    return datetime.utcnow().isoformat()


def _print_block(title: str, data: Any) -> None:
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)
    if isinstance(data, (dict, list)):
        print(json.dumps(data, indent=2, default=str))
    else:
        print(data)


def _make_test_chat_session(db_session) -> ChatSession:
    now = _utc_iso()

    chat_session = ChatSession(
        user_id="intent-test-user",
        start_time=now,
        last_interaction=now,
        session_data=[
            {
                "role": "user",
                "content": "I am having RF arc outs on the Bag Fabricator. What should I check?",
                "request_id": "seed-1",
                "created_at": now,
                "metadata": {},
            },
            {
                "role": "assistant",
                "content": (
                    "Check the arc suppressor setting, generator cabinet connections, "
                    "transmission line area, and oscillator tube if plate/grid current peg."
                ),
                "request_id": "seed-1",
                "created_at": now,
                "metadata": {},
            },
        ],
        conversation_summary=[
            {
                "request_id": "seed-1",
                "qanda_id": str(uuid.uuid4()),
                "created_at": now,
                "question": "I am having RF arc outs on the Bag Fabricator. What should I check?",
                "answer_preview": (
                    "Check arc suppressor settings, loose generator cabinet connections, "
                    "metal objects near transmission lines, and the oscillator tube."
                ),
                "document_scope": None,
                "document_scope_enabled": False,
            }
        ],
    )

    db_session.add(chat_session)
    db_session.flush()
    return chat_session


def run_intent_tests() -> None:
    db = DatabaseConfig()
    coordinator = ChatIntentCoordinator()

    test_cases: List[Dict[str, Any]] = [
        {
            "name": "RAG / New standalone question",
            "question": "Where are the transfer grippers located on the auto filler?",
            "document_scope": None,
            "expected_path": "RAG_ONLY",
        },
        {
            "name": "Current conversation follow-up",
            "question": "What about on the rotary filler?",
            "document_scope": None,
            "expected_path": "CURRENT_SESSION_MEMORY",
        },
        {
            "name": "Prior conversation recall",
            "question": "What did we talk about earlier with RF arc outs?",
            "document_scope": None,
            "expected_path": "SEMANTIC_CHAT_RECALL",
        },
        {
            "name": "Recent Q/A clarification",
            "question": "Can you explain that in simpler steps?",
            "document_scope": None,
            "expected_path": "CURRENT_SESSION_MEMORY",
        },
        {
            "name": "Document scoped follow-up",
            "question": "What does this document say about the stacker?",
            "document_scope": {
                "enabled": True,
                "scope_type": "complete_document",
                "document_id": 237,
                "complete_document_id": 237,
                "document_name": "POPSFab4",
            },
            "expected_path": "DOCUMENT_SCOPE",
        },
    ]

    with db.main_session() as session:
        chat_session = _make_test_chat_session(session)
        session.commit()

        _print_block(
            "Created test ChatSession",
            {
                "session_id": str(chat_session.session_id),
                "summary_items": len(chat_session.conversation_summary or []),
                "message_items": len(chat_session.session_data or []),
            },
        )

        for case in test_cases:
            decision = coordinator.classify_question(
                question=case["question"],
                chat_session=chat_session,
                document_scope=case["document_scope"],
                request_id=REQUEST_ID,
            )

            actual_path = _resolve_path(decision.to_dict())

            result = {
                "test": case["name"],
                "question": case["question"],
                "expected_path": case["expected_path"],
                "actual_path": actual_path,
                "passed": actual_path == case["expected_path"],
                "decision": decision.to_dict(),
            }

            _print_block(case["name"], result)

    print("\nDone.")


def _resolve_path(decision: Dict[str, Any]) -> str:
    intent = decision.get("intent")

    if decision.get("needs_document_scope"):
        return "DOCUMENT_SCOPE"

    if decision.get("needs_semantic_chat_recall"):
        return "SEMANTIC_CHAT_RECALL"

    if decision.get("needs_current_session_memory"):
        return "CURRENT_SESSION_MEMORY"

    if intent == "NEW_TOPIC":
        return "RAG_ONLY"

    return "UNKNOWN"


if __name__ == "__main__":
    try:
        run_intent_tests()
    except Exception as exc:
        error_id(
            f"[test_chat_intent_pathway] Runner failed: {exc}",
            REQUEST_ID,
            exc_info=True,
        )
        raise