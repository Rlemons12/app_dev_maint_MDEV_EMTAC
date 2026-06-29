# tests/test_chat_intent_pathway_comprehensive.py

from __future__ import annotations

import json
import sys
import uuid
import time
from pathlib import Path
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from modules.configuration.config_env import DatabaseConfig
from modules.configuration.log_config import info_id, warning_id, error_id

from modules.coordinators.chat_intent_coordinator import ChatIntentCoordinator
from modules.emtacdb.emtacdb_fts import ChatSession


REQUEST_ID = "test-chat-intent-pathway-comprehensive"
PASS_TARGET_PERCENT = 95.0


def _utc_iso() -> str:
    return datetime.utcnow().isoformat()


def _print_block(title: str, data: Any) -> None:
    print("\n" + "=" * 100)
    print(title)
    print("=" * 100)

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
            {
                "role": "user",
                "content": "Where is the arc suppressor located?",
                "request_id": "seed-2",
                "created_at": now,
                "metadata": {},
            },
            {
                "role": "assistant",
                "content": (
                    "The arc suppressor is part of the RF sealing/generator system. "
                    "Check the cabinet area, suppressor adjustment, and wiring condition."
                ),
                "request_id": "seed-2",
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
            },
            {
                "request_id": "seed-2",
                "qanda_id": str(uuid.uuid4()),
                "created_at": now,
                "question": "Where is the arc suppressor located?",
                "answer_preview": (
                    "The arc suppressor is associated with the RF sealing/generator system "
                    "and should be checked in the RF generator/cabinet area."
                ),
                "document_scope": None,
                "document_scope_enabled": False,
            },
        ],
    )

    db_session.add(chat_session)
    db_session.flush()

    return chat_session


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


def _decision_to_dict(decision: Any) -> Dict[str, Any]:
    if hasattr(decision, "to_dict"):
        return decision.to_dict()

    return {
        "intent": getattr(getattr(decision, "intent", None), "value", None),
        "confidence": getattr(decision, "confidence", None),
        "needs_current_session_memory": getattr(
            decision,
            "needs_current_session_memory",
            None,
        ),
        "needs_semantic_chat_recall": getattr(
            decision,
            "needs_semantic_chat_recall",
            None,
        ),
        "needs_document_scope": getattr(
            decision,
            "needs_document_scope",
            None,
        ),
        "rewritten_question": getattr(decision, "rewritten_question", None),
        "reason": getattr(decision, "reason", None),
    }


def _is_valid_decision(decision: Dict[str, Any]) -> Tuple[bool, List[str]]:
    errors: List[str] = []

    allowed_intents = {
        "NEW_TOPIC",
        "FOLLOW_UP_CURRENT_SESSION",
        "RECALL_PRIOR_CONVERSATION",
        "DOCUMENT_SCOPED_FOLLOW_UP",
        "CLARIFICATION",
    }

    intent = decision.get("intent")

    if intent not in allowed_intents:
        errors.append(f"Invalid intent: {intent!r}")

    confidence = decision.get("confidence")

    if not isinstance(confidence, (int, float)):
        errors.append(f"Confidence is not numeric: {confidence!r}")
    elif confidence < 0 or confidence > 1:
        errors.append(f"Confidence out of range: {confidence!r}")

    for key in [
        "needs_current_session_memory",
        "needs_semantic_chat_recall",
        "needs_document_scope",
    ]:
        if not isinstance(decision.get(key), bool):
            errors.append(f"{key} is not bool: {decision.get(key)!r}")

    if decision.get("rewritten_question") is None:
        errors.append("rewritten_question is None")

    if decision.get("reason") is None:
        errors.append("reason is None")

    return len(errors) == 0, errors


def _case(
    *,
    name: str,
    question: str,
    expected_path: str,
    document_scope: Optional[Dict[str, Any]] = None,
    min_confidence: float = 0.0,
) -> Dict[str, Any]:
    return {
        "name": name,
        "question": question,
        "document_scope": document_scope,
        "expected_path": expected_path,
        "min_confidence": min_confidence,
    }


def _build_test_cases() -> List[Dict[str, Any]]:
    active_doc_scope = {
        "enabled": True,
        "scope_type": "complete_document",
        "document_id": 237,
        "complete_document_id": 237,
        "document_name": "POPSFab4",
    }

    return [
        _case(
            name="01 standalone transfer gripper question",
            question="Where are the transfer grippers located on the auto filler?",
            expected_path="RAG_ONLY",
        ),
        _case(
            name="02 standalone sensor question",
            question="What does a photo eye do on an infeed conveyor?",
            expected_path="RAG_ONLY",
        ),
        _case(
            name="03 standalone maintenance question",
            question="How do I check a pneumatic cylinder for air leaks?",
            expected_path="RAG_ONLY",
        ),
        _case(
            name="04 current-session what about follow-up",
            question="What about on the rotary filler?",
            expected_path="CURRENT_SESSION_MEMORY",
        ),
        _case(
            name="05 current-session pronoun follow-up",
            question="Where is that located?",
            expected_path="CURRENT_SESSION_MEMORY",
        ),
        _case(
            name="06 current-session it follow-up",
            question="How do I adjust it?",
            expected_path="CURRENT_SESSION_MEMORY",
        ),
        _case(
            name="07 current-session same thing follow-up",
            question="Is it the same thing on the other machine?",
            expected_path="CURRENT_SESSION_MEMORY",
        ),
        _case(
            name="08 clarification simpler steps",
            question="Can you explain that in simpler steps?",
            expected_path="CURRENT_SESSION_MEMORY",
        ),
        _case(
            name="09 clarification continue",
            question="Continue from there.",
            expected_path="CURRENT_SESSION_MEMORY",
        ),
        _case(
            name="10 clarification reword",
            question="Can you reword that?",
            expected_path="CURRENT_SESSION_MEMORY",
        ),
        _case(
            name="11 prior recall earlier",
            question="What did we talk about earlier with RF arc outs?",
            expected_path="SEMANTIC_CHAT_RECALL",
        ),
        _case(
            name="12 prior recall last time",
            question="What did you tell me last time about the arc suppressor?",
            expected_path="SEMANTIC_CHAT_RECALL",
        ),
        _case(
            name="13 prior recall previous conversation",
            question="Do you remember what we discussed before about RF problems?",
            expected_path="SEMANTIC_CHAT_RECALL",
        ),
        _case(
            name="14 prior recall what did we discuss",
            question="What did we discuss earlier?",
            expected_path="SEMANTIC_CHAT_RECALL",
        ),
        _case(
            name="15 document scoped this document",
            question="What does this document say about the stacker?",
            document_scope=active_doc_scope,
            expected_path="DOCUMENT_SCOPE",
        ),
        _case(
            name="16 document scoped in this document",
            question="In this document, what is the setup procedure?",
            document_scope=active_doc_scope,
            expected_path="DOCUMENT_SCOPE",
        ),
        _case(
            name="17 document scoped how does it work",
            question="How does it work?",
            document_scope=active_doc_scope,
            expected_path="DOCUMENT_SCOPE",
        ),
        _case(
            name="18 document scoped summarize",
            question="Summarize this document.",
            document_scope=active_doc_scope,
            expected_path="DOCUMENT_SCOPE",
        ),
        _case(
            name="19 inactive document scope should not force document route",
            question="What does this document say about the stacker?",
            document_scope={
                "enabled": False,
                "scope_type": "complete_document",
                "document_id": 237,
                "complete_document_id": 237,
                "document_name": "POPSFab4",
            },
            expected_path="CURRENT_SESSION_MEMORY",
        ),
        _case(
            name="20 standalone despite history",
            question="What is the purpose of a vacuum generator?",
            expected_path="RAG_ONLY",
        ),
    ]


def run_intent_tests() -> None:
    db = DatabaseConfig()
    coordinator = ChatIntentCoordinator()

    test_cases = _build_test_cases()

    results: List[Dict[str, Any]] = []

    with db.main_session() as session:
        chat_session = _make_test_chat_session(session)
        session.commit()

        _print_block(
            "Created real DB ChatSession",
            {
                "session_id": str(chat_session.session_id),
                "user_id": chat_session.user_id,
                "summary_items": len(chat_session.conversation_summary or []),
                "message_items": len(chat_session.session_data or []),
            },
        )

        for index, case in enumerate(test_cases, start=1):
            started = time.perf_counter()

            try:
                decision = coordinator.classify_question(
                    question=case["question"],
                    chat_session=chat_session,
                    document_scope=case.get("document_scope"),
                    request_id=f"{REQUEST_ID}-{index}",
                )

                elapsed_ms = round((time.perf_counter() - started) * 1000, 2)

                decision_dict = _decision_to_dict(decision)
                actual_path = _resolve_path(decision_dict)

                valid_shape, shape_errors = _is_valid_decision(decision_dict)

                path_passed = actual_path == case["expected_path"]
                confidence = decision_dict.get("confidence") or 0.0
                confidence_passed = confidence >= case["min_confidence"]

                passed = bool(path_passed and valid_shape and confidence_passed)

                result = {
                    "test_number": index,
                    "name": case["name"],
                    "question": case["question"],
                    "expected_path": case["expected_path"],
                    "actual_path": actual_path,
                    "path_passed": path_passed,
                    "valid_shape": valid_shape,
                    "shape_errors": shape_errors,
                    "confidence": confidence,
                    "min_confidence": case["min_confidence"],
                    "confidence_passed": confidence_passed,
                    "passed": passed,
                    "elapsed_ms": elapsed_ms,
                    "decision": decision_dict,
                }

            except Exception as exc:
                elapsed_ms = round((time.perf_counter() - started) * 1000, 2)

                result = {
                    "test_number": index,
                    "name": case["name"],
                    "question": case["question"],
                    "expected_path": case["expected_path"],
                    "actual_path": "EXCEPTION",
                    "passed": False,
                    "elapsed_ms": elapsed_ms,
                    "error": f"{type(exc).__name__}: {exc}",
                }

                error_id(
                    f"[test_chat_intent_pathway] Case failed with exception: "
                    f"{case['name']} -> {exc}",
                    REQUEST_ID,
                    exc_info=True,
                )

            results.append(result)
            _print_block(case["name"], result)

    total = len(results)
    passed_count = sum(1 for item in results if item.get("passed") is True)
    failed_count = total - passed_count
    pass_percent = round((passed_count / total) * 100, 2) if total else 0.0

    failed = [item for item in results if not item.get("passed")]

    summary = {
        "total_tests": total,
        "passed": passed_count,
        "failed": failed_count,
        "pass_percent": pass_percent,
        "target_percent": PASS_TARGET_PERCENT,
        "target_met": pass_percent >= PASS_TARGET_PERCENT,
    }

    _print_block("FINAL TEST SUMMARY", summary)

    if failed:
        _print_block(
            "FAILED CASES",
            [
                {
                    "test_number": item.get("test_number"),
                    "name": item.get("name"),
                    "question": item.get("question"),
                    "expected_path": item.get("expected_path"),
                    "actual_path": item.get("actual_path"),
                    "confidence": item.get("confidence"),
                    "shape_errors": item.get("shape_errors"),
                    "error": item.get("error"),
                    "decision": item.get("decision"),
                }
                for item in failed
            ],
        )

    if pass_percent < PASS_TARGET_PERCENT:
        raise SystemExit(
            f"FAILED: intent pathway pass rate {pass_percent}% "
            f"is below target {PASS_TARGET_PERCENT}%."
        )

    print(
        f"\nPASSED: intent pathway pass rate {pass_percent}% "
        f"meets target {PASS_TARGET_PERCENT}%."
    )


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