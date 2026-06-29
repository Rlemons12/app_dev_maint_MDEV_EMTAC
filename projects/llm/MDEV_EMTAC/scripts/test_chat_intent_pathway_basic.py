# scripts/test_chat_intent_pathway_basic.py

from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Optional

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from modules.coordinators.chat_intent_coordinator import ChatIntentCoordinator


class FakeIntentAI:
    """
    Fake model so we test the intent pathway without RAG or GPU.
    """

    def classify_intent(
        self,
        *,
        prompt: str,
        request_id: Optional[str] = None,
    ) -> str:
        prompt_lower = prompt.lower()

        if "current user question:" in prompt_lower:
            question = prompt_lower.split("current user question:", 1)[1].strip()
        else:
            question = prompt_lower

        if "what were we talking about" in question or "before" in question:
            intent = "RECALL_PRIOR_CONVERSATION"
        elif "this document" in question or "current document" in question:
            intent = "DOCUMENT_SCOPED_FOLLOW_UP"
        elif "explain that" in question or "break that down" in question or "continue" in question:
            intent = "CLARIFICATION"
        elif question.startswith(("what sensor", "what switch", "where is it", "what controls it")):
            intent = "FOLLOW_UP_CURRENT_SESSION"
        else:
            intent = "NEW_TOPIC"

        return json.dumps(
            {
                "intent": intent,
                "confidence": 0.95,
                "needs_current_session_memory": intent in {
                    "FOLLOW_UP_CURRENT_SESSION",
                    "RECALL_PRIOR_CONVERSATION",
                    "DOCUMENT_SCOPED_FOLLOW_UP",
                    "CLARIFICATION",
                },
                "needs_semantic_chat_recall": intent == "RECALL_PRIOR_CONVERSATION",
                "needs_document_scope": intent == "DOCUMENT_SCOPED_FOLLOW_UP",
                "rewritten_question": question,
                "reason": f"Fake classifier selected {intent}",
            }
        )


def make_chat_session() -> SimpleNamespace:
    return SimpleNamespace(
        session_data=[
            {
                "role": "user",
                "content": "What causes a bag indexing fault?",
            },
            {
                "role": "assistant",
                "content": "A bag indexing fault can involve the index dwell prox, photo eyes, or timing issues.",
            },
        ],
        conversation_summary=[
            {
                "question": "What causes a bag indexing fault?",
                "answer_preview": "Discussed bag indexing, sensors, and index dwell prox troubleshooting.",
            }
        ],
    )


def run_case(
    *,
    name: str,
    question: str,
    expected_intent: str,
    chat_session: Optional[Any] = None,
    document_scope: Optional[Dict[str, Any]] = None,
) -> bool:
    coordinator = ChatIntentCoordinator(
        ai_service=FakeIntentAI(),
    )

    decision = coordinator.classify_question(
        question=question,
        chat_session=chat_session,
        document_scope=document_scope,
        request_id=f"test-{name}",
    )

    actual = decision.intent.value

    passed = actual == expected_intent

    print("\n" + "=" * 80)
    print(name)
    print("-" * 80)
    print(f"Question: {question}")
    print(f"Expected: {expected_intent}")
    print(f"Actual:   {actual}")
    print(f"Passed:   {passed}")
    print(f"Decision: {decision}")

    return passed


def main() -> int:
    chat_session = make_chat_session()

    document_scope = {
        "enabled": True,
        "scope_type": "complete_document",
        "complete_document_id": 29,
        "document_name": "Bag Loader Manual",
    }

    cases = [
        {
            "name": "standalone_rag_question",
            "question": "What does a photo eye do on an infeed conveyor?",
            "expected_intent": "NEW_TOPIC",
            "chat_session": chat_session,
        },
        {
            "name": "current_session_follow_up",
            "question": "What sensor does it use?",
            "expected_intent": "FOLLOW_UP_CURRENT_SESSION",
            "chat_session": chat_session,
        },
        {
            "name": "prior_conversation_recall",
            "question": "What were we talking about before this question?",
            "expected_intent": "RECALL_PRIOR_CONVERSATION",
            "chat_session": chat_session,
        },
        {
            "name": "document_scoped_follow_up",
            "question": "What does this document say about setup?",
            "expected_intent": "DOCUMENT_SCOPED_FOLLOW_UP",
            "chat_session": chat_session,
            "document_scope": document_scope,
        },
        {
            "name": "clarification",
            "question": "Can you explain that better?",
            "expected_intent": "CLARIFICATION",
            "chat_session": chat_session,
        },
    ]

    passed_count = 0

    for case in cases:
        if run_case(**case):
            passed_count += 1

    total = len(cases)

    print("\n" + "=" * 80)
    print("INTENT PATHWAY RESULT")
    print("=" * 80)
    print(f"Passed: {passed_count}/{total}")

    if passed_count == total:
        print("SUCCESS: Intent pathway returns properly.")
        return 0

    print("FAILED: One or more intent decisions returned incorrectly.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())