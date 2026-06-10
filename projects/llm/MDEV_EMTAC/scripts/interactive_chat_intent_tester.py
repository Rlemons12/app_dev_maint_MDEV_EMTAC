# scripts/interactive_chat_intent_tester.py

from __future__ import annotations

import json
import sys
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from modules.coordinators.chat_intent_coordinator import ChatIntentCoordinator


REQUEST_ID = "interactive-chat-intent"


class FakeChatSession:
    def __init__(self, *, seed: str = "rf"):
        now = datetime.utcnow().isoformat()

        self.session_id = uuid.uuid4()
        self.user_id = "intent-test-user"
        self.start_time = now
        self.last_interaction = now
        self.session_data: List[Dict[str, Any]] = []
        self.conversation_summary: List[Dict[str, Any]] = []

        if seed == "stacker":
            self.seed_stacker()
        else:
            self.seed_rf()

    def seed_rf(self) -> None:
        self.session_data.clear()
        self.conversation_summary.clear()

        self.add_turn(
            question="I am having RF arc outs on the Bag Fabricator. What should I check?",
            answer=(
                "Check arc suppressor settings, loose generator cabinet connections, "
                "metal objects near transmission lines, and the oscillator tube."
            ),
        )

    def seed_stacker(self) -> None:
        self.session_data.clear()
        self.conversation_summary.clear()

        self.add_turn(
            question="What does this document talk about?",
            answer=(
                "The document talks about the operation of the TTP scanners/verifiers system "
                "in Fab 4, including the pusher, mandrel, port checker, stacker, and inspection process."
            ),
            document_scope={
                "enabled": True,
                "scope_type": "complete_document",
                "document_id": 237,
                "complete_document_id": 237,
                "document_name": "POPSFab4",
            },
        )

    def add_turn(
        self,
        *,
        question: str,
        answer: str,
        document_scope: Optional[Dict[str, Any]] = None,
    ) -> None:
        now = datetime.utcnow().isoformat()

        self.session_data.append(
            {
                "role": "user",
                "content": question,
                "request_id": REQUEST_ID,
                "created_at": now,
                "metadata": {
                    "document_scope": document_scope,
                    "document_scope_enabled": bool(document_scope),
                },
            }
        )

        self.session_data.append(
            {
                "role": "assistant",
                "content": answer,
                "request_id": REQUEST_ID,
                "created_at": now,
                "metadata": {},
            }
        )

        self.conversation_summary.append(
            {
                "request_id": REQUEST_ID,
                "qanda_id": str(uuid.uuid4()),
                "created_at": now,
                "question": question,
                "answer_preview": answer[:1500],
                "document_scope": document_scope,
                "document_scope_enabled": bool(document_scope),
            }
        )

        self.last_interaction = now


def make_document_scope(enabled: bool) -> Optional[Dict[str, Any]]:
    if not enabled:
        return None

    return {
        "enabled": True,
        "scope_type": "complete_document",
        "document_id": 237,
        "complete_document_id": 237,
        "document_name": "POPSFab4",
    }


def resolve_path(decision: Dict[str, Any]) -> str:
    if decision.get("needs_document_scope"):
        return "DOCUMENT_SCOPE"

    if decision.get("needs_semantic_chat_recall"):
        return "SEMANTIC_CHAT_RECALL"

    if decision.get("needs_current_session_memory"):
        return "CURRENT_SESSION_MEMORY"

    if decision.get("intent") == "NEW_TOPIC":
        return "RAG_ONLY"

    return "UNKNOWN"


def print_help() -> None:
    print(
        """
Commands:
  /help              Show this help
  /exit              Quit
  /scope on          Enable fake document scope
  /scope off         Disable fake document scope
  /seed rf           Reset memory to RF arc-out seed
  /seed stacker      Reset memory to POPSFab4 document/stacker seed
  /summary           Show conversation_summary
  /messages          Show recent session_data
  /reset             Reset to RF seed and scope off

Good tests:
  What does a photo eye do on an infeed conveyor?
  I need help with the auto filler
  What about the transfer grippers?
  if the bag indexes what sensor does it use then?
  What were we talking about before this question?
  What does this document say about the stacker?
""".strip()
    )


def print_status(chat_session: FakeChatSession, document_scope_enabled: bool) -> None:
    print("\nState")
    print("-" * 80)
    print(
        json.dumps(
            {
                "session_id": str(chat_session.session_id),
                "summary_items": len(chat_session.conversation_summary),
                "message_items": len(chat_session.session_data),
                "document_scope_enabled": document_scope_enabled,
                "document_scope": make_document_scope(document_scope_enabled),
            },
            indent=2,
            default=str,
        )
    )


def main() -> None:
    coordinator = ChatIntentCoordinator()
    chat_session = FakeChatSession(seed="rf")
    document_scope_enabled = False

    print("\nInteractive Chat Intent Tester")
    print("=" * 80)
    print("Thin runner: calls ChatIntentCoordinator only.")
    print("No DB writes. No RAG execution. Type /help for commands.")
    print("=" * 80)

    while True:
        try:
            question = input("\nintent> ").strip()
        except KeyboardInterrupt:
            print("\nExiting.")
            break

        if not question:
            continue

        command = question.lower()

        if command in {"/exit", "exit", "quit", "/quit"}:
            print("Exiting.")
            break

        if command == "/help":
            print_help()
            continue

        if command == "/scope on":
            document_scope_enabled = True
            print("Document scope enabled.")
            print_status(chat_session, document_scope_enabled)
            continue

        if command == "/scope off":
            document_scope_enabled = False
            print("Document scope disabled.")
            print_status(chat_session, document_scope_enabled)
            continue

        if command == "/seed rf":
            chat_session = FakeChatSession(seed="rf")
            document_scope_enabled = False
            print("Seeded RF arc-out conversation.")
            print_status(chat_session, document_scope_enabled)
            continue

        if command == "/seed stacker":
            chat_session = FakeChatSession(seed="stacker")
            document_scope_enabled = True
            print("Seeded POPSFab4 stacker/document conversation.")
            print_status(chat_session, document_scope_enabled)
            continue

        if command == "/reset":
            chat_session = FakeChatSession(seed="rf")
            document_scope_enabled = False
            print("Reset to RF seed.")
            print_status(chat_session, document_scope_enabled)
            continue

        if command == "/summary":
            print(json.dumps(chat_session.conversation_summary, indent=2, default=str))
            continue

        if command == "/messages":
            print(json.dumps(chat_session.session_data[-10:], indent=2, default=str))
            continue

        document_scope = make_document_scope(document_scope_enabled)

        decision = coordinator.classify_question(
            question=question,
            chat_session=chat_session,
            document_scope=document_scope,
            request_id=REQUEST_ID,
        )

        decision_dict = decision.to_dict()
        route = resolve_path(decision_dict)

        print("\nDecision")
        print("-" * 80)
        print(json.dumps(decision_dict, indent=2, default=str))

        print("\nRoute")
        print("-" * 80)
        print(route)

        fake_answer = (
            f"[TEST ONLY] Route={route}. "
            f"Intent={decision_dict.get('intent')}. "
            f"Rewritten={decision_dict.get('rewritten_question')}"
        )

        chat_session.add_turn(
            question=question,
            answer=fake_answer,
            document_scope=document_scope,
        )


if __name__ == "__main__":
    main()