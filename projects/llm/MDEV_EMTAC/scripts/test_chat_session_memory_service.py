from __future__ import annotations

import argparse
import json
import sys
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional


# ---------------------------------------------------------------------
# Project path setup
# ---------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from modules.configuration.config_env import DatabaseConfig
from modules.services.chat_session_memory_service import ChatSessionMemoryService
from modules.services.qanda_embedding_service import QandAEmbeddingService
from modules.emtacdb.emtacdb_fts import ChatSession


REQUEST_ID = "test-chat-session-memory-service"


# ---------------------------------------------------------------------
# Printing helpers
# ---------------------------------------------------------------------

def print_block(title: str, value: Any) -> None:
    print("\n" + "=" * 100)
    print(title)
    print("=" * 100)

    if isinstance(value, (dict, list)):
        print(json.dumps(value, indent=2, default=str))
    else:
        print(value)


def fail(message: str) -> None:
    raise AssertionError(message)


def require(condition: bool, message: str) -> None:
    if not condition:
        fail(message)


def safe_list(value: Any) -> List[Any]:
    """
    Safely normalize JSON/list values.

    Used for:
        - session_data
        - raw conversation_summary model value
    """

    if value is None:
        return []

    if isinstance(value, list):
        return value

    if isinstance(value, tuple):
        return list(value)

    if isinstance(value, str):
        text = value.strip()

        if not text:
            return []

        parsed = json.loads(text)

        if isinstance(parsed, list):
            return parsed

        return []

    return []


def safe_dict(value: Any) -> Dict[str, Any]:
    """
    Safely normalize JSON/dict values.

    Important:
        ChatSession.conversation_summary is list-like in the current model.

        The clean summary is stored as:

            [
                {
                    "schema_version": 2,
                    "facts": {...},
                    "summary_text": "..."
                }
            ]

        This helper unwraps that single-item list into the inner dict.
    """

    if value is None:
        return {}

    if isinstance(value, dict):
        return value

    if isinstance(value, list):
        if len(value) == 1 and isinstance(value[0], dict):
            return value[0]
        return {}

    if isinstance(value, tuple):
        items = list(value)

        if len(items) == 1 and isinstance(items[0], dict):
            return items[0]

        return {}

    if isinstance(value, str):
        text = value.strip()

        if not text:
            return {}

        parsed = json.loads(text)

        if isinstance(parsed, dict):
            return parsed

        if isinstance(parsed, list):
            if len(parsed) == 1 and isinstance(parsed[0], dict):
                return parsed[0]

        return {}

    return {}


def embedding_is_present(value: Any) -> bool:
    """
    True when summary_embedding appears populated.

    Handles pgvector/list/tuple/numpy-ish values without requiring numpy.
    """

    if value is None:
        return False

    try:
        return len(value) > 0
    except Exception:
        return True


def describe_embedding(value: Any) -> Dict[str, Any]:
    """
    Small safe diagnostic. Does not print the full vector.
    """

    result: Dict[str, Any] = {
        "present": value is not None,
        "type": type(value).__name__ if value is not None else None,
        "length": None,
        "preview": None,
    }

    if value is None:
        return result

    try:
        result["length"] = len(value)
    except Exception:
        result["length"] = None

    try:
        preview_items = list(value)[:5]
        result["preview"] = preview_items
    except Exception:
        result["preview"] = None

    return result


def try_direct_embedding(
    *,
    qanda_embedding_service: Optional[QandAEmbeddingService],
    text: str,
    request_id: str,
) -> Dict[str, Any]:
    """
    Diagnostic only.

    This checks whether QandAEmbeddingService.embed_text works independently
    from ChatSessionMemoryService.update_chat_session_summary_embedding().
    """

    if qanda_embedding_service is None:
        return {
            "attempted": False,
            "success": False,
            "reason": "qanda_embedding_service is None",
        }

    if not hasattr(qanda_embedding_service, "embed_text"):
        return {
            "attempted": False,
            "success": False,
            "reason": "qanda_embedding_service has no embed_text method",
        }

    try:
        vector = qanda_embedding_service.embed_text(
            text=text,
            request_id=request_id,
        )

        return {
            "attempted": True,
            "success": embedding_is_present(vector),
            "embedding": describe_embedding(vector),
        }

    except Exception as exc:
        return {
            "attempted": True,
            "success": False,
            "error_type": type(exc).__name__,
            "error": str(exc),
        }


# ---------------------------------------------------------------------
# Main test
# ---------------------------------------------------------------------

def run_test(*, commit: bool, embedding: bool, user_id: str) -> int:
    """
    Tests ChatSessionMemoryService without touching ChatOrchestrator.

    By default this test rolls back all DB changes.

    Use:
        python scripts/test_chat_session_memory_service.py

    Optional persistent test:
        python scripts/test_chat_session_memory_service.py --commit

    Optional summary embedding test:
        python scripts/test_chat_session_memory_service.py --embedding
    """

    db = DatabaseConfig()

    qanda_embedding_service: Optional[QandAEmbeddingService] = None

    if embedding:
        qanda_embedding_service = QandAEmbeddingService()

    service = ChatSessionMemoryService(
        qanda_embedding_service=qanda_embedding_service,
        store_summary_embedding=embedding,
        include_assistant_messages_in_memory_context=False,
    )

    print_block(
        "TEST CONFIG",
        {
            "commit": commit,
            "embedding": embedding,
            "embedding_service_loaded": qanda_embedding_service is not None,
            "embedding_service_type": (
                type(qanda_embedding_service).__name__
                if qanda_embedding_service is not None
                else None
            ),
            "user_id": user_id,
            "request_id": REQUEST_ID,
        },
    )

    with db.main_session() as session:
        try:
            # ---------------------------------------------------------
            # 1. Create test ChatSession
            # ---------------------------------------------------------

            chat_session, created = service.get_or_create_chat_session(
                session=session,
                conversation_id=None,
                user_id=user_id,
                request_id=REQUEST_ID,
            )

            session.flush()

            conversation_id = str(chat_session.session_id)

            print_block(
                "CREATED CHAT SESSION",
                {
                    "conversation_id": conversation_id,
                    "created": created,
                    "user_id": chat_session.user_id,
                    "session_data": chat_session.session_data,
                    "conversation_summary": chat_session.conversation_summary,
                    "summary_embedding": describe_embedding(
                        getattr(chat_session, "summary_embedding", None)
                    ),
                },
            )

            require(created is True, "Expected a new ChatSession to be created.")
            require(conversation_id, "Expected conversation_id/session_id to exist.")

            raw_created_summary = safe_list(chat_session.conversation_summary)

            require(
                isinstance(chat_session.conversation_summary, list),
                "Raw chat_session.conversation_summary should be list-like for current model compatibility.",
            )

            require(
                len(raw_created_summary) == 1,
                f"Expected new conversation_summary to contain exactly one clean summary object, got {len(raw_created_summary)}.",
            )

            require(
                isinstance(raw_created_summary[0], dict),
                "Expected conversation_summary[0] to be a clean dict summary object.",
            )

            require(
                raw_created_summary[0].get("schema_version") == 2,
                "Expected initial conversation_summary[0].schema_version to be 2.",
            )

            # ---------------------------------------------------------
            # 2. Store first user fact: name
            # ---------------------------------------------------------

            service.store_user_message(
                chat_session=chat_session,
                content="Hi my name is Robert",
                request_id="test-001",
                client_type="web",
                created_session=True,
                document_scope=None,
            )

            service.store_assistant_memory(
                session=session,
                conversation_id=conversation_id,
                answer="Nice to meet you, Robert.",
                question="Hi my name is Robert",
                request_id="test-001",
                qanda_id=None,
                client_type="web",
                document_scope=None,
                intent_decision=None,
            )

            session.flush()
            session.refresh(chat_session)

            # ---------------------------------------------------------
            # 3. Store second user fact: work area
            # ---------------------------------------------------------

            service.store_user_message(
                chat_session=chat_session,
                content="I work in Overwrap",
                request_id="test-002",
                client_type="web",
                created_session=False,
                document_scope=None,
            )

            service.store_assistant_memory(
                session=session,
                conversation_id=conversation_id,
                answer="Got it — you work in Overwrap.",
                question="I work in Overwrap",
                request_id="test-002",
                qanda_id=None,
                client_type="web",
                document_scope=None,
                intent_decision=None,
            )

            session.flush()
            session.refresh(chat_session)

            # ---------------------------------------------------------
            # 4. Add an equipment question and a deliberately bad assistant
            #    answer to prove assistant text does not pollute memory.
            # ---------------------------------------------------------

            service.store_user_message(
                chat_session=chat_session,
                content="What should I do if the indexer is not at the forward or back position?",
                request_id="test-003",
                client_type="web",
                created_session=False,
                document_scope={
                    "enabled": True,
                    "scope_type": "complete_document",
                    "document_id": 251,
                    "complete_document_id": 251,
                    "document_name": "IrrigationFab 7 TSG",
                },
            )

            service.store_assistant_memory(
                session=session,
                conversation_id=conversation_id,
                answer="=251)",
                question="What should I do if the indexer is not at the forward or back position?",
                request_id="test-003",
                qanda_id=None,
                client_type="web",
                document_scope={
                    "enabled": True,
                    "scope_type": "complete_document",
                    "document_id": 251,
                    "complete_document_id": 251,
                    "document_name": "IrrigationFab 7 TSG",
                },
                intent_decision=None,
            )

            session.flush()
            session.refresh(chat_session)

            # ---------------------------------------------------------
            # 5. Append current recall question, then build memory.
            #    The service should exclude this last user message.
            # ---------------------------------------------------------

            service.store_user_message(
                chat_session=chat_session,
                content="What is my name?",
                request_id="test-004",
                client_type="web",
                created_session=False,
                document_scope=None,
            )

            session.flush()
            session.refresh(chat_session)

            memory_context = service.build_memory_context_text(
                chat_session=chat_session,
                exclude_last_user_message=True,
            )

            # ---------------------------------------------------------
            # 6. Validate raw session_data
            # ---------------------------------------------------------

            session_data = safe_list(chat_session.session_data)

            print_block("SESSION DATA", session_data)

            require(
                len(session_data) >= 7,
                f"Expected at least 7 messages in session_data, got {len(session_data)}.",
            )

            require(
                any(
                    isinstance(item, dict)
                    and item.get("role") == "user"
                    and "Hi my name is Robert" in str(item.get("content") or "")
                    for item in session_data
                ),
                "Expected session_data to contain the name user message.",
            )

            require(
                any(
                    isinstance(item, dict)
                    and item.get("role") == "assistant"
                    and "Nice to meet you" in str(item.get("content") or "")
                    for item in session_data
                ),
                "Expected session_data to contain assistant acknowledgement.",
            )

            # ---------------------------------------------------------
            # 7. Validate raw and unwrapped clean conversation_summary
            # ---------------------------------------------------------

            raw_conversation_summary = safe_list(chat_session.conversation_summary)
            conversation_summary = safe_dict(chat_session.conversation_summary)

            print_block("RAW CONVERSATION SUMMARY", raw_conversation_summary)
            print_block("UNWRAPPED CONVERSATION SUMMARY", conversation_summary)

            require(
                isinstance(chat_session.conversation_summary, list),
                "Raw conversation_summary should remain list-like for the current SQLAlchemy model.",
            )

            require(
                len(raw_conversation_summary) == 1,
                f"Expected raw conversation_summary to contain exactly one clean summary object, got {len(raw_conversation_summary)}.",
            )

            require(
                isinstance(conversation_summary, dict),
                "conversation_summary should unwrap to a clean schema_version=2 dict.",
            )

            require(
                conversation_summary.get("schema_version") == 2,
                "conversation_summary.schema_version should be 2.",
            )

            require(
                "answer_preview" not in conversation_summary,
                "conversation_summary should not have top-level answer_preview.",
            )

            require(
                "question" not in conversation_summary,
                "conversation_summary should not have top-level question.",
            )

            facts = conversation_summary.get("facts")

            require(isinstance(facts, dict), "conversation_summary.facts should be a dict.")

            user_name = (
                facts.get("user_name", {}).get("value")
                if isinstance(facts.get("user_name"), dict)
                else None
            )

            work_area = (
                facts.get("work_area", {}).get("value")
                if isinstance(facts.get("work_area"), dict)
                else None
            )

            require(
                user_name == "Robert",
                f"Expected user_name fact to be Robert, got {user_name!r}.",
            )

            require(
                work_area == "Overwrap",
                f"Expected work_area fact to be Overwrap, got {work_area!r}.",
            )

            summary_text = str(conversation_summary.get("summary_text") or "").strip()

            require(
                "Robert" in summary_text,
                "summary_text should mention Robert.",
            )

            require(
                "Overwrap" in summary_text,
                "summary_text should mention Overwrap.",
            )

            require(
                "=251)" not in summary_text,
                "summary_text should not contain leaked bad assistant answer '=251)'.",
            )

            require(
                summary_text,
                "summary_text should not be empty before embedding test.",
            )

            # ---------------------------------------------------------
            # 8. Validate memory context
            # ---------------------------------------------------------

            print_block("MEMORY CONTEXT", memory_context)

            require(
                "User name: Robert" in memory_context,
                "memory_context should contain clean User name fact.",
            )

            require(
                "User work area: Overwrap" in memory_context,
                "memory_context should contain clean work area fact.",
            )

            require(
                "What is my name?" not in memory_context,
                "memory_context should exclude the current in-flight user question.",
            )

            require(
                "=251)" not in memory_context,
                "memory_context should not include bad assistant leakage '=251)'.",
            )

            require(
                "--- USER MESSAGE START ---" not in memory_context,
                "memory_context should not include prompt wrapper leakage.",
            )

            require(
                "--- ASSISTANT MESSAGE START ---" not in memory_context,
                "memory_context should not include assistant wrapper leakage.",
            )

            # ---------------------------------------------------------
            # 9. Validate summary embedding behavior
            # ---------------------------------------------------------

            if embedding:
                print_block(
                    "SUMMARY EMBEDDING INPUT",
                    {
                        "summary_text_chars": len(summary_text),
                        "summary_text": summary_text,
                        "service_store_summary_embedding": getattr(
                            service,
                            "store_summary_embedding",
                            None,
                        ),
                        "service_has_qanda_embedding_service": getattr(
                            service,
                            "qanda_embedding_service",
                            None,
                        )
                        is not None,
                        "test_has_qanda_embedding_service": qanda_embedding_service is not None,
                    },
                )

                direct_embedding_check = try_direct_embedding(
                    qanda_embedding_service=qanda_embedding_service,
                    text=summary_text,
                    request_id="test-direct-summary-embedding",
                )

                print_block("DIRECT EMBEDDING DIAGNOSTIC", direct_embedding_check)

                update_result = service.update_chat_session_summary_embedding(
                    chat_session=chat_session,
                    request_id="test-forced-summary-embedding",
                )

                session.flush()
                session.refresh(chat_session)

                summary_embedding = getattr(chat_session, "summary_embedding", None)

                print_block(
                    "SUMMARY EMBEDDING",
                    {
                        "update_result": update_result,
                        "summary_embedding": describe_embedding(summary_embedding),
                    },
                )

                require(
                    direct_embedding_check.get("success") is True,
                    "Direct QandAEmbeddingService.embed_text() did not return a usable embedding. "
                    "Check the DIRECT EMBEDDING DIAGNOSTIC block above.",
                )

                require(
                    update_result is True,
                    "ChatSessionMemoryService.update_chat_session_summary_embedding() returned False. "
                    "Check the SUMMARY EMBEDDING INPUT and SUMMARY EMBEDDING blocks above.",
                )

                require(
                    embedding_is_present(summary_embedding),
                    "Expected summary_embedding to be populated when --embedding is used.",
                )

            else:
                print_block(
                    "SUMMARY EMBEDDING",
                    "Skipped because --embedding was not used.",
                )

            # ---------------------------------------------------------
            # 10. Optional reload check by session_id
            # ---------------------------------------------------------

            reloaded = session.get(ChatSession, chat_session.session_id)

            require(reloaded is not None, "Expected to reload ChatSession by session_id.")

            raw_reloaded_summary = safe_list(reloaded.conversation_summary)
            reloaded_summary = safe_dict(reloaded.conversation_summary)

            require(
                isinstance(reloaded.conversation_summary, list),
                "Reloaded raw conversation_summary should remain list-like.",
            )

            require(
                len(raw_reloaded_summary) == 1,
                f"Reloaded raw conversation_summary should contain exactly one item, got {len(raw_reloaded_summary)}.",
            )

            require(
                reloaded_summary.get("schema_version") == 2,
                "Reloaded conversation_summary should still be schema_version 2.",
            )

            print_block(
                "RELOAD CHECK",
                {
                    "conversation_id": str(reloaded.session_id),
                    "raw_summary_items": len(raw_reloaded_summary),
                    "summary_schema_version": reloaded_summary.get("schema_version"),
                    "facts": reloaded_summary.get("facts"),
                    "last_interaction": reloaded.last_interaction,
                    "summary_embedding": describe_embedding(
                        getattr(reloaded, "summary_embedding", None)
                    ),
                },
            )

            if commit:
                session.commit()
                print_block(
                    "RESULT",
                    {
                        "status": "PASS",
                        "committed": True,
                        "conversation_id": conversation_id,
                    },
                )
            else:
                session.rollback()
                print_block(
                    "RESULT",
                    {
                        "status": "PASS",
                        "committed": False,
                        "rolled_back": True,
                        "conversation_id": conversation_id,
                    },
                )

            return 0

        except Exception:
            session.rollback()

            print_block("RESULT", {"status": "FAIL", "rolled_back": True})
            traceback.print_exc()

            return 1


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Test ChatSessionMemoryService before wiring it into ChatOrchestrator."
    )

    parser.add_argument(
        "--commit",
        action="store_true",
        help="Commit the test ChatSession row instead of rolling back.",
    )

    parser.add_argument(
        "--embedding",
        action="store_true",
        help="Also test summary_embedding generation.",
    )

    parser.add_argument(
        "--user-id",
        default="memory-service-test",
        help="Test user_id to use.",
    )

    args = parser.parse_args()

    return run_test(
        commit=bool(args.commit),
        embedding=bool(args.embedding),
        user_id=str(args.user_id or "memory-service-test"),
    )


if __name__ == "__main__":
    raise SystemExit(main())