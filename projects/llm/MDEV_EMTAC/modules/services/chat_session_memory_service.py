from __future__ import annotations

import json
import re
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple
from uuid import UUID

from sqlalchemy.orm import Session

from modules.configuration.log_config import (
    debug_id,
    info_id,
    warning_id,
    error_id,
)

from modules.emtacdb.emtacdb_fts import ChatSession

try:
    from modules.services.qanda_embedding_service import QandAEmbeddingService
except Exception:  # pragma: no cover
    QandAEmbeddingService = None  # type: ignore


class ChatSessionMemoryService:
    """
    Owns all ChatSession/session-memory behavior.

    This service is intentionally responsible for:
        - Creating/loading ChatSession rows
        - Appending raw user/assistant messages to session_data
        - Maintaining a CLEAN conversation_summary object
        - Building safe current-session memory context for prompts
        - Updating summary_embedding from the clean summary only
        - Updating last_interaction

    This service intentionally does NOT:
        - Own database transactions
        - Commit
        - Rollback
        - Call the AI answer pipeline
        - Write QandA rows
        - Write audit records

    Important design rule:
        session_data is the raw chat transcript.
        conversation_summary is NOT a rolling Q&A log.
        conversation_summary is clean state/facts only.
    """

    CHAT_SESSION_MAX_MESSAGES = 100

    MEMORY_RECENT_MESSAGE_LIMIT = 10
    MEMORY_PREVIEW_CHARS = 1000

    SUMMARY_SCHEMA_VERSION = 2
    SUMMARY_TOPIC_LIMIT = 8
    SUMMARY_TEXT_MAX_CHARS = 2000

    # Keep this enabled because your existing schema already has summary_embedding.
    # The embedding is now created from clean summary_text, not polluted Q&A previews.
    DEFAULT_STORE_SUMMARY_EMBEDDING = True

    def __init__(
        self,
        *,
        qanda_embedding_service: Optional[Any] = None,
        store_summary_embedding: bool = DEFAULT_STORE_SUMMARY_EMBEDDING,
        include_assistant_messages_in_memory_context: bool = False,
        chat_session_max_messages: int = CHAT_SESSION_MAX_MESSAGES,
        memory_recent_message_limit: int = MEMORY_RECENT_MESSAGE_LIMIT,
        memory_preview_chars: int = MEMORY_PREVIEW_CHARS,
        summary_topic_limit: int = SUMMARY_TOPIC_LIMIT,
        summary_text_max_chars: int = SUMMARY_TEXT_MAX_CHARS,
    ) -> None:
        self.qanda_embedding_service = qanda_embedding_service

        if self.qanda_embedding_service is None and QandAEmbeddingService is not None:
            try:
                self.qanda_embedding_service = QandAEmbeddingService()
            except Exception:
                self.qanda_embedding_service = None

        self.store_summary_embedding = bool(store_summary_embedding)
        self.include_assistant_messages_in_memory_context = bool(
            include_assistant_messages_in_memory_context
        )

        self.chat_session_max_messages = int(
            chat_session_max_messages or self.CHAT_SESSION_MAX_MESSAGES
        )
        self.memory_recent_message_limit = int(
            memory_recent_message_limit or self.MEMORY_RECENT_MESSAGE_LIMIT
        )
        self.memory_preview_chars = int(memory_preview_chars or self.MEMORY_PREVIEW_CHARS)
        self.summary_topic_limit = int(summary_topic_limit or self.SUMMARY_TOPIC_LIMIT)
        self.summary_text_max_chars = int(
            summary_text_max_chars or self.SUMMARY_TEXT_MAX_CHARS
        )

    # -------------------------------------------------------------------------
    # ChatSession create/load
    # -------------------------------------------------------------------------

    def get_or_create_chat_session(
        self,
        *,
        session: Session,
        conversation_id: Optional[str],
        user_id: str,
        request_id: Optional[str] = None,
    ) -> Tuple[ChatSession, bool]:
        """
        Load an existing ChatSession by conversation_id, or create a new one.

        Returns:
            (chat_session, created_session)
        """

        parsed_conversation_id = self._coerce_uuid_or_none(conversation_id)

        if parsed_conversation_id is not None:
            existing_session = session.get(ChatSession, parsed_conversation_id)

            if existing_session is not None:
                existing_user_id = str(getattr(existing_session, "user_id", "") or "")
                requested_user_id = str(user_id or "")

                if existing_user_id == requested_user_id:
                    self._ensure_chat_session_defaults(existing_session)
                    return existing_session, False

                warning_id(
                    "[ChatSessionMemoryService] Conversation ID belongs to a different user. "
                    f"incoming_conversation_id={conversation_id} "
                    f"existing_user_id={existing_user_id} "
                    f"requested_user_id={requested_user_id}. "
                    "Creating a new ChatSession.",
                    request_id,
                )

            else:
                warning_id(
                    "[ChatSessionMemoryService] Conversation ID was supplied but no "
                    f"ChatSession was found. incoming_conversation_id={conversation_id}. "
                    "Creating a new ChatSession.",
                    request_id,
                )

        elif conversation_id:
            warning_id(
                "[ChatSessionMemoryService] Invalid conversation_id supplied: "
                f"{conversation_id!r}. Creating a new ChatSession.",
                request_id,
            )

        now = self._utc_iso()

        chat_session = ChatSession(
            user_id=str(user_id or "anonymous"),
            start_time=now,
            last_interaction=now,
            session_data=[],
            conversation_summary=self._empty_summary(now=now),
        )

        session.add(chat_session)
        session.flush()

        return chat_session, True

    # Backward-compatible alias name.
    def get_or_create_chat_session(
            self,
            *,
            session: Session,
            conversation_id: Optional[str],
            user_id: str,
            request_id: Optional[str] = None,
    ) -> Tuple[ChatSession, bool]:
        """
        Load an existing ChatSession by conversation_id, or create a new one.

        Returns:
            (chat_session, created_session)

        Important:
            ChatSession.conversation_summary is currently mapped as a list-like
            mutable JSON field. Therefore the clean summary object is stored as:

                [
                    {
                        "schema_version": 2,
                        "facts": {},
                        "summary_text": "",
                        ...
                    }
                ]

            Do NOT assign a raw dict directly to conversation_summary unless the
            SQLAlchemy model is changed from MutableList to MutableDict.
        """

        normalized_user_id = str(user_id or "anonymous").strip() or "anonymous"
        normalized_conversation_id = (
            str(conversation_id).strip()
            if conversation_id is not None and str(conversation_id).strip()
            else None
        )

        parsed_conversation_id = self._coerce_uuid_or_none(normalized_conversation_id)

        # ------------------------------------------------------------------
        # 1. Try to load existing session when a valid conversation_id arrives.
        # ------------------------------------------------------------------

        if parsed_conversation_id is not None:
            existing_session = session.get(ChatSession, parsed_conversation_id)

            if existing_session is not None:
                existing_user_id = str(getattr(existing_session, "user_id", "") or "")
                requested_user_id = normalized_user_id

                if existing_user_id == requested_user_id:
                    self._ensure_chat_session_defaults(existing_session)

                    debug_id(
                        "[ChatSessionMemoryService] Loaded existing ChatSession "
                        f"conversation_id={existing_session.session_id} "
                        f"user_id={requested_user_id}",
                        request_id,
                    )

                    return existing_session, False

                warning_id(
                    "[ChatSessionMemoryService] Conversation ID belongs to a different user. "
                    f"incoming_conversation_id={normalized_conversation_id} "
                    f"existing_user_id={existing_user_id} "
                    f"requested_user_id={requested_user_id}. "
                    "Creating a new ChatSession.",
                    request_id,
                )

            else:
                warning_id(
                    "[ChatSessionMemoryService] Conversation ID was supplied but no "
                    f"ChatSession was found. incoming_conversation_id={normalized_conversation_id}. "
                    "Creating a new ChatSession.",
                    request_id,
                )

        elif normalized_conversation_id:
            warning_id(
                "[ChatSessionMemoryService] Invalid conversation_id supplied: "
                f"{normalized_conversation_id!r}. Creating a new ChatSession.",
                request_id,
            )

        # ------------------------------------------------------------------
        # 2. Create a new ChatSession.
        # ------------------------------------------------------------------
        # conversation_summary MUST be list-like for the current model.
        # Store the clean schema_version=2 summary object inside a single-item list.
        # ------------------------------------------------------------------

        now = self._utc_iso()

        chat_session = ChatSession(
            user_id=normalized_user_id,
            start_time=now,
            last_interaction=now,
            session_data=[],
            conversation_summary=[self._empty_summary(now=now)],
        )

        session.add(chat_session)
        session.flush()

        info_id(
            "[ChatSessionMemoryService] Created new ChatSession "
            f"conversation_id={chat_session.session_id} "
            f"user_id={normalized_user_id}",
            request_id,
        )

        return chat_session, True

    def _ensure_chat_session_defaults(self, chat_session: ChatSession) -> None:
        """
        Ensures existing sessions have safe default JSON fields.

        Important:
            ChatSession.conversation_summary is currently mapped as a list-like
            mutable JSON field.

            So the clean schema_version=2 summary must be stored as:

                [
                    {
                        "schema_version": 2,
                        "facts": {},
                        "summary_text": "",
                        ...
                    }
                ]

            Do NOT assign a raw dict directly to conversation_summary unless the
            SQLAlchemy model is changed from MutableList to MutableDict.

        This method does not erase session_data.
        It also does not try to migrate legacy polluted summary rows here.
        Legacy cleanup is handled by _safe_summary_dict() when building/updating
        the clean summary.
        """

        current_session_data = getattr(chat_session, "session_data", None)
        current_summary = getattr(chat_session, "conversation_summary", None)

        # session_data should always be a list because it stores the raw transcript.
        if current_session_data is None:
            chat_session.session_data = []

        elif isinstance(current_session_data, str):
            text = current_session_data.strip()

            if not text:
                chat_session.session_data = []
            else:
                try:
                    parsed = json.loads(text)

                    if isinstance(parsed, list):
                        chat_session.session_data = parsed
                    else:
                        chat_session.session_data = []
                except Exception:
                    chat_session.session_data = []

        elif isinstance(current_session_data, tuple):
            chat_session.session_data = list(current_session_data)

        elif not isinstance(current_session_data, list):
            chat_session.session_data = []

        # conversation_summary must remain list-like for the current SQLAlchemy model.
        if current_summary is None:
            chat_session.conversation_summary = [self._empty_summary()]
            return

        if isinstance(current_summary, str):
            text = current_summary.strip()

            if not text:
                chat_session.conversation_summary = [self._empty_summary()]
                return

            try:
                parsed = json.loads(text)
            except Exception:
                chat_session.conversation_summary = [self._empty_summary()]
                return

            if isinstance(parsed, list):
                chat_session.conversation_summary = parsed
                return

            if isinstance(parsed, dict):
                chat_session.conversation_summary = [parsed]
                return

            chat_session.conversation_summary = [self._empty_summary()]
            return

        if isinstance(current_summary, dict):
            # Defensive only. Your current model usually raises before allowing this,
            # but this keeps the service safe if the model changes later.
            chat_session.conversation_summary = [current_summary]
            return

        if isinstance(current_summary, tuple):
            chat_session.conversation_summary = list(current_summary)
            return

        if isinstance(current_summary, list):
            if not current_summary:
                chat_session.conversation_summary = [self._empty_summary()]
            return

        chat_session.conversation_summary = [self._empty_summary()]

    # -------------------------------------------------------------------------
    # Message storage
    # -------------------------------------------------------------------------

    def append_chat_message(
        self,
        *,
        chat_session: ChatSession,
        role: str,
        content: str,
        request_id: Optional[str],
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Append a raw message to chat_session.session_data.

        session_data is allowed to contain raw user/assistant messages.
        It is the transcript.
        """

        messages = self._safe_list(getattr(chat_session, "session_data", None))

        normalized_role = str(role or "unknown").strip().lower() or "unknown"

        messages.append(
            {
                "role": normalized_role,
                "content": self._clean_raw_message_content(content),
                "request_id": request_id,
                "created_at": self._utc_iso(),
                "metadata": metadata or {},
            }
        )

        if len(messages) > self.chat_session_max_messages:
            messages = messages[-self.chat_session_max_messages :]

        chat_session.session_data = messages

    # Backward-compatible alias.
    def _append_chat_message(
        self,
        *,
        chat_session: ChatSession,
        role: str,
        content: str,
        request_id: Optional[str],
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.append_chat_message(
            chat_session=chat_session,
            role=role,
            content=content,
            request_id=request_id,
            metadata=metadata,
        )

    def store_user_message(
        self,
        *,
        chat_session: ChatSession,
        content: str,
        request_id: Optional[str],
        client_type: str,
        created_session: bool,
        document_scope: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Convenience method for storing the user message.
        """

        normalized_document_scope = self.normalize_document_scope(document_scope)

        self.append_chat_message(
            chat_session=chat_session,
            role="user",
            content=content,
            request_id=request_id,
            metadata={
                "client_type": str(client_type or "web").strip().lower() or "web",
                "created_session": bool(created_session),
                "document_scope": normalized_document_scope,
                "document_scope_enabled": bool(normalized_document_scope),
            },
        )

        self.touch_chat_session(chat_session)

    def store_assistant_memory(
        self,
        *,
        session: Session,
        conversation_id: str,
        answer: str,
        question: str,
        request_id: Optional[str],
        qanda_id: Optional[Any],
        client_type: str,
        document_scope: Optional[Dict[str, Any]] = None,
        intent_decision: Optional[Any] = None,
    ) -> None:
        """
        Store the assistant answer in session_data and update clean summary state.

        This replaces the old behavior where conversation_summary stored a rolling
        list of:
            question / answer_preview / qanda_id / intent / document_scope

        That old structure polluted memory. This method keeps the assistant answer
        only in session_data and stores clean facts/topics in conversation_summary.
        """

        parsed_conversation_id = self._coerce_uuid_or_none(conversation_id)
        normalized_document_scope = self.normalize_document_scope(document_scope)

        if parsed_conversation_id is None:
            warning_id(
                "[ChatSessionMemoryService] Cannot store assistant memory because "
                f"conversation_id is invalid: {conversation_id!r}",
                request_id,
            )
            return

        chat_session = session.get(ChatSession, parsed_conversation_id)

        if chat_session is None:
            warning_id(
                "[ChatSessionMemoryService] Cannot store assistant memory because "
                f"ChatSession was not found. conversation_id={conversation_id}",
                request_id,
            )
            return

        self._ensure_chat_session_defaults(chat_session)

        self.append_chat_message(
            chat_session=chat_session,
            role="assistant",
            content=answer or "",
            request_id=request_id,
            metadata={
                "client_type": str(client_type or "web").strip().lower() or "web",
                "qanda_id": str(qanda_id) if qanda_id else None,
                "document_scope": normalized_document_scope,
                "document_scope_enabled": bool(normalized_document_scope),
                "intent": self.intent_decision_to_dict(intent_decision),
            },
        )

        self.update_conversation_summary(
            chat_session=chat_session,
            latest_question=question,
            latest_answer=answer or "",
            request_id=request_id,
            qanda_id=qanda_id,
            document_scope=normalized_document_scope,
            intent_decision=intent_decision,
        )

        self.update_chat_session_summary_embedding(
            chat_session=chat_session,
            request_id=request_id,
        )

        self.touch_chat_session(chat_session)

    # Backward-compatible alias.
    def _store_assistant_memory(
        self,
        *,
        session: Session,
        conversation_id: str,
        answer: str,
        question: str,
        request_id: Optional[str],
        qanda_id: Optional[Any],
        client_type: str,
        document_scope: Optional[Dict[str, Any]] = None,
        intent_decision: Optional[Any] = None,
    ) -> None:
        self.store_assistant_memory(
            session=session,
            conversation_id=conversation_id,
            answer=answer,
            question=question,
            request_id=request_id,
            qanda_id=qanda_id,
            client_type=client_type,
            document_scope=document_scope,
            intent_decision=intent_decision,
        )

    # -------------------------------------------------------------------------
    # Clean conversation_summary
    # -------------------------------------------------------------------------
    def update_conversation_summary(
            self,
            *,
            chat_session: ChatSession,
            latest_question: str,
            latest_answer: str,
            request_id: Optional[str],
            qanda_id: Optional[Any],
            document_scope: Optional[Dict[str, Any]] = None,
            intent_decision: Optional[Any] = None,
    ) -> Dict[str, Any]:
        """
        Updates conversation_summary as a clean summary object.

        Important:
            ChatSession.conversation_summary is currently mapped as a list-like
            mutable JSON field.

            Therefore the clean summary object must be stored as:

                [
                    {
                        "schema_version": 2,
                        "facts": {...},
                        "summary_text": "...",
                        ...
                    }
                ]

            Do NOT assign the clean summary dict directly to
            chat_session.conversation_summary.
        """

        now = self._utc_iso()
        normalized_document_scope = self.normalize_document_scope(document_scope)

        messages = self._safe_list(getattr(chat_session, "session_data", None))

        previous_summary = self._safe_summary_dict(
            getattr(chat_session, "conversation_summary", None)
        )

        facts = self.extract_user_facts_from_messages(messages)

        # Preserve previous facts if the current message scan finds nothing.
        previous_facts = previous_summary.get("facts")
        if not facts and isinstance(previous_facts, dict):
            facts = previous_facts

        recent_user_topics = self.extract_recent_user_topics_from_messages(
            messages=messages,
            limit=self.summary_topic_limit,
        )

        current_topic = (
            recent_user_topics[-1].get("text")
            if recent_user_topics
            else previous_summary.get("current_topic")
        )

        clean_summary: Dict[str, Any] = {
            "schema_version": self.SUMMARY_SCHEMA_VERSION,
            "facts": facts,
            "current_topic": current_topic,
            "recent_user_topics": recent_user_topics,
            "active_document": normalized_document_scope,
            "summary_text": "",
            "last_request_id": request_id,
            "last_qanda_id": str(qanda_id) if qanda_id else None,
            "last_intent": self.intent_decision_to_dict(intent_decision),
            "updated_at": now,
        }

        clean_summary["summary_text"] = self.build_summary_text(clean_summary)

        # CRITICAL:
        # conversation_summary must stay list-like for the current SQLAlchemy model.
        # The clean summary object is stored as the only item in the list.
        chat_session.conversation_summary = [clean_summary]

        debug_id(
            "[ChatSessionMemoryService] Clean conversation_summary updated "
            f"conversation_id={getattr(chat_session, 'session_id', None)} "
            f"facts={list(facts.keys()) if isinstance(facts, dict) else []} "
            f"recent_topics={len(recent_user_topics)} "
            f"summary_text_chars={len(clean_summary.get('summary_text') or '')}",
            request_id,
        )

        return clean_summary

    def append_conversation_summary_item(
        self,
        *,
        chat_session: ChatSession,
        question: str,
        answer: str,
        request_id: Optional[str],
        qanda_id: Optional[Any],
        document_scope: Optional[Dict[str, Any]] = None,
        intent_decision: Optional[Any] = None,
    ) -> None:
        """
        Backward-compatible method name.

        The old orchestrator used this name to append polluted Q&A preview items.
        This method intentionally DOES NOT append Q&A preview items anymore.
        It updates the clean summary object instead.
        """

        self.update_conversation_summary(
            chat_session=chat_session,
            latest_question=question,
            latest_answer=answer,
            request_id=request_id,
            qanda_id=qanda_id,
            document_scope=document_scope,
            intent_decision=intent_decision,
        )

    # Backward-compatible alias.
    def _append_conversation_summary_item(
        self,
        *,
        chat_session: ChatSession,
        question: str,
        answer: str,
        request_id: Optional[str],
        qanda_id: Optional[Any],
        document_scope: Optional[Dict[str, Any]] = None,
        intent_decision: Optional[Any] = None,
    ) -> None:
        self.append_conversation_summary_item(
            chat_session=chat_session,
            question=question,
            answer=answer,
            request_id=request_id,
            qanda_id=qanda_id,
            document_scope=document_scope,
            intent_decision=intent_decision,
        )

    def update_chat_session_summary_embedding(
            self,
            *,
            chat_session: ChatSession,
            request_id: Optional[str],
    ) -> bool:
        """
        Update chat_sessions.summary_embedding from the clean summary_text.

        Important:
            conversation_summary is list-like in the current SQLAlchemy model.

            Clean format:

                [
                    {
                        "schema_version": 2,
                        "summary_text": "..."
                    }
                ]

            SQLAlchemy may return the value as MutableList. This method relies on
            _safe_summary_dict() to unwrap that safely before reading summary_text.
        """

        if not self.store_summary_embedding:
            debug_id(
                "[ChatSessionMemoryService] Summary embedding skipped because "
                "store_summary_embedding=False.",
                request_id,
            )
            return False

        raw_summary = getattr(chat_session, "conversation_summary", None)
        clean_summary = self._safe_summary_dict(raw_summary)

        summary_text = str(clean_summary.get("summary_text") or "").strip()

        if not summary_text:
            chat_session.summary_embedding = None

            warning_id(
                "[ChatSessionMemoryService] Chat session summary embedding skipped "
                "because clean summary_text is empty. "
                f"conversation_id={getattr(chat_session, 'session_id', None)} "
                f"raw_summary_type={type(raw_summary).__name__} "
                f"clean_summary_keys={list(clean_summary.keys()) if isinstance(clean_summary, dict) else None} "
                f"raw_summary_preview={str(raw_summary)[:500]!r}",
                request_id,
            )

            return False

        if self.is_bad_assistant_memory_content(summary_text):
            chat_session.summary_embedding = None

            warning_id(
                "[ChatSessionMemoryService] Chat session summary embedding skipped "
                "because summary_text looked unsafe/polluted. "
                f"conversation_id={getattr(chat_session, 'session_id', None)} "
                f"summary_text_preview={summary_text[:300]!r}",
                request_id,
            )

            return False

        if self.qanda_embedding_service is None:
            warning_id(
                "[ChatSessionMemoryService] Chat session summary embedding skipped "
                "because qanda_embedding_service is missing.",
                request_id,
            )
            return False

        if not hasattr(self.qanda_embedding_service, "embed_text"):
            warning_id(
                "[ChatSessionMemoryService] Chat session summary embedding skipped "
                "because qanda_embedding_service does not expose embed_text().",
                request_id,
            )
            return False

        try:
            embedding = self.qanda_embedding_service.embed_text(
                text=summary_text,
                request_id=request_id,
            )

            if not embedding:
                warning_id(
                    "[ChatSessionMemoryService] Chat session summary embedding returned empty. "
                    f"conversation_id={getattr(chat_session, 'session_id', None)} "
                    f"summary_text_chars={len(summary_text)}",
                    request_id,
                )
                return False

            chat_session.summary_embedding = embedding

            info_id(
                "[ChatSessionMemoryService] Chat session summary embedding updated "
                f"conversation_id={getattr(chat_session, 'session_id', None)} "
                f"summary_text_chars={len(summary_text)} "
                f"embedding_dims={len(embedding)}",
                request_id,
            )

            return True

        except Exception as exc:
            error_id(
                "[ChatSessionMemoryService] Failed to update chat session summary embedding: "
                f"{type(exc).__name__}: {exc}",
                request_id,
                exc_info=True,
            )
            return False

    # Backward-compatible alias.
    def _update_chat_session_summary_embedding(
        self,
        *,
        chat_session: ChatSession,
        request_id: Optional[str],
    ) -> bool:
        return self.update_chat_session_summary_embedding(
            chat_session=chat_session,
            request_id=request_id,
        )

    # -------------------------------------------------------------------------
    # Memory context builder
    # -------------------------------------------------------------------------

    def build_memory_context_text(
        self,
        *,
        chat_session: ChatSession,
        exclude_last_user_message: bool = True,
    ) -> str:
        """
        Build safe prompt-ready current-session memory context.

        This intentionally avoids the old polluted format:

            Rolling conversation summary:
            - Prior question: ...
              Prior answer summary: ...

        Instead, it uses:
            - clean facts from conversation_summary
            - recent prior user messages from session_data

        Assistant messages are excluded by default because bad assistant answers
        can poison future answers. Enable include_assistant_messages_in_memory_context
        only after the recall pathway is stable.
        """

        sections: List[str] = []

        messages = self._safe_list(getattr(chat_session, "session_data", None))

        prior_messages = list(messages)

        if exclude_last_user_message and prior_messages:
            last_item = prior_messages[-1]

            if isinstance(last_item, dict):
                last_role = str(last_item.get("role") or "").strip().lower()

                if last_role == "user":
                    prior_messages = prior_messages[:-1]

        summary = self._safe_summary_dict(
            getattr(chat_session, "conversation_summary", None)
        )

        facts = summary.get("facts")

        if not isinstance(facts, dict) or not facts:
            facts = self.extract_user_facts_from_messages(prior_messages)

        fact_lines = self.render_fact_lines(facts)

        if fact_lines:
            sections.append("Known facts from this conversation:\n" + "\n".join(fact_lines))

        recent_lines = self.render_recent_prior_messages(
            messages=prior_messages,
            limit=self.memory_recent_message_limit,
            include_assistant_messages=self.include_assistant_messages_in_memory_context,
        )

        if recent_lines:
            sections.append(
                "Recent prior conversation messages:\n" + "\n".join(recent_lines)
            )

        if not sections:
            return ""

        return (
            "Conversation memory context:\n"
            "Use this only for current-session context and user-stated facts. "
            "Do not let prior assistant answers override current retrieved document evidence.\n\n"
            + "\n\n".join(sections)
        ).strip()

    # Backward-compatible alias.
    def _build_memory_context_text(
        self,
        *,
        chat_session: ChatSession,
    ) -> str:
        return self.build_memory_context_text(chat_session=chat_session)

    def render_recent_prior_messages(
        self,
        *,
        messages: List[Any],
        limit: int,
        include_assistant_messages: bool,
    ) -> List[str]:
        if limit <= 0:
            return []

        recent_messages = messages[-limit:]

        rendered_messages: List[str] = []

        for item in recent_messages:
            if not isinstance(item, dict):
                continue

            role = str(item.get("role") or "unknown").strip().lower()

            if role not in {"user", "assistant"}:
                continue

            if role == "assistant" and not include_assistant_messages:
                continue

            content = self.clean_memory_content(item.get("content"))

            if not content:
                continue

            if role == "user" and self.is_low_value_memory_user_message(content):
                continue

            if role == "assistant" and self.is_bad_assistant_memory_content(content):
                continue

            rendered_messages.append(f"{role}: {content}")

        return rendered_messages

    # -------------------------------------------------------------------------
    # Fact and topic extraction
    # -------------------------------------------------------------------------

    def extract_user_facts_from_messages(self, messages: List[Any]) -> Dict[str, Any]:
        """
        Extract stable facts from USER messages only.

        Latest fact wins.
        This intentionally avoids extracting facts from assistant answers.
        """

        facts: Dict[str, Any] = {}

        for item in messages:
            if not isinstance(item, dict):
                continue

            role = str(item.get("role") or "").strip().lower()

            if role != "user":
                continue

            content = str(item.get("content") or "").strip()

            if not content:
                continue

            created_at = str(item.get("created_at") or "").strip() or self._utc_iso()
            source_request_id = item.get("request_id")

            extracted = self.extract_user_facts_from_text(
                text=content,
                source_request_id=str(source_request_id) if source_request_id else None,
                updated_at=created_at,
            )

            facts.update(extracted)

        return facts

    def extract_user_facts_from_text(
        self,
        *,
        text: str,
        source_request_id: Optional[str],
        updated_at: str,
    ) -> Dict[str, Any]:
        facts: Dict[str, Any] = {}

        text_value = str(text or "").strip()

        if not text_value:
            return facts

        # Handles:
        #   My name is Robert
        #   myname is Robert
        #   Hi my name is Robert
        name_match = re.search(
            r"\bmy\s*name\s+is\s+([A-Za-z][A-Za-z'\-]{1,40})\b",
            text_value,
            flags=re.IGNORECASE,
        )

        if name_match:
            user_name = self.clean_fact_value(name_match.group(1), max_chars=40)

            if user_name:
                facts["user_name"] = {
                    "value": user_name,
                    "source": "user_message",
                    "source_request_id": source_request_id,
                    "updated_at": updated_at,
                }

        # Handles:
        #   I work in Bag Fab
        #   I work on Overwrap
        #   I work at Line 7
        work_match = re.search(
            r"\bi\s+work\s+(?:in|on|at)\s+([A-Za-z0-9][A-Za-z0-9 &/_\-.]{1,80})",
            text_value,
            flags=re.IGNORECASE,
        )

        if work_match:
            work_area = self.clean_fact_value(work_match.group(1), max_chars=80)

            if work_area:
                facts["work_area"] = {
                    "value": work_area,
                    "source": "user_message",
                    "source_request_id": source_request_id,
                    "updated_at": updated_at,
                }

        return facts

    def extract_recent_user_topics_from_messages(
        self,
        *,
        messages: List[Any],
        limit: int,
    ) -> List[Dict[str, Any]]:
        topics: List[Dict[str, Any]] = []

        for item in messages:
            if not isinstance(item, dict):
                continue

            role = str(item.get("role") or "").strip().lower()

            if role != "user":
                continue

            content = str(item.get("content") or "").strip()

            if not content:
                continue

            if self.is_low_value_memory_user_message(content):
                continue

            clean_content = self.clean_memory_content(content)

            if not clean_content:
                continue

            topics.append(
                {
                    "text": self.preview_text(clean_content, 180),
                    "request_id": item.get("request_id"),
                    "created_at": item.get("created_at"),
                }
            )

        if limit > 0 and len(topics) > limit:
            topics = topics[-limit:]

        return topics

    def render_fact_lines(self, facts: Any) -> List[str]:
        if not isinstance(facts, dict):
            return []

        lines: List[str] = []

        user_name = self.get_fact_value(facts, "user_name")
        work_area = self.get_fact_value(facts, "work_area")

        if user_name:
            lines.append(f"- User name: {user_name}")

        if work_area:
            lines.append(f"- User work area: {work_area}")

        return lines

    @staticmethod
    def get_fact_value(facts: Dict[str, Any], key: str) -> str:
        value = facts.get(key)

        if isinstance(value, dict):
            return str(value.get("value") or "").strip()

        return str(value or "").strip()

    def build_summary_text(self, summary: Dict[str, Any]) -> str:
        """
        Build clean human-readable summary text.

        Important:
            Do not include raw debug IDs, qanda IDs, request IDs, document IDs,
            complete_document_id values, or assistant answer previews here.

            Structured fields like active_document can still store IDs, but
            summary_text should stay clean and safe for embedding/prompt memory.
        """

        parts: List[str] = []

        facts = summary.get("facts") if isinstance(summary, dict) else {}

        if isinstance(facts, dict):
            user_name = self.get_fact_value(facts, "user_name")
            work_area = self.get_fact_value(facts, "work_area")

            if user_name:
                parts.append(f"The user's name is {user_name}.")

            if work_area:
                parts.append(f"The user works in {work_area}.")

        current_topic = str(summary.get("current_topic") or "").strip()

        if current_topic:
            clean_topic = self.clean_memory_content(current_topic)

            if clean_topic and not self.is_bad_assistant_memory_content(clean_topic):
                parts.append(f"The current topic is: {clean_topic}.")

        recent_user_topics = summary.get("recent_user_topics")

        if isinstance(recent_user_topics, list) and recent_user_topics:
            topic_texts: List[str] = []

            for item in recent_user_topics[-3:]:
                if not isinstance(item, dict):
                    continue

                text = str(item.get("text") or "").strip()

                if not text:
                    continue

                clean_text = self.clean_memory_content(text)

                if not clean_text:
                    continue

                if self.is_bad_assistant_memory_content(clean_text):
                    continue

                topic_texts.append(clean_text)

            if topic_texts:
                parts.append(
                    "Recent user topics include: " + "; ".join(topic_texts) + "."
                )

        active_document = summary.get("active_document")

        if isinstance(active_document, dict):
            document_name = str(active_document.get("document_name") or "").strip()

            # Keep the document name only. Do NOT include complete_document_id in
            # summary_text because it can look like leaked prompt/debug text and
            # it is already stored safely in active_document.
            if document_name:
                parts.append(f"The active selected document is {document_name}.")

        summary_text = " ".join(parts).strip()

        # Final guardrails against known bad/prompt leakage.
        blocked_fragments = [
            "=251)",
            "--- USER MESSAGE START ---",
            "--- USER MESSAGE END ---",
            "--- ASSISTANT MESSAGE START ---",
            "--- ASSISTANT MESSAGE END ---",
            "RAW MODEL OUTPUT START",
            "RAW MODEL OUTPUT END",
            "RETRIEVED DOCUMENT CONTEXT:",
            "FINAL ANSWER:",
        ]

        for fragment in blocked_fragments:
            summary_text = summary_text.replace(fragment, "")

        summary_text = re.sub(r"\s+", " ", summary_text).strip()

        if len(summary_text) > self.summary_text_max_chars:
            summary_text = summary_text[: self.summary_text_max_chars - 3].rstrip() + "..."

        return summary_text

    # -------------------------------------------------------------------------
    # Cleaners / validators
    # -------------------------------------------------------------------------

    def clean_memory_content(self, value: Any) -> str:
        text_value = str(value or "").strip()

        if not text_value:
            return ""

        blocked_markers = [
            "--- USER MESSAGE START ---",
            "--- USER MESSAGE END ---",
            "--- ASSISTANT MESSAGE START ---",
            "--- ASSISTANT MESSAGE END ---",
            "RAW MODEL OUTPUT START",
            "RAW MODEL OUTPUT END",
            "RETRIEVED DOCUMENT CONTEXT:",
            "--- CONTEXT END ---",
            "MEMORY RECALL OVERRIDE:",
            "FINAL ANSWER:",
            "[AIModelsService]",
        ]

        for marker in blocked_markers:
            if marker in text_value:
                return ""

        text_value = text_value.replace("\x00", " ")
        text_value = re.sub(r"\s+", " ", text_value).strip()

        return self.preview_text(text_value, self.memory_preview_chars)

    @staticmethod
    def _clean_raw_message_content(value: Any) -> str:
        text_value = str(value or "")
        text_value = text_value.replace("\x00", " ")
        return text_value

    @staticmethod
    def clean_fact_value(value: str, max_chars: int = 80) -> str:
        text_value = str(value or "").strip()
        text_value = re.sub(r"\s+", " ", text_value)
        text_value = text_value.strip(" .,!?:;\"'`<>")
        text_value = text_value[:max_chars].strip(" .,!?:;\"'`<>")
        return text_value

    @staticmethod
    def is_bad_assistant_memory_content(content: str) -> bool:
        text_value = str(content or "").strip()

        if not text_value:
            return True

        compact = text_value.strip()

        # Bad leakage seen in your test data: "=251)"
        if len(compact) <= 12 and re.search(r"[=)#]", compact):
            return True

        blocked_fragments = [
            "I cannot determine your name from the provided context",
            "The provided context does not contain information about your name",
            "To find your name, you may need to check other records",
            "--- USER MESSAGE START ---",
            "--- ASSISTANT MESSAGE START ---",
        ]

        lowered = compact.lower()

        for fragment in blocked_fragments:
            if fragment.lower() in lowered:
                return True

        return False

    @staticmethod
    def is_low_value_memory_user_message(content: str) -> bool:
        """
        These are valid user messages, but they should not be stored as topic context.
        Facts are extracted separately before this filter is used.
        """

        text_value = str(content or "").strip().lower()

        if not text_value:
            return True

        text_value = re.sub(r"\s+", " ", text_value)

        low_value_patterns = [
            r"^what'?s my name\??$",
            r"^what is my name\??$",
            r"^do you know my name\??$",
            r"^where do i work\??$",
            r"^what area do i work in\??$",
            r"\bmy\s*name\s+is\b",
            r"\bi\s+work\s+(in|on|at)\b",
        ]

        for pattern in low_value_patterns:
            if re.search(pattern, text_value, flags=re.IGNORECASE):
                return True

        return False

    # -------------------------------------------------------------------------
    # Summary JSON helpers
    # -------------------------------------------------------------------------

    def _empty_summary(self, *, now: Optional[str] = None) -> Dict[str, Any]:
        timestamp = now or self._utc_iso()

        return {
            "schema_version": self.SUMMARY_SCHEMA_VERSION,
            "facts": {},
            "current_topic": None,
            "recent_user_topics": [],
            "active_document": None,
            "summary_text": "",
            "last_request_id": None,
            "last_qanda_id": None,
            "last_intent": None,
            "updated_at": timestamp,
        }

    def _safe_summary_dict(self, value: Any) -> Dict[str, Any]:
        """
        Safely parse conversation_summary.

        Current DB/model compatibility:
            conversation_summary is mapped as a list-like mutable JSON field.

        Clean format:

            [
                {
                    "schema_version": 2,
                    "facts": {...},
                    "summary_text": "..."
                }
            ]

        Important:
            SQLAlchemy may return this field as MutableList instead of a plain list.
            This method therefore handles:
                - dict
                - JSON string
                - list
                - tuple
                - SQLAlchemy MutableList / other list-like values

        Legacy polluted format was also a list, but with many Q&A preview dicts.
        If legacy format is detected, do not trust answer_preview/question history.
        """

        if value is None:
            return self._empty_summary()

        # ------------------------------------------------------------
        # Direct dict support.
        # ------------------------------------------------------------

        if isinstance(value, dict):
            schema_version = value.get("schema_version")

            if schema_version == self.SUMMARY_SCHEMA_VERSION:
                summary = dict(value)

                summary.setdefault("facts", {})
                summary.setdefault("current_topic", None)
                summary.setdefault("recent_user_topics", [])
                summary.setdefault("active_document", None)
                summary.setdefault("summary_text", "")
                summary.setdefault("last_request_id", None)
                summary.setdefault("last_qanda_id", None)
                summary.setdefault("last_intent", None)
                summary.setdefault("updated_at", self._utc_iso())

                return summary

            summary = self._empty_summary()

            facts = value.get("facts")

            if isinstance(facts, dict):
                summary["facts"] = facts

            summary_text = str(value.get("summary_text") or "").strip()

            if summary_text:
                summary["summary_text"] = self.preview_text(
                    summary_text,
                    self.summary_text_max_chars,
                )

            return summary

        # ------------------------------------------------------------
        # JSON string support.
        # ------------------------------------------------------------

        if isinstance(value, str):
            text = value.strip()

            if not text:
                return self._empty_summary()

            try:
                parsed = json.loads(text)
            except Exception:
                return self._empty_summary()

            return self._safe_summary_dict(parsed)

        # ------------------------------------------------------------
        # List-like support.
        #
        # This intentionally handles SQLAlchemy MutableList too.
        # Do not rely only on isinstance(value, list), because your logs show:
        #
        #     raw_summary_type=MutableList
        #
        # ------------------------------------------------------------

        items: Optional[List[Any]] = None

        if isinstance(value, list):
            items = list(value)

        elif isinstance(value, tuple):
            items = list(value)

        else:
            try:
                # Defensive fallback for SQLAlchemy MutableList or other
                # list-like mutable JSON containers.
                if hasattr(value, "__iter__") and hasattr(value, "__len__"):
                    items = list(value)
            except Exception:
                items = None

        if items is not None:
            if len(items) == 1 and isinstance(items[0], dict):
                first_item = items[0]

                if first_item.get("schema_version") == self.SUMMARY_SCHEMA_VERSION:
                    return self._safe_summary_dict(first_item)

                # One dict but not schema v2. Treat as partial summary, not legacy Q&A.
                if "facts" in first_item or "summary_text" in first_item:
                    return self._safe_summary_dict(first_item)

            # Legacy polluted rolling Q&A format.
            summary = self._empty_summary()
            summary["legacy_summary_discarded"] = True
            summary["legacy_summary_items_count"] = len(items)
            return summary

        return self._empty_summary()

    @staticmethod
    def _safe_list(value: Any) -> List[Any]:
        if value is None:
            return []

        if isinstance(value, list):
            return list(value)

        if isinstance(value, tuple):
            return list(value)

        if isinstance(value, str):
            text = value.strip()

            if not text:
                return []

            try:
                parsed = json.loads(text)
            except Exception:
                return []

            if isinstance(parsed, list):
                return parsed

            return []

        return []

    # -------------------------------------------------------------------------
    # Document scope / intent serialization
    # -------------------------------------------------------------------------

    @staticmethod
    def normalize_document_scope(
        document_scope: Optional[Dict[str, Any]],
    ) -> Optional[Dict[str, Any]]:
        if not document_scope or not isinstance(document_scope, dict):
            return None

        enabled = document_scope.get("enabled", True)

        if enabled is False:
            return None

        scope_type = (
            document_scope.get("scope_type")
            or document_scope.get("scopeType")
            or "complete_document"
        )

        scope_type = str(scope_type or "").strip() or "complete_document"

        if scope_type != "complete_document":
            return None

        complete_document_id = (
            document_scope.get("complete_document_id")
            or document_scope.get("completed_document_id")
            or document_scope.get("completeDocumentId")
            or document_scope.get("completeDocumentID")
        )

        complete_document_id = ChatSessionMemoryService._coerce_int_or_none(
            complete_document_id
        )

        if complete_document_id is None:
            return None

        document_id = (
            document_scope.get("document_id")
            or document_scope.get("documentId")
        )

        document_id = ChatSessionMemoryService._coerce_int_or_none(document_id)

        document_name = (
            document_scope.get("document_name")
            or document_scope.get("documentName")
            or document_scope.get("name")
            or document_scope.get("title")
            or f"Document #{complete_document_id}"
        )

        document_name = (
            str(document_name or "").strip()
            or f"Document #{complete_document_id}"
        )

        return {
            "enabled": True,
            "scope_type": "complete_document",
            "document_id": document_id,
            "complete_document_id": complete_document_id,
            "document_name": document_name,
        }

    # Backward-compatible alias.
    @staticmethod
    def _normalize_document_scope(
        document_scope: Optional[Dict[str, Any]],
    ) -> Optional[Dict[str, Any]]:
        return ChatSessionMemoryService.normalize_document_scope(document_scope)

    @staticmethod
    def intent_decision_to_dict(intent_decision: Optional[Any]) -> Optional[Dict[str, Any]]:
        if intent_decision is None:
            return None

        intent = getattr(intent_decision, "intent", None)
        intent_value = getattr(intent, "value", intent)

        return {
            "intent": str(intent_value or ""),
            "confidence": float(getattr(intent_decision, "confidence", 0.0) or 0.0),
            "needs_current_session_memory": bool(
                getattr(intent_decision, "needs_current_session_memory", False)
            ),
            "needs_semantic_chat_recall": bool(
                getattr(intent_decision, "needs_semantic_chat_recall", False)
            ),
            "needs_document_scope": bool(
                getattr(intent_decision, "needs_document_scope", False)
            ),
            "rewritten_question": str(
                getattr(intent_decision, "rewritten_question", "") or ""
            ),
            "reason": str(getattr(intent_decision, "reason", "") or ""),
        }

    # Backward-compatible alias.
    @staticmethod
    def _intent_decision_to_dict(
        intent_decision: Optional[Any],
    ) -> Optional[Dict[str, Any]]:
        return ChatSessionMemoryService.intent_decision_to_dict(intent_decision)

    # -------------------------------------------------------------------------
    # Touch / utility
    # -------------------------------------------------------------------------

    @staticmethod
    def touch_chat_session(chat_session: ChatSession) -> None:
        chat_session.last_interaction = ChatSessionMemoryService._utc_iso()

    # Backward-compatible alias.
    @staticmethod
    def _touch_chat_session(chat_session: ChatSession) -> None:
        ChatSessionMemoryService.touch_chat_session(chat_session)

    @staticmethod
    def preview_text(value: str, max_chars: int) -> str:
        text = str(value or "").strip()

        safe_max = int(max_chars or 500)

        if safe_max <= 3:
            safe_max = 500

        if len(text) <= safe_max:
            return text

        return text[: safe_max - 3].rstrip() + "..."

    # Backward-compatible alias.
    @staticmethod
    def _preview_text(value: str, max_chars: int) -> str:
        return ChatSessionMemoryService.preview_text(value, max_chars)

    @staticmethod
    def _utc_iso() -> str:
        return datetime.utcnow().isoformat()

    @staticmethod
    def _coerce_uuid_or_none(value: Any) -> Optional[UUID]:
        if value is None:
            return None

        if isinstance(value, UUID):
            return value

        try:
            return UUID(str(value))
        except (TypeError, ValueError, AttributeError):
            return None

    @staticmethod
    def _coerce_int_or_none(value: Any) -> Optional[int]:
        if value is None:
            return None

        if isinstance(value, bool):
            return None

        try:
            text = str(value).strip()

            if not text:
                return None

            return int(text)

        except (TypeError, ValueError):
            return None