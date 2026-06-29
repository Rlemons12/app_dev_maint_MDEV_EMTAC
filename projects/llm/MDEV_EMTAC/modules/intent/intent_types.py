# modules/intent/intent_types.py

from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import Enum
from typing import Any, Dict


class ChatIntent(str, Enum):
    NEW_TOPIC = "NEW_TOPIC"
    FOLLOW_UP_CURRENT_SESSION = "FOLLOW_UP_CURRENT_SESSION"
    RECALL_PRIOR_CONVERSATION = "RECALL_PRIOR_CONVERSATION"
    DOCUMENT_SCOPED_FOLLOW_UP = "DOCUMENT_SCOPED_FOLLOW_UP"
    CLARIFICATION = "CLARIFICATION"
    PERSONAL_MEMORY_UPDATE = "PERSONAL_MEMORY_UPDATE"


@dataclass
class ChatIntentDecision:
    intent: ChatIntent
    confidence: float
    needs_current_session_memory: bool
    needs_semantic_chat_recall: bool
    needs_document_scope: bool
    rewritten_question: str
    reason: str = ""

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data["intent"] = self.intent.value
        return data

    @classmethod
    def fallback_new_topic(cls, question: str) -> "ChatIntentDecision":
        return cls(
            intent=ChatIntent.NEW_TOPIC,
            confidence=0.0,
            needs_current_session_memory=False,
            needs_semantic_chat_recall=False,
            needs_document_scope=False,
            rewritten_question=question or "",
            reason="Fallback decision.",
        )

    @classmethod
    def personal_memory_update(
        cls,
        question: str,
        *,
        confidence: float = 1.0,
        reason: str = "Personal memory update.",
    ) -> "ChatIntentDecision":
        return cls(
            intent=ChatIntent.PERSONAL_MEMORY_UPDATE,
            confidence=confidence,
            needs_current_session_memory=True,
            needs_semantic_chat_recall=False,
            needs_document_scope=False,
            rewritten_question=question or "",
            reason=reason,
        )

    @classmethod
    def follow_up_current_session(
        cls,
        question: str,
        *,
        confidence: float = 1.0,
        reason: str = "Current-session follow-up.",
    ) -> "ChatIntentDecision":
        return cls(
            intent=ChatIntent.FOLLOW_UP_CURRENT_SESSION,
            confidence=confidence,
            needs_current_session_memory=True,
            needs_semantic_chat_recall=False,
            needs_document_scope=False,
            rewritten_question=question or "",
            reason=reason,
        )

    @classmethod
    def recall_prior_conversation(
        cls,
        question: str,
        *,
        confidence: float = 1.0,
        reason: str = "Prior conversation recall.",
    ) -> "ChatIntentDecision":
        return cls(
            intent=ChatIntent.RECALL_PRIOR_CONVERSATION,
            confidence=confidence,
            needs_current_session_memory=True,
            needs_semantic_chat_recall=True,
            needs_document_scope=False,
            rewritten_question=question or "",
            reason=reason,
        )

    @classmethod
    def document_scoped_follow_up(
        cls,
        question: str,
        *,
        confidence: float = 1.0,
        reason: str = "Document-scoped follow-up.",
    ) -> "ChatIntentDecision":
        return cls(
            intent=ChatIntent.DOCUMENT_SCOPED_FOLLOW_UP,
            confidence=confidence,
            needs_current_session_memory=True,
            needs_semantic_chat_recall=False,
            needs_document_scope=True,
            rewritten_question=question or "",
            reason=reason,
        )

    @classmethod
    def clarification(
        cls,
        question: str,
        *,
        confidence: float = 1.0,
        reason: str = "Clarification.",
    ) -> "ChatIntentDecision":
        return cls(
            intent=ChatIntent.CLARIFICATION,
            confidence=confidence,
            needs_current_session_memory=True,
            needs_semantic_chat_recall=False,
            needs_document_scope=False,
            rewritten_question=question or "",
            reason=reason,
        )