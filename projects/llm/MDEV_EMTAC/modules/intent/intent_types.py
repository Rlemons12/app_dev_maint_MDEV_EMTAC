# modules/intent/intent_types.py

from __future__ import annotations

from enum import Enum
from typing import Any, Dict, Optional
from dataclasses import dataclass, asdict


class ChatIntent(str, Enum):
    NEW_TOPIC = "NEW_TOPIC"
    FOLLOW_UP_CURRENT_SESSION = "FOLLOW_UP_CURRENT_SESSION"
    RECALL_PRIOR_CONVERSATION = "RECALL_PRIOR_CONVERSATION"
    DOCUMENT_SCOPED_FOLLOW_UP = "DOCUMENT_SCOPED_FOLLOW_UP"
    CLARIFICATION = "CLARIFICATION"


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