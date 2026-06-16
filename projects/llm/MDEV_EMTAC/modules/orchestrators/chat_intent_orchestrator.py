# modules/orchestrators/chat_intent_orchestrator.py

from __future__ import annotations

from typing import Any, Dict, List, Optional

from modules.configuration.log_config import debug_id, error_id
from modules.intent.intent_types import ChatIntentDecision
from modules.services.intent.chat_intent_classifier_service import ChatIntentClassifierService
from modules.services.intent.chat_intent_prompt_service import ChatIntentPromptService


class ChatIntentOrchestrator:
    """
    Orchestrates intent classification before the main RAG/chat pathway.

    This class:
    - does not own DB sessions
    - does not call RAG
    - does not call AIStewardManagerService
    - does not call AIModelsService directly
    - only reads already-loaded ChatSession data
    - returns a ChatIntentDecision

    The actual classifier should be the local DistilBERT classifier.
    """

    def __init__(
        self,
        *,
        classifier_service: ChatIntentClassifierService,
        prompt_service: Optional[ChatIntentPromptService] = None,
    ):
        self.classifier_service = classifier_service
        self.prompt_service = prompt_service or ChatIntentPromptService()

    def classify_question(
        self,
        *,
        question: str,
        chat_session: Optional[Any] = None,
        document_scope: Optional[Dict[str, Any]] = None,
        request_id: Optional[str] = None,
    ) -> ChatIntentDecision:
        normalized_question = (question or "").strip()

        if not normalized_question:
            return ChatIntentDecision.fallback_new_topic("")

        try:
            recent_messages = self._safe_list(
                getattr(chat_session, "session_data", None)
                if chat_session is not None
                else None
            )

            conversation_summary = self._safe_list(
                getattr(chat_session, "conversation_summary", None)
                if chat_session is not None
                else None
            )

            decision = self.classifier_service.classify(
                question=normalized_question,
                recent_messages=recent_messages,
                conversation_summary=conversation_summary,
                document_scope=document_scope,
                request_id=request_id,
            )

            debug_id(
                f"[ChatIntentOrchestrator] Intent decision "
                f"intent={decision.intent.value} "
                f"confidence={decision.confidence:.2f} "
                f"current_memory={decision.needs_current_session_memory} "
                f"semantic_recall={decision.needs_semantic_chat_recall} "
                f"document_scope={decision.needs_document_scope} "
                f"rewritten_question={decision.rewritten_question!r}",
                request_id,
            )

            return decision

        except Exception as exc:
            error_id(
                f"[ChatIntentOrchestrator] Intent orchestration failed: {exc}",
                request_id,
                exc_info=True,
            )
            return ChatIntentDecision.fallback_new_topic(normalized_question)

    @staticmethod
    def _safe_list(value: Any) -> List[Any]:
        if isinstance(value, list):
            return list(value)

        if isinstance(value, dict):
            messages = value.get("messages")
            if isinstance(messages, list):
                return list(messages)

        return []