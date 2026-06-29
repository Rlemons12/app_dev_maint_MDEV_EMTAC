# modules/coordinators/chat_intent_coordinator.py

from __future__ import annotations

from typing import Any, Dict, Optional

from modules.intent.intent_types import ChatIntentDecision
from modules.orchestrators.chat_intent_orchestrator import ChatIntentOrchestrator
from modules.services.intent.chat_intent_classifier_service import ChatIntentClassifierService
from modules.services.intent.chat_intent_prompt_service import ChatIntentPromptService


class ChatIntentCoordinator:
    """
    Public coordinator for chat intent classification.

    IMPORTANT:
    This coordinator stays lightweight.

    It does NOT default to AIStewardManagerService.
    It does NOT run RAG/search.
    It does NOT call the main chat AI pathway.

    Flow:
        question + chat_session + document_scope
            -> lightweight local DistilBERT classifier
            -> ChatIntentDecision
    """

    def __init__(
        self,
        *,
        ai_service: Optional[Any] = None,
        classifier_service: Optional[ChatIntentClassifierService] = None,
        prompt_service: Optional[ChatIntentPromptService] = None,
        orchestrator: Optional[ChatIntentOrchestrator] = None,
    ):
        """
        ai_service is accepted only for backward compatibility.

        The production pathway should use ChatIntentClassifierService,
        which loads the local DistilBERT model from MODELS_DISTILBERT_INTENT.
        """

        if orchestrator is not None:
            self.orchestrator = orchestrator
            return

        resolved_classifier_service = classifier_service or ChatIntentClassifierService(
            ai_service=ai_service,
        )

        resolved_prompt_service = prompt_service or ChatIntentPromptService()

        self.orchestrator = ChatIntentOrchestrator(
            classifier_service=resolved_classifier_service,
            prompt_service=resolved_prompt_service,
        )

    def classify_question(
        self,
        *,
        question: str,
        chat_session: Optional[Any] = None,
        document_scope: Optional[Dict[str, Any]] = None,
        request_id: Optional[str] = None,
    ) -> ChatIntentDecision:
        return self.orchestrator.classify_question(
            question=question,
            chat_session=chat_session,
            document_scope=document_scope,
            request_id=request_id,
        )