from __future__ import annotations

import logging
from typing import Any, Dict, Optional

from modules.orchestrators.feedback_orchestrator import FeedbackOrchestrator


logger = logging.getLogger(__name__)


class FeedbackCoordinator:
    """
    Coordinator layer for QandA feedback.

    The route calls this.
    The orchestrator owns transaction/session flow.
    The service updates the existing QandA row.
    """

    def __init__(
        self,
        *,
        orchestrator: Optional[FeedbackOrchestrator] = None,
    ):
        self.orchestrator = orchestrator or FeedbackOrchestrator()

    def process_feedback(
        self,
        *,
        user_id: Optional[Any],
        question: Optional[str],
        answer: Optional[str],
        rating: Optional[Any],
        comment: Optional[Any],
        request_id: Optional[str] = None,
        qa_id: Optional[Any] = None,
    ) -> Dict[str, Any]:
        logger.info(
            "[FeedbackCoordinator] process_feedback called "
            "user_id=%s request_id=%s has_question=%s has_answer=%s has_rating=%s has_comment=%s",
            user_id,
            request_id,
            bool(question),
            bool(answer),
            rating is not None,
            bool(comment),
        )

        return self.orchestrator.process_feedback(
            user_id=user_id,
            question=question,
            answer=answer,
            rating=rating,
            comment=comment,
            request_id=request_id,
            qa_id=qa_id,
        )