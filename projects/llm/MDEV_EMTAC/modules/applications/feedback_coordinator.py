# modules/application/feedback_coordinator.py

from typing import Dict, Any

from modules.configuration.log_config import (
    with_request_id,
    info_id,
)
from modules.orchestrators.feedback_orchestrator import (
    FeedbackOrchestrator,
)


class FeedbackCoordinator:
    """
    Application-layer coordinator for feedback updates.

    Responsibilities:
        - Input validation
        - Delegation to orchestrator
        - Response normalization
    """

    def __init__(self):
        self.orchestrator = FeedbackOrchestrator()

    @with_request_id
    def process_feedback(
        self,
        *,
        user_id: str,
        question: str,
        answer: str,
        rating: int,
        comment: str,
        request_id: str = None,
    ) -> Dict[str, Any]:

        info_id("FeedbackCoordinator.process_feedback called", request_id)

        if not question:
            return {
                "status": "invalid_input",
                "message": "Question is required",
            }

        return self.orchestrator.handle_feedback(
            user_id=user_id,
            question=question,
            answer=answer,
            rating=rating,
            comment=comment,
            request_id=request_id,
        )