# modules/orchestrators/feedback_orchestrator.py

from typing import Dict, Any

from modules.orchestrators.base_orchestrator import BaseOrchestrator
from modules.services.feedback_service import FeedbackService
from modules.configuration.log_config import (
    with_request_id,
    info_id,
    warning_id,
    error_id,
)


class FeedbackOrchestrator(BaseOrchestrator):
    """
    Owns transaction boundary for feedback updates.
    """

    def __init__(self):
        super().__init__()
        self.feedback_service = FeedbackService()

    @with_request_id
    def handle_feedback(
        self,
        *,
        user_id: str,
        question: str,
        answer: str,
        rating: int,
        comment: str,
        request_id: str = None,
    ) -> Dict[str, Any]:

        try:
            with self.transaction() as session:

                success = self.feedback_service.update_feedback(
                    session=session,
                    user_id=user_id,
                    question=question,
                    answer=answer,
                    rating=rating,
                    comment=comment,
                    request_id=request_id,
                )

                if not success:
                    warning_id("Feedback update failed", request_id)
                    return {
                        "status": "failed",
                        "message": "Failed to update Q&A entry",
                    }

            info_id("Feedback updated successfully", request_id)

            return {
                "status": "success",
                "message": "Q&A updated successfully",
            }

        except Exception as e:
            error_id(
                f"FeedbackOrchestrator failure: {e}",
                request_id,
                exc_info=True,
            )

            return {
                "status": "error",
                "message": "An unexpected error occurred",
            }