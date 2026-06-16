from __future__ import annotations

import logging
from typing import Any, Dict, Optional

from sqlalchemy.exc import SQLAlchemyError

from modules.configuration.config_env import get_db_config
from modules.services.feedback_service import FeedbackService


logger = logging.getLogger(__name__)


class FeedbackOrchestrator:
    """
    Orchestrator layer for QandA feedback.

    Owns:
        - database session lifecycle
        - commit / rollback

    Does not:
        - create new QandA rows
    """

    def __init__(
        self,
        *,
        db_config=None,
        service: Optional[FeedbackService] = None,
    ):
        self.db_config = db_config or get_db_config()
        self.service = service or FeedbackService()

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
        session = None

        try:
            logger.info(
                "[FeedbackOrchestrator] process_feedback started "
                "user_id=%s request_id=%s has_question=%s has_answer=%s",
                user_id,
                request_id,
                bool(question),
                bool(answer),
            )

            session = self.db_config.get_main_session()

            result = self.service.update_feedback(
                session=session,
                user_id=user_id,
                question=question,
                answer=answer,
                rating=rating,
                comment=comment,
                request_id=request_id,
                qa_id=qa_id,
            )

            if result.get("status") != "success":
                session.rollback()
                return result

            session.commit()

            logger.info(
                "[FeedbackOrchestrator] process_feedback committed qanda_id=%s matched_by=%s",
                result.get("qanda_id"),
                result.get("matched_by"),
            )

            return result

        except SQLAlchemyError as exc:
            if session is not None:
                session.rollback()

            logger.error(
                "[FeedbackOrchestrator] Database error while saving feedback: %s",
                exc,
                exc_info=True,
            )

            return {
                "status": "error",
                "message": "Database error while saving feedback.",
                "qanda_id": None,
                "matched_by": None,
            }

        except Exception as exc:
            if session is not None:
                session.rollback()

            logger.error(
                "[FeedbackOrchestrator] Unexpected error while saving feedback: %s",
                exc,
                exc_info=True,
            )

            return {
                "status": "error",
                "message": "Unexpected error while saving feedback.",
                "qanda_id": None,
                "matched_by": None,
            }

        finally:
            if session is not None:
                session.close()