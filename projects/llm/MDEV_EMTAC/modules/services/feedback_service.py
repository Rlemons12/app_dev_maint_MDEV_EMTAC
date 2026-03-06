# modules/services/feedback_service.py

from typing import Optional
from sqlalchemy.orm import Session
from datetime import datetime

from modules.emtacdb.emtacdb_fts import QandA
from modules.configuration.log_config import (
    debug_id,
    warning_id,
)


class FeedbackService:
    """
    Pure domain service for Q&A feedback updates.

    HARD RULES:
        - Never open sessions
        - Never commit
        - Never rollback
        - Orchestrator owns transaction
    """

    def update_feedback(
        self,
        *,
        session: Session,
        user_id: str,
        question: str,
        answer: str,
        rating: Optional[int],
        comment: Optional[str],
        request_id: str = None,
    ) -> bool:

        if not session:
            return False

        # --------------------------------------------------
        # If model has helper method → use it
        # --------------------------------------------------
        if hasattr(QandA, "update_or_create_feedback"):
            return QandA.update_or_create_feedback(
                user_id=user_id,
                question=question,
                answer=answer,
                rating=rating,
                comment=comment,
                session=session,
            )

        # --------------------------------------------------
        # Fallback logic (legacy parity)
        # --------------------------------------------------
        last_entry = (
            session.query(QandA)
            .order_by(QandA.id.desc())
            .first()
        )

        if (
            last_entry
            and last_entry.rating is None
            and last_entry.comment is None
        ):
            last_entry.rating = rating
            last_entry.comment = comment
            debug_id("Updated last Q&A entry with feedback", request_id)
            return True

        # Otherwise create new feedback entry
        new_entry = QandA(
            user_id=user_id,
            question=question,
            answer=answer,
            rating=rating,
            comment=comment,
            timestamp=datetime.now().isoformat(),
        )

        session.add(new_entry)
        debug_id("Created new Q&A feedback entry", request_id)

        return True