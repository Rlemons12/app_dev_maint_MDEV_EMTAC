from __future__ import annotations

import logging
from typing import Any, Dict, Optional
from uuid import UUID

from sqlalchemy import desc
from sqlalchemy.orm import Session

from modules.emtacdb.emtacdb_fts import QandA


logger = logging.getLogger(__name__)


class FeedbackService:
    """
    Service layer for QandA feedback.

    Important:
        This service updates an existing qanda row.
        It does NOT create a new qanda row.

    Expected behavior:
        /chatbot/ask creates the QandA record.
        /chatbot/update_qanda updates rating/comment on that same record.
    """

    def update_feedback(
        self,
        *,
        session: Session,
        user_id: Optional[Any],
        question: Optional[str],
        answer: Optional[str],
        rating: Optional[Any],
        comment: Optional[Any],
        request_id: Optional[str] = None,
        qa_id: Optional[Any] = None,
    ) -> Dict[str, Any]:
        clean_user_id = self._normalize_optional_text(user_id)
        clean_question = self._normalize_optional_text(question)
        clean_answer = self._normalize_optional_text(answer)
        clean_rating = self._normalize_optional_text(rating)
        clean_comment = self._normalize_optional_text(comment)
        clean_request_id = self._normalize_optional_text(request_id)

        logger.info(
            "[FeedbackService] update_feedback called user_id=%s request_id=%s "
            "has_question=%s has_answer=%s has_rating=%s has_comment=%s qa_id=%s",
            clean_user_id,
            clean_request_id,
            bool(clean_question),
            bool(clean_answer),
            clean_rating is not None,
            clean_comment is not None,
            qa_id,
        )

        if clean_rating is None and clean_comment is None:
            return {
                "status": "invalid_input",
                "message": "Rating or comment is required.",
                "qanda_id": None,
                "matched_by": None,
            }

        qa_record = None
        matched_by = None

        # Best match: original /chatbot/ask request_id.
        if clean_request_id:
            qa_record = self._find_by_request_id(
                session=session,
                request_id=clean_request_id,
                user_id=clean_user_id,
            )
            matched_by = "request_id" if qa_record else None

        # Optional direct fallback if later you send qanda UUID from the UI.
        if qa_record is None and qa_id:
            qa_record = self._find_by_qa_id(
                session=session,
                qa_id=qa_id,
                user_id=clean_user_id,
            )
            matched_by = "qa_id" if qa_record else None

        # Last fallback: exact latest question/answer/user match.
        # This prevents creating duplicates if request_id is unavailable.
        if qa_record is None and clean_question and clean_answer:
            qa_record = self._find_latest_by_question_answer(
                session=session,
                user_id=clean_user_id,
                question=clean_question,
                answer=clean_answer,
            )
            matched_by = "question_answer" if qa_record else None

        if qa_record is None:
            logger.warning(
                "[FeedbackService] No existing QandA row found for feedback. "
                "user_id=%s request_id=%s question_present=%s answer_present=%s",
                clean_user_id,
                clean_request_id,
                bool(clean_question),
                bool(clean_answer),
            )

            return {
                "status": "not_found",
                "message": "No existing Q&A record was found to update.",
                "qanda_id": None,
                "matched_by": None,
            }

        qa_record.rating = clean_rating
        qa_record.comment = clean_comment

        logger.info(
            "[FeedbackService] Updated existing QandA feedback. "
            "qanda_id=%s user_id=%s request_id=%s matched_by=%s rating=%s has_comment=%s",
            qa_record.id,
            qa_record.user_id,
            qa_record.request_id,
            matched_by,
            clean_rating,
            bool(clean_comment),
        )

        return {
            "status": "success",
            "message": "Feedback saved.",
            "qanda_id": str(qa_record.id),
            "matched_by": matched_by,
            "request_id": qa_record.request_id,
            "user_id": qa_record.user_id,
        }

    def _find_by_request_id(
        self,
        *,
        session: Session,
        request_id: str,
        user_id: Optional[str] = None,
    ) -> Optional[QandA]:
        query = session.query(QandA).filter(QandA.request_id == request_id)

        if user_id:
            query = query.filter(QandA.user_id == str(user_id))

        return query.order_by(desc(QandA.timestamp)).first()

    def _find_by_qa_id(
        self,
        *,
        session: Session,
        qa_id: Any,
        user_id: Optional[str] = None,
    ) -> Optional[QandA]:
        normalized_qa_id = self._normalize_qa_id(qa_id)

        if normalized_qa_id is None:
            return None

        query = session.query(QandA).filter(QandA.id == normalized_qa_id)

        if user_id:
            query = query.filter(QandA.user_id == str(user_id))

        return query.first()

    def _find_latest_by_question_answer(
        self,
        *,
        session: Session,
        user_id: Optional[str],
        question: str,
        answer: str,
    ) -> Optional[QandA]:
        query = session.query(QandA).filter(
            QandA.question == question,
            QandA.answer == answer,
        )

        if user_id:
            query = query.filter(QandA.user_id == str(user_id))

        # Prefer the real /ask row if duplicates already exist.
        query = query.order_by(
            desc(QandA.request_id.isnot(None)),
            desc(QandA.raw_response.isnot(None)),
            desc(QandA.timestamp),
        )

        return query.first()

    def _normalize_optional_text(self, value: Optional[Any]) -> Optional[str]:
        if value is None:
            return None

        text_value = str(value).strip()

        if not text_value:
            return None

        if text_value.lower() in {"none", "null", "undefined"}:
            return None

        return text_value

    def _normalize_qa_id(self, qa_id: Any):
        if qa_id is None:
            return None

        try:
            return UUID(str(qa_id))
        except Exception:
            logger.warning("[FeedbackService] Invalid qa_id supplied: %s", qa_id)
            return None