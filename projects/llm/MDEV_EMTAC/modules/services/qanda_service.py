# modules/services/qanda_service.py

from __future__ import annotations

from typing import Optional, Dict, Any
from datetime import datetime

from sqlalchemy.orm import Session

from modules.configuration.log_config import (
    debug_id,
    warning_id,
    with_request_id,
    get_request_id,
)

from modules.emtacdb.emtacdb_fts import QandA


class QandAService:
    """
    Pure domain service for QandA persistence.

    HARD RULES:
    - NEVER open sessions
    - NEVER close sessions
    - NEVER commit
    - NEVER rollback
    - Orchestrator owns transactions

    Responsibilities:
    - Create QandA interaction records
    - Store intent classification metadata
    - Update feedback
    - Retrieve raw saved response seeds for payload loading
    - Lightweight QandA analytics
    """

    @with_request_id
    def create_interaction(
        self,
        session: Session,
        *,
        user_id: str,
        question: str,
        answer: str,
        request_id: Optional[str] = None,
        question_embedding: Optional[list] = None,
        answer_embedding: Optional[list] = None,
        processing_time_ms: Optional[int] = None,
        raw_response: Optional[Dict[str, Any]] = None,
        rating: Optional[str] = None,
        comment: Optional[str] = None,
        intent_type: Optional[str] = None,
        intent_confidence: Optional[float] = None,
        intent_reason: Optional[str] = None,
        intent_rewritten_question: Optional[str] = None,
        intent_needs_current_session_memory: bool = False,
        intent_needs_semantic_chat_recall: bool = False,
        intent_needs_document_scope: bool = False,
    ) -> QandA:
        """
        Create and attach a QandA record to the session.
        Does NOT commit.
        """

        resolved_request_id = request_id or get_request_id()
        timestamp = datetime.utcnow().isoformat()

        qa = QandA(
            user_id=user_id,
            question=question,
            answer=answer,
            timestamp=timestamp,
            rating=rating,
            comment=comment,
            request_id=resolved_request_id,
            raw_response=raw_response,
            processing_time_ms=processing_time_ms,
            intent_type=intent_type,
            intent_confidence=intent_confidence,
            intent_reason=intent_reason,
            intent_rewritten_question=intent_rewritten_question,
            intent_needs_current_session_memory=intent_needs_current_session_memory,
            intent_needs_semantic_chat_recall=intent_needs_semantic_chat_recall,
            intent_needs_document_scope=intent_needs_document_scope,
        )

        qa.question_length = len(question) if question else 0
        qa.answer_length = len(answer) if answer else 0

        if question_embedding is not None:
            qa.question_embedding = question_embedding

        if answer_embedding is not None:
            qa.answer_embedding = answer_embedding

        session.add(qa)

        debug_id(
            f"QandA interaction staged for persistence "
            f"(user={user_id}, req={resolved_request_id}, intent={intent_type})",
            resolved_request_id,
        )

        return qa

    @with_request_id
    def get_interaction_by_request_id(
        self,
        session: Session,
        *,
        request_id: str,
    ) -> Optional[QandA]:
        """
        Load the most recent QandA interaction by request_id.
        """

        resolved_request_id = request_id or get_request_id()

        if not resolved_request_id:
            warning_id(
                "QandA lookup skipped: missing request_id",
                resolved_request_id,
            )
            return None

        qa = (
            session.query(QandA)
            .filter(QandA.request_id == resolved_request_id)
            .order_by(QandA.timestamp.desc())
            .first()
        )

        if qa:
            debug_id(
                f"QandA interaction loaded for request_id={resolved_request_id} "
                f"(id={qa.id})",
                resolved_request_id,
            )
            return qa

        warning_id(
            f"No QandA interaction found for request_id={resolved_request_id}",
            resolved_request_id,
        )
        return None

    @with_request_id
    def get_raw_response_by_request_id(
        self,
        session: Session,
        *,
        request_id: str,
    ) -> Dict[str, Any]:
        """
        Retrieve raw_response from the QandA row for request_id.

        This raw_response is the seed used by the payload route to build:
            - documents
            - images
            - parts
            - drawings
        """

        resolved_request_id = request_id or get_request_id()

        qa = self.get_interaction_by_request_id(
            session=session,
            request_id=resolved_request_id,
        )

        if not qa:
            return {}

        raw_response = getattr(qa, "raw_response", None)

        if isinstance(raw_response, dict):
            debug_id(
                f"QandA raw_response loaded for request_id={resolved_request_id}",
                resolved_request_id,
            )
            return raw_response

        if raw_response is None:
            warning_id(
                f"QandA raw_response is empty for request_id={resolved_request_id}",
                resolved_request_id,
            )
            return {}

        warning_id(
            f"QandA raw_response is not a dict for request_id={resolved_request_id} "
            f"type={type(raw_response).__name__}",
            resolved_request_id,
        )

        return {}

    @with_request_id
    def update_feedback(
        self,
        session: Session,
        *,
        qa_id,
        rating: Optional[str] = None,
        comment: Optional[str] = None,
        request_id: Optional[str] = None,
    ) -> Optional[QandA]:
        """
        Update rating/comment on an existing QandA record.
        Does NOT commit.
        """

        resolved_request_id = request_id or get_request_id()

        qa = session.get(QandA, qa_id)

        if not qa:
            warning_id(
                f"QandA id={qa_id} not found for feedback update",
                resolved_request_id,
            )
            return None

        qa.rating = rating
        qa.comment = comment

        debug_id(
            f"QandA feedback updated (id={qa_id})",
            resolved_request_id,
        )

        return qa

    @with_request_id
    def update_feedback_by_request_id(
        self,
        session: Session,
        *,
        request_id: str,
        rating: Optional[str] = None,
        comment: Optional[str] = None,
    ) -> Optional[QandA]:
        """
        Update rating/comment using the original /chatbot/ask request_id.
        Does NOT commit.
        """

        qa = self.get_interaction_by_request_id(
            session=session,
            request_id=request_id,
        )

        if not qa:
            return None

        qa.rating = rating
        qa.comment = comment

        debug_id(
            f"QandA feedback updated by request_id={request_id} (id={qa.id})",
            request_id,
        )

        return qa

    def find_similar_questions(
        self,
        session: Session,
        *,
        query_embedding,
        user_id: Optional[str] = None,
        limit: int = 5,
        similarity_threshold: float = 0.8,
    ):
        """
        Wrapper around vector similarity search.

        Returns:
            list[(QandA, similarity_score)]
        """

        if query_embedding is None:
            return []

        query = (
            session.query(
                QandA,
                QandA.question_embedding.cosine_distance(query_embedding).label(
                    "distance"
                ),
            )
            .filter(
                QandA.question_embedding.is_not(None),
                QandA.question_embedding.cosine_distance(query_embedding)
                < (1 - similarity_threshold),
            )
        )

        if user_id:
            query = query.filter(QandA.user_id == user_id)

        results = query.order_by("distance").limit(limit).all()

        return [(qa, 1 - distance) for qa, distance in results]

    def get_user_analytics(
        self,
        session: Session,
        *,
        user_id: str,
    ) -> Dict[str, Any]:
        """
        Lightweight analytics wrapper.
        No raw SQL. Keeps logic in service layer.
        """

        qas = session.query(QandA).filter_by(user_id=user_id).all()

        if not qas:
            return {}

        total = len(qas)
        rated = [qa for qa in qas if qa.rating is not None]

        numeric_ratings = [
            int(qa.rating)
            for qa in rated
            if str(qa.rating).isdigit()
        ]

        avg_rating = (
            sum(numeric_ratings) / len(numeric_ratings)
            if numeric_ratings
            else 0
        )

        intent_counts: Dict[str, int] = {}

        for qa in qas:
            intent_type = getattr(qa, "intent_type", None) or "UNKNOWN"
            intent_counts[intent_type] = intent_counts.get(intent_type, 0) + 1

        return {
            "total_questions": total,
            "avg_question_length": (
                sum((qa.question_length or 0) for qa in qas) / total
            ),
            "avg_answer_length": (
                sum((qa.answer_length or 0) for qa in qas) / total
            ),
            "avg_processing_time_ms": (
                sum((qa.processing_time_ms or 0) for qa in qas) / total
            ),
            "rated_answers": len(rated),
            "avg_rating": avg_rating,
            "intent_counts": intent_counts,
        }