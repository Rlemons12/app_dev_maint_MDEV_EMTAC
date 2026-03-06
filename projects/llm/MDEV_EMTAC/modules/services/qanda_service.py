# modules/services/qanda_service.py

from typing import Optional, Dict, Any
from datetime import datetime
from sqlalchemy.orm import Session

from modules.configuration.log_config import (
    debug_id,
    warning_id,
    error_id,
    with_request_id,
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
    """

    # ---------------------------------------------------------
    # CREATE INTERACTION
    # ---------------------------------------------------------

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
    ) -> QandA:
        """
        Create and attach a QandA record to the session.
        Does NOT commit.

        Returns:
            QandA instance (attached to session)
        """

        timestamp = datetime.utcnow().isoformat()

        qa = QandA(
            user_id=user_id,
            question=question,
            answer=answer,
            timestamp=timestamp,
            rating=rating,
            comment=comment,
        )

        # Optional metadata
        qa.request_id = request_id
        qa.raw_response = raw_response
        qa.processing_time_ms = processing_time_ms

        # Length metadata (defensive in case constructor changes)
        qa.question_length = len(question) if question else 0
        qa.answer_length = len(answer) if answer else 0

        # Embeddings (pgvector)
        if question_embedding is not None:
            qa.question_embedding = question_embedding

        if answer_embedding is not None:
            qa.answer_embedding = answer_embedding

        session.add(qa)

        debug_id(
            f"QandA interaction staged for persistence (user={user_id}, req={request_id})"
        )

        return qa

    # ---------------------------------------------------------
    # UPDATE FEEDBACK
    # ---------------------------------------------------------

    @with_request_id
    def update_feedback(
        self,
        session: Session,
        *,
        qa_id,
        rating: Optional[str] = None,
        comment: Optional[str] = None,
    ) -> Optional[QandA]:
        """
        Update rating/comment on an existing QandA record.
        Does NOT commit.
        """

        qa = session.get(QandA, qa_id)

        if not qa:
            warning_id(f"QandA id={qa_id} not found for feedback update")
            return None

        qa.rating = rating
        qa.comment = comment

        debug_id(f"QandA feedback updated (id={qa_id})")

        return qa

    # ---------------------------------------------------------
    # FIND SIMILAR QUESTIONS
    # ---------------------------------------------------------

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
        Returns list[(QandA, similarity_score)]
        """

        if query_embedding is None:
            return []

        query = session.query(
            QandA,
            QandA.question_embedding.cosine_distance(query_embedding).label("distance"),
        ).filter(
            QandA.question_embedding.is_not(None),
            QandA.question_embedding.cosine_distance(query_embedding)
            < (1 - similarity_threshold),
        )

        if user_id:
            query = query.filter(QandA.user_id == user_id)

        results = (
            query.order_by("distance")
            .limit(limit)
            .all()
        )

        return [(qa, 1 - distance) for qa, distance in results]

    # ---------------------------------------------------------
    # USER ANALYTICS
    # ---------------------------------------------------------

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

        return {
            "total_questions": total,
            "avg_question_length": sum(qa.question_length for qa in qas) / total,
            "avg_answer_length": sum(qa.answer_length for qa in qas) / total,
            "avg_processing_time_ms": (
                sum(qa.processing_time_ms or 0 for qa in qas) / total
            ),
            "rated_answers": len(rated),
            "avg_rating": (
                sum(int(qa.rating) for qa in rated if str(qa.rating).isdigit())
                / len(rated)
                if rated
                else 0
            ),
        }