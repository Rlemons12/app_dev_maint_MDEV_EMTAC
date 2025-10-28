"""
feedback.py
-----------
Handles user feedback and learning from interactions.
Includes:
- FeedbackLearner class for DB-driven feedback learning
- Functional helpers for recording/querying feedback
"""

from typing import Dict, Any, Optional, List
from sqlalchemy.orm import Session

from modules.configuration.log_config import get_request_id, debug_id, info_id
from .models import UserFeedback, SearchQuery


class FeedbackLearner:
    """
    Learns from feedback and click patterns to improve search quality.
    Originally defined in nlp_search.py, now split into this module.
    """

    def __init__(self, session: Optional[Session] = None):
        self.session = session
        self.request_id = get_request_id()

    def record_feedback(
        self,
        query_id: int,
        rating: Optional[int] = None,
        comment: Optional[str] = None,
        user_id: Optional[str] = None,
    ) -> Optional[int]:
        """
        Store feedback for a given query.
        """
        if not self.session:
            debug_id("[FeedbackLearner] No session available", self.request_id)
            return None

        feedback = UserFeedback(
            query_id=query_id,
            rating=rating,
            comment=comment,
            user_id=user_id,
        )
        self.session.add(feedback)
        self.session.commit()
        info_id(f"[FeedbackLearner] Recorded feedback for query_id={query_id}", self.request_id)
        return feedback.id

    def learn_from_clicks(self, query_id: int) -> Dict[str, Any]:
        """
        Learn from click patterns on search results.
        """
        if not self.session:
            return {"status": "noop", "reason": "no session"}

        clicks = (
            self.session.query(SearchQuery)
            .filter(SearchQuery.id == query_id)
            .all()
        )
        debug_id(f"[FeedbackLearner] learn_from_clicks for query_id={query_id}", self.request_id)
        return {"status": "ok", "clicks": len(clicks)}

    def get_popular_patterns(self, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Retrieve top search patterns based on feedback/clicks.
        """
        if not self.session:
            return []

        rows = (
            self.session.query(SearchQuery.query_text)
            .order_by(SearchQuery.id.desc())
            .limit(limit)
            .all()
        )
        return [{"pattern": r.query_text} for r in rows]


# ---------------- Functional helpers ---------------- #

def record_feedback(
    session: Session,
    query_id: int,
    rating: Optional[int] = None,
    comment: Optional[str] = None,
    user_id: Optional[str] = None,
) -> Optional[int]:
    """
    Record feedback without instantiating FeedbackLearner.
    """
    learner = FeedbackLearner(session=session)
    return learner.record_feedback(query_id, rating, comment, user_id)


def get_feedback_for_query(session: Session, query_id: int) -> List[UserFeedback]:
    """
    Fetch all feedback entries for a given query.
    """
    return session.query(UserFeedback).filter(UserFeedback.query_id == query_id).all()


def average_rating_for_query(session: Session, query_id: int) -> Optional[float]:
    """
    Compute the average rating for a given query.
    """
    rows = session.query(UserFeedback.rating).filter(UserFeedback.query_id == query_id).all()
    ratings = [r[0] for r in rows if r[0] is not None]
    return sum(ratings) / len(ratings) if ratings else None
