"""
factories.py
------------
Factory helpers for creating search-related ORM objects.
Keeps object creation logic separate from the models.
"""

from datetime import datetime
from sqlalchemy.orm import Session
from .models import SearchSession, SearchQuery


def create_search_session(db: Session, user_id: str, context_data: dict = None) -> SearchSession:
    """
    Create and persist a new SearchSession.
    """
    session = SearchSession(
        user_id=user_id,
        context_data=context_data or {},
        started_at=datetime.utcnow(),
        is_active=True,
    )
    db.add(session)
    db.commit()
    db.refresh(session)
    return session


def create_search_query(db: Session, session_id: int, query_text: str,
                        intent: str = None, entities: dict = None) -> SearchQuery:
    """
    Create and persist a new SearchQuery for a given session.
    """
    query = SearchQuery(
        session_id=session_id,
        query_text=query_text,
        intent=intent,
        extracted_entities=entities or {},
        created_at=datetime.utcnow(),
    )
    db.add(query)
    db.commit()
    db.refresh(query)
    return query
