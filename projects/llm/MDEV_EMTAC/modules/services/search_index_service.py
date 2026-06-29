# modules/services/search_index_service.py

from sqlalchemy import text
from sqlalchemy.orm import Session
from typing import Optional


class SearchIndexService:
    """
    Handles PostgreSQL full-text indexing.
    """

    def index_complete_document(
        self,
        session: Session,
        *,
        title: str,
        content: str,
    ) -> None:

        sql = text("""
            INSERT INTO documents_fts (title, content, search_vector)
            VALUES (:title, :content,
                    to_tsvector('english', :title || ' ' || :content))
            ON CONFLICT (title)
            DO UPDATE SET
                content = EXCLUDED.content,
                search_vector = EXCLUDED.search_vector,
                updated_at = CURRENT_TIMESTAMP
        """)

        session.execute(sql, {
            "title": title,
            "content": content[:1_000_000]  # safety cap
        })