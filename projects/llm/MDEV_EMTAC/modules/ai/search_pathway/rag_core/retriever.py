from __future__ import annotations

from typing import List, Dict, Any, Optional
from sqlalchemy import text, bindparam
from sqlalchemy.types import Integer, String
from sqlalchemy.exc import SQLAlchemyError

from modules.configuration.config_env import DatabaseConfig
from modules.configuration.log_config import (
    debug_id,
    info_id,
    error_id,
    with_request_id,
)


class PgVectorRetriever:

    def __init__(
        self,
        db_config: Optional[DatabaseConfig] = None,
        default_top_k: int = 5,
        distance_metric: str = "cosine",
    ):
        self.db_config = db_config or DatabaseConfig()
        self.default_top_k = default_top_k
        self.distance_metric = distance_metric

    def _distance_operator(self) -> str:
        if self.distance_metric.lower() in ("l2", "euclidean"):
            return "<->"
        return "<=>"

    @with_request_id
    def retrieve(
            self,
            query_embedding: List[float],
            top_k: Optional[int] = None,
            request_id: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        if not query_embedding:
            raise ValueError("query_embedding must be non-empty")
        k = int(top_k or self.default_top_k)
        debug_id(
            f"[PgVectorRetriever] Retrieving (top_k={k}, metric={self.distance_metric})",
            request_id,
        )
        # Build vector literal for pgvector
        vector_literal = "[" + ", ".join(str(float(x)) for x in query_embedding) + "]"
        op = self._distance_operator()

        # Use f-string to embed the vector literal directly in the SQL
        # Only bind the k parameter
        sql = text(f"""
            SELECT
                de.id AS embedding_id,
                de.document_id AS document_id,
                (de.embedding_vector {op} '{vector_literal}'::vector) AS dist,
                d.content AS content,
                d.file_path AS file_path,
                d.complete_document_id AS complete_document_id,
                d.rev AS rev
            FROM document_embedding de
            JOIN document d ON d.id = de.document_id
            WHERE de.embedding_vector IS NOT NULL
            ORDER BY dist ASC
            LIMIT :k
        """)

        # Only bind the k parameter now
        sql = sql.bindparams(
            bindparam("k", type_=Integer),
        )

        try:
            with self.db_config.main_session() as session:
                rows = session.execute(
                    sql,
                    {"k": k},
                ).fetchall()
                info_id(f"[PgVectorRetriever] Retrieved {len(rows)} chunks", request_id)
                return [
                    {
                        "embedding_id": r.embedding_id,
                        "document_id": r.document_id,
                        "content": r.content,
                        "file_path": r.file_path,
                        "complete_document_id": r.complete_document_id,
                        "rev": r.rev,
                        "distance": float(r.dist),
                    }
                    for r in rows
                ]
        except Exception as e:
            error_id(f"[PgVectorRetriever] Unexpected error: {e}", request_id)
            raise
