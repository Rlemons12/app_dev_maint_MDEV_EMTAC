from typing import Dict, Any, List, Optional
from sqlalchemy import text
from modules.orchestrators.base_orchestrator import BaseOrchestrator
from modules.configuration.log_config import with_request_id


class ChunkSimilarityOrchestrator(BaseOrchestrator):

    @with_request_id
    def find_similar_chunks(
        self,
        *,
        query_embedding: List[float],
        model_name: str,
        limit: int = 10,
        similarity_threshold: float = 0.5,
        request_id: Optional[str] = None,
    ) -> List[Dict[str, Any]]:

        if not query_embedding:
            return []

        # PostgreSQL pgvector expects array-like format
        vector_literal = "[" + ",".join(map(str, query_embedding)) + "]"

        with self.transaction() as session:

            sql = text("""
                SELECT
                    d.id,
                    d.name,
                    d.content,
                    d.complete_document_id,
                    1 - (e.embedding_vector <=> :vec) AS score
                FROM document_embedding e
                JOIN document d ON d.id = e.document_id
                WHERE e.model_name = :model
                  AND (1 - (e.embedding_vector <=> :vec)) >= :threshold
                ORDER BY e.embedding_vector <=> :vec
                LIMIT :limit
            """)

            rows = session.execute(
                sql,
                {
                    "vec": vector_literal,
                    "model": model_name,
                    "threshold": similarity_threshold,
                    "limit": limit,
                },
            ).fetchall()

            results = []

            for r in rows:
                results.append(
                    {
                        "chunk_id": r.id,
                        "chunk_name": r.name,
                        "content_preview": (
                            r.content[:300] if r.content else None
                        ),
                        "complete_document_id": r.complete_document_id,
                        "similarity": float(r.score),
                    }
                )

            return results