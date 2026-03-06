# modules/services/document_embedding_service.py

from typing import List, Optional, Dict
from sqlalchemy.orm import Session
from sqlalchemy import text

from modules.configuration.log_config import (
    info_id,
    debug_id,
    error_id,
    with_request_id,
)
from modules.emtacdb.emtacdb_fts import DocumentEmbedding


class DocumentEmbeddingService:
    """
    Pure domain service for DocumentEmbedding.

    HARD RULES:
    - NEVER open sessions
    - NEVER close sessions
    - NEVER commit
    - NEVER rollback
    - Orchestrator owns transactions
    """

    # ==========================================================
    # CREATE
    # ==========================================================

    @with_request_id
    def add_embedding(
        self,
        session: Session,
        *,
        document_id: int,
        model_name: str,
        embedding: List[float],
        use_pgvector: bool = True,
        metadata: Optional[Dict] = None,
        request_id: Optional[str] = None,
    ) -> int:

        if not session:
            raise RuntimeError("Session required for add_embedding")

        if use_pgvector:
            embedding_obj = DocumentEmbedding.create_with_pgvector(
                document_id=document_id,
                model_name=model_name,
                embedding=embedding,
                embedding_metadata=metadata or {},
            )
        else:
            embedding_obj = DocumentEmbedding.create_with_legacy(
                document_id=document_id,
                model_name=model_name,
                embedding=embedding,
                embedding_metadata=metadata or {},
            )

        session.add(embedding_obj)
        session.flush()

        debug_id(
            f"Embedding staged doc={document_id}, "
            f"model={model_name}, id={embedding_obj.id}",
            request_id,
        )

        return embedding_obj.id

    # ==========================================================
    # READ
    # ==========================================================

    @with_request_id
    def get_by_id(
        self,
        session: Session,
        *,
        embedding_id: int,
        request_id: Optional[str] = None,
    ) -> Optional[DocumentEmbedding]:

        return session.get(DocumentEmbedding, embedding_id)

    @with_request_id
    def get_by_document(
        self,
        session: Session,
        *,
        document_id: int,
        request_id: Optional[str] = None,
    ) -> List[DocumentEmbedding]:

        return (
            session.query(DocumentEmbedding)
            .filter(DocumentEmbedding.document_id == document_id)
            .all()
        )

    # ==========================================================
    # VECTOR SEARCH (DIMENSION SAFE)
    # ==========================================================

    @with_request_id
    def search_similar(
        self,
        session: Session,
        *,
        query_embedding: List[float],
        model_name: Optional[str] = None,
        limit: int = 5,
        threshold: float = 0.5,
        request_id: Optional[str] = None,
    ) -> List[Dict]:

        if not query_embedding:
            return []

        dims = len(query_embedding)
        vector_str = "[" + ",".join(map(str, query_embedding)) + "]"

        sql = text("""
            SELECT 
                id,
                document_id,
                model_name,
                actual_dimensions,
                1 - (embedding_vector <=> :query_vector::vector) AS similarity
            FROM document_embedding
            WHERE embedding_vector IS NOT NULL
              AND actual_dimensions = :dims
              AND (:model_name IS NULL OR model_name = :model_name)
              AND (1 - (embedding_vector <=> :query_vector::vector)) >= :threshold
            ORDER BY embedding_vector <=> :query_vector::vector ASC
            LIMIT :limit
        """)

        rows = session.execute(
            sql,
            {
                "query_vector": vector_str,
                "dims": dims,
                "model_name": model_name,
                "threshold": threshold,
                "limit": limit,
            },
        ).fetchall()

        results = [
            {
                "embedding_id": r[0],
                "document_id": r[1],
                "model_name": r[2],
                "dimensions": r[3],
                "similarity": float(r[4]),
            }
            for r in rows
        ]

        info_id(f"Vector search returned {len(results)} results", request_id)

        return results

    # ==========================================================
    # MIGRATION
    # ==========================================================

    @with_request_id
    def migrate_all_to_pgvector(
        self,
        session: Session,
        *,
        request_id: Optional[str] = None,
    ) -> Dict:

        legacy_embeddings = (
            session.query(DocumentEmbedding)
            .filter(
                DocumentEmbedding.embedding_vector.is_(None),
                DocumentEmbedding.model_embedding.isnot(None),
            )
            .all()
        )

        migrated = 0
        failed = 0

        for emb in legacy_embeddings:
            if emb.migrate_to_pgvector():
                session.add(emb)
                migrated += 1
            else:
                failed += 1

        session.flush()

        info_id(
            f"Migrated {migrated} embeddings (no commit here)",
            request_id,
        )

        return {"migrated": migrated, "failed": failed}

    # ==========================================================
    # INDEX CREATION (PRODUCTION READY)
    # ==========================================================

    @with_request_id
    def setup_pgvector_indexes(
        self,
        session: Session,
        *,
        request_id: Optional[str] = None,
    ) -> bool:

        index_statements = [
            # Core indexes
            "CREATE INDEX IF NOT EXISTS idx_doc_embedding_doc_id ON document_embedding (document_id);",
            "CREATE INDEX IF NOT EXISTS idx_doc_embedding_model ON document_embedding (model_name);",
            "CREATE INDEX IF NOT EXISTS idx_doc_embedding_dims ON document_embedding (actual_dimensions);",
            "CREATE INDEX IF NOT EXISTS idx_doc_embedding_metadata ON document_embedding USING gin (embedding_metadata);",

            # HNSW cosine indexes by dimension
            """
            CREATE INDEX IF NOT EXISTS idx_doc_embedding_cosine_384d
            ON document_embedding
            USING hnsw (embedding_vector vector_cosine_ops)
            WHERE actual_dimensions = 384;
            """,

            """
            CREATE INDEX IF NOT EXISTS idx_doc_embedding_cosine_768d
            ON document_embedding
            USING hnsw (embedding_vector vector_cosine_ops)
            WHERE actual_dimensions = 768;
            """,

            """
            CREATE INDEX IF NOT EXISTS idx_doc_embedding_cosine_1536d
            ON document_embedding
            USING hnsw (embedding_vector vector_cosine_ops)
            WHERE actual_dimensions = 1536;
            """,
        ]

        try:
            for stmt in index_statements:
                session.execute(text(stmt))

            session.flush()

            info_id("pgvector indexes ensured", request_id)
            return True

        except Exception as e:
            error_id(f"Index creation failed: {e}", request_id)
            raise
