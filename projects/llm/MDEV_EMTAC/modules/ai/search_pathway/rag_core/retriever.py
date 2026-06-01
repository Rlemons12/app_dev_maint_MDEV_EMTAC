from __future__ import annotations

from typing import List, Dict, Any, Optional

from sqlalchemy import text, bindparam
from sqlalchemy.types import Integer

from modules.configuration.config_env import DatabaseConfig
from modules.configuration.log_config import (
    debug_id,
    info_id,
    error_id,
    warning_id,
    with_request_id,
)


class PgVectorRetriever:
    """
    pgvector-backed document chunk retriever.

    Supports:
        - normal semantic retrieval across all document chunks
        - document-scoped retrieval filtered by complete_document_id

    Document-scoped conversation mode:
        The RAG pipeline may call this in either form:

            retrieve(..., document_scope={
                "enabled": true,
                "scope_type": "complete_document",
                "complete_document_id": 29,
                "document_name": "Document #29"
            })

        or:

            retrieve(..., complete_document_id=29)

        In both cases, this retriever applies the SQL filter:

            d.complete_document_id = :complete_document_id

        That means pgvector only ranks chunks from the selected manual/document.
    """

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
        """
        Return the pgvector distance operator.

        cosine:
            <=> cosine distance

        l2 / euclidean:
            <-> L2 distance
        """

        if self.distance_metric.lower() in ("l2", "euclidean"):
            return "<->"

        return "<=>"

    @with_request_id
    def retrieve(
        self,
        query_embedding: List[float],
        top_k: Optional[int] = None,
        request_id: Optional[str] = None,
        document_scope: Optional[Dict[str, Any]] = None,
        complete_document_id: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """
        Retrieve semantically similar document chunks.

        Args:
            query_embedding:
                Query vector produced by the active embedding model.

            top_k:
                Number of chunks to return.

            request_id:
                Request ID for logging.

            document_scope:
                Optional document mode scope object.

            complete_document_id:
                Optional direct complete_document_id filter.

        Returns:
            List of chunk dictionaries compatible with the RAG pipeline.
        """

        if not query_embedding:
            raise ValueError("query_embedding must be non-empty")

        k = self._normalize_top_k(top_k)
        selected_complete_document_id = self._resolve_complete_document_id(
            document_scope=document_scope,
            complete_document_id=complete_document_id,
        )

        debug_id(
            "[PgVectorRetriever] Retrieving "
            f"(top_k={k}, metric={self.distance_metric}, "
            f"document_scope_enabled={selected_complete_document_id is not None}, "
            f"complete_document_id={selected_complete_document_id})",
            request_id,
        )

        vector_literal = self._build_vector_literal(query_embedding)
        op = self._distance_operator()

        where_clauses = [
            "de.embedding_vector IS NOT NULL",
        ]

        params: Dict[str, Any] = {
            "k": k,
        }

        bind_params = [
            bindparam("k", type_=Integer),
        ]

        if selected_complete_document_id is not None:
            where_clauses.append("d.complete_document_id = :complete_document_id")
            params["complete_document_id"] = selected_complete_document_id
            bind_params.append(bindparam("complete_document_id", type_=Integer))

        where_sql = "\n              AND ".join(where_clauses)

        # Note:
        # The distance operator is controlled by _distance_operator(), not user input.
        # The vector literal is built from float-casted values.
        sql = text(f"""
            SELECT
                de.id AS embedding_id,
                de.document_id AS document_id,
                d.id AS chunk_id,
                (de.embedding_vector {op} '{vector_literal}'::vector) AS dist,
                d.content AS content,
                d.file_path AS file_path,
                d.complete_document_id AS complete_document_id,
                d.rev AS rev
            FROM document_embedding de
            JOIN document d ON d.id = de.document_id
            WHERE {where_sql}
            ORDER BY dist ASC
            LIMIT :k
        """)

        sql = sql.bindparams(*bind_params)

        try:
            with self.db_config.main_session() as session:
                rows = session.execute(sql, params).fetchall()

            info_id(
                "[PgVectorRetriever] Retrieved "
                f"{len(rows)} chunks "
                f"(document_scope_enabled={selected_complete_document_id is not None}, "
                f"complete_document_id={selected_complete_document_id})",
                request_id,
            )

            return [
                {
                    "embedding_id": row.embedding_id,

                    # Keep both names because different layers use different keys.
                    "id": row.chunk_id,
                    "chunk_id": row.chunk_id,
                    "document_id": row.document_id,

                    "content": row.content,
                    "text": row.content,
                    "file_path": row.file_path,
                    "complete_document_id": row.complete_document_id,
                    "rev": row.rev,
                    "distance": float(row.dist),
                }
                for row in rows
            ]

        except Exception as e:
            error_id(
                "[PgVectorRetriever] Unexpected error: "
                f"{e} "
                f"(document_scope_enabled={selected_complete_document_id is not None}, "
                f"complete_document_id={selected_complete_document_id})",
                request_id,
            )
            raise

    @staticmethod
    def _normalize_top_k(top_k: Optional[int]) -> int:
        try:
            k = int(top_k or 5)
        except (TypeError, ValueError):
            k = 5

        if k <= 0:
            return 5

        return k

    @staticmethod
    def _build_vector_literal(query_embedding: List[float]) -> str:
        """
        Build a pgvector literal from a list of numeric values.

        Every value is float-casted before being placed in the vector literal.
        """

        return "[" + ", ".join(str(float(value)) for value in query_embedding) + "]"

    @classmethod
    def _resolve_complete_document_id(
        cls,
        *,
        document_scope: Optional[Dict[str, Any]],
        complete_document_id: Optional[int],
    ) -> Optional[int]:
        """
        Resolve complete_document_id from either direct argument or document_scope.
        Direct complete_document_id wins if supplied.
        """

        direct_id = cls._safe_int(complete_document_id)

        if direct_id is not None:
            return direct_id

        normalized_scope = cls._normalize_document_scope(document_scope)

        if not normalized_scope:
            return None

        return cls._safe_int(normalized_scope.get("complete_document_id"))

    @classmethod
    def _normalize_document_scope(
        cls,
        document_scope: Optional[Dict[str, Any]],
    ) -> Optional[Dict[str, Any]]:
        if not document_scope or not isinstance(document_scope, dict):
            return None

        enabled = document_scope.get("enabled", True)

        if enabled is False:
            return None

        scope_type = (
            document_scope.get("scope_type")
            or document_scope.get("scopeType")
            or "complete_document"
        )

        scope_type = str(scope_type or "").strip() or "complete_document"

        if scope_type != "complete_document":
            return None

        complete_document_id = (
            document_scope.get("complete_document_id")
            or document_scope.get("completed_document_id")
            or document_scope.get("completeDocumentId")
            or document_scope.get("completeDocumentID")
        )

        complete_document_id = cls._safe_int(complete_document_id)

        if complete_document_id is None:
            return None

        document_id = (
            document_scope.get("document_id")
            or document_scope.get("documentId")
        )

        document_id = cls._safe_int(document_id)

        document_name = (
            document_scope.get("document_name")
            or document_scope.get("documentName")
            or document_scope.get("name")
            or document_scope.get("title")
            or f"Document #{complete_document_id}"
        )

        document_name = str(document_name or "").strip() or f"Document #{complete_document_id}"

        return {
            "enabled": True,
            "scope_type": "complete_document",
            "document_id": document_id,
            "complete_document_id": complete_document_id,
            "document_name": document_name,
        }

    @staticmethod
    def _safe_int(value: Any) -> Optional[int]:
        if value in (None, "", "None"):
            return None

        if isinstance(value, bool):
            return None

        try:
            return int(value)
        except (TypeError, ValueError):
            return None