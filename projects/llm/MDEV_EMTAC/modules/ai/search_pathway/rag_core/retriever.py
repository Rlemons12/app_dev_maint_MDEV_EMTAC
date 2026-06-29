from __future__ import annotations

import os
from typing import List, Dict, Any, Optional, Tuple

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

    Environment options:
        EMTAC_RAG_LOG_RETRIEVED_CHUNKS=true
        EMTAC_RAG_MIN_SIMILARITY=0.40
        EMTAC_RAG_WARN_DISTANCE=0.65
        EMTAC_RAG_REJECT_WEAK_RESULTS=true

    Important behavior:
        Normal RAG:
            Uses the configured similarity quality gate.

        Ask This Document / document_scope mode:
            The selected complete_document_id is already the trust boundary.
            Do NOT discard selected-document chunks just because the score is
            below the global threshold. Return the best scoped chunks and mark
            weak matches with metadata.
    """

    DEFAULT_WARN_DISTANCE = 0.65

    def __init__(
        self,
        db_config: Optional[DatabaseConfig] = None,
        default_top_k: int = 5,
        distance_metric: str = "cosine",
        min_similarity: Optional[float] = None,
        warn_distance: Optional[float] = None,
        reject_weak_results: Optional[bool] = None,
        log_retrieved_chunks: Optional[bool] = None,
    ):
        self.db_config = db_config or DatabaseConfig()
        self.default_top_k = default_top_k
        self.distance_metric = distance_metric

        self.min_similarity = (
            self._safe_float(min_similarity)
            if min_similarity is not None
            else self._env_float("EMTAC_RAG_MIN_SIMILARITY", 0.0)
        )

        self.warn_distance = (
            self._safe_float(warn_distance)
            if warn_distance is not None
            else self._env_float("EMTAC_RAG_WARN_DISTANCE", self.DEFAULT_WARN_DISTANCE)
        )

        self.reject_weak_results = (
            bool(reject_weak_results)
            if reject_weak_results is not None
            else self._env_bool("EMTAC_RAG_REJECT_WEAK_RESULTS", False)
        )

        self.log_retrieved_chunks = (
            bool(log_retrieved_chunks)
            if log_retrieved_chunks is not None
            else self._env_bool("EMTAC_RAG_LOG_RETRIEVED_CHUNKS", True)
        )

    def _refresh_runtime_settings(self) -> None:
        self.min_similarity = self._env_float(
            "EMTAC_RAG_MIN_SIMILARITY",
            self.min_similarity or 0.0,
        )
        self.warn_distance = self._env_float(
            "EMTAC_RAG_WARN_DISTANCE",
            self.warn_distance or self.DEFAULT_WARN_DISTANCE,
        )
        self.reject_weak_results = self._env_bool(
            "EMTAC_RAG_REJECT_WEAK_RESULTS",
            self.reject_weak_results,
        )
        self.log_retrieved_chunks = self._env_bool(
            "EMTAC_RAG_LOG_RETRIEVED_CHUNKS",
            self.log_retrieved_chunks,
        )

    def _distance_operator(self) -> str:
        if self.distance_metric.lower() in ("l2", "euclidean"):
            return "<->"
        return "<=>"

    def _using_cosine(self) -> bool:
        return self.distance_metric.lower() not in ("l2", "euclidean")

    @with_request_id
    def retrieve(
        self,
        query_embedding: List[float],
        top_k: Optional[int] = None,
        request_id: Optional[str] = None,
        document_scope: Optional[Dict[str, Any]] = None,
        complete_document_id: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        self._refresh_runtime_settings()

        if not query_embedding:
            raise ValueError("query_embedding must be non-empty")

        k = self._normalize_top_k(top_k)

        selected_complete_document_id = self._resolve_complete_document_id(
            document_scope=document_scope,
            complete_document_id=complete_document_id,
        )

        document_scope_enabled = selected_complete_document_id is not None

        (
            effective_min_similarity,
            effective_reject_weak_results,
        ) = self._resolve_effective_quality_settings(
            document_scope_enabled=document_scope_enabled,
        )

        debug_id(
            "[PgVectorRetriever] Retrieving "
            f"(top_k={k}, metric={self.distance_metric}, "
            f"document_scope_enabled={document_scope_enabled}, "
            f"complete_document_id={selected_complete_document_id}, "
            f"min_similarity={self.min_similarity}, "
            f"effective_min_similarity={effective_min_similarity}, "
            f"warn_distance={self.warn_distance}, "
            f"reject_weak_results={self.reject_weak_results}, "
            f"effective_reject_weak_results={effective_reject_weak_results})",
            request_id,
        )

        vector_literal = self._build_vector_literal(query_embedding)
        op = self._distance_operator()

        where_clauses = ["de.embedding_vector IS NOT NULL"]
        params: Dict[str, Any] = {"k": k}
        bind_params = [bindparam("k", type_=Integer)]

        if selected_complete_document_id is not None:
            where_clauses.append("d.complete_document_id = :complete_document_id")
            params["complete_document_id"] = selected_complete_document_id
            bind_params.append(bindparam("complete_document_id", type_=Integer))

        where_sql = "\n              AND ".join(where_clauses)

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

            raw_chunks = self._rows_to_chunks(rows)

            self._mark_chunk_quality(
                chunks=raw_chunks,
                configured_min_similarity=self.min_similarity,
                effective_min_similarity=effective_min_similarity,
                document_scope_enabled=document_scope_enabled,
            )

            self._log_retrieval_summary(
                chunks=raw_chunks,
                request_id=request_id,
                selected_complete_document_id=selected_complete_document_id,
            )

            filtered_chunks, rejected_chunks = self._apply_quality_gate(
                chunks=raw_chunks,
                min_similarity=effective_min_similarity,
            )

            weak_returned_count = len(
                [
                    chunk
                    for chunk in raw_chunks
                    if bool(chunk.get("weak_match"))
                ]
            )

            if rejected_chunks:
                warning_id(
                    "[PgVectorRetriever] Weak chunks rejected "
                    f"rejected_count={len(rejected_chunks)} "
                    f"returned_count={len(filtered_chunks)} "
                    f"min_similarity={self.min_similarity} "
                    f"effective_min_similarity={effective_min_similarity} "
                    f"reject_weak_results={self.reject_weak_results} "
                    f"effective_reject_weak_results={effective_reject_weak_results} "
                    f"document_scope_enabled={document_scope_enabled} "
                    f"complete_document_id={selected_complete_document_id}",
                    request_id,
                )

                for chunk in rejected_chunks[:5]:
                    warning_id(
                        "[PgVectorRetriever] Rejected weak chunk "
                        f"chunk_id={chunk.get('chunk_id')} "
                        f"distance={self._fmt(chunk.get('distance'))} "
                        f"similarity={self._fmt(chunk.get('similarity'))} "
                        f"complete_document_id={chunk.get('complete_document_id')} "
                        f"preview={self._preview(chunk.get('content'))}",
                        request_id,
                    )

            elif document_scope_enabled and weak_returned_count:
                debug_id(
                    "[PgVectorRetriever] Document-scope weak chunks preserved "
                    f"weak_returned_count={weak_returned_count} "
                    f"raw_count={len(raw_chunks)} "
                    f"configured_min_similarity={self.min_similarity} "
                    f"effective_min_similarity={effective_min_similarity} "
                    f"configured_reject_weak_results={self.reject_weak_results} "
                    f"effective_reject_weak_results={effective_reject_weak_results} "
                    f"complete_document_id={selected_complete_document_id}",
                    request_id,
                )

            final_chunks = (
                filtered_chunks
                if effective_reject_weak_results
                else raw_chunks
            )

            info_id(
                "[PgVectorRetriever] Retrieved "
                f"{len(final_chunks)} chunks "
                f"(raw_count={len(raw_chunks)}, "
                f"rejected_count={len(rejected_chunks) if effective_reject_weak_results else 0}, "
                f"weak_returned_count={weak_returned_count}, "
                f"document_scope_enabled={document_scope_enabled}, "
                f"complete_document_id={selected_complete_document_id}, "
                f"min_similarity={self.min_similarity}, "
                f"effective_min_similarity={effective_min_similarity}, "
                f"reject_weak_results={self.reject_weak_results}, "
                f"effective_reject_weak_results={effective_reject_weak_results})",
                request_id,
            )

            return final_chunks

        except Exception as e:
            error_id(
                "[PgVectorRetriever] Unexpected error: "
                f"{e} "
                f"(document_scope_enabled={document_scope_enabled}, "
                f"complete_document_id={selected_complete_document_id})",
                request_id,
                exc_info=True,
            )
            raise

    def _rows_to_chunks(self, rows: Any) -> List[Dict[str, Any]]:
        chunks: List[Dict[str, Any]] = []

        for row in rows or []:
            distance = self._safe_float(row.dist)
            similarity = None

            if distance is not None and self._using_cosine():
                similarity = 1.0 - distance

            chunks.append(
                {
                    "embedding_id": row.embedding_id,
                    "id": row.chunk_id,
                    "chunk_id": row.chunk_id,
                    "document_id": row.document_id,
                    "content": row.content,
                    "text": row.content,
                    "file_path": row.file_path,
                    "complete_document_id": row.complete_document_id,
                    "rev": row.rev,
                    "distance": distance,
                    "similarity": similarity,
                    "retrieval_metric": self.distance_metric,
                    "weak_match": False,
                    "document_scope_enabled": False,
                }
            )

        return chunks

    def _resolve_effective_quality_settings(
        self,
        *,
        document_scope_enabled: bool,
    ) -> Tuple[float, bool]:
        """
        Resolves the quality gate used for the current retrieval.

        Normal/global RAG:
            Honor configured EMTAC_RAG_MIN_SIMILARITY and
            EMTAC_RAG_REJECT_WEAK_RESULTS.

        Document-scoped mode:
            Disable hard rejection. The selected complete_document_id is already
            the trust boundary. The best chunks from that selected document
            should be returned to the RAG pipeline, even if their similarity is
            below the global threshold.

        This does not hide weak matches. Chunks are still annotated with
        weak_match=True when they fall below the configured global threshold.
        """

        configured_min_similarity = self._safe_float(self.min_similarity) or 0.0
        configured_reject_weak_results = bool(self.reject_weak_results)

        if document_scope_enabled:
            return 0.0, False

        return configured_min_similarity, configured_reject_weak_results

    def _mark_chunk_quality(
        self,
        *,
        chunks: List[Dict[str, Any]],
        configured_min_similarity: Optional[float],
        effective_min_similarity: Optional[float],
        document_scope_enabled: bool,
    ) -> None:
        """
        Adds debug/useful metadata to chunks without removing them.

        weak_match:
            True when the chunk similarity is below the configured global
            min_similarity.

        rejected_by_effective_gate:
            True when the chunk would be rejected by the actual active gate.
        """

        if not chunks:
            return

        configured_min = self._safe_float(configured_min_similarity) or 0.0
        effective_min = self._safe_float(effective_min_similarity) or 0.0

        for chunk in chunks:
            if not isinstance(chunk, dict):
                continue

            chunk["document_scope_enabled"] = bool(document_scope_enabled)

            similarity = self._safe_float(chunk.get("similarity"))

            if not self._using_cosine() or similarity is None:
                chunk["weak_match"] = False
                chunk["rejected_by_effective_gate"] = False
                continue

            chunk["weak_match"] = (
                configured_min > 0
                and similarity < configured_min
            )

            chunk["rejected_by_effective_gate"] = (
                effective_min > 0
                and similarity < effective_min
            )

    def _apply_quality_gate(
        self,
        *,
        chunks: List[Dict[str, Any]],
        min_similarity: Optional[float],
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        if not chunks:
            return [], []

        if not self._using_cosine():
            return chunks, []

        effective_min_similarity = self._safe_float(min_similarity) or 0.0

        if effective_min_similarity <= 0:
            return chunks, []

        kept: List[Dict[str, Any]] = []
        rejected: List[Dict[str, Any]] = []

        for chunk in chunks:
            similarity = self._safe_float(chunk.get("similarity"))

            if similarity is None:
                rejected.append(chunk)
            elif similarity >= effective_min_similarity:
                kept.append(chunk)
            else:
                rejected.append(chunk)

        return kept, rejected

    def _log_retrieval_summary(
        self,
        *,
        chunks: List[Dict[str, Any]],
        request_id: Optional[str],
        selected_complete_document_id: Optional[int],
    ) -> None:
        if not chunks:
            warning_id(
                "[PgVectorRetriever] Retrieved zero raw chunks "
                f"(document_scope_enabled={selected_complete_document_id is not None}, "
                f"complete_document_id={selected_complete_document_id})",
                request_id,
            )
            return

        distances = [
            self._safe_float(chunk.get("distance"))
            for chunk in chunks
            if self._safe_float(chunk.get("distance")) is not None
        ]

        similarities = [
            self._safe_float(chunk.get("similarity"))
            for chunk in chunks
            if self._safe_float(chunk.get("similarity")) is not None
        ]

        best_distance = min(distances) if distances else None
        worst_distance = max(distances) if distances else None
        best_similarity = max(similarities) if similarities else None
        worst_similarity = min(similarities) if similarities else None

        debug_id(
            "[PgVectorRetriever] Retrieval score summary "
            f"count={len(chunks)} "
            f"best_distance={self._fmt(best_distance)} "
            f"worst_distance={self._fmt(worst_distance)} "
            f"best_similarity={self._fmt(best_similarity)} "
            f"worst_similarity={self._fmt(worst_similarity)}",
            request_id,
        )

        if (
            self.warn_distance is not None
            and worst_distance is not None
            and worst_distance >= self.warn_distance
        ):
            warning_id(
                "[PgVectorRetriever] Weak retrieval distance warning "
                f"worst_distance={self._fmt(worst_distance)} "
                f"warn_distance={self._fmt(self.warn_distance)} "
                f"best_distance={self._fmt(best_distance)} "
                f"count={len(chunks)} "
                f"document_scope_enabled={selected_complete_document_id is not None} "
                f"complete_document_id={selected_complete_document_id}",
                request_id,
            )

        if not self.log_retrieved_chunks:
            return

        for rank, chunk in enumerate(chunks, start=1):
            debug_id(
                "[PgVectorRetriever] Retrieved chunk "
                f"rank={rank} "
                f"chunk_id={chunk.get('chunk_id')} "
                f"document_id={chunk.get('document_id')} "
                f"complete_document_id={chunk.get('complete_document_id')} "
                f"distance={self._fmt(chunk.get('distance'))} "
                f"similarity={self._fmt(chunk.get('similarity'))} "
                f"weak_match={bool(chunk.get('weak_match'))} "
                f"rejected_by_effective_gate={bool(chunk.get('rejected_by_effective_gate'))} "
                f"file_path={chunk.get('file_path')} "
                f"preview={self._preview(chunk.get('content'))}",
                request_id,
            )

    @staticmethod
    def _normalize_top_k(top_k: Optional[int]) -> int:
        try:
            k = int(top_k or 5)
        except (TypeError, ValueError):
            k = 5

        return k if k > 0 else 5

    @staticmethod
    def _build_vector_literal(query_embedding: List[float]) -> str:
        return "[" + ", ".join(str(float(value)) for value in query_embedding) + "]"

    @classmethod
    def _resolve_complete_document_id(
        cls,
        *,
        document_scope: Optional[Dict[str, Any]],
        complete_document_id: Optional[int],
    ) -> Optional[int]:
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

        if document_scope.get("enabled", True) is False:
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

    @staticmethod
    def _safe_float(value: Any) -> Optional[float]:
        if value in (None, "", "None"):
            return None

        if isinstance(value, bool):
            return None

        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _env_bool(name: str, default: bool) -> bool:
        raw = os.getenv(name)

        if raw is None:
            return default

        value = str(raw).strip().strip('"').strip("'").lower()

        if value in ("1", "true", "yes", "y", "on"):
            return True

        if value in ("0", "false", "no", "n", "off"):
            return False

        return default

    @classmethod
    def _env_float(cls, name: str, default: float) -> float:
        raw = os.getenv(name)

        if raw is None:
            return default

        value = cls._safe_float(str(raw).strip().strip('"').strip("'"))

        return default if value is None else value

    @staticmethod
    def _preview(value: Any, limit: int = 180) -> str:
        text_value = " ".join(str(value or "").split())

        if len(text_value) <= limit:
            return repr(text_value)

        return repr(text_value[:limit] + "...")

    @staticmethod
    def _fmt(value: Any) -> str:
        number = PgVectorRetriever._safe_float(value)

        if number is None:
            return "None"

        return f"{number:.4f}"