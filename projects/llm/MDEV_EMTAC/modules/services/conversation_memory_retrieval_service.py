from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from sqlalchemy import text
from sqlalchemy.orm import Session

from modules.configuration.log_config import (
    debug_id,
    warning_id,
    error_id,
)

from modules.services.qanda_embedding_service import QandAEmbeddingService
from modules.emtacdb.emtacdb_fts import QandA
from modules.ai.search_pathway.rag_core.embedder import DBConfiguredEmbedder

class ConversationMemoryRetrievalService:
    """
    Semantic conversation memory retrieval service.

    Purpose:
        Finds older Q&A interactions that are semantically related to the
        current user question, then renders them as prompt-ready memory text.

    Why this exists:
        ChatOrchestrator already stores conversation_summary/session_data and
        QandA embeddings. This service adds long-range memory recall by searching
        QandA.question_embedding and QandA.answer_embedding.

    This service does NOT:
        - Own database transactions
        - Commit or rollback
        - Generate answers
        - Modify ChatSession
        - Modify QandA rows unless explicitly asked to backfill missing embeddings

    Expected QandA fields:
        - id
        - user_id
        - question
        - answer
        - timestamp
        - request_id
        - question_embedding
        - answer_embedding
    """

    DEFAULT_TOP_K = 5
    DEFAULT_CANDIDATE_LIMIT = 25
    DEFAULT_MAX_QUESTION_CHARS = 500
    DEFAULT_MAX_ANSWER_CHARS = 900
    DEFAULT_MAX_MEMORY_CHARS = 5000

    def __init__(
            self,
            *,
            embedder: Optional[BaseEmbedder] = None,
            top_k: int = DEFAULT_TOP_K,
            candidate_limit: int = DEFAULT_CANDIDATE_LIMIT,
            max_question_chars: int = DEFAULT_MAX_QUESTION_CHARS,
            max_answer_chars: int = DEFAULT_MAX_ANSWER_CHARS,
            max_memory_chars: int = DEFAULT_MAX_MEMORY_CHARS,
    ) -> None:
        self.embedder = embedder or DBConfiguredEmbedder()
        self.top_k = int(top_k or self.DEFAULT_TOP_K)
        self.candidate_limit = int(candidate_limit or self.DEFAULT_CANDIDATE_LIMIT)
        self.max_question_chars = int(max_question_chars or self.DEFAULT_MAX_QUESTION_CHARS)
        self.max_answer_chars = int(max_answer_chars or self.DEFAULT_MAX_ANSWER_CHARS)
        self.max_memory_chars = int(max_memory_chars or self.DEFAULT_MAX_MEMORY_CHARS)

    def build_relevant_memory_context(
        self,
        *,
        session: Session,
        user_id: str,
        question: str,
        request_id: Optional[str] = None,
        exclude_request_id: Optional[str] = None,
        conversation_id: Optional[str] = None,
        top_k: Optional[int] = None,
    ) -> str:
        """
        Main entry point.

        Returns:
            Prompt-ready text such as:

            Relevant long-term conversation memories:
            - Prior question: ...
              Prior answer: ...

        Safe failure behavior:
            Returns "" if anything fails.
        """

        normalized_question = (question or "").strip()
        normalized_user_id = str(user_id or "anonymous").strip() or "anonymous"

        if not normalized_question:
            return ""

        try:
            query_embedding = self._embed_question(
                question=normalized_question,
                request_id=request_id,
            )

            if not query_embedding:
                warning_id(
                    "[ConversationMemoryRetrievalService] No query embedding produced.",
                    request_id,
                )
                return ""

            memories = self.search_relevant_qanda_memories(
                session=session,
                user_id=normalized_user_id,
                query_embedding=query_embedding,
                request_id=request_id,
                exclude_request_id=exclude_request_id,
                top_k=top_k or self.top_k,
            )

            if not memories:
                debug_id(
                    "[ConversationMemoryRetrievalService] No relevant semantic memories found.",
                    request_id,
                )
                return ""

            rendered = self.render_memories_for_prompt(
                memories=memories,
                conversation_id=conversation_id,
            )

            debug_id(
                "[ConversationMemoryRetrievalService] Built relevant memory context "
                f"user_id={normalized_user_id} "
                f"memories={len(memories)} "
                f"chars={len(rendered)}",
                request_id,
            )

            return rendered

        except Exception as exc:
            error_id(
                f"[ConversationMemoryRetrievalService] Memory retrieval failed: {exc}",
                request_id,
                exc_info=True,
            )
            return ""

    def search_relevant_qanda_memories(
        self,
        *,
        session: Session,
        user_id: str,
        query_embedding: List[float],
        request_id: Optional[str] = None,
        exclude_request_id: Optional[str] = None,
        top_k: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """
        Search QandA rows using pgvector cosine distance.

        This uses raw SQL because it is more reliable across pgvector/sqlalchemy
        versions than relying on ORM comparator methods.
        """

        safe_top_k = int(top_k or self.top_k)
        safe_candidate_limit = max(int(self.candidate_limit), safe_top_k)

        if not query_embedding:
            return []

        embedding_literal = self._vector_literal(query_embedding)

        params: Dict[str, Any] = {
            "user_id": str(user_id or "anonymous"),
            "embedding": embedding_literal,
            "limit": safe_candidate_limit,
        }

        exclude_clause = ""

        if exclude_request_id:
            exclude_clause = "AND COALESCE(request_id, '') <> :exclude_request_id"
            params["exclude_request_id"] = str(exclude_request_id)

        sql = text(
            f"""
            WITH question_matches AS (
                SELECT
                    id,
                    user_id,
                    question,
                    answer,
                    timestamp,
                    request_id,
                    'question_embedding' AS match_source,
                    (question_embedding <=> CAST(:embedding AS vector)) AS distance
                FROM qanda
                WHERE user_id = :user_id
                  AND question_embedding IS NOT NULL
                  {exclude_clause}
                ORDER BY question_embedding <=> CAST(:embedding AS vector)
                LIMIT :limit
            ),
            answer_matches AS (
                SELECT
                    id,
                    user_id,
                    question,
                    answer,
                    timestamp,
                    request_id,
                    'answer_embedding' AS match_source,
                    (answer_embedding <=> CAST(:embedding AS vector)) AS distance
                FROM qanda
                WHERE user_id = :user_id
                  AND answer_embedding IS NOT NULL
                  {exclude_clause}
                ORDER BY answer_embedding <=> CAST(:embedding AS vector)
                LIMIT :limit
            ),
            combined AS (
                SELECT * FROM question_matches
                UNION ALL
                SELECT * FROM answer_matches
            ),
            ranked AS (
                SELECT DISTINCT ON (id)
                    id,
                    user_id,
                    question,
                    answer,
                    timestamp,
                    request_id,
                    match_source,
                    distance
                FROM combined
                ORDER BY id, distance ASC
            )
            SELECT
                id,
                user_id,
                question,
                answer,
                timestamp,
                request_id,
                match_source,
                distance
            FROM ranked
            ORDER BY distance ASC
            LIMIT :limit
            """
        )

        try:
            rows = session.execute(sql, params).mappings().all()

            memories: List[Dict[str, Any]] = []

            for row in rows:
                item = dict(row)

                distance = item.get("distance")
                similarity = None

                try:
                    if distance is not None:
                        similarity = 1.0 - float(distance)
                except Exception:
                    similarity = None

                memories.append(
                    {
                        "id": str(item.get("id")),
                        "user_id": item.get("user_id"),
                        "question": item.get("question") or "",
                        "answer": item.get("answer") or "",
                        "timestamp": item.get("timestamp"),
                        "request_id": item.get("request_id"),
                        "match_source": item.get("match_source"),
                        "distance": float(distance) if distance is not None else None,
                        "similarity": similarity,
                    }
                )

            return memories[:safe_top_k]

        except Exception as raw_sql_error:
            warning_id(
                "[ConversationMemoryRetrievalService] Raw pgvector memory search failed. "
                f"Falling back to ORM recency search. error={raw_sql_error}",
                request_id,
                exc_info=True,
            )

            return self._fallback_recent_qanda_memories(
                session=session,
                user_id=user_id,
                exclude_request_id=exclude_request_id,
                top_k=safe_top_k,
                request_id=request_id,
            )

    def render_memories_for_prompt(
            self,
            *,
            memories: List[Dict[str, Any]],
            conversation_id: Optional[str] = None,
    ) -> str:
        """
        Convert retrieved memory rows into clean prompt-ready conversation context.

        Important:
            Do NOT expose retrieval/debug metadata to the model.
            The model should see useful conversation facts, not:
                - similarity scores
                - embedding source names
                - distance values
                - database labels
                - debug/retrieval wording
        """

        if not memories:
            return ""

        lines: List[str] = []

        lines.append(
            "Conversation memory context:\n"
            "Use these prior conversation details only when they are relevant to the user's current question. "
            "Do not mention that these came from memory retrieval."
        )

        for memory in memories:
            question = self._preview_text(
                str(memory.get("question") or ""),
                self.max_question_chars,
            )
            answer = self._preview_text(
                str(memory.get("answer") or ""),
                self.max_answer_chars,
            )

            question = self._clean_memory_text(question)
            answer = self._clean_memory_text(answer)

            if not question and not answer:
                continue

            if question and answer:
                lines.append(
                    f"- The user previously asked: {question}\n"
                    f"  The assistant previously answered: {answer}"
                )
            elif question:
                lines.append(
                    f"- The user previously said or asked: {question}"
                )
            elif answer:
                lines.append(
                    f"- A previous assistant response was: {answer}"
                )

        rendered = "\n".join(lines).strip()

        if len(rendered) > self.max_memory_chars:
            rendered = rendered[: self.max_memory_chars - 3].rstrip() + "..."

        return rendered

    @staticmethod
    def _clean_memory_text(value: str) -> str:
        """
        Remove prompt/debug artifacts from memory text before sending it to the LLM.
        """

        import re

        text_value = (value or "").strip()

        if not text_value:
            return ""

        debug_markers = [
            "Prior question:",
            "Prior answer summary:",
            "Memory source:",
            "Similarity:",
            "RETRIEVED DOCUMENT CONTEXT:",
            "--- CONTEXT END ---",
            "QUESTION:",
            "INSTRUCTIONS:",
            "MEMORY RECALL OVERRIDE:",
            "FINAL ANSWER:",
            "[AIModelsService]",
            "RAW MODEL OUTPUT START",
            "RAW MODEL OUTPUT END",
        ]

        for marker in debug_markers:
            text_value = text_value.replace(marker, "")

        text_value = re.sub(r"\s+", " ", text_value).strip()

        return text_value

    def _embed_question(
        self,
        *,
        question: str,
        request_id: Optional[str],
    ) -> List[float]:
        """
        Produce an embedding for the current question using the same embedder
        used by the RAG pipeline.
        """

        text_value = (question or "").strip()

        if not text_value:
            return []

        try:
            from modules.ai.search_pathway.rag_core.embedder import DBConfiguredEmbedder

            embedder = getattr(self, "embedder", None)

            if embedder is None:
                embedder = DBConfiguredEmbedder()
                self.embedder = embedder

            embedding = embedder.embed_query(
                text_value,
                request_id=request_id,
            )

            normalized = self._normalize_embedding(embedding)

            if normalized:
                return normalized

            warning_id(
                "[ConversationMemoryRetrievalService] DBConfiguredEmbedder returned empty embedding.",
                request_id,
            )
            return []

        except Exception as exc:
            warning_id(
                f"[ConversationMemoryRetrievalService] DBConfiguredEmbedder failed: {exc}",
                request_id,
                exc_info=True,
            )
            return []

    @staticmethod
    def _normalize_embedding(value: Any) -> List[float]:
        """
        Normalize different embedding return shapes into List[float].
        """

        if value is None:
            return []

        if isinstance(value, dict):
            for key in ("embedding", "vector", "query_embedding", "data"):
                candidate = value.get(key)

                normalized = ConversationMemoryRetrievalService._normalize_embedding(
                    candidate
                )

                if normalized:
                    return normalized

            return []

        if isinstance(value, tuple) and value:
            return ConversationMemoryRetrievalService._normalize_embedding(value[0])

        if isinstance(value, list):
            output: List[float] = []

            for item in value:
                try:
                    output.append(float(item))
                except Exception:
                    return []

            return output

        return []

    @staticmethod
    def _vector_literal(embedding: List[float]) -> str:
        """
        Convert [0.1, 0.2] into pgvector literal '[0.1,0.2]'.
        """

        safe_values: List[str] = []

        for value in embedding:
            safe_values.append(str(float(value)))

        return "[" + ",".join(safe_values) + "]"

    def _fallback_recent_qanda_memories(
        self,
        *,
        session: Session,
        user_id: str,
        exclude_request_id: Optional[str],
        top_k: int,
        request_id: Optional[str],
    ) -> List[Dict[str, Any]]:
        """
        Fallback when pgvector SQL fails.

        This is not semantic. It simply returns recent QandA rows so the chat
        still gets some long-term memory instead of none.
        """

        try:
            query = session.query(QandA).filter(QandA.user_id == str(user_id))

            if exclude_request_id:
                query = query.filter(QandA.request_id != str(exclude_request_id))

            rows = (
                query
                .order_by(QandA.timestamp.desc())
                .limit(int(top_k or self.top_k))
                .all()
            )

            memories: List[Dict[str, Any]] = []

            for row in rows:
                memories.append(
                    {
                        "id": str(getattr(row, "id", "")),
                        "user_id": getattr(row, "user_id", None),
                        "question": getattr(row, "question", "") or "",
                        "answer": getattr(row, "answer", "") or "",
                        "timestamp": getattr(row, "timestamp", None),
                        "request_id": getattr(row, "request_id", None),
                        "match_source": "fallback_recent_qanda",
                        "distance": None,
                        "similarity": None,
                    }
                )

            debug_id(
                "[ConversationMemoryRetrievalService] Fallback recent memory loaded "
                f"count={len(memories)}",
                request_id,
            )

            return memories

        except Exception as exc:
            warning_id(
                f"[ConversationMemoryRetrievalService] Fallback memory retrieval failed: {exc}",
                request_id,
                exc_info=True,
            )
            return []

    @staticmethod
    def _preview_text(value: str, max_chars: int) -> str:
        text_value = (value or "").strip()

        if not text_value:
            return ""

        safe_max = int(max_chars or 500)

        if len(text_value) <= safe_max:
            return text_value

        return text_value[: safe_max - 3].rstrip() + "..."