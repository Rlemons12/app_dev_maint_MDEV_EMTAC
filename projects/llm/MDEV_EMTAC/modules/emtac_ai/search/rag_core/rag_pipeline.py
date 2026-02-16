from __future__ import annotations
from typing import Dict, Any, Optional, List

from modules.configuration.config_env import DatabaseConfig
from modules.configuration.log_config import (
    info_id,
    error_id,
    debug_id,
    warning_id,
    with_request_id,
)

# DB-driven AI component loaders
from .embedder import DBConfiguredEmbedder, BaseEmbedder
from .retriever import PgVectorRetriever
from .context_builder import ContextBuilder
from .answer_generator import DBConfiguredAnswerGenerator, BaseAnswerGenerator
from .document_ui_payload import DocumentUIPayload


class RAGPipeline:
    """
    High-level orchestration for RAG (Retrieval-Augmented Generation).

    Steps:
        1. Embed question using DB-selected embedding model
        2. Retrieve top document chunks using pgvector
        3. Build a merged context string
        4. Aggregate documents from used chunks
        5. Generate answer using DB-selected LLM
        6. Return structured outputs for UI
    """

    def __init__(
        self,
        db_config: Optional[DatabaseConfig] = None,
        embedder: Optional[BaseEmbedder] = None,
        retriever: Optional[PgVectorRetriever] = None,
        context_builder: Optional[ContextBuilder] = None,
        answer_generator: Optional[BaseAnswerGenerator] = None,
    ):
        self.db_config = db_config or DatabaseConfig()

        self.embedder = embedder or DBConfiguredEmbedder()
        self.retriever = retriever or PgVectorRetriever(db_config=self.db_config)
        self.context_builder = context_builder or ContextBuilder()
        self.answer_generator = answer_generator or DBConfiguredAnswerGenerator()

    # ------------------------------------------------------------------
    # MAIN EXECUTION FUNCTION
    # ------------------------------------------------------------------
    @with_request_id
    def run(
            self,
            question: str,
            top_k: int = 5,
            request_id: Optional[str] = None,
            **answer_kwargs: Any,
    ) -> Dict[str, Any]:

        debug_id(f"[RAGPipeline] Start RAG: '{question[:80]}...'", request_id)

        try:
            # -------------------------------
            # 1. EMBEDDING STAGE
            # -------------------------------
            query_embedding = self.embedder.embed_query(
                question,
                request_id=request_id,
            )

            if not query_embedding:
                warning_id(
                    "[RAGPipeline] No embedding produced — stopping RAG.",
                    request_id,
                )
                return {
                    "answer": "Embedding model failed or returned nothing.",
                    "documents": [],
                    "used_chunks": [],
                    "query_embedding": [],
                    "retriever_top_k": top_k,
                }

            # -------------------------------
            # 2. RETRIEVAL STAGE
            # -------------------------------
            retrieved_chunks = self.retriever.retrieve(
                query_embedding=query_embedding,
                top_k=top_k,
                request_id=request_id,
            )

            if not retrieved_chunks:
                warning_id(
                    "[RAGPipeline] No documents retrieved — answering from no context.",
                    request_id,
                )

            # -------------------------------
            # 3. CONTEXT BUILDING
            # -------------------------------
            ctx = self.context_builder.build_context(
                retrieved_chunks=retrieved_chunks,
                request_id=request_id,
            )

            context: str = ctx.get("context", "")
            used_chunks: List[Dict[str, Any]] = ctx.get("used_chunks", [])

            # -------------------------------
            # 4. DOCUMENT UI PAYLOAD
            # -------------------------------
            documents = (
                DocumentUIPayload()
                .aggregate_from_chunks(used_chunks)
                .build()
            )

            debug_id(
                f"[RAGPipeline] Built UI payload with {len(documents)} documents "
                f"from {len(used_chunks)} chunks",
                request_id,
            )

            # -------------------------------
            # 5. ANSWER GENERATION
            # -------------------------------
            answer_result = self.answer_generator.generate_answer(
                question=question,
                context=context,
                request_id=request_id,
                **answer_kwargs,
            )

            answer: str = answer_result.get("answer", "")

            # -------------------------------
            # 6. STRUCTURED RETURN
            # -------------------------------
            return {
                "answer": answer,
                "documents": documents,
                "used_chunks": used_chunks,
                "query_embedding": query_embedding,
                "retriever_top_k": top_k,
            }

        except Exception as e:
            error_id(f"[RAGPipeline] Pipeline failed: {e}", request_id, exc_info=True)
            raise

# --------------------------------------------------------------------
# Global Singleton Instance
# --------------------------------------------------------------------
_default_rag: Optional[RAGPipeline] = None


def get_default_rag() -> RAGPipeline:
    """
    Factory / accessor for the global RAGPipeline instance.
    Ensures that any caller receives a valid object.
    """
    global _default_rag

    if _default_rag is None:
        try:
            db_config = DatabaseConfig()
            _default_rag = RAGPipeline(db_config=db_config)
            info_id("[RAG] Default RAGPipeline initialized")
        except Exception as e:
            error_id(f"[RAG] Failed to construct RAGPipeline: {e}")
            raise

    return _default_rag


__all__ = ["RAGPipeline", "get_default_rag"]
