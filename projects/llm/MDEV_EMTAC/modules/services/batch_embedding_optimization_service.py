from __future__ import annotations

from typing import List, Optional, Any

from modules.configuration.log_config import (
    debug_id,
    info_id,
    warning_id,
    with_request_id,
    get_request_id,
)


class BatchEmbeddingOptimizationService:
    """
    Optimizes embedding generation by batching calls to the embedding model.

    Behavior:
    - Uses embedding_model_service.get_embeddings(text)
    - Falls back to per-item embedding if batch not supported

    HARD RULES:
    - No session creation
    - No commit/rollback
    """

    @with_request_id
    def embed_and_store(
        self,
        *,
        session,
        chunks: List[Any],  # ORM Document rows
        embedding_model_service,
        document_embedding_service,
        complete_document_id: int,
        batch_size: int = 16,
        request_id: Optional[str] = None,
    ) -> int:

        rid = request_id or get_request_id()

        if not chunks:
            return 0

        texts: List[str] = []
        ids: List[int] = []

        for ch in chunks:
            content = getattr(ch, "content", None)
            if not content or not content.strip():
                continue

            ids.append(int(ch.id))
            texts.append(content)

        if not texts:
            return 0

        info_id(f"[BATCH EMBED] chunks={len(texts)} batch_size={batch_size}", rid)

        # -------------------------------------------------
        # Resolve model name (NEW CONTRACT — no category)
        # -------------------------------------------------
        model_name = embedding_model_service.get_current_model_name(
            request_id=rid
        )

        total = 0

        # -------------------------------------------------
        # Process in batches
        # -------------------------------------------------
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            batch_ids = ids[i:i + batch_size]

            vectors = []

            # -------------------------------------------------
            # Try batch embedding if supported
            # -------------------------------------------------
            embed_batch = getattr(embedding_model_service, "embed_batch", None)

            if callable(embed_batch):
                try:
                    vectors = embed_batch(batch_texts, request_id=rid)
                except Exception as e:
                    warning_id(
                        f"[BATCH EMBED] batch call failed: {e}",
                        rid,
                    )
                    vectors = []

            # -------------------------------------------------
            # Fallback to per-item embedding
            # -------------------------------------------------
            if not vectors or len(vectors) != len(batch_ids):
                warning_id(
                    "[BATCH EMBED] vector count mismatch; falling back per-item",
                    rid,
                )

                vectors = [
                    embedding_model_service.get_embeddings(
                        text,
                        request_id=rid,
                    )
                    for text in batch_texts
                ]

            # -------------------------------------------------
            # Store embeddings
            # -------------------------------------------------
            for doc_id, vec in zip(batch_ids, vectors):
                if not vec:
                    continue

                document_embedding_service.add_embedding(
                    session=session,
                    document_id=doc_id,
                    model_name=model_name,
                    embedding=vec,
                    metadata={
                        "complete_document_id": complete_document_id,
                        "pipeline": "process_upload",
                    },
                    request_id=rid,
                )

                total += 1

        debug_id(f"[BATCH EMBED] stored embeddings={total}", rid)

        return total