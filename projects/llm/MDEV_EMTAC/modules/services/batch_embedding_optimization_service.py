from __future__ import annotations

from typing import List, Optional, Any

from modules.configuration.log_config import (
    debug_id,
    info_id,
    warning_id,
    error_id,
    with_request_id,
    get_request_id,
)


class BatchEmbeddingOptimizationService:
    """
    Optimizes embedding generation by batching calls to the embedding model.

    Behavior:
    - Prefers embedding_model_service.get_embeddings_batch(texts, ...)
    - Supports legacy embed_batch(texts, ...) if present
    - Falls back to per-item embedding only when batch truly fails

    HARD RULES:
    - No session creation
    - No commit/rollback
    """

    @staticmethod
    def _normalize_vectors(vectors: Any) -> List[List[float]]:
        """
        Normalize possible return shapes into List[List[float]].

        Supported:
        - [[...], [...]]
        - [...]  -> treated as a single vector
        """
        if vectors is None:
            return []

        if not isinstance(vectors, list):
            return []

        if not vectors:
            return []

        # Single vector shape: [0.1, 0.2, ...]
        if all(isinstance(x, (int, float)) for x in vectors):
            return [vectors]

        # Multi-vector shape: [[...], [...]]
        if all(isinstance(v, list) for v in vectors):
            cleaned: List[List[float]] = []
            for vec in vectors:
                if not all(isinstance(x, (int, float)) for x in vec):
                    return []
                cleaned.append(vec)
            return cleaned

        return []

    @with_request_id
    def embed_and_store(
        self,
        *,
        session,
        chunks: List[Any],  # ORM chunk/document rows
        embedding_model_service,
        document_embedding_service,
        complete_document_id: int,
        batch_size: int = 16,
        request_id: Optional[str] = None,
    ) -> int:
        rid = request_id or get_request_id()

        if not chunks:
            debug_id("[BATCH EMBED] no chunks provided", rid)
            return 0

        texts: List[str] = []
        ids: List[int] = []

        for ch in chunks:
            content = getattr(ch, "content", None)
            if not content or not isinstance(content, str) or not content.strip():
                continue

            chunk_id = getattr(ch, "id", None)
            if chunk_id is None:
                continue

            ids.append(int(chunk_id))
            texts.append(content)

        if not texts:
            debug_id("[BATCH EMBED] no valid chunk text found", rid)
            return 0

        info_id(f"[BATCH EMBED] chunks={len(texts)} batch_size={batch_size}", rid)

        # -------------------------------------------------
        # Resolve model name once
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

            vectors: List[List[float]] = []

            # -------------------------------------------------
            # Preferred batch API: get_embeddings_batch
            # -------------------------------------------------
            get_embeddings_batch = getattr(
                embedding_model_service,
                "get_embeddings_batch",
                None,
            )

            if callable(get_embeddings_batch):
                try:
                    debug_id(
                        f"[BATCH EMBED] attempting get_embeddings_batch | batch_len={len(batch_texts)}",
                        rid,
                    )

                    raw_vectors = get_embeddings_batch(
                        batch_texts,
                        request_id=rid,
                        batch_size=batch_size,
                    )
                    vectors = self._normalize_vectors(raw_vectors)

                    debug_id(
                        f"[BATCH EMBED] get_embeddings_batch returned vectors={len(vectors)}",
                        rid,
                    )

                except Exception as e:
                    warning_id(
                        f"[BATCH EMBED] get_embeddings_batch failed: {e}",
                        rid,
                    )
                    vectors = []

            # -------------------------------------------------
            # Legacy batch API support: embed_batch
            # -------------------------------------------------
            if not vectors:
                embed_batch = getattr(embedding_model_service, "embed_batch", None)

                if callable(embed_batch):
                    try:
                        debug_id(
                            f"[BATCH EMBED] attempting legacy embed_batch | batch_len={len(batch_texts)}",
                            rid,
                        )

                        raw_vectors = embed_batch(
                            batch_texts,
                            request_id=rid,
                        )
                        vectors = self._normalize_vectors(raw_vectors)

                        debug_id(
                            f"[BATCH EMBED] embed_batch returned vectors={len(vectors)}",
                            rid,
                        )

                    except Exception as e:
                        warning_id(
                            f"[BATCH EMBED] legacy embed_batch failed: {e}",
                            rid,
                        )
                        vectors = []

            # -------------------------------------------------
            # Validate batch result
            # -------------------------------------------------
            if vectors and len(vectors) != len(batch_ids):
                warning_id(
                    f"[BATCH EMBED] vector count mismatch | expected={len(batch_ids)} returned={len(vectors)}",
                    rid,
                )
                vectors = []

            # -------------------------------------------------
            # Fallback to per-item embedding
            # -------------------------------------------------
            if not vectors:
                warning_id(
                    "[BATCH EMBED] falling back to per-item embedding",
                    rid,
                )

                vectors = []
                for text in batch_texts:
                    try:
                        vec = embedding_model_service.get_embeddings(
                            text,
                            request_id=rid,
                        )
                        if vec:
                            vectors.append(vec)
                        else:
                            vectors.append([])
                    except Exception as e:
                        warning_id(
                            f"[BATCH EMBED] per-item embedding failed: {e}",
                            rid,
                        )
                        vectors.append([])

            # -------------------------------------------------
            # Store embeddings
            # -------------------------------------------------
            stored_this_batch = 0

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
                stored_this_batch += 1

            debug_id(
                f"[BATCH EMBED] batch stored={stored_this_batch} running_total={total}",
                rid,
            )

        debug_id(f"[BATCH EMBED] stored embeddings={total}", rid)

        return total