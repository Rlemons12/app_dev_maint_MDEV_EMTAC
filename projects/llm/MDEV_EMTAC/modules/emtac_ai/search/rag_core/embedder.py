# modules/emtac_ai/search/rag_core/embedder.py

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List, Optional

from modules.configuration.log_config import (
    with_request_id,
    debug_id,
    info_id,
    warning_id,
    error_id,
    get_request_id,
)

from plugins.ai_modules.ai_models import ModelsConfig
from modules.services.ai_models_embedding_service import AIModelsEmbeddingService


# ============================================================
# Base Interface
# ============================================================

class BaseEmbedder(ABC):
    """Abstract base for all embedders."""

    @abstractmethod
    def embed_query(self, text: str, request_id: Optional[str] = None) -> List[float]:
        raise NotImplementedError


# ============================================================
# DBConfiguredEmbedder (Primary)
# ============================================================

class DBConfiguredEmbedder(BaseEmbedder):
    """
    Primary embedder for EMTAC RAG.

    Behavior:
    - Reads CURRENT_MODEL for embeddings from DB
    - Backend-aware:
        * gpu_service → NEVER load local model
        * local       → load + cache local embedding model
    - All embedding calls route through AIModelsEmbeddingService
    """

    def __init__(self):
        rid = get_request_id()

        # Resolve backend & model name
        self.backend = ModelsConfig.get_execution_backend("embedding")
        self.model_name = AIModelsEmbeddingService.get_current_model_name(request_id=rid)

        self._model = None  # only used for local backend

        info_id(
            f"[Embedder] init backend={self.backend} model={self.model_name}",
            rid,
        )

        # --------------------------------------------------
        # LOCAL BACKEND → load model immediately
        # --------------------------------------------------
        if self.backend != "gpu_service":
            try:
                info_id(
                    f"[Embedder] Loading local embedding model '{self.model_name}'",
                    rid,
                )
                self._model = AIModelsEmbeddingService._load_model(request_id=rid)
            except Exception as e:
                error_id(
                    f"[Embedder] Failed to load local embedding model: {e}",
                    rid,
                )
                raise
        else:
            # GPU service path → DO NOT load local model
            info_id(
                f"[Embedder] Using GPU embedding service for '{self.model_name}'",
                rid,
            )

    # -----------------------------
    # Embed a query string → vector
    # -----------------------------
    @with_request_id
    def embed_query(self, text: str, request_id: Optional[str] = None) -> List[float]:
        rid = request_id or get_request_id()

        if not text or not text.strip():
            warning_id("[Embedder] Empty text passed to embed_query()", rid)
            return []

        debug_id(
            f"[Embedder] embed_query backend={self.backend} chars={len(text)}",
            rid,
        )

        # --------------------------------------------------
        # GPU SERVICE PATH
        # --------------------------------------------------
        if self.backend == "gpu_service":
            try:
                vec = AIModelsEmbeddingService.get_embeddings(text, request_id=rid)
                return list(vec) if vec else []
            except Exception as e:
                error_id(f"[Embedder] GPU embedding failed: {e}", rid)
                return []

        # --------------------------------------------------
        # LOCAL EMBEDDING PATH
        # --------------------------------------------------
        if not self._model:
            warning_id(
                "[Embedder] Local embedding model not loaded; returning empty vector",
                rid,
            )
            return []

        if not hasattr(self._model, "get_embeddings"):
            warning_id(
                f"[Embedder] Model {self.model_name} "
                f"does not implement get_embeddings()",
                rid,
            )
            return []

        try:
            vec = self._model.get_embeddings(text)

            # normalize output
            if hasattr(vec, "tolist"):
                vec = vec.tolist()

            return list(vec)

        except Exception as e:
            error_id(f"[Embedder] Local get_embeddings() failed: {e}", rid, exc_info=True)
            return []
