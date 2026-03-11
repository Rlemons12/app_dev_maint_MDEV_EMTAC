"""
AIModelsEmbeddingService - Unified Embedding Model Facade for EMTAC

- Registry-driven (ModelsConfig)
- Backend-aware (local vs gpu_service)
- Strict behavior (no silent fallback)
"""

from __future__ import annotations

from typing import Optional, List, Sequence

from modules.runtime.gpu_service_adapter_emtac import GPUServerAdapter
from modules.configuration.log_config import (
    debug_id,
    info_id,
    warning_id,
    error_id,
    with_request_id,
)
from plugins.ai_modules.ai_models import ModelsConfig


class AIModelsEmbeddingService:
    """
    High-level facade for embedding model usage.
    Instance-based service (no classmethod usage).

    Public API:
    - get_embeddings(text) -> List[float]
    - get_embeddings_batch(texts) -> List[List[float]]
    """

    def __init__(self):
        self._model_cache: dict = {}
        self._current_model_name: Optional[str] = None
        self._gpu_adapter: Optional[GPUServerAdapter] = None

    # ----------------------------------------------------
    # GPU Adapter (instance-safe)
    # ----------------------------------------------------
    def _get_gpu_adapter(self) -> Optional[GPUServerAdapter]:
        if self._gpu_adapter is None:
            self._gpu_adapter = GPUServerAdapter()

        return self._gpu_adapter if self._gpu_adapter.is_available() else None

    # ----------------------------------------------------
    # DB Lookup
    # ----------------------------------------------------
    @with_request_id
    def get_current_model_name(self, request_id=None) -> str:
        name = ModelsConfig.get_config_value(
            "embedding",
            "CURRENT_MODEL",
            default=None,
        )

        if not name:
            raise RuntimeError(
                "No embedding model configured "
                "(models_config.embedding.CURRENT_MODEL missing)"
            )

        debug_id(
            f"[AIModelsEmbeddingService] Current embedding model from DB: {name}",
            request_id,
        )

        return name

    # ----------------------------------------------------
    # Local Model Loader (instance-safe cache)
    # ----------------------------------------------------
    @with_request_id
    def _load_local_model(self, model_name: str, request_id=None):
        if model_name in self._model_cache:
            debug_id(
                f"[AIModelsEmbeddingService] Using cached local model '{model_name}'",
                request_id,
            )
            return self._model_cache[model_name]

        debug_id(
            f"[AIModelsEmbeddingService] Loading local embedding model '{model_name}'",
            request_id,
        )

        model = ModelsConfig.load_embedding_model(model_name)

        if not model:
            raise RuntimeError(f"Failed to load embedding model '{model_name}'")

        self._model_cache[model_name] = model
        self._current_model_name = model_name

        info_id(
            f"[AIModelsEmbeddingService] Loaded local embedding model '{model_name}'",
            request_id,
        )

        return model

    # ----------------------------------------------------
    # Input validation
    # ----------------------------------------------------
    @staticmethod
    def _validate_single_text(text: str) -> str:
        if not isinstance(text, str) or not text.strip():
            raise RuntimeError("Embedding text must be a non-empty string")
        return text

    @staticmethod
    def _validate_texts(texts: Sequence[str]) -> List[str]:
        if not isinstance(texts, (list, tuple)):
            raise RuntimeError("Embedding texts must be a list or tuple of strings")

        cleaned: List[str] = []
        for idx, text in enumerate(texts):
            if not isinstance(text, str):
                raise RuntimeError(f"Embedding texts[{idx}] must be a string")
            if not text.strip():
                raise RuntimeError(f"Embedding texts[{idx}] must be non-empty")
            cleaned.append(text)

        if not cleaned:
            raise RuntimeError("Embedding texts must contain at least one non-empty string")

        return cleaned

    # ----------------------------------------------------
    # Public API - Single
    # ----------------------------------------------------
    @with_request_id
    def get_embeddings(self, text: str, request_id=None) -> List[float]:
        """
        Backward-compatible single-text embedding API.
        Internally routes through batch API for consistency.
        """
        text = self._validate_single_text(text)

        vectors = self.get_embeddings_batch(
            [text],
            request_id=request_id,
        )

        if not vectors or not vectors[0]:
            raise RuntimeError("Embedding service returned empty vector")

        return vectors[0]

    # ----------------------------------------------------
    # Public API - Batch
    # ----------------------------------------------------
    @with_request_id
    def get_embeddings_batch(
        self,
        texts: Sequence[str],
        request_id=None,
        batch_size: int = 32,
    ) -> List[List[float]]:
        """
        Generate embeddings for multiple texts using the configured model.
        Strict routing. No fallback.
        """
        texts = self._validate_texts(texts)

        model_name = self.get_current_model_name(request_id=request_id)
        model_info = ModelsConfig.get_current_model_info("embedding")

        if not model_info:
            raise RuntimeError(
                f"Embedding model '{model_name}' is not registered in ModelsConfig"
            )

        backend = (model_info.get("backend") or "local").lower()

        debug_id(
            f"[AIModelsEmbeddingService] Batch embedding request | "
            f"model={model_name} backend={backend} texts={len(texts)} batch_size={batch_size}",
            request_id,
        )

        # -------------------------------------------------
        # GPU SERVICE PATH (STRICT)
        # -------------------------------------------------
        if backend == "gpu_service":
            gpu = self._get_gpu_adapter()
            if not gpu:
                raise RuntimeError(
                    "GPU embedding backend configured but GPU service is unavailable"
                )

            gpu_key = model_info.get("gpu_key")
            if not gpu_key:
                raise RuntimeError(
                    f"Embedding model '{model_name}' missing gpu_key"
                )

            debug_id(
                f"[AIModelsEmbeddingService] Using GPU embedding backend "
                f"(gpu_key={gpu_key}, texts={len(texts)})",
                request_id,
            )

            vecs = gpu.embed(
                texts=list(texts),
                gpu_model=gpu_key,
                batch_size=batch_size,
            )

            if not vecs:
                raise RuntimeError("GPU embedding service returned no vectors")

            if len(vecs) != len(texts):
                warning_id(
                    f"[AIModelsEmbeddingService] GPU vector count mismatch | "
                    f"requested={len(texts)} returned={len(vecs)}",
                    request_id,
                )
            else:
                debug_id(
                    f"[AIModelsEmbeddingService] GPU batch embeddings returned {len(vecs)} vectors",
                    request_id,
                )

            return vecs

        # -------------------------------------------------
        # LOCAL PATH
        # -------------------------------------------------
        model = self._load_local_model(model_name, request_id=request_id)

        debug_id(
            f"[AIModelsEmbeddingService] Using local embedding backend "
            f"(model={model_name}, texts={len(texts)})",
            request_id,
        )

        if hasattr(model, "encode"):
            vecs = model.encode(list(texts))
        elif hasattr(model, "get_embeddings_batch"):
            vecs = model.get_embeddings_batch(list(texts))
        elif hasattr(model, "get_embeddings"):
            # Fallback only within local model interface compatibility,
            # not across different backends.
            vecs = [model.get_embeddings(text) for text in texts]
        else:
            raise RuntimeError(
                f"Local embedding model '{model_name}' has no supported embedding method"
            )

        if vecs is None:
            raise RuntimeError("Local embedding model returned no vectors")

        try:
            vecs = list(vecs)
        except Exception as exc:
            raise RuntimeError(
                f"Local embedding model returned non-iterable vectors: {exc}"
            ) from exc

        if not vecs:
            raise RuntimeError("Local embedding model returned empty vectors")

        if len(vecs) != len(texts):
            warning_id(
                f"[AIModelsEmbeddingService] Local vector count mismatch | "
                f"requested={len(texts)} returned={len(vecs)}",
                request_id,
            )
        else:
            debug_id(
                f"[AIModelsEmbeddingService] Local batch embeddings returned {len(vecs)} vectors",
                request_id,
            )

        return vecs