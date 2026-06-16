"""
AIModelsEmbeddingService - Unified Embedding Model Facade for EMTAC

- Registry-driven (ModelsConfig)
- Backend-aware (local vs gpu_service)
- Strict behavior (no silent fallback)
- Normalizes vectors for PostgreSQL pgvector storage
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

from modules.runtime.gpu_service_adapter_emtac import GPUServerAdapter
from modules.configuration.log_config import (
    debug_id,
    info_id,
    warning_id,
    error_id,
    with_request_id,
)
from modules.ai.config.models_config import ModelsConfig


class AIModelsEmbeddingService:
    """
    High-level facade for embedding model usage.

    This service is responsible for generating embeddings only.

    It should NOT:
        - Create QandA rows
        - Update QandA rows
        - Own database transactions
        - Commit or rollback application data

    Public API:
        - get_embeddings(text) -> List[float]
        - get_embeddings_batch(texts) -> List[List[float]]
        - get_current_model_details() -> Dict[str, Any]
    """

    def __init__(self):
        self._model_cache: dict = {}
        self._current_model_name: Optional[str] = None
        self._gpu_adapter: Optional[GPUServerAdapter] = None

    # ----------------------------------------------------
    # GPU Adapter
    # ----------------------------------------------------

    def _get_gpu_adapter(self) -> Optional[GPUServerAdapter]:
        """
        Lazily creates and returns the GPU adapter if the GPU service is available.
        """

        if self._gpu_adapter is None:
            self._gpu_adapter = GPUServerAdapter()

        return self._gpu_adapter if self._gpu_adapter.is_available() else None

    # ----------------------------------------------------
    # DB / Config Lookup
    # ----------------------------------------------------

    @with_request_id
    def get_current_model_name(self, request_id=None) -> str:
        """
        Gets the currently configured embedding model name from ModelsConfig.
        """

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

    @with_request_id
    def get_current_model_details(self, request_id=None) -> Dict[str, Any]:
        """
        Returns useful metadata about the active embedding model.

        This is helpful when another service wants to store:
            - embedding_model
            - embedding_backend
            - embedding_dimension
        """

        model_name = self.get_current_model_name(request_id=request_id)
        model_info = ModelsConfig.get_current_model_info("embedding")

        if not model_info:
            raise RuntimeError(
                f"Embedding model '{model_name}' is not registered in ModelsConfig"
            )

        backend = (model_info.get("backend") or "local").lower()
        expected_dimension = self._resolve_expected_dimension(model_info)

        details = {
            "model_name": model_name,
            "backend": backend,
            "expected_dimension": expected_dimension,
            "model_info": model_info,
        }

        debug_id(
            "[AIModelsEmbeddingService] Current model details "
            f"model={model_name} backend={backend} "
            f"expected_dimension={expected_dimension}",
            request_id,
        )

        return details

    @staticmethod
    def _resolve_expected_dimension(model_info: Dict[str, Any]) -> Optional[int]:
        """
        Attempts to resolve the configured embedding dimension.

        Supported config keys:
            - dimension
            - dimensions
            - embedding_dimension
            - vector_size
            - embedding_size

        Returns None if the dimension is not configured.
        """

        if not isinstance(model_info, dict):
            return None

        for key in (
                "dimension",
                "dim",
                "dimensions",
                "embedding_dimension",
                "vector_size",
                "embedding_size",
        ):
            value = model_info.get(key)

            if value is None:
                continue

            try:
                dimension = int(value)
            except (TypeError, ValueError):
                continue

            if dimension > 0:
                return dimension

        return None

    # ----------------------------------------------------
    # Local Model Loader
    # ----------------------------------------------------

    @with_request_id
    def _load_local_model(self, model_name: str, request_id=None):
        """
        Loads and caches a local embedding model.
        """

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
    # Input Validation
    # ----------------------------------------------------

    @staticmethod
    def _validate_single_text(text: str) -> str:
        """
        Validates one text input for embedding generation.
        """

        if not isinstance(text, str) or not text.strip():
            raise RuntimeError("Embedding text must be a non-empty string")

        return text.strip()

    @staticmethod
    def _validate_texts(texts: Sequence[str]) -> List[str]:
        """
        Validates multiple text inputs for embedding generation.
        """

        if not isinstance(texts, (list, tuple)):
            raise RuntimeError("Embedding texts must be a list or tuple of strings")

        cleaned: List[str] = []

        for idx, text in enumerate(texts):
            if not isinstance(text, str):
                raise RuntimeError(f"Embedding texts[{idx}] must be a string")

            text = text.strip()

            if not text:
                raise RuntimeError(f"Embedding texts[{idx}] must be non-empty")

            cleaned.append(text)

        if not cleaned:
            raise RuntimeError("Embedding texts must contain at least one non-empty string")

        return cleaned

    # ----------------------------------------------------
    # Vector Normalization
    # ----------------------------------------------------

    @staticmethod
    def _coerce_vector_to_float_list(vector: Any, vector_index: int) -> List[float]:
        """
        Converts one vector into a plain List[float].

        This is important because local models often return:
            - numpy.ndarray
            - torch.Tensor
            - list of numpy.float32
            - tuple

        pgvector/SQLAlchemy works best with a clean Python list of floats.
        """

        if vector is None:
            raise RuntimeError(f"Embedding vector[{vector_index}] is None")

        if hasattr(vector, "detach"):
            vector = vector.detach()

        if hasattr(vector, "cpu"):
            vector = vector.cpu()

        if hasattr(vector, "tolist"):
            vector = vector.tolist()

        if isinstance(vector, tuple):
            vector = list(vector)

        if not isinstance(vector, list):
            raise RuntimeError(
                f"Embedding vector[{vector_index}] must be list-like, "
                f"got {type(vector).__name__}"
            )

        if not vector:
            raise RuntimeError(f"Embedding vector[{vector_index}] is empty")

        cleaned: List[float] = []

        for value_index, value in enumerate(vector):
            try:
                cleaned.append(float(value))
            except (TypeError, ValueError) as exc:
                raise RuntimeError(
                    f"Embedding vector[{vector_index}][{value_index}] "
                    f"cannot be converted to float: {value!r}"
                ) from exc

        return cleaned

    @classmethod
    def _normalize_vectors(
        cls,
        *,
        raw_vectors: Any,
        expected_count: int,
        expected_dimension: Optional[int],
        source_label: str,
        request_id=None,
    ) -> List[List[float]]:
        """
        Normalizes raw embedding output into List[List[float]].

        Also validates:
            - vector count
            - vector dimension, if configured
        """

        if raw_vectors is None:
            raise RuntimeError(f"{source_label} embedding model returned no vectors")

        try:
            vectors_iterable = list(raw_vectors)
        except Exception as exc:
            raise RuntimeError(
                f"{source_label} embedding model returned non-iterable vectors: {exc}"
            ) from exc

        if not vectors_iterable:
            raise RuntimeError(f"{source_label} embedding model returned empty vectors")

        if len(vectors_iterable) != expected_count:
            warning_id(
                f"[AIModelsEmbeddingService] {source_label} vector count mismatch | "
                f"requested={expected_count} returned={len(vectors_iterable)}",
                request_id,
            )

        normalized_vectors: List[List[float]] = []

        for idx, vector in enumerate(vectors_iterable):
            normalized = cls._coerce_vector_to_float_list(vector, idx)

            if expected_dimension is not None and len(normalized) != expected_dimension:
                raise RuntimeError(
                    f"{source_label} embedding dimension mismatch for vector[{idx}]. "
                    f"expected={expected_dimension} actual={len(normalized)}"
                )

            normalized_vectors.append(normalized)

        return normalized_vectors

    # ----------------------------------------------------
    # Public API - Single
    # ----------------------------------------------------

    @with_request_id
    def get_embeddings(self, text: str, request_id=None) -> List[float]:
        """
        Backward-compatible single-text embedding API.

        Internally routes through batch API for consistent behavior.
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
        Generates embeddings for multiple texts using the configured model.

        Strict behavior:
            - No silent backend fallback
            - GPU model must use GPU service if configured
            - Local model must load locally if configured

        Returns:
            List[List[float]]
        """

        texts = self._validate_texts(texts)

        model_name = self.get_current_model_name(request_id=request_id)
        model_info = ModelsConfig.get_current_model_info("embedding")

        if not model_info:
            raise RuntimeError(
                f"Embedding model '{model_name}' is not registered in ModelsConfig"
            )

        backend = (model_info.get("backend") or "local").lower()
        expected_dimension = self._resolve_expected_dimension(model_info)

        debug_id(
            f"[AIModelsEmbeddingService] Batch embedding request | "
            f"model={model_name} backend={backend} texts={len(texts)} "
            f"batch_size={batch_size} expected_dimension={expected_dimension}",
            request_id,
        )

        if expected_dimension is None:
            warning_id(
                "[AIModelsEmbeddingService] No expected embedding dimension configured. "
                "Dimension validation will be skipped. For qanda Vector(1536), "
                "the configured embedding model should declare dimension=1536.",
                request_id,
            )

        # -------------------------------------------------
        # GPU SERVICE PATH
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

            raw_vectors = gpu.embed(
                texts=list(texts),
                gpu_model=gpu_key,
                batch_size=batch_size,
            )

            vectors = self._normalize_vectors(
                raw_vectors=raw_vectors,
                expected_count=len(texts),
                expected_dimension=expected_dimension,
                source_label="GPU",
                request_id=request_id,
            )

            debug_id(
                f"[AIModelsEmbeddingService] GPU batch embeddings returned "
                f"{len(vectors)} normalized vectors",
                request_id,
            )

            return vectors

        # -------------------------------------------------
        # LOCAL PATH
        # -------------------------------------------------

        if backend != "local":
            raise RuntimeError(
                f"Unsupported embedding backend '{backend}' for model '{model_name}'"
            )

        model = self._load_local_model(
            model_name,
            request_id=request_id,
        )

        debug_id(
            f"[AIModelsEmbeddingService] Using local embedding backend "
            f"(model={model_name}, texts={len(texts)})",
            request_id,
        )

        if hasattr(model, "encode"):
            raw_vectors = self._encode_with_local_model(
                model=model,
                texts=texts,
                batch_size=batch_size,
                request_id=request_id,
            )

        elif hasattr(model, "get_embeddings_batch"):
            raw_vectors = model.get_embeddings_batch(list(texts))

        elif hasattr(model, "get_embeddings"):
            raw_vectors = [model.get_embeddings(text) for text in texts]

        else:
            raise RuntimeError(
                f"Local embedding model '{model_name}' has no supported embedding method"
            )

        vectors = self._normalize_vectors(
            raw_vectors=raw_vectors,
            expected_count=len(texts),
            expected_dimension=expected_dimension,
            source_label="Local",
            request_id=request_id,
        )

        debug_id(
            f"[AIModelsEmbeddingService] Local batch embeddings returned "
            f"{len(vectors)} normalized vectors",
            request_id,
        )

        return vectors

    @staticmethod
    def _encode_with_local_model(
        *,
        model: Any,
        texts: Sequence[str],
        batch_size: int,
        request_id=None,
    ) -> Any:
        """
        Calls model.encode() with sensible options when supported.

        Some embedding models support:
            - batch_size
            - show_progress_bar
            - convert_to_numpy

        Others do not, so this method gracefully retries with a simpler call.
        This is local interface compatibility, not backend fallback.
        """

        try:
            return model.encode(
                list(texts),
                batch_size=batch_size,
                show_progress_bar=False,
                convert_to_numpy=True,
            )
        except TypeError:
            debug_id(
                "[AIModelsEmbeddingService] Local model.encode() did not accept "
                "batch_size/show_progress_bar/convert_to_numpy. Retrying with "
                "basic encode(texts).",
                request_id,
            )
            return model.encode(list(texts))
        except Exception as exc:
            error_id(
                f"[AIModelsEmbeddingService] Local model.encode() failed: {exc}",
                request_id,
                exc_info=True,
            )
            raise