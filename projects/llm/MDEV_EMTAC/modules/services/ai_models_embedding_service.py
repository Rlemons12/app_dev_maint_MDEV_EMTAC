"""
AIModelsEmbeddingService - Unified Embedding Model Facade for EMTAC

- Registry-driven (ModelsConfig)
- Backend-aware (local vs gpu_service)
- Strict behavior (no silent fallback)
"""

from typing import Optional, List

from modules.runtime.gpu_service_adapter_emtac import GPUServerAdapter
from modules.configuration.log_config import (
    debug_id,
    info_id,
    error_id,
    with_request_id,
)
from plugins.ai_modules.ai_models import ModelsConfig


class AIModelsEmbeddingService:
    """
    High-level facade for embedding model usage.
    Instance-based service (no classmethod usage).
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
            raise RuntimeError(
                f"Failed to load embedding model '{model_name}'"
            )

        self._model_cache[model_name] = model
        self._current_model_name = model_name

        info_id(
            f"[AIModelsEmbeddingService] Loaded local embedding model '{model_name}'",
            request_id,
        )

        return model

    # ----------------------------------------------------
    # Public API
    # ----------------------------------------------------
    @with_request_id
    def get_embeddings(self, text: str, request_id=None) -> List[float]:
        """
        Generate embeddings using the configured model.
        Strict routing. No fallback.
        """

        if not isinstance(text, str) or not text.strip():
            raise RuntimeError("Embedding text must be a non-empty string")

        model_name = self.get_current_model_name(request_id=request_id)
        model_info = ModelsConfig.get_current_model_info("embedding")

        if not model_info:
            raise RuntimeError(
                f"Embedding model '{model_name}' is not registered in ModelsConfig"
            )

        backend = (model_info.get("backend") or "local").lower()

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
                f"(gpu_key={gpu_key}, chars={len(text)})",
                request_id,
            )

            vecs = gpu.embed(
                texts=[text],
                gpu_model=gpu_key,
            )

            if not vecs or not vecs[0]:
                raise RuntimeError("GPU embedding service returned empty vector")

            return vecs[0]

        # -------------------------------------------------
        # LOCAL PATH
        # -------------------------------------------------
        model = self._load_local_model(model_name, request_id=request_id)

        debug_id(
            f"[AIModelsEmbeddingService] Using local embedding backend "
            f"(model={model_name}, chars={len(text)})",
            request_id,
        )

        # Support both possible local APIs
        if hasattr(model, "get_embeddings"):
            vec = model.get_embeddings(text)
        elif hasattr(model, "encode"):
            vec = model.encode([text])[0]
        else:
            raise RuntimeError(
                f"Local embedding model '{model_name}' has no supported embedding method"
            )

        if not vec:
            raise RuntimeError("Local embedding model returned empty vector")

        return vec