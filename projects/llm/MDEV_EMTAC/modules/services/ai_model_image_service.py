#modules/services/ai_model_image_service.py
"""
AIModelImageService - Image Model Facade for EMTAC

- Registry-driven (ModelsConfig)
- Backend-aware (local vs gpu_service)
- Strict routing (no silent fallback mismatches)
- Handles:
    • image processing
    • comparison
    • description generation
    • image embeddings
"""

from __future__ import annotations

import os
from typing import Optional, Dict, Any, List, Tuple

from modules.runtime.gpu_service_adapter_emtac import GPUServerAdapter
from modules.configuration.log_config import (
    debug_id,
    info_id,
    warning_id,
    error_id,
    with_request_id,
    get_request_id,
)
from modules.ai.config.models_config import ModelsConfig
from modules.configuration.config import DATABASE_DIR


class AIModelImageService:
    """
    High-level facade for IMAGE model usage.
    """

    CATEGORY = "image"

    _model_cache: Dict[str, Any] = {}
    _gpu_adapter: Optional[GPUServerAdapter] = None

    # ----------------------------------------------------
    # Path Resolver (DB stores relative to DATABASE_DIR)
    # ----------------------------------------------------
    @classmethod
    def _resolve_image_path(cls, image_path: str) -> str:
        if not image_path or not isinstance(image_path, str):
            raise RuntimeError("image_path must be a non-empty string")

        if os.path.isabs(image_path):
            resolved = image_path
        else:
            resolved = os.path.join(DATABASE_DIR, image_path)

        resolved = os.path.normpath(resolved)

        if not os.path.exists(resolved):
            raise FileNotFoundError(
                f"[AIModelImage] Image file not found: {resolved}"
            )

        return resolved

    # ----------------------------------------------------
    # GPU Adapter (lazy singleton)
    # ----------------------------------------------------
    @classmethod
    def _get_gpu_adapter(cls) -> Optional[GPUServerAdapter]:
        if cls._gpu_adapter is None:
            cls._gpu_adapter = GPUServerAdapter()

        return cls._gpu_adapter if cls._gpu_adapter.is_available() else None

    # ----------------------------------------------------
    # Model Resolution
    # ----------------------------------------------------
    @classmethod
    @with_request_id
    def get_current_model_name(cls, request_id: Optional[str] = None) -> str:
        rid = request_id or get_request_id()

        name = ModelsConfig.get_config_value(
            cls.CATEGORY,
            "CURRENT_MODEL",
            default=None,
        )

        if not name:
            raise RuntimeError(
                f"No model configured for category '{cls.CATEGORY}'"
            )

        debug_id(
            f"[AIModelImage] Current image model from DB: {name}",
            rid,
        )

        return name

    @classmethod
    def _resolve_model_or_raise(
        cls,
        request_id: Optional[str],
    ) -> Tuple[str, Dict[str, Any], str]:
        rid = request_id or get_request_id()

        model_name = cls.get_current_model_name(request_id=rid)
        model_info = ModelsConfig.get_current_model_info(cls.CATEGORY)

        if not model_info:
            raise RuntimeError(
                f"Model '{model_name}' not registered under category '{cls.CATEGORY}'"
            )

        backend = (model_info.get("backend") or "local").lower().strip()

        debug_id(
            f"[AIModelImage] Resolved model | "
            f"category={cls.CATEGORY} model={model_name} backend={backend}",
            rid,
        )

        return model_name, model_info, backend

    # ----------------------------------------------------
    # Validation Helpers
    # ----------------------------------------------------
    @classmethod
    def _validate_local_model_interface(
        cls,
        model: Any,
        *,
        model_name: str,
        request_id: Optional[str] = None,
    ) -> Any:
        rid = request_id or get_request_id()

        if model is None:
            raise RuntimeError(
                f"Local image model '{model_name}' resolved to None"
            )

        debug_id(
            f"[AIModelImage] Local model instance type for '{model_name}': "
            f"{type(model).__name__}",
            rid,
        )

        return model

    @classmethod
    def _ensure_supports_method(
        cls,
        model: Any,
        *,
        model_name: str,
        method_name: str,
        request_id: Optional[str] = None,
    ) -> None:
        rid = request_id or get_request_id()

        if not hasattr(model, method_name):
            available = [
                attr for attr in dir(model)
                if not attr.startswith("_")
            ]
            raise RuntimeError(
                f"Local model '{model_name}' does not support '{method_name}'. "
                f"Resolved object type: {type(model).__name__}. "
                f"Available public attrs sample: {available[:20]}"
            )

        debug_id(
            f"[AIModelImage] Verified method '{method_name}' on local model '{model_name}'",
            rid,
        )

    @classmethod
    def _normalize_vector(
        cls,
        vector: Any,
        *,
        source: str,
    ) -> List[float]:
        if vector is None:
            raise RuntimeError(f"{source} returned None")

        if hasattr(vector, "tolist"):
            vector = vector.tolist()

        if isinstance(vector, tuple):
            vector = list(vector)

        if not isinstance(vector, list):
            raise RuntimeError(
                f"{source} returned unsupported vector type: {type(vector).__name__}"
            )

        if not vector:
            raise RuntimeError(f"{source} returned empty vector")

        try:
            normalized = [float(x) for x in vector]
        except Exception as e:
            raise RuntimeError(
                f"{source} returned non-numeric vector values: {e}"
            ) from e

        return normalized

    @classmethod
    def _try_clear_cuda_cache(
        cls,
        request_id: Optional[str] = None,
    ) -> None:
        rid = request_id or get_request_id()

        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                debug_id("[AIModelImage] Cleared CUDA cache before local model load", rid)
        except Exception as e:
            debug_id(f"[AIModelImage] Could not clear CUDA cache: {e}", rid)

    # ----------------------------------------------------
    # Local Loader (cached)
    # ----------------------------------------------------
    @classmethod
    @with_request_id
    def _load_local_model(
        cls,
        model_name: str,
        request_id: Optional[str] = None,
    ):
        rid = request_id or get_request_id()

        if model_name in cls._model_cache:
            cached_model = cls._model_cache[model_name]
            debug_id(
                f"[AIModelImage] Using cached local model '{model_name}' "
                f"type={type(cached_model).__name__}",
                rid,
            )
            return cached_model

        debug_id(
            f"[AIModelImage] Loading local image model '{model_name}'",
            rid,
        )

        cls._try_clear_cuda_cache(request_id=rid)

        try:
            model = ModelsConfig.load_image_model(model_name)
        except Exception as e:
            error_id(
                f"[AIModelImage] load_image_model failed for '{model_name}': {e}",
                rid,
                exc_info=True,
            )
            raise RuntimeError(
                f"Failed to load local image model '{model_name}': {e}"
            ) from e

        model = cls._validate_local_model_interface(
            model,
            model_name=model_name,
            request_id=rid,
        )

        cls._model_cache[model_name] = model

        info_id(
            f"[AIModelImage] Loaded local image model '{model_name}' "
            f"type={type(model).__name__}",
            rid,
        )

        return model

    # ----------------------------------------------------
    # IMAGE EMBEDDING
    # ----------------------------------------------------
    @classmethod
    @with_request_id
    def get_image_embedding(
        cls,
        image_path: str,
        request_id: Optional[str] = None,
    ) -> List[float]:
        rid = request_id or get_request_id()

        resolved_path = cls._resolve_image_path(image_path)

        model_name, model_info, backend = cls._resolve_model_or_raise(
            request_id=rid,
        )

        # GPU BACKEND
        if backend == "gpu_service":
            gpu = cls._get_gpu_adapter()
            if not gpu:
                raise RuntimeError(
                    "GPU image backend configured but GPU service unavailable"
                )

            gpu_key = model_info.get("gpu_key")
            if not gpu_key:
                raise RuntimeError(
                    f"Image model '{model_name}' missing gpu_key configuration"
                )

            vec = gpu.embed_image(
                image_path=resolved_path,
                gpu_model=gpu_key,
            )

            return cls._normalize_vector(
                vec,
                source=f"GPU image embedding ({model_name})",
            )

        # LOCAL BACKEND
        model = cls._load_local_model(model_name, request_id=rid)

        cls._ensure_supports_method(
            model,
            model_name=model_name,
            method_name="get_image_embedding",
            request_id=rid,
        )

        from PIL import Image

        try:
            with Image.open(resolved_path) as pil_img:
                img = pil_img.convert("RGB")
                vec = model.get_image_embedding(img)
        except Exception as e:
            raise RuntimeError(
                f"Local image embedding failed for '{resolved_path}' using model '{model_name}': {e}"
            ) from e

        return cls._normalize_vector(
            vec,
            source=f"Local image embedding ({model_name})",
        )

    # ----------------------------------------------------
    # IMAGE PROCESSING
    # ----------------------------------------------------
    @classmethod
    @with_request_id
    def process_image(
        cls,
        image_path: str,
        request_id: Optional[str] = None,
    ) -> Any:
        rid = request_id or get_request_id()

        resolved_path = cls._resolve_image_path(image_path)

        model_name, model_info, backend = cls._resolve_model_or_raise(
            request_id=rid,
        )

        if backend == "gpu_service":
            gpu = cls._get_gpu_adapter()
            if not gpu:
                raise RuntimeError(
                    "GPU image backend configured but GPU service unavailable"
                )

            return gpu.process_image(
                image_path=resolved_path,
                model=model_name,
            )

        model = cls._load_local_model(model_name, request_id=rid)
        cls._ensure_supports_method(
            model,
            model_name=model_name,
            method_name="process_image",
            request_id=rid,
        )
        return model.process_image(resolved_path)

    # ----------------------------------------------------
    # IMAGE COMPARISON
    # ----------------------------------------------------
    @classmethod
    @with_request_id
    def compare_images(
        cls,
        image_a: str,
        image_b: str,
        request_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        rid = request_id or get_request_id()

        resolved_a = cls._resolve_image_path(image_a)
        resolved_b = cls._resolve_image_path(image_b)

        model_name, model_info, backend = cls._resolve_model_or_raise(
            request_id=rid,
        )

        if backend == "gpu_service":
            gpu = cls._get_gpu_adapter()
            if not gpu:
                raise RuntimeError(
                    "GPU image backend configured but GPU service unavailable"
                )

            return gpu.compare_images(
                image_a=resolved_a,
                image_b=resolved_b,
                model=model_name,
            )

        model = cls._load_local_model(model_name, request_id=rid)
        cls._ensure_supports_method(
            model,
            model_name=model_name,
            method_name="compare_images",
            request_id=rid,
        )
        return model.compare_images(resolved_a, resolved_b)

    # ----------------------------------------------------
    # IMAGE DESCRIPTION
    # ----------------------------------------------------
    @classmethod
    @with_request_id
    def generate_description(
        cls,
        image_path: str,
        request_id: Optional[str] = None,
    ) -> str:
        rid = request_id or get_request_id()

        resolved_path = cls._resolve_image_path(image_path)

        model_name, model_info, backend = cls._resolve_model_or_raise(
            request_id=rid,
        )

        if backend == "gpu_service":
            gpu = cls._get_gpu_adapter()
            if not gpu:
                raise RuntimeError(
                    "GPU image backend configured but GPU service unavailable"
                )

            return gpu.generate_description(
                image_path=resolved_path,
                model=model_name,
            )

        model = cls._load_local_model(model_name, request_id=rid)
        cls._ensure_supports_method(
            model,
            model_name=model_name,
            method_name="generate_description",
            request_id=rid,
        )
        return model.generate_description(resolved_path)

    # ----------------------------------------------------
    # Cache Utility
    # ----------------------------------------------------
    @classmethod
    @with_request_id
    def clear_model_cache(cls, request_id: Optional[str] = None) -> None:
        rid = request_id or get_request_id()

        cls._model_cache.clear()
        info_id("[AIModelImage] Cleared local model cache", rid)

        cls._try_clear_cuda_cache(request_id=rid)