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
    with_request_id,
    get_request_id,
)
from plugins.ai_modules.ai_models import ModelsConfig
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
            debug_id(
                f"[AIModelImage] Using cached local model '{model_name}'",
                rid,
            )
            return cls._model_cache[model_name]

        debug_id(
            f"[AIModelImage] Loading local image model '{model_name}'",
            rid,
        )

        model = ModelsConfig.load_image_model(model_name)

        if model is None:
            raise RuntimeError(
                f"load_image_model returned None for '{model_name}'"
            )

        cls._model_cache[model_name] = model

        info_id(
            f"[AIModelImage] Loaded local image model '{model_name}'",
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

            if not vec:
                raise RuntimeError("GPU image embedding returned empty vector")

            return vec

        # LOCAL BACKEND
        model = cls._load_local_model(model_name, request_id=rid)

        if not hasattr(model, "get_image_embedding"):
            raise RuntimeError(
                f"Local model '{model_name}' does not support image embeddings"
            )

        from PIL import Image

        img = Image.open(resolved_path).convert("RGB")
        vec = model.get_image_embedding(img)

        if not vec:
            raise RuntimeError("Local image embedding returned empty vector")

        return vec

    # ----------------------------------------------------
    # IMAGE PROCESSING
    # ----------------------------------------------------
    @classmethod
    @with_request_id
    def process_image(cls, image_path: str, request_id=None) -> Any:

        resolved_path = cls._resolve_image_path(image_path)

        model_name, model_info, backend = cls._resolve_model_or_raise(
            request_id=request_id,
        )

        if backend == "gpu_service":
            gpu = cls._get_gpu_adapter()
            return gpu.process_image(
                image_path=resolved_path,
                model=model_name,
            )

        model = cls._load_local_model(model_name, request_id=request_id)
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
        request_id=None,
    ) -> Dict[str, Any]:

        resolved_a = cls._resolve_image_path(image_a)
        resolved_b = cls._resolve_image_path(image_b)

        model_name, model_info, backend = cls._resolve_model_or_raise(
            request_id=request_id,
        )

        if backend == "gpu_service":
            gpu = cls._get_gpu_adapter()
            return gpu.compare_images(
                image_a=resolved_a,
                image_b=resolved_b,
                model=model_name,
            )

        model = cls._load_local_model(model_name, request_id=request_id)
        return model.compare_images(resolved_a, resolved_b)

    # ----------------------------------------------------
    # IMAGE DESCRIPTION
    # ----------------------------------------------------
    @classmethod
    @with_request_id
    def generate_description(
        cls,
        image_path: str,
        request_id=None,
    ) -> str:

        resolved_path = cls._resolve_image_path(image_path)

        model_name, model_info, backend = cls._resolve_model_or_raise(
            request_id=request_id,
        )

        if backend == "gpu_service":
            gpu = cls._get_gpu_adapter()
            return gpu.generate_description(
                image_path=resolved_path,
                model=model_name,
            )

        model = cls._load_local_model(model_name, request_id=request_id)
        return model.generate_description(resolved_path)