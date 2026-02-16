# plugins/ai_modules/services/ai_model_image_service.py

"""
AIModelImageService - Unified Image Model Facade for EMTAC

This service:
- Reads image model config from models_config
- Routes execution via EXECUTION_BACKEND (local | gpu_service)
- Loads/caches local image models
- Uses GPU service when configured
"""

from typing import Optional, Dict, Any

from modules.runtime.gpu_service_adapter_emtac import GPUServerAdapter
from modules.configuration.log_config import (
    debug_id,
    info_id,
    warning_id,
    error_id,
    with_request_id,
)
from plugins.ai_modules.ai_models import ModelsConfig


class AIModelImageService:
    """
    High-level facade / wrapper for image models.
    """

    _model_cache = {}
    _current_model_name: Optional[str] = None
    _gpu_adapter: Optional[GPUServerAdapter] = None

    # ----------------------------------------------------
    # GPU Adapter (lazy singleton)
    # ----------------------------------------------------
    @classmethod
    def _get_gpu_adapter(cls) -> Optional[GPUServerAdapter]:
        if cls._gpu_adapter is None:
            cls._gpu_adapter = GPUServerAdapter()
        return cls._gpu_adapter if cls._gpu_adapter.is_available() else None

    # ----------------------------------------------------
    # Config helpers
    # ----------------------------------------------------
    @classmethod
    @with_request_id
    def get_current_model_name(cls, request_id: Optional[str] = None) -> str:
        name = ModelsConfig.get_config_value(
            "image", "CURRENT_MODEL", default="NoImageModel"
        )
        debug_id(f"[AIModelImage] Current image model from DB: {name}", request_id)
        return name

    @classmethod
    @with_request_id
    def set_current_model(cls, model_name: str, request_id: Optional[str] = None) -> bool:
        ok = ModelsConfig.set_current_image_model(model_name)
        if ok:
            info_id(
                f"[AIModelImage] Updated CURRENT_MODEL=image → '{model_name}'",
                request_id,
            )
            cls._current_model_name = None
            cls._model_cache.clear()
        else:
            error_id(
                f"[AIModelImage] Failed updating CURRENT_MODEL=image → '{model_name}'",
                request_id,
            )
        return ok

    # ----------------------------------------------------
    # Local model loading / caching
    # ----------------------------------------------------
    @classmethod
    @with_request_id
    def _load_local_model(
        cls,
        model_name: Optional[str] = None,
        force_reload: bool = False,
        request_id: Optional[str] = None,
    ):
        if model_name is None:
            model_name = cls.get_current_model_name(request_id=request_id)

        if not force_reload and model_name in cls._model_cache:
            debug_id(
                f"[AIModelImage] Using cached local model '{model_name}'",
                request_id,
            )
            return cls._model_cache[model_name]

        debug_id(
            f"[AIModelImage] Loading local image model '{model_name}'",
            request_id,
        )

        model = ModelsConfig.load_image_model(model_name)

        if model is None:
            warning_id(
                f"[AIModelImage] load_image_model returned None for '{model_name}'",
                request_id,
            )
            return None

        cls._model_cache[model_name] = model
        cls._current_model_name = model_name

        info_id(
            f"[AIModelImage] Loaded local image model '{model_name}'",
            request_id,
        )
        return model

    # ----------------------------------------------------
    # Public APIs
    # ----------------------------------------------------
    @classmethod
    @with_request_id
    def process_image(cls, image_path: str, request_id: Optional[str] = None) -> Any:
        backend = ModelsConfig.get_execution_backend("image")

        # -------------------------------
        # GPU SERVICE PATH
        # -------------------------------
        if backend == "gpu_service":
            gpu = cls._get_gpu_adapter()
            if gpu:
                model_name = cls.get_current_model_name(request_id=request_id)
                debug_id(
                    f"[AIModelImage] GPU process_image "
                    f"(model={model_name}, path={image_path})",
                    request_id,
                )
                return gpu.process_image(
                    image_path=image_path,
                    model=model_name,
                )

            warning_id(
                "[AIModelImage] GPU service unavailable — falling back to local",
                request_id,
            )

        # -------------------------------
        # LOCAL PATH
        # -------------------------------
        model = cls._load_local_model(request_id=request_id)
        if model is None:
            error_id(
                "[AIModelImage] No local image model available for process_image()",
                request_id,
            )
            return "Image processing unavailable."

        try:
            debug_id(
                f"[AIModelImage] Local process_image(path='{image_path}')",
                request_id,
            )
            return model.process_image(image_path)
        except Exception as e:
            error_id(
                f"[AIModelImage] process_image exception: {e}",
                request_id,
            )
            return "Image processing error."

    @classmethod
    @with_request_id
    def compare_images(
        cls,
        image_a: str,
        image_b: str,
        request_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        backend = ModelsConfig.get_execution_backend("image")

        # -------------------------------
        # GPU SERVICE PATH
        # -------------------------------
        if backend == "gpu_service":
            gpu = cls._get_gpu_adapter()
            if gpu:
                model_name = cls.get_current_model_name(request_id=request_id)
                debug_id(
                    f"[AIModelImage] GPU compare_images "
                    f"(model={model_name}, a={image_a}, b={image_b})",
                    request_id,
                )
                return gpu.compare_images(
                    image_a=image_a,
                    image_b=image_b,
                    model=model_name,
                )

            warning_id(
                "[AIModelImage] GPU service unavailable — falling back to local",
                request_id,
            )

        # -------------------------------
        # LOCAL PATH
        # -------------------------------
        model = cls._load_local_model(request_id=request_id)
        if model is None:
            return {"similarity": 0.0, "error": "Model unavailable"}

        try:
            debug_id(
                f"[AIModelImage] Local compare_images(a='{image_a}', b='{image_b}')",
                request_id,
            )
            result = model.compare_images(image_a, image_b)

            if not isinstance(result, dict):
                warning_id(
                    "[AIModelImage] Non-dict result from compare_images; normalizing",
                    request_id,
                )
                return {"similarity": 0.0, "raw": result}

            return result

        except Exception as e:
            error_id(
                f"[AIModelImage] compare_images exception: {e}",
                request_id,
            )
            return {"similarity": 0.0, "error": str(e)}

    @classmethod
    @with_request_id
    def generate_description(
        cls,
        image_path: str,
        request_id: Optional[str] = None,
    ) -> str:
        backend = ModelsConfig.get_execution_backend("image")

        # -------------------------------
        # GPU SERVICE PATH
        # -------------------------------
        if backend == "gpu_service":
            gpu = cls._get_gpu_adapter()
            if gpu:
                model_name = cls.get_current_model_name(request_id=request_id)
                debug_id(
                    f"[AIModelImage] GPU generate_description "
                    f"(model={model_name}, path={image_path})",
                    request_id,
                )
                return gpu.generate_description(
                    image_path=image_path,
                    model=model_name,
                )

            warning_id(
                "[AIModelImage] GPU service unavailable — falling back to local",
                request_id,
            )

        # -------------------------------
        # LOCAL PATH
        # -------------------------------
        model = cls._load_local_model(request_id=request_id)
        if model is None:
            return "Image description unavailable."

        try:
            debug_id(
                f"[AIModelImage] Local generate_description(path='{image_path}')",
                request_id,
            )
            return model.generate_description(image_path)
        except Exception as e:
            error_id(
                f"[AIModelImage] generate_description exception: {e}",
                request_id,
            )
            return "Image description error."
