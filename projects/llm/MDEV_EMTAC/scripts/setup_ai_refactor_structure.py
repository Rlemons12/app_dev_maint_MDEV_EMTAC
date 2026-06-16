from __future__ import annotations

from pathlib import Path
from textwrap import dedent


PROJECT_ROOT = Path(r"E:\emtac\projects\llm\MDEV_EMTAC")
OVERWRITE_EXISTING = False


FILE_CONTENTS = {
    "modules/__init__.py": """
    \"\"\"Top-level modules package.\"\"\"
    """,

    "modules/ai/__init__.py": """
    \"\"\"AI package for model loading, adapters, and AI-facing services.\"\"\"
    """,

    "modules/ai/adapters/__init__.py": """
    \"\"\"Adapters for concrete model implementations.\"\"\"

    from .clip_image_adapter import CLIPImageAdapter
    from .no_image_adapter import NoImageAdapter

    __all__ = [
        "CLIPImageAdapter",
        "NoImageAdapter",
    ]
    """,

    "modules/ai/adapters/clip_image_adapter.py": """
    from __future__ import annotations

    import os
    import time
    from typing import Any, Dict, List, Optional

    import torch
    from PIL import Image
    from transformers import CLIPModel, CLIPProcessor

    from modules.configuration.log_config import logger


    class CLIPImageAdapter:
        \"\"\"
        Adapter around a locally stored CLIP model.

        Responsibilities:
        - Load CLIP from local disk
        - Generate image embeddings
        - Compare two images
        - Provide simple metadata/description helpers

        Notes:
        - This is an adapter, not a service.
        - It should not open DB sessions.
        - It should not know about orchestrators.
        \"\"\"

        _model_cache: Dict[str, CLIPModel] = {}
        _processor_cache: Dict[str, CLIPProcessor] = {}

        def __init__(self, model_dir: str):
            if not model_dir:
                raise ValueError("model_dir is required")

            if not os.path.isdir(model_dir):
                raise FileNotFoundError(f"CLIP model directory not found: {model_dir}")

            self.model_dir = model_dir
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.model: Optional[CLIPModel] = None
            self.processor: Optional[CLIPProcessor] = None

            self._load_model()

        def _load_model(self) -> None:
            cache_key = f"{self.model_dir}|{self.device}"

            if cache_key in self._model_cache and cache_key in self._processor_cache:
                self.model = self._model_cache[cache_key]
                self.processor = self._processor_cache[cache_key]
                logger.info("[CLIPImageAdapter] Using cached CLIP model")
                return

            start = time.time()
            logger.info(
                "[CLIPImageAdapter] Loading CLIP model from %s on %s",
                self.model_dir,
                self.device,
            )

            self.processor = CLIPProcessor.from_pretrained(
                self.model_dir,
                local_files_only=True,
            )
            self.model = CLIPModel.from_pretrained(
                self.model_dir,
                local_files_only=True,
            ).to(self.device)

            self._model_cache[cache_key] = self.model
            self._processor_cache[cache_key] = self.processor

            logger.info(
                "[CLIPImageAdapter] CLIP model loaded in %.2fs",
                time.time() - start,
            )

        def get_image_embedding(self, image: Image.Image) -> Optional[List[float]]:
            if self.model is None or self.processor is None:
                raise RuntimeError("CLIP model is not loaded")

            if image is None:
                raise ValueError("image is required")

            try:
                image = image.convert("RGB")
                inputs = self.processor(images=image, return_tensors="pt", padding=True)
                inputs = {k: v.to(self.device) for k, v in inputs.items()}

                with torch.no_grad():
                    features = self.model.get_image_features(**inputs)
                    features = features / features.norm(dim=-1, keepdim=True)

                vector = features.cpu().numpy().flatten().tolist()
                return vector
            except Exception as exc:
                logger.error("[CLIPImageAdapter] Failed to generate image embedding: %s", exc)
                return None

        def process_image(self, image_path: str) -> str:
            image = Image.open(image_path).convert("RGB")
            embedding = self.get_image_embedding(image)
            if not embedding:
                return f"Failed to generate embedding for {image_path}"
            return f"Processed image successfully ({len(embedding)} dims)"

        def compare_images(self, image_a: str, image_b: str) -> Dict[str, Any]:
            import math

            try:
                img_a = Image.open(image_a).convert("RGB")
                img_b = Image.open(image_b).convert("RGB")

                emb_a = self.get_image_embedding(img_a)
                emb_b = self.get_image_embedding(img_b)

                if not emb_a or not emb_b:
                    return {
                        "similarity": 0.0,
                        "error": "Failed to generate one or both embeddings",
                    }

                dot = sum(a * b for a, b in zip(emb_a, emb_b))
                norm_a = math.sqrt(sum(a * a for a in emb_a))
                norm_b = math.sqrt(sum(b * b for b in emb_b))

                similarity = dot / (norm_a * norm_b) if norm_a and norm_b else 0.0

                return {
                    "similarity": float(similarity),
                    "image_a": image_a,
                    "image_b": image_b,
                    "model": "CLIPImageAdapter",
                }
            except Exception as exc:
                logger.error("[CLIPImageAdapter] compare_images failed: %s", exc)
                return {
                    "similarity": 0.0,
                    "error": str(exc),
                }

        def generate_description(self, image_path: str) -> str:
            try:
                image = Image.open(image_path).convert("RGB")
                width, height = image.size
                return f"Image {width}x{height}"
            except Exception as exc:
                logger.error("[CLIPImageAdapter] generate_description failed: %s", exc)
                return f"Error: {exc}"

        def get_model_info(self) -> Dict[str, Any]:
            return {
                "adapter": self.__class__.__name__,
                "model_dir": self.model_dir,
                "device": str(self.device),
                "loaded": self.model is not None and self.processor is not None,
            }
    """,

    "modules/ai/adapters/no_image_adapter.py": """
    from __future__ import annotations

    from typing import Any, Dict, List, Optional


    class NoImageAdapter:
        \"\"\"Disabled image adapter.\"\"\"

        def get_image_embedding(self, image) -> Optional[List[float]]:
            return None

        def process_image(self, image_path: str) -> str:
            return "Image processing disabled."

        def compare_images(self, image_a: str, image_b: str) -> Dict[str, Any]:
            return {
                "similarity": 0.0,
                "error": "Image comparison disabled.",
            }

        def generate_description(self, image_path: str) -> str:
            return "Image description disabled."

        def get_model_info(self) -> Dict[str, Any]:
            return {
                "adapter": self.__class__.__name__,
                "loaded": True,
                "enabled": False,
            }
    """,

    "modules/ai/loaders/__init__.py": """
    \"\"\"Loader helpers for AI, embedding, and image models.\"\"\"

    from .image_model_loader import ImageModelLoader
    from .ai_model_loader import AIModelLoader
    from .embedding_model_loader import EmbeddingModelLoader

    __all__ = [
        "ImageModelLoader",
        "AIModelLoader",
        "EmbeddingModelLoader",
    ]
    """,

    "modules/ai/loaders/image_model_loader.py": """
    from __future__ import annotations

    import os
    from typing import Any

    from modules.ai.adapters.clip_image_adapter import CLIPImageAdapter
    from modules.ai.adapters.no_image_adapter import NoImageAdapter
    from modules.ai.config.model_registry import ModelRegistry
    from modules.configuration.log_config import logger


    class ImageModelLoader:
        \"\"\"
        Loads image model adapters based on registry/config values.

        This should become the single place where image model instances are created.
        \"\"\"

        @classmethod
        def load(cls, model_name: str | None = None) -> Any:
            resolved_name = model_name or ModelRegistry.get_current_model_name("image")

            if resolved_name == "CLIPModelHandler":
                model_dir = ModelRegistry.get_model_path("image", resolved_name)
                logger.info("[ImageModelLoader] Loading CLIP adapter from %s", model_dir)
                return CLIPImageAdapter(model_dir=model_dir)

            if resolved_name in ("NoImageModel", None, ""):
                logger.info("[ImageModelLoader] Using NoImageAdapter")
                return NoImageAdapter()

            raise RuntimeError(f"Unsupported image model: {resolved_name}")
    """,

    "modules/ai/loaders/ai_model_loader.py": """
    from __future__ import annotations

    from typing import Any

    from modules.ai.config.model_registry import ModelRegistry
    from modules.configuration.log_config import logger


    class AIModelLoader:
        \"\"\"
        Stub loader for text-generation / AI models.

        This is the future replacement point for legacy ai_modules.py loading behavior.
        \"\"\"

        @classmethod
        def load(cls, model_name: str | None = None) -> Any:
            resolved_name = model_name or ModelRegistry.get_current_model_name("ai")
            logger.info("[AIModelLoader] Requested AI model: %s", resolved_name)

            raise NotImplementedError(
                "AIModelLoader.load() has not been implemented yet. "
                "Move legacy AI model construction here next."
            )
    """,

    "modules/ai/loaders/embedding_model_loader.py": """
    from __future__ import annotations

    from typing import Any

    from modules.ai.config.model_registry import ModelRegistry
    from modules.configuration.log_config import logger


    class EmbeddingModelLoader:
        \"\"\"
        Stub loader for embedding models.

        This is the future replacement point for legacy embedding loading behavior.
        \"\"\"

        @classmethod
        def load(cls, model_name: str | None = None) -> Any:
            resolved_name = model_name or ModelRegistry.get_current_model_name("embedding")
            logger.info("[EmbeddingModelLoader] Requested embedding model: %s", resolved_name)

            raise NotImplementedError(
                "EmbeddingModelLoader.load() has not been implemented yet. "
                "Move legacy embedding model construction here next."
            )
    """,

    "modules/ai/config/__init__.py": """
    \"\"\"AI config helpers and registry access.\"\"\"

    from .model_registry import ModelRegistry

    __all__ = ["ModelRegistry"]
    """,

    "modules/ai/config/model_registry.py": """
    from __future__ import annotations

    import os
    from typing import Optional

    from modules.ai.config.models_config import ModelsConfig
    from modules.configuration.log_config import logger


    class ModelRegistry:
        \"\"\"
        Thin wrapper around ModelsConfig.

        Purpose:
        - central place for model name lookup
        - central place for path resolution
        - reduce direct legacy imports across the codebase
        \"\"\"

        @staticmethod
        def get_current_model_name(category: str) -> Optional[str]:
            return ModelsConfig.get_config_value(category, "CURRENT_MODEL", default=None)

        @staticmethod
        def get_execution_backend(category: str, default: str = "local") -> str:
            return ModelsConfig.get_config_value(category, "EXECUTION_BACKEND", default=default)

        @staticmethod
        def get_model_path(category: str, model_name: str) -> str:
            if category == "image" and model_name == "CLIPModelHandler":
                model_dir = os.getenv("MODEL_PATH_CLIP") or os.getenv("MODEL_CLIP_DIR")
                if not model_dir:
                    raise RuntimeError(
                        "MODEL_PATH_CLIP or MODEL_CLIP_DIR must be set for CLIP image loading"
                    )
                return model_dir

            raise RuntimeError(
                f"No path resolver implemented for category='{category}' model='{model_name}'"
            )
    """,

    "modules/ai/services/__init__.py": """
    \"\"\"AI-facing services package.\"\"\"
    """,
}


def validate_project_root(project_root: Path) -> None:
    if not project_root.exists():
        raise FileNotFoundError(f"Project root does not exist: {project_root}")
    if not project_root.is_dir():
        raise NotADirectoryError(f"Project root is not a directory: {project_root}")


def write_file(path: Path, content: str, overwrite: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    if path.exists() and not overwrite:
        print(f"[SKIP] {path}")
        return

    path.write_text(dedent(content).lstrip(), encoding="utf-8")
    print(f"[WRITE] {path}")


def main() -> None:
    validate_project_root(PROJECT_ROOT)

    print("=" * 90)
    print("SETTING UP AI REFACTOR STRUCTURE WITH STARTER BOILERPLATE")
    print("=" * 90)
    print(f"PROJECT ROOT: {PROJECT_ROOT}")
    print(f"OVERWRITE_EXISTING: {OVERWRITE_EXISTING}")
    print("=" * 90)

    for relative_path, content in FILE_CONTENTS.items():
        full_path = PROJECT_ROOT / relative_path
        write_file(full_path, content, overwrite=OVERWRITE_EXISTING)

    print("=" * 90)
    print("DONE")
    print("=" * 90)
    print("Next suggested moves:")
    print("1. Move CLIP loading calls to modules.ai.loaders.image_model_loader")
    print("2. Move plugin image model logic into modules.ai.adapters")
    print("3. Keep legacy files as wrappers temporarily during migration")
    print("=" * 90)


if __name__ == "__main__":
    main()