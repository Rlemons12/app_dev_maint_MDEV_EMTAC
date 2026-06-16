# modules/ai/image/services/image_handler_service.py

"""Image handler service extracted from legacy plugins/image_modules/image_handler.py."""


from __future__ import annotations

import imghdr
import os
from typing import Any, Dict, Optional

from PIL import Image, UnidentifiedImageError

from modules.ai.image.models import CLIPModelHandler, NoImageModel
from modules.configuration.log_config import logger


class ImageHandler:
    """
    Image handler service with lazy model loading.

    Key improvements:
    - lazy-loads handlers instead of instantiating all handlers at startup
    - uses a single cache for handler reuse
    - normalizes model aliases consistently
    - reduces GPU pressure by avoiding unnecessary CLIP construction
    - keeps backward compatibility for legacy model names
    """

    MODEL_ALIASES = {
        "clip": "CLIPModelHandler",
        "CLIPModelHandler": "CLIPModelHandler",
        "no_model": "NoImageModel",
        "NoImageModel": "NoImageModel",
        None: "NoImageModel",
        "": "NoImageModel",
    }

    def __init__(self):
        # lazy imports to avoid startup coupling
        from modules.emtacdb.emtacdb_fts import Session, load_image_model_config_from_db

        self.Session = Session
        self._load_image_model_config_from_db = load_image_model_config_from_db

        self.current_model = self._normalize_model_name(
            self._load_image_model_config_from_db()
        )

        # Cache for model handlers to avoid recreation
        self._handler_cache: Dict[str, Any] = {}

        logger.info(
            f"ImageHandler initialized with current model: {self.current_model}"
        )

    # ---------------------------------------------------------
    # INTERNAL MODEL RESOLUTION
    # ---------------------------------------------------------
    def _normalize_model_name(self, model_name: Optional[str]) -> str:
        normalized = self.MODEL_ALIASES.get(model_name)
        if normalized:
            return normalized

        # unknown models fall back to NoImageModel
        logger.warning(f"Unknown model alias '{model_name}', falling back to NoImageModel")
        return "NoImageModel"

    def _build_handler(self, model_name: str):
        if model_name == "CLIPModelHandler":
            return CLIPModelHandler()
        if model_name == "NoImageModel":
            return NoImageModel()

        logger.warning(f"Unknown model '{model_name}', building NoImageModel instead")
        return NoImageModel()

    def _get_handler(self, model_name: Optional[str] = None):
        resolved_name = self._normalize_model_name(model_name or self.current_model)

        if resolved_name not in self._handler_cache:
            logger.info(f"Creating image handler for model: {resolved_name}")
            self._handler_cache[resolved_name] = self._build_handler(resolved_name)

        return self._handler_cache[resolved_name]

    def refresh_current_model_from_db(self) -> str:
        self.current_model = self._normalize_model_name(
            self._load_image_model_config_from_db()
        )
        logger.info(f"Refreshed current image model from DB: {self.current_model}")
        return self.current_model

    # ---------------------------------------------------------
    # CORE MODEL OPERATIONS
    # ---------------------------------------------------------
    def allowed_file(self, filename, model_name=None):
        """Check if file extension is allowed for the specified model."""
        handler = self._get_handler(model_name)
        return handler.allowed_file(filename)

    def preprocess_image(self, image, model_name=None):
        """Preprocess image using the specified model."""
        handler = self._get_handler(model_name)
        return handler.preprocess_image(image)

    def get_image_embedding(self, image, model_name=None):
        """Get image embedding using the specified model."""
        handler = self._get_handler(model_name)
        embedding = handler.get_image_embedding(image)

        if embedding is None:
            return None

        if hasattr(embedding, "tolist"):
            return embedding.tolist()

        if isinstance(embedding, list):
            return embedding

        return list(embedding)

    def is_valid_image(self, image, model_name=None):
        """Validate if image meets requirements for the specified model."""
        handler = self._get_handler(model_name)
        return handler.is_valid_image(image)

    def store_image_metadata(self, session, title, description, file_path, embedding, model_name=None):
        """Store image metadata and embedding using pgvector-compatible methods."""
        handler = self._get_handler(model_name)

        if embedding is not None:
            if hasattr(embedding, "tolist"):
                embedding = embedding.tolist()
            elif not isinstance(embedding, list):
                embedding = list(embedding)

        resolved_model_name = self._normalize_model_name(model_name or self.current_model)

        handler.store_image_metadata(
            session=session,
            title=title,
            description=description,
            file_path=file_path,
            embedding=embedding,
            model_name=resolved_model_name,
        )

    # ---------------------------------------------------------
    # SAFE IMAGE LOADING
    # ---------------------------------------------------------
    def load_image_safe(self, file_path):
        """Safely loads an image with enhanced error handling and validation."""
        if not os.path.exists(file_path):
            logger.error(f"File does not exist: {file_path}")
            return None

        if os.path.getsize(file_path) == 0:
            logger.error(f"File is empty: {file_path}")
            return None

        # Verify file type using imghdr first
        file_type = imghdr.what(file_path)
        allowed_types = {"jpeg", "png", "jpg", "webp", "bmp", "tiff"}
        if file_type not in allowed_types:
            logger.warning(
                f"Invalid image type detected: {file_path}. Type: {file_type}"
            )
            return None

        try:
            with Image.open(file_path) as img:
                img_format = img.format.lower() if img.format else "unknown"

                supported_formats = {
                    "jpeg", "jpg", "png", "webp", "bmp", "tiff", "tif"
                }
                if img_format not in supported_formats:
                    logger.warning(
                        f"Image format mismatch: {file_path}. "
                        f"Detected format: {img_format}"
                    )

                rgb_image = img.convert("RGB")
                logger.debug(
                    f"Successfully loaded image: {file_path} "
                    f"({rgb_image.size[0]}x{rgb_image.size[1]})"
                )
                return rgb_image

        except UnidentifiedImageError:
            logger.error(f"Cannot identify image file: {file_path}")
            return None
        except Exception as exc:
            logger.error(f"Unexpected error while opening image {file_path}: {exc}")
            return None

    # ---------------------------------------------------------
    # COMPARISON / SEARCH
    # ---------------------------------------------------------
    def compare_images(self, image1_path, image2_path, model_name=None):
        """Compare two images using the specified model."""
        resolved_model_name = self._normalize_model_name(model_name or self.current_model)
        handler = self._get_handler(resolved_model_name)

        if hasattr(handler, "compare_images"):
            return handler.compare_images(image1_path, image2_path)

        logger.warning(f"Model {resolved_model_name} does not support image comparison")
        return {
            "similarity": 0.0,
            "image1": image1_path,
            "image2": image2_path,
            "model": resolved_model_name,
            "error": "Comparison not supported",
            "message": "Model does not support comparison",
        }

    def search_similar_images(
        self,
        session,
        query_image_path,
        model_name=None,
        limit=10,
        similarity_threshold=0.7,
    ):
        """Search for similar images in the database using pgvector."""
        resolved_model_name = self._normalize_model_name(model_name or self.current_model)
        handler = self._get_handler(resolved_model_name)

        if hasattr(handler, "search_similar_images_in_db"):
            return handler.search_similar_images_in_db(
                session=session,
                query_image_path=query_image_path,
                limit=limit,
                similarity_threshold=similarity_threshold,
            )

        logger.warning(f"Model {resolved_model_name} does not support similarity search")
        return []

    # ---------------------------------------------------------
    # MODEL INFO
    # ---------------------------------------------------------
    def get_model_info(self, model_name=None):
        """Get information about the specified model."""
        resolved_model_name = self._normalize_model_name(model_name or self.current_model)
        handler = self._get_handler(resolved_model_name)

        if hasattr(handler, "get_model_info"):
            return handler.get_model_info()

        return {
            "model_name": resolved_model_name,
            "model_loaded": True,
            "capabilities": ["basic"],
            "pgvector_compatible": False,
        }

    def get_available_models(self):
        """Get list of available models and their status."""
        model_names = ["CLIPModelHandler", "NoImageModel"]
        models_info = {}

        for model_name in model_names:
            try:
                handler = self._get_handler(model_name)
                if hasattr(handler, "get_model_info"):
                    models_info[model_name] = handler.get_model_info()
                else:
                    models_info[model_name] = {
                        "model_name": model_name,
                        "model_loaded": True,
                        "capabilities": ["basic"],
                    }
            except Exception as exc:
                models_info[model_name] = {
                    "model_name": model_name,
                    "model_loaded": False,
                    "error": str(exc),
                }

        models_info["current_model"] = self.current_model
        return models_info

    def set_current_model(self, model_name):
        """Set the current model for image processing."""
        resolved_model_name = self._normalize_model_name(model_name)

        if resolved_model_name not in {"CLIPModelHandler", "NoImageModel"}:
            logger.error(f"Unknown model: {model_name}")
            return False

        self.current_model = resolved_model_name
        logger.info(f"Current model set to: {resolved_model_name}")
        return True

    # ---------------------------------------------------------
    # PIPELINES
    # ---------------------------------------------------------
    def process_image_with_embedding(self, image_path, model_name=None):
        """Complete image processing pipeline: load, validate, and generate embedding."""
        resolved_model_name = self._normalize_model_name(model_name or self.current_model)

        try:
            image = self.load_image_safe(image_path)
            if image is None:
                return {
                    "success": False,
                    "error": "Failed to load image",
                    "image_path": image_path,
                }

            if not self.is_valid_image(image, resolved_model_name):
                return {
                    "success": False,
                    "error": "Image validation failed",
                    "image_path": image_path,
                    "image_size": image.size,
                }

            embedding = self.get_image_embedding(image, resolved_model_name)
            if embedding is None:
                return {
                    "success": False,
                    "error": "Failed to generate embedding",
                    "image_path": image_path,
                    "model": resolved_model_name,
                }

            return {
                "success": True,
                "image_path": image_path,
                "model": resolved_model_name,
                "embedding": embedding,
                "embedding_dimensions": len(embedding),
                "image_size": image.size,
            }

        except Exception as exc:
            logger.error(f"Error processing image {image_path}: {exc}")
            return {
                "success": False,
                "error": str(exc),
                "image_path": image_path,
            }

    def batch_process_images(self, image_paths, model_name=None, progress_callback=None):
        """Process multiple images in batch with optional progress callback."""
        resolved_model_name = self._normalize_model_name(model_name or self.current_model)
        results = []

        total_images = len(image_paths)
        successful = 0
        failed = 0

        logger.info(
            f"Starting batch processing of {total_images} images using {resolved_model_name}"
        )

        for i, image_path in enumerate(image_paths):
            try:
                result = self.process_image_with_embedding(image_path, resolved_model_name)
                results.append(result)

                if result["success"]:
                    successful += 1
                else:
                    failed += 1

                if progress_callback:
                    progress_callback(i + 1, total_images, successful, failed)

            except Exception as exc:
                logger.error(f"Error in batch processing image {image_path}: {exc}")
                results.append({
                    "success": False,
                    "error": str(exc),
                    "image_path": image_path,
                })
                failed += 1

        logger.info(f"Batch processing complete: {successful} successful, {failed} failed")

        return {
            "results": results,
            "summary": {
                "total": total_images,
                "successful": successful,
                "failed": failed,
                "success_rate": (successful / total_images * 100) if total_images > 0 else 0,
            },
        }

    # ---------------------------------------------------------
    # DB UTILITIES
    # ---------------------------------------------------------
    def get_embedding_statistics(self, session):
        """Get statistics about embeddings in the database."""
        try:
            from modules.emtacdb.emtacdb_fts import Image as ImageRecord
            return ImageRecord.get_embedding_statistics(session)
        except Exception as exc:
            logger.error(f"Error getting embedding statistics: {exc}")
            return {}

    def migrate_embeddings_to_pgvector(self, session):
        """Migrate all legacy embeddings to pgvector format."""
        try:
            from modules.emtacdb.emtacdb_fts import Image as ImageRecord
            return ImageRecord.migrate_all_embeddings_to_pgvector(session)
        except Exception as exc:
            logger.error(f"Error migrating embeddings: {exc}")
            return {
                "total": 0,
                "migrated": 0,
                "failed": 0,
                "success_rate": 0,
            }

    def setup_pgvector_indexes(self, session):
        """Setup pgvector indexes for optimal performance."""
        try:
            from modules.emtacdb.emtacdb_fts import Image as ImageRecord
            return ImageRecord.setup_pgvector_indexes(session)
        except Exception as exc:
            logger.error(f"Error setting up pgvector indexes: {exc}")
            return False