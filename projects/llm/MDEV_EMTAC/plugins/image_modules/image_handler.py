#plugins/image_modules/image_handler.py

import os
import imghdr
from PIL import Image, UnidentifiedImageError
from plugins.image_modules import CLIPModelHandler, NoImageModel
from modules.emtacdb.emtacdb_fts import Session, load_image_model_config_from_db, ImageEmbedding
from modules.configuration.log_config import logger

class ImageHandler:
    """
    Enhanced ImageHandler with pgvector support and advanced image processing capabilities.
    """

    def __init__(self):
        self.model_handlers = {
            "clip": CLIPModelHandler(),
            "no_model": NoImageModel()
        }
        self.Session = Session
        self.current_model = load_image_model_config_from_db()

        # Cache for model handlers to avoid recreation
        self._handler_cache = {}

        logger.info(f"ImageHandler initialized with current model: {self.current_model}")

    def allowed_file(self, filename, model_name=None):
        """Check if file extension is allowed for the specified model."""
        model_name = model_name or self.current_model
        try:
            return self.model_handlers[model_name].allowed_file(filename)
        except KeyError:
            logger.error(f"Unknown model: {model_name}")
            return self.model_handlers["no_model"].allowed_file(filename)

    def preprocess_image(self, image, model_name=None):
        """Preprocess image using the specified model."""
        model_name = model_name or self.current_model
        try:
            return self.model_handlers[model_name].preprocess_image(image)
        except KeyError:
            logger.error(f"Unknown model: {model_name}")
            return self.model_handlers["no_model"].preprocess_image(image)

    def get_image_embedding(self, image, model_name=None):
        """Get image embedding using the specified model - returns list for pgvector compatibility."""
        model_name = model_name or self.current_model
        try:
            embedding = self.model_handlers[model_name].get_image_embedding(image)

            # Ensure the embedding is in list format for pgvector compatibility
            if embedding is not None:
                if hasattr(embedding, 'tolist'):
                    return embedding.tolist()
                elif isinstance(embedding, list):
                    return embedding
                else:
                    return list(embedding)
            return None
        except KeyError:
            logger.error(f"Unknown model: {model_name}")
            return self.model_handlers["no_model"].get_image_embedding(image)

    def is_valid_image(self, image, model_name=None):
        """Validate if image meets requirements for the specified model."""
        model_name = model_name or self.current_model
        try:
            return self.model_handlers[model_name].is_valid_image(image)
        except KeyError:
            logger.error(f"Unknown model: {model_name}")
            return self.model_handlers["no_model"].is_valid_image(image)

    def store_image_metadata(self, session, title, description, file_path, embedding, model_name=None):
        """Store image metadata and embedding using pgvector-compatible methods."""
        model_name = model_name or self.current_model
        try:
            # Ensure embedding is in the right format
            if embedding is not None:
                if hasattr(embedding, 'tolist'):
                    embedding = embedding.tolist()
                elif not isinstance(embedding, list):
                    embedding = list(embedding)

            self.model_handlers[model_name].store_image_metadata(
                session, title, description, file_path, embedding, model_name
            )
        except KeyError:
            logger.error(f"Unknown model: {model_name}")
            self.model_handlers["no_model"].store_image_metadata(
                session, title, description, file_path, embedding, model_name
            )

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
        allowed_types = {"jpeg", "png", "jpg", "webp", "bmp", "tiff"}  # Expanded supported types
        if file_type not in allowed_types:
            logger.warning(f"Invalid image type detected: {file_path}. Type: {file_type}")
            return None

        try:
            with Image.open(file_path) as img:
                img_format = img.format.lower() if img.format else "unknown"

                # Be more flexible with format checking
                supported_formats = {"jpeg", "jpg", "png", "webp", "bmp", "tiff", "tif"}
                if img_format not in supported_formats:
                    logger.warning(f"Image format mismatch: {file_path}. Detected format: {img_format}")
                    # Don't return None immediately, try to convert anyway

                # Convert to RGB and return a copy
                rgb_image = img.convert("RGB")
                logger.debug(f"Successfully loaded image: {file_path} ({img.size[0]}x{img.size[1]})")
                return rgb_image

        except UnidentifiedImageError:
            logger.error(f"Cannot identify image file: {file_path}")
            return None
        except Exception as e:
            logger.error(f"Unexpected error while opening image {file_path}: {e}")
            return None

    def compare_images(self, image1_path, image2_path, model_name=None):
        """Compare two images using the specified model."""
        model_name = model_name or self.current_model
        try:
            handler = self.model_handlers[model_name]
            if hasattr(handler, 'compare_images'):
                return handler.compare_images(image1_path, image2_path)
            else:
                logger.warning(f"Model {model_name} does not support image comparison")
                return {
                    "similarity": 0.0,
                    "image1": image1_path,
                    "image2": image2_path,
                    "model": model_name,
                    "error": "Comparison not supported",
                    "message": "Model does not support comparison"
                }
        except KeyError:
            logger.error(f"Unknown model: {model_name}")
            return {
                "similarity": 0.0,
                "image1": image1_path,
                "image2": image2_path,
                "model": model_name,
                "error": "Unknown model",
                "message": "Comparison failed"
            }

    def search_similar_images(self, session, query_image_path, model_name=None,
                              limit=10, similarity_threshold=0.7):
        """Search for similar images in the database using pgvector."""
        model_name = model_name or self.current_model
        try:
            handler = self.model_handlers[model_name]
            if hasattr(handler, 'search_similar_images_in_db'):
                return handler.search_similar_images_in_db(
                    session, query_image_path, limit, similarity_threshold
                )
            else:
                logger.warning(f"Model {model_name} does not support similarity search")
                return []
        except KeyError:
            logger.error(f"Unknown model: {model_name}")
            return []

    def get_model_info(self, model_name=None):
        """Get information about the specified model."""
        model_name = model_name or self.current_model
        try:
            handler = self.model_handlers[model_name]
            if hasattr(handler, 'get_model_info'):
                return handler.get_model_info()
            else:
                return {
                    "model_name": model_name,
                    "model_loaded": True,
                    "capabilities": ["basic"],
                    "pgvector_compatible": False
                }
        except KeyError:
            logger.error(f"Unknown model: {model_name}")
            return {
                "model_name": model_name,
                "model_loaded": False,
                "error": "Unknown model"
            }

    def get_available_models(self):
        """Get list of available models and their status."""
        models_info = {}
        for model_name, handler in self.model_handlers.items():
            try:
                if hasattr(handler, 'get_model_info'):
                    models_info[model_name] = handler.get_model_info()
                else:
                    models_info[model_name] = {
                        "model_name": model_name,
                        "model_loaded": True,
                        "capabilities": ["basic"]
                    }
            except Exception as e:
                models_info[model_name] = {
                    "model_name": model_name,
                    "model_loaded": False,
                    "error": str(e)
                }

        models_info["current_model"] = self.current_model
        return models_info

    def set_current_model(self, model_name):
        """Set the current model for image processing."""
        if model_name in self.model_handlers:
            self.current_model = model_name
            logger.info(f"Current model set to: {model_name}")
            return True
        else:
            logger.error(f"Unknown model: {model_name}")
            return False

    def process_image_with_embedding(self, image_path, model_name=None):
        """Complete image processing pipeline: load, validate, and generate embedding."""
        model_name = model_name or self.current_model

        try:
            # Load image safely
            image = self.load_image_safe(image_path)
            if image is None:
                return {
                    "success": False,
                    "error": "Failed to load image",
                    "image_path": image_path
                }

            # Validate image
            if not self.is_valid_image(image, model_name):
                return {
                    "success": False,
                    "error": "Image validation failed",
                    "image_path": image_path,
                    "image_size": image.size
                }

            # Generate embedding
            embedding = self.get_image_embedding(image, model_name)
            if embedding is None:
                return {
                    "success": False,
                    "error": "Failed to generate embedding",
                    "image_path": image_path,
                    "model": model_name
                }

            return {
                "success": True,
                "image_path": image_path,
                "model": model_name,
                "embedding": embedding,
                "embedding_dimensions": len(embedding),
                "image_size": image.size
            }

        except Exception as e:
            logger.error(f"Error processing image {image_path}: {e}")
            return {
                "success": False,
                "error": str(e),
                "image_path": image_path
            }

    def batch_process_images(self, image_paths, model_name=None, progress_callback=None):
        """Process multiple images in batch with optional progress callback."""
        model_name = model_name or self.current_model
        results = []

        total_images = len(image_paths)
        successful = 0
        failed = 0

        logger.info(f"Starting batch processing of {total_images} images using {model_name}")

        for i, image_path in enumerate(image_paths):
            try:
                result = self.process_image_with_embedding(image_path, model_name)
                results.append(result)

                if result["success"]:
                    successful += 1
                else:
                    failed += 1

                # Call progress callback if provided
                if progress_callback:
                    progress_callback(i + 1, total_images, successful, failed)

            except Exception as e:
                logger.error(f"Error in batch processing image {image_path}: {e}")
                results.append({
                    "success": False,
                    "error": str(e),
                    "image_path": image_path
                })
                failed += 1

        logger.info(f"Batch processing complete: {successful} successful, {failed} failed")

        return {
            "results": results,
            "summary": {
                "total": total_images,
                "successful": successful,
                "failed": failed,
                "success_rate": (successful / total_images * 100) if total_images > 0 else 0
            }
        }

    def get_embedding_statistics(self, session):
        """Get statistics about embeddings in the database."""
        try:
            from modules.emtacdb.emtacdb_fts import Image
            return Image.get_embedding_statistics(session)
        except Exception as e:
            logger.error(f"Error getting embedding statistics: {e}")
            return {}

    def migrate_embeddings_to_pgvector(self, session):
        """Migrate all legacy embeddings to pgvector format."""
        try:
            from modules.emtacdb.emtacdb_fts import Image
            return Image.migrate_all_embeddings_to_pgvector(session)
        except Exception as e:
            logger.error(f"Error migrating embeddings: {e}")
            return {'total': 0, 'migrated': 0, 'failed': 0, 'success_rate': 0}

    def setup_pgvector_indexes(self, session):
        """Setup pgvector indexes for optimal performance."""
        try:
            from modules.emtacdb.emtacdb_fts import Image
            return Image.setup_pgvector_indexes(session)
        except Exception as e:
            logger.error(f"Error setting up pgvector indexes: {e}")
            return False
