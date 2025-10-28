# plugins/image_modules/image_models.py
import os
import logging
import time  # ← This was missing!
from typing import Dict, Any, Optional
from PIL import Image
import torch
from transformers import CLIPProcessor, CLIPModel
import numpy as np
import sys
from sqlalchemy.orm import sessionmaker
from sqlalchemy import create_engine
import logging
from PIL import ImageFile
from abc import ABC, abstractmethod
import torch
# Import config variables from config.py
from modules.configuration.config import DATABASE_URL, ALLOWED_EXTENSIONS
from typing import Dict, Any, Optional
from modules.configuration.config import BASE_DIR  # Import BASE_DIR

ImageFile.LOAD_TRUNCATED_IMAGES = True

logger = logging.getLogger(__name__)

# SQLAlchemy setup
engine = create_engine(DATABASE_URL)
Session = sessionmaker(bind=engine)
# Function to dynamically import and instantiate the correct model handler
def get_image_model_handler(model_name):
    module = sys.modules[__name__]
    try:
        model_class = getattr(module, model_name)
        if issubclass(model_class, BaseImageModelHandler):
            return model_class()
        else:
            raise ValueError(f"{model_name} is not a subclass of BaseImageModelHandler")
    except AttributeError:
        logger.error(f"{model_name} not found in {__name__}")
        return NoImageModel()

# Define the BaseImageModelHandler interface
class BaseImageModelHandler(ABC):
    @abstractmethod
    def allowed_file(self, filename):
        pass

    @abstractmethod
    def preprocess_image(self, image):
        pass

    @abstractmethod
    def get_image_embedding(self, image):
        pass

    @abstractmethod
    def is_valid_image(self, image):
        pass

    def store_image_metadata(self, session, title, description, file_path, embedding, model_name):
        from modules.emtacdb.emtacdb_fts import Image, ImageEmbedding
        # Ensure file_path is relative
        if os.path.isabs(file_path):
            relative_file_path = os.path.relpath(file_path, BASE_DIR)
            logger.debug(f"Converted absolute file path '{file_path}' to relative path '{relative_file_path}'.")
        else:
            relative_file_path = file_path
            logger.debug(f"Using existing relative file path '{relative_file_path}'.")

        # Create Image entry with relative path
        image = Image(title=title, description=description, file_path=relative_file_path)
        session.add(image)
        session.commit()

        # Create ImageEmbedding entry
        image_embedding = ImageEmbedding(image_id=image.id, model_name=model_name, model_embedding=embedding.tobytes())
        session.add(image_embedding)
        session.commit()

        logger.info(f"Stored image metadata and embedding for '{relative_file_path}' using '{model_name}'.")

# Implement the NoImageModel handler
class NoImageModel(BaseImageModelHandler):
    def allowed_file(self, filename):
        return False

    def preprocess_image(self, image):
        return None

    def get_image_embedding(self, image):
        return None

    def is_valid_image(self, image):
        return False

    def store_image_metadata(self, session, title, description, file_path, embedding, model_name):
        logger.info("No image model selected, not storing image metadata.")


class CLIPModelHandler(BaseImageModelHandler):
    """Enhanced CLIP model handler with pgvector integration."""

    # Class-level cache to persist models across instances
    _model_cache = {}
    _processor_cache = {}

    def __init__(self):
        self.model_name = "CLIPModelHandler"
        self.clip_model_id = "openai/clip-vit-base-patch32"
        self.model = None
        self.processor = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Initializing CLIP model handler on device: {self.device}")

        # Load model immediately (with caching)
        self._load_model()

    def _load_model(self):
        """Load CLIP model with caching to avoid repeated loading."""
        cache_key = f"{self.clip_model_id}_{self.device}"

        # Check if model is already cached
        if cache_key in self._model_cache:
            logger.info("Using cached CLIP model - INSTANT LOAD!")
            self.model = self._model_cache[cache_key]
            self.processor = self._processor_cache[cache_key]
            return

        start_time = time.time()
        logger.info(f"Loading CLIP model: {self.clip_model_id}")

        try:
            # Load model and processor (offline mode prevents network calls)
            self.model = CLIPModel.from_pretrained(
                self.clip_model_id,
                local_files_only=True,  # Force local files only
                cache_dir=None  # Use default cache
            ).to(self.device)

            self.processor = CLIPProcessor.from_pretrained(
                self.clip_model_id,
                local_files_only=True,  # Force local files only
                cache_dir=None  # Use default cache
            )

            # Cache the loaded model and processor
            self._model_cache[cache_key] = self.model
            self._processor_cache[cache_key] = self.processor

            load_time = time.time() - start_time
            logger.info(f"Successfully loaded and cached CLIP model in {load_time:.2f}s")

        except Exception as e:
            logger.error(f"Error loading CLIP model: {e}")
            # Fallback: try without local_files_only if offline loading fails
            try:
                logger.warning("Attempting fallback model loading with network access")
                self.model = CLIPModel.from_pretrained(self.clip_model_id).to(self.device)
                self.processor = CLIPProcessor.from_pretrained(self.clip_model_id)

                # Cache the fallback model
                self._model_cache[cache_key] = self.model
                self._processor_cache[cache_key] = self.processor

                load_time = time.time() - start_time
                logger.info(f"Fallback model loaded in {load_time:.2f}s")
            except Exception as fallback_error:
                logger.error(f"Failed to load CLIP model even with fallback: {fallback_error}")
                raise

    def allowed_file(self, filename):
        """Check if file extension is allowed."""
        return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

    def preprocess_image(self, image):
        """Preprocess image for CLIP model."""
        if not self.processor:
            raise RuntimeError("CLIP processor not loaded")

        # Resize image while maintaining aspect ratio
        image = image.resize((224, 224))
        inputs = self.processor(images=image, return_tensors="pt", padding=True)
        return inputs.to(self.device)

    def get_image_embedding(self, image):
        """Get CLIP embedding for an image - returns list for pgvector compatibility."""
        try:
            if not self.model or not self.processor:
                logger.error("CLIP model or processor not loaded")
                return None

            inputs = self.preprocess_image(image)
            with torch.no_grad():
                image_features = self.model.get_image_features(**inputs)
                # Normalize the features
                image_features = image_features / image_features.norm(dim=-1, keepdim=True)

            # Convert to Python list for pgvector compatibility
            embedding = image_features.cpu().numpy().flatten().tolist()
            logger.info(f"Generated CLIP embedding (dimensions: {len(embedding)})")
            return embedding

        except Exception as e:
            logger.error(f"Error processing image: {e}")
            return None

    def is_valid_image(self, image):
        """Validate if image meets requirements."""
        try:
            width, height = image.size
            logger.info(f"Image dimensions: width={width}, height={height}")

            # Define minimum and maximum dimensions
            min_dimension = 100  # Minimum acceptable dimension
            max_dimension = 5000  # Maximum acceptable dimension

            if width < min_dimension or height < min_dimension:
                logger.info(f"Image is too small: width={width}, height={height}")
                return False
            if width > max_dimension or height > max_dimension:
                logger.info(f"Image is too large: width={width}, height={height}")
                return False

            # Define acceptable aspect ratio range
            min_aspect_ratio = 1 / 5  # Minimum aspect ratio (height/width)
            max_aspect_ratio = 5  # Maximum aspect ratio (width/height)

            aspect_ratio = width / height
            if not (min_aspect_ratio <= aspect_ratio <= max_aspect_ratio):
                logger.info(f"Image aspect ratio {aspect_ratio} is outside the acceptable range. "
                            f"Min aspect ratio: {min_aspect_ratio}, Max aspect ratio: {max_aspect_ratio}")
                return False

            return True

        except Exception as e:
            logger.error(f"Image validation failed: {e}")
            return False

    def store_image_metadata(self, session, title, description, file_path, embedding, model_name):
        """
        Updated to use pgvector-compatible ImageEmbedding creation.
        """
        from modules.emtacdb.emtacdb_fts import Image, ImageEmbedding
        # Ensure file_path is relative
        if os.path.isabs(file_path):
            relative_file_path = os.path.relpath(file_path, BASE_DIR)
            logger.debug(f"Converted absolute file path '{file_path}' to relative path '{relative_file_path}'.")
        else:
            relative_file_path = file_path
            logger.debug(f"Using existing relative file path '{relative_file_path}'.")

        # Create Image entry with relative path
        image = Image(title=title, description=description, file_path=relative_file_path)
        session.add(image)
        session.commit()

        # Create ImageEmbedding entry using pgvector method
        try:
            # Ensure embedding is a list
            if isinstance(embedding, list):
                embedding_list = embedding
            elif hasattr(embedding, 'tolist'):
                embedding_list = embedding.tolist()
            elif isinstance(embedding, np.ndarray):
                embedding_list = embedding.flatten().tolist()
            else:
                embedding_list = list(embedding)

            # Use the new pgvector creation method
            image_embedding = ImageEmbedding.create_with_pgvector(
                image_id=image.id,
                model_name=model_name,
                embedding=embedding_list
            )
            session.add(image_embedding)
            session.commit()

            logger.info(
                f"Stored image metadata and pgvector embedding for '{relative_file_path}' using '{model_name}'.")
        except Exception as e:
            logger.warning(f"Failed to store pgvector embedding, falling back to legacy format: {e}")
            # Fallback to legacy format
            image_embedding = ImageEmbedding.create_with_legacy(
                image_id=image.id,
                model_name=model_name,
                embedding=embedding_list if 'embedding_list' in locals() else embedding
            )
            session.add(image_embedding)
            session.commit()
            logger.info(f"Stored image metadata and legacy embedding for '{relative_file_path}' using '{model_name}'.")

    def compare_images(self, image1_path: str, image2_path: str) -> dict:
        """Enhanced image comparison using CLIP embeddings."""
        try:
            logger.info(f"Comparing images with CLIP: {image1_path} vs {image2_path}")

            if not self.model or not self.processor:
                return {
                    "similarity": 0.0,
                    "image1": image1_path,
                    "image2": image2_path,
                    "model": self.model_name,
                    "error": "Model not loaded",
                    "message": "Comparison failed"
                }

            # Load both images
            image1 = Image.open(image1_path).convert('RGB')
            image2 = Image.open(image2_path).convert('RGB')

            # Get embeddings for both images
            embedding1 = self.get_image_embedding(image1)
            embedding2 = self.get_image_embedding(image2)

            if embedding1 is None or embedding2 is None:
                return {
                    "similarity": 0.0,
                    "image1": image1_path,
                    "image2": image2_path,
                    "model": self.model_name,
                    "error": "Failed to generate embeddings",
                    "message": "Comparison failed"
                }

            # Calculate cosine similarity using the embeddings
            import math

            # Calculate dot product
            dot_product = sum(a * b for a, b in zip(embedding1, embedding2))

            # Calculate norms
            norm1 = math.sqrt(sum(a * a for a in embedding1))
            norm2 = math.sqrt(sum(b * b for b in embedding2))

            # Calculate cosine similarity
            if norm1 == 0 or norm2 == 0:
                similarity = 0.0
            else:
                similarity = dot_product / (norm1 * norm2)

            logger.info(f"Image comparison similarity: {similarity:.4f}")

            return {
                "similarity": float(similarity),
                "image1": image1_path,
                "image2": image2_path,
                "model": self.model_name,
                "message": "Comparison completed successfully"
            }

        except Exception as e:
            logger.error(f"Error comparing images with CLIP: {e}")
            return {
                "similarity": 0.0,
                "image1": image1_path,
                "image2": image2_path,
                "model": self.model_name,
                "error": str(e),
                "message": "Comparison failed"
            }

    def search_similar_images_in_db(self, session, query_image_path: str,
                                    limit: int = 10, similarity_threshold: float = 0.7) -> list:
        """
        New method to search for similar images in the database using pgvector.
        """
        try:
            logger.info(f"Searching for similar images to: {query_image_path}")

            # Load and process query image
            query_image = Image.open(query_image_path).convert('RGB')
            if not self.is_valid_image(query_image):
                logger.warning(f"Query image is not valid: {query_image_path}")
                return []

            # Get embedding for query image
            query_embedding = self.get_image_embedding(query_image)
            if query_embedding is None:
                logger.error(f"Failed to generate embedding for query image: {query_image_path}")
                return []

            # Use ImageEmbedding's search method
            from modules.emtacdb.emtacdb_fts import ImageEmbedding
            similar_images = ImageEmbedding.search_similar_images(
                session=session,
                query_embedding=query_embedding,
                model_name=self.model_name,
                limit=limit,
                similarity_threshold=similarity_threshold
            )

            logger.info(f"Found {len(similar_images)} similar images")
            return similar_images

        except Exception as e:
            logger.error(f"Error searching for similar images: {e}")
            return []

    @classmethod
    def preload_model(cls) -> bool:
        """Class method to preload model during application startup."""
        try:
            logger.info("Preloading CLIP model for faster subsequent access...")
            start_time = time.time()

            # Create temporary instance to trigger model loading
            temp_handler = cls()

            preload_time = time.time() - start_time
            logger.info(f"CLIP model preloaded successfully in {preload_time:.2f}s")
            return True

        except Exception as e:
            logger.error(f"Failed to preload CLIP model: {e}")
            return False

    def get_cache_stats(self) -> Dict[str, Any]:
        """Get information about the model cache."""
        return {
            "models_cached": len(self._model_cache),
            "processors_cached": len(self._processor_cache),
            "cache_keys": list(self._model_cache.keys()),
            "model_loaded": self.model is not None,
            "processor_loaded": self.processor is not None,
            "device": str(self.device),
            "offline_mode": os.environ.get("TRANSFORMERS_OFFLINE", "0") == "1"
        }

    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the loaded model."""
        return {
            "model_name": self.model_name,
            "model_id": self.clip_model_id,
            "device": str(self.device),
            "model_loaded": self.model is not None,
            "processor_loaded": self.processor is not None,
            "cache_size": len(self._model_cache),
            "offline_mode": os.environ.get("TRANSFORMERS_OFFLINE", "0") == "1",
            "pgvector_compatible": True
        }


