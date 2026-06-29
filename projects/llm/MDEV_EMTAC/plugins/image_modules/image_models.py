# plugins/image_modules/image_models.py
from dotenv import load_dotenv
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
    """
    Clean, corrected, and unified CLIP model handler.

    - Loads CLIP strictly offline using MODEL_PATH_CLIP
    - Supports local Hugging Face folders with:
        * model.safetensors
        * or pytorch_model.bin
    - Uses cache to avoid repeated reloads
    - Retries on CPU if initial CUDA load fails
    - Returns list embeddings for pgvector compatibility
    """

    _model_cache = {}
    _processor_cache = {}

    def __init__(self):
        load_dotenv()

        self.model_name = "CLIPModelHandler"
        self.clip_model_dir = os.getenv("MODEL_PATH_CLIP") or os.getenv("MODEL_CLIP_DIR")

        if not self.clip_model_dir:
            raise RuntimeError("MODEL_PATH_CLIP or MODEL_CLIP_DIR not set in .env")

        if not os.path.isdir(self.clip_model_dir):
            raise RuntimeError(
                f"CLIP model directory does not exist: {self.clip_model_dir}"
            )

        self._validate_model_dir()

        self.model = None
        self.processor = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        logger.info(f"[CLIP] Initializing CLIP on device: {self.device}")
        logger.info(f"[CLIP] Model directory: {self.clip_model_dir}")

        self._load_model()

    # -------------------------------------------------------------
    # VALIDATE LOCAL MODEL DIRECTORY
    # -------------------------------------------------------------
    def _validate_model_dir(self):
        required_files = [
            "config.json",
            "preprocessor_config.json",
        ]

        missing = [
            name for name in required_files
            if not os.path.exists(os.path.join(self.clip_model_dir, name))
        ]
        if missing:
            raise RuntimeError(
                f"Offline CLIP model missing required files: {', '.join(missing)}"
            )

        has_weights = (
            os.path.exists(os.path.join(self.clip_model_dir, "model.safetensors"))
            or os.path.exists(os.path.join(self.clip_model_dir, "pytorch_model.bin"))
        )

        if not has_weights:
            raise RuntimeError(
                "Offline CLIP model missing weights file. "
                "Expected 'model.safetensors' or 'pytorch_model.bin'."
            )

    # -------------------------------------------------------------
    # UNIFIED MODEL LOADER
    # -------------------------------------------------------------
    def _load_model(self):
        cache_key = self.clip_model_dir

        if cache_key in self._model_cache:
            logger.info("[CLIP] Loaded from cache")
            self.model = self._model_cache[cache_key]
            self.processor = self._processor_cache[cache_key]
            return

        logger.info(f"[CLIP] Loading offline CLIP model from: {self.clip_model_dir}")
        start = time.time()

        try:
            # Load processor first
            processor = CLIPProcessor.from_pretrained(
                self.clip_model_dir,
                local_files_only=True,
            )

            # First try normal device path
            try:
                model = CLIPModel.from_pretrained(
                    self.clip_model_dir,
                    local_files_only=True,
                )
                model = model.to(self.device)

            except Exception as first_error:
                logger.warning(f"[CLIP] First model load attempt failed: {first_error}")

                # If CUDA is available, clear cache and retry on CPU
                try:
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                        logger.info("[CLIP] CUDA cache cleared after failed load")
                except Exception:
                    pass

                logger.info("[CLIP] Retrying CLIP load on CPU")
                model = CLIPModel.from_pretrained(
                    self.clip_model_dir,
                    local_files_only=True,
                )
                self.device = torch.device("cpu")
                model = model.to(self.device)

            self.model = model
            self.processor = processor

        except Exception as e:
            logger.error(f"[CLIP] FAILED to load offline model from {self.clip_model_dir}: {e}", exc_info=True)
            raise RuntimeError(
                f"Failed to load offline CLIP model from '{self.clip_model_dir}': {e}"
            ) from e

        self._model_cache[cache_key] = self.model
        self._processor_cache[cache_key] = self.processor

        logger.info(f"[CLIP] Model loaded successfully in {time.time() - start:.2f}s on {self.device}")

    # -------------------------------------------------------------
    # ALLOWED FILE EXTENSIONS
    # -------------------------------------------------------------
    def allowed_file(self, filename):
        return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS

    # -------------------------------------------------------------
    # PREPROCESSING
    # -------------------------------------------------------------
    def preprocess_image(self, image):
        if self.processor is None:
            raise RuntimeError("CLIP processor not loaded")

        if image.mode != "RGB":
            image = image.convert("RGB")

        inputs = self.processor(images=image, return_tensors="pt", padding=True)
        return {k: v.to(self.device) for k, v in inputs.items()}

    # -------------------------------------------------------------
    # EMBEDDING
    # -------------------------------------------------------------
    def get_image_embedding(self, image):
        try:
            if not self.is_valid_image(image):
                logger.warning("[CLIP] Invalid image supplied for embedding")
                return None

            inputs = self.preprocess_image(image)

            with torch.no_grad():
                img_features = self.model.get_image_features(**inputs)
                img_features = img_features / img_features.norm(dim=-1, keepdim=True)

            embedding = img_features.detach().cpu().numpy().flatten().tolist()

            logger.debug(f"[CLIP] Embedding generated ({len(embedding)} dims)")
            return embedding

        except Exception as e:
            logger.error(f"[CLIP] Error generating embedding: {e}", exc_info=True)
            return None

    # -------------------------------------------------------------
    # VALIDATION
    # -------------------------------------------------------------
    def is_valid_image(self, image):
        try:
            w, h = image.size

            if w < 32 or h < 32:
                logger.warning(f"[CLIP] Image too small: {w}x{h}")
                return False

            if w > 10000 or h > 10000:
                logger.warning(f"[CLIP] Image too large: {w}x{h}")
                return False

            aspect = w / h if h else 0
            if not (0.1 <= aspect <= 10.0):
                logger.warning(f"[CLIP] Invalid image aspect ratio: {aspect}")
                return False

            return True

        except Exception as e:
            logger.error(f"[CLIP] Image validation failed: {e}", exc_info=True)
            return False

    # -------------------------------------------------------------
    # DB STORAGE
    # -------------------------------------------------------------
    def store_image_metadata(self, session, title, description, file_path, embedding, model_name):
        from modules.emtacdb.emtacdb_fts import Image, ImageEmbedding

        if os.path.isabs(file_path):
            file_path = os.path.relpath(file_path, BASE_DIR)

        img = Image(title=title, description=description, file_path=file_path)
        session.add(img)
        session.flush()

        try:
            if embedding is None:
                logger.warning(f"[CLIP] No embedding generated for {file_path}")
                return

            if not isinstance(embedding, list):
                if hasattr(embedding, "tolist"):
                    embedding = embedding.tolist()
                else:
                    embedding = list(embedding)

            emb = ImageEmbedding.create_with_pgvector(
                image_id=img.id,
                model_name=model_name,
                embedding=embedding,
            )
            session.add(emb)
            session.flush()

            logger.info(f"[CLIP] Stored pgvector embedding for {file_path}")

        except Exception as e:
            logger.error(f"[CLIP] Failed to store pgvector embedding: {e}", exc_info=True)
            raise

    # -------------------------------------------------------------
    # IMAGE COMPARISON
    # -------------------------------------------------------------
    def compare_images(self, image1_path, image2_path):
        import math

        try:
            img1 = Image.open(image1_path).convert("RGB")
            img2 = Image.open(image2_path).convert("RGB")

            emb1 = self.get_image_embedding(img1)
            emb2 = self.get_image_embedding(img2)

            if not emb1 or not emb2:
                return {"similarity": 0.0, "error": "Embedding failed"}

            dot = sum(a * b for a, b in zip(emb1, emb2))
            n1 = math.sqrt(sum(a * a for a in emb1))
            n2 = math.sqrt(sum(b * b for b in emb2))
            sim = dot / (n1 * n2) if n1 and n2 else 0.0

            return {
                "similarity": float(sim),
                "image1": image1_path,
                "image2": image2_path,
                "model": self.model_name,
            }

        except Exception as e:
            logger.error(f"[CLIP] compare_images failed: {e}", exc_info=True)
            return {"similarity": 0.0, "error": str(e)}

    # -------------------------------------------------------------
    # SEARCH SIMILAR IMAGES
    # -------------------------------------------------------------
    def search_similar_images_in_db(self, session, query_image_path, limit=10, similarity_threshold=0.7):
        try:
            query_img = Image.open(query_image_path).convert("RGB")

            if not self.is_valid_image(query_img):
                return []

            emb = self.get_image_embedding(query_img)
            if emb is None:
                return []

            from modules.emtacdb.emtacdb_fts import ImageEmbedding

            return ImageEmbedding.search_similar_images(
                session=session,
                query_embedding=emb,
                model_name=self.model_name,
                limit=limit,
                similarity_threshold=similarity_threshold,
            )

        except Exception as e:
            logger.error(f"[CLIP] search_similar_images_in_db failed: {e}", exc_info=True)
            return []

    # -------------------------------------------------------------
    # MODEL INFO
    # -------------------------------------------------------------
    def get_model_info(self):
        return {
            "model_name": self.model_name,
            "model_dir": self.clip_model_dir,
            "device": str(self.device),
            "cached_models": len(self._model_cache),
            "cached_processors": len(self._processor_cache),
            "model_loaded": self.model is not None,
            "processor_loaded": self.processor is not None,
            "offline_mode": True,
        }

    # -------------------------------------------------------------
    # PRELOAD
    # -------------------------------------------------------------
    @classmethod
    def preload_model(cls):
        try:
            logger.info("[CLIP] Preloading offline model...")
            start = time.time()
            _ = cls()
            logger.info(f"[CLIP] Preloaded in {time.time() - start:.2f}s")
            return True
        except Exception as e:
            logger.error(f"[CLIP] Preload failed: {e}", exc_info=True)
            return False

