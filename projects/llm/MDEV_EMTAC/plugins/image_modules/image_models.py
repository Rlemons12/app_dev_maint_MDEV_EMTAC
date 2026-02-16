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
    - Removes ALL duplicated _load_model() logic
    - Loads CLIP strictly offline using MODEL_PATH_CLIP
    - Provides consistent caching
    - Maintains pgvector support
    """

    _model_cache = {}
    _processor_cache = {}

    def __init__(self):
        load_dotenv()
        self.model_name = "CLIPModelHandler"
        self.clip_model_dir = os.getenv("MODEL_PATH_CLIP")

        if not self.clip_model_dir:
            raise RuntimeError("MODEL_PATH_CLIP not set in .env")

        if not os.path.exists(self.clip_model_dir):
            raise RuntimeError(f"MODEL_PATH_CLIP directory does not exist: {self.clip_model_dir}")

        self.model = None
        self.processor = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        logger.info(f"[CLIP] Initializing CLIP on device: {self.device}")
        logger.info(f"[CLIP] Model directory: {self.clip_model_dir}")

        self._load_model()

    # -------------------------------------------------------------
    # UNIFIED MODEL LOADER  ✔ Only one version
    # -------------------------------------------------------------
    def _load_model(self):
        cache_key = f"{self.clip_model_dir}_{self.device}"

        # Return cached model instantly
        if cache_key in self._model_cache:
            logger.info("[CLIP] Loaded from cache (instant)")
            self.model = self._model_cache[cache_key]
            self.processor = self._processor_cache[cache_key]
            return

        logger.info(f"[CLIP] Loading offline CLIP model from: {self.clip_model_dir}")
        start = time.time()

        try:
            self.model = CLIPModel.from_pretrained(
                self.clip_model_dir,
                local_files_only=True
            ).to(self.device)

            self.processor = CLIPProcessor.from_pretrained(
                self.clip_model_dir,
                local_files_only=True
            )

        except Exception as e:
            logger.error(f"[CLIP] FAILED to load offline model: {e}")
            raise RuntimeError(
                "Offline CLIP model missing required files (config.json, pytorch_model.bin)"
            )

        # Cache after successful load
        self._model_cache[cache_key] = self.model
        self._processor_cache[cache_key] = self.processor

        logger.info(f"[CLIP] Model loaded in {time.time() - start:.2f}s")

    # -------------------------------------------------------------
    # ALLOWED FILE EXTENSIONS
    # -------------------------------------------------------------
    def allowed_file(self, filename):
        return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS

    # -------------------------------------------------------------
    # PREPROCESSING
    # -------------------------------------------------------------
    def preprocess_image(self, image):
        if not self.processor:
            raise RuntimeError("CLIP processor not loaded")

        image = image.resize((224, 224))
        inputs = self.processor(images=image, return_tensors="pt", padding=True)
        return inputs.to(self.device)

    # -------------------------------------------------------------
    # EMBEDDING
    # -------------------------------------------------------------
    def get_image_embedding(self, image):
        try:
            inputs = self.preprocess_image(image)

            with torch.no_grad():
                img_features = self.model.get_image_features(**inputs)
                img_features = img_features / img_features.norm(dim=-1, keepdim=True)

            embedding = img_features.cpu().numpy().flatten().tolist()

            logger.debug(f"[CLIP] Embedding generated ({len(embedding)} dims)")
            return embedding

        except Exception as e:
            logger.error(f"[CLIP] Error generating embedding: {e}")
            return None

    # -------------------------------------------------------------
    # VALIDATION
    # -------------------------------------------------------------
    def is_valid_image(self, image):
        try:
            w, h = image.size
            if w < 100 or h < 100:
                return False
            if w > 5000 or h > 5000:
                return False

            aspect = w / h
            return 0.2 <= aspect <= 5.0

        except Exception as e:
            logger.error(f"Image validation failed: {e}")
            return False

    # -------------------------------------------------------------
    # DB STORAGE (single clean version)
    # -------------------------------------------------------------
    def store_image_metadata(self, session, title, description, file_path, embedding, model_name):
        from modules.emtacdb.emtacdb_fts import Image, ImageEmbedding

        # Ensure relative path
        if os.path.isabs(file_path):
            file_path = os.path.relpath(file_path, BASE_DIR)

        img = Image(title=title, description=description, file_path=file_path)
        session.add(img)
        session.commit()

        try:
            if isinstance(embedding, list):
                emb_list = embedding
            elif hasattr(embedding, "tolist"):
                emb_list = embedding.tolist()
            else:
                emb_list = list(embedding)

            emb = ImageEmbedding.create_with_pgvector(
                image_id=img.id,
                model_name=model_name,
                embedding=emb_list
            )

            session.add(emb)
            session.commit()
            logger.info(f"[CLIP] Stored pgvector embedding for {file_path}")

        except Exception as e:
            logger.error(f"[CLIP] Failed pgvector store: {e}")
            # legacy fallback
            emb = ImageEmbedding.create_with_legacy(
                image_id=img.id,
                model_name=model_name,
                embedding=embedding
            )
            session.add(emb)
            session.commit()

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
                return {"similarity": 0, "error": "Embedding failed"}

            dot = sum(a * b for a, b in zip(emb1, emb2))
            n1 = math.sqrt(sum(a * a for a in emb1))
            n2 = math.sqrt(sum(b * b for b in emb2))
            sim = dot / (n1 * n2) if n1 and n2 else 0

            return {
                "similarity": float(sim),
                "image1": image1_path,
                "image2": image2_path,
                "model": self.model_name
            }

        except Exception as e:
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
                similarity_threshold=similarity_threshold
            )

        except Exception as e:
            logger.error(f"[CLIP] search_similar_images_in_db failed: {e}")
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
    # PRELOAD FOR STARTUP
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
            logger.error(f"[CLIP] Preload failed: {e}")
            return False

