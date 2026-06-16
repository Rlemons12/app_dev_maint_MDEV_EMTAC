# modules/ai/config/models_config.py
"""
Extracted ModelsConfig and related configuration helpers
from legacy plugins/ai_modules/ai_models/ai_models.py.

Updated for the refactored modules/ai layout:
- uses shared database config via get_db_config()
- removes self-import recursion
- removes local engine/session duplication
- loads image models from modules.ai.image.models.*
- keeps legacy-safe helper functions still used elsewhere
"""

from __future__ import annotations

import importlib
import json
import logging
import os
from datetime import datetime
from typing import Optional

import torch
from sqlalchemy import (
    Column,
    DateTime,
    Enum,
    Integer,
    String,
    UniqueConstraint,
)

from modules.configuration.base import Base
from modules.configuration.config import (
    ANTHROPIC_API_KEY,
    DATABASE_URL,
    HF_HOME,
    HF_HUB_CACHE,
    MODEL_CLIP_DIR,
    MODELS_DIR,
    MODELS_IMAGE_DIR,
    MODELS_LLM_DIR,
    MODELS_QWEN_DIR,
    MODELS_TINY_LLAMA_DIR,
    OPENAI_API_KEY,
    OPENAI_MODEL_NAME,
    SENTENCE_TRANSFORMERS_MODELS_PATH,
)
from modules.configuration.config_env import get_db_config

# ---------------------------------------------------------------------
# Shared DB objects (single engine / pool per process)
# ---------------------------------------------------------------------
db_config = get_db_config()
engine = db_config.get_engine()
Session = db_config.get_main_session_registry()

# ---------------------------------------------------------------------
# Capability flags
# ---------------------------------------------------------------------
QUANTIZATION_AVAILABLE = True
TORCH_COMPILE_AVAILABLE = hasattr(torch, "compile")
TRANSFORMERS_AVAILABLE = True
MODEL_MINILM_DIR_ENV = os.getenv("MODEL_MINILM_DIR")

logger = logging.getLogger(__name__)


class ModelsConfig(Base):
    __tablename__ = "models_config"

    id = Column(Integer, primary_key=True)
    model_type = Column(
        Enum("ai", "image", "embedding", name="model_type_enum"),
        nullable=False,
    )
    key = Column(String(255), nullable=False)
    value = Column(String(1000), nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    __table_args__ = (
        UniqueConstraint("model_type", "key", name="unique_model_type_key"),
    )

    def __repr__(self):
        return f"<ModelsConfig(model_type='{self.model_type}', key='{self.key}')>"

    # ---------------------------------------------------
    # Internal session helper
    # ---------------------------------------------------
    @staticmethod
    def _get_session():
        return Session()

    # ---------------------------------------------------
    # Config loading helpers (legacy-safe)
    # ---------------------------------------------------
    @staticmethod
    def load_config_from_db():
        session = ModelsConfig._get_session()
        try:
            ai = session.query(ModelsConfig).filter_by(
                model_type="ai",
                key="CURRENT_MODEL",
            ).first()

            emb = session.query(ModelsConfig).filter_by(
                model_type="embedding",
                key="CURRENT_MODEL",
            ).first()

            return (
                ai.value if ai else "NoAIModel",
                emb.value if emb else "NoEmbeddingModel",
            )
        finally:
            session.close()

    @staticmethod
    def load_image_model_config_from_db():
        session = ModelsConfig._get_session()
        try:
            img = session.query(ModelsConfig).filter_by(
                model_type="image",
                key="CURRENT_MODEL",
            ).first()
            return img.value if img else "NoImageModel"
        finally:
            session.close()

    # ---------------------------------------------------
    # Generic config access
    # ---------------------------------------------------
    @classmethod
    def set_config_value(cls, model_type, key, value):
        session = cls._get_session()
        try:
            config = session.query(cls).filter_by(
                model_type=model_type,
                key=key,
            ).first()

            if config:
                config.value = value
                config.updated_at = datetime.utcnow()
            else:
                session.add(
                    cls(
                        model_type=model_type,
                        key=key,
                        value=value,
                    )
                )

            session.commit()
            logger.info("[ModelsConfig] Set %s.%s = %s", model_type, key, value)
            return True

        except Exception as e:
            session.rollback()
            logger.error("[ModelsConfig] Error setting %s.%s: %s", model_type, key, e)
            return False

        finally:
            session.close()

    @classmethod
    def get_config_value(cls, model_type, key, default=None):
        session = cls._get_session()
        try:
            config = session.query(cls).filter_by(
                model_type=model_type,
                key=key,
            ).first()
            return config.value if config else default

        except Exception as e:
            logger.error("[ModelsConfig] Error reading %s.%s: %s", model_type, key, e)
            return default

        finally:
            session.close()

    # ---------------------------------------------------
    # Convenience setters
    # ---------------------------------------------------
    @classmethod
    def set_current_ai_model(cls, model_name):
        return cls.set_config_value("ai", "CURRENT_MODEL", model_name)

    @classmethod
    def set_current_embedding_model(cls, model_name):
        return cls.set_config_value("embedding", "CURRENT_MODEL", model_name)

    @classmethod
    def set_current_image_model(cls, model_name):
        return cls.set_config_value("image", "CURRENT_MODEL", model_name)

    # ---------------------------------------------------
    # Initialization (safe defaults)
    # ---------------------------------------------------
    @classmethod
    def initialize_models_config_table(cls):
        """
        Initialize model configuration with safe, explicit defaults.

        - Does NOT overwrite existing DB values
        - Uses registry-based embedding model names
        - Explicitly sets execution backend where required
        """

        # -------------------------------
        # AI (LLM)
        # -------------------------------
        if not cls.get_config_value("ai", "CURRENT_MODEL"):
            cls.set_current_ai_model("TinyLlamaModel")

        if not cls.get_config_value("ai", "EXECUTION_BACKEND"):
            cls.set_config_value("ai", "EXECUTION_BACKEND", "gpu_service")

        # -------------------------------
        # EMBEDDING
        # -------------------------------
        if not cls.get_config_value("embedding", "CURRENT_MODEL"):
            cls.set_current_embedding_model("all-MiniLM-L6-v2")

        if not cls.get_config_value("embedding", "EXECUTION_BACKEND"):
            cls.set_config_value("embedding", "EXECUTION_BACKEND", "gpu_service")

        # -------------------------------
        # IMAGE
        # -------------------------------
        if not cls.get_config_value("image", "CURRENT_MODEL"):
            cls.set_current_image_model("CLIPModelHandler")

        if not cls.get_config_value("image", "EXECUTION_BACKEND"):
            cls.set_config_value("image", "EXECUTION_BACKEND", "local")

        logger.info("[ModelsConfig] Model configuration initialized")

    # ---------------------------------------------------
    # Model lists / metadata
    # ---------------------------------------------------
    @classmethod
    def get_available_models(cls, model_type):
        raw = cls.get_config_value(model_type, "available_models", "[]")
        try:
            parsed = json.loads(raw)
            return parsed if isinstance(parsed, list) else []
        except Exception:
            logger.error("[ModelsConfig] Invalid available_models for %s", model_type)
            return []

    @classmethod
    def get_enabled_models(cls, model_type):
        return [
            m for m in cls.get_available_models(model_type)
            if m.get("enabled", True)
        ]

    @classmethod
    def get_current_model_info(cls, model_type):
        name = cls.get_config_value(model_type, "CURRENT_MODEL")
        if not name:
            return None

        for model in cls.get_available_models(model_type):
            if model.get("name") == name:
                return model

        return None

    # ---------------------------------------------------
    # Deprecated AI model loading
    # ---------------------------------------------------
    @classmethod
    def load_ai_model(cls, model_name=None):
        """
        DEPRECATED.

        AI models are executed via GPU service.
        This method exists only to prevent crashes in legacy paths.
        """
        logger.warning(
            "[ModelsConfig] load_ai_model() is deprecated. "
            "AI execution must go through AIModelsService + GPU service."
        )
        return cls._fallback_ai_model()

    # ---------------------------------------------------
    # Embedding models
    # ---------------------------------------------------
    @classmethod
    def load_embedding_model(cls, model_name: str | None = None):
        if model_name is None:
            model_name = cls.get_config_value(
                "embedding",
                "CURRENT_MODEL",
                "NoEmbeddingModel",
            )

        model_info = cls.get_current_model_info("embedding")
        if not model_info:
            raise RuntimeError(
                f"[ModelsConfig] Embedding model '{model_name}' not registered"
            )

        backend = (model_info.get("backend") or "").strip().lower()
        path = model_info.get("path")

        logger.info(
            "[ModelsConfig] Loading embedding model: name=%s backend=%s path=%s",
            model_name,
            backend,
            path,
        )

        if backend == "gpu_service":
            raise RuntimeError(
                "GPU embedding models are loaded by AIModelsEmbeddingService "
                "via GPUServerAdapter — not locally"
            )

        if backend == "local":
            if not path:
                raise RuntimeError(
                    f"Embedding model '{model_name}' missing path"
                )

            from sentence_transformers import SentenceTransformer

            logger.info(
                "[ModelsConfig] Loading SentenceTransformer from: %s",
                path,
            )

            st_model = SentenceTransformer(path)

            class _SentenceTransformerAdapter:
                def __init__(self, model):
                    self.model = model

                def get_embeddings(self, text: str):
                    if not isinstance(text, str) or not text.strip():
                        raise RuntimeError("Embedding text must be non-empty")

                    vec = self.model.encode(text)

                    if vec is None or len(vec) == 0:
                        raise RuntimeError("SentenceTransformer returned empty vector")

                    return vec.tolist()

                def get_embeddings_batch(self, texts):
                    if not isinstance(texts, (list, tuple)) or not texts:
                        raise RuntimeError(
                            "Embedding texts must be a non-empty list or tuple"
                        )

                    cleaned = []
                    for idx, text in enumerate(texts):
                        if not isinstance(text, str) or not text.strip():
                            raise RuntimeError(
                                f"Embedding texts[{idx}] must be a non-empty string"
                            )
                        cleaned.append(text)

                    vecs = self.model.encode(cleaned)

                    if vecs is None or len(vecs) == 0:
                        raise RuntimeError(
                            "SentenceTransformer returned empty batch vectors"
                        )

                    return [
                        v.tolist() if hasattr(v, "tolist") else list(v)
                        for v in vecs
                    ]

            return _SentenceTransformerAdapter(st_model)

        raise RuntimeError(
            f"Unknown embedding backend '{backend}' for model '{model_name}'"
        )

    # ---------------------------------------------------
    # Image models
    # ---------------------------------------------------
    @classmethod
    def load_image_model(cls, model_name=None):
        if model_name is None:
            model_name = cls.get_config_value(
                "image",
                "CURRENT_MODEL",
                "NoImageModel",
            )

        registry = {
            "CLIPModelHandler": (
                "modules.ai.image.models.clip_model_handler",
                "CLIPModelHandler",
            ),
            "NoImageModel": (
                "modules.ai.image.models.no_image_model",
                "NoImageModel",
            ),
        }

        entry = registry.get(model_name)
        if not entry:
            raise RuntimeError(
                f"[ModelsConfig] Unknown image model '{model_name}'. "
                f"Registered image models: {list(registry.keys())}"
            )

        module_name, class_name = entry

        try:
            logger.info(
                "[ModelsConfig] Loading image model '%s' from %s.%s",
                model_name,
                module_name,
                class_name,
            )

            module = importlib.import_module(module_name)
            model_cls = getattr(module, class_name)
            model = model_cls()

            logger.info(
                "[ModelsConfig] Image model loaded successfully: %s -> %s",
                model_name,
                type(model).__name__,
            )
            return model

        except Exception as e:
            logger.error(
                "[ModelsConfig] Image model load failed for '%s': %s",
                model_name,
                e,
                exc_info=True,
            )
            raise RuntimeError(
                f"Failed to load image model '{model_name}': {e}"
            ) from e

    # ---------------------------------------------------
    # Fallback implementations
    # ---------------------------------------------------
    @staticmethod
    def _fallback_ai_model():
        class FallbackAI:
            def get_response(self, prompt):
                return "AI service unavailable."

            def generate_description(self, image_path):
                return "Unavailable."

        return FallbackAI()

    @staticmethod
    def _fallback_embedding_model():
        class FallbackEmbedding:
            def get_embeddings(self, text):
                return []

        return FallbackEmbedding()

    @staticmethod
    def _fallback_image_model():
        class FallbackImage:
            def generate_description(self, image_path):
                return "Unavailable."

        return FallbackImage()

    # ---------------------------------------------------
    # Execution backend
    # ---------------------------------------------------
    @classmethod
    def get_execution_backend(cls, model_type, default="gpu_service"):
        return cls.get_config_value(model_type, "EXECUTION_BACKEND", default)

    # ---------------------------------------------------
    # Auto-promote latest trained model
    # ---------------------------------------------------
    @classmethod
    def auto_promote_latest_trained_model(cls):
        """
        Detect latest trained model using ModelResolverConfig
        and set it as CURRENT_MODEL (ai).

        Stores only the model name, not absolute path.
        """
        try:
            from modules.configuration.model_resolver_config import (
                resolve_latest_trained_model,
            )

            resolved = resolve_latest_trained_model(refresh=True)

            model_name = resolved["model_name"]
            version = resolved["version"]

            logger.info(
                "[ModelsConfig] Latest trained model detected → %s (v%s)",
                model_name,
                version,
            )

            ok = cls.set_current_ai_model(model_name)

            if ok:
                logger.info(
                    "[ModelsConfig] Auto-promoted latest trained model → %s (v%s)",
                    model_name,
                    version,
                )
            else:
                logger.error(
                    "[ModelsConfig] Failed to update CURRENT_MODEL to %s",
                    model_name,
                )

            return ok

        except Exception as e:
            logger.error(
                "[ModelsConfig] Auto-promote failed: %s",
                e,
                exc_info=True,
            )
            return False



def get_tinyllama_config():
    """Get TinyLlama-specific configuration values."""
    try:
        model_path = ModelsConfig.get_config_value(
            "ai",
            "TINYLLAMA_MODEL_PATH",
            r"C:\Users\10169062\PycharmProjects\MDEV_EMTAC\plugins\ai_modules\TinyLlama_1_1B",
        )
        timeout = int(ModelsConfig.get_config_value("ai", "TINYLLAMA_TIMEOUT", "120"))
        max_tokens = int(
            ModelsConfig.get_config_value("ai", "TINYLLAMA_MAX_TOKENS", "256")
        )

        return {
            "model_path": model_path,
            "timeout": timeout,
            "max_tokens": max_tokens,
        }

    except Exception as e:
        logger.error("Error getting TinyLlama config: %s", e)
        return {
            "model_path": r"C:\Users\10169062\PycharmProjects\MDEV_EMTAC\plugins\ai_modules\TinyLlama_1_1B",
            "timeout": 120,
            "max_tokens": 256,
        }


def update_tinyllama_config(model_path=None, timeout=None, max_tokens=None):
    """Update TinyLlama-specific configuration values."""
    try:
        updated = False

        if model_path is not None:
            success = ModelsConfig.set_config_value(
                "ai",
                "TINYLLAMA_MODEL_PATH",
                model_path,
            )
            if success:
                logger.info("Updated TinyLlama model path: %s", model_path)
                updated = True

        if timeout is not None:
            success = ModelsConfig.set_config_value(
                "ai",
                "TINYLLAMA_TIMEOUT",
                str(timeout),
            )
            if success:
                logger.info("Updated TinyLlama timeout: %ss", timeout)
                updated = True

        if max_tokens is not None:
            success = ModelsConfig.set_config_value(
                "ai",
                "TINYLLAMA_MAX_TOKENS",
                str(max_tokens),
            )
            if success:
                logger.info("Updated TinyLlama max tokens: %s", max_tokens)
                updated = True

        return updated

    except Exception as e:
        logger.error("Error updating TinyLlama config: %s", e)
        return False


def get_current_models():
    """Get information about all currently active models."""
    try:
        ai_model = ModelsConfig.get_config_value("ai", "CURRENT_MODEL", "NoAIModel")
        embedding_model = ModelsConfig.get_config_value(
            "embedding",
            "CURRENT_MODEL",
            "NoEmbeddingModel",
        )
        image_model = ModelsConfig.get_config_value(
            "image",
            "CURRENT_MODEL",
            "NoImageModel",
        )

        return {
            "ai": ai_model,
            "embedding": embedding_model,
            "image": image_model,
        }

    except Exception as e:
        logger.error("Error getting current models: %s", e)
        return {
            "ai": "NoAIModel",
            "embedding": "NoEmbeddingModel",
            "image": "NoImageModel",
        }


def get_tinyllama_embedding_config():
    """Get TinyLlama embedding-specific configuration values."""
    try:
        embedding_model = ModelsConfig.get_config_value(
            "embedding",
            "TINYLLAMA_EMBEDDING_MODEL",
            "all-MiniLM-L6-v2",
        )
        cache_path = ModelsConfig.get_config_value(
            "embedding",
            "TINYLLAMA_EMBEDDING_CACHE_PATH",
            os.path.join(
                SENTENCE_TRANSFORMERS_MODELS_PATH,
                "tinyllama_embeddings",
            ),
        )

        return {
            "embedding_model": embedding_model,
            "cache_path": cache_path,
        }

    except Exception as e:
        logger.error("Error getting TinyLlama embedding config: %s", e)
        return {
            "embedding_model": "all-MiniLM-L6-v2",
            "cache_path": os.path.join(
                SENTENCE_TRANSFORMERS_MODELS_PATH,
                "tinyllama_embeddings",
            ),
        }
