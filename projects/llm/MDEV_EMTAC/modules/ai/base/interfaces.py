"""Abstract interfaces for AI, embedding, and image model types."""


from __future__ import annotations


import json

from abc import ABC, abstractmethod
from modules.configuration.log_config import logger, with_request_id

class AIModel(ABC):
    @abstractmethod
    def get_response(self, prompt: str) -> str:
        pass

    @abstractmethod
    def generate_description(self, image_path: str) -> str:
        pass

    @classmethod
    def register_available_model(cls, model_type: str, model_info: dict):
        session = cls._get_session()
        try:
            raw = cls.get_config_value(model_type, "available_models", "[]")
            models = json.loads(raw)

            # Prevent duplicates
            if any(m.get("name") == model_info.get("name") for m in models):
                logger.warning(
                    f"[ModelsConfig] Model '{model_info['name']}' already registered"
                )
                return False

            models.append(model_info)

            cls.set_config_value(
                model_type,
                "available_models",
                json.dumps(models),
            )

            logger.info(
                f"[ModelsConfig] Registered new {model_type} model: {model_info['name']}"
            )
            return True

        except Exception as e:
            session.rollback()
            logger.error(f"[ModelsConfig] Failed to register model: {e}")
            return False
        finally:
            session.close()


class EmbeddingModel(ABC):
    @abstractmethod
    def get_embeddings(self, text: str) -> list:
        pass


class ImageModel(ABC):
    @abstractmethod
    def process_image(self, image_path: str) -> str:
        pass

    @abstractmethod
    def compare_images(self, image1_path: str, image2_path: str) -> dict:
        pass

    @abstractmethod
    def generate_description(self, image_path: str) -> str:
        pass
