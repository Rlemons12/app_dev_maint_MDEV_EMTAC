"""Base abstract interface for image model handlers."""


from __future__ import annotations


import os

from abc import ABC, abstractmethod
from modules.configuration.config import BASE_DIR
from modules.configuration.log_config import logger, with_request_id, debug_id, info_id, warning_id, error_id, get_request_id, log_timed_operation

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
