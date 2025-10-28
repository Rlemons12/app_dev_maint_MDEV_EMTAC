# services/image_service.py
from typing import Optional, List, Dict, Any
from sqlalchemy.exc import SQLAlchemyError

from modules.emtacdb.emtacdb_fts import Image
from modules.configuration.config_env import DatabaseConfig
from modules.configuration.log_config import info_id, error_id, with_request_id


class ImageService:
    """
    Service layer for managing Image entities.

    Provides:
      - `add_to_db`       → Wrapper for Image.add_to_db
      - `find`            → Wraps Image.search
      - `get`             → Retrieve an Image by ID
      - `remove`          → Delete an Image by ID
      - `find_related`    → Collect associations for an Image
      - `search_images`   → Wrapper for Image.search_images with pgvector support
      - `similarity`      → Embedding-based similarity search
    """

    def __init__(self, db_config: DatabaseConfig = None):
        self.db_config = db_config or DatabaseConfig()

    # ------------------------
    # CORE WRAPPERS
    # ------------------------

    @with_request_id
    def add_to_db(self,
                  title: str,
                  file_path: str,
                  description: str,
                  position_id: Optional[int] = None,
                  complete_document_id: Optional[int] = None,
                  metadata: Optional[Dict[str, Any]] = None) -> Optional[int]:
        """Wrapper for Image.add_to_db → returns new image ID or None on failure."""
        with self.db_config.main_session() as session:
            try:
                return Image.add_to_db(session,
                                       title=title,
                                       file_path=file_path,
                                       description=description,
                                       position_id=position_id,
                                       complete_document_id=complete_document_id,
                                       metadata=metadata)
            except SQLAlchemyError as e:
                error_id(f"ImageService.add_to_db failed: {e}", None)
                raise

    @with_request_id
    def find(self, **filters) -> List[Image]:
        """Search images using Image.search."""
        try:
            return Image.search(**filters)
        except SQLAlchemyError as e:
            error_id(f"ImageService.find failed: {e}", None)
            raise

    @with_request_id
    def get(self, image_id: int) -> Optional[Image]:
        """Retrieve an Image by ID."""
        with self.db_config.main_session() as session:
            try:
                return session.query(Image).filter_by(id=image_id).first()
            except SQLAlchemyError as e:
                error_id(f"ImageService.get failed: {e}", None)
                raise

    @with_request_id
    def remove(self, image_id: int) -> bool:
        """Delete an Image by ID."""
        with self.db_config.main_session() as session:
            try:
                image = session.query(Image).filter_by(id=image_id).first()
                if image:
                    session.delete(image)
                    info_id(f"Deleted Image id={image_id}", None)
                    return True
                return False
            except SQLAlchemyError as e:
                error_id(f"ImageService.remove failed: {e}", None)
                raise

    # ------------------------
    # ADVANCED HELPERS
    # ------------------------

    @with_request_id
    def find_related(self, image_id: int) -> Optional[Dict[str, Any]]:
        """Return related associations for an image."""
        with self.db_config.main_session() as session:
            try:
                image = session.query(Image).filter_by(id=image_id).first()
                if not image:
                    return None
                return {
                    "image": image,
                    "downward": {
                        "parts_positions": image.parts_position_image,
                        "problems": image.image_problem,
                        "tasks": image.image_task,
                        "completed_documents": image.image_completed_document_association,
                        "embeddings": image.image_embedding,
                        "positions": image.image_position_association,
                        "tools": image.tool_image_association,
                    }
                }
            except SQLAlchemyError as e:
                error_id(f"ImageService.find_related failed: {e}", None)
                raise

    @with_request_id
    def search_images(self, **kwargs) -> List[Dict[str, Any]]:
        """Wrapper for Image.search_images (with pgvector/hybrid ranking support)."""
        with self.db_config.main_session() as session:
            try:
                return Image.search_images(session=session, **kwargs)
            except SQLAlchemyError as e:
                error_id(f"ImageService.search_images failed: {e}", None)
                raise

    @with_request_id
    def similarity(self, query_embedding, model_name="CLIPModelHandler",
                   limit: int = 10, threshold: float = 0.7) -> List[Dict[str, Any]]:
        """Find similar images via pgvector embedding search."""
        with self.db_config.main_session() as session:
            try:
                return Image.search_similar_images_by_embedding(
                    session=session,
                    query_embedding=query_embedding,
                    model_name=model_name,
                    limit=limit,
                    similarity_threshold=threshold
                )
            except SQLAlchemyError as e:
                error_id(f"ImageService.similarity failed: {e}", None)
                raise
