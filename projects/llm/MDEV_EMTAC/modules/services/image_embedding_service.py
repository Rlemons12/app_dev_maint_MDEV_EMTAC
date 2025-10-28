# services/image_embedding_service.py
from typing import Optional, List, Dict, Any
from sqlalchemy.exc import SQLAlchemyError

from modules.emtacdb.emtacdb_fts import ImageEmbedding
from modules.configuration.config_env import DatabaseConfig
from modules.configuration.log_config import info_id, error_id, with_request_id


class ImageEmbeddingService:
    """
    Service layer for managing ImageEmbedding entities.

    Provides:
      - `add_pgvector`    → Create a new embedding in pgvector format
      - `add_legacy`      → Create a new embedding in legacy LargeBinary format
      - `get`             → Retrieve an embedding by ID
      - `remove`          → Delete an embedding by ID
      - `migrate`         → Migrate legacy embeddings to pgvector
      - `stats`           → Return embedding statistics
      - `similarity`      → Search similar images using query embedding
      - `find_similar_to` → Find images similar to a given image ID
      - `create_indexes`  → Ensure pgvector indexes exist
    """

    def __init__(self, db_config: DatabaseConfig = None):
        self.db_config = db_config or DatabaseConfig()

    # ------------------------
    # CORE CRUD
    # ------------------------

    @with_request_id
    def add_pgvector(self, image_id: int, model_name: str,
                     embedding: List[float]) -> Optional[ImageEmbedding]:
        """Insert a new pgvector embedding for an image."""
        with self.db_config.main_session() as session:
            try:
                embedding_obj = ImageEmbedding.create_with_pgvector(
                    image_id=image_id,
                    model_name=model_name,
                    embedding=embedding
                )
                session.add(embedding_obj)
                session.commit()
                info_id(f"Added pgvector embedding for image {image_id}", None)
                return embedding_obj
            except SQLAlchemyError as e:
                error_id(f"ImageEmbeddingService.add_pgvector failed: {e}", None)
                session.rollback()
                raise

    @with_request_id
    def add_legacy(self, image_id: int, model_name: str,
                   embedding: List[float]) -> Optional[ImageEmbedding]:
        """Insert a new legacy (LargeBinary) embedding for an image."""
        with self.db_config.main_session() as session:
            try:
                embedding_obj = ImageEmbedding.create_with_legacy(
                    image_id=image_id,
                    model_name=model_name,
                    embedding=embedding
                )
                session.add(embedding_obj)
                session.commit()
                info_id(f"Added legacy embedding for image {image_id}", None)
                return embedding_obj
            except SQLAlchemyError as e:
                error_id(f"ImageEmbeddingService.add_legacy failed: {e}", None)
                session.rollback()
                raise

    @with_request_id
    def get(self, embedding_id: int) -> Optional[ImageEmbedding]:
        """Retrieve an ImageEmbedding by ID."""
        with self.db_config.main_session() as session:
            try:
                return session.query(ImageEmbedding).filter_by(id=embedding_id).first()
            except SQLAlchemyError as e:
                error_id(f"ImageEmbeddingService.get failed: {e}", None)
                raise

    @with_request_id
    def remove(self, embedding_id: int) -> bool:
        """Delete an ImageEmbedding by ID."""
        with self.db_config.main_session() as session:
            try:
                embedding = session.query(ImageEmbedding).filter_by(id=embedding_id).first()
                if embedding:
                    session.delete(embedding)
                    info_id(f"Deleted embedding id={embedding_id}", None)
                    return True
                return False
            except SQLAlchemyError as e:
                error_id(f"ImageEmbeddingService.remove failed: {e}", None)
                raise

    # ------------------------
    # MIGRATION & STATS
    # ------------------------

    @with_request_id
    def migrate(self, embedding_id: int) -> bool:
        """Migrate a single legacy embedding to pgvector."""
        with self.db_config.main_session() as session:
            try:
                embedding = session.query(ImageEmbedding).filter_by(id=embedding_id).first()
                if not embedding:
                    return False
                if embedding.migrate_to_pgvector():
                    session.add(embedding)
                    session.commit()
                    info_id(f"Migrated embedding {embedding_id} to pgvector", None)
                    return True
                return False
            except SQLAlchemyError as e:
                error_id(f"ImageEmbeddingService.migrate failed: {e}", None)
                session.rollback()
                raise

    @with_request_id
    def stats(self) -> Dict[str, Any]:
        """Return statistics about embeddings."""
        with self.db_config.main_session() as session:
            try:
                total = session.query(ImageEmbedding).count()
                pgvector_count = session.query(ImageEmbedding).filter(
                    ImageEmbedding.embedding_vector.isnot(None)).count()
                legacy_count = session.query(ImageEmbedding).filter(
                    ImageEmbedding.model_embedding.isnot(None)).count()
                return {
                    "total": total,
                    "pgvector": pgvector_count,
                    "legacy": legacy_count,
                    "pgvector_percentage": (pgvector_count / total * 100) if total else 0,
                    "legacy_percentage": (legacy_count / total * 100) if total else 0,
                }
            except SQLAlchemyError as e:
                error_id(f"ImageEmbeddingService.stats failed: {e}", None)
                raise

    # ------------------------
    # SIMILARITY SEARCH
    # ------------------------

    @with_request_id
    def similarity(self, query_embedding: List[float], model_name: str = None,
                   limit: int = 10, threshold: float = 0.5) -> List[Dict[str, Any]]:
        """Search for images similar to a query embedding."""
        with self.db_config.main_session() as session:
            try:
                return ImageEmbedding.search_similar_images(
                    session=session,
                    query_embedding=query_embedding,
                    model_name=model_name,
                    limit=limit,
                    similarity_threshold=threshold
                )
            except SQLAlchemyError as e:
                error_id(f"ImageEmbeddingService.similarity failed: {e}", None)
                raise

    @with_request_id
    def find_similar_to(self, image_id: int, model_name: str = None,
                        limit: int = 10, exclude_self: bool = True) -> List[Dict[str, Any]]:
        """Find images similar to a given image ID."""
        with self.db_config.main_session() as session:
            try:
                return ImageEmbedding.find_similar_to_image(
                    session=session,
                    image_id=image_id,
                    model_name=model_name,
                    limit=limit,
                    exclude_self=exclude_self
                )
            except SQLAlchemyError as e:
                error_id(f"ImageEmbeddingService.find_similar_to failed: {e}", None)
                raise

    # ------------------------
    # INDEXES
    # ------------------------

    @with_request_id
    def create_indexes(self) -> bool:
        """Ensure pgvector indexes exist for embeddings."""
        with self.db_config.main_session() as session:
            try:
                return ImageEmbedding.create_pgvector_indexes(session)
            except SQLAlchemyError as e:
                error_id(f"ImageEmbeddingService.create_indexes failed: {e}", None)
                session.rollback()
                raise
