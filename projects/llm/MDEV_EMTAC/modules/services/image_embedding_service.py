# services/image_embedding_service.py

from typing import Optional, List, Dict, Any
from sqlalchemy.orm import Session
from sqlalchemy.exc import SQLAlchemyError

from modules.emtacdb.emtacdb_fts import ImageEmbedding
from modules.configuration.log_config import (
    info_id,
    error_id,
    debug_id,
    warning_id,
    with_request_id,
)


class ImageEmbeddingService:
    """
    Pure domain service for ImageEmbedding.

    HARD RULES:
    - NEVER open sessions
    - NEVER close sessions
    - NEVER commit
    - NEVER rollback
    - Orchestrator owns transactions
    """

    # ---------------------------------------------------------
    # CREATE
    # ---------------------------------------------------------

    @with_request_id
    def add_pgvector(
        self,
        session: Session,
        *,
        image_id: int,
        model_name: str,
        embedding: List[float],
        request_id: Optional[str] = None,
    ) -> ImageEmbedding:

        if not session:
            raise RuntimeError("Session required for add_pgvector")

        embedding_obj = ImageEmbedding.create_with_pgvector(
            image_id=image_id,
            model_name=model_name,
            embedding=embedding,
        )

        session.add(embedding_obj)
        session.flush()

        debug_id(
            f"Embedding staged image={image_id}, id={embedding_obj.id}",
            request_id,
        )

        return embedding_obj

    @with_request_id
    def add_legacy(
        self,
        session: Session,
        *,
        image_id: int,
        model_name: str,
        embedding: List[float],
        request_id: Optional[str] = None,
    ) -> ImageEmbedding:

        if not session:
            raise RuntimeError("Session required for add_legacy")

        embedding_obj = ImageEmbedding.create_with_legacy(
            image_id=image_id,
            model_name=model_name,
            embedding=embedding,
        )

        session.add(embedding_obj)
        session.flush()

        debug_id(
            f"Legacy embedding staged image={image_id}, id={embedding_obj.id}",
            request_id,
        )

        return embedding_obj

    # ---------------------------------------------------------
    # READ
    # ---------------------------------------------------------

    @with_request_id
    def get(
        self,
        session: Session,
        *,
        embedding_id: int,
        request_id: Optional[str] = None,
    ) -> Optional[ImageEmbedding]:

        return session.get(ImageEmbedding, embedding_id)

    @with_request_id
    def remove(
        self,
        session: Session,
        *,
        embedding_id: int,
        request_id: Optional[str] = None,
    ) -> bool:

        embedding = session.get(ImageEmbedding, embedding_id)

        if not embedding:
            warning_id(f"Embedding id={embedding_id} not found", request_id)
            return False

        session.delete(embedding)

        info_id(f"Embedding staged for deletion id={embedding_id}", request_id)

        return True

    # ---------------------------------------------------------
    # MIGRATION
    # ---------------------------------------------------------

    @with_request_id
    def migrate(
        self,
        session: Session,
        *,
        embedding_id: int,
        request_id: Optional[str] = None,
    ) -> bool:

        embedding = session.get(ImageEmbedding, embedding_id)

        if not embedding:
            warning_id(f"Embedding id={embedding_id} not found", request_id)
            return False

        if embedding.migrate_to_pgvector():
            session.add(embedding)
            session.flush()

            info_id(
                f"Embedding migrated to pgvector id={embedding_id}",
                request_id,
            )
            return True

        return False

    @with_request_id
    def stats(
        self,
        session: Session,
        *,
        request_id: Optional[str] = None,
    ) -> Dict[str, Any]:

        total = session.query(ImageEmbedding).count()

        pgvector_count = session.query(ImageEmbedding).filter(
            ImageEmbedding.embedding_vector.isnot(None)
        ).count()

        legacy_count = session.query(ImageEmbedding).filter(
            ImageEmbedding.model_embedding.isnot(None)
        ).count()

        return {
            "total": total,
            "pgvector": pgvector_count,
            "legacy": legacy_count,
            "pgvector_percentage": (pgvector_count / total * 100) if total else 0,
            "legacy_percentage": (legacy_count / total * 100) if total else 0,
        }

    # ---------------------------------------------------------
    # SIMILARITY SEARCH
    # ---------------------------------------------------------

    @with_request_id
    def similarity(
        self,
        session: Session,
        *,
        query_embedding: List[float],
        model_name: Optional[str] = None,
        limit: int = 10,
        threshold: float = 0.5,
        request_id: Optional[str] = None,
    ) -> List[Dict[str, Any]]:

        return ImageEmbedding.search_similar_images(
            session=session,
            query_embedding=query_embedding,
            model_name=model_name,
            limit=limit,
            similarity_threshold=threshold,
        )

    @with_request_id
    def find_similar_to(
        self,
        session: Session,
        *,
        image_id: int,
        model_name: Optional[str] = None,
        limit: int = 10,
        exclude_self: bool = True,
        request_id: Optional[str] = None,
    ) -> List[Dict[str, Any]]:

        return ImageEmbedding.find_similar_to_image(
            session=session,
            image_id=image_id,
            model_name=model_name,
            limit=limit,
            exclude_self=exclude_self,
        )

    # ---------------------------------------------------------
    # INDEX CREATION
    # ---------------------------------------------------------

    @with_request_id
    def create_indexes(
        self,
        session: Session,
        *,
        request_id: Optional[str] = None,
    ) -> bool:

        return ImageEmbedding.create_pgvector_indexes(session)
