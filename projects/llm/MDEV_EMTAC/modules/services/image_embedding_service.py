# services/image_embedding_service.py

from __future__ import annotations

from typing import Optional, List, Dict, Any

from sqlalchemy.orm import Session

from modules.emtacdb.emtacdb_fts import ImageEmbedding
from modules.configuration.log_config import (
    info_id,
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
    # INTERNAL HELPERS
    # ---------------------------------------------------------

    @staticmethod
    def _require_session(session: Session, method_name: str) -> None:
        if session is None:
            raise RuntimeError(
                f"Session required for ImageEmbeddingService.{method_name}"
            )

    @staticmethod
    def _require_image_id(image_id: int) -> None:
        if image_id is None:
            raise ValueError("image_id is required")

    @staticmethod
    def _require_embedding_id(embedding_id: int) -> None:
        if embedding_id is None:
            raise ValueError("embedding_id is required")

    @staticmethod
    def _normalize_embedding(embedding: List[float]) -> List[float]:
        if embedding is None:
            raise ValueError("embedding is required")

        if not isinstance(embedding, (list, tuple)):
            raise TypeError("embedding must be a list or tuple")

        normalized = [float(x) for x in embedding]

        if not normalized:
            raise ValueError("embedding must not be empty")

        return normalized

    @staticmethod
    def serialize(embedding_obj: Optional[ImageEmbedding]) -> Optional[Dict[str, Any]]:
        if embedding_obj is None:
            return None

        return {
            "id": getattr(embedding_obj, "id", None),
            "image_id": getattr(embedding_obj, "image_id", None),
            "model_name": getattr(embedding_obj, "model_name", None),
            "storage_type": embedding_obj.get_storage_type() if hasattr(embedding_obj, "get_storage_type") else None,
            "has_pgvector": getattr(embedding_obj, "embedding_vector", None) is not None,
            "has_legacy": getattr(embedding_obj, "model_embedding", None) is not None,
            "created_at": getattr(embedding_obj, "created_at", None),
            "updated_at": getattr(embedding_obj, "updated_at", None),
        }

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
        self._require_session(session, "add_pgvector")
        self._require_image_id(image_id)

        if not model_name or not isinstance(model_name, str):
            raise ValueError("model_name is required")

        normalized_embedding = self._normalize_embedding(embedding)

        embedding_obj = ImageEmbedding.create_with_pgvector(
            image_id=image_id,
            model_name=model_name.strip(),
            embedding=normalized_embedding,
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
        self._require_session(session, "add_legacy")
        self._require_image_id(image_id)

        if not model_name or not isinstance(model_name, str):
            raise ValueError("model_name is required")

        normalized_embedding = self._normalize_embedding(embedding)

        embedding_obj = ImageEmbedding.create_with_legacy(
            image_id=image_id,
            model_name=model_name.strip(),
            embedding=normalized_embedding,
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
        self._require_session(session, "get")
        self._require_embedding_id(embedding_id)

        return session.get(ImageEmbedding, embedding_id)

    @with_request_id
    def get_all_for_image(
        self,
        session: Session,
        *,
        image_id: int,
        request_id: Optional[str] = None,
    ) -> List[ImageEmbedding]:
        self._require_session(session, "get_all_for_image")
        self._require_image_id(image_id)

        rows = (
            session.query(ImageEmbedding)
            .filter(ImageEmbedding.image_id == image_id)
            .order_by(ImageEmbedding.id.asc())
            .all()
        )

        debug_id(
            f"[ImageEmbeddingService] get_all_for_image returned {len(rows)} row(s) for image_id={image_id}",
            request_id,
        )

        return rows

    # ---------------------------------------------------------
    # DELETE
    # ---------------------------------------------------------

    @with_request_id
    def remove(
        self,
        session: Session,
        *,
        embedding_id: int,
        request_id: Optional[str] = None,
    ) -> bool:
        self._require_session(session, "remove")
        self._require_embedding_id(embedding_id)

        embedding = session.get(ImageEmbedding, embedding_id)

        if not embedding:
            warning_id(f"Embedding id={embedding_id} not found", request_id)
            return False

        session.delete(embedding)
        session.flush()

        info_id(f"Embedding staged for deletion id={embedding_id}", request_id)
        return True

    @with_request_id
    def remove_all_for_image(
        self,
        session: Session,
        *,
        image_id: int,
        request_id: Optional[str] = None,
    ) -> int:
        self._require_session(session, "remove_all_for_image")
        self._require_image_id(image_id)

        rows = (
            session.query(ImageEmbedding)
            .filter(ImageEmbedding.image_id == image_id)
            .all()
        )

        removed_count = 0
        for row in rows:
            session.delete(row)
            removed_count += 1

        session.flush()

        info_id(
            f"[ImageEmbeddingService] Removed {removed_count} embedding row(s) for image_id={image_id}",
            request_id,
        )

        return removed_count

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
        self._require_session(session, "migrate")
        self._require_embedding_id(embedding_id)

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
        self._require_session(session, "stats")

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
        self._require_session(session, "similarity")

        normalized_embedding = self._normalize_embedding(query_embedding)

        return ImageEmbedding.search_similar_images(
            session=session,
            query_embedding=normalized_embedding,
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
        self._require_session(session, "find_similar_to")
        self._require_image_id(image_id)

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
        self._require_session(session, "create_indexes")
        return ImageEmbedding.create_pgvector_indexes(session)