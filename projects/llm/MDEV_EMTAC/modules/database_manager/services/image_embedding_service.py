from __future__ import annotations

import json
from typing import Any, Optional

import numpy as np
from sqlalchemy import func, text
from sqlalchemy.orm import Session

from modules.emtacdb.emtacdb_fts import ImageEmbedding


class ImageEmbeddingService:
    """
    Service layer for ImageEmbedding.

    Rules:
    - accepts an active SQLAlchemy Session
    - does not open/close sessions
    - does not commit/rollback
    """

    @staticmethod
    def get_by_id(session: Session, embedding_id: int) -> Optional[ImageEmbedding]:
        return (
            session.query(ImageEmbedding)
            .filter(ImageEmbedding.id == embedding_id)
            .first()
        )

    @staticmethod
    def get_by_image_and_model(
        session: Session,
        *,
        image_id: int,
        model_name: str,
    ) -> Optional[ImageEmbedding]:
        return (
            session.query(ImageEmbedding)
            .filter(
                ImageEmbedding.image_id == image_id,
                ImageEmbedding.model_name == model_name,
            )
            .first()
        )

    @staticmethod
    def create_with_pgvector(
        session: Session,
        *,
        image_id: int,
        model_name: str,
        embedding: list[float],
        **kwargs,
    ) -> ImageEmbedding:
        embedding_row = ImageEmbedding.create_with_pgvector(
            image_id=image_id,
            model_name=model_name,
            embedding=embedding,
            **kwargs,
        )
        session.add(embedding_row)
        session.flush()
        return embedding_row

    @staticmethod
    def create_with_legacy(
        session: Session,
        *,
        image_id: int,
        model_name: str,
        embedding: list[float],
        **kwargs,
    ) -> ImageEmbedding:
        embedding_row = ImageEmbedding.create_with_legacy(
            image_id=image_id,
            model_name=model_name,
            embedding=embedding,
            **kwargs,
        )
        session.add(embedding_row)
        session.flush()
        return embedding_row

    @staticmethod
    def create_or_update_for_image(
        session: Session,
        *,
        image_id: int,
        model_name: str,
        embedding: list[float],
        prefer_pgvector: bool = True,
    ) -> ImageEmbedding:
        existing = ImageEmbeddingService.get_by_image_and_model(
            session,
            image_id=image_id,
            model_name=model_name,
        )

        if existing:
            if prefer_pgvector:
                existing.embedding_vector = embedding
            else:
                existing.model_embedding = np.array(embedding, dtype=np.float32).tobytes()
            session.flush()
            return existing

        if prefer_pgvector:
            return ImageEmbeddingService.create_with_pgvector(
                session,
                image_id=image_id,
                model_name=model_name,
                embedding=embedding,
            )

        return ImageEmbeddingService.create_with_legacy(
            session,
            image_id=image_id,
            model_name=model_name,
            embedding=embedding,
        )

    @staticmethod
    def embedding_to_list(embedding_row: ImageEmbedding) -> list[float]:
        return embedding_row.embedding_as_list

    @staticmethod
    def get_storage_type(embedding_row: ImageEmbedding) -> str:
        return embedding_row.get_storage_type()

    @staticmethod
    def migrate_to_pgvector(
        session: Session,
        *,
        embedding_row: ImageEmbedding,
    ) -> bool:
        migrated = embedding_row.migrate_to_pgvector()
        if migrated:
            session.add(embedding_row)
            session.flush()
        return migrated

    @staticmethod
    def migrate_all_to_pgvector(session: Session) -> dict:
        legacy_embeddings = (
            session.query(ImageEmbedding)
            .filter(
                ImageEmbedding.embedding_vector.is_(None),
                ImageEmbedding.model_embedding.isnot(None),
            )
            .all()
        )

        total_count = len(legacy_embeddings)
        migrated_count = 0
        failed_count = 0

        for embedding in legacy_embeddings:
            try:
                if embedding.migrate_to_pgvector():
                    session.add(embedding)
                    migrated_count += 1
                else:
                    failed_count += 1
            except Exception:
                failed_count += 1

        session.flush()

        return {
            "total": total_count,
            "migrated": migrated_count,
            "failed": failed_count,
            "success_rate": (migrated_count / total_count * 100) if total_count > 0 else 0,
        }

    @staticmethod
    def create_pgvector_indexes(session: Session) -> bool:
        indexes = [
            """
            CREATE INDEX IF NOT EXISTS idx_image_embedding_cosine
            ON image_embedding
            USING hnsw (embedding_vector vector_cosine_ops)
            WITH (m = 16, ef_construction = 64);
            """,
            """
            CREATE INDEX IF NOT EXISTS idx_image_embedding_l2
            ON image_embedding
            USING hnsw (embedding_vector vector_l2_ops)
            WITH (m = 16, ef_construction = 64);
            """,
            "CREATE INDEX IF NOT EXISTS idx_image_embedding_model ON image_embedding (model_name);",
            "CREATE INDEX IF NOT EXISTS idx_image_embedding_image_id ON image_embedding (image_id);",
            "CREATE INDEX IF NOT EXISTS idx_image_embedding_created ON image_embedding (created_at);",
        ]

        try:
            for index_sql in indexes:
                session.execute(text(index_sql))
            session.flush()
            return True
        except Exception:
            return False

    @staticmethod
    def has_pgvector_embedding(
        session: Session,
        *,
        image_id: int,
        model_name: str = "CLIPModelHandler",
    ) -> bool:
        embedding = (
            session.query(ImageEmbedding)
            .filter(
                ImageEmbedding.image_id == image_id,
                ImageEmbedding.model_name == model_name,
                ImageEmbedding.embedding_vector.isnot(None),
            )
            .first()
        )
        return embedding is not None

    @staticmethod
    def get_statistics(session: Session) -> dict:
        total_embeddings = session.query(ImageEmbedding).count()

        pgvector_embeddings = (
            session.query(ImageEmbedding)
            .filter(ImageEmbedding.embedding_vector.isnot(None))
            .count()
        )

        legacy_embeddings = (
            session.query(ImageEmbedding)
            .filter(
                ImageEmbedding.model_embedding.isnot(None),
                ImageEmbedding.embedding_vector.is_(None),
            )
            .count()
        )

        both_formats = (
            session.query(ImageEmbedding)
            .filter(
                ImageEmbedding.embedding_vector.isnot(None),
                ImageEmbedding.model_embedding.isnot(None),
            )
            .count()
        )

        model_stats = (
            session.query(
                ImageEmbedding.model_name,
                func.count(ImageEmbedding.id).label("count"),
            )
            .group_by(ImageEmbedding.model_name)
            .all()
        )

        return {
            "total_embeddings": total_embeddings,
            "pgvector_embeddings": pgvector_embeddings,
            "legacy_embeddings": legacy_embeddings,
            "both_formats": both_formats,
            "pgvector_percentage": (pgvector_embeddings / total_embeddings * 100) if total_embeddings > 0 else 0,
            "legacy_percentage": (legacy_embeddings / total_embeddings * 100) if total_embeddings > 0 else 0,
            "models": {model: count for model, count in model_stats},
        }

    @staticmethod
    def search_similar_images(
        session: Session,
        *,
        query_embedding: list[float],
        model_name: str = "CLIPModelHandler",
        limit: int = 10,
        similarity_threshold: float = 0.7,
    ) -> list[dict]:
        """
        Thin wrapper around the model implementation so the orchestrator/service
        layer owns the workflow while reusing the current pgvector search logic.
        """
        return ImageEmbedding.search_similar_images(
            session=session,
            query_embedding=query_embedding,
            model_name=model_name,
            limit=limit,
            similarity_threshold=similarity_threshold,
        )

    @staticmethod
    def find_similar_to_image(
        session: Session,
        *,
        image_id: int,
        model_name: str = "CLIPModelHandler",
        limit: int = 10,
        exclude_self: bool = True,
    ) -> list[dict]:
        return ImageEmbedding.find_similar_to_image(
            session=session,
            image_id=image_id,
            model_name=model_name,
            limit=limit,
            exclude_self=exclude_self,
        )