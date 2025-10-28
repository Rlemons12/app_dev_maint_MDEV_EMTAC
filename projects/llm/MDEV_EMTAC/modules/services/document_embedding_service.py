# modules/services/document_embedding_service.py

from typing import List, Optional, Dict
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session
from sqlalchemy import text

from modules.configuration.config_env import DatabaseConfig
from modules.configuration.log_config import (
    info_id, debug_id, warning_id, error_id, with_request_id, get_request_id
)
from modules.emtacdb.emtacdb_fts import Document, DocumentEmbedding


class DocumentEmbeddingService:
    """Service layer for managing DocumentEmbedding objects."""

    def __init__(self, db_config: Optional[DatabaseConfig] = None):
        self.db_config = db_config or DatabaseConfig()

    # ----------------------------
    # CREATE / ADD
    # ----------------------------
    @with_request_id
    def add_embedding(
        self,
        session: Session,
        document_id: int,
        model_name: str,
        embedding: List[float],
        use_pgvector: bool = True,
        metadata: Optional[Dict] = None,
        request_id: Optional[str] = None
    ) -> Optional[int]:
        """
        Add a new document embedding (pgvector preferred, fallback to legacy).
        """
        try:
            if use_pgvector:
                embedding_obj = DocumentEmbedding.create_with_pgvector(
                    document_id=document_id,
                    model_name=model_name,
                    embedding=embedding,
                    embedding_metadata=metadata or {}
                )
            else:
                embedding_obj = DocumentEmbedding.create_with_legacy(
                    document_id=document_id,
                    model_name=model_name,
                    embedding=embedding,
                    embedding_metadata=metadata or {}
                )

            session.add(embedding_obj)
            session.commit()

            info_id(
                f"Created new DocumentEmbedding for document_id={document_id}, "
                f"model={model_name}, id={embedding_obj.id}",
                request_id
            )
            return embedding_obj.id
        except SQLAlchemyError as e:
            session.rollback()
            error_id(f"Failed to add DocumentEmbedding: {e}", request_id)
            return None

    # ----------------------------
    # RETRIEVE
    # ----------------------------
    @with_request_id
    def get_by_id(self, embedding_id: int, session: Optional[Session] = None, request_id: Optional[str] = None) -> Optional[DocumentEmbedding]:
        """Retrieve embedding by its ID."""
        rid = request_id or get_request_id()
        session_provided = session is not None

        if not session_provided:
            session = self.db_config.get_main_session()
            debug_id("Created new session in get_by_id", rid)

        try:
            embedding = session.query(DocumentEmbedding).filter_by(id=embedding_id).first()
            if embedding:
                debug_id(f"Found embedding ID={embedding_id}, dims={embedding.actual_dimensions}", rid)
            else:
                warning_id(f"No embedding found with ID={embedding_id}", rid)
            return embedding
        finally:
            if not session_provided:
                session.close()

    @with_request_id
    def get_by_document(self, document_id: int, session: Optional[Session] = None, request_id: Optional[str] = None) -> List[DocumentEmbedding]:
        """Retrieve all embeddings for a given document ID."""
        rid = request_id or get_request_id()
        session_provided = session is not None

        if not session_provided:
            session = self.db_config.get_main_session()
            debug_id("Created new session in get_by_document", rid)

        try:
            embeddings = session.query(DocumentEmbedding).filter_by(document_id=document_id).all()
            debug_id(f"Found {len(embeddings)} embeddings for document_id={document_id}", rid)
            return embeddings
        finally:
            if not session_provided:
                session.close()

    # ----------------------------
    # SEARCH
    # ----------------------------
    @with_request_id
    def search_similar(
        self,
        query_embedding: List[float],
        model_name: Optional[str] = None,
        limit: int = 5,
        threshold: float = 0.5,
        session: Optional[Session] = None,
        request_id: Optional[str] = None
    ) -> List[Dict]:
        """
        Search for document chunks similar to a given embedding.
        """
        rid = request_id or get_request_id()
        session_provided = session is not None

        if not session_provided:
            session = self.db_config.get_main_session()
            debug_id("Created new session in search_similar", rid)

        try:
            results = DocumentEmbedding.search_similar_images(  # NOTE: we can rename this to search_similar_chunks later
                session=session,
                query_embedding=query_embedding,
                model_name=model_name,
                limit=limit,
                similarity_threshold=threshold
            )
            info_id(f"Found {len(results)} similar document embeddings", rid)
            return results
        except Exception as e:
            error_id(f"Error in search_similar: {e}", rid)
            return []
        finally:
            if not session_provided:
                session.close()

    # ----------------------------
    # MIGRATION & MAINTENANCE
    # ----------------------------
    @with_request_id
    def migrate_all_to_pgvector(self, session: Optional[Session] = None, request_id: Optional[str] = None) -> Dict:
        """
        Migrate all legacy embeddings to pgvector format.
        """
        rid = request_id or get_request_id()
        session_provided = session is not None

        if not session_provided:
            session = self.db_config.get_main_session()
            debug_id("Created new session in migrate_all_to_pgvector", rid)

        try:
            legacy_embeddings = session.query(DocumentEmbedding).filter(
                DocumentEmbedding.embedding_vector.is_(None),
                DocumentEmbedding.model_embedding.isnot(None)
            ).all()

            migrated = 0
            failed = 0
            for emb in legacy_embeddings:
                if emb.migrate_to_pgvector():
                    migrated += 1
                    session.add(emb)
                else:
                    failed += 1

            session.commit()
            info_id(f"Migrated {migrated} embeddings, failed {failed}", rid)
            return {"migrated": migrated, "failed": failed}
        except Exception as e:
            error_id(f"Error in migrate_all_to_pgvector: {e}", rid)
            session.rollback()
            return {"migrated": 0, "failed": 0}
        finally:
            if not session_provided:
                session.close()

    @with_request_id
    def setup_pgvector_indexes(self, session: Optional[Session] = None, request_id: Optional[str] = None) -> bool:
        """
        Create optimized indexes for pgvector searches on document embeddings.
        """
        rid = request_id or get_request_id()
        session_provided = session is not None

        if not session_provided:
            session = self.db_config.get_main_session()

        try:
            success = DocumentEmbedding.create_pgvector_indexes(session)
            if success:
                info_id("pgvector indexes for DocumentEmbedding created successfully", rid)
            else:
                warning_id("Some or all pgvector indexes for DocumentEmbedding failed", rid)
            return success
        finally:
            if not session_provided:
                session.close()
