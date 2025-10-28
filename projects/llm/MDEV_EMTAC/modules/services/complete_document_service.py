# modules/services/complete_document_service.py

from typing import List, Optional, Dict, Tuple
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session

from modules.configuration.config_env import DatabaseConfig
from modules.configuration.log_config import (
    info_id, debug_id, warning_id, error_id, with_request_id, get_request_id
)
from modules.emtacdb.emtacdb_fts import CompleteDocument


class CompleteDocumentService:
    """Service layer for managing CompleteDocument objects."""

    def __init__(self, db_config: Optional[DatabaseConfig] = None):
        self.db_config = db_config or DatabaseConfig()

    # ----------------------------
    # CREATE / ADD
    # ----------------------------
    @with_request_id
    def add(
        self,
        title: str,
        file_path: Optional[str] = None,
        content: Optional[str] = None,
        position_id: Optional[int] = None,
        request_id: Optional[str] = None,
    ) -> Optional[int]:
        """Add a new CompleteDocument (and trigger associations)."""
        try:
            with self.db_config.main_session() as session:
                doc = CompleteDocument(
                    title=title,
                    file_path=file_path,
                    content=content,
                )
                session.add(doc)
                session.flush()

                # Create associations if position provided
                if position_id:
                    CompleteDocument._create_associations(session, doc.id, position_id)

                session.commit()
                info_id(f"Added CompleteDocument id={doc.id}, title='{title}'", request_id)
                return doc.id
        except SQLAlchemyError as e:
            error_id(f"Failed to add CompleteDocument: {e}", request_id)
            return None

    # ----------------------------
    # RETRIEVE
    # ----------------------------
    @with_request_id
    def get_by_id(self, doc_id: int, session: Optional[Session] = None, request_id: Optional[str] = None) -> Optional[CompleteDocument]:
        """Retrieve a CompleteDocument by ID."""
        rid = request_id or get_request_id()
        local_session = None
        if session is None:
            local_session = self.db_config.get_main_session()

        try:
            doc = (session or local_session).query(CompleteDocument).filter_by(id=doc_id).first()
            if doc:
                debug_id(f"Found CompleteDocument {doc}", rid)
            else:
                warning_id(f"No CompleteDocument found for id={doc_id}", rid)
            return doc
        finally:
            if local_session:
                local_session.close()

    @with_request_id
    def list_all(self, limit: int = 50, request_id: Optional[str] = None) -> List[CompleteDocument]:
        """List CompleteDocuments with limit."""
        with self.db_config.main_session() as session:
            return session.query(CompleteDocument).limit(limit).all()

    # ----------------------------
    # SEARCH
    # ----------------------------
    @with_request_id
    def search_by_text(self, query: str, limit: int = 25, request_id: Optional[str] = None):
        """Full-text search wrapper."""
        return CompleteDocument.search_by_text(query, limit=limit, request_id=request_id)

    @with_request_id
    def search_similar_by_embedding(self, query_text: str, limit: int = 10, threshold: float = 0.7, request_id: Optional[str] = None):
        """Vector similarity search wrapper."""
        return CompleteDocument.search_similar_by_embedding(
            query_text=query_text,
            limit=limit,
            threshold=threshold,
            request_id=request_id
        )

    # ----------------------------
    # UPLOAD + IMAGE HANDLING
    # ----------------------------
    @with_request_id
    def process_upload(self, files, metadata: dict, request_id: Optional[str] = None) -> Tuple[bool, dict, int]:
        """
        Delegate to CompleteDocument.process_upload.
        Returns (success, payload, status_code).
        """
        return CompleteDocument.process_upload(files, metadata, request_id=request_id)

    @with_request_id
    def get_images_with_context(self, doc_id: int, request_id: Optional[str] = None):
        """Get all images associated with a CompleteDocument."""
        return CompleteDocument.get_images_with_chunk_context(doc_id, request_id=request_id)

    # ----------------------------
    # ANALYTICS / STATS
    # ----------------------------
    @with_request_id
    def get_embedding_stats(self, request_id: Optional[str] = None) -> Dict:
        """Return embedding distribution stats for all CompleteDocuments."""
        return CompleteDocument.get_embedding_statistics(request_id=request_id)

    @with_request_id
    def get_association_stats(self, doc_id: int, request_id: Optional[str] = None) -> Dict:
        """Return association statistics for images linked to this document."""
        return CompleteDocument.get_association_statistics(doc_id, request_id=request_id)

    # ----------------------------
    # DELETE
    # ----------------------------
    @with_request_id
    def delete(self, doc_id: int, request_id: Optional[str] = None) -> bool:
        """Delete a CompleteDocument by ID."""
        try:
            with self.db_config.main_session() as session:
                doc = session.query(CompleteDocument).filter_by(id=doc_id).first()
                if not doc:
                    warning_id(f"No CompleteDocument found for deletion id={doc_id}", request_id)
                    return False
                session.delete(doc)
                session.commit()
                info_id(f"Deleted CompleteDocument id={doc_id}", request_id)
                return True
        except SQLAlchemyError as e:
            error_id(f"Failed to delete CompleteDocument {doc_id}: {e}", request_id)
            return False
