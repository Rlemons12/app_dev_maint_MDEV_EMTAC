# services/document_service.py
from typing import Optional, List, Dict, Any
from sqlalchemy.exc import SQLAlchemyError

from modules.emtacdb.emtacdb_fts import Document, Image
from modules.configuration.config_env import DatabaseConfig
from modules.configuration.log_config import info_id, error_id, with_request_id


class DocumentService:
    """
    Service layer for managing Document entities.

    Provides:
      - add_to_db        → Create a new document chunk
      - get              → Retrieve by ID
      - remove           → Delete a document
      - find_chunks_with_images → Return chunks in a document that have images
      - get_images_for_chunk    → Get all images linked to a chunk
      - create_fts_table        → Initialize enhanced FTS
      - search_fts              → Full-text search across chunks (optional)
    """

    def __init__(self, db_config: DatabaseConfig = None):
        self.db_config = db_config or DatabaseConfig()

    # ------------------------
    # CORE CRUD
    # ------------------------

    @with_request_id
    def add_to_db(self,
                  name: str,
                  file_path: str,
                  content: Optional[str] = None,
                  complete_document_id: Optional[int] = None,
                  rev: str = "R0",
                  doc_metadata: Optional[Dict[str, Any]] = None) -> int:
        """Insert a new Document chunk."""
        with self.db_config.main_session() as session:
            try:
                doc = Document(
                    name=name,
                    file_path=file_path,
                    content=content,
                    complete_document_id=complete_document_id,
                    rev=rev,
                    doc_metadata=doc_metadata or {}
                )
                session.add(doc)
                session.commit()
                info_id(f"Created Document chunk '{name}' (id={doc.id})", None)
                return doc.id
            except SQLAlchemyError as e:
                session.rollback()
                error_id(f"DocumentService.add_to_db failed: {e}", None)
                raise

    @with_request_id
    def get(self, doc_id: int) -> Optional[Document]:
        """Retrieve a Document by ID."""
        with self.db_config.main_session() as session:
            try:
                return session.query(Document).filter_by(id=doc_id).first()
            except SQLAlchemyError as e:
                error_id(f"DocumentService.get failed: {e}", None)
                raise

    @with_request_id
    def remove(self, doc_id: int) -> bool:
        """Delete a Document by ID."""
        with self.db_config.main_session() as session:
            try:
                doc = session.query(Document).filter_by(id=doc_id).first()
                if doc:
                    session.delete(doc)
                    session.commit()
                    info_id(f"Deleted Document id={doc_id}", None)
                    return True
                return False
            except SQLAlchemyError as e:
                session.rollback()
                error_id(f"DocumentService.remove failed: {e}", None)
                raise

    # ------------------------
    # ASSOCIATIONS
    # ------------------------

    @with_request_id
    def get_images_for_chunk(self, chunk_id: int) -> List[Dict[str, Any]]:
        """Return all images linked to a document chunk."""
        try:
            return Document.get_images_for_chunk(chunk_id)
        except SQLAlchemyError as e:
            error_id(f"DocumentService.get_images_for_chunk failed: {e}", None)
            raise

    @with_request_id
    def find_chunks_with_images(self, complete_document_id: int) -> List[Document]:
        """Return all chunks in a document that have images associated."""
        try:
            return Document.find_chunks_with_images(complete_document_id)
        except SQLAlchemyError as e:
            error_id(f"DocumentService.find_chunks_with_images failed: {e}", None)
            raise

    # ------------------------
    # FULL-TEXT SEARCH
    # ------------------------

    @with_request_id
    def create_fts_table(self) -> bool:
        """Initialize enhanced FTS table for documents with image-chunk support."""
        try:
            return Document.create_fts_table()
        except SQLAlchemyError as e:
            error_id(f"DocumentService.create_fts_table failed: {e}", None)
            return False

    @with_request_id
    def search_fts(self, search_text: str, limit: int = 20) -> List[Dict[str, Any]]:
        """
        Example full-text search across document chunks.
        This assumes `documents_fts` is synced with Document chunks.
        """
        from sqlalchemy import text
        with self.db_config.main_session() as session:
            try:
                query = text("""
                    SELECT id, title, content, complete_document_id, chunk_id, has_images
                    FROM documents_fts
                    WHERE search_vector @@ plainto_tsquery(:query)
                    ORDER BY ts_rank(search_vector, plainto_tsquery(:query)) DESC
                    LIMIT :limit
                """)
                rows = session.execute(query, {"query": search_text, "limit": limit}).fetchall()
                return [dict(row._mapping) for row in rows]
            except SQLAlchemyError as e:
                error_id(f"DocumentService.search_fts failed: {e}", None)
                return []
