# modules/services/document_service.py

from typing import Optional, List, Dict, Any
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy import text

from modules.configuration.config_env import DatabaseConfig
from modules.configuration.log_config import (
    info_id, error_id, warning_id, debug_id, with_request_id
)
from modules.emtacdb.emtacdb_fts import (
    Document,
    Image,
    ImageCompletedDocumentAssociation,
    CompleteDocument
)


class DocumentService:
    """
    Service layer for managing Document (text chunk) entities.

    Provides:
      - save()                   → Create / update chunk
      - get()                    → Retrieve by ID
      - remove()                 → Delete a chunk
      - find()                   → Search by name/path/metadata
      - find_related()           → Upward/downward relationships
      - get_images_for_chunk()   → All images linked to this chunk
      - find_chunks_with_images()
      - create_fts_table()       → Initialize enhanced FTS
      - search_fts()             → FTS query for RAG
    """

    def __init__(self, db_config: DatabaseConfig = None):
        self.db_config = db_config or DatabaseConfig()

    # ----------------------------------------------------------------------
    # CREATE / UPDATE
    # ----------------------------------------------------------------------
    @with_request_id
    def save(
        self,
        name: str,
        file_path: str,
        content: Optional[str] = None,
        complete_document_id: Optional[int] = None,
        rev: str = "R0",
        doc_metadata: Optional[Dict[str, Any]] = None,
        doc_id: Optional[int] = None,
        request_id: Optional[str] = None,
    ) -> Document:
        """
        Create or update a document chunk.
        """
        with self.db_config.main_session() as session:
            try:
                # ----------------------------------------------------------
                # UPDATE
                # ----------------------------------------------------------
                if doc_id:
                    chunk = session.query(Document).filter_by(id=doc_id).first()
                    if not chunk:
                        raise ValueError(f"Document id={doc_id} not found")

                    chunk.name = name
                    chunk.file_path = file_path
                    chunk.content = content
                    chunk.rev = rev
                    chunk.doc_metadata = doc_metadata or chunk.doc_metadata

                    info_id(f"Updated Document chunk id={doc_id}", request_id)
                    return chunk

                # ----------------------------------------------------------
                # CREATE
                # ----------------------------------------------------------
                chunk = Document(
                    name=name,
                    file_path=file_path,
                    content=content,
                    complete_document_id=complete_document_id,
                    rev=rev,
                    doc_metadata=doc_metadata or {}
                )
                session.add(chunk)
                session.flush()

                info_id(f"Created Document chunk '{name}' (id={chunk.id})", request_id)
                return chunk

            except SQLAlchemyError as e:
                error_id(f"DocumentService.save failed: {e}", request_id)
                raise

    # ----------------------------------------------------------------------
    # GET
    # ----------------------------------------------------------------------
    @with_request_id
    def get(self, doc_id: int, request_id: Optional[str] = None) -> Optional[Document]:
        with self.db_config.main_session() as session:
            try:
                return session.query(Document).filter_by(id=doc_id).first()
            except SQLAlchemyError as e:
                error_id(f"DocumentService.get failed: {e}", request_id)
                raise

    # ----------------------------------------------------------------------
    # DELETE
    # ----------------------------------------------------------------------
    @with_request_id
    def remove(self, doc_id: int, request_id: Optional[str] = None) -> bool:
        with self.db_config.main_session() as session:
            try:
                doc = session.query(Document).filter_by(id=doc_id).first()
                if not doc:
                    warning_id(f"Document id={doc_id} not found for deletion", request_id)
                    return False

                session.delete(doc)
                info_id(f"Deleted Document id={doc_id}", request_id)
                return True
            except SQLAlchemyError as e:
                error_id(f"DocumentService.remove failed: {e}", request_id)
                raise

    # ----------------------------------------------------------------------
    # FIND (Generic Search)
    # ----------------------------------------------------------------------
    @with_request_id
    def find(
        self,
        name: Optional[str] = None,
        file_path: Optional[str] = None,
        complete_document_id: Optional[int] = None,
        limit: int = 50,
        request_id: Optional[str] = None,
    ) -> List[Document]:
        """Search Document chunks by simple filters."""
        with self.db_config.main_session() as session:
            try:
                query = session.query(Document)

                if name:
                    query = query.filter(Document.name.ilike(f"%{name}%"))

                if file_path:
                    query = query.filter(Document.file_path.ilike(f"%{file_path}%"))

                if complete_document_id:
                    query = query.filter(Document.complete_document_id == complete_document_id)

                results = query.limit(limit).all()
                info_id(f"Found {len(results)} Document chunks", request_id)
                return results

            except SQLAlchemyError as e:
                error_id(f"DocumentService.find failed: {e}", request_id)
                raise

    # ----------------------------------------------------------------------
    # RELATIONSHIP TRAVERSAL
    # ----------------------------------------------------------------------
    @with_request_id
    def find_related(self, doc_id: int, request_id: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """Return related CompleteDocument and images."""
        with self.db_config.main_session() as session:
            try:
                chunk = session.query(Document).filter_by(id=doc_id).first()
                if not chunk:
                    warning_id(f"No Document chunk found for id={doc_id}", request_id)
                    return None

                complete_doc = chunk.complete_document

                images = (
                    session.query(Image)
                    .join(ImageCompletedDocumentAssociation,
                          ImageCompletedDocumentAssociation.image_id == Image.id)
                    .filter(ImageCompletedDocumentAssociation.document_id == doc_id)
                    .all()
                )

                return {
                    "chunk": chunk,
                    "upward": {"complete_document": complete_doc},
                    "downward": {"images": images},
                }

            except SQLAlchemyError as e:
                error_id(f"DocumentService.find_related failed: {e}", request_id)
                raise

    # ----------------------------------------------------------------------
    # IMAGE-CHUNK ASSOCIATIONS
    # ----------------------------------------------------------------------
    @with_request_id
    def get_images_for_chunk(self, chunk_id: int, request_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """Wrapper around model-level helper."""
        try:
            return Document.get_images_for_chunk(chunk_id)
        except SQLAlchemyError as e:
            error_id(f"DocumentService.get_images_for_chunk failed: {e}", request_id)
            raise

    @with_request_id
    def find_chunks_with_images(self, complete_document_id: int, request_id: Optional[str] = None) -> List[Document]:
        """Wrapper around model-level helper."""
        try:
            return Document.find_chunks_with_images(complete_document_id)
        except SQLAlchemyError as e:
            error_id(f"DocumentService.find_chunks_with_images failed: {e}", request_id)
            raise

    # ----------------------------------------------------------------------
    # FULL-TEXT SEARCH
    # ----------------------------------------------------------------------
    @with_request_id
    def create_fts_table(self, request_id: Optional[str] = None) -> bool:
        try:
            return Document.create_fts_table()
        except SQLAlchemyError as e:
            error_id(f"DocumentService.create_fts_table failed: {e}", request_id)
            return False

    @with_request_id
    def search_fts(
            self,
            search_text: str,
            limit: int = 20,
            request_id: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Run FTS by delegating to Document.search_fts().
        The model owns the SQL logic; the service acts as a thin wrapper.
        """
        try:
            results = Document.search_fts(
                query_text=search_text,
                limit=limit,
                request_id=request_id
            )
            return results

        except Exception as e:
            error_id(f"DocumentService.search_fts failed: {e}", request_id)
            return []


