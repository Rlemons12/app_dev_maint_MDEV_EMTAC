# modules/services/complete_document_service.py

from typing import List, Optional, Dict, Any, Tuple
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session

from modules.configuration.config_env import DatabaseConfig
from modules.configuration.log_config import (
    info_id, debug_id, warning_id, error_id, with_request_id, get_request_id
)
from modules.emtacdb.emtacdb_fts import (
    CompleteDocument,
    Image,
    ImageCompletedDocumentAssociation,
    Position,
    CompletedDocumentPositionAssociation,
)


class CompleteDocumentService:
    """
    Service layer for managing CompleteDocument entities.

    Provides:
      - find()              → search by title or metadata
      - get()               → load a document by ID
      - save()              → create or update a record
      - remove()            → delete by ID
      - find_or_create()
      - find_related()      → return upward/downward relationships
      - search_text()       → call model FTS search
      - search_embedding()  → vector similarity search
      - process_upload()    → delegate upload pipeline
      - get_summary()       → document metadata summary
      - get_images(), get_chunks(), get_embeddings()
      - update()            → metadata-only update unless flags specify otherwise
    """

    def __init__(self, db_config: Optional[DatabaseConfig] = None):
        self.db_config = db_config or DatabaseConfig()

    # ------------------------------------------------------------------
    # BASIC QUERY OPERATIONS
    # ------------------------------------------------------------------
    @with_request_id
    def find(
        self,
        title: Optional[str] = None,
        file_path: Optional[str] = None,
        has_embeddings: Optional[bool] = None,
        limit: int = 50,
        request_id: Optional[str] = None
    ) -> List[CompleteDocument]:
        """Search CompleteDocuments by title/file_path/flags."""
        with self.db_config.main_session() as session:
            try:
                query = session.query(CompleteDocument)

                if title:
                    query = query.filter(CompleteDocument.title.ilike(f"%{title}%"))

                if file_path:
                    query = query.filter(CompleteDocument.file_path.ilike(f"%{file_path}%"))

                if has_embeddings is True:
                    query = query.filter(CompleteDocument.embedding != None)

                if has_embeddings is False:
                    query = query.filter(CompleteDocument.embedding == None)

                results = query.limit(limit).all()
                info_id(f"Found {len(results)} CompleteDocuments", request_id)
                return results
            except SQLAlchemyError as e:
                error_id(f"CompleteDocumentService.find failed: {e}", request_id)
                raise

    @with_request_id
    def get(
            self,
            complete_document_id: int,
            *,
            session: Optional[Session] = None,
            request_id: Optional[str] = None,
    ) -> Optional[CompleteDocument]:
        """
        Load a CompleteDocument by ID.

        Session-safe:
        - Uses caller session if provided
        - Otherwise creates and closes its own
        """

        close_session = False

        if session is None:
            session = self.db_config.get_main_session()
            close_session = True

        try:
            doc = (
                session.query(CompleteDocument)
                .filter(CompleteDocument.id == complete_document_id)
                .first()
            )

            if not doc:
                warning_id(
                    f"CompleteDocument id={complete_document_id} not found",
                    request_id,
                )
                return None

            return doc

        except SQLAlchemyError as e:
            error_id(f"CompleteDocumentService.get failed: {e}", request_id)
            raise

        finally:
            if close_session:
                session.close()

    # ------------------------------------------------------------------
    # CREATE / SAVE
    # ------------------------------------------------------------------
    @with_request_id
    def save(
        self,
        title: str,
        file_path: Optional[str] = None,
        content: Optional[str] = None,
        position_id: Optional[int] = None,
        doc_id: Optional[int] = None,
        request_id: Optional[str] = None
    ) -> CompleteDocument:
        """
        Create or update a CompleteDocument.

        Note:
          - This performs a LIGHT UPDATE (Option A).
          - It does NOT regenerate chunks/embeddings unless done manually.
        """
        with self.db_config.main_session() as session:
            try:
                if doc_id:
                    # ----------------------------------------
                    # UPDATE
                    # ----------------------------------------
                    doc = session.query(CompleteDocument).filter_by(id=doc_id).first()
                    if not doc:
                        raise ValueError(f"CompleteDocument id={doc_id} not found")

                    doc.title = title
                    if file_path is not None:
                        doc.file_path = file_path
                    if content is not None:
                        doc.content = content

                    info_id(f"Updated CompleteDocument id={doc_id}", request_id)
                else:
                    # ----------------------------------------
                    # CREATE
                    # ----------------------------------------
                    doc = CompleteDocument(
                        title=title,
                        file_path=file_path,
                        content=content,
                    )
                    session.add(doc)
                    session.flush()

                    # Link to Position if provided
                    if position_id:
                        CompleteDocument._create_associations(session, doc.id, position_id)

                    info_id(f"Created CompleteDocument '{title}' id={doc.id}", request_id)

                return doc

            except SQLAlchemyError as e:
                error_id(f"CompleteDocumentService.save failed: {e}", request_id)
                raise

    # ------------------------------------------------------------------
    # DELETE
    # ------------------------------------------------------------------
    @with_request_id
    def remove(self, doc_id: int, request_id: Optional[str] = None) -> bool:
        """Delete a CompleteDocument."""
        with self.db_config.main_session() as session:
            try:
                doc = session.query(CompleteDocument).filter_by(id=doc_id).first()
                if not doc:
                    warning_id(f"No CompleteDocument found for deletion id={doc_id}", request_id)
                    return False

                session.delete(doc)
                info_id(f"Deleted CompleteDocument id={doc_id}", request_id)
                return True
            except SQLAlchemyError as e:
                error_id(f"CompleteDocumentService.remove failed: {e}", request_id)
                raise

    # ------------------------------------------------------------------
    # FIND OR CREATE
    # ------------------------------------------------------------------
    @with_request_id
    def find_or_create(
        self,
        title: str,
        file_path: Optional[str] = None,
        request_id: Optional[str] = None
    ) -> CompleteDocument:

        with self.db_config.main_session() as session:
            try:
                doc = session.query(CompleteDocument).filter_by(title=title).first()
                if doc:
                    return doc

                doc = CompleteDocument(title=title, file_path=file_path)
                session.add(doc)
                info_id(f"Created new CompleteDocument '{title}'", request_id)
                return doc
            except SQLAlchemyError as e:
                error_id(f"CompleteDocumentService.find_or_create failed: {e}", request_id)
                raise

    # ------------------------------------------------------------------
    # RELATED ENTITIES
    # ------------------------------------------------------------------
    @with_request_id
    def find_related(self, doc_id: int, request_id: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """Return upward/downward relationships."""
        with self.db_config.main_session() as session:
            doc = session.query(CompleteDocument).filter_by(id=doc_id).first()
            if not doc:
                return None

            # Positions linked
            positions = (
                session.query(Position)
                .join(CompletedDocumentPositionAssociation,
                      CompletedDocumentPositionAssociation.position_id == Position.id)
                .filter(CompletedDocumentPositionAssociation.complete_document_id == doc_id)
                .all()
            )

            # Images linked
            images = (
                session.query(Image)
                .join(ImageCompletedDocumentAssociation,
                      ImageCompletedDocumentAssociation.image_id == Image.id)
                .filter(ImageCompletedDocumentAssociation.complete_document_id == doc_id)
                .all()
            )

            return {
                "document": doc,
                "downward": {
                    "images": images,
                    "positions": positions,
                },
            }

    # ------------------------------------------------------------------
    # DOCUMENT SUMMARY
    # ------------------------------------------------------------------
    @with_request_id
    def get_summary(self, doc_id: int, request_id: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """Return metadata summary for RAG UI."""
        with self.db_config.main_session() as session:
            doc = session.query(CompleteDocument).filter_by(id=doc_id).first()
            if not doc:
                warning_id(f"No CompleteDocument found for summary id={doc_id}", request_id)
                return None

            images = doc.get_images_with_chunk_context(request_id=request_id)

            return {
                "id": doc.id,
                "title": doc.title,
                "file_path": doc.file_path,
                "has_embedding": doc.embedding is not None,
                "image_count": len(images) if images else 0,
                "position_count": len(doc.position) if hasattr(doc, "position") else 0,
            }

    # ------------------------------------------------------------------
    # SEARCH OPS (TEXT / VECTOR)
    # ------------------------------------------------------------------
    @with_request_id
    def search_text(self, query: str, limit: int = 25, request_id: Optional[str] = None):
        return CompleteDocument.search_by_text(query, limit=limit, request_id=request_id)

    @with_request_id
    def search_embedding(
        self,
        query_text: str,
        limit: int = 10,
        threshold: float = 0.7,
        request_id: Optional[str] = None
    ):
        return CompleteDocument.search_similar_by_embedding(
            query_text=query_text,
            limit=limit,
            threshold=threshold,
            request_id=request_id
        )

    # ------------------------------------------------------------------
    # UPLOAD PROCESSOR
    # ------------------------------------------------------------------
    @with_request_id
    def process_upload(self, files, metadata: dict, request_id: Optional[str] = None):
        """Wrap upload; return (success, payload, http_code)."""
        return CompleteDocument.process_upload(files, metadata, request_id=request_id)

