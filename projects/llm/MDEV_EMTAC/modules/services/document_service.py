# modules/services/document_service.py

from typing import Optional, List, Dict, Any
from sqlalchemy.orm import Session
from sqlalchemy import text

from modules.configuration.log_config import (
    info_id,
    warning_id,
    with_request_id,
)

from modules.emtacdb.emtacdb_fts import (
    Document,
    Image,
    ImageCompletedDocumentAssociation,
)


class DocumentService:
    """
    Pure domain service for Document (text chunk).

    HARD RULES:
    - NEVER open sessions
    - NEVER close sessions
    - NEVER commit
    - NEVER rollback
    - Orchestrator owns transactions
    """

    # ----------------------------------------------------------------------
    # CREATE / UPDATE
    # ----------------------------------------------------------------------

    @with_request_id
    def save(
        self,
        session: Session,
        *,
        name: str,
        file_path: str,
        content: Optional[str] = None,
        complete_document_id: Optional[int] = None,
        rev: str = "R0",
        doc_metadata: Optional[Dict[str, Any]] = None,
        doc_id: Optional[int] = None,
        request_id: Optional[str] = None,
    ) -> Document:

        if doc_id:
            chunk = session.get(Document, doc_id)
            if not chunk:
                raise ValueError(f"Document id={doc_id} not found")

            chunk.name = name
            chunk.file_path = file_path
            chunk.content = content
            chunk.rev = rev

            if doc_metadata is not None:
                chunk.doc_metadata = doc_metadata

            return chunk

        chunk = Document(
            name=name,
            file_path=file_path,
            content=content,
            complete_document_id=complete_document_id,
            rev=rev,
            doc_metadata=doc_metadata or {},
        )

        session.add(chunk)
        session.flush()

        return chunk

    # ----------------------------------------------------------------------
    # CREATE CHUNKS (Text Splitting + Insert)
    # ----------------------------------------------------------------------

    @with_request_id
    def create_chunks(
        self,
        session: Session,
        *,
        complete_document_id: int,
        text: str,
        file_path: Optional[str] = None,
        chunk_size: int = 1000,
        overlap: int = 100,
        request_id: Optional[str] = None,
    ) -> List[int]:
        """
        Split text into overlapping chunks and persist as Document rows.
        """

        if not text:
            return []

        created_ids: List[int] = []

        start = 0
        text_length = len(text)
        index = 0

        while start < text_length:
            end = start + chunk_size
            chunk_text = text[start:end]

            chunk = Document(
                name=f"chunk_{complete_document_id}_{index}",
                file_path=file_path,
                content=chunk_text,
                complete_document_id=complete_document_id,
                rev="R0",
                doc_metadata={
                    "chunk_index": index,
                    "char_start": start,
                    "char_end": min(end, text_length),
                },
            )

            session.add(chunk)
            session.flush()

            created_ids.append(chunk.id)

            start += chunk_size - overlap
            index += 1

        return created_ids

    # ----------------------------------------------------------------------
    # GET
    # ----------------------------------------------------------------------

    @with_request_id
    def get(
            self,
            session: Session,
            *,
            doc_id: Optional[int] = None,
            document_id: Optional[int] = None,
            id: Optional[int] = None,
            request_id: Optional[str] = None,
    ) -> Optional[Document]:

        # Normalize primary key
        resolved_id = doc_id or document_id or id

        if resolved_id is None:
            raise ValueError(
                "DocumentService.get() requires one of: doc_id, document_id, or id"
            )

        return session.get(Document, resolved_id)

    # ----------------------------------------------------------------------
    # GET MULTIPLE
    # ----------------------------------------------------------------------

    @with_request_id
    def get_by_ids(
        self,
        session: Session,
        *,
        ids: List[int],
        request_id: Optional[str] = None,
    ) -> List[Document]:

        if not ids:
            return []

        return (
            session.query(Document)
            .filter(Document.id.in_(ids))
            .all()
        )

    # ----------------------------------------------------------------------
    # DELETE
    # ----------------------------------------------------------------------

    @with_request_id
    def remove(
        self,
        session: Session,
        *,
        doc_id: int,
        request_id: Optional[str] = None,
    ) -> bool:

        doc = session.get(Document, doc_id)
        if not doc:
            warning_id(f"Document id={doc_id} not found", request_id)
            return False

        session.delete(doc)
        return True

    # ----------------------------------------------------------------------
    # FIND (Dynamic Filtering)
    # ----------------------------------------------------------------------

    @with_request_id
    def find(
        self,
        session: Session,
        *,
        name: Optional[str] = None,
        file_path: Optional[str] = None,
        complete_document_id: Optional[int] = None,
        has_images: Optional[bool] = None,
        limit: int = 50,
        request_id: Optional[str] = None,
    ) -> List[Document]:

        query = session.query(Document)

        if name:
            query = query.filter(Document.name.ilike(f"%{name}%"))

        if file_path:
            query = query.filter(Document.file_path.ilike(f"%{file_path}%"))

        if complete_document_id:
            query = query.filter(
                Document.complete_document_id == complete_document_id
            )

        if has_images is True:
            query = query.join(
                ImageCompletedDocumentAssociation,
                ImageCompletedDocumentAssociation.document_id == Document.id,
            ).distinct()

        return query.limit(limit).all()

    # ----------------------------------------------------------------------
    # IMAGE RELATIONSHIPS
    # ----------------------------------------------------------------------

    @with_request_id
    def get_images_for_chunk(
        self,
        session: Session,
        *,
        chunk_id: int,
        request_id: Optional[str] = None,
    ) -> List[Image]:

        return (
            session.query(Image)
            .join(
                ImageCompletedDocumentAssociation,
                ImageCompletedDocumentAssociation.image_id == Image.id,
            )
            .filter(
                ImageCompletedDocumentAssociation.document_id == chunk_id
            )
            .all()
        )

    # ----------------------------------------------------------------------
    # FULL-TEXT SEARCH (FTS DIRECT ON DOCUMENT)
    # ----------------------------------------------------------------------

    @with_request_id
    def search_fts(
        self,
        session: Session,
        *,
        search_text: str,
        limit: int = 20,
        request_id: Optional[str] = None,
    ) -> List[Dict[str, Any]]:

        sql = text("""
            SELECT 
                id,
                name,
                file_path,
                rev,
                ts_rank(
                    to_tsvector('english', COALESCE(content, '')),
                    plainto_tsquery('english', :query)
                ) AS rank
            FROM document
            WHERE to_tsvector('english', COALESCE(content, ''))
                  @@ plainto_tsquery('english', :query)
            ORDER BY rank DESC
            LIMIT :limit
        """)

        rows = session.execute(
            sql,
            {"query": search_text, "limit": limit},
        ).fetchall()

        return [
            {
                "id": r[0],
                "name": r[1],
                "file_path": r[2],
                "rev": r[3],
                "rank": float(r[4]),
            }
            for r in rows
        ]