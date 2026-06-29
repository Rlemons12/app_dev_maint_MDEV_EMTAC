from typing import Optional, List, Dict, Any

from sqlalchemy.orm import Session
from sqlalchemy import text, and_

from modules.configuration.log_config import (
    info_id,
    warning_id,
    with_request_id,
)

from modules.emtacdb.emtacdb_fts import (
    CompleteDocument,
    Position,
    Image,
    CompletedDocumentPositionAssociation,
    ImageCompletedDocumentAssociation,
)


class CompleteDocumentService:
    """
    Pure domain service for CompleteDocument.

    RULES:
    - Does NOT open sessions
    - Does NOT commit
    - Does NOT rollback
    - Does NOT coordinate workflows
    """

    # ==============================================================
    # BASIC QUERY
    # ==============================================================

    @with_request_id
    def find(
        self,
        session: Session,
        *,
        title: Optional[str] = None,
        file_path: Optional[str] = None,
        limit: int = 50,
        request_id: Optional[str] = None,
    ) -> List[CompleteDocument]:

        query = session.query(CompleteDocument)

        if title:
            query = query.filter(CompleteDocument.title.ilike(f"%{title}%"))

        if file_path:
            query = query.filter(CompleteDocument.file_path.ilike(f"%{file_path}%"))

        results = query.limit(limit).all()
        info_id(f"Found {len(results)} CompleteDocuments", request_id)
        return results

    # ==============================================================

    @with_request_id
    def get(
        self,
        session: Session,
        *,
        document_id: Optional[int] = None,
        complete_document_id: Optional[int] = None,
        id: Optional[int] = None,
        request_id: Optional[str] = None,
    ) -> Optional[CompleteDocument]:

        resolved_id = document_id or complete_document_id or id

        if resolved_id is None:
            raise ValueError(
                "CompleteDocumentService.get() requires one of: "
                "document_id, complete_document_id, or id"
            )

        doc = session.get(CompleteDocument, resolved_id)

        if not doc:
            warning_id(f"CompleteDocument id={resolved_id} not found", request_id)

        return doc

    @with_request_id
    def get_by_id(
        self,
        session: Session,
        *,
        complete_document_id: int,
        request_id: Optional[str] = None,
    ) -> Optional[CompleteDocument]:
        return self.get(
            session=session,
            complete_document_id=complete_document_id,
            request_id=request_id,
        )

    # ==============================================================
    # CREATE
    # ==============================================================

    @with_request_id
    def create(
        self,
        session: Session,
        *,
        title: str,
        file_path: Optional[str] = None,
        content: Optional[str] = None,
        position_id: Optional[int] = None,

        # RAG / AI metadata
        summary: Optional[str] = None,
        rag_metadata: Optional[Dict[str, Any]] = None,
        topics: Optional[List[str]] = None,
        keywords: Optional[List[str]] = None,
        questions_answered: Optional[List[str]] = None,
        equipment: Optional[List[str]] = None,

        # File/extraction metadata
        file_sha256: Optional[str] = None,
        file_size: Optional[int] = None,
        file_basename: Optional[str] = None,
        source_type: Optional[str] = None,
        extraction_method: Optional[str] = None,

        request_id: Optional[str] = None,
    ) -> CompleteDocument:

        doc = CompleteDocument(
            title=title,
            file_path=file_path,
            content=content,
            rev="R0",
        )

        self._apply_optional_metadata(
            doc=doc,
            summary=summary,
            rag_metadata=rag_metadata,
            topics=topics,
            keywords=keywords,
            questions_answered=questions_answered,
            equipment=equipment,
            file_sha256=file_sha256,
            file_size=file_size,
            file_basename=file_basename,
            source_type=source_type,
            extraction_method=extraction_method,
            request_id=request_id,
        )

        session.add(doc)
        session.flush()

        if position_id:
            self.attach_to_position(
                session=session,
                document_id=doc.id,
                position_id=position_id,
                request_id=request_id,
            )

        info_id(f"Created CompleteDocument id={doc.id}", request_id)
        return doc

    # ==============================================================
    # UPDATE
    # ==============================================================

    @with_request_id
    def update(
        self,
        session: Session,
        *,
        document_id: int,
        title: Optional[str] = None,
        file_path: Optional[str] = None,
        content: Optional[str] = None,

        # RAG / AI metadata
        summary: Optional[str] = None,
        rag_metadata: Optional[Dict[str, Any]] = None,
        topics: Optional[List[str]] = None,
        keywords: Optional[List[str]] = None,
        questions_answered: Optional[List[str]] = None,
        equipment: Optional[List[str]] = None,

        # File/extraction metadata
        file_sha256: Optional[str] = None,
        file_size: Optional[int] = None,
        file_basename: Optional[str] = None,
        source_type: Optional[str] = None,
        extraction_method: Optional[str] = None,

        request_id: Optional[str] = None,
    ) -> Optional[CompleteDocument]:

        doc = self.get(session, document_id=document_id, request_id=request_id)

        if not doc:
            return None

        updated = False

        if title is not None and doc.title != title:
            doc.title = title
            updated = True

        if file_path is not None and doc.file_path != file_path:
            doc.file_path = file_path
            updated = True

        if content is not None and doc.content != content:
            doc.content = content
            updated = True

        metadata_updated = self._apply_optional_metadata(
            doc=doc,
            summary=summary,
            rag_metadata=rag_metadata,
            topics=topics,
            keywords=keywords,
            questions_answered=questions_answered,
            equipment=equipment,
            file_sha256=file_sha256,
            file_size=file_size,
            file_basename=file_basename,
            source_type=source_type,
            extraction_method=extraction_method,
            request_id=request_id,
        )

        if updated or metadata_updated:
            self._bump_revision(doc)

        info_id(f"Updated CompleteDocument id={document_id}", request_id)
        return doc

    # ==============================================================
    # UPSERT BY FILE_PATH
    # ==============================================================

    @with_request_id
    def upsert(
        self,
        session: Session,
        *,
        title: str,
        file_path: str,
        content: Optional[str] = None,

        # RAG / AI metadata
        summary: Optional[str] = None,
        rag_metadata: Optional[Dict[str, Any]] = None,
        topics: Optional[List[str]] = None,
        keywords: Optional[List[str]] = None,
        questions_answered: Optional[List[str]] = None,
        equipment: Optional[List[str]] = None,

        # File/extraction metadata
        file_sha256: Optional[str] = None,
        file_size: Optional[int] = None,
        file_basename: Optional[str] = None,
        source_type: Optional[str] = None,
        extraction_method: Optional[str] = None,

        request_id: Optional[str] = None,
    ) -> CompleteDocument:

        existing = (
            session.query(CompleteDocument)
            .filter(CompleteDocument.file_path == file_path)
            .first()
        )

        if existing:
            updated = False

            if title and existing.title != title:
                existing.title = title
                updated = True

            if content is not None and existing.content != content:
                existing.content = content
                updated = True

            metadata_updated = self._apply_optional_metadata(
                doc=existing,
                summary=summary,
                rag_metadata=rag_metadata,
                topics=topics,
                keywords=keywords,
                questions_answered=questions_answered,
                equipment=equipment,
                file_sha256=file_sha256,
                file_size=file_size,
                file_basename=file_basename,
                source_type=source_type,
                extraction_method=extraction_method,
                request_id=request_id,
            )

            if updated or metadata_updated:
                self._bump_revision(existing)

            info_id(
                f"Upsert updated CompleteDocument id={existing.id}",
                request_id,
            )

            return existing

        return self.create(
            session=session,
            title=title,
            file_path=file_path,
            content=content,
            summary=summary,
            rag_metadata=rag_metadata,
            topics=topics,
            keywords=keywords,
            questions_answered=questions_answered,
            equipment=equipment,
            file_sha256=file_sha256,
            file_size=file_size,
            file_basename=file_basename,
            source_type=source_type,
            extraction_method=extraction_method,
            request_id=request_id,
        )

    # ==============================================================
    # METADATA HELPERS
    # ==============================================================

    def _apply_optional_metadata(
        self,
        *,
        doc: CompleteDocument,
        summary: Optional[str] = None,
        rag_metadata: Optional[Dict[str, Any]] = None,
        topics: Optional[List[str]] = None,
        keywords: Optional[List[str]] = None,
        questions_answered: Optional[List[str]] = None,
        equipment: Optional[List[str]] = None,
        file_sha256: Optional[str] = None,
        file_size: Optional[int] = None,
        file_basename: Optional[str] = None,
        source_type: Optional[str] = None,
        extraction_method: Optional[str] = None,
        request_id: Optional[str] = None,
    ) -> bool:
        """
        Applies optional metadata only when values are provided.

        Returns:
            True if any field changed.
        """

        changed = False

        updates = {
            "summary": summary,
            "rag_metadata": rag_metadata,
            "topics": topics,
            "keywords": keywords,
            "questions_answered": questions_answered,
            "equipment": equipment,
            "file_sha256": file_sha256,
            "file_size": file_size,
            "file_basename": file_basename,
            "source_type": source_type,
            "extraction_method": extraction_method,
        }

        for field_name, new_value in updates.items():
            if new_value is None:
                continue

            if not hasattr(doc, field_name):
                warning_id(
                    f"CompleteDocument has no field '{field_name}'. "
                    f"Skipping metadata assignment.",
                    request_id,
                )
                continue

            old_value = getattr(doc, field_name)

            if old_value != new_value:
                setattr(doc, field_name, new_value)
                changed = True

        return changed

    # ==============================================================
    # REVISION
    # ==============================================================

    def _bump_revision(self, doc: CompleteDocument) -> None:
        try:
            rev_num = int(doc.rev[1:]) if doc.rev and doc.rev.startswith("R") else 0
        except Exception:
            rev_num = 0

        doc.rev = f"R{rev_num + 1}"

    # ==============================================================
    # DELETE
    # ==============================================================

    @with_request_id
    def delete(
        self,
        session: Session,
        *,
        document_id: int,
        request_id: Optional[str] = None,
    ) -> bool:

        doc = self.get(
            session=session,
            document_id=document_id,
            request_id=request_id,
        )

        if not doc:
            return False

        session.delete(doc)
        info_id(f"Deleted CompleteDocument id={document_id}", request_id)
        return True

    # ==============================================================
    # ASSOCIATIONS
    # ==============================================================

    @with_request_id
    def attach_to_position(
        self,
        session: Session,
        *,
        document_id: int,
        position_id: int,
        request_id: Optional[str] = None,
    ) -> None:

        existing = (
            session.query(CompletedDocumentPositionAssociation)
            .filter_by(
                complete_document_id=document_id,
                position_id=position_id,
            )
            .first()
        )

        if existing:
            return

        assoc = CompletedDocumentPositionAssociation(
            complete_document_id=document_id,
            position_id=position_id,
        )

        session.add(assoc)

        info_id(
            f"Attached document {document_id} to position {position_id}",
            request_id,
        )