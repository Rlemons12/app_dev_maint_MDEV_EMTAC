from typing import List, Optional
from sqlalchemy.orm import Session
from sqlalchemy import select

from modules.configuration.log_config import (
    info_id,
    warning_id,
    debug_id,
    with_request_id,
)

from modules.emtacdb.emtacdb_fts import (
    CompletedDocumentPositionAssociation,
    CompleteDocument,
    Position,
)


class CompletedDocumentPositionService:
    """
    Domain service for CompletedDocumentPositionAssociation.

    HARD RULES:
    - NEVER open sessions
    - NEVER commit
    - NEVER rollback
    - NEVER close sessions
    - Orchestrator owns transaction scope
    """

    # =========================================================
    # ASSOCIATE
    # =========================================================

    @with_request_id
    def associate(
        self,
        session: Session,
        *,
        position_id: int,
        complete_document_id: int,
        request_id: Optional[str] = None,
    ) -> Optional[CompletedDocumentPositionAssociation]:

        if not session:
            raise RuntimeError("Session required for associate")

        if not position_id or not complete_document_id:
            warning_id("Invalid IDs provided to associate()", request_id)
            return None

        position = session.get(Position, position_id)
        if not position:
            warning_id(f"Position id={position_id} not found", request_id)
            return None

        document = session.get(CompleteDocument, complete_document_id)
        if not document:
            warning_id(
                f"CompleteDocument id={complete_document_id} not found",
                request_id,
            )
            return None

        existing = (
            session.query(CompletedDocumentPositionAssociation)
            .filter_by(
                position_id=position_id,
                complete_document_id=complete_document_id,
            )
            .first()
        )

        if existing:
            debug_id(
                f"Association already exists id={existing.id}",
                request_id,
            )
            return existing

        assoc = CompletedDocumentPositionAssociation(
            position_id=position_id,
            complete_document_id=complete_document_id,
        )

        session.add(assoc)
        session.flush()

        info_id(
            f"Association staged document={complete_document_id}, "
            f"position={position_id}",
            request_id,
        )

        return assoc

    # =========================================================
    # BULK ASSOCIATE (Optimized)
    # =========================================================

    @with_request_id
    def bulk_associate(
        self,
        session: Session,
        *,
        position_id: int,
        document_ids: List[int],
        request_id: Optional[str] = None,
    ) -> List[CompletedDocumentPositionAssociation]:

        if not session:
            raise RuntimeError("Session required for bulk_associate")

        if not document_ids:
            return []

        created = []

        for doc_id in document_ids:
            assoc = self.associate(
                session=session,
                position_id=position_id,
                complete_document_id=doc_id,
                request_id=request_id,
            )
            if assoc:
                created.append(assoc)

        return created

    # =========================================================
    # REMOVE
    # =========================================================

    @with_request_id
    def dissociate(
        self,
        session: Session,
        *,
        position_id: int,
        complete_document_id: int,
        request_id: Optional[str] = None,
    ) -> bool:

        assoc = (
            session.query(CompletedDocumentPositionAssociation)
            .filter_by(
                position_id=position_id,
                complete_document_id=complete_document_id,
            )
            .first()
        )

        if not assoc:
            warning_id(
                f"No association found document={complete_document_id}, "
                f"position={position_id}",
                request_id,
            )
            return False

        session.delete(assoc)

        info_id(
            f"Association staged for removal document={complete_document_id}, "
            f"position={position_id}",
            request_id,
        )

        return True

    # =========================================================
    # QUERIES
    # =========================================================

    @with_request_id
    def get_documents_for_position(
        self,
        session: Session,
        *,
        position_id: int,
        request_id: Optional[str] = None,
    ) -> List[CompleteDocument]:

        if not position_id:
            return []

        stmt = (
            select(CompleteDocument)
            .join(
                CompletedDocumentPositionAssociation,
                CompleteDocument.id ==
                CompletedDocumentPositionAssociation.complete_document_id,
            )
            .where(
                CompletedDocumentPositionAssociation.position_id ==
                position_id
            )
            .distinct()
        )

        results = session.execute(stmt).scalars().all()

        debug_id(
            f"Found {len(results)} documents for position_id={position_id}",
            request_id,
        )

        return results

    # ---------------------------------------------------------

    @with_request_id
    def get_positions_for_document(
        self,
        session: Session,
        *,
        complete_document_id: int,
        request_id: Optional[str] = None,
    ) -> List[Position]:

        if not complete_document_id:
            return []

        stmt = (
            select(Position)
            .join(
                CompletedDocumentPositionAssociation,
                CompletedDocumentPositionAssociation.position_id ==
                Position.id,
            )
            .where(
                CompletedDocumentPositionAssociation.complete_document_id ==
                complete_document_id
            )
            .distinct()
        )

        results = session.execute(stmt).scalars().all()

        debug_id(
            f"Found {len(results)} positions for "
            f"document_id={complete_document_id}",
            request_id,
        )

        return results

    # ---------------------------------------------------------

    @with_request_id
    def get_position_ids_for_document(
        self,
        session: Session,
        *,
        complete_document_id: int,
        request_id: Optional[str] = None,
    ) -> List[int]:

        if not complete_document_id:
            return []

        stmt = (
            select(CompletedDocumentPositionAssociation.position_id)
            .where(
                CompletedDocumentPositionAssociation.complete_document_id ==
                complete_document_id
            )
            .distinct()
        )

        results = session.execute(stmt).scalars().all()

        debug_id(
            f"Found {len(results)} position IDs for "
            f"document_id={complete_document_id}",
            request_id,
        )

        return results
