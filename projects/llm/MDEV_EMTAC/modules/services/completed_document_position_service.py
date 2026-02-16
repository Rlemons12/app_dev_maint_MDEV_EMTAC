# modules/services/completed_document_position_service.py

from typing import Optional, List, Tuple
from sqlalchemy.orm import Session
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session as SASession
from sqlalchemy import select

from modules.configuration.config_env import DatabaseConfig
from modules.configuration.log_config import (
    info_id,
    error_id,
    warning_id,
    debug_id,
    with_request_id,
    log_timed_operation,
    get_request_id,
)

from modules.emtacdb.emtacdb_fts import (
    CompletedDocumentPositionAssociation,
    CompleteDocument,
    Position,
)


class CompletedDocumentPositionService:
    """
    Service layer for CompletedDocumentPositionAssociation.

    Responsibilities:
    - Associate positions and complete documents
    - List documents for a position
    - List positions for a document
    - Manage association lifecycle
    """

    def __init__(self, db_config: Optional[DatabaseConfig] = None):
        self.db_config = db_config or DatabaseConfig()

    def _get_session(self, session: Optional[SASession]) -> Tuple[SASession, bool]:
        if session is not None:
            return session, False
        return self.db_config.get_main_session(), True

    # ---------------------------------------------------------
    # Associate
    # ---------------------------------------------------------
    @with_request_id
    def associate(
        self,
        position_id: int,
        complete_document_id: int,
        session: Optional[SASession] = None,
    ) -> Optional[CompletedDocumentPositionAssociation]:
        """
        Create or reuse a CompletedDocumentPositionAssociation.

        Uses the model's associate method under the hood.
        """
        rid = get_request_id()
        sess, created_here = self._get_session(session)
        debug_id(
            f"[CompletedDocumentPositionService.associate] position_id={position_id}, "
            f"complete_document_id={complete_document_id}",
            rid,
        )

        try:
            with log_timed_operation("CompletedDocumentPositionService.associate", rid):
                position = sess.query(Position).filter(Position.id == position_id).first()
                doc = (
                    sess.query(CompleteDocument)
                    .filter(CompleteDocument.id == complete_document_id)
                    .first()
                )

                if not position:
                    warning_id(f"Position id={position_id} not found", rid)
                    return None
                if not doc:
                    warning_id(f"CompleteDocument id={complete_document_id} not found", rid)
                    return None

                assoc = CompletedDocumentPositionAssociation.associate(
                    session=sess,
                    position=position,
                    complete_document=doc,
                )
                debug_id(f"Association id={assoc.id} created/reused", rid)
                return assoc

        except SQLAlchemyError as e:
            error_id(f"Error in CompletedDocumentPositionService.associate: {e}", rid, exc_info=True)
            if created_here:
                sess.rollback()
            return None
        finally:
            if created_here:
                sess.close()

    # ---------------------------------------------------------
    # Queries
    # ---------------------------------------------------------
    @with_request_id
    def get_documents_for_position(
        self,
        position_id: int,
        session: Optional[SASession] = None,
    ) -> List[CompleteDocument]:
        """
        Get all CompleteDocuments linked to a given Position.
        """
        rid = get_request_id()
        sess, created_here = self._get_session(session)
        debug_id(
            f"[CompletedDocumentPositionService.get_documents_for_position] position_id={position_id}",
            rid,
        )

        try:
            with log_timed_operation(
                "CompletedDocumentPositionService.get_documents_for_position", rid
            ):
                q = (
                    sess.query(CompleteDocument)
                    .join(
                        CompletedDocumentPositionAssociation,
                        CompleteDocument.id
                        == CompletedDocumentPositionAssociation.complete_document_id,
                    )
                    .filter(CompletedDocumentPositionAssociation.position_id == position_id)
                    .distinct()
                )
                docs = q.all()
                debug_id(f"Found {len(docs)} documents for position_id={position_id}", rid)
                return docs

        except SQLAlchemyError as e:
            error_id(f"Error in get_documents_for_position: {e}", rid, exc_info=True)
            return []
        finally:
            if created_here:
                sess.close()

    @with_request_id
    def get_positions_for_document(
            self,
            complete_document_id: int,
            session: Session,
            request_id: str | None = None,
    ) -> List[Position]:
        """
        Return Position ORM objects linked to a CompleteDocument.

        HARD RULE:
        - Uses caller-provided session ONLY
        - Never opens its own session
        - No ORM traversal
        """

        if not complete_document_id:
            return []

        stmt = (
            select(Position)
            .join(
                CompletedDocumentPositionAssociation,
                CompletedDocumentPositionAssociation.position_id == Position.id,
            )
            .where(
                CompletedDocumentPositionAssociation.complete_document_id
                == complete_document_id
            )
        )

        return session.execute(stmt).scalars().all()

    # ---------------------------------------------------------
    # LIGHTWEIGHT HELPERS (USED BY CHUNK RELATIONSHIP MAP)
    # ---------------------------------------------------------
    @with_request_id
    def get_position_ids_for_document(
        self,
        complete_document_id: int,
        session: Optional[SASession] = None,
    ) -> List[int]:
        """
        Return position IDs associated with a complete document.

        Lightweight helper for chunk relationship graphs.
        """
        rid = get_request_id()

        if not complete_document_id:
            debug_id(
                "[CompletedDocumentPositionService.get_position_ids_for_document] "
                "No complete_document_id provided",
                rid,
            )
            return []

        sess, created_here = self._get_session(session)

        try:
            with log_timed_operation(
                "CompletedDocumentPositionService.get_position_ids_for_document", rid
            ):
                rows = (
                    sess.query(CompletedDocumentPositionAssociation.position_id)
                    .filter(
                        CompletedDocumentPositionAssociation.complete_document_id
                        == complete_document_id
                    )
                    .distinct()
                    .all()
                )

                position_ids = [r[0] for r in rows]

                debug_id(
                    f"Found {len(position_ids)} positions for "
                    f"complete_document_id={complete_document_id}",
                    rid,
                )

                return position_ids

        except Exception as e:
            error_id(
                f"Error getting position IDs for document {complete_document_id}: {e}",
                rid,
                exc_info=True,
            )
            return []
        finally:
            if created_here:
                sess.close()
