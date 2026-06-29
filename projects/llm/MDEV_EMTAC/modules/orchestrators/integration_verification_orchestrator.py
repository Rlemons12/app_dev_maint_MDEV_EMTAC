from __future__ import annotations

from typing import Dict, Any, Optional, List

from sqlalchemy.orm import joinedload

from modules.orchestrators.base_orchestrator import BaseOrchestrator
from modules.configuration.log_config import (
    with_request_id,
    debug_id,
    info_id,
    warning_id,
    error_id,
)

from modules.emtacdb.emtacdb_fts import (
    CompleteDocument,
    Document,
    ImageCompletedDocumentAssociation,
    CompletedDocumentPositionAssociation,
)

from modules.emtac_ai.search.rag_core.document_ui_payload import (
    DocumentUIPayload,
)


class IntegrationVerificationOrchestrator(BaseOrchestrator):
    """
    System-level integration verification.

    Used for:
        - Pipeline validation
        - Payload integrity checks
        - Graph consistency verification
        - Admin debugging tools

    No AI execution.
    No RAG.
    No retriever.
    """

    # ---------------------------------------------------------
    # PUBLIC ENTRY
    # ---------------------------------------------------------
    @with_request_id
    def verify_complete_document(
        self,
        *,
        document_id: int,
        request_id: Optional[str] = None,
    ) -> Dict[str, Any]:

        debug_id(
            f"[IntegrationVerification] Verifying document_id={document_id}",
            request_id,
        )

        try:
            with self.session_scope() as session:

                # -----------------------------------------------------
                # 1. Load Full Graph
                # -----------------------------------------------------
                complete_document: Optional[CompleteDocument] = (
                    session.query(CompleteDocument)
                    .options(
                        joinedload(CompleteDocument.documents)
                        .joinedload(Document.embeddings),

                        joinedload(
                            CompleteDocument.image_completed_document_association
                        ).joinedload(ImageCompletedDocumentAssociation.image),

                        joinedload(
                            CompleteDocument.completed_document_position_association
                        ),
                    )
                    .filter(CompleteDocument.id == document_id)
                    .first()
                )

                if not complete_document:
                    warning_id(
                        f"[IntegrationVerification] Document {document_id} not found",
                        request_id,
                    )
                    return {
                        "error": "Document not found",
                        "payload": [],
                        "validation": {},
                    }

                # -----------------------------------------------------
                # 2. Build RAG-Compatible Chunk Structure
                # -----------------------------------------------------
                used_chunks: List[Dict[str, Any]] = []

                missing_embeddings = 0
                empty_chunks = 0

                for doc in complete_document.documents:

                    has_embedding = bool(doc.embeddings)

                    if not has_embedding:
                        missing_embeddings += 1

                    if not doc.content:
                        empty_chunks += 1

                    used_chunks.append(
                        {
                            "chunk_id": doc.id,
                            "text": doc.content or "",
                            "complete_document_id": complete_document.id,
                            "complete_document_title": complete_document.title,
                            "metadata": {
                                "document_id": doc.id,
                                "has_embedding": has_embedding,
                            },
                        }
                    )

                # -----------------------------------------------------
                # 3. Build UI Payload (same projection as RAG)
                # -----------------------------------------------------
                payload = (
                    DocumentUIPayload()
                    .aggregate_from_chunks(used_chunks)
                    .build()
                )

                # -----------------------------------------------------
                # 4. Structural Validation
                # -----------------------------------------------------
                image_count = len(
                    complete_document.image_completed_document_association or []
                )

                position_count = len(
                    complete_document.completed_document_position_association or []
                )

                validation = {
                    "document_id": document_id,
                    "chunk_count": len(used_chunks),
                    "image_count": image_count,
                    "position_count": position_count,
                    "missing_embeddings": missing_embeddings,
                    "empty_chunks": empty_chunks,
                    "payload_document_count": len(payload),
                }

                info_id(
                    f"[IntegrationVerification] document_id={document_id} "
                    f"| chunks={len(used_chunks)} "
                    f"| missing_embeddings={missing_embeddings}",
                    request_id,
                )

                return {
                    "payload": payload,
                    "used_chunks": used_chunks,
                    "validation": validation,
                }

        except Exception as e:
            error_id(
                f"[IntegrationVerification] Verification failed: {e}",
                request_id,
                exc_info=True,
            )
            raise