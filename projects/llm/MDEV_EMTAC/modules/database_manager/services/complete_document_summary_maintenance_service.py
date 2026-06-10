from __future__ import annotations

from typing import Dict, Any, List, Optional

from sqlalchemy import text
from sqlalchemy.orm import Session

from modules.configuration.log_config import (
    info_id,
    error_id,
    with_request_id,
)

from modules.services.complete_document_summary_service import (
    CompleteDocumentSummaryService,
)


class CompleteDocumentSummaryMaintenanceService:
    """
    Maintenance service for complete_document RAG metadata cleanup.

    Rules:
      - Does NOT open sessions
      - Does NOT commit
      - Does NOT rollback

    Important:
      This service supports cursor-based batching using after_id.

      That prevents force=True from repeatedly selecting the same first batch:

          [1, 2, 3, 4, 5]
          [1, 2, 3, 4, 5]
          [1, 2, 3, 4, 5]

      Instead it advances through IDs:

          [1, 2, 3, 4, 5]
          [6, 7, 8, 9, 10]
          [11, 12, 13, 14, 15]
    """

    def __init__(
        self,
        *,
        store_summary_embedding: bool = True,
    ):
        self.summary_service = CompleteDocumentSummaryService(
            store_summary_embedding=store_summary_embedding,
        )

    def count_targets(
        self,
        *,
        session: Session,
        include_missing_embedding: bool,
        force: bool = False,
    ) -> int:
        sql = self._build_count_sql(
            include_missing_embedding=include_missing_embedding,
            force=force,
        )

        return int(session.execute(sql).scalar() or 0)

    def get_target_ids(
        self,
        *,
        session: Session,
        limit: int,
        include_missing_embedding: bool,
        force: bool = False,
        after_id: int = 0,
    ) -> List[int]:
        """
        Return the next target complete_document IDs.

        Args:
            session:
                Existing SQLAlchemy session. This method does not commit or rollback.

            limit:
                Maximum number of IDs to return.

            include_missing_embedding:
                When True, include documents that already have summary/rag_metadata
                but are missing summary_embedding_vector.

            force:
                When True, include all documents with content, even if already summarized.

            after_id:
                Cursor value. Only rows with id > after_id are returned.

        Why after_id matters:
            With force=True, already-summarized rows are still valid targets.
            Without after_id, every loop returns the same first ordered batch forever.
        """

        if limit <= 0:
            return []

        if after_id < 0:
            after_id = 0

        sql = self._build_select_sql(
            include_missing_embedding=include_missing_embedding,
            force=force,
        )

        rows = session.execute(
            sql,
            {
                "limit": int(limit),
                "after_id": int(after_id),
            },
        ).fetchall()

        return [int(row[0]) for row in rows]

    @with_request_id
    def process_one(
        self,
        *,
        session: Session,
        complete_document_id: int,
        force: bool = False,
        request_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        try:
            result = self.summary_service.summarize_complete_document_chunks(
                session=session,
                complete_document_id=complete_document_id,
                force=force,
                request_id=request_id,
            )

            info_id(
                f"[CompleteDocumentSummaryMaintenanceService] Processed "
                f"complete_document_id={complete_document_id} "
                f"status={result.get('status')} "
                f"embedding={result.get('summary_embedding_stored')}",
                request_id,
            )

            return result

        except Exception as e:
            error_id(
                f"[CompleteDocumentSummaryMaintenanceService] Failed processing "
                f"complete_document_id={complete_document_id}: {e}",
                request_id,
                exc_info=True,
            )

            return {
                "success": False,
                "complete_document_id": complete_document_id,
                "status": "error",
                "error": str(e),
                "summary_embedding_stored": False,
            }

    def _build_select_sql(
        self,
        *,
        include_missing_embedding: bool,
        force: bool = False,
    ):
        """
        Build SQL for selecting target document IDs.

        All branches include:

            id > :after_id

        This is required so the orchestrator can move forward through the table
        instead of repeatedly selecting the same lowest IDs.
        """

        if force:
            return text(
                """
                SELECT id
                FROM public.complete_document
                WHERE id > :after_id
                  AND content IS NOT NULL
                  AND TRIM(content) <> ''
                ORDER BY id ASC
                LIMIT :limit
                """
            )

        if include_missing_embedding:
            return text(
                """
                SELECT id
                FROM public.complete_document
                WHERE id > :after_id
                  AND content IS NOT NULL
                  AND TRIM(content) <> ''
                  AND (
                        summary IS NULL
                     OR TRIM(summary) = ''
                     OR rag_metadata IS NULL
                     OR summary_embedding_vector IS NULL
                  )
                ORDER BY id ASC
                LIMIT :limit
                """
            )

        return text(
            """
            SELECT id
            FROM public.complete_document
            WHERE id > :after_id
              AND content IS NOT NULL
              AND TRIM(content) <> ''
              AND (
                    summary IS NULL
                 OR TRIM(summary) = ''
                 OR rag_metadata IS NULL
              )
            ORDER BY id ASC
            LIMIT :limit
            """
        )

    def _build_count_sql(
        self,
        *,
        include_missing_embedding: bool,
        force: bool = False,
    ):
        """
        Build SQL for counting starting targets.

        This intentionally does not use after_id because count_targets() is used
        to calculate the total number of matching rows at the start of the run.
        """

        if force:
            return text(
                """
                SELECT COUNT(*)
                FROM public.complete_document
                WHERE content IS NOT NULL
                  AND TRIM(content) <> ''
                """
            )

        if include_missing_embedding:
            return text(
                """
                SELECT COUNT(*)
                FROM public.complete_document
                WHERE content IS NOT NULL
                  AND TRIM(content) <> ''
                  AND (
                        summary IS NULL
                     OR TRIM(summary) = ''
                     OR rag_metadata IS NULL
                     OR summary_embedding_vector IS NULL
                  )
                """
            )

        return text(
            """
            SELECT COUNT(*)
            FROM public.complete_document
            WHERE content IS NOT NULL
              AND TRIM(content) <> ''
              AND (
                    summary IS NULL
                 OR TRIM(summary) = ''
                 OR rag_metadata IS NULL
              )
            """
        )