from __future__ import annotations

from typing import Dict, Any, List, Optional

from sqlalchemy import text
from sqlalchemy.orm import Session

from modules.configuration.log_config import (
    info_id,
    error_id,
    warning_id,
    with_request_id,
)

from modules.services.complete_document_summary_service import (
    CompleteDocumentSummaryService,
)


class CompleteDocumentSummaryMaintenanceService:
    """
    Database maintenance service for complete_document summary/profile quality.

    Purpose:
      - Find complete_document rows whose summary/profile data is missing,
        stale, too short, too long, or polluted with old chunk-summary text.
      - Send those rows through CompleteDocumentSummaryService.
      - Make sure CompleteDocument.summary remains concise and human-readable.
      - Make sure rag_metadata["retrieval_text"] exists for retrieval/profile use.
      - Optionally make sure summary_embedding_vector exists.

    Rules:
      - This class does NOT open sessions.
      - This class does NOT commit.
      - This class does NOT rollback directly.

    Important:
      CompleteDocumentSummaryService may release/rollback the active transaction
      before local AI generation when release_transaction_during_ai=True. This is
      intentional to prevent PostgreSQL idle-in-transaction timeouts during long
      summary generation.

    Cursor batching:
      This service supports cursor-based batching using after_id.

      That prevents force=True or broad maintenance mode from repeatedly selecting
      the same first batch:

          [1, 2, 3, 4, 5]
          [1, 2, 3, 4, 5]
          [1, 2, 3, 4, 5]

      Instead it advances through IDs:

          [1, 2, 3, 4, 5]
          [6, 7, 8, 9, 10]
          [11, 12, 13, 14, 15]
    """

    MIN_SUMMARY_CHARS = 80
    MAX_SUMMARY_CHARS = 1800

    # Only patterns that should never appear in the human-facing
    # complete_document.summary field.
    #
    # Important:
    # Do NOT include "Document title:" here. That phrase is valid in
    # rag_metadata["retrieval_text"].
    SUMMARY_BAD_PATTERNS = (
        "Combined Chunk Summaries",
        "CHUNK SUMMARY",
        "Chunk ID:",
        "Name: chunk_",
        "chunk_",
        "Useful extracted document signals",
    )

    # Only patterns that indicate rag_metadata is polluted with old
    # chunk-summary dump text.
    #
    # Important:
    # Do NOT include "Document title:" here. CompleteDocumentSummaryService
    # intentionally stores retrieval_text with a "Document title:" line.
    RAG_METADATA_BAD_PATTERNS = (
        "Combined Chunk Summaries",
        "CHUNK SUMMARY",
        "Chunk ID:",
        "Name: chunk_",
        "Useful extracted document signals",
    )

    REQUIRED_RAG_METADATA_KEYS = (
        "summary",
        "retrieval_text",
    )

    def __init__(
        self,
        *,
        store_summary_embedding: bool = True,
        release_transaction_during_ai: bool = True,
        min_summary_chars: int = MIN_SUMMARY_CHARS,
        max_summary_chars: int = MAX_SUMMARY_CHARS,
    ):
        self.store_summary_embedding = bool(store_summary_embedding)
        self.release_transaction_during_ai = bool(release_transaction_during_ai)

        self.min_summary_chars = max(
            1,
            int(min_summary_chars or self.MIN_SUMMARY_CHARS),
        )
        self.max_summary_chars = max(
            self.min_summary_chars + 1,
            int(max_summary_chars or self.MAX_SUMMARY_CHARS),
        )

        self.summary_service = CompleteDocumentSummaryService(
            store_summary_embedding=store_summary_embedding,
            release_transaction_during_ai=release_transaction_during_ai,
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

        return int(session.execute(sql, self._sql_params()).scalar() or 0)

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
                When True, include documents that already have good summary/rag_metadata
                but are missing summary_embedding_vector.

            force:
                When True, include every complete_document with either document-level
                content or child chunk content.

            after_id:
                Cursor value. Only rows with id > after_id are returned.

        Maintenance target rules:
            A document is selected when it has usable source text and any of these
            are true:

              - force=True
              - summary is missing
              - summary is too short
              - summary is too long
              - summary looks like old chunk-summary dump text
              - rag_metadata is missing
              - rag_metadata["summary"] is missing
              - rag_metadata["retrieval_text"] is missing
              - rag_metadata contains old chunk-summary dump text
              - summary_embedding_vector is missing and include_missing_embedding=True
        """

        if limit <= 0:
            return []

        if after_id < 0:
            after_id = 0

        sql = self._build_select_sql(
            include_missing_embedding=include_missing_embedding,
            force=force,
        )

        params = self._sql_params()
        params.update(
            {
                "limit": int(limit),
                "after_id": int(after_id),
            }
        )

        rows = session.execute(sql, params).fetchall()

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
        """
        Process one complete_document row.

        This method decides whether the document needs a full summary regeneration
        or only an embedding/profile refresh.

        If the row has a bad old summary, this forces regeneration even when the
        caller passed force=False. That prevents old "Combined Chunk Summaries"
        text from being treated as already summarized.
        """

        try:
            maintenance_state = self.get_document_maintenance_state(
                session=session,
                complete_document_id=complete_document_id,
            )

            if not maintenance_state.get("exists"):
                return {
                    "success": False,
                    "complete_document_id": complete_document_id,
                    "status": "document_not_found",
                    "summary_embedding_stored": False,
                    "maintenance_state": maintenance_state,
                }

            if not maintenance_state.get("has_source_text"):
                return {
                    "success": False,
                    "complete_document_id": complete_document_id,
                    "status": "no_source_text",
                    "summary_embedding_stored": False,
                    "maintenance_state": maintenance_state,
                }

            regenerate_required = bool(maintenance_state.get("regenerate_required"))
            effective_force = bool(force or regenerate_required)

            if regenerate_required and not force:
                warning_id(
                    f"[CompleteDocumentSummaryMaintenanceService] Forcing regeneration "
                    f"because summary/profile failed maintenance checks "
                    f"complete_document_id={complete_document_id} "
                    f"reasons={maintenance_state.get('reasons')}",
                    request_id,
                )

            result = self.summary_service.summarize_complete_document_chunks(
                session=session,
                complete_document_id=complete_document_id,
                force=effective_force,
                request_id=request_id,
            )

            result["maintenance_state_before"] = maintenance_state
            result["maintenance_force_applied"] = effective_force
            result["maintenance_regenerate_required"] = regenerate_required

            info_id(
                f"[CompleteDocumentSummaryMaintenanceService] Processed "
                f"complete_document_id={complete_document_id} "
                f"status={result.get('status')} "
                f"force={effective_force} "
                f"regenerate_required={regenerate_required} "
                f"embedding={result.get('summary_embedding_stored')} "
                f"reasons={maintenance_state.get('reasons')}",
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

    def get_document_maintenance_state(
        self,
        *,
        session: Session,
        complete_document_id: int,
    ) -> Dict[str, Any]:
        """
        Inspect one complete_document row and explain why it does or does not
        need maintenance.

        This is useful for debugging:
            - why a document was selected
            - why regeneration was forced
            - whether only an embedding is missing
        """

        sql = text(
            f"""
            SELECT
                cd.id,
                cd.title,
                ({self._has_source_text_sql()}) AS has_source_text,
                cd.summary IS NULL
                    OR TRIM(COALESCE(cd.summary, '')) = '' AS missing_summary,
                LENGTH(TRIM(COALESCE(cd.summary, ''))) AS summary_chars,
                ({self._bad_summary_sql()}) AS bad_summary,
                ({self._summary_too_short_sql()}) AS summary_too_short,
                ({self._summary_too_long_sql()}) AS summary_too_long,
                cd.rag_metadata IS NULL AS missing_rag_metadata,
                ({self._missing_required_rag_metadata_sql()})
                    AS missing_required_rag_metadata,
                ({self._bad_rag_metadata_sql()}) AS bad_rag_metadata,
                cd.summary_embedding_vector IS NULL AS missing_summary_embedding
            FROM public.complete_document cd
            WHERE cd.id = :complete_document_id
            LIMIT 1
            """
        )

        params = self._sql_params()
        params["complete_document_id"] = int(complete_document_id)

        row = session.execute(sql, params).mappings().first()

        if not row:
            return {
                "exists": False,
                "complete_document_id": complete_document_id,
                "has_source_text": False,
                "reasons": ["document_not_found"],
                "regenerate_required": False,
                "embedding_required": False,
            }

        reasons: List[str] = []

        has_source_text = bool(row.get("has_source_text"))

        if not has_source_text:
            reasons.append("no_source_text")

        checks = {
            "missing_summary": bool(row.get("missing_summary")),
            "bad_summary": bool(row.get("bad_summary")),
            "summary_too_short": bool(row.get("summary_too_short")),
            "summary_too_long": bool(row.get("summary_too_long")),
            "missing_rag_metadata": bool(row.get("missing_rag_metadata")),
            "missing_required_rag_metadata": bool(
                row.get("missing_required_rag_metadata")
            ),
            "bad_rag_metadata": bool(row.get("bad_rag_metadata")),
            "missing_summary_embedding": bool(row.get("missing_summary_embedding")),
        }

        for reason, enabled in checks.items():
            if enabled:
                reasons.append(reason)

        regenerate_required = bool(
            has_source_text
            and (
                checks["missing_summary"]
                or checks["bad_summary"]
                or checks["summary_too_short"]
                or checks["summary_too_long"]
                or checks["missing_rag_metadata"]
                or checks["missing_required_rag_metadata"]
                or checks["bad_rag_metadata"]
            )
        )

        embedding_required = bool(
            has_source_text and checks["missing_summary_embedding"]
        )

        return {
            "exists": True,
            "complete_document_id": int(row.get("id")),
            "title": row.get("title"),
            "has_source_text": has_source_text,
            "summary_chars": int(row.get("summary_chars") or 0),
            "reasons": reasons,
            "regenerate_required": regenerate_required,
            "embedding_required": embedding_required,
            **checks,
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

            cd.id > :after_id

        This is required so the runner can move forward through the table
        instead of repeatedly selecting the same lowest IDs.
        """

        if force:
            return text(
                f"""
                SELECT cd.id
                FROM public.complete_document cd
                WHERE cd.id > :after_id
                  AND ({self._has_source_text_sql()})
                ORDER BY cd.id ASC
                LIMIT :limit
                """
            )

        maintenance_predicate = self._maintenance_predicate_sql(
            include_missing_embedding=include_missing_embedding,
        )

        return text(
            f"""
            SELECT cd.id
            FROM public.complete_document cd
            WHERE cd.id > :after_id
              AND ({self._has_source_text_sql()})
              AND ({maintenance_predicate})
            ORDER BY cd.id ASC
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
                f"""
                SELECT COUNT(*)
                FROM public.complete_document cd
                WHERE ({self._has_source_text_sql()})
                """
            )

        maintenance_predicate = self._maintenance_predicate_sql(
            include_missing_embedding=include_missing_embedding,
        )

        return text(
            f"""
            SELECT COUNT(*)
            FROM public.complete_document cd
            WHERE ({self._has_source_text_sql()})
              AND ({maintenance_predicate})
            """
        )

    def _maintenance_predicate_sql(
        self,
        *,
        include_missing_embedding: bool,
    ) -> str:
        predicates = [
            "cd.summary IS NULL",
            "TRIM(COALESCE(cd.summary, '')) = ''",
            self._summary_too_short_sql(),
            self._summary_too_long_sql(),
            self._bad_summary_sql(),
            "cd.rag_metadata IS NULL",
            self._missing_required_rag_metadata_sql(),
            self._bad_rag_metadata_sql(),
        ]

        if include_missing_embedding:
            predicates.append("cd.summary_embedding_vector IS NULL")

        return "\n OR ".join(f"({predicate})" for predicate in predicates)

    @staticmethod
    def _has_source_text_sql() -> str:
        """
        A complete_document can be summarized from either:
          - complete_document.content
          - public.document chunks linked by complete_document_id

        Do not require complete_document.content only, because many uploaded
        documents are represented primarily by child chunks.
        """

        return """
            (
                cd.content IS NOT NULL
                AND TRIM(cd.content) <> ''
            )
            OR EXISTS (
                SELECT 1
                FROM public.document d
                WHERE d.complete_document_id = cd.id
                  AND d.content IS NOT NULL
                  AND TRIM(d.content) <> ''
            )
        """

    def _summary_too_short_sql(self) -> str:
        return """
            cd.summary IS NOT NULL
            AND TRIM(COALESCE(cd.summary, '')) <> ''
            AND LENGTH(TRIM(COALESCE(cd.summary, ''))) < :min_summary_chars
        """

    def _summary_too_long_sql(self) -> str:
        return """
            cd.summary IS NOT NULL
            AND LENGTH(TRIM(COALESCE(cd.summary, ''))) > :max_summary_chars
        """

    def _bad_summary_sql(self) -> str:
        pattern_checks = []

        for index, _pattern in enumerate(self.SUMMARY_BAD_PATTERNS):
            pattern_checks.append(
                f"COALESCE(cd.summary, '') ILIKE :bad_summary_pattern_{index}"
            )

        return "\n OR ".join(pattern_checks) or "FALSE"

    def _bad_rag_metadata_sql(self) -> str:
        """
        Detect polluted human-facing rag_metadata fields.

        Important:
        Do NOT scan the entire rag_metadata::text blob because valid metadata
        contains internal keys/values such as:
            - chunk_summary_count
            - clean_chunk_map_reduce_summary

        Those are valid implementation metadata and should not trigger
        bad_rag_metadata=True.

        Only inspect the human-facing fields:
            - rag_metadata["summary"]
            - rag_metadata["retrieval_text"]
        """

        pattern_checks = []

        for index, _pattern in enumerate(self.RAG_METADATA_BAD_PATTERNS):
            pattern_checks.append(
                f"""
                (
                    COALESCE(cd.rag_metadata::jsonb ->> 'summary', '')
                        ILIKE :bad_rag_metadata_pattern_{index}
                    OR COALESCE(cd.rag_metadata::jsonb ->> 'retrieval_text', '')
                        ILIKE :bad_rag_metadata_pattern_{index}
                )
                """
            )

        return "\n OR ".join(pattern_checks) or "FALSE"

    def _missing_required_rag_metadata_sql(self) -> str:
        """
        Detect profile metadata that is missing the fields used by the RAG profile path.

        Supports JSON or JSONB rag_metadata by casting to jsonb before key checks.
        """

        checks = []

        for key in self.REQUIRED_RAG_METADATA_KEYS:
            checks.append(
                f"""
                (
                    cd.rag_metadata IS NOT NULL
                    AND (
                        NOT (cd.rag_metadata::jsonb ? '{key}')
                        OR TRIM(COALESCE(cd.rag_metadata::jsonb ->> '{key}', '')) = ''
                    )
                )
                """
            )

        return "\n OR ".join(checks) or "FALSE"

    def _sql_params(self) -> Dict[str, Any]:
        params: Dict[str, Any] = {
            "min_summary_chars": int(self.min_summary_chars),
            "max_summary_chars": int(self.max_summary_chars),
        }

        for index, pattern in enumerate(self.SUMMARY_BAD_PATTERNS):
            params[f"bad_summary_pattern_{index}"] = f"%{pattern}%"

        for index, pattern in enumerate(self.RAG_METADATA_BAD_PATTERNS):
            params[f"bad_rag_metadata_pattern_{index}"] = f"%{pattern}%"

        return params