from __future__ import annotations

from typing import Dict, Any, Optional, List

from modules.configuration.config_env import DatabaseConfig
from modules.configuration.log_config import (
    info_id,
    error_id,
    with_request_id,
)

from modules.database_manager.services.complete_document_summary_maintenance_service import (
    CompleteDocumentSummaryMaintenanceService,
)


class CompleteDocumentSummaryMaintenanceOrchestrator:
    """
    Orchestrates complete_document summary/profile database maintenance.

    Responsibilities:
      - Owns DB session lifecycle
      - Owns commit / rollback
      - Calls CompleteDocumentSummaryMaintenanceService
      - Supports dry run, batching, max document limits
      - Supports force regeneration
      - Supports summary quality maintenance
      - Supports cursor batching with after_id

    Maintenance goals:
      - Regenerate missing summaries
      - Regenerate polluted chunk-dump summaries
      - Regenerate summaries that are too short or too long
      - Regenerate missing/incomplete rag_metadata
      - Create missing summary embeddings when requested

    Important:
      The maintenance service itself does not own the DB session. This
      orchestrator owns commit/rollback.

      CompleteDocumentSummaryService may release/rollback the active transaction
      before long AI generation when release_transaction_during_ai=True. That is
      intentional to prevent PostgreSQL idle-in-transaction timeouts.
    """

    def __init__(
        self,
        *,
        store_summary_embedding: bool = True,
        release_transaction_during_ai: bool = True,
        min_summary_chars: int = 80,
        max_summary_chars: int = 1800,
    ):
        self.db_config = DatabaseConfig()

        self.store_summary_embedding = bool(store_summary_embedding)
        self.release_transaction_during_ai = bool(release_transaction_during_ai)
        self.min_summary_chars = max(1, int(min_summary_chars or 80))
        self.max_summary_chars = max(
            self.min_summary_chars + 1,
            int(max_summary_chars or 1800),
        )

        self.maintenance_service = CompleteDocumentSummaryMaintenanceService(
            store_summary_embedding=self.store_summary_embedding,
            release_transaction_during_ai=self.release_transaction_during_ai,
            min_summary_chars=self.min_summary_chars,
            max_summary_chars=self.max_summary_chars,
        )

    @with_request_id
    def run(
        self,
        *,
        batch_size: int = 5,
        max_documents: int = 0,
        force: bool = False,
        dry_run: bool = False,
        include_missing_embedding: bool = True,
        maintenance_mode: bool = True,
        min_summary_chars: Optional[int] = None,
        max_summary_chars: Optional[int] = None,
        request_id: Optional[str] = None,
    ) -> Dict[str, Any]:

        if batch_size <= 0:
            raise ValueError("batch_size must be greater than 0")

        if max_documents < 0:
            raise ValueError("max_documents cannot be negative")

        # Allow the runner to override these at run time.
        if min_summary_chars is not None:
            self.min_summary_chars = max(1, int(min_summary_chars))

        if max_summary_chars is not None:
            self.max_summary_chars = max(
                self.min_summary_chars + 1,
                int(max_summary_chars),
            )

        # Keep the service in sync with runtime overrides.
        if hasattr(self.maintenance_service, "min_summary_chars"):
            self.maintenance_service.min_summary_chars = self.min_summary_chars

        if hasattr(self.maintenance_service, "max_summary_chars"):
            self.maintenance_service.max_summary_chars = self.max_summary_chars

        processed = 0
        summarized = 0
        skipped = 0
        failed = 0
        embeddings_stored = 0
        maintenance_forced = 0
        regeneration_required = 0
        embedding_required = 0
        batches_processed = 0
        last_seen_id = 0

        results: List[Dict[str, Any]] = []

        try:
            with self.db_config.main_session() as session:
                total_targets = self.maintenance_service.count_targets(
                    session=session,
                    include_missing_embedding=include_missing_embedding,
                    force=force,
                )

                effective_target_total = (
                    total_targets
                    if not max_documents
                    else min(total_targets, max_documents)
                )

                info_id(
                    f"[CompleteDocumentSummaryMaintenanceOrchestrator] "
                    f"Targets found={total_targets} "
                    f"effective_target_total={effective_target_total} "
                    f"batch_size={batch_size} "
                    f"max_documents={max_documents} "
                    f"force={force} "
                    f"dry_run={dry_run} "
                    f"maintenance_mode={maintenance_mode} "
                    f"include_missing_embedding={include_missing_embedding} "
                    f"store_summary_embedding={self.store_summary_embedding} "
                    f"release_transaction_during_ai={self.release_transaction_during_ai} "
                    f"min_summary_chars={self.min_summary_chars} "
                    f"max_summary_chars={self.max_summary_chars}",
                    request_id,
                )

                if dry_run:
                    preview_limit = (
                        batch_size
                        if not max_documents
                        else min(batch_size, max_documents)
                    )

                    preview_ids = self.maintenance_service.get_target_ids(
                        session=session,
                        limit=preview_limit,
                        include_missing_embedding=include_missing_embedding,
                        force=force,
                        after_id=0,
                    )

                    preview_states: List[Dict[str, Any]] = []

                    for complete_document_id in preview_ids:
                        try:
                            state = (
                                self.maintenance_service.get_document_maintenance_state(
                                    session=session,
                                    complete_document_id=complete_document_id,
                                )
                            )
                            preview_states.append(state)
                        except Exception as state_error:
                            preview_states.append(
                                {
                                    "exists": None,
                                    "complete_document_id": complete_document_id,
                                    "status": "maintenance_state_error",
                                    "error": str(state_error),
                                }
                            )

                    return {
                        "success": True,
                        "dry_run": True,
                        "force": force,
                        "maintenance_mode": maintenance_mode,
                        "include_missing_embedding": include_missing_embedding,
                        "store_summary_embedding": self.store_summary_embedding,
                        "release_transaction_during_ai": self.release_transaction_during_ai,
                        "min_summary_chars": self.min_summary_chars,
                        "max_summary_chars": self.max_summary_chars,
                        "batch_size": batch_size,
                        "max_documents": max_documents,
                        "total_targets": total_targets,
                        "effective_target_total": effective_target_total,
                        "preview_ids": preview_ids,
                        "preview_states": preview_states,
                    }

                while True:
                    if max_documents and processed >= max_documents:
                        info_id(
                            f"[CompleteDocumentSummaryMaintenanceOrchestrator] "
                            f"Stopping because max_documents reached. "
                            f"processed={processed} max_documents={max_documents}",
                            request_id,
                        )
                        break

                    remaining_allowed = (
                        batch_size
                        if not max_documents
                        else min(batch_size, max_documents - processed)
                    )

                    if remaining_allowed <= 0:
                        break

                    target_ids = self.maintenance_service.get_target_ids(
                        session=session,
                        limit=remaining_allowed,
                        include_missing_embedding=include_missing_embedding,
                        force=force,
                        after_id=last_seen_id,
                    )

                    if not target_ids:
                        info_id(
                            f"[CompleteDocumentSummaryMaintenanceOrchestrator] "
                            f"No more target ids found after_id={last_seen_id}.",
                            request_id,
                        )
                        break

                    batches_processed += 1
                    batch_start_id = target_ids[0]
                    batch_end_id = target_ids[-1]
                    last_seen_id = max(target_ids)

                    info_id(
                        f"[CompleteDocumentSummaryMaintenanceOrchestrator] "
                        f"Processing batch #{batches_processed} "
                        f"ids={target_ids} "
                        f"range={batch_start_id}-{batch_end_id} "
                        f"last_seen_id={last_seen_id} "
                        f"force={force} "
                        f"maintenance_mode={maintenance_mode}",
                        request_id,
                    )

                    for complete_document_id in target_ids:
                        try:
                            result = self.maintenance_service.process_one(
                                session=session,
                                complete_document_id=complete_document_id,
                                force=force,
                                request_id=request_id,
                            )

                            session.commit()

                            processed += 1
                            results.append(result)

                            status = result.get("status")
                            success = bool(result.get("success"))

                            if status in {
                                "summarized_from_chunks",
                                "summarized_from_content",
                            }:
                                summarized += 1
                            elif status == "already_summarized":
                                skipped += 1

                            if not success:
                                failed += 1

                            if result.get("summary_embedding_stored"):
                                embeddings_stored += 1

                            if result.get("maintenance_force_applied"):
                                maintenance_forced += 1

                            if result.get("maintenance_regenerate_required"):
                                regeneration_required += 1

                            maintenance_state = (
                                result.get("maintenance_state_before")
                                if isinstance(
                                    result.get("maintenance_state_before"),
                                    dict,
                                )
                                else {}
                            )

                            if maintenance_state.get("embedding_required"):
                                embedding_required += 1

                        except Exception as e:
                            session.rollback()

                            processed += 1
                            failed += 1

                            error_id(
                                f"[CompleteDocumentSummaryMaintenanceOrchestrator] "
                                f"Failed complete_document_id={complete_document_id}: {e}",
                                request_id,
                                exc_info=True,
                            )

                            results.append(
                                {
                                    "success": False,
                                    "complete_document_id": complete_document_id,
                                    "status": "error",
                                    "error": str(e),
                                    "summary_embedding_stored": False,
                                }
                            )

                    info_id(
                        f"[CompleteDocumentSummaryMaintenanceOrchestrator] "
                        f"Progress processed={processed}/{effective_target_total} "
                        f"summarized={summarized} "
                        f"skipped={skipped} "
                        f"failed={failed} "
                        f"embeddings_stored={embeddings_stored} "
                        f"regeneration_required={regeneration_required} "
                        f"maintenance_forced={maintenance_forced} "
                        f"embedding_required={embedding_required} "
                        f"batches_processed={batches_processed} "
                        f"last_seen_id={last_seen_id}",
                        request_id,
                    )

            return {
                "success": True,
                "dry_run": False,
                "force": force,
                "maintenance_mode": maintenance_mode,
                "include_missing_embedding": include_missing_embedding,
                "store_summary_embedding": self.store_summary_embedding,
                "release_transaction_during_ai": self.release_transaction_during_ai,
                "min_summary_chars": self.min_summary_chars,
                "max_summary_chars": self.max_summary_chars,
                "batch_size": batch_size,
                "max_documents": max_documents,
                "total_targets_start": total_targets,
                "effective_target_total": effective_target_total,
                "processed": processed,
                "summarized": summarized,
                "skipped": skipped,
                "failed": failed,
                "summary_embeddings_stored": embeddings_stored,
                "regeneration_required": regeneration_required,
                "maintenance_forced": maintenance_forced,
                "embedding_required": embedding_required,
                "batches_processed": batches_processed,
                "last_seen_id": last_seen_id,
                "results": results,
            }

        except Exception as e:
            error_id(
                f"[CompleteDocumentSummaryMaintenanceOrchestrator] Run failed: {e}",
                request_id,
                exc_info=True,
            )
            raise