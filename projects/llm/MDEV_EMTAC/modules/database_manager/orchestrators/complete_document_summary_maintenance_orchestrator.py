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
    Orchestrates CompleteDocument summary cleanup.

    Uses CompleteDocumentSummaryMaintenanceService.process_one().

    Responsibilities:
      - Owns DB session lifecycle
      - Owns commit / rollback
      - Calls maintenance service
      - Supports dry run, batching, max document limits
      - Supports force regeneration
      - Prevents force=True from looping over the same batch forever

    Important:
      When force=True, the target query intentionally includes documents
      even if they already have summaries. Because of that, this orchestrator
      must use an ID cursor so each document is only selected once per run.
    """

    def __init__(
        self,
        *,
        store_summary_embedding: bool = True,
    ):
        self.db_config = DatabaseConfig()
        self.maintenance_service = CompleteDocumentSummaryMaintenanceService(
            store_summary_embedding=store_summary_embedding,
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
        request_id: Optional[str] = None,
    ) -> Dict[str, Any]:

        if batch_size <= 0:
            raise ValueError("batch_size must be greater than 0")

        if max_documents < 0:
            raise ValueError("max_documents cannot be negative")

        processed = 0
        summarized = 0
        skipped = 0
        failed = 0
        embeddings_stored = 0
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
                    f"include_missing_embedding={include_missing_embedding}",
                    request_id,
                )

                if dry_run:
                    preview_ids = self.maintenance_service.get_target_ids(
                        session=session,
                        limit=batch_size,
                        include_missing_embedding=include_missing_embedding,
                        force=force,
                        after_id=0,
                    )

                    return {
                        "success": True,
                        "dry_run": True,
                        "force": force,
                        "include_missing_embedding": include_missing_embedding,
                        "batch_size": batch_size,
                        "max_documents": max_documents,
                        "total_targets": total_targets,
                        "effective_target_total": effective_target_total,
                        "preview_ids": preview_ids,
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
                        f"force={force}",
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

                            if status in {
                                "summarized_from_chunks",
                                "summarized_from_content",
                            }:
                                summarized += 1
                            elif status == "already_summarized":
                                skipped += 1

                            if not result.get("success"):
                                failed += 1

                            if result.get("summary_embedding_stored"):
                                embeddings_stored += 1

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
                        f"batches_processed={batches_processed} "
                        f"last_seen_id={last_seen_id}",
                        request_id,
                    )

            return {
                "success": True,
                "dry_run": False,
                "force": force,
                "include_missing_embedding": include_missing_embedding,
                "batch_size": batch_size,
                "max_documents": max_documents,
                "total_targets_start": total_targets,
                "effective_target_total": effective_target_total,
                "processed": processed,
                "summarized": summarized,
                "skipped": skipped,
                "failed": failed,
                "summary_embeddings_stored": embeddings_stored,
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