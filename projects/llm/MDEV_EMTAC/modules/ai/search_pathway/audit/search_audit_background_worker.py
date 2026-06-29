# ============================================================
# File:
#   modules/ai/search_pathway/audit/search_audit_background_worker.py
#
# Purpose:
#   Run heavy search-audit detail capture after the payload response has
#   already been built.
#
# Main use case:
#   /ask/payload should return quickly.
#   Detailed audit item inserts can happen in a background thread.
#
# Architecture rule:
#   This worker owns its own DB session and commit/rollback boundary.
#   SearchAuditService still does not create sessions or commit.
# ============================================================

from __future__ import annotations

import atexit
import copy
import threading
import time
import traceback
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any
from uuid import UUID

from modules.configuration.config_env import get_db_config

from modules.ai.search_pathway.audit.search_audit_logger import (
    get_search_audit_log_manager,
)
from modules.ai.search_pathway.audit.search_audit_service import SearchAuditService


@dataclass(frozen=True)
class SearchAuditBackgroundJob:
    """
    Immutable job description for background search audit detail capture.
    """

    audit_run_id: UUID | str
    request_id: str
    pathway_name: str
    response: dict[str, Any]
    replace_existing: bool = False


class SearchAuditBackgroundWorker:
    """
    Background worker for heavy search-audit operations.

    This is intended for delayed payload item capture.

    The worker:
        - receives a completed payload/search response
        - opens its own DB session
        - calls SearchAuditService.record_payload_items_for_existing_run()
        - commits independently of the user-facing request
        - logs success/failure to the dedicated search audit logger

    This prevents /ask/payload from waiting while thousands of audit item
    rows are inserted.
    """

    def __init__(
        self,
        *,
        max_workers: int = 1,
        thread_name_prefix: str = "search-audit-bg",
    ) -> None:
        self.max_workers = max_workers
        self.thread_name_prefix = thread_name_prefix

        self._executor = ThreadPoolExecutor(
            max_workers=max_workers,
            thread_name_prefix=thread_name_prefix,
        )

        self._lock = threading.Lock()
        self._submitted_count = 0
        self._completed_count = 0
        self._failed_count = 0
        self._shutdown = False

        self.audit_log_manager = get_search_audit_log_manager()

    def enqueue_payload_item_capture(
        self,
        *,
        audit_run_id: UUID | str | None,
        request_id: str | None,
        pathway_name: str | None,
        response: dict[str, Any] | None,
        replace_existing: bool = False,
    ) -> Future | None:
        """
        Queue detailed payload item capture.

        Returns:
            Future if queued.
            None if the job could not be queued.

        This method is intentionally fast. It does not write to the database.
        """

        if not audit_run_id:
            self.audit_log_manager.log_validation_result(
                request_id=request_id or "unknown",
                pathway_name=pathway_name or "unknown",
                check_name="background_audit_enqueue",
                check_status="failed",
                details={
                    "message": "Cannot enqueue background audit without audit_run_id.",
                },
            )
            return None

        if not isinstance(response, dict):
            self.audit_log_manager.log_validation_result(
                request_id=request_id or "unknown",
                pathway_name=pathway_name or "unknown",
                check_name="background_audit_enqueue",
                check_status="failed",
                details={
                    "message": "Cannot enqueue background audit because response is not a dict.",
                    "response_type": type(response).__name__,
                },
            )
            return None

        with self._lock:
            if self._shutdown:
                self.audit_log_manager.log_validation_result(
                    request_id=request_id or "unknown",
                    pathway_name=pathway_name or "unknown",
                    check_name="background_audit_enqueue",
                    check_status="failed",
                    details={
                        "message": "Cannot enqueue background audit because worker is shut down.",
                    },
                )
                return None

            self._submitted_count += 1
            job_number = self._submitted_count

        # Make a detached copy so the background thread is not sharing a mutable
        # response object with the request thread.
        response_snapshot = copy.deepcopy(response)

        job = SearchAuditBackgroundJob(
            audit_run_id=audit_run_id,
            request_id=request_id or "unknown",
            pathway_name=pathway_name or "unknown",
            response=response_snapshot,
            replace_existing=replace_existing,
        )

        self.audit_log_manager.log_validation_result(
            request_id=job.request_id,
            pathway_name=job.pathway_name,
            check_name="background_audit_enqueue",
            check_status="passed",
            details={
                "message": "Background audit detail capture queued.",
                "audit_run_id": str(job.audit_run_id),
                "job_number": job_number,
                "replace_existing": replace_existing,
            },
        )

        future = self._executor.submit(self._run_payload_item_capture_job, job)
        future.add_done_callback(
            lambda completed_future: self._handle_future_complete(
                completed_future=completed_future,
                job=job,
                job_number=job_number,
            )
        )

        return future

    def _run_payload_item_capture_job(
        self,
        job: SearchAuditBackgroundJob,
    ) -> dict[str, Any]:
        """
        Worker thread entry point.

        This owns the database session and transaction boundary.
        """

        started = time.perf_counter()
        session = None

        self.audit_log_manager.log_validation_result(
            request_id=job.request_id,
            pathway_name=job.pathway_name,
            check_name="background_audit_detail_capture_started",
            check_status="passed",
            details={
                "audit_run_id": str(job.audit_run_id),
            },
        )

        try:
            db_config = get_db_config()
            session = db_config.get_main_session()

            result = SearchAuditService.record_payload_items_for_existing_run(
                session=session,
                audit_run_id=job.audit_run_id,
                response=job.response,
                request_id=job.request_id,
                pathway_name=job.pathway_name,
                replace_existing=job.replace_existing,
            )

            session.commit()

            duration_ms = int((time.perf_counter() - started) * 1000)

            self.audit_log_manager.log_validation_result(
                request_id=job.request_id,
                pathway_name=job.pathway_name,
                check_name="background_audit_detail_capture_completed",
                check_status="passed",
                details={
                    "audit_run_id": str(job.audit_run_id),
                    "duration_ms": duration_ms,
                    "result": result,
                },
            )

            return {
                "success": True,
                "audit_run_id": str(job.audit_run_id),
                "request_id": job.request_id,
                "pathway_name": job.pathway_name,
                "duration_ms": duration_ms,
                "result": result,
            }

        except Exception as exc:
            if session:
                session.rollback()

            duration_ms = int((time.perf_counter() - started) * 1000)

            self.audit_log_manager.log_validation_result(
                request_id=job.request_id,
                pathway_name=job.pathway_name,
                check_name="background_audit_detail_capture_failed",
                check_status="failed",
                details={
                    "audit_run_id": str(job.audit_run_id),
                    "duration_ms": duration_ms,
                    "error": str(exc),
                    "traceback": traceback.format_exc(),
                },
            )

            raise

        finally:
            if session:
                session.close()

    def _handle_future_complete(
        self,
        *,
        completed_future: Future,
        job: SearchAuditBackgroundJob,
        job_number: int,
    ) -> None:
        """
        Callback executed when a background job completes.
        """

        try:
            result = completed_future.result()

            with self._lock:
                self._completed_count += 1

            self.audit_log_manager.log_validation_result(
                request_id=job.request_id,
                pathway_name=job.pathway_name,
                check_name="background_audit_future_completed",
                check_status="passed",
                details={
                    "audit_run_id": str(job.audit_run_id),
                    "job_number": job_number,
                    "result": result,
                    "worker_stats": self.stats(),
                },
            )

        except Exception as exc:
            with self._lock:
                self._failed_count += 1

            self.audit_log_manager.log_validation_result(
                request_id=job.request_id,
                pathway_name=job.pathway_name,
                check_name="background_audit_future_completed",
                check_status="failed",
                details={
                    "audit_run_id": str(job.audit_run_id),
                    "job_number": job_number,
                    "error": str(exc),
                    "worker_stats": self.stats(),
                },
            )

    def stats(self) -> dict[str, Any]:
        """
        Return worker stats for logging/debugging.
        """

        with self._lock:
            return {
                "submitted_count": self._submitted_count,
                "completed_count": self._completed_count,
                "failed_count": self._failed_count,
                "shutdown": self._shutdown,
                "max_workers": self.max_workers,
            }

    def shutdown(
        self,
        *,
        wait: bool = False,
        cancel_futures: bool = False,
    ) -> None:
        """
        Shut down the background executor.

        Default wait=False because this is normally called at process exit and
        we do not want shutdown to hang indefinitely.
        """

        with self._lock:
            if self._shutdown:
                return

            self._shutdown = True

        self.audit_log_manager.log_validation_result(
            request_id="system",
            pathway_name="search_audit_background_worker",
            check_name="background_audit_worker_shutdown",
            check_status="passed",
            details={
                "wait": wait,
                "cancel_futures": cancel_futures,
                "stats": self.stats(),
            },
        )

        self._executor.shutdown(
            wait=wait,
            cancel_futures=cancel_futures,
        )


# ============================================================
# Global worker accessor
# ============================================================

_WORKER_LOCK = threading.Lock()
_GLOBAL_WORKER: SearchAuditBackgroundWorker | None = None


def get_search_audit_background_worker() -> SearchAuditBackgroundWorker:
    """
    Returns a process-local singleton background worker.

    This prevents each request from creating its own ThreadPoolExecutor.
    """

    global _GLOBAL_WORKER

    if _GLOBAL_WORKER is not None:
        return _GLOBAL_WORKER

    with _WORKER_LOCK:
        if _GLOBAL_WORKER is None:
            _GLOBAL_WORKER = SearchAuditBackgroundWorker(max_workers=1)

    return _GLOBAL_WORKER


def enqueue_search_audit_payload_details(
    *,
    audit_run_id: UUID | str | None,
    request_id: str | None,
    pathway_name: str | None,
    response: dict[str, Any] | None,
    replace_existing: bool = False,
) -> Future | None:
    """
    Convenience helper for orchestrators.

    Example:
        enqueue_search_audit_payload_details(
            audit_run_id=audit_summary.get("audit_run_id"),
            request_id=request_id,
            pathway_name="payload_projection",
            response=audit_response,
        )
    """

    worker = get_search_audit_background_worker()

    return worker.enqueue_payload_item_capture(
        audit_run_id=audit_run_id,
        request_id=request_id,
        pathway_name=pathway_name,
        response=response,
        replace_existing=replace_existing,
    )


def shutdown_search_audit_background_worker() -> None:
    """
    Explicit shutdown helper.

    Usually not needed during normal Flask request handling.
    Useful for scripts/tests.
    """

    global _GLOBAL_WORKER

    with _WORKER_LOCK:
        worker = _GLOBAL_WORKER
        _GLOBAL_WORKER = None

    if worker:
        worker.shutdown(wait=False, cancel_futures=False)


atexit.register(shutdown_search_audit_background_worker)