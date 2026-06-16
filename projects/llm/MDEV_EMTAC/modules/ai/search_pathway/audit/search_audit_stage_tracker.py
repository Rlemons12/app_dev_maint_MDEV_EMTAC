from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

from modules.ai.search_pathway.audit.search_audit_logger import (
    get_search_audit_logger,
)


logger = get_search_audit_logger()


@dataclass
class SearchAuditStageRecord:
    stage_name: str
    started_at: float
    completed_at: float | None = None
    duration_ms: int | None = None
    status: str = "started"
    input_snapshot: dict[str, Any] | None = None
    output_snapshot: dict[str, Any] | None = None
    error_text: str | None = None


@dataclass
class SearchAuditStageTracker:
    """
    Lightweight in-memory tracker for search pathway stages.

    This can later be connected to audit.search_audit_stage.
    """

    request_id: str
    pathway_name: str = "rag"
    stages: list[SearchAuditStageRecord] = field(default_factory=list)

    def start_stage(
        self,
        stage_name: str,
        input_snapshot: dict[str, Any] | None = None,
    ) -> SearchAuditStageRecord:
        record = SearchAuditStageRecord(
            stage_name=stage_name,
            started_at=time.perf_counter(),
            input_snapshot=input_snapshot,
        )

        self.stages.append(record)

        logger.debug(
            "AUDIT_STAGE_START request_id=%s pathway=%s stage=%s",
            self.request_id,
            self.pathway_name,
            stage_name,
        )

        return record

    def complete_stage(
        self,
        record: SearchAuditStageRecord,
        output_snapshot: dict[str, Any] | None = None,
    ) -> None:
        completed_at = time.perf_counter()
        record.completed_at = completed_at
        record.duration_ms = int((completed_at - record.started_at) * 1000)
        record.status = "completed"
        record.output_snapshot = output_snapshot

        logger.debug(
            "AUDIT_STAGE_SUCCESS request_id=%s pathway=%s stage=%s duration_ms=%s",
            self.request_id,
            self.pathway_name,
            record.stage_name,
            record.duration_ms,
        )

    def fail_stage(
        self,
        record: SearchAuditStageRecord,
        error: Exception,
    ) -> None:
        completed_at = time.perf_counter()
        record.completed_at = completed_at
        record.duration_ms = int((completed_at - record.started_at) * 1000)
        record.status = "failed"
        record.error_text = str(error)

        logger.error(
            "AUDIT_STAGE_FAILURE request_id=%s pathway=%s stage=%s duration_ms=%s error=%s",
            self.request_id,
            self.pathway_name,
            record.stage_name,
            record.duration_ms,
            error,
            exc_info=True,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "request_id": self.request_id,
            "pathway_name": self.pathway_name,
            "stages": [
                {
                    "stage_name": stage.stage_name,
                    "duration_ms": stage.duration_ms,
                    "status": stage.status,
                    "input_snapshot": stage.input_snapshot,
                    "output_snapshot": stage.output_snapshot,
                    "error_text": stage.error_text,
                }
                for stage in self.stages
            ],
        }
