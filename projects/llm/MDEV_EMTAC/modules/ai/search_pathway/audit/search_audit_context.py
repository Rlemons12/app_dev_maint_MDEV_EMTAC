from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any
from uuid import UUID

from modules.ai.search_pathway.audit.search_audit_stage_tracker import (
    SearchAuditStageTracker,
)


@dataclass
class SearchAuditContext:
    """
    Per-request audit context for one search pathway run.

    This object travels through the search pathway so decorators and services
    can capture audit information without opening sessions or committing.

    The orchestrator should create this object near the beginning of the request.
    """

    request_id: str
    pathway_name: str = "rag"
    pathway_version: str = "1.0"

    user_id: str | None = None
    session_id: UUID | None = None
    qanda_id: UUID | None = None

    question: str | None = None
    normalized_question: str | None = None
    answer: str | None = None

    model_name: str | None = None
    duration_ms: int | None = None

    metadata: dict[str, Any] = field(default_factory=dict)

    stage_tracker: SearchAuditStageTracker = field(init=False)

    def __post_init__(self) -> None:
        self.stage_tracker = SearchAuditStageTracker(
            request_id=self.request_id,
            pathway_name=self.pathway_name,
        )

    def to_summary(self) -> dict[str, Any]:
        return {
            "request_id": self.request_id,
            "pathway_name": self.pathway_name,
            "pathway_version": self.pathway_version,
            "user_id": self.user_id,
            "session_id": str(self.session_id) if self.session_id else None,
            "qanda_id": str(self.qanda_id) if self.qanda_id else None,
            "question": self.question,
            "normalized_question": self.normalized_question,
            "model_name": self.model_name,
            "duration_ms": self.duration_ms,
            "metadata": self.metadata,
            "stages": self.stage_tracker.to_dict(),
        }