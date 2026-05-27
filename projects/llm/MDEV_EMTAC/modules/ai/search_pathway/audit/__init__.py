"""
Search pathway auditing package.

Public runtime API for auditing AI/search pathways.

Initial focus:
    - RAG search auditing
    - payload projection auditing

Future expansion:
    - keyword search auditing
    - vector search auditing
    - hybrid search auditing
    - part search auditing
    - drawing search auditing
    - image search auditing
    - AI-based result review

Do not import CLI/setup scripts here:
    - create_search_audit_schema.py
    - search_audit_reporter.py

Those scripts should be run directly with:
    python -m modules.ai.search_pathway.audit.create_search_audit_schema
    python -m modules.ai.search_pathway.audit.search_audit_reporter
"""

from modules.ai.search_pathway.audit.search_audit_types import (
    SearchPathwayName,
    SearchAuditStageName,
    SearchAuditItemType,
    SearchAuditStatus,
    SearchAuditValidationStatus,
)

from modules.ai.search_pathway.audit.search_audit_logger import (
    get_search_audit_logger,
    get_search_audit_log_manager,
)

from modules.ai.search_pathway.audit.search_audit_payload_extractor import (
    SearchAuditPayloadExtractor,
)

from modules.ai.search_pathway.audit.search_audit_service import (
    SearchAuditService,
)

from modules.ai.search_pathway.audit.search_audit_background_worker import (
    SearchAuditBackgroundWorker,
    get_search_audit_background_worker,
    enqueue_search_audit_payload_details,
    shutdown_search_audit_background_worker,
)

from modules.ai.search_pathway.audit.search_audit_stage_tracker import (
    SearchAuditStageRecord,
    SearchAuditStageTracker,
)

from modules.ai.search_pathway.audit.search_audit_validator import (
    SearchAuditValidator,
)

from modules.ai.search_pathway.audit.search_audit_evidence_builder import (
    SearchAuditEvidenceBuilder,
)

from modules.ai.search_pathway.audit.search_audit_ai_reviewer import (
    SearchAuditAIReviewer,
)


__all__ = [
    # Enums / types
    "SearchPathwayName",
    "SearchAuditStageName",
    "SearchAuditItemType",
    "SearchAuditStatus",
    "SearchAuditValidationStatus",

    # Logging
    "get_search_audit_logger",
    "get_search_audit_log_manager",

    # Payload extraction
    "SearchAuditPayloadExtractor",

    # Main service
    "SearchAuditService",

    # Background worker
    "SearchAuditBackgroundWorker",
    "get_search_audit_background_worker",
    "enqueue_search_audit_payload_details",
    "shutdown_search_audit_background_worker",

    # Stage tracking
    "SearchAuditStageRecord",
    "SearchAuditStageTracker",

    # Validation / evidence / AI review
    "SearchAuditValidator",
    "SearchAuditEvidenceBuilder",
    "SearchAuditAIReviewer",
]