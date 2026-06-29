from __future__ import annotations

from enum import Enum


class SearchPathwayName(str, Enum):
    """
    Names for supported search pathways.

    These values help identify which pathway produced the result.
    """

    RAG = "rag"
    UNIFIED_SEARCH = "unified_search"
    FORCED_CHUNK_DEBUG = "forced_chunk_debug"
    KEYWORD_SEARCH = "keyword_search"
    VECTOR_SEARCH = "vector_search"
    HYBRID_SEARCH = "hybrid_search"
    PAYLOAD_PROJECTION = "payload_projection"


class SearchAuditStageName(str, Enum):
    """
    Common stages inside a search pathway.
    """

    NORMALIZE_QUESTION = "normalize_question"
    CLASSIFY_INTENT = "classify_intent"
    GENERATE_EMBEDDING = "generate_embedding"
    RETRIEVE_CANDIDATES = "retrieve_candidates"
    VECTOR_RETRIEVE_CHUNKS = "vector_retrieve_chunks"
    KEYWORD_RETRIEVE_CHUNKS = "keyword_retrieve_chunks"
    RERANK_CHUNKS = "rerank_chunks"
    BUILD_CONTEXT = "build_context"
    GENERATE_ANSWER = "generate_answer"
    RESOLVE_RELATIONSHIPS = "resolve_relationships"
    BUILD_PAYLOAD = "build_payload"
    VALIDATE_PAYLOAD = "validate_payload"
    AI_REVIEW = "ai_review"


class SearchAuditItemType(str, Enum):
    """
    Types of items that may be returned by search pathways.
    """

    CHUNK = "chunk"
    DOCUMENT = "document"
    COMPLETE_DOCUMENT = "complete_document"
    IMAGE = "image"
    DRAWING = "drawing"
    PART = "part"
    POSITION = "position"
    PROBLEM = "problem"
    SOLUTION = "solution"
    TASK = "task"


class SearchAuditStatus(str, Enum):
    """
    General audit status values.
    """

    STARTED = "started"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


class SearchAuditValidationStatus(str, Enum):
    """
    Validation result values.
    """

    NOT_VALIDATED = "not_validated"
    PASSED = "passed"
    WARNING = "warning"
    FAILED = "failed"
