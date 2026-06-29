"""
Query expansion package init.

This exposes:
- Core query expansion utilities
- Orchestrator & synonym loader
- Unified search hub
- DB search layer (AggregateSearch, REPOManager, PartRepository, etc.)
"""

# Core expansion utils & engine
from .query_utils import (
    dedup_preserve_order,
    replace_word_boundary,
    tokenize_words_lower,
)
from .synonym_loader import SynonymLoader
from .query_expansion_core import QueryExpansionRAG
from .orchestrator import EMTACQueryExpansionOrchestrator

# Unified search hub
from .UnifiedSearch import UnifiedSearch

# DB search layer
from .db_search_repo import (
    REPOManager,
    BaseRepository,
    PartRepository,
    ImageRepository,
    DocumentRepository,
    DrawingRepository,
    PositionRepository,
    CompleteDocumentRepository,
    AggregateSearch,
    PositionFilters,
    PartSearchParams,
)

__all__ = [
    # Core utils
    "dedup_preserve_order",
    "replace_word_boundary",
    "tokenize_words_lower",

    # Expansion engine
    "SynonymLoader",
    "QueryExpansionRAG",
    "EMTACQueryExpansionOrchestrator",

    # Unified search hub
    "UnifiedSearch",

    # DB search layer
    "REPOManager",
    "BaseRepository",
    "PartRepository",
    "ImageRepository",
    "DocumentRepository",
    "DrawingRepository",
    "PositionRepository",
    "CompleteDocumentRepository",
    "AggregateSearch",
    "PositionFilters",
    "PartSearchParams",
]
