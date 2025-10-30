"""
emtac_ai package init.

This package unifies:
- Query expansion utilities + orchestrator
- Search backends + DB repositories
- AistManager orchestration
- Intent/NER plugins
- Response formatting
"""

__version__ = "0.3.0"

def get_version() -> str:
    return __version__

# --- Core orchestrator ---
from .aist_manager import AistManager, get_or_create_aist_manager

# --- Query expansion ---
from .query_expansion import (
    dedup_preserve_order,
    replace_word_boundary,
    tokenize_words_lower,
    SynonymLoader,
    QueryExpansionRAG,
    EMTACQueryExpansionOrchestrator,
)

# --- Search repositories ---
from .search.db_search_repo import (
    REPOManager,
    AggregateSearch,
)
# --- Search repositories ---
from .search.db_search_repo import REPOManager, AggregateSearch
from .search.db_search_repo.base_repository import BaseRepository
from .search.db_search_repo.part_repository import PartRepository
from .search.db_search_repo.image_repository import ImageRepository
from .search.db_search_repo.document_repository import DocumentRepository
from .search.db_search_repo.drawing_repository import DrawingRepository
from .search.db_search_repo.position_repository import PositionRepository
from .search.db_search_repo.complete_document_repository import CompleteDocumentRepository
from .search.db_search_repo.aggregate_search import PositionFilters, PartSearchParams


# --- NLP (Intent/NER) ---
from .emtac_intent_entity import IntentEntityPlugin

# --- Formatting ---
from .response_formatter import ResponseFormatter

__all__ = [
    "__version__",
    "get_version",
    # Orchestrator
    "AistManager",
    "get_or_create_aist_manager",
    # Query expansion
    "dedup_preserve_order",
    "replace_word_boundary",
    "tokenize_words_lower",
    "SynonymLoader",
    "QueryExpansionRAG",
    "EMTACQueryExpansionOrchestrator",
    # DB repositories
    "REPOManager",
    "AggregateSearch",
    "BaseRepository",
    "PartRepository",
    "ImageRepository",
    "DocumentRepository",
    "DrawingRepository",
    "PositionRepository",
    "CompleteDocumentRepository",
    "PositionFilters",
    "PartSearchParams",
    # NLP
    "IntentEntityPlugin",
    # Formatting
    "ResponseFormatter",
]
