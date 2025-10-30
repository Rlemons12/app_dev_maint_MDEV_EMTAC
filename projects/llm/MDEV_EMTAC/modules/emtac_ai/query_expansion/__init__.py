# modules/emtac_ai/query_expansion/__init__.py

from .orchestrator import EMTACQueryExpansionOrchestrator
from .query_expansion_core import QueryExpansionRAG
from .synonym_loader import SynonymLoader
from .query_utils import (
    dedup_preserve_order,
    replace_word_boundary,
    tokenize_words_lower,
)

# --------------------------
# Singleton-style accessor
# --------------------------

class _QueryExpansionFacade:
    """
    Singleton-style accessor for query expansion.
    Provides one orchestrator instance + access to core classes/utilities.
    """

    _instance = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self, ai_model=None, intent_model_dir=None, ner_model_dir=None, domain="maintenance"):
        if not hasattr(self, "_initialized"):
            # One orchestrator instance for the whole app
            self.orch = EMTACQueryExpansionOrchestrator(
                ai_model=ai_model,
                intent_model_dir=intent_model_dir,
                ner_model_dir=ner_model_dir,
                domain=domain
            )
            self._initialized = True

    # Expose underlying classes/utilities
    @property
    def orchestrator(self) -> EMTACQueryExpansionOrchestrator:
        return self.orch

    @property
    def rag_engine(self) -> QueryExpansionRAG:
        return self.orch.engine  # direct access to expansion core

    @property
    def synonym_loader(self) -> SynonymLoader:
        return self.orch.engine.syn_loader


# Global singleton instance
query_expansion = _QueryExpansionFacade()

# --------------------------
# Explicit exports
# --------------------------

__all__ = [
    # High-level entrypoint
    "EMTACQueryExpansionOrchestrator",
    # Core expansion engine
    "QueryExpansionRAG",
    # Synonym/acronym loader
    "SynonymLoader",
    # Utility helpers
    "dedup_preserve_order",
    "replace_word_boundary",
    "tokenize_words_lower",
    # Singleton facade
    "query_expansion",
]
