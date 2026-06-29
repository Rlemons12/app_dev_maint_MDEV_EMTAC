# modules/emtac_ai/search/db_search_repo/__init__.py
from __future__ import annotations

from typing import Optional
from .repo_manager import REPOManager
from .aggregate_search import AggregateSearch
from modules.configuration.log_config import get_request_id, debug_id


class _DBSearchRepoFacade:
    """
    Singleton-style facade around all repositories + aggregate search.
    Usage:
        from modules.emtac_ai.search.db_search_repo import db_search_repo
        parts = db_search_repo.parts.search(search_text="valve")
    """
    _instance: Optional["_DBSearchRepoFacade"] = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        if not hasattr(self, "_initialized"):
            self._manager = REPOManager()
            # ✅ FIX: pass the REPOManager, not session
            self.aggregate_search = AggregateSearch(self._manager)
            self._initialized = True
            rid = get_request_id()
            debug_id("db_search_repo singleton initialized", rid)

    # --- Repo accessors (forward to REPOManager) ---
    @property
    def positions(self):
        return self._manager.positions

    @property
    def parts(self):
        return self._manager.parts

    @property
    def drawings(self):
        return self._manager.drawings

    @property
    def images(self):
        return self._manager.images

    @property
    def complete_documents(self):
        return self._manager.complete_documents

    # --- Session lifecycle ---
    def close(self):
        self._manager.close()


# Global singleton accessor
db_search_repo = _DBSearchRepoFacade()
