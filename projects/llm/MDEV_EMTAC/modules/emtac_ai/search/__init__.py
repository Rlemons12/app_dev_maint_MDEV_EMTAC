# modules/search/__init__.py

from .UnifiedSearch import UnifiedSearch
from . import db_search_repo  # exposes the whole facade + repos

__all__ = [
    "UnifiedSearch",
    "db_search_repo",
    "unified_search_with_tracking"
]
