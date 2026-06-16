from __future__ import annotations

from functools import lru_cache
from logging import LoggerAdapter

from modules.configuration.log_config import SearchAuditLogManager


@lru_cache(maxsize=1)
def get_search_audit_log_manager() -> SearchAuditLogManager:
    """
    Returns the global search audit log manager.

    This prevents duplicate logger handlers from being attached repeatedly.
    """

    return SearchAuditLogManager(
        run_name="global",
        to_console=False,
    )


def get_search_audit_logger() -> LoggerAdapter:
    """
    Returns the dedicated search audit logger adapter.
    """

    return get_search_audit_log_manager().logger
