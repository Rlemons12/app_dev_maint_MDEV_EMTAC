from __future__ import annotations

import logging
from typing import Any, Dict


LOGGER_NAME = "postgres_server"


def configure_logging(level: int = logging.INFO) -> None:
    """
    Configure standard logging for the postgres_server service.
    Safe to call from launcher startup.
    """
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


def get_logger(name: str = LOGGER_NAME) -> logging.Logger:
    """
    Return a named logger for the postgres_server service.
    """
    return logging.getLogger(name)


def mask_sensitive_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Return a copy of config with sensitive values masked.
    """
    safe = dict(config)

    for key in ("password", "database_url"):
        if key in safe and safe[key]:
            if key == "database_url":
                safe[key] = _mask_database_url(str(safe[key]))
            else:
                safe[key] = "********"

    return safe


def _mask_database_url(url: str) -> str:
    """
    Mask passwords in database URLs like:
    postgresql+psycopg2://user:password@host:5432/db
    """
    if not url or "://" not in url or "@" not in url:
        return url

    try:
        scheme, rest = url.split("://", 1)
        creds, host_part = rest.split("@", 1)

        if ":" in creds:
            user, _password = creds.split(":", 1)
            creds = f"{user}:********"

        return f"{scheme}://{creds}@{host_part}"
    except Exception:
        return url