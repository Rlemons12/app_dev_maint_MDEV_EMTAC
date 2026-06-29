from __future__ import annotations

import logging
import os
import sys
import threading
import uuid
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Optional


# ---------------------------------------------------------
# Filesystem paths
# ---------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parents[2]   # .../service_dashboard
LOG_DIR = BASE_DIR / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)

DASHBOARD_LOG_FILE = LOG_DIR / "service_dashboard.log"


# ---------------------------------------------------------
# Thread-local request ID
# ---------------------------------------------------------
_local = threading.local()


def get_request_id() -> str:
    if hasattr(_local, "request_id"):
        return _local.request_id
    rid = str(uuid.uuid4())[:8]
    _local.request_id = rid
    return rid


def set_request_id(request_id: Optional[str] = None) -> str:
    rid = request_id or str(uuid.uuid4())[:8]
    _local.request_id = rid
    return rid


def clear_request_id() -> None:
    if hasattr(_local, "request_id"):
        delattr(_local, "request_id")


# ---------------------------------------------------------
# Logger setup
# ---------------------------------------------------------
dashboard_logger = logging.getLogger("emtac_service_dashboard")

LOG_LEVEL = os.getenv("SERVICE_DASHBOARD_LOG_LEVEL", "INFO").upper()
dashboard_logger.setLevel(getattr(logging, LOG_LEVEL, logging.INFO))
dashboard_logger.propagate = False

if not dashboard_logger.handlers:
    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | DASHBOARD | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    file_handler = RotatingFileHandler(
        DASHBOARD_LOG_FILE,
        maxBytes=10 * 1024 * 1024,
        backupCount=5,
        encoding="utf-8",
    )
    file_handler.setLevel(getattr(logging, LOG_LEVEL, logging.INFO))
    file_handler.setFormatter(formatter)

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(getattr(logging, LOG_LEVEL, logging.INFO))
    console_handler.setFormatter(formatter)

    dashboard_logger.addHandler(file_handler)
    dashboard_logger.addHandler(console_handler)


# ---------------------------------------------------------
# Core logging helpers
# ---------------------------------------------------------
def dash_log(level: int, message: str, request_id: Optional[str] = None) -> None:
    rid = request_id or get_request_id()
    dashboard_logger.log(level, f"[REQ-{rid}] {message}")


def dash_debug(msg: str, request_id: Optional[str] = None) -> None:
    dash_log(logging.DEBUG, msg, request_id)


def dash_info(msg: str, request_id: Optional[str] = None) -> None:
    dash_log(logging.INFO, msg, request_id)


def dash_warning(msg: str, request_id: Optional[str] = None) -> None:
    dash_log(logging.WARNING, msg, request_id)


def dash_error(msg: str, request_id: Optional[str] = None) -> None:
    dash_log(logging.ERROR, msg, request_id)


def dash_critical(msg: str, request_id: Optional[str] = None) -> None:
    dash_log(logging.CRITICAL, msg, request_id)