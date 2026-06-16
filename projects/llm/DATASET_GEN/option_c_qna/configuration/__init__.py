"""
Configuration package for the Option C Q&A pipeline.

Exports:
    cfg               – Global configuration loaded from the master .env
    get_qna_logger    – Unified logger for all pipeline modules
    LOG_DIR           – Path to log output directory
"""

from .config import cfg
from .logging_config import get_qna_logger, LOG_DIR
from .pg_db_config import *
__all__ = [
    "cfg",
    "get_qna_logger",
    "LOG_DIR",
]
