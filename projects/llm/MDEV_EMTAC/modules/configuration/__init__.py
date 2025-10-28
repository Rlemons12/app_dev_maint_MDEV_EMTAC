"""
Configuration package initializer.

This file exposes the key configuration objects so they can be imported
directly from `modules.configuration`.
"""

# Import database base class
from .base import Base

# Import project-wide configuration variables and directory paths
from .config import (
    BASE_DIR,
    DATABASE_URL,
    ENABLE_REVISION_CONTROL,
    DATABASE_DIR,
    DATABASE_PATH,
    REVISION_CONTROL_DB_PATH,
    DATABASE_PATH_IMAGES_FOLDER,
    DB_LOADSHEET,
    DB_LOADSHEETS_BACKUP,
    DB_LOADSHEET_BOMS,
    BACKUP_DIR,
    directories_to_check,
    MODEL_DIRS,
    TRAIN_DATA_DIRS,
    PROJECT_DIRS,
    ALL_DIRS,
    OPENAI_MODEL_NAME,
    CURRENT_AI_MODEL,
    CURRENT_EMBEDDING_MODEL,
)

# Import environment-aware database configuration
from .config_env import DatabaseConfig

__all__ = [
    "Base",
    "BASE_DIR",
    "DATABASE_URL",
    "ENABLE_REVISION_CONTROL",
    "DATABASE_DIR",
    "DATABASE_PATH",
    "REVISION_CONTROL_DB_PATH",
    "DATABASE_PATH_IMAGES_FOLDER",
    "DB_LOADSHEET",
    "DB_LOADSHEETS_BACKUP",
    "DB_LOADSHEET_BOMS",
    "BACKUP_DIR",
    "directories_to_check",
    "MODEL_DIRS",
    "TRAIN_DATA_DIRS",
    "PROJECT_DIRS",
    "ALL_DIRS",
    "OPENAI_MODEL_NAME",
    "CURRENT_AI_MODEL",
    "CURRENT_EMBEDDING_MODEL",
    "DatabaseConfig",
]
