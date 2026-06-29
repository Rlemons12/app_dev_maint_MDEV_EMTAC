# option_c_qna/qanda_db/__init__.py
"""
Q&A Database Package
====================

This package bundles:

    • ORM models
    • Database configuration
    • Q&A service layer
"""

# ---------------------------------------------------------
# ORM Models (exported from qa_db.py or models_qa_db.py)
# ---------------------------------------------------------
from .qa_db import (
    Document,
    Chunk,
    Question,
    Answer,
    LLMModel,
    Embedding,
    PipelineRun,           # <-- ADDED
    PipelineRunItem,       # <-- ADDED
    create_schema,
)

# ---------------------------------------------------------
# Database Configuration (PostgreSQL engines + sessions)
# ---------------------------------------------------------
from option_c_qna.configuration.pg_db_config import (
    emtac_engine,
    EMTACSessionLocal,
    train_engine,
    TrainSessionLocal,
    get_training_session,
    get_emtac_session,
)

# ---------------------------------------------------------
# Service Layer
# ---------------------------------------------------------
from .service_qa_db import (
    QADatabaseService,
    get_qa_service,
)

# ---------------------------------------------------------
# What gets imported when using: from option_c_qna.qanda_db import *
# ---------------------------------------------------------
__all__ = [
    # ORM models
    "Document",
    "Chunk",
    "Question",
    "Answer",
    "LLMModel",
    "Embedding",
    "PipelineRun",          # <-- ADDED
    "PipelineRunItem",      # <-- ADDED
    "create_schema",

    # DB configs
    "emtac_engine",
    "EMTACSessionLocal",
    "train_engine",
    "TrainSessionLocal",
    "get_training_session",
    "get_emtac_session",

    # Service
    "QADatabaseService",
    "get_qa_service",
]
