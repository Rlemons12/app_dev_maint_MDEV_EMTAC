# modules/emtacdb/AI_Steward/__init__.py
"""
AI Steward package for EMTAC application.

This package provides intelligent question answering capabilities
using multiple search strategies including keyword search, full-text search,
and vector similarity search.
"""

from .aist import AistManager


def __init__(self, ai_model=None, db_session=None):
    """Initialize with optional AI model and database session."""
    self.ai_model = ai_model
    self.db_session = db_session
    self.start_time = None
    self.db_config = DatabaseConfig()

    # Initialize vector search client
    try:
        self.vector_search_client = VectorSearchClient()
        logger.debug("Vector search client initialized successfully")
    except Exception as e:
        logger.warning(f"Failed to initialize vector search client: {e}")
        self.vector_search_client = None

    logger.debug("AistManager initialized")

__version__ = "1.0.0"
__all__ = ['AistManager']