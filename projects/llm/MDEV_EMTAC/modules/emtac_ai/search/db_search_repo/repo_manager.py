# modules/search/db_search_repo/repo_manager.py
from __future__ import annotations

from contextlib import contextmanager
from typing import Optional

from modules.configuration.config_env import DatabaseConfig
from modules.configuration.log_config import (
    with_request_id, get_request_id, debug_id, info_id
)
from .position_repository import PositionRepository
from .part_repository import PartRepository
from .drawing_repository import DrawingRepository
from .image_repository import ImageRepository
from .complete_document_repository import CompleteDocumentRepository  # <-- FIXED


class REPOManager:
    """
    Central factory and lifecycle manager for repositories.
    - Owns a single SQLAlchemy session (per instance/request).
    - Hands out repositories that share this session.
    """
    def __init__(self, session=None):
        self._db = DatabaseConfig()
        self._session = session or self._db.get_main_session()

        self._positions: Optional[PositionRepository] = None
        self._parts: Optional[PartRepository] = None
        self._drawings: Optional[DrawingRepository] = None
        self._images: Optional[ImageRepository] = None
        self._complete_docs: Optional[CompleteDocumentRepository] = None  # <-- FIXED

        rid = get_request_id()
        debug_id("REPOManager initialized", rid)

    # --- Accessors (lazily construct) ---
    @property
    def session(self):
        return self._session

    @property
    def positions(self) -> PositionRepository:
        if self._positions is None:
            self._positions = PositionRepository(session=self._session)
        return self._positions

    @property
    def parts(self) -> PartRepository:
        if self._parts is None:
            self._parts = PartRepository(session=self._session)
        return self._parts

    @property
    def drawings(self) -> DrawingRepository:
        if self._drawings is None:
            self._drawings = DrawingRepository(session=self._session)
        return self._drawings

    @property
    def images(self) -> ImageRepository:
        if self._images is None:
            self._images = ImageRepository(session=self._session)
        return self._images

    @property
    def complete_documents(self) -> CompleteDocumentRepository:  # <-- FIXED
        if self._complete_docs is None:
            self._complete_docs = CompleteDocumentRepository(session=self._session)
        return self._complete_docs

    # --- Lifecycle helpers ---
    def close(self):
        rid = get_request_id()
        if self._session:
            try:
                self._session.close()
                debug_id("REPOManager session closed", rid)
            finally:
                self._session = None

    @contextmanager
    def session_scope(self):
        rid = get_request_id()
        info_id("REPOManager.session_scope enter", rid)
        try:
            yield self
        except Exception:
            raise
        finally:
            info_id("REPOManager.session_scope exit", rid)
            self.close()
