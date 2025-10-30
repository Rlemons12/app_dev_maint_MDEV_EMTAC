from __future__ import annotations
from typing import Optional

# Shared config/logging
from modules.configuration.config_env import DatabaseConfig
from modules.configuration.log_config import get_request_id,debug_id



class BaseRepository:
    """
    Shared session lifecycle for repositories.
    - If a session is provided, we reuse it and do NOT close it.
    - If not provided, we create a session and close it after each op.
    """
    def __init__(self, session=None):
        self._db = DatabaseConfig()
        self._external_session = session
        rid = get_request_id()
        debug_id(f"{self.__class__.__name__} initialized (owns_session={self._external_session is None})", rid)

    def _session(self):
        return self._external_session or self._db.get_main_session()

    def _owns_session(self) -> bool:
        return self._external_session is None
