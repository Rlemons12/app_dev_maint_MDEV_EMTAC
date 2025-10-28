from typing import Dict, Optional, List
import uuid
import logging

logger = logging.getLogger(__name__)


class SearchQueryTracker:
    """
    Tracks individual queries, intents, and clicked results.
    Current implementation is a safe stub (no DB writes).
    """

    def __init__(self, session_factory=None):
        self.session_factory = session_factory
        self._queries = []  # in-memory log for testing

    def start_session(self, user_id: str, context_data: Optional[Dict] = None) -> int:
        """
        Start a search session for a given user.
        Returns a synthetic session ID.
        """
        session_id = uuid.uuid4().int & (1 << 31) - 1
        logger.debug(f"[SearchQueryTracker] start_session user={user_id}, context={context_data}, id={session_id}")
        return session_id

    def record_query(self, session_id: int, query_text: str, intent: str, entities: List[str]):
        """
        Record a query made in a session.
        """
        record = {
            "session_id": session_id,
            "query_text": query_text,
            "intent": intent,
            "entities": entities,
        }
        self._queries.append(record)
        logger.debug(f"[SearchQueryTracker] record_query {record}")
        return True

    def record_click(self, session_id: int, query_id: int, result_id: str):
        """
        Record a click on a result for a given query.
        """
        record = {
            "session_id": session_id,
            "query_id": query_id,
            "result_id": result_id,
        }
        logger.debug(f"[SearchQueryTracker] record_click {record}")
        return True


class SearchSessionManager:
    """
    Manages search sessions for users.
    Current implementation is a safe stub (no DB writes).
    """

    def __init__(self, session_factory=None):
        self.session_factory = session_factory
        self._sessions = {}  # in-memory

    def start_session(self, user_id: str, context_data: Optional[Dict] = None) -> int:
        """
        Start and return a session ID.
        """
        session_id = uuid.uuid4().int & (1 << 31) - 1
        self._sessions[session_id] = {
            "user_id": user_id,
            "context_data": context_data or {},
        }
        logger.debug(f"[SearchSessionManager] start_session user={user_id}, id={session_id}, context={context_data}")
        return session_id

    def get_session(self, session_id: int) -> Optional[Dict]:
        """
        Retrieve session metadata.
        """
        return self._sessions.get(session_id)
