# modules/services/search_query_analytics_tracker.py

from typing import Optional
from datetime import datetime
from sqlalchemy.orm import Session

from modules.configuration.log_config import debug_id, error_id
from typing import Optional
from datetime import datetime

from sqlalchemy.orm import Session

from modules.configuration.log_config import debug_id, error_id
from modules.emtacdb.emtacdb_fts import SearchQueryAnalytics


class SearchQueryAnalyticsTracker:
    """
    Chat-level analytics tracker.

    Responsibilities:
        - Persist chat performance analytics
        - Never commit
        - Never create sessions
        - Never raise fatal exceptions
        - Fully transaction-safe (orchestrator owns commit)

    This is NOT the NLP tracker.
    This is chat/system telemetry.
    """

    # ---------------------------------------------------------
    # Record Query Analytics
    # ---------------------------------------------------------

    def record_query(
        self,
        *,
        session: Session,
        session_id: Optional[int],
        query: str,
        user_id: str,
        result_count: int,
        success: bool,
        method: str,
        duration_ms: int,
        request_id: Optional[str],
    ) -> None:
        """
        Persist analytics record.

        Orchestrator owns transaction.
        This method never commits and never raises.
        """

        try:
            record = SearchQueryAnalytics(
                session_id=session_id,
                user_id=user_id,
                query=query,
                result_count=result_count,
                success=success,
                method=method,
                duration_ms=duration_ms,
                timestamp=datetime.utcnow(),
            )

            session.add(record)

            debug_id(
                f"[ChatAnalytics] "
                f"user={user_id} "
                f"method={method} "
                f"duration={duration_ms}ms "
                f"results={result_count} "
                f"success={success}",
                request_id,
            )

        except Exception as e:
            error_id(
                f"[ChatAnalytics] record_query failed: {e}",
                request_id,
            )
            # NEVER RAISE