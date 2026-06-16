# modules/services/search_tracking_service.py

import time
from typing import Dict, Any, Optional
from sqlalchemy.orm import Session

from modules.configuration.log_config import debug_id, error_id
from modules.services.search_query_analytics_tracker import (
    SearchQueryAnalyticsTracker,
)


class SearchTrackingService:
    """
    Thin wrapper over SearchQueryAnalyticsTracker.

    Exists only for compatibility.
    No NLP tracking.
    No session creation.
    No start/finish lifecycle.
    """

    def __init__(self):
        self.analytics = SearchQueryAnalyticsTracker()

    def record(
        self,
        *,
        session: Session,
        user_id: str,
        question: str,
        ai_result: Dict[str, Any],
        duration_ms: int,
        request_id: Optional[str],
    ) -> None:

        try:
            self.analytics.record_query(
                session=session,
                session_id=None,
                query=question,
                user_id=user_id,
                result_count=len(ai_result.get("documents", [])),
                success=ai_result.get("strategy") != "error",
                method=ai_result.get("strategy", "rag"),
                duration_ms=duration_ms,
                request_id=request_id,
            )

            debug_id(
                f"[SearchTrackingService] Analytics recorded ({duration_ms}ms)",
                request_id,
            )

        except Exception as e:
            error_id(f"Tracking failed: {e}", request_id)
            # NEVER raise