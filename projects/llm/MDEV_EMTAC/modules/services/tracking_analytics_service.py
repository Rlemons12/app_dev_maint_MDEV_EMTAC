# modules/services/tracking_analytics_service.py

from typing import Dict, Any
from sqlalchemy.orm import Session
from datetime import datetime, timedelta

from modules.configuration.log_config import debug_id
from modules.services.search_query_analytics_tracker import SearchQueryTracker


class TrackingAnalyticsService:
    """
    Pure domain analytics service.

    HARD RULES:
        - Never open sessions
        - Never commit
        - Never rollback
        - Only read data
    """

    def get_analytics(
        self,
        *,
        session: Session,
        hours: int,
        request_id: str = None,
    ) -> Dict[str, Any]:

        since = datetime.utcnow() - timedelta(hours=hours)

        queries = (
            session.query(SearchQueryTracker)
            .filter(SearchQueryTracker.timestamp >= since)
            .all()
        )

        if not queries:
            return {
                "total_requests": 0,
                "avg_response_time": 0,
                "avg_performance_score": 0,
                "step_performance": {},
                "method_performance": {},
                "performance_distribution": {},
            }

        total_requests = len(queries)

        total_time = sum(q.total_time or 0 for q in queries)
        avg_response_time = total_time / total_requests

        # Method performance
        method_perf = {}
        for q in queries:
            method = q.method or "unknown"
            if method not in method_perf:
                method_perf[method] = {"count": 0, "total_time": 0}
            method_perf[method]["count"] += 1
            method_perf[method]["total_time"] += q.total_time or 0

        for method, stats in method_perf.items():
            stats["avg_time"] = stats["total_time"] / stats["count"]

        # Performance distribution
        distribution = {
            "excellent": 0,
            "good": 0,
            "poor": 0,
            "very_poor": 0,
        }

        for q in queries:
            t = q.total_time or 0
            if t < 1.0:
                distribution["excellent"] += 1
            elif t < 2.5:
                distribution["good"] += 1
            elif t < 4.0:
                distribution["poor"] += 1
            else:
                distribution["very_poor"] += 1

        debug_id(f"Tracking analytics computed for {total_requests} records", request_id)

        return {
            "total_requests": total_requests,
            "avg_response_time": avg_response_time,
            "avg_performance_score": max(0, 100 - (avg_response_time * 10)),
            "method_performance": method_perf,
            "performance_distribution": distribution,
            "step_performance": {},  # Expand later if needed
        }