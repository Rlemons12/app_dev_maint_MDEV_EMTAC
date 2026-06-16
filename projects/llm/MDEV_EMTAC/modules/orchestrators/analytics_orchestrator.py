# modules/orchestrators/analytics_orchestrator.py

from datetime import datetime, timedelta
from typing import Dict, Any
from sqlalchemy import func

from modules.configuration.log_config import (
    with_request_id,
    error_id,
)

from modules.orchestrators.base_orchestrator import BaseOrchestrator
from modules.emtacdb.emtacdb_fts import QandA


class AnalyticsOrchestrator(BaseOrchestrator):
    """
    Read-only analytics aggregation layer.

    Responsibilities:
        - Aggregate chatbot metrics
        - Compute performance windows
        - Never mutate data
        - Own transaction scope
    """

    def __init__(self):
        super().__init__()

    # ---------------------------------------------------------
    # Public Entry
    # ---------------------------------------------------------

    @with_request_id
    def get_metrics(
        self,
        *,
        hours: int = 24,
        request_id: str | None = None,
    ) -> Dict[str, Any]:

        cutoff = datetime.utcnow() - timedelta(hours=hours)

        metrics = {
            "status": "ok",
            "window_hours": hours,
            "metrics": {},
        }

        try:
            with self.transaction() as session:

                total_questions = (
                    session.query(func.count(QandA.id))
                    .filter(QandA.created_at >= cutoff)
                    .scalar()
                )

                successful = (
                    session.query(func.count(QandA.id))
                    .filter(
                        QandA.created_at >= cutoff,
                        QandA.answer.isnot(None),
                    )
                    .scalar()
                )

                avg_length = (
                    session.query(func.avg(func.length(QandA.answer)))
                    .filter(QandA.created_at >= cutoff)
                    .scalar()
                )

                metrics["metrics"] = {
                    "total_questions": total_questions or 0,
                    "successful_answers": successful or 0,
                    "avg_answer_length": float(avg_length or 0),
                }

        except Exception as e:
            error_id(f"Analytics failure: {e}", request_id, exc_info=True)
            return {
                "status": "error",
                "message": "Analytics unavailable",
            }

        return metrics

    # ---------------------------------------------------------
    # Performance Recommendations
    # ---------------------------------------------------------

    @with_request_id
    def get_recommendations(
        self,
        *,
        hours: int = 24,
        request_id: str | None = None,
    ) -> Dict[str, Any]:

        metrics = self.get_metrics(hours=hours, request_id=request_id)

        if metrics.get("status") != "ok":
            return metrics

        recommendations = []

        total = metrics["metrics"]["total_questions"]
        avg_length = metrics["metrics"]["avg_answer_length"]

        if total > 1000:
            recommendations.append("Consider horizontal scaling.")

        if avg_length < 50:
            recommendations.append("Model responses may be too short.")

        return {
            "status": "ok",
            "hours": hours,
            "recommendations": recommendations,
        }

    # ---------------------------------------------------------
    # Dashboard Aggregation (Lightweight)
    # ---------------------------------------------------------

    @with_request_id
    def get_dashboard(
        self,
        *,
        hours: int = 24,
        request_id: str | None = None,
    ) -> Dict[str, Any]:

        metrics = self.get_metrics(hours=hours, request_id=request_id)

        if metrics.get("status") != "ok":
            return metrics

        return {
            "status": "ok",
            "hours": hours,
            "dashboard": {
                "summary": metrics["metrics"],
                "generated_at": datetime.utcnow().isoformat(),
            },
        }