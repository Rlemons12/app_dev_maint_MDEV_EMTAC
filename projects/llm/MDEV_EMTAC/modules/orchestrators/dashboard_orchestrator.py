# modules/orchestrators/dashboard_orchestrator.py

from datetime import datetime
from typing import Dict, Any

from modules.configuration.log_config import with_request_id
from modules.orchestrators.base_orchestrator import BaseOrchestrator

from modules.orchestrators.health_orchestrator import HealthOrchestrator
from modules.orchestrators.analytics_orchestrator import AnalyticsOrchestrator


class DashboardOrchestrator(BaseOrchestrator):
    """
    High-level system dashboard aggregation layer.

    Responsibilities:
        - Combine health + analytics
        - Provide system summary snapshot
        - No domain logic
        - No direct DB queries
        - No persistence
    """

    def __init__(self):
        super().__init__()

        # Each orchestrator handles its own BaseOrchestrator initialization
        self.health_orchestrator = HealthOrchestrator()
        self.analytics_orchestrator = AnalyticsOrchestrator()

    # ---------------------------------------------------------
    # Public Entry
    # ---------------------------------------------------------

    @with_request_id
    def get_dashboard(
        self,
        *,
        hours: int = 24,
        request_id: str | None = None,
    ) -> Dict[str, Any]:

        # -----------------------------------------------------
        # 1. System Health
        # -----------------------------------------------------
        health = self.health_orchestrator.check_health(
            request_id=request_id
        )

        # -----------------------------------------------------
        # 2. Performance Metrics
        # -----------------------------------------------------
        metrics = self.analytics_orchestrator.get_metrics(
            hours=hours,
            request_id=request_id,
        )

        # -----------------------------------------------------
        # 3. Compose Dashboard Snapshot
        # -----------------------------------------------------
        dashboard = {
            "status": "ok",
            "hours": hours,
            "generated_at": datetime.utcnow().isoformat(),
            "health": health,
            "metrics": metrics.get("metrics")
            if metrics.get("status") == "ok"
            else None,
        }

        # Degrade overall status if any subsystem degraded
        if health.get("status") != "healthy":
            dashboard["status"] = "degraded"

        if metrics.get("status") != "ok":
            dashboard["status"] = "degraded"

        return dashboard