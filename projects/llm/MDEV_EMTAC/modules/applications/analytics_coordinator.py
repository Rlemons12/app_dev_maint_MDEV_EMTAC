# modules/application/analytics_coordinator.py

from typing import Dict, Any

from modules.configuration.log_config import with_request_id
from modules.orchestrators.analytics_orchestrator import AnalyticsOrchestrator


class AnalyticsCoordinator:
    """
    Application layer for analytics endpoints.
    """

    def __init__(self):
        self.orchestrator = AnalyticsOrchestrator()

    @with_request_id
    def get_metrics(self, *, request_id: str = None) -> Dict[str, Any]:
        return self.orchestrator.compute_metrics(request_id=request_id)

    @with_request_id
    def get_recommendations(
        self,
        *,
        hours: int,
        request_id: str = None,
    ) -> Dict[str, Any]:
        return self.orchestrator.compute_recommendations(
            hours=hours,
            request_id=request_id,
        )