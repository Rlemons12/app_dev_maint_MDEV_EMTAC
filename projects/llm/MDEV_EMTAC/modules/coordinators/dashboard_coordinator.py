# modules/application/dashboard_coordinator.py

from typing import Dict, Any

from modules.configuration.log_config import with_request_id
from modules.orchestrators.dashboard_orchestrator import DashboardOrchestrator


class DashboardCoordinator:

    def __init__(self):
        self.orchestrator = DashboardOrchestrator()

    @with_request_id
    def get_dashboard(
        self,
        *,
        hours: int,
        request_id: str = None,
    ) -> Dict[str, Any]:
        return self.orchestrator.build_dashboard(
            hours=hours,
            request_id=request_id,
        )