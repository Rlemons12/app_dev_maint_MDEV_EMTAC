# modules/application/health_coordinator.py

from typing import Dict, Any
from modules.configuration.log_config import with_request_id
from modules.orchestrators.health_orchestrator import HealthOrchestrator


class HealthCoordinator:

    def __init__(self):
        self.orchestrator = HealthOrchestrator()

    @with_request_id
    def check_health(self, *, request_id: str = None) -> Dict[str, Any]:
        return self.orchestrator.perform_health_check(request_id=request_id)