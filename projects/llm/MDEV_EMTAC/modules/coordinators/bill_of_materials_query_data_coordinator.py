from __future__ import annotations

from typing import Any, Dict, Optional

from modules.configuration.log_config import logger, with_request_id
from modules.orchestrators.bill_of_materials_query_data_orchestrator import (
    BillOfMaterialsQueryDataOrchestrator,
)


class BillOfMaterialsQueryDataCoordinator:
    """
    Coordinator for BOM query lookup data.

    Responsibilities:
    - Keep route layer thin
    - Call orchestrator
    - Return response-safe dictionaries
    """

    def __init__(
        self,
        orchestrator: Optional[BillOfMaterialsQueryDataOrchestrator] = None,
    ) -> None:
        self.orchestrator = orchestrator or BillOfMaterialsQueryDataOrchestrator()

    @with_request_id
    def get_parts_position_data(self) -> Dict[str, Any]:
        logger.debug("Coordinator calling orchestrator for parts position data")
        return self.orchestrator.get_parts_position_data()