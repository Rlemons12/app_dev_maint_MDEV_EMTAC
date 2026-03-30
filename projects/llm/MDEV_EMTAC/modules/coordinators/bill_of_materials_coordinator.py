from __future__ import annotations

from typing import Any, Dict, Optional

from modules.configuration.log_config import logger, with_request_id
from modules.orchestrators.position_orchestrator import PositionOrchestrator


class BillOfMaterialsCoordinator:
    """
    Coordinator layer for BOM workflows.

    Rules:
    - No session ownership
    - No direct DB access
    - No commit / rollback
    - Normalize inputs and call orchestrator
    """

    def __init__(
        self,
        position_orchestrator: Optional[PositionOrchestrator] = None,
    ) -> None:
        self.position_orchestrator = (
            position_orchestrator or PositionOrchestrator()
        )

    @with_request_id
    def get_bom_lookup_data(self) -> Dict[str, Any]:
        logger.info("BillOfMaterialsCoordinator.get_bom_lookup_data called")

        result = self.position_orchestrator.get_position_lookup_data()

        raw_data = result.get("data") or {}

        response = {
            "areas": raw_data.get("areas", []),
            "equipment_groups": raw_data.get("equipment_groups", []),
            "models": raw_data.get("models", []),
            "asset_numbers": raw_data.get("asset_numbers", []),
            "locations": raw_data.get("locations", []),
            "status_code": result.get("status_code", 200),
        }

        logger.info(
            "BOM lookup data normalized for legacy JSON contract | "
            "areas=%d equipment_groups=%d models=%d asset_numbers=%d locations=%d",
            len(response["areas"]),
            len(response["equipment_groups"]),
            len(response["models"]),
            len(response["asset_numbers"]),
            len(response["locations"]),
        )

        return response

    @with_request_id
    def get_parts_position_data(self, args: dict) -> Dict[str, Any]:
        logger.info("BillOfMaterialsCoordinator.get_parts_position_data called")

        result = self.position_orchestrator.get_position_lookup_data()

        raw_data = result.get("data") or {}

        return {
            "areas": raw_data.get("areas", []),
            "equipment_groups": raw_data.get("equipment_groups", []),
            "models": raw_data.get("models", []),
            "asset_numbers": raw_data.get("asset_numbers", []),
            "locations": raw_data.get("locations", []),
            "status_code": result.get("status_code", 200),
        }