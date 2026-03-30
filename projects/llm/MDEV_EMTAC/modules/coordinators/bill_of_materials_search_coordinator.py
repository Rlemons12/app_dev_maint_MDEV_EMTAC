from __future__ import annotations

from typing import Any, Dict, Optional

from modules.configuration.log_config import logger, with_request_id
from modules.orchestrators.bill_of_materials_search_orchestrator import (
    BillOfMaterialsSearchOrchestrator,
)


class BillOfMaterialsSearchCoordinator:
    """
    Coordinator for BOM search workflow.

    RESPONSIBILITIES:
    - Normalize inbound request data
    - Keep request parsing out of orchestrator
    - Do light, non-transactional staging only
    """

    def __init__(
        self,
        orchestrator: Optional[BillOfMaterialsSearchOrchestrator] = None,
    ) -> None:
        self.orchestrator = orchestrator or BillOfMaterialsSearchOrchestrator()

    @with_request_id
    def search_bill_of_materials(
        self,
        *,
        form_data,
    ) -> Dict[str, Any]:
        logger.info("BillOfMaterialsSearchCoordinator.search_bill_of_materials called")

        normalized = {
            "area_id": self._normalize_int(form_data.get("area") or form_data.get("area_id")),
            "equipment_group_id": self._normalize_int(
                form_data.get("equipment_group") or form_data.get("equipment_group_id")
            ),
            "model_id": self._normalize_int(form_data.get("model") or form_data.get("model_id")),
            "asset_number_id": self._normalize_int(
                form_data.get("asset_number") or form_data.get("asset_number_id")
            ),
            "location_id": self._normalize_int(
                form_data.get("location") or form_data.get("location_id")
            ),
            "index": self._normalize_int(form_data.get("index"), default=0),
            "per_page": self._normalize_int(form_data.get("per_page"), default=25),
        }

        logger.info(
            "Normalized BOM search payload | area_id=%s | equipment_group_id=%s | "
            "model_id=%s | asset_number_id=%s | location_id=%s | index=%s | per_page=%s",
            normalized["area_id"],
            normalized["equipment_group_id"],
            normalized["model_id"],
            normalized["asset_number_id"],
            normalized["location_id"],
            normalized["index"],
            normalized["per_page"],
        )

        return self.orchestrator.search_bill_of_materials(normalized)

    @staticmethod
    def _normalize_int(value: Any, default: Optional[int] = None) -> Optional[int]:
        if value in (None, "", "None"):
            return default
        try:
            return int(value)
        except (TypeError, ValueError):
            return default