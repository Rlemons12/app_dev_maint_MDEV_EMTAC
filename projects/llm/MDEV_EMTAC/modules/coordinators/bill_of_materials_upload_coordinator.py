from __future__ import annotations

from typing import Any, Dict, Optional

from modules.configuration.log_config import logger, with_request_id
from modules.orchestrators.bill_of_materials_upload_orchestrator import (
    BillOfMaterialsUploadOrchestrator,
)


class BillOfMaterialsUploadCoordinator:
    """
    Coordinator for BOM upload workflow.
    """

    def __init__(
        self,
        orchestrator: Optional[BillOfMaterialsUploadOrchestrator] = None,
    ) -> None:
        self.orchestrator = orchestrator or BillOfMaterialsUploadOrchestrator()

    @with_request_id
    def submit_bill_of_materials_upload(
        self,
        *,
        form_data,
        files,
    ) -> Dict[str, Any]:
        logger.info("BillOfMaterialsUploadCoordinator.submit_bill_of_materials_upload called")

        normalized = {
            "image_path": form_data.get("image_path"),
            "area_id": form_data.get("area"),
            "equipment_group_id": form_data.get("equipment_group"),
            "model_id": form_data.get("model"),
            "asset_number_id": form_data.get("asset_number"),
            "location_id": form_data.get("location"),
            "site_location_id": form_data.get("site_location"),
            "file": files.get("file"),
        }

        return self.orchestrator.submit_bill_of_materials_upload(normalized)