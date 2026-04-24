from __future__ import annotations

from typing import Any, Dict, List, Optional

from modules.configuration.log_config import logger, with_request_id
from modules.emtacdb.emtacdb_fts import (
    Area,
    EquipmentGroup,
    Model,
    AssetNumber,
    Location,
)


class BillOfMaterialsQueryDataService:
    """
    Service for BOM query lookup data.

    HARD RULES:
    - No session creation
    - No session closing
    - No commit
    - No rollback
    """

    @staticmethod
    def _safe_name(value: Any) -> str:
        """
        Convert nullable text fields into a safe string for JSON responses.
        """
        if value is None:
            return ""
        return str(value)

    @staticmethod
    def _safe_int(value: Any) -> Optional[int]:
        """
        Normalize nullable foreign keys into either int or None.
        """
        if value is None:
            return None
        try:
            return int(value)
        except (TypeError, ValueError):
            return None

    @with_request_id
    def get_parts_position_data(
        self,
        *,
        session,
        args: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, List[Dict[str, Any]]]:
        """
        Return lookup data used by the BOM search / position selection UI.

        This method preserves the existing payload shape expected by the front end:
        - areas
        - equipment_groups
        - models
        - asset_numbers
        - locations
        """
        logger.debug(
            "BillOfMaterialsQueryDataService.get_parts_position_data called | args=%s",
            args,
        )

        logger.debug("Querying areas from database")
        areas = session.query(Area).all()
        logger.debug("Fetched %s areas", len(areas))

        logger.debug("Querying equipment groups from database")
        equipment_groups = session.query(EquipmentGroup).all()
        logger.debug("Fetched %s equipment groups", len(equipment_groups))

        logger.debug("Querying models from database")
        models = session.query(Model).all()
        logger.debug("Fetched %s models", len(models))

        logger.debug("Querying asset numbers from database")
        asset_numbers = session.query(AssetNumber).all()
        logger.debug("Fetched %s asset numbers", len(asset_numbers))

        logger.debug("Querying locations from database")
        locations = session.query(Location).all()
        logger.debug("Fetched %s locations", len(locations))

        data = {
            "areas": [
                {
                    "id": area.id,
                    "name": self._safe_name(area.name),
                }
                for area in areas
            ],
            "equipment_groups": [
                {
                    "id": group.id,
                    "name": self._safe_name(group.name),
                    "area_id": self._safe_int(getattr(group, "area_id", None)),
                }
                for group in equipment_groups
            ],
            "models": [
                {
                    "id": model.id,
                    "name": self._safe_name(model.name),
                    "equipment_group_id": self._safe_int(
                        getattr(model, "equipment_group_id", None)
                    ),
                }
                for model in models
            ],
            "asset_numbers": [
                {
                    "id": asset_number.id,
                    "number": self._safe_name(asset_number.number),
                    "model_id": self._safe_int(getattr(asset_number, "model_id", None)),
                }
                for asset_number in asset_numbers
            ],
            "locations": [
                {
                    "id": location.id,
                    "name": self._safe_name(location.name),
                    "model_id": self._safe_int(getattr(location, "model_id", None)),
                }
                for location in locations
            ],
        }

        logger.debug(
            "Returning parts position data | areas=%s | equipment_groups=%s | models=%s | asset_numbers=%s | locations=%s",
            len(data["areas"]),
            len(data["equipment_groups"]),
            len(data["models"]),
            len(data["asset_numbers"]),
            len(data["locations"]),
        )
        return data

    @with_request_id
    def get_bom_list_data(
        self,
        *,
        session,
        args: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, List[Dict[str, Any]]]:
        """
        Return lookup data for the BOM page dropdowns.

        Right now this intentionally matches the same payload structure as
        get_parts_position_data so existing BOM UI code can keep working.

        If later the BOM page needs a slightly different payload shape, this
        method can be customized without affecting get_parts_position_data.
        """
        logger.debug(
            "BillOfMaterialsQueryDataService.get_bom_list_data called | args=%s",
            args,
        )

        logger.debug(
            "get_bom_list_data currently reuses get_parts_position_data payload structure"
        )

        data = self.get_parts_position_data(
            session=session,
            args=args,
        )

        logger.debug(
            "Returning BOM list data | areas=%s | equipment_groups=%s | models=%s | asset_numbers=%s | locations=%s",
            len(data.get("areas", [])),
            len(data.get("equipment_groups", [])),
            len(data.get("models", [])),
            len(data.get("asset_numbers", [])),
            len(data.get("locations", [])),
        )
        return data