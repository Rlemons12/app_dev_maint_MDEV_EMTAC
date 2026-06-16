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
    - Apply lightweight request safety limits
    - Do light, non-transactional staging only

    PERFORMANCE NOTES:
    - This layer should not build SQL queries.
    - This layer should not own database sessions.
    - This layer can protect downstream layers from expensive request values,
      such as very large per_page values or invalid pagination indexes.
    """

    DEFAULT_INDEX = 0
    DEFAULT_PER_PAGE = 25
    MAX_PER_PAGE = 100

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
            "area_id": self._normalize_int(
                self._first_present(form_data, "area", "area_id"),
                default=None,
                min_value=1,
            ),
            "equipment_group_id": self._normalize_int(
                self._first_present(form_data, "equipment_group", "equipment_group_id"),
                default=None,
                min_value=1,
            ),
            "model_id": self._normalize_int(
                self._first_present(form_data, "model", "model_id"),
                default=None,
                min_value=1,
            ),
            "asset_number_id": self._normalize_int(
                self._first_present(form_data, "asset_number", "asset_number_id"),
                default=None,
                min_value=1,
            ),
            "location_id": self._normalize_int(
                self._first_present(form_data, "location", "location_id"),
                default=None,
                min_value=1,
            ),
            "index": self._normalize_int(
                form_data.get("index"),
                default=self.DEFAULT_INDEX,
                min_value=0,
            ),
            "per_page": self._normalize_int(
                form_data.get("per_page"),
                default=self.DEFAULT_PER_PAGE,
                min_value=1,
                max_value=self.MAX_PER_PAGE,
            ),
        }

        has_filter = any(
            normalized[key] is not None
            for key in (
                "area_id",
                "equipment_group_id",
                "model_id",
                "asset_number_id",
                "location_id",
            )
        )

        if not has_filter:
            logger.warning(
                "BOM search requested with no filters. "
                "This may become expensive if the orchestrator/service does not enforce "
                "safe default limits. index=%s | per_page=%s",
                normalized["index"],
                normalized["per_page"],
            )

        logger.info(
            "Normalized BOM search payload | area_id=%s | equipment_group_id=%s | "
            "model_id=%s | asset_number_id=%s | location_id=%s | index=%s | per_page=%s | "
            "has_filter=%s",
            normalized["area_id"],
            normalized["equipment_group_id"],
            normalized["model_id"],
            normalized["asset_number_id"],
            normalized["location_id"],
            normalized["index"],
            normalized["per_page"],
            has_filter,
        )

        return self.orchestrator.search_bill_of_materials(normalized)

    @staticmethod
    def _first_present(form_data: Any, *keys: str) -> Any:
        """
        Return the first meaningful value from form_data.

        This avoids using:

            form_data.get("area") or form_data.get("area_id")

        because Python's `or` treats values like 0, "", None, and False
        as missing. For request parsing, it is safer to explicitly define
        what counts as missing.
        """

        for key in keys:
            value = form_data.get(key)

            if value not in (None, "", "None"):
                return value

        return None

    @staticmethod
    def _normalize_int(
        value: Any,
        default: Optional[int] = None,
        min_value: Optional[int] = None,
        max_value: Optional[int] = None,
    ) -> Optional[int]:
        """
        Normalize request values into integers with optional bounds.

        Examples:
        - None, "", "None" -> default
        - invalid values -> default
        - values below min_value -> min_value
        - values above max_value -> max_value
        """

        if value in (None, "", "None"):
            return default

        try:
            normalized = int(value)
        except (TypeError, ValueError):
            return default

        if min_value is not None and normalized < min_value:
            return min_value

        if max_value is not None and normalized > max_value:
            return max_value

        return normalized