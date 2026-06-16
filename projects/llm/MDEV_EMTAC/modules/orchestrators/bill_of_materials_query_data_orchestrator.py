from __future__ import annotations

from typing import Any, Dict, Optional

from modules.configuration.log_config import logger, with_request_id
from modules.configuration.config_env import get_db_config
from modules.services.bill_of_materials_query_data_service import (
    BillOfMaterialsQueryDataService,
)


class BillOfMaterialsQueryDataOrchestrator:
    """
    Orchestrator for BOM query lookup data.

    Responsibilities:
    - own session lifecycle
    - call service methods
    - return response-safe dictionaries
    """

    def __init__(
        self,
        query_data_service: Optional[BillOfMaterialsQueryDataService] = None,
    ) -> None:
        self.query_data_service = query_data_service or BillOfMaterialsQueryDataService()

    @with_request_id
    def get_parts_position_data(
        self,
        *,
        args: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        logger.debug(
            "Orchestrator opened DB session for parts position data | args=%s",
            args,
        )

        db_config = get_db_config()

        with db_config.main_session() as session:
            data = self.query_data_service.get_parts_position_data(
                session=session,
                args=args or {},
            )

            # 🔥 RETURN RAW DATA (frontend expects this)
            return data

    @with_request_id
    def get_bom_list_data(
        self,
        *,
        args: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        logger.debug(
            "Orchestrator opened DB session for BOM list data | args=%s",
            args,
        )

        db_config = get_db_config()

        with db_config.main_session() as session:
            data = self.query_data_service.get_bom_list_data(
                session=session,
                args=args or {},
            )

            # 🔥 RETURN RAW DATA (frontend expects this)
            return data