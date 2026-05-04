from __future__ import annotations

from collections import OrderedDict
from typing import Any, Dict, Optional

from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session

from modules.configuration.config_env import get_db_config
from modules.configuration.log_config import logger, with_request_id
from modules.services.part_service import PartService
from modules.services.parts_position_image_service import PartsPositionImageService


class BillOfMaterialsSearchOrchestrator:
    """
    Orchestrator for BOM search workflow.

    RESPONSIBILITIES:
    - Own session lifecycle
    - Use existing services for lookup/query work
    - Assemble response-safe payloads
    """

    def __init__(
        self,
        part_service: Optional[PartService] = None,
        parts_position_image_service: Optional[PartsPositionImageService] = None,
        db_config=None,
    ) -> None:
        self.part_service = part_service or PartService()
        self.parts_position_image_service = (
            parts_position_image_service or PartsPositionImageService()
        )

        # ✅ FIX — use shared singleton config
        self.db_config = db_config or get_db_config()

    @with_request_id
    def search_bill_of_materials(
        self,
        data: Dict[str, Any],
    ) -> Dict[str, Any]:
        logger.info("BillOfMaterialsSearchOrchestrator.search_bill_of_materials started")

        session: Session = self.db_config.get_main_session()

        try:
            position_ids = self.parts_position_image_service.get_corresponding_position_ids(
                area_id=data.get("area_id"),
                equipment_group_id=data.get("equipment_group_id"),
                model_id=data.get("model_id"),
                asset_number_id=data.get("asset_number_id"),
                location_id=data.get("location_id"),
                session=session,
            )

            logger.info("Resolved corresponding position IDs | count=%s", len(position_ids))

            if not position_ids:
                return {
                    "success": False,
                    "message": "No positions found matching the search criteria.",
                    "template_name": "bill_of_materials/partials/bom_search_message.html",
                    "context": {
                        "message": "No positions found matching the search criteria.",
                    },
                    "status_code": 200,
                }

            parts = self.part_service.get_parts_for_positions(
                position_ids=position_ids,
                session=session,
            )

            logger.info("Resolved parts for positions | count=%s", len(parts))

            if not parts:
                return {
                    "success": False,
                    "message": "No parts found for the selected positions.",
                    "template_name": "bill_of_materials/partials/bom_search_message.html",
                    "context": {
                        "message": "No parts found for the selected positions.",
                    },
                    "status_code": 200,
                }

            unique_parts = OrderedDict()
            for part in parts:
                if part and getattr(part, "id", None) not in unique_parts:
                    unique_parts[part.id] = part

            all_parts = list(unique_parts.values())

            index = max(data.get("index", 0) or 0, 0)
            per_page = max(data.get("per_page", 25) or 25, 1)
            total_count = len(all_parts)
            page_parts = all_parts[index:index + per_page]

            return {
                "success": True,
                "message": "BOM search completed successfully.",
                "template_name": "parts_results.html",
                "context": {
                    "parts": page_parts,
                    "results_count": len(page_parts),
                    "total_count": total_count,
                    "index": index,
                    "per_page": per_page,
                    "has_prev": index > 0,
                    "has_next": index + per_page < total_count,
                    "prev_index": max(index - per_page, 0),
                    "next_index": index + per_page,
                    "position_ids": position_ids,
                },
                "status_code": 200,
            }

        except SQLAlchemyError as exc:
            logger.error(
                "SQLAlchemy error during BOM search | error=%s",
                exc,
                exc_info=True,
            )
            return {
                "success": False,
                "message": "A database error occurred while searching bill of materials.",
                "template_name": "bill_of_materials/partials/bom_search_message.html",
                "context": {
                    "message": "A database error occurred while searching bill of materials.",
                },
                "status_code": 500,
            }

        except Exception as exc:
            logger.error(
                "Unexpected error during BOM search | error=%s",
                exc,
                exc_info=True,
            )
            return {
                "success": False,
                "message": "An unexpected error occurred while searching bill of materials.",
                "template_name": "bill_of_materials/partials/bom_search_message.html",
                "context": {
                    "message": "An unexpected error occurred while searching bill of materials.",
                },
                "status_code": 500,
            }

        finally:
            logger.debug("Closing DB session in search orchestrator")
            try:
                session.close()
            except Exception:
                logger.exception("Failed closing DB session")