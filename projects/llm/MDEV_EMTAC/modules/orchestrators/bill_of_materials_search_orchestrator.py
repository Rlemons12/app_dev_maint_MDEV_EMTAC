from __future__ import annotations

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

    PERFORMANCE NOTES:
    - Position matching is delegated to PartsPositionImageService.
    - Paged/distinct part lookup is delegated to PartService.
    - This orchestrator does not deduplicate parts in Python.
    - This orchestrator does not paginate after loading all rows.
    """

    DEFAULT_INDEX = 0
    DEFAULT_PER_PAGE = 25
    MAX_PER_PAGE = 100

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

        # Use shared singleton config.
        self.db_config = db_config or get_db_config()

    @with_request_id
    def search_bill_of_materials(
        self,
        data: Dict[str, Any],
    ) -> Dict[str, Any]:
        logger.info("BillOfMaterialsSearchOrchestrator.search_bill_of_materials started")

        session: Session = self.db_config.get_main_session()

        try:
            index = self._normalize_index(data.get("index"))
            per_page = self._normalize_per_page(data.get("per_page"))

            logger.info(
                "BOM search pagination normalized | index=%s | per_page=%s",
                index,
                per_page,
            )

            position_ids = self.parts_position_image_service.get_corresponding_position_ids(
                area_id=data.get("area_id"),
                equipment_group_id=data.get("equipment_group_id"),
                model_id=data.get("model_id"),
                asset_number_id=data.get("asset_number_id"),
                location_id=data.get("location_id"),
                session=session,
            )

            position_count = len(position_ids) if position_ids else 0

            logger.info(
                "Resolved corresponding position IDs | count=%s",
                position_count,
            )

            if not position_ids:
                return self._message_response(
                    message="No positions found matching the search criteria.",
                    status_code=200,
                )

            search_result = self.part_service.search_distinct_parts_for_positions(
                position_ids=position_ids,
                session=session,
                index=index,
                per_page=per_page,
            )

            page_parts = search_result.get("parts", [])
            total_count = search_result.get("total_count", 0)

            logger.info(
                "Resolved paged BOM parts | position_count=%s | page_count=%s | "
                "total_count=%s | index=%s | per_page=%s",
                position_count,
                len(page_parts),
                total_count,
                index,
                per_page,
            )

            if not page_parts:
                return self._message_response(
                    message="No parts found for the selected positions.",
                    status_code=200,
                )

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
            return self._message_response(
                message="A database error occurred while searching bill of materials.",
                status_code=500,
            )

        except Exception as exc:
            logger.error(
                "Unexpected error during BOM search | error=%s",
                exc,
                exc_info=True,
            )
            return self._message_response(
                message="An unexpected error occurred while searching bill of materials.",
                status_code=500,
            )

        finally:
            logger.debug("Closing DB session in search orchestrator")
            try:
                session.close()
            except Exception:
                logger.exception("Failed closing DB session")

    @classmethod
    def _normalize_index(cls, value: Any) -> int:
        try:
            index = int(value)
        except (TypeError, ValueError):
            return cls.DEFAULT_INDEX

        return max(index, 0)

    @classmethod
    def _normalize_per_page(cls, value: Any) -> int:
        try:
            per_page = int(value)
        except (TypeError, ValueError):
            return cls.DEFAULT_PER_PAGE

        per_page = max(per_page, 1)
        per_page = min(per_page, cls.MAX_PER_PAGE)

        return per_page

    @staticmethod
    def _message_response(
        *,
        message: str,
        status_code: int,
    ) -> Dict[str, Any]:
        return {
            "success": False,
            "message": message,
            "template_name": "bill_of_materials/partials/bom_search_message.html",
            "context": {
                "message": message,
            },
            "status_code": status_code,
        }