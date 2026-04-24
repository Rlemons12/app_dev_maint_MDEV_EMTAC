from __future__ import annotations

from typing import Any, Dict, Optional

from modules.configuration.log_config import logger, with_request_id
from modules.orchestrators.bill_of_materials_update_part_orchestrator import (
    BillOfMaterialsUpdatePartOrchestrator,
)


class BillOfMaterialsUpdatePartCoordinator:
    """
    Coordinator for BOM Edit Part workflows.

    Responsibilities:
    - Normalize request data
    - Convert request/form/files into clean payloads
    - Call orchestrator
    - Return response-safe dictionaries
    """

    def __init__(
        self,
        orchestrator: Optional[BillOfMaterialsUpdatePartOrchestrator] = None,
    ) -> None:
        self.orchestrator = orchestrator or BillOfMaterialsUpdatePartOrchestrator()

    @staticmethod
    def _coerce_ajax_flag(request) -> bool:
        return (
            request.form.get("ajax") == "true"
            or request.args.get("ajax") == "true"
            or request.args.get("ajax") == "1"
            or request.headers.get("X-Requested-With") == "XMLHttpRequest"
        )

    @staticmethod
    def _normalize_position_id(raw_value: Optional[str]) -> Optional[int]:
        if raw_value is None:
            return None

        value = str(raw_value).strip()
        if not value or value in {"", "None", "__None"}:
            return None

        try:
            return int(value)
        except (ValueError, TypeError):
            return None

    @staticmethod
    def _extract_remove_image_ids(request) -> list[int]:
        raw_values = request.form.getlist("remove_image")
        image_ids: list[int] = []

        for raw in raw_values:
            try:
                image_ids.append(int(raw))
            except (ValueError, TypeError):
                continue

        return image_ids

    @with_request_id
    def build_edit_view_request(self, request, part_id: int) -> Dict[str, Any]:
        payload = {
            "part_id": part_id,
            "is_ajax": self._coerce_ajax_flag(request),
            "search_query": request.args.get("search_query", ""),
        }
        logger.debug("Built edit view payload: %s", payload)
        return payload

    @with_request_id
    def build_search_request(self, request) -> Dict[str, Any]:
        payload = {
            "search_query": request.args.get("search_query", "").strip(),
            "is_ajax": self._coerce_ajax_flag(request),
            "limit": 10,
        }
        logger.debug("Built search payload: %s", payload)
        return payload

    @with_request_id
    def build_ajax_search_request(self, request) -> Dict[str, Any]:
        payload = {
            "search_query": request.args.get("search_query", "").strip(),
            "is_ajax": True,
            "limit": 10,
        }
        logger.debug("Built AJAX search payload: %s", payload)
        return payload

    @with_request_id
    def build_update_request(self, request, part_id: int) -> Dict[str, Any]:
        uploaded_file = request.files.get("part_image")

        payload = {
            "part_id": part_id,
            "is_ajax": self._coerce_ajax_flag(request),
            "part_fields": {
                "part_number": request.form.get("part_number"),
                "name": request.form.get("name"),
                "oem_mfg": request.form.get("oem_mfg"),
                "model": request.form.get("model"),
                "class_flag": request.form.get("class_flag"),
                "ud6": request.form.get("ud6"),
                "type": request.form.get("type"),
                "notes": request.form.get("notes"),
                "documentation": request.form.get("documentation"),
            },
            "uploaded_file": uploaded_file,
            "image_title": request.form.get("image_title"),
            "image_description": request.form.get("image_description"),
            "position_id": self._normalize_position_id(request.form.get("position_id")),
            "remove_image_ids": self._extract_remove_image_ids(request),
            "search_query": request.form.get("search_query", "").strip(),
        }

        logger.debug(
            "Built update payload for part_id=%s: is_ajax=%s, position_id=%s, remove_image_ids=%s, has_upload=%s",
            part_id,
            payload["is_ajax"],
            payload["position_id"],
            payload["remove_image_ids"],
            bool(uploaded_file and getattr(uploaded_file, "filename", "")),
        )
        return payload

    @with_request_id
    def build_image_request(self, image_id: int) -> Dict[str, Any]:
        payload = {"image_id": image_id}
        logger.debug("Built image request payload: %s", payload)
        return payload

    @with_request_id
    def get_edit_part_view_data(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        return self.orchestrator.get_edit_part_view_data(
            part_id=payload["part_id"],
            search_query=payload.get("search_query", ""),
        )

    @with_request_id
    def search_parts(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        return self.orchestrator.search_parts(
            search_text=payload["search_query"],
            limit=payload.get("limit", 10),
            is_ajax=payload.get("is_ajax", False),
        )

    @with_request_id
    def update_part(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        return self.orchestrator.update_part(payload=payload)

    @with_request_id
    def get_part_image_file_data(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        return self.orchestrator.get_part_image_file_data(image_id=payload["image_id"])