from __future__ import annotations

from typing import Any, Dict, Optional

from modules.configuration.log_config import logger, with_request_id
from modules.orchestrators.bill_of_materials_query_data_orchestrator import (
    BillOfMaterialsQueryDataOrchestrator,
)
from modules.orchestrators.bill_of_materials_search_orchestrator import (
    BillOfMaterialsSearchOrchestrator,
)
from modules.orchestrators.bill_of_materials_upload_orchestrator import (
    BillOfMaterialsUploadOrchestrator,
)
from modules.orchestrators.bill_of_materials_update_part_orchestrator import (
    BillOfMaterialsUpdatePartOrchestrator,
)


class BillOfMaterialsCoordinator:
    """
    Unified coordinator for Bill of Materials workflows.

    RESPONSIBILITIES:
    - Keep route layer thin
    - Normalize inbound request/form/query/file data
    - Build clean payloads for orchestrators
    - Delegate workflow execution to orchestrators
    - Return response-safe dictionaries

    HARD RULES:
    - no session creation
    - no session closing
    - no commit
    - no rollback

    NOTES:
    - This replaces separate BOM coordinator classes such as:
        * BillOfMaterialsQueryDataCoordinator
        * BillOfMaterialsSearchCoordinator
        * BillOfMaterialsUploadCoordinator
        * BillOfMaterialsUpdatePartCoordinator
    - Orchestrators remain responsible for workflow execution and transaction ownership.
    """

    def __init__(
        self,
        query_data_orchestrator: Optional[BillOfMaterialsQueryDataOrchestrator] = None,
        search_orchestrator: Optional[BillOfMaterialsSearchOrchestrator] = None,
        upload_orchestrator: Optional[BillOfMaterialsUploadOrchestrator] = None,
        update_part_orchestrator: Optional[BillOfMaterialsUpdatePartOrchestrator] = None,
    ) -> None:
        self.query_data_orchestrator = (
            query_data_orchestrator or BillOfMaterialsQueryDataOrchestrator()
        )
        self.search_orchestrator = (
            search_orchestrator or BillOfMaterialsSearchOrchestrator()
        )
        self.upload_orchestrator = (
            upload_orchestrator or BillOfMaterialsUploadOrchestrator()
        )
        self.update_part_orchestrator = (
            update_part_orchestrator or BillOfMaterialsUpdatePartOrchestrator()
        )

    # ------------------------------------------------------------------
    # SHARED HELPERS
    # ------------------------------------------------------------------

    @staticmethod
    def _normalize_int(value: Any, default: Optional[int] = None) -> Optional[int]:
        """
        Convert incoming form/query values to int where possible.

        Returns default for empty/invalid values.
        """
        if value in (None, "", "None", "__None"):
            return default

        try:
            return int(value)
        except (TypeError, ValueError):
            return default

    @staticmethod
    def _coerce_ajax_flag(request) -> bool:
        """
        Determine whether the incoming request should be treated as AJAX.
        """
        return (
            request.form.get("ajax") == "true"
            or request.args.get("ajax") == "true"
            or request.args.get("ajax") == "1"
            or request.headers.get("X-Requested-With") == "XMLHttpRequest"
        )

    @staticmethod
    def _extract_remove_image_ids(request) -> list[int]:
        """
        Extract image ids marked for removal from request.form.
        """
        raw_values = request.form.getlist("remove_image")
        image_ids: list[int] = []

        for raw in raw_values:
            try:
                image_ids.append(int(raw))
            except (ValueError, TypeError):
                continue

        return image_ids

    @staticmethod
    def _normalize_mapping(data: Optional[Any]) -> Dict[str, Any]:
        """
        Normalize MultiDict / dict / mapping-like inputs into a plain dictionary.

        This is useful for request.args or other mapping-like objects coming from Flask.
        """
        if not data:
            return {}

        try:
            return dict(data)
        except Exception:
            return {}

    # ------------------------------------------------------------------
    # QUERY LOOKUP DATA
    # ------------------------------------------------------------------

    @with_request_id
    def get_parts_position_data(
        self,
        *,
        args: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Normalize query-string args and delegate position lookup data retrieval.
        """
        normalized_args = self._normalize_mapping(args)
        logger.debug(
            "BillOfMaterialsCoordinator.get_parts_position_data args=%s",
            normalized_args,
        )

        return self.query_data_orchestrator.get_parts_position_data(
            args=normalized_args
        )

    @with_request_id
    def get_bom_list_data(
        self,
        *,
        args: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Normalize query-string args and delegate BOM dropdown/list lookup retrieval.
        """
        normalized_args = self._normalize_mapping(args)
        logger.debug(
            "BillOfMaterialsCoordinator.get_bom_list_data args=%s",
            normalized_args,
        )

        return self.query_data_orchestrator.get_bom_list_data(
            args=normalized_args
        )

    # ------------------------------------------------------------------
    # BOM SEARCH
    # ------------------------------------------------------------------

    @with_request_id
    def search_bill_of_materials(
        self,
        *,
        form_data,
    ) -> Dict[str, Any]:
        """
        Normalize BOM search inputs and delegate to search orchestrator.
        """
        logger.info(
            "BillOfMaterialsCoordinator.search_bill_of_materials called"
        )

        normalized = {
            "area_id": self._normalize_int(
                form_data.get("area") or form_data.get("area_id")
            ),
            "equipment_group_id": self._normalize_int(
                form_data.get("equipment_group") or form_data.get("equipment_group_id")
            ),
            "model_id": self._normalize_int(
                form_data.get("model") or form_data.get("model_id")
            ),
            "asset_number_id": self._normalize_int(
                form_data.get("asset_number") or form_data.get("asset_number_id")
            ),
            "location_id": self._normalize_int(
                form_data.get("location") or form_data.get("location_id")
            ),
            "index": self._normalize_int(
                form_data.get("index"),
                default=0,
            ),
            "per_page": self._normalize_int(
                form_data.get("per_page"),
                default=25,
            ),
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

        return self.search_orchestrator.search_bill_of_materials(normalized)

    # ------------------------------------------------------------------
    # BOM UPLOAD
    # ------------------------------------------------------------------

    @with_request_id
    def submit_bill_of_materials_upload(
        self,
        *,
        form_data,
        files,
    ) -> Dict[str, Any]:
        """
        Normalize BOM upload request data and delegate to upload orchestrator.
        """
        logger.info(
            "BillOfMaterialsCoordinator.submit_bill_of_materials_upload called"
        )

        normalized = {
            "image_path": form_data.get("image_path"),
            "area_id": self._normalize_int(
                form_data.get("area") or form_data.get("area_id")
            ),
            "equipment_group_id": self._normalize_int(
                form_data.get("equipment_group") or form_data.get("equipment_group_id")
            ),
            "model_id": self._normalize_int(
                form_data.get("model") or form_data.get("model_id")
            ),
            "asset_number_id": self._normalize_int(
                form_data.get("asset_number") or form_data.get("asset_number_id")
            ),
            "location_id": self._normalize_int(
                form_data.get("location") or form_data.get("location_id")
            ),
            "site_location_id": self._normalize_int(
                form_data.get("site_location") or form_data.get("site_location_id")
            ),
            "file": files.get("file"),
        }

        logger.info(
            "Normalized BOM upload payload | area_id=%s | equipment_group_id=%s | "
            "model_id=%s | asset_number_id=%s | location_id=%s | site_location_id=%s | "
            "has_file=%s | image_path=%s",
            normalized["area_id"],
            normalized["equipment_group_id"],
            normalized["model_id"],
            normalized["asset_number_id"],
            normalized["location_id"],
            normalized["site_location_id"],
            bool(normalized["file"]),
            normalized["image_path"],
        )

        return self.upload_orchestrator.submit_bill_of_materials_upload(normalized)

    # ------------------------------------------------------------------
    # UPDATE PART - PAYLOAD BUILDERS
    # ------------------------------------------------------------------

    @with_request_id
    def build_edit_view_request(self, request, part_id: int) -> Dict[str, Any]:
        """
        Build payload for edit-part page/view requests.
        """
        payload = {
            "part_id": part_id,
            "is_ajax": self._coerce_ajax_flag(request),
            "search_query": request.args.get("search_query", "").strip(),
        }
        logger.debug("Built edit view payload: %s", payload)
        return payload

    @with_request_id
    def build_search_request(self, request) -> Dict[str, Any]:
        """
        Build payload for standard part search requests.
        """
        payload = {
            "search_query": request.args.get("search_query", "").strip(),
            "is_ajax": self._coerce_ajax_flag(request),
            "limit": 10,
        }
        logger.debug("Built search payload: %s", payload)
        return payload

    @with_request_id
    def build_ajax_search_request(self, request) -> Dict[str, Any]:
        """
        Build payload for AJAX-only part search requests.
        """
        payload = {
            "search_query": request.args.get("search_query", "").strip(),
            "is_ajax": True,
            "limit": 10,
        }
        logger.debug("Built AJAX search payload: %s", payload)
        return payload

    @with_request_id
    def build_update_request(self, request, part_id: int) -> Dict[str, Any]:
        """
        Build payload for part update requests.
        """
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
            "position_id": self._normalize_int(request.form.get("position_id")),
            "remove_image_ids": self._extract_remove_image_ids(request),
            "search_query": request.form.get("search_query", "").strip(),
        }

        logger.debug(
            "Built update payload for part_id=%s | is_ajax=%s | position_id=%s | "
            "remove_image_ids=%s | has_upload=%s",
            part_id,
            payload["is_ajax"],
            payload["position_id"],
            payload["remove_image_ids"],
            bool(uploaded_file and getattr(uploaded_file, "filename", "")),
        )
        return payload

    @with_request_id
    def build_image_request(self, image_id: int) -> Dict[str, Any]:
        """
        Build payload for image/file fetch requests.
        """
        payload = {"image_id": image_id}
        logger.debug("Built image request payload: %s", payload)
        return payload

    # ------------------------------------------------------------------
    # UPDATE PART - ORCHESTRATOR DELEGATION
    # ------------------------------------------------------------------

    @with_request_id
    def get_edit_part_view_data(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """
        Retrieve edit-part page data.
        """
        return self.update_part_orchestrator.get_edit_part_view_data(
            part_id=payload["part_id"],
            search_query=payload.get("search_query", ""),
        )

    @with_request_id
    def search_parts(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """
        Search parts for edit/search workflows.
        """
        return self.update_part_orchestrator.search_parts(
            search_text=payload["search_query"],
            limit=payload.get("limit", 10),
            is_ajax=payload.get("is_ajax", False),
        )

    @with_request_id
    def update_part(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """
        Submit part update workflow to orchestrator.
        """
        return self.update_part_orchestrator.update_part(payload=payload)

    @with_request_id
    def get_part_image_file_data(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """
        Retrieve file data for a part image.
        """
        return self.update_part_orchestrator.get_part_image_file_data(
            image_id=payload["image_id"]
        )