# modules/emtac_ai/query_expansion/db_search_repo/aggregate_search.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, List, Optional
from .repo_manager import REPOManager
from .part_repository import PartRepository
from .position_repository import PositionRepository
from .document_repository import DocumentRepository
from .image_repository import ImageRepository
from .drawing_repository import DrawingRepository
from .complete_document_repository import CompleteDocumentRepository

# Shared logging/config
from modules.configuration.log_config import (
    info_id, debug_id, warning_id, error_id, with_request_id, get_request_id,log_timed_operation)


# -------------------- DTOs --------------------

@dataclass
class PositionFilters:
    area_id: Optional[int] = None
    equipment_group_id: Optional[int] = None
    model_id: Optional[int] = None
    asset_number_id: Optional[int] = None
    location_id: Optional[int] = None
    subassembly_id: Optional[int] = None
    component_assembly_id: Optional[int] = None
    assembly_view_id: Optional[int] = None
    site_location_id: Optional[int] = None


@dataclass
class PartSearchParams:
    search_text: Optional[str] = None
    fields: Optional[List[str]] = None
    exact_match: bool = False
    use_fts: bool = True
    part_id: Optional[int] = None
    part_number: Optional[str] = None
    name: Optional[str] = None
    oem_mfg: Optional[str] = None
    model: Optional[str] = None
    class_flag: Optional[str] = None
    ud6: Optional[str] = None
    type_: Optional[str] = None
    notes: Optional[str] = None
    documentation: Optional[str] = None
    limit: int = 100


# -------------------- Orchestrator --------------------

class AggregateSearch:
    """
    Cross-entity orchestration that composes repositories via REPOManager.
    This class no longer manages SQLAlchemy sessions nor imports ORM models directly.
    """

    def __init__(self, repo_manager: REPOManager = None):
        # Default to creating a new REPOManager if not provided
        self.repos = repo_manager or REPOManager()

    # ---------- Position convenience wrappers ----------

    @with_request_id
    def search_positions(self, filters: PositionFilters, request_id: Optional[str] = None) -> List[int]:
        rid = request_id or get_request_id()
        debug_id(f"[AggregateSearch.search_positions] filters={filters}", rid)
        with log_timed_operation("AggregateSearch.search_positions", rid):
            return self.repos.positions.search_position_ids(
                area_id=filters.area_id,
                equipment_group_id=filters.equipment_group_id,
                model_id=filters.model_id,
                asset_number_id=filters.asset_number_id,
                location_id=filters.location_id,
                request_id=rid,
            )

    @with_request_id
    def ensure_position(self, filters: PositionFilters, request_id: Optional[str] = None) -> int:
        rid = request_id or get_request_id()
        debug_id(f"[AggregateSearch.ensure_position] filters={filters}", rid)
        with log_timed_operation("AggregateSearch.ensure_position", rid):
            return self.repos.positions.add(
                area_id=filters.area_id,
                equipment_group_id=filters.equipment_group_id,
                model_id=filters.model_id,
                asset_number_id=filters.asset_number_id,
                location_id=filters.location_id,
                subassembly_id=filters.subassembly_id,
                component_assembly_id=filters.component_assembly_id,
                assembly_view_id=filters.assembly_view_id,
                site_location_id=filters.site_location_id,
                request_id=rid,
            )

    def children_of(self, parent_type: str, parent_id: int, child_type: Optional[str] = None,
                    request_id: Optional[str] = None) -> List[Any]:
        return self.repos.positions.children_of(parent_type, parent_id, child_type, request_id=request_id)

    def next_level_type(self, current_level: str) -> Optional[str]:
        return self.repos.positions.next_level_type(current_level)

    # ---------- Part convenience wrappers ----------

    @with_request_id
    def search_parts(self, params: PartSearchParams, request_id: Optional[str] = None):
        rid = request_id or get_request_id()
        debug_id(f"[AggregateSearch.search_parts] params={params}", rid)
        with log_timed_operation("AggregateSearch.search_parts", rid):
            return self.repos.parts.search(
                search_text=params.search_text,
                fields=params.fields,
                exact_match=params.exact_match,
                use_fts=params.use_fts,
                part_id=params.part_id,
                part_number=params.part_number,
                name=params.name,
                oem_mfg=params.oem_mfg,
                model=params.model,
                class_flag=params.class_flag,
                ud6=params.ud6,
                type_=params.type_,
                notes=params.notes,
                documentation=params.documentation,
                limit=params.limit,
                request_id=rid,
            )

    # ---------- Aggregate (cross-entity) use cases ----------

    @with_request_id
    def parts_for_position_filters(
        self,
        pos_filters: PositionFilters,
        part_text: Optional[str] = None,
        limit: int = 100,
        request_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        End-to-end example:
          1) Find positions via filters
          2) Pull only parts actually linked to those positions (via PartsPositionImageAssociation)
          3) Optionally, further filter by a free-text (FTS) query if provided
        """
        rid = request_id or get_request_id()
        with log_timed_operation("AggregateSearch.parts_for_position_filters", rid):
            # 1) positions
            pos_ids = self.search_positions(pos_filters, request_id=rid)
            info_id(f"[AggregateSearch] positions matched: {len(pos_ids)}", rid)

            if not pos_ids:
                return {"position_ids": [], "parts": []}

            # 2) parts linked to those positions
            linked_parts = self.repos.parts.parts_by_position_ids(pos_ids, limit=limit, request_id=rid)

            # 3) optional additional text filter (use FTS for better ranking)
            if part_text:
                # We do a separate FTS search and intersect on part IDs to keep relevance
                fts_results = self.repos.parts.search(
                    search_text=part_text, use_fts=True, limit=limit, request_id=rid
                )
                fts_ids = {p.id for p in fts_results}
                linked_parts = [p for p in linked_parts if p.id in fts_ids]

            return {"position_ids": pos_ids, "parts": linked_parts[:limit]}

    @with_request_id
    def drawings_for_position_filters(
        self,
        pos_filters: PositionFilters,
        limit: int = 100,
        request_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Variant: return drawings actually linked to the filtered positions.
        """
        rid = request_id or get_request_id()
        with log_timed_operation("AggregateSearch.drawings_for_position_filters", rid):
            pos_ids = self.search_positions(pos_filters, request_id=rid)
            if not pos_ids:
                return {"position_ids": [], "drawings": []}
            rows = self.repos.drawings.drawings_by_position_ids(pos_ids, limit=limit, request_id=rid)
            return {"position_ids": pos_ids, "drawings": rows}

    @with_request_id
    def images_for_position_filters(
        self,
        pos_filters: PositionFilters,
        limit: int = 100,
        request_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Variant: return images actually linked to the filtered positions.
        """
        rid = request_id or get_request_id()
        with log_timed_operation("AggregateSearch.images_for_position_filters", rid):
            pos_ids = self.search_positions(pos_filters, request_id=rid)
            if not pos_ids:
                return {"position_ids": [], "images": []}
            rows = self.repos.images.images_by_position_ids(pos_ids, limit=limit, request_id=rid)
            return {"position_ids": pos_ids, "images": rows}
