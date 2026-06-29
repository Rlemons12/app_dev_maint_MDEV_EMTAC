from __future__ import annotations
from typing import Any, List, Optional

from sqlalchemy.orm import Session

from modules.configuration.log_config import (
    with_request_id, get_request_id, log_timed_operation, debug_id, info_id
)
from .base_repository import BaseRepository

# Robust imports for your project structure
try:
    from emtacdb_fts import Position  # type: ignore
except Exception:
    from modules.emtacdb.emtacdb_fts import Position  # type: ignore


class PositionRepository(BaseRepository):
    @with_request_id
    def search_position_ids(
        self,
        *,
        area_id: Optional[int] = None,
        equipment_group_id: Optional[int] = None,
        model_id: Optional[int] = None,
        asset_number_id: Optional[int] = None,
        location_id: Optional[int] = None,
        request_id: Optional[str] = None,
    ) -> List[int]:
        rid = request_id or get_request_id()
        sess: Session = self._session()
        debug_id(f"[PositionRepository.search_position_ids] "
                 f"area={area_id}, eg={equipment_group_id}, model={model_id}, asset={asset_number_id}, loc={location_id}", rid)
        try:
            with log_timed_operation("PositionRepository.search_position_ids", rid):
                ids = Position.get_corresponding_position_ids(
                    session=sess,
                    area_id=area_id,
                    equipment_group_id=equipment_group_id,
                    model_id=model_id,
                    asset_number_id=asset_number_id,
                    location_id=location_id,
                    request_id=rid,
                )
                info_id(f"found {len(ids)} positions", rid)
                return ids
        finally:
            if self._owns_session():
                sess.close()

    @with_request_id
    def add(
        self,
        *,
        area_id: Optional[int] = None,
        equipment_group_id: Optional[int] = None,
        model_id: Optional[int] = None,
        asset_number_id: Optional[int] = None,
        location_id: Optional[int] = None,
        subassembly_id: Optional[int] = None,
        component_assembly_id: Optional[int] = None,
        assembly_view_id: Optional[int] = None,
        site_location_id: Optional[int] = None,
        request_id: Optional[str] = None,
    ) -> int:
        rid = request_id or get_request_id()
        sess: Session = self._session()
        try:
            with log_timed_operation("PositionRepository.add", rid):
                pos_id = Position.add_to_db(
                    session=sess,
                    area_id=area_id,
                    equipment_group_id=equipment_group_id,
                    model_id=model_id,
                    asset_number_id=asset_number_id,
                    location_id=location_id,
                    subassembly_id=subassembly_id,
                    component_assembly_id=component_assembly_id,
                    assembly_view_id=assembly_view_id,
                    site_location_id=site_location_id,
                )
                info_id(f"created/ensured position id={pos_id}", rid)
                return pos_id
        finally:
            if self._owns_session():
                sess.close()

    @with_request_id
    def children_of(
        self,
        parent_type: str,
        parent_id: int,
        child_type: Optional[str] = None,
        request_id: Optional[str] = None,
    ) -> List[Any]:
        rid = request_id or get_request_id()
        sess: Session = self._session()
        try:
            with log_timed_operation("PositionRepository.children_of", rid):
                items = Position.get_dependent_items(
                    session=sess,
                    parent_type=parent_type,
                    parent_id=parent_id,
                    child_type=child_type,
                )
                return items
        finally:
            if self._owns_session():
                sess.close()

    def next_level_type(self, current_level: str) -> Optional[str]:
        return Position.get_next_level_type(current_level)
