from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session

from modules.configuration.log_config import (
    debug_id,
    error_id,
    info_id,
    warning_id,
    with_request_id,
)

from modules.emtacdb.emtacdb_fts import Position


class PositionService:
    """
    Pure domain/service layer for Position operations.

    HARD RULES:
    - NEVER open sessions
    - NEVER close sessions
    - NEVER commit
    - NEVER rollback
    - Orchestrator owns transaction lifecycle

    Notes:
    - This service intentionally does NOT call Position.add_to_db()
      because that ORM helper still commits/rolls back internally.
    - For create/get-or-create flows, this service uses session.flush()
      so the orchestrator can still own the transaction boundary.
    """

    VALID_FIELDS = {
        "id",
        "area_id",
        "equipment_group_id",
        "model_id",
        "asset_number_id",
        "location_id",
        "subassembly_id",
        "component_assembly_id",
        "assembly_view_id",
        "site_location_id",
        "campus_id",
        "building_id",
    }

    POSITION_FK_FIELDS = (
        "area_id",
        "equipment_group_id",
        "model_id",
        "asset_number_id",
        "location_id",
        "subassembly_id",
        "component_assembly_id",
        "assembly_view_id",
        "site_location_id",
        "campus_id",
        "building_id",
    )

    # ------------------------------------------------------------------
    # READ HELPERS
    # ------------------------------------------------------------------

    @with_request_id
    def find(self, session: Session, **filters: Any) -> List[Position]:
        """
        Find Position rows by allowed filters.
        """
        return self._find_internal(session=session, filters=filters)

    @with_request_id
    def get(self, session: Session, position_id: int) -> Optional[Position]:
        """
        Get one Position by primary key.
        """
        if not position_id:
            warning_id("PositionService.get called without a valid position_id")
            return None

        return self._get_internal(session=session, position_id=position_id)

    @with_request_id
    def get_positions_by_filters(
        self,
        session: Session,
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
        campus_id: Optional[int] = None,
        building_id: Optional[int] = None,
    ) -> List[Position]:
        """
        Read helper for filtered Position lookup.
        """
        filters = self._build_position_filters(
            area_id=area_id,
            equipment_group_id=equipment_group_id,
            model_id=model_id,
            asset_number_id=asset_number_id,
            location_id=location_id,
            subassembly_id=subassembly_id,
            component_assembly_id=component_assembly_id,
            assembly_view_id=assembly_view_id,
            site_location_id=site_location_id,
            campus_id=campus_id,
            building_id=building_id,
        )

        debug_id(f"PositionService.get_positions_by_filters filters={filters}")
        return self._find_internal(session=session, filters=filters)

    @with_request_id
    def get_corresponding_position_ids(
        self,
        session: Session,
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
        campus_id: Optional[int] = None,
        building_id: Optional[int] = None,
    ) -> List[int]:
        """
        Return Position IDs matching the provided hierarchy filters.
        """
        positions = self.get_positions_by_filters(
            session=session,
            area_id=area_id,
            equipment_group_id=equipment_group_id,
            model_id=model_id,
            asset_number_id=asset_number_id,
            location_id=location_id,
            subassembly_id=subassembly_id,
            component_assembly_id=component_assembly_id,
            assembly_view_id=assembly_view_id,
            site_location_id=site_location_id,
            campus_id=campus_id,
            building_id=building_id,
        )

        position_ids = [position.id for position in positions]
        info_id(f"PositionService.get_corresponding_position_ids found {len(position_ids)} ids")
        return position_ids

    @with_request_id
    def get_dependents(
        self,
        session: Session,
        *,
        parent_type: str,
        parent_id: Optional[int],
        child_type: Optional[str] = None,
    ) -> List[Any]:
        """
        Get dependent hierarchy items using Position model helper.
        Safe for service use because it only reads and uses caller-owned session.
        """
        if not parent_type:
            warning_id("PositionService.get_dependents called without parent_type")
            return []

        return Position.get_dependent_items(
            session=session,
            parent_type=parent_type,
            parent_id=parent_id,
            child_type=child_type,
        )

    @with_request_id
    def get_next_level_type(self, current_level: str) -> Optional[str]:
        """
        Return the next hierarchy level type.
        """
        if not current_level:
            warning_id("PositionService.get_next_level_type called without current_level")
            return None

        return Position.get_next_level_type(current_level)

    # ------------------------------------------------------------------
    # WRITE HELPERS
    # ------------------------------------------------------------------

    @with_request_id
    def create(
        self,
        session: Session,
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
        campus_id: Optional[int] = None,
        building_id: Optional[int] = None,
    ) -> Position:
        """
        Create a new Position row and flush it.
        No commit happens here.
        """
        filters = self._build_position_filters(
            area_id=area_id,
            equipment_group_id=equipment_group_id,
            model_id=model_id,
            asset_number_id=asset_number_id,
            location_id=location_id,
            subassembly_id=subassembly_id,
            component_assembly_id=component_assembly_id,
            assembly_view_id=assembly_view_id,
            site_location_id=site_location_id,
            campus_id=campus_id,
            building_id=building_id,
        )

        position = Position(**filters)
        session.add(position)
        session.flush()

        info_id(f"PositionService.create created Position id={position.id}")
        return position

    @with_request_id
    def get_or_create(
        self,
        session: Session,
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
        campus_id: Optional[int] = None,
        building_id: Optional[int] = None,
    ) -> Position:
        """
        Get-or-create Position using exact FK match.
        No commit happens here.
        """
        filters = self._build_position_filters(
            area_id=area_id,
            equipment_group_id=equipment_group_id,
            model_id=model_id,
            asset_number_id=asset_number_id,
            location_id=location_id,
            subassembly_id=subassembly_id,
            component_assembly_id=component_assembly_id,
            assembly_view_id=assembly_view_id,
            site_location_id=site_location_id,
            campus_id=campus_id,
            building_id=building_id,
            include_none=True,
        )

        existing = session.query(Position).filter_by(**filters).first()
        if existing:
            info_id(f"PositionService.get_or_create found existing Position id={existing.id}")
            return existing

        position = Position(**filters)
        session.add(position)
        session.flush()

        info_id(f"PositionService.get_or_create created Position id={position.id}")
        return position

    @with_request_id
    def save(self, session: Session, position: Position) -> Position:
        """
        Add or merge Position into the current session and flush.
        No commit happens here.
        """
        if position is None:
            raise ValueError("PositionService.save requires a non-null position")

        session.add(position)
        session.flush()

        info_id(f"PositionService.save flushed Position id={getattr(position, 'id', None)}")
        return position

    @with_request_id
    def remove(self, session: Session, position: Position) -> None:
        """
        Delete Position from the current session and flush.
        No commit happens here.
        """
        if position is None:
            raise ValueError("PositionService.remove requires a non-null position")

        session.delete(position)
        session.flush()

        info_id(f"PositionService.remove flushed delete for Position id={getattr(position, 'id', None)}")

    # ------------------------------------------------------------------
    # SERIALIZATION / RESPONSE HELPERS
    # ------------------------------------------------------------------

    @with_request_id
    def serialize_position(self, position: Position) -> Dict[str, Any]:
        """
        Convert a Position ORM object into a response-safe dictionary.
        """
        if position is None:
            return {}

        return {
            "id": position.id,
            "area_id": position.area_id,
            "equipment_group_id": position.equipment_group_id,
            "model_id": position.model_id,
            "asset_number_id": position.asset_number_id,
            "location_id": position.location_id,
            "subassembly_id": position.subassembly_id,
            "component_assembly_id": position.component_assembly_id,
            "assembly_view_id": position.assembly_view_id,
            "site_location_id": position.site_location_id,
            "campus_id": position.campus_id,
            "building_id": position.building_id,
        }

    @with_request_id
    def serialize_positions(self, positions: Sequence[Position]) -> List[Dict[str, Any]]:
        """
        Serialize multiple Position rows.
        """
        return [self.serialize_position(position) for position in positions]

    # ------------------------------------------------------------------
    # INTERNAL HELPERS
    # ------------------------------------------------------------------

    def _find_internal(self, session: Session, filters: Dict[str, Any]) -> List[Position]:
        """
        Internal filtered query helper.
        """
        query = session.query(Position)

        cleaned_filters: Dict[str, Any] = {}
        for key, value in filters.items():
            if key not in self.VALID_FIELDS:
                warning_id(f"PositionService._find_internal ignored invalid filter '{key}'")
                continue

            if value is None:
                continue

            cleaned_filters[key] = value

        debug_id(f"PositionService._find_internal cleaned_filters={cleaned_filters}")

        if cleaned_filters:
            query = query.filter_by(**cleaned_filters)

        return query.all()

    def _get_internal(self, session: Session, position_id: int) -> Optional[Position]:
        """
        Internal PK lookup helper.
        """
        return session.query(Position).filter(Position.id == position_id).first()

    def _build_position_filters(
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
        campus_id: Optional[int] = None,
        building_id: Optional[int] = None,
        include_none: bool = False,
    ) -> Dict[str, Optional[int]]:
        """
        Build canonical Position FK filter dictionary.

        include_none=True is important for exact get-or-create matching,
        because Position rows are defined by the full FK combination.
        """
        raw_filters: Dict[str, Optional[int]] = {
            "area_id": area_id,
            "equipment_group_id": equipment_group_id,
            "model_id": model_id,
            "asset_number_id": asset_number_id,
            "location_id": location_id,
            "subassembly_id": subassembly_id,
            "component_assembly_id": component_assembly_id,
            "assembly_view_id": assembly_view_id,
            "site_location_id": site_location_id,
            "campus_id": campus_id,
            "building_id": building_id,
        }

        if include_none:
            return raw_filters

        return {key: value for key, value in raw_filters.items() if value is not None}