# services/position_service.py

from typing import Optional, List
from sqlalchemy.orm import Session
from sqlalchemy.exc import SQLAlchemyError

from modules.emtacdb.emtacdb_fts import (
    Position,
    EquipmentGroup, Model, AssetNumber, Location,
    Subassembly, ComponentAssembly, AssemblyView,
    SiteLocation, Campus, Building,
    CompletedDocumentPositionAssociation
)

from modules.configuration.config_env import DatabaseConfig
from modules.configuration.log_config import with_request_id


class PositionService:
    """
    Transaction-aware Position service.

    RULE:
    - If session provided → NEVER commit
    - If session not provided → standalone mode allowed
    """

    VALID_FIELDS = {
        "area_id", "equipment_group_id", "model_id",
        "asset_number_id", "location_id", "subassembly_id",
        "component_assembly_id", "assembly_view_id",
        "site_location_id", "campus_id", "building_id"
    }

    def __init__(self, db_config: DatabaseConfig = None):
        self.db_config = db_config or DatabaseConfig()

        self.MODEL_MAP = {
            "EquipmentGroup": EquipmentGroup,
            "Model": Model,
            "AssetNumber": AssetNumber,
            "Location": Location,
            "Subassembly": Subassembly,
            "ComponentAssembly": ComponentAssembly,
            "AssemblyView": AssemblyView,
            "SiteLocation": SiteLocation,
            "Campus": Campus,
            "Building": Building,
        }

        if getattr(Position, "MODELS_MAP", None) is None:
            Position.MODELS_MAP = self.MODEL_MAP

    # ----------------------------------------------------
    # FIND
    # ----------------------------------------------------

    @with_request_id
    def find(self, session: Optional[Session] = None, **filters) -> List[Position]:

        if session:
            return self._find_internal(session, filters)

        with self.db_config.main_session() as s:
            return self._find_internal(s, filters)

    def _find_internal(self, session, filters):

        query = session.query(Position)

        for key, value in filters.items():
            if key in self.VALID_FIELDS and value not in (None, "", "null"):
                query = query.filter(getattr(Position, key) == value)

        return query.all()

    # ----------------------------------------------------
    # GET
    # ----------------------------------------------------

    @with_request_id
    def get(self, position_id: int, session: Optional[Session] = None):

        if session:
            return session.get(Position, position_id)

        with self.db_config.main_session() as s:
            return s.get(Position, position_id)

    # ----------------------------------------------------
    # SAVE
    # ----------------------------------------------------

    @with_request_id
    def save(self, session: Optional[Session] = None, **kwargs) -> Position:

        if session:
            return self._save_internal(session, kwargs)

        with self.db_config.main_session() as s:
            try:
                pos = self._save_internal(s, kwargs)
                s.commit()
                return pos
            except:
                s.rollback()
                raise

    def _save_internal(self, session, kwargs):

        if "id" in kwargs and kwargs["id"]:
            pos = session.get(Position, kwargs["id"])
            if not pos:
                raise ValueError(f"Position {kwargs['id']} not found")

            for k, v in kwargs.items():
                if hasattr(pos, k):
                    setattr(pos, k, v)
        else:
            pos = Position(**kwargs)
            session.add(pos)

        session.flush()
        return pos

    # ----------------------------------------------------
    # REMOVE
    # ----------------------------------------------------

    @with_request_id
    def remove(self, position_id: int, session: Optional[Session] = None) -> bool:

        if session:
            return self._remove_internal(session, position_id)

        with self.db_config.main_session() as s:
            try:
                result = self._remove_internal(s, position_id)
                s.commit()
                return result
            except:
                s.rollback()
                raise

    def _remove_internal(self, session, position_id):

        pos = session.get(Position, position_id)
        if not pos:
            return False

        session.delete(pos)
        return True

    # ----------------------------------------------------
    # HIERARCHY
    # ----------------------------------------------------

    @with_request_id
    def get_dependents(
        self,
        parent_type: str,
        parent_id: int,
        child_type: Optional[str] = None,
        session: Optional[Session] = None,
    ):

        if session:
            return Position.get_dependent_items(
                session=session,
                parent_type=parent_type,
                parent_id=parent_id,
                child_type=child_type
            )

        with self.db_config.main_session() as s:
            return Position.get_dependent_items(
                session=s,
                parent_type=parent_type,
                parent_id=parent_id,
                child_type=child_type
            )

    @with_request_id
    def get_corresponding_ids(
        self,
        session: Optional[Session] = None,
        **filters
    ):

        if session:
            return Position.get_corresponding_position_ids(
                session=session,
                area_id=filters.get("area_id"),
                equipment_group_id=filters.get("equipment_group_id"),
                model_id=filters.get("model_id"),
                asset_number_id=filters.get("asset_number_id"),
                location_id=filters.get("location_id"),
                campus_id=filters.get("campus_id"),
                building_id=filters.get("building_id"),
                site_location_id=filters.get("site_location_id"),
                request_id=None,
            )

        with self.db_config.main_session() as s:
            return Position.get_corresponding_position_ids(
                session=s,
                area_id=filters.get("area_id"),
                equipment_group_id=filters.get("equipment_group_id"),
                model_id=filters.get("model_id"),
                asset_number_id=filters.get("asset_number_id"),
                location_id=filters.get("location_id"),
                campus_id=filters.get("campus_id"),
                building_id=filters.get("building_id"),
                site_location_id=filters.get("site_location_id"),
                request_id=None,
            )

    @with_request_id
    def get_next_level_type(self, current_level: str):
        return Position.get_next_level_type(current_level)

    # ----------------------------------------------------
    # GET OR CREATE
    # ----------------------------------------------------

    @with_request_id
    def add_to_db(self, session: Optional[Session] = None, **kwargs) -> int:

        if session:
            return Position.add_to_db(session=session, **kwargs)

        with self.db_config.main_session() as s:
            try:
                position_id = Position.add_to_db(session=s, **kwargs)
                s.commit()
                return position_id
            except:
                s.rollback()
                raise

    # ----------------------------------------------------
    # DOCUMENT LINKING
    # ----------------------------------------------------

    @with_request_id
    def get_positions_for_complete_document(
        self,
        complete_document_id: int,
        session: Optional[Session] = None,
    ):

        if not session:
            with self.db_config.main_session() as s:
                return self._get_positions_internal(s, complete_document_id)

        return self._get_positions_internal(session, complete_document_id)

    def _get_positions_internal(self, session, complete_document_id):

        position_ids = [
            a.position_id
            for a in (
                session.query(CompletedDocumentPositionAssociation)
                .filter(
                    CompletedDocumentPositionAssociation.complete_document_id
                    == complete_document_id
                )
                .all()
            )
            if a.position_id
        ]

        if not position_ids:
            return [], []

        positions = (
            session.query(Position)
            .filter(Position.id.in_(position_ids))
            .all()
        )

        serialized = [
            {
                "id": p.id,
                "area_id": p.area_id,
                "equipment_group_id": p.equipment_group_id,
                "model_id": p.model_id,
                "asset_number_id": p.asset_number_id,
                "location_id": p.location_id,
                "subassembly_id": p.subassembly_id,
                "component_assembly_id": p.component_assembly_id,
                "assembly_view_id": p.assembly_view_id,
                "site_location_id": p.site_location_id,
                "campus_id": p.campus_id,
                "building_id": p.building_id,
            }
            for p in positions
        ]

        return serialized, position_ids