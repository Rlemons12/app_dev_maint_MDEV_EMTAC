# services/position_service.py

from typing import Optional, List
from sqlalchemy.exc import SQLAlchemyError
from flask import g

from modules.emtacdb.emtacdb_fts import (
    Position,
    EquipmentGroup, Model, AssetNumber, Location,
    Subassembly, ComponentAssembly, AssemblyView,
    SiteLocation, Campus, Building
)

from modules.configuration.config_env import DatabaseConfig
from modules.configuration.log_config import info_id, error_id, with_request_id


class PositionService:
    """
    Service layer for managing Position entities.

    Provides:
      - find                   → Search by FK filters
      - get                    → Retrieve a single position
      - save                   → Create or update a position
      - remove                 → Delete a position
      - get_dependents         → Hierarchy traversal
      - get_corresponding_ids  → Multi-filter lookups
      - get_next_level_type    → What is the next hierarchy level?
      - add_to_db              → get-or-create pattern
    """

    VALID_FIELDS = {
        "area_id", "equipment_group_id", "model_id",
        "asset_number_id", "location_id", "subassembly_id",
        "component_assembly_id", "assembly_view_id",
        "site_location_id", "campus_id", "building_id"
    }

    def __init__(self, db_config: DatabaseConfig = None):
        self.db_config = db_config or DatabaseConfig()

        # Ensures Position.get_dependent_items() can resolve model strings
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

        # Inject model map into Position class if not already set
        if getattr(Position, "MODELS_MAP", None) is None:
            Position.MODELS_MAP = self.MODEL_MAP

    # ----------------------------------------------------
    # CRUD
    # ----------------------------------------------------

    @with_request_id
    def find(self, **filters) -> List[Position]:
        """
        Safe search with validated FK filters only.
        """
        with self.db_config.main_session() as session:
            query = session.query(Position)

            for key, value in filters.items():
                if key in self.VALID_FIELDS and value not in (None, "", "null"):
                    query = query.filter(getattr(Position, key) == value)

            return query.all()

    @with_request_id
    def get(self, position_id: int) -> Optional[Position]:
        with self.db_config.main_session() as session:
            return session.get(Position, position_id)

    @with_request_id
    def save(self, **kwargs) -> Position:
        with self.db_config.main_session() as session:
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

            session.commit()
            return pos

    @with_request_id
    def remove(self, position_id: int) -> bool:
        with self.db_config.main_session() as session:
            pos = session.get(Position, position_id)
            if not pos:
                return False
            session.delete(pos)
            session.commit()
            return True

    # ----------------------------------------------------
    # Hierarchy Traversal
    # ----------------------------------------------------

    @with_request_id
    def get_dependents(self, parent_type: str, parent_id: int, child_type: Optional[str] = None):
        with self.db_config.main_session() as session:
            return Position.get_dependent_items(
                session=session,
                parent_type=parent_type,
                parent_id=parent_id,
                child_type=child_type
            )

    @with_request_id
    def get_corresponding_ids(self, **filters):
        """
        Calls Position.get_corresponding_position_ids with safe request_id handling.
        """
        with self.db_config.main_session() as session:

            request_id = getattr(g, "request_id", "no_request_id")

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
                request_id=request_id,
            )

    @with_request_id
    def get_next_level_type(self, current_level: str):
        return Position.get_next_level_type(current_level)

    # ----------------------------------------------------
    # Get-or-Create
    # ----------------------------------------------------

    @with_request_id
    def add_to_db(self, **kwargs) -> int:
        with self.db_config.main_session() as session:
            return Position.add_to_db(session=session, **kwargs)

    @with_request_id
    def get_positions_for_complete_document(
            self,
            complete_document_id: int,
            session,
    ):
        from modules.emtacdb.emtacdb_fts import CompletedDocumentPositionAssociation

        # 1. Get position IDs
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

        # 2. Load positions USING THE SAME SESSION
        positions = (
            session.query(Position)
            .filter(Position.id.in_(position_ids))
            .all()
        )

        # 3. Serialize while still attached
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

