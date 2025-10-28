# services/position_service.py
from typing import Optional, List, Dict, Any
from sqlalchemy.exc import SQLAlchemyError
from modules.emtacdb.emtacdb_fts import Position
from modules.configuration.config_env import DatabaseConfig
from modules.configuration.log_config import info_id, error_id, with_request_id


class PositionService:
    """
    Service layer for managing Position entities.

    Provides:
      - `find`               → Search positions by foreign key filters.
      - `get`                → Retrieve a single position by ID.
      - `save`               → Create or update a position.
      - `remove`             → Delete a position.
      - `get_dependents`     → Traverse hierarchy (Area → EquipmentGroup → Model → …).
      - `get_corresponding_ids` → Look up position IDs based on filters.
      - `get_next_level_type`   → Get the next hierarchy level (e.g., model → location).
      - `add_to_db`          → Get-or-create a position by FK set.
    """

    def __init__(self, db_config: DatabaseConfig = None):
        self.db_config = db_config or DatabaseConfig()

    # ------------------------
    # CRUD
    # ------------------------

    @with_request_id
    def find(self, **filters) -> List[Position]:
        """Search for Positions by optional filters (area_id, model_id, etc.)."""
        with self.db_config.main_session() as session:
            query = session.query(Position)
            for field, value in filters.items():
                if value is not None and hasattr(Position, field):
                    query = query.filter(getattr(Position, field) == value)
            return query.all()

    @with_request_id
    def get(self, position_id: int) -> Optional[Position]:
        """Retrieve a single Position by ID."""
        with self.db_config.main_session() as session:
            return session.query(Position).get(position_id)

    @with_request_id
    def save(self, **kwargs) -> Position:
        """Create or update a Position (update if id present)."""
        with self.db_config.main_session() as session:
            if "id" in kwargs and kwargs["id"]:
                pos = session.query(Position).get(kwargs["id"])
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
        """Delete a Position by ID. Returns True if deleted."""
        with self.db_config.main_session() as session:
            pos = session.query(Position).get(position_id)
            if pos:
                session.delete(pos)
                session.commit()
                return True
            return False

    # ------------------------
    # Hierarchy helpers
    # ------------------------

    @with_request_id
    def get_dependents(self, parent_type: str, parent_id: int, child_type: Optional[str] = None):
        """Traverse hierarchy to get dependent items."""
        with self.db_config.main_session() as session:
            return Position.get_dependent_items(session, parent_type, parent_id, child_type)

    @with_request_id
    def get_corresponding_ids(self, **filters):
        """Find corresponding Position IDs for the given filters."""
        with self.db_config.main_session() as session:
            return Position.get_corresponding_position_ids(session=session, **filters)

    @with_request_id
    def get_next_level_type(self, current_level: str):
        """Get the next level type in the hierarchy (e.g., area → equipment_group)."""
        return Position.get_next_level_type(current_level)

    @with_request_id
    def add_to_db(self, **kwargs) -> int:
        """Get-or-create a Position with the given FK values. Returns Position ID."""
        with self.db_config.main_session() as session:
            return Position.add_to_db(session=session, **kwargs)
