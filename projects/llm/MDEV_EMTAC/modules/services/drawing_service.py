# services/drawing_service.py
from typing import Optional, List, Dict, Any
from sqlalchemy.exc import SQLAlchemyError

from modules.emtacdb.emtacdb_fts import Drawing, DrawingType
from modules.configuration.config_env import DatabaseConfig
from modules.configuration.log_config import info_id, error_id, with_request_id


class DrawingService:
    """
    Service layer for managing Drawing entities.

    Provides:
      - `add_to_db`       → Create a new drawing
      - `find`            → Wraps Drawing.search
      - `find_formatted`  → Wraps Drawing.search_and_format
      - `get`             → Retrieve a Drawing by ID
      - `remove`          → Delete a Drawing
      - `find_by_asset`   → Search drawings related to asset numbers
      - `find_by_type`    → Search drawings by type
      - `available_types` → Get all supported drawing types
      - `find_related`    → Traverse relationships (positions, problems, tasks, parts)
    """

    def __init__(self, db_config: DatabaseConfig = None):
        self.db_config = db_config or DatabaseConfig()

    # ------------------------
    # CORE CRUD
    # ------------------------

    @with_request_id
    def add_to_db(self,
                  drw_equipment_name: Optional[str],
                  drw_number: str,
                  drw_name: Optional[str] = None,
                  drw_revision: Optional[str] = None,
                  drw_spare_part_number: Optional[str] = None,
                  drw_type: str = "Other",
                  file_path: str = None) -> int:
        """Create a new Drawing and return its ID."""
        with self.db_config.main_session() as session:
            try:
                drawing = Drawing(
                    drw_equipment_name=drw_equipment_name,
                    drw_number=drw_number,
                    drw_name=drw_name,
                    drw_revision=drw_revision,
                    drw_spare_part_number=drw_spare_part_number,
                    drw_type=drw_type,
                    file_path=file_path
                )
                session.add(drawing)
                session.commit()
                info_id(f"Created new Drawing '{drw_number}' (id={drawing.id})", None)
                return drawing.id
            except SQLAlchemyError as e:
                session.rollback()
                error_id(f"DrawingService.add_to_db failed: {e}", None)
                raise

    @with_request_id
    def get(self, drawing_id: int) -> Optional[Drawing]:
        """Retrieve a Drawing by ID."""
        try:
            return Drawing.get_by_id(drawing_id)
        except SQLAlchemyError as e:
            error_id(f"DrawingService.get failed: {e}", None)
            raise

    @with_request_id
    def remove(self, drawing_id: int) -> bool:
        """Delete a Drawing by ID."""
        with self.db_config.main_session() as session:
            try:
                drawing = session.query(Drawing).filter_by(id=drawing_id).first()
                if drawing:
                    session.delete(drawing)
                    info_id(f"Deleted Drawing id={drawing_id}", None)
                    return True
                return False
            except SQLAlchemyError as e:
                error_id(f"DrawingService.remove failed: {e}", None)
                raise

    # ------------------------
    # SEARCH HELPERS
    # ------------------------

    @with_request_id
    def find(self, **filters) -> List[Drawing]:
        """Search for Drawings (wrapper for Drawing.search)."""
        try:
            return Drawing.search(**filters)
        except SQLAlchemyError as e:
            error_id(f"DrawingService.find failed: {e}", None)
            raise

    @with_request_id
    def find_formatted(self, **filters) -> Dict[str, Any]:
        """Search and return formatted results (wrapper for Drawing.search_and_format)."""
        try:
            return Drawing.search_and_format(**filters)
        except SQLAlchemyError as e:
            error_id(f"DrawingService.find_formatted failed: {e}", None)
            raise

    @with_request_id
    def find_by_asset(self, asset_number: str) -> List[Drawing]:
        """Search drawings by asset number."""
        try:
            return Drawing.search_by_asset_number(asset_number)
        except SQLAlchemyError as e:
            error_id(f"DrawingService.find_by_asset failed: {e}", None)
            raise

    @with_request_id
    def find_by_type(self, drawing_type: str) -> List[Drawing]:
        """Search drawings by type."""
        try:
            return Drawing.search_by_type(drawing_type)
        except SQLAlchemyError as e:
            error_id(f"DrawingService.find_by_type failed: {e}", None)
            raise

    @with_request_id
    def available_types(self) -> List[str]:
        """Return all available drawing types."""
        return Drawing.get_available_types()

    # ------------------------
    # RELATIONSHIPS
    # ------------------------

    @with_request_id
    def find_related(self, drawing_id: int) -> Optional[Dict[str, Any]]:
        """Return related entities for a Drawing (positions, problems, tasks, parts)."""
        with self.db_config.main_session() as session:
            try:
                drawing = session.query(Drawing).filter_by(id=drawing_id).first()
                if not drawing:
                    return None
                return {
                    "drawing": drawing,
                    "downward": {
                        "positions": drawing.drawing_position,
                        "problems": drawing.drawing_problem,
                        "tasks": drawing.drawing_task,
                        "parts": drawing.drawing_part,
                    }
                }
            except SQLAlchemyError as e:
                error_id(f"DrawingService.find_related failed: {e}", None)
                raise
