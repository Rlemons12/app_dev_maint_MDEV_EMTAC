# services/assembly_view_service.py
from typing import List, Optional, Dict, Any
from sqlalchemy.exc import SQLAlchemyError

from modules.configuration.log_config import info_id, error_id, with_request_id
from modules.database.emtacdb_fts import AssemblyView, ComponentAssembly
from modules.configuration.config_env import DatabaseConfig


class AssemblyViewService:
    """
    Service layer for managing AssemblyView (also called ComponentView) entities.

    Provides CRUD and helpers:
      - `find`          → Search for AssemblyViews with optional filters.
      - `get`           → Retrieve a single AssemblyView by ID.
      - `save`          → Create a new AssemblyView or update an existing one.
      - `remove`        → Delete an AssemblyView by ID.
      - `find_or_create`→ Find by name+parent or create new if not found.
      - `find_related`  → Traverse relationships (parent ComponentAssembly, child Positions).
    """

    def __init__(self, db_config: DatabaseConfig = None):
        self.db_config = db_config or DatabaseConfig()

    # ---------- Basic Queries ----------

    @with_request_id
    def find(self, name: Optional[str] = None,
             component_assembly_id: Optional[int] = None,
             limit: int = 100) -> List[AssemblyView]:
        """
        Search for AssemblyViews with optional filters.

        Args:
            name (str, optional): Filter by name (partial match).
            component_assembly_id (int, optional): Filter by parent ComponentAssembly ID.
            limit (int): Max results (default 100).

        Returns:
            List[AssemblyView]: Matching records.
        """
        with self.db_config.main_session() as session:
            try:
                query = session.query(AssemblyView)
                if name:
                    query = query.filter(AssemblyView.name.ilike(f"%{name}%"))
                if component_assembly_id:
                    query = query.filter_by(component_assembly_id=component_assembly_id)
                results = query.limit(limit).all()
                info_id(f"Found {len(results)} AssemblyViews", None)
                return results
            except SQLAlchemyError as e:
                error_id(f"AssemblyViewService.find failed: {e}", None)
                raise

    @with_request_id
    def get(self, assembly_view_id: int) -> Optional[AssemblyView]:
        """
        Retrieve a single AssemblyView by ID.
        """
        with self.db_config.main_session() as session:
            try:
                return session.query(AssemblyView).filter_by(id=assembly_view_id).first()
            except SQLAlchemyError as e:
                error_id(f"AssemblyViewService.get failed: {e}", None)
                raise

    # ---------- Create / Update ----------

    @with_request_id
    def save(self, name: str, component_assembly_id: int,
             description: Optional[str] = None,
             assembly_view_id: Optional[int] = None) -> AssemblyView:
        """
        Create or update an AssemblyView.

        Args:
            name (str): Name of the AssemblyView.
            component_assembly_id (int): Parent ComponentAssembly ID.
            description (str, optional): Description text.
            assembly_view_id (int, optional): If provided, updates that record.

        Returns:
            AssemblyView: Created or updated object.
        """
        with self.db_config.main_session() as session:
            try:
                if assembly_view_id:
                    av = session.query(AssemblyView).filter_by(id=assembly_view_id).first()
                    if not av:
                        raise ValueError(f"AssemblyView with id {assembly_view_id} not found")
                    av.name = name
                    av.description = description
                    av.component_assembly_id = component_assembly_id
                    info_id(f"Updated AssemblyView id={assembly_view_id}", None)
                else:
                    av = AssemblyView(
                        name=name,
                        description=description,
                        component_assembly_id=component_assembly_id
                    )
                    session.add(av)
                    info_id(f"Created AssemblyView '{name}'", None)
                return av
            except SQLAlchemyError as e:
                error_id(f"AssemblyViewService.save failed: {e}", None)
                raise

    @with_request_id
    def remove(self, assembly_view_id: int) -> bool:
        """
        Delete an AssemblyView by ID.

        Returns:
            bool: True if deleted, False if not found.
        """
        with self.db_config.main_session() as session:
            try:
                av = session.query(AssemblyView).filter_by(id=assembly_view_id).first()
                if av:
                    session.delete(av)
                    info_id(f"Deleted AssemblyView id={assembly_view_id}", None)
                    return True
                return False
            except SQLAlchemyError as e:
                error_id(f"AssemblyViewService.remove failed: {e}", None)
                raise

    # ---------- Helpers ----------

    @with_request_id
    def find_or_create(self, name: str, component_assembly_id: int,
                       description: Optional[str] = None) -> AssemblyView:
        """
        Find by (name, component_assembly_id) or create a new AssemblyView.

        Returns:
            AssemblyView: Found or newly created object.
        """
        with self.db_config.main_session() as session:
            av = (session.query(AssemblyView)
                        .filter_by(name=name, component_assembly_id=component_assembly_id)
                        .first())
            if av:
                info_id(f"Found existing AssemblyView '{name}'", None)
                return av
            av = AssemblyView(name=name, description=description, component_assembly_id=component_assembly_id)
            session.add(av)
            info_id(f"Created new AssemblyView '{name}'", None)
            return av

    @with_request_id
    def find_related(self, assembly_view_id: int) -> Optional[Dict[str, Any]]:
        """
        Get related entities for an AssemblyView.

        Traverses:
          - Upward: parent ComponentAssembly
          - Downward: positions

        Returns:
            dict | None: {
                "assembly_view": AssemblyView,
                "upward": {"component_assembly": ...},
                "downward": {"positions": [...]}
            }
        """
        with self.db_config.main_session() as session:
            av = session.query(AssemblyView).filter_by(id=assembly_view_id).first()
            if not av:
                return None

            upward = {"component_assembly": av.component_assembly}
            downward = {"positions": av.position}
            return {"assembly_view": av, "upward": upward, "downward": downward}

