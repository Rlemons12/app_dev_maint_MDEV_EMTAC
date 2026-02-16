# services/assembly_view_service.py

from typing import List, Optional, Dict, Any
from sqlalchemy.exc import SQLAlchemyError

from modules.configuration.log_config import info_id, error_id, warning_id, with_request_id
from modules.emtacdb.emtacdb_fts import AssemblyView, ComponentAssembly
from modules.configuration.config_env import DatabaseConfig


class AssemblyViewService:
    """
    Service layer for managing AssemblyView (ComponentView) entities.

    Responsibilities:
    ------------------
    CRUD OPERATIONS
      - find:
            Search AssemblyViews by name, component_assembly_id, or asset_number_id.
      - get:
            Retrieve a single AssemblyView by ID.
      - save:
            Create or update an AssemblyView,
            including:
                • name
                • description
                • component_assembly_id
                • asset_number_id
      - remove:
            Delete an AssemblyView IF it has no dependent Position records.

    CREATION HELPERS
      - find_or_create:
            Uniqueness is defined by:
                (name, component_assembly_id, asset_number_id)
            Creates a new view only if one does not already exist.

    RELATIONSHIP TRAVERSAL
      - find_related:
            Returns upward and downward hierarchy:
                upward:
                    • component_assembly
                    • asset_number
                downward:
                    • positions mapped to this AssemblyView
    """

    def __init__(self, db_config: DatabaseConfig = None):
        self.db_config = db_config or DatabaseConfig()

    # ----------------------------------------------------------------------
    # FIND
    # ----------------------------------------------------------------------
    @with_request_id
    def find(
        self,
        name: Optional[str] = None,
        component_assembly_id: Optional[int] = None,
        asset_number_id: Optional[int] = None,
        limit: int = 100
    ) -> List[AssemblyView]:

        with self.db_config.main_session() as session:
            try:
                query = session.query(AssemblyView)

                if name:
                    query = query.filter(AssemblyView.name.ilike(f"%{name}%"))
                if component_assembly_id:
                    query = query.filter(AssemblyView.component_assembly_id == component_assembly_id)
                if asset_number_id:
                    query = query.filter(AssemblyView.asset_number_id == asset_number_id)

                results = query.limit(limit).all()
                info_id(f"Found {len(results)} AssemblyViews", None)
                return results

            except SQLAlchemyError as e:
                error_id(f"AssemblyViewService.find failed: {e}", None)
                raise

    # ----------------------------------------------------------------------
    # GET
    # ----------------------------------------------------------------------
    @with_request_id
    def get(self, assembly_view_id: int) -> Optional[AssemblyView]:

        with self.db_config.main_session() as session:
            try:
                return session.query(AssemblyView).filter_by(id=assembly_view_id).first()
            except SQLAlchemyError as e:
                error_id(f"AssemblyViewService.get failed: {e}", None)
                raise

    # ----------------------------------------------------------------------
    # SAVE (create or update)
    # ----------------------------------------------------------------------
    @with_request_id
    def save(
        self,
        name: str,
        component_assembly_id: int,
        description: Optional[str] = None,
        asset_number_id: Optional[int] = None,
        assembly_view_id: Optional[int] = None
    ) -> AssemblyView:

        with self.db_config.main_session() as session:
            try:
                if assembly_view_id:
                    av = session.query(AssemblyView).filter_by(id=assembly_view_id).first()
                    if not av:
                        raise ValueError(f"AssemblyView with id {assembly_view_id} not found")

                    av.name = name
                    av.description = description
                    av.component_assembly_id = component_assembly_id
                    av.asset_number_id = asset_number_id

                    info_id(f"Updated AssemblyView id={assembly_view_id}", None)

                else:
                    av = AssemblyView(
                        name=name,
                        description=description,
                        component_assembly_id=component_assembly_id,
                        asset_number_id=asset_number_id
                    )
                    session.add(av)
                    info_id(f"Created AssemblyView '{name}'", None)

                return av

            except SQLAlchemyError as e:
                error_id(f"AssemblyViewService.save failed: {e}", None)
                raise

    # ----------------------------------------------------------------------
    # REMOVE (safe delete)
    # ----------------------------------------------------------------------
    @with_request_id
    def remove(self, assembly_view_id: int) -> bool:

        with self.db_config.main_session() as session:
            try:
                av = session.query(AssemblyView).filter_by(id=assembly_view_id).first()
                if not av:
                    return False

                # Prevent deleting assembly views that still have positions
                if av.position and len(av.position) > 0:
                    warning_id(
                        f"Cannot delete AssemblyView id={assembly_view_id} "
                        f"because {len(av.position)} Positions depend on it",
                        None
                    )
                    return False

                session.delete(av)
                info_id(f"Deleted AssemblyView id={assembly_view_id}", None)
                return True

            except SQLAlchemyError as e:
                error_id(f"AssemblyViewService.remove failed: {e}", None)
                raise

    # ----------------------------------------------------------------------
    # FIND_OR_CREATE (matches ORM uniqueness)
    # ----------------------------------------------------------------------
    @with_request_id
    def find_or_create(
        self,
        name: str,
        component_assembly_id: int,
        asset_number_id: Optional[int] = None,
        description: Optional[str] = None
    ) -> AssemblyView:

        with self.db_config.main_session() as session:
            filters = {
                "name": name,
                "component_assembly_id": component_assembly_id,
                "asset_number_id": asset_number_id
            }

            av = session.query(AssemblyView).filter_by(**filters).first()
            if av:
                info_id(f"Found existing AssemblyView '{name}'", None)
                return av

            av = AssemblyView(
                name=name,
                description=description,
                component_assembly_id=component_assembly_id,
                asset_number_id=asset_number_id
            )
            session.add(av)
            info_id(f"Created new AssemblyView '{name}'", None)
            return av

    # ----------------------------------------------------------------------
    # FIND_RELATED
    # ----------------------------------------------------------------------
    @with_request_id
    def find_related(self, assembly_view_id: int) -> Optional[Dict[str, Any]]:

        with self.db_config.main_session() as session:
            av = session.query(AssemblyView).filter_by(id=assembly_view_id).first()
            if not av:
                return None

            return {
                "assembly_view": av,
                "upward": {
                    "component_assembly": av.component_assembly,
                    "asset_number": av.asset_number,
                },
                "downward": {
                    "positions": av.position
                }
            }
