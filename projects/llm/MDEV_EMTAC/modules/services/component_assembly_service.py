# services/component_assembly_service.py
from typing import List, Optional, Dict, Any, Union
from sqlalchemy.exc import SQLAlchemyError

from modules.emtacdb.emtacdb_fts import ComponentAssembly, Subassembly, AssemblyView, Position
from modules.configuration.database_config import DatabaseConfig
from modules.configuration.log_config import info_id, error_id, with_request_id


class ComponentAssemblyService:
    """
    Service layer for managing ComponentAssembly entities.

    Provides a clean API with session handling and logging:
      - `find`          → Search ComponentAssemblies with optional filters.
      - `get`           → Retrieve a single ComponentAssembly by ID.
      - `save`          → Create or update a ComponentAssembly.
      - `remove`        → Delete a ComponentAssembly by ID.
      - `find_or_create`→ Get or create a ComponentAssembly by unique fields.
      - `find_related`  → Traverse relationships (Subassembly ↑, AssemblyViews ↓, Positions ↓).
    """

    def __init__(self, db_config: DatabaseConfig = None):
        self.db_config = db_config or DatabaseConfig()

    # ------------------------
    # BASIC CRUD METHODS
    # ------------------------

    @with_request_id
    def find(self, name: Optional[str] = None,
             subassembly_id: Optional[int] = None,
             limit: int = 100) -> List[ComponentAssembly]:
        """
        Search for ComponentAssemblies with optional filters.

        Args:
            name (str, optional): Partial match on ComponentAssembly name.
            subassembly_id (int, optional): Filter by Subassembly ID.
            limit (int): Max results (default 100).

        Returns:
            List[ComponentAssembly]: Matching records.
        """
        with self.db_config.main_session() as session:
            try:
                query = session.query(ComponentAssembly)
                if name:
                    query = query.filter(ComponentAssembly.name.ilike(f"%{name}%"))
                if subassembly_id:
                    query = query.filter_by(subassembly_id=subassembly_id)
                results = query.limit(limit).all()
                info_id(f"Found {len(results)} ComponentAssemblies", None)
                return results
            except SQLAlchemyError as e:
                error_id(f"ComponentAssemblyService.find failed: {e}", None)
                raise

    @with_request_id
    def get(self, component_assembly_id: int) -> Optional[ComponentAssembly]:
        """
        Retrieve a ComponentAssembly by ID.

        Args:
            component_assembly_id (int): Primary key.

        Returns:
            ComponentAssembly | None
        """
        with self.db_config.main_session() as session:
            try:
                return session.query(ComponentAssembly).filter_by(id=component_assembly_id).first()
            except SQLAlchemyError as e:
                error_id(f"ComponentAssemblyService.get failed: {e}", None)
                raise

    @with_request_id
    def save(self, name: str, subassembly_id: int,
             description: Optional[str] = None,
             component_assembly_id: Optional[int] = None) -> ComponentAssembly:
        """
        Create or update a ComponentAssembly.

        Args:
            name (str): Name of the ComponentAssembly.
            subassembly_id (int): Parent Subassembly ID.
            description (str, optional): Description.
            component_assembly_id (int, optional): Existing record ID for update.

        Returns:
            ComponentAssembly: Created or updated object.
        """
        with self.db_config.main_session() as session:
            try:
                if component_assembly_id:
                    ca = session.query(ComponentAssembly).filter_by(id=component_assembly_id).first()
                    if not ca:
                        raise ValueError(f"ComponentAssembly with id {component_assembly_id} not found")
                    ca.name = name
                    ca.description = description
                    ca.subassembly_id = subassembly_id
                    info_id(f"Updated ComponentAssembly id={component_assembly_id}", None)
                else:
                    ca = ComponentAssembly(name=name,
                                           description=description,
                                           subassembly_id=subassembly_id)
                    session.add(ca)
                    info_id(f"Created ComponentAssembly '{name}'", None)
                return ca
            except SQLAlchemyError as e:
                error_id(f"ComponentAssemblyService.save failed: {e}", None)
                raise

    @with_request_id
    def remove(self, component_assembly_id: int) -> bool:
        """
        Delete a ComponentAssembly by ID.

        Args:
            component_assembly_id (int): Record ID.

        Returns:
            bool: True if deleted, False if not found.
        """
        with self.db_config.main_session() as session:
            try:
                ca = session.query(ComponentAssembly).filter_by(id=component_assembly_id).first()
                if ca:
                    session.delete(ca)
                    info_id(f"Deleted ComponentAssembly id={component_assembly_id}", None)
                    return True
                return False
            except SQLAlchemyError as e:
                error_id(f"ComponentAssemblyService.remove failed: {e}", None)
                raise

    # ------------------------
    # HELPERS
    # ------------------------

    @with_request_id
    def find_or_create(self, name: str, subassembly_id: int,
                       description: Optional[str] = None) -> ComponentAssembly:
        """
        Find a ComponentAssembly by name + subassembly, or create it if missing.

        Args:
            name (str): ComponentAssembly name.
            subassembly_id (int): Parent Subassembly ID.
            description (str, optional): Description.

        Returns:
            ComponentAssembly: Existing or new record.
        """
        with self.db_config.main_session() as session:
            try:
                ca = (session.query(ComponentAssembly)
                            .filter_by(name=name, subassembly_id=subassembly_id)
                            .first())
                if ca:
                    info_id(f"Found existing ComponentAssembly '{name}'", None)
                    return ca
                ca = ComponentAssembly(name=name,
                                       description=description,
                                       subassembly_id=subassembly_id)
                session.add(ca)
                info_id(f"Created ComponentAssembly '{name}'", None)
                return ca
            except SQLAlchemyError as e:
                error_id(f"ComponentAssemblyService.find_or_create failed: {e}", None)
                raise

    @with_request_id
    def find_related(self, identifier: Union[int, str], is_id: bool = True) -> Optional[Dict[str, Any]]:
        """
        Traverse relationships for a ComponentAssembly.

        Upward:
          - Subassembly
        Downward:
          - AssemblyViews
          - Positions

        Args:
            identifier (int | str): ID or name of the ComponentAssembly.
            is_id (bool): Treat identifier as ID (True) or name (False).

        Returns:
            dict | None
        """
        with self.db_config.main_session() as session:
            try:
                if is_id:
                    ca = session.query(ComponentAssembly).filter_by(id=identifier).first()
                else:
                    ca = session.query(ComponentAssembly).filter_by(name=identifier).first()

                if not ca:
                    return None

                upward = {"subassembly": ca.subassembly}
                downward = {"assembly_views": ca.assembly_view, "positions": ca.position}

                return {"component_assembly": ca,
                        "upward": upward,
                        "downward": downward}
            except SQLAlchemyError as e:
                error_id(f"ComponentAssemblyService.find_related failed: {e}", None)
                raise

