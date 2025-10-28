# services/subassembly_service.py
from typing import List, Optional, Dict, Any
from sqlalchemy.exc import SQLAlchemyError

from modules.configuration.log_config import info_id, error_id, with_request_id
from modules.configuration.config_env import DatabaseConfig
from modules.emtacdb.emtacdb_fts import Subassembly, Location, ComponentAssembly, Position


class SubassemblyService:
    """
    Service layer for managing Subassembly entities.

    Provides CRUD and relationship helpers:
      - `find`          → Search for Subassemblies with optional filters.
      - `get`           → Retrieve a Subassembly by ID.
      - `save`          → Create a new Subassembly or update an existing one.
      - `remove`        → Delete a Subassembly by ID.
      - `find_or_create`→ Get or create a Subassembly by name + location.
      - `find_related`  → Traverse related entities (Location, ComponentAssemblies, Positions).
    """

    def __init__(self, db_config: DatabaseConfig = None):
        self.db_config = db_config or DatabaseConfig()

    # ------------------------
    # BASIC CRUD
    # ------------------------

    @with_request_id
    def find(self, name: Optional[str] = None,
             location_id: Optional[int] = None,
             limit: int = 100) -> List[Subassembly]:
        """
        Search for Subassemblies with optional filters.

        Args:
            name (str, optional): Filter by name (partial match).
            location_id (int, optional): Restrict to a specific Location.
            limit (int): Maximum results (default 100).

        Returns:
            List[Subassembly]: Matching records.
        """
        with self.db_config.main_session() as session:
            try:
                query = session.query(Subassembly)
                if name:
                    query = query.filter(Subassembly.name.ilike(f"%{name}%"))
                if location_id:
                    query = query.filter_by(location_id=location_id)
                results = query.limit(limit).all()
                info_id(f"Found {len(results)} Subassemblies", None)
                return results
            except SQLAlchemyError as e:
                error_id(f"SubassemblyService.find failed: {e}", None)
                raise

    @with_request_id
    def get(self, subassembly_id: int) -> Optional[Subassembly]:
        """
        Retrieve a Subassembly by its ID.

        Args:
            subassembly_id (int): Primary key.

        Returns:
            Subassembly | None
        """
        with self.db_config.main_session() as session:
            try:
                return session.query(Subassembly).filter_by(id=subassembly_id).first()
            except SQLAlchemyError as e:
                error_id(f"SubassemblyService.get failed: {e}", None)
                raise

    @with_request_id
    def save(self, name: str, location_id: int,
             description: Optional[str] = None,
             subassembly_id: Optional[int] = None) -> Subassembly:
        """
        Create or update a Subassembly.

        Args:
            name (str): Subassembly name.
            location_id (int): Parent Location ID.
            description (str, optional): Description text.
            subassembly_id (int, optional): If provided → update existing.

        Returns:
            Subassembly: The created or updated record.
        """
        with self.db_config.main_session() as session:
            try:
                if subassembly_id:
                    sub = session.query(Subassembly).filter_by(id=subassembly_id).first()
                    if not sub:
                        raise ValueError(f"Subassembly with id {subassembly_id} not found")
                    sub.name = name
                    sub.location_id = location_id
                    sub.description = description
                    info_id(f"Updated Subassembly id={subassembly_id}", None)
                else:
                    sub = Subassembly(name=name, location_id=location_id, description=description)
                    session.add(sub)
                    info_id(f"Created Subassembly '{name}'", None)
                return sub
            except SQLAlchemyError as e:
                error_id(f"SubassemblyService.save failed: {e}", None)
                raise

    @with_request_id
    def remove(self, subassembly_id: int) -> bool:
        """
        Delete a Subassembly by ID.

        Args:
            subassembly_id (int): Primary key.

        Returns:
            bool: True if deleted, False if not found.
        """
        with self.db_config.main_session() as session:
            try:
                sub = session.query(Subassembly).filter_by(id=subassembly_id).first()
                if sub:
                    session.delete(sub)
                    info_id(f"Deleted Subassembly id={subassembly_id}", None)
                    return True
                return False
            except SQLAlchemyError as e:
                error_id(f"SubassemblyService.remove failed: {e}", None)
                raise

    # ------------------------
    # EXTENSIONS
    # ------------------------

    @with_request_id
    def find_or_create(self, name: str, location_id: int,
                       description: Optional[str] = None) -> Subassembly:
        """
        Find a Subassembly by name+location or create it if missing.

        Args:
            name (str): Subassembly name.
            location_id (int): Parent Location ID.
            description (str, optional): Description text.

        Returns:
            Subassembly: Existing or new record.
        """
        with self.db_config.main_session() as session:
            sub = session.query(Subassembly).filter_by(name=name, location_id=location_id).first()
            if sub:
                info_id(f"Found existing Subassembly '{name}' at location {location_id}", None)
                return sub
            sub = Subassembly(name=name, location_id=location_id, description=description)
            session.add(sub)
            session.commit()
            info_id(f"Created new Subassembly '{name}' at location {location_id}", None)
            return sub

    @with_request_id
    def find_related(self, identifier: int, is_id: bool = True) -> Optional[Dict[str, Any]]:
        """
        Retrieve related entities for a Subassembly.

        Traverses upward:
          - Location

        Traverses downward:
          - ComponentAssemblies
          - Positions

        Args:
            identifier (int): Either ID or name.
            is_id (bool): If True, treat identifier as ID, else as name.

        Returns:
            dict | None: { "subassembly": Subassembly, "upward": {...}, "downward": {...} }
        """
        with self.db_config.main_session() as session:
            try:
                if is_id:
                    sub = session.query(Subassembly).filter_by(id=identifier).first()
                else:
                    sub = session.query(Subassembly).filter_by(name=identifier).first()

                if not sub:
                    return None

                upward = {"location": sub.location}
                downward = {
                    "component_assemblies": sub.component_assembly,
                    "positions": sub.position
                }

                return {"subassembly": sub, "upward": upward, "downward": downward}
            except SQLAlchemyError as e:
                error_id(f"SubassemblyService.find_related failed: {e}", None)
                raise

