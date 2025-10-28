# services/area_service.py
from typing import Optional, List
from sqlalchemy.exc import SQLAlchemyError
from modules.emtacdb.emtacdb_fts import Area
from modules.configuration.config_env import DatabaseConfig
from modules.configuration.log_config import info_id, error_id, with_request_id


class AreaService:
    """
    Service layer for managing Area entities.

    Provides CRUD operations and search helpers:
      - `find`          → Search areas by name/description.
      - `get`           → Retrieve an area by ID.
      - `save`          → Create or update an area.
      - `remove`        → Delete an area by ID.
      - `find_or_create`→ Get an existing area by name or create a new one.
    """

    def __init__(self, db_config: DatabaseConfig = None):
        self.db_config = db_config or DatabaseConfig()

    @with_request_id
    def find(self, name: Optional[str] = None,
             description: Optional[str] = None,
             limit: int = 100) -> List[Area]:
        """
        Search for Areas by name and/or description.

        Args:
            name (str, optional): Case-insensitive match on area name.
            description (str, optional): Case-insensitive match on description.
            limit (int): Max number of results (default 100).

        Returns:
            List[Area]: Matching Area objects.
        """
        with self.db_config.main_session() as session:
            try:
                query = session.query(Area)
                if name:
                    query = query.filter(Area.name.ilike(f"%{name}%"))
                if description:
                    query = query.filter(Area.description.ilike(f"%{description}%"))
                results = query.limit(limit).all()
                info_id(f"Found {len(results)} Areas", None)
                return results
            except SQLAlchemyError as e:
                error_id(f"AreaService.find failed: {e}", None)
                raise

    @with_request_id
    def get(self, area_id: int) -> Optional[Area]:
        """
        Retrieve an Area by its ID.

        Args:
            area_id (int): Primary key.

        Returns:
            Area | None
        """
        with self.db_config.main_session() as session:
            try:
                return session.query(Area).filter_by(id=area_id).first()
            except SQLAlchemyError as e:
                error_id(f"AreaService.get failed: {e}", None)
                raise

    @with_request_id
    def save(self, name: str, description: Optional[str] = None,
             area_id: Optional[int] = None) -> Area:
        """
        Create or update an Area.

        Args:
            name (str): Area name.
            description (str, optional): Area description.
            area_id (int, optional): Update existing record if provided.

        Returns:
            Area: Created or updated Area object.
        """
        with self.db_config.main_session() as session:
            try:
                if area_id:
                    area = session.query(Area).filter_by(id=area_id).first()
                    if not area:
                        raise ValueError(f"Area with id {area_id} not found")
                    area.name = name
                    area.description = description
                    info_id(f"Updated Area id={area_id}", None)
                else:
                    area = Area(name=name, description=description)
                    session.add(area)
                    info_id(f"Created Area '{name}'", None)
                return area
            except SQLAlchemyError as e:
                error_id(f"AreaService.save failed: {e}", None)
                raise

    @with_request_id
    def remove(self, area_id: int) -> bool:
        """
        Delete an Area by ID.

        Args:
            area_id (int): Area ID to delete.

        Returns:
            bool: True if deleted, False if not found.
        """
        with self.db_config.main_session() as session:
            try:
                area = session.query(Area).filter_by(id=area_id).first()
                if area:
                    session.delete(area)
                    info_id(f"Deleted Area id={area_id}", None)
                    return True
                return False
            except SQLAlchemyError as e:
                error_id(f"AreaService.remove failed: {e}", None)
                raise

    @with_request_id
    def find_or_create(self, name: str, description: Optional[str] = None) -> Area:
        """
        Find an Area by name, or create it if it doesn't exist.

        Args:
            name (str): Name of the area.
            description (str, optional): Description if creating new.

        Returns:
            Area: The found or newly created Area object.
        """
        with self.db_config.main_session() as session:
            try:
                area = session.query(Area).filter_by(name=name).first()
                if area:
                    info_id(f"Found existing Area '{name}'", None)
                else:
                    area = Area(name=name, description=description)
                    session.add(area)
                    session.commit()
                    info_id(f"Created new Area '{name}'", None)
                return area
            except SQLAlchemyError as e:
                error_id(f"AreaService.find_or_create failed: {e}", None)
                raise

