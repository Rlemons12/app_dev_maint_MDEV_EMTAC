# services/location_service.py
from typing import Optional, List, Union, Dict, Any
from sqlalchemy.exc import SQLAlchemyError
from modules.emtacdb.emtacdb_fts import Location
from modules.configuration.config_env import DatabaseConfig
from modules.configuration.log_config import info_id, error_id, with_request_id


class LocationService:
    """
    Service layer for managing Location entities.

    This class provides a simplified API to search, retrieve, create/update,
    delete, and explore related entities for Locations. It automatically
    manages database sessions and logs all operations.
    """

    def __init__(self, db_config: DatabaseConfig = None):
        # Use provided DatabaseConfig or fall back to a default one.
        self.db_config = db_config or DatabaseConfig()

    @with_request_id
    def find(self, name: Optional[str] = None,
             model_id: Optional[int] = None,
             limit: int = 100) -> List[Location]:
        """
        Broad search for Locations with optional filters.

        Args:
            name (str, optional): Case-insensitive partial match on Location name.
            model_id (int, optional): Filter Locations by parent Model ID.
            limit (int, optional): Max number of results to return (default 100).

        Returns:
            List[Location]: List of matching Location objects.
        """
        with self.db_config.main_session() as session:
            try:
                query = session.query(Location)
                if name:
                    query = query.filter(Location.name.ilike(f"%{name}%"))
                if model_id:
                    query = query.filter_by(model_id=model_id)
                results = query.limit(limit).all()
                info_id(f"Found {len(results)} Locations", None)
                return results
            except SQLAlchemyError as e:
                error_id(f"LocationService.find failed: {e}", None)
                raise

    @with_request_id
    def get(self, location_id: int) -> Optional[Location]:
        """
        Retrieve a Location by its primary key ID.

        Args:
            location_id (int): ID of the Location to retrieve.

        Returns:
            Location | None: The Location if found, otherwise None.
        """
        with self.db_config.main_session() as session:
            try:
                return session.query(Location).filter_by(id=location_id).first()
            except SQLAlchemyError as e:
                error_id(f"LocationService.get failed: {e}", None)
                raise

    @with_request_id
    def save(self, name: str, model_id: int,
             description: Optional[str] = None,
             location_id: Optional[int] = None) -> Location:
        """
        Create a new Location or update an existing one.

        Behavior:
            - If location_id is provided → updates the existing record.
            - Otherwise → inserts a new Location.

        Args:
            name (str): Name of the Location.
            model_id (int): ID of the parent Model.
            description (str, optional): Optional description.
            location_id (int, optional): ID of an existing Location to update.

        Returns:
            Location: The created or updated Location object.
        """
        with self.db_config.main_session() as session:
            try:
                if location_id:
                    # Update existing Location
                    location = session.query(Location).filter_by(id=location_id).first()
                    if not location:
                        raise ValueError(f"Location with id {location_id} not found")
                    location.name = name
                    location.description = description
                    location.model_id = model_id
                    info_id(f"Updated Location id={location_id}", None)
                else:
                    # Insert new Location
                    location = Location(name=name, model_id=model_id, description=description)
                    session.add(location)
                    info_id(f"Created Location '{name}'", None)
                return location
            except SQLAlchemyError as e:
                error_id(f"LocationService.save failed: {e}", None)
                raise

    @with_request_id
    def remove(self, location_id: int) -> bool:
        """
        Delete a Location by its ID.

        Args:
            location_id (int): ID of the Location to delete.

        Returns:
            bool: True if the Location was deleted, False if not found.
        """
        with self.db_config.main_session() as session:
            try:
                location = session.query(Location).filter_by(id=location_id).first()
                if location:
                    session.delete(location)
                    info_id(f"Deleted Location id={location_id}", None)
                    return True
                return False
            except SQLAlchemyError as e:
                error_id(f"LocationService.remove failed: {e}", None)
                raise

    @with_request_id
    def find_related(self, identifier: Union[int, str],
                     is_id: bool = True) -> Optional[Dict[str, Any]]:
        """
        Find related entities for a Location.

        Traverses:
          - Upward: Model → EquipmentGroup → Area
          - Downward: Positions, Subassemblies

        Args:
            identifier (int | str): Either the Location ID or Location name.
            is_id (bool): Whether identifier is an ID (default True).

        Returns:
            dict | None: Dictionary containing:
              - 'location': the Location object
              - 'upward': model, equipment_group, area
              - 'downward': positions, subassemblies
              None if the Location is not found.
        """
        with self.db_config.main_session() as session:
            try:
                if is_id:
                    location = session.query(Location).filter_by(id=identifier).first()
                else:
                    location = session.query(Location).filter_by(name=identifier).first()

                if not location:
                    return None

                upward = {
                    "model": location.model,
                    "equipment_group": location.model.equipment_group if location.model else None,
                    "area": location.model.equipment_group.area if location.model and location.model.equipment_group else None,
                }

                downward = {
                    "positions": location.position,
                    "subassemblies": location.subassembly,
                }

                return {"location": location, "upward": upward, "downward": downward}
            except SQLAlchemyError as e:
                error_id(f"LocationService.find_related failed: {e}", None)
                raise

    @with_request_id
    def find_or_create(self, name: str, model_id: int,
                       description: Optional[str] = None) -> Location:
        """
        Find a Location by name and model, or create it if not found.

        Args:
            name (str): Name of the location.
            model_id (int): ID of the parent Model.
            description (str, optional): Description text.

        Returns:
            Location: The found or newly created Location.
        """
        with self.db_config.main_session() as session:
            loc = session.query(Location).filter_by(name=name, model_id=model_id).first()
            if loc:
                return loc
            loc = Location(name=name, model_id=model_id, description=description)
            session.add(loc)
            session.commit()
            return loc