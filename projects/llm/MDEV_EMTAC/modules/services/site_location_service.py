# services/site_location_service.py
from typing import Optional, List, Union, Dict, Any
from sqlalchemy.exc import SQLAlchemyError
from modules.emtacdb.emtacdb_fts import SiteLocation
from modules.configuration.config_env import DatabaseConfig
from modules.configuration.log_config import info_id, error_id, with_request_id


class SiteLocationService:
    """
    Service layer for managing SiteLocation entities.

    Provides CRUD operations and helpers for relationship traversal:
      - `find`            → Search SiteLocations with filters.
      - `get`             → Retrieve a SiteLocation by ID.
      - `save`            → Create or update a SiteLocation.
      - `remove`          → Delete a SiteLocation by ID.
      - `find_related`    → Get positions at a given site location.
      - `find_or_create`  → Lookup by title, or create if not found.
    """

    def __init__(self, db_config: DatabaseConfig = None):
        # DatabaseConfig handles session lifecycle management
        self.db_config = db_config or DatabaseConfig()

    # ------------------------
    # BASIC CRUD
    # ------------------------

    @with_request_id
    def find(self, title: Optional[str] = None,
             room_number: Optional[str] = None,
             site_area: Optional[str] = None,
             limit: int = 100) -> List[SiteLocation]:
        """Search SiteLocations by optional filters."""
        with self.db_config.main_session() as session:
            try:
                query = session.query(SiteLocation)
                if title:
                    query = query.filter(SiteLocation.title.ilike(f"%{title}%"))
                if room_number:
                    query = query.filter(SiteLocation.room_number.ilike(f"%{room_number}%"))
                if site_area:
                    query = query.filter(SiteLocation.site_area.ilike(f"%{site_area}%"))
                results = query.limit(limit).all()
                info_id(f"Found {len(results)} SiteLocations", None)
                return results
            except SQLAlchemyError as e:
                error_id(f"SiteLocationService.find failed: {e}", None)
                raise

    @with_request_id
    def get(self, site_location_id: int) -> Optional[SiteLocation]:
        """Retrieve a single SiteLocation by its ID."""
        with self.db_config.main_session() as session:
            try:
                return session.query(SiteLocation).filter_by(id=site_location_id).first()
            except SQLAlchemyError as e:
                error_id(f"SiteLocationService.get failed: {e}", None)
                raise

    @with_request_id
    def save(self, title: str,
             room_number: str,
             site_area: str,
             site_location_id: Optional[int] = None) -> SiteLocation:
        """Insert or update a SiteLocation record."""
        with self.db_config.main_session() as session:
            try:
                if site_location_id:
                    sl = session.query(SiteLocation).filter_by(id=site_location_id).first()
                    if not sl:
                        raise ValueError(f"SiteLocation with id {site_location_id} not found")
                    sl.title = title
                    sl.room_number = room_number
                    sl.site_area = site_area
                    info_id(f"Updated SiteLocation id={site_location_id}", None)
                else:
                    sl = SiteLocation(title=title, room_number=room_number, site_area=site_area)
                    session.add(sl)
                    info_id(f"Created SiteLocation '{title}'", None)
                return sl
            except SQLAlchemyError as e:
                error_id(f"SiteLocationService.save failed: {e}", None)
                raise

    @with_request_id
    def remove(self, site_location_id: int) -> bool:
        """Delete a SiteLocation by ID."""
        with self.db_config.main_session() as session:
            try:
                sl = session.query(SiteLocation).filter_by(id=site_location_id).first()
                if sl:
                    session.delete(sl)
                    info_id(f"Deleted SiteLocation id={site_location_id}", None)
                    return True
                return False
            except SQLAlchemyError as e:
                error_id(f"SiteLocationService.remove failed: {e}", None)
                raise

    # ------------------------
    # RELATIONSHIP HELPERS
    # ------------------------

    @with_request_id
    def find_related(self, identifier: Union[int, str], is_id: bool = True) -> Optional[Dict[str, Any]]:
        """Retrieve related Positions for a SiteLocation (by ID or title)."""
        with self.db_config.main_session() as session:
            try:
                if is_id:
                    sl = session.query(SiteLocation).filter_by(id=identifier).first()
                else:
                    sl = session.query(SiteLocation).filter_by(title=identifier).first()

                if not sl:
                    return None

                downward = {"positions": sl.position}
                return {"site_location": sl, "downward": downward}
            except SQLAlchemyError as e:
                error_id(f"SiteLocationService.find_related failed: {e}", None)
                raise

    # ------------------------
    # CONVENIENCE HELPERS
    # ------------------------

    @with_request_id
    def find_or_create(self, title: str,
                       room_number: str = "Unknown",
                       site_area: str = "General") -> SiteLocation:
        """
        Find a SiteLocation by title, or create it if it doesn't exist.

        Args:
            title (str): Site location title.
            room_number (str): Room number (default="Unknown").
            site_area (str): Site area (default="General").

        Returns:
            SiteLocation: Found or newly created SiteLocation.
        """
        with self.db_config.main_session() as session:
            try:
                sl = session.query(SiteLocation).filter_by(title=title).first()
                if sl:
                    info_id(f"Found existing SiteLocation '{title}'", None)
                else:
                    sl = SiteLocation(title=title, room_number=room_number, site_area=site_area)
                    session.add(sl)
                    info_id(f"Created SiteLocation '{title}'", None)
                return sl
            except SQLAlchemyError as e:
                error_id(f"SiteLocationService.find_or_create failed: {e}", None)
                raise

