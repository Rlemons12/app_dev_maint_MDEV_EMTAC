# services/site_location_service.py

from typing import Optional, List, Dict, Any, Union
from sqlalchemy.exc import SQLAlchemyError

from modules.emtacdb.emtacdb_fts import SiteLocation
from modules.configuration.config_env import DatabaseConfig
from modules.configuration.log_config import info_id, error_id, debug_id, with_request_id


class SiteLocationService:
    """
    Service layer for managing SiteLocation entities.

    Supports CRUD operations PLUS hierarchical lookups:
        - find()                → search by title/room/area or building
        - get()                 → retrieve a site location by ID
        - save()                → create or update a site location
        - remove()              → safe delete, checks for child entities
        - find_related()        → return upward/downward relationships
        - find_or_create()      → convenience helper
    """

    def __init__(self, db_config: DatabaseConfig = None):
        self.db_config = db_config or DatabaseConfig()

    # -------------------------------------------------------------------------
    # FIND (SEARCH)
    # -------------------------------------------------------------------------

    @with_request_id
    def find(
        self,
        title: Optional[str] = None,
        room_number: Optional[str] = None,
        site_area: Optional[str] = None,
        building_id: Optional[int] = None,
        limit: int = 100,
        request_id=None
    ) -> List[SiteLocation]:

        with self.db_config.main_session() as session:
            try:
                query = session.query(SiteLocation)

                if title:
                    query = query.filter(SiteLocation.title.ilike(f"%{title}%"))
                if room_number:
                    query = query.filter(SiteLocation.room_number.ilike(f"%{room_number}%"))
                if site_area:
                    query = query.filter(SiteLocation.site_area.ilike(f"%{site_area}%"))
                if building_id:
                    query = query.filter(SiteLocation.building_id == building_id)

                results = query.limit(limit).all()
                info_id(f"Found {len(results)} SiteLocations", request_id)
                return results

            except SQLAlchemyError as e:
                error_id(f"SiteLocationService.find failed: {e}", request_id)
                raise

    # -------------------------------------------------------------------------
    # GET
    # -------------------------------------------------------------------------

    @with_request_id
    def get(self, site_location_id: int, request_id=None) -> Optional[SiteLocation]:
        with self.db_config.main_session() as session:
            try:
                sl = session.query(SiteLocation).get(site_location_id)
                debug_id(f"Retrieved SiteLocation id={site_location_id}", request_id)
                return sl
            except SQLAlchemyError as e:
                error_id(f"SiteLocationService.get failed: {e}", request_id)
                raise

    # -------------------------------------------------------------------------
    # SAVE (CREATE OR UPDATE)
    # -------------------------------------------------------------------------

    @with_request_id
    def save(
        self,
        title: str,
        room_number: str,
        site_area: str,
        building_id: int,
        site_location_id: Optional[int] = None,
        request_id=None
    ) -> SiteLocation:

        with self.db_config.main_session() as session:
            try:
                if site_location_id:
                    sl = session.query(SiteLocation).get(site_location_id)
                    if not sl:
                        raise ValueError(f"SiteLocation id={site_location_id} not found")

                    sl.title = title
                    sl.room_number = room_number
                    sl.site_area = site_area
                    sl.building_id = building_id

                    info_id(f"Updated SiteLocation id={site_location_id}", request_id)

                else:
                    sl = SiteLocation(
                        title=title,
                        room_number=room_number,
                        site_area=site_area,
                        building_id=building_id
                    )
                    session.add(sl)
                    info_id(f"Created SiteLocation '{title}'", request_id)

                session.commit()
                session.refresh(sl)
                return sl

            except SQLAlchemyError as e:
                session.rollback()
                error_id(f"SiteLocationService.save failed: {e}", request_id)
                raise

    # -------------------------------------------------------------------------
    # REMOVE (SAFE DELETE)
    # -------------------------------------------------------------------------

    @with_request_id
    def remove(self, site_location_id: int, request_id=None) -> bool:
        with self.db_config.main_session() as session:
            try:
                sl = session.query(SiteLocation).get(site_location_id)
                if not sl:
                    return False

                if sl.position:
                    error_id(
                        f"Cannot delete SiteLocation id={site_location_id}: "
                        f"{len(sl.position)} positions exist.",
                        request_id
                    )
                    return False

                session.delete(sl)
                session.commit()
                info_id(f"Deleted SiteLocation id={site_location_id}", request_id)
                return True

            except SQLAlchemyError as e:
                session.rollback()
                error_id(f"SiteLocationService.remove failed: {e}", request_id)
                raise

    # -------------------------------------------------------------------------
    # RELATIONSHIP LOOKUP
    # -------------------------------------------------------------------------

    @with_request_id
    def find_related(
        self,
        identifier: Union[int, str],
        is_id: bool = True,
        request_id=None
    ) -> Optional[Dict[str, Any]]:

        with self.db_config.main_session() as session:
            try:
                if is_id:
                    sl = session.query(SiteLocation).get(identifier)
                else:
                    sl = session.query(SiteLocation).filter_by(title=identifier).first()

                if not sl:
                    return None

                return {
                    "site_location": sl,
                    "upward": {
                        "building": sl.building,
                        "campus": sl.building.campus if sl.building else None
                    },
                    "downward": {
                        "positions": sl.position
                    }
                }

            except SQLAlchemyError as e:
                error_id(f"SiteLocationService.find_related failed: {e}", request_id)
                raise

    # -------------------------------------------------------------------------
    # FIND OR CREATE
    # -------------------------------------------------------------------------

    @with_request_id
    def find_or_create(
        self,
        title: str,
        building_id: int,
        room_number: str = "Unknown",
        site_area: str = "General",
        request_id=None
    ) -> SiteLocation:

        with self.db_config.main_session() as session:
            try:
                sl = (
                    session.query(SiteLocation)
                    .filter_by(title=title, building_id=building_id)
                    .first()
                )

                if sl:
                    debug_id(f"Found existing SiteLocation '{title}'", request_id)
                    return sl

                sl = SiteLocation(
                    title=title,
                    room_number=room_number,
                    site_area=site_area,
                    building_id=building_id
                )
                session.add(sl)
                session.commit()
                session.refresh(sl)

                info_id(f"Created new SiteLocation '{title}'", request_id)
                return sl

            except SQLAlchemyError as e:
                session.rollback()
                error_id(f"SiteLocationService.find_or_create failed: {e}", request_id)
                raise
