# services/site_location_service.py

from typing import Optional, List, Dict, Any, Union
from sqlalchemy.orm import Session
from sqlalchemy.exc import SQLAlchemyError

from modules.emtacdb.emtacdb_fts import SiteLocation
from modules.configuration.config_env import DatabaseConfig
from modules.configuration.log_config import with_request_id


class SiteLocationService:
    """
    Transaction-aware SiteLocation service.

    RULE:
    - If session provided → NEVER commit
    - If session not provided → standalone mode allowed
    """

    def __init__(self, db_config: DatabaseConfig = None):
        self.db_config = db_config or DatabaseConfig()

    # -------------------------------------------------------------------------
    # FIND
    # -------------------------------------------------------------------------

    @with_request_id
    def find(
        self,
        title: Optional[str] = None,
        room_number: Optional[str] = None,
        site_area: Optional[str] = None,
        building_id: Optional[int] = None,
        limit: int = 100,
        session: Optional[Session] = None,
    ) -> List[SiteLocation]:

        if session:
            return self._find_internal(
                session, title, room_number, site_area, building_id, limit
            )

        with self.db_config.main_session() as s:
            return self._find_internal(
                s, title, room_number, site_area, building_id, limit
            )

    def _find_internal(
        self, session, title, room_number, site_area, building_id, limit
    ):

        query = session.query(SiteLocation)

        if title:
            query = query.filter(SiteLocation.title.ilike(f"%{title}%"))

        if room_number:
            query = query.filter(SiteLocation.room_number.ilike(f"%{room_number}%"))

        if site_area:
            query = query.filter(SiteLocation.site_area.ilike(f"%{site_area}%"))

        if building_id:
            query = query.filter(SiteLocation.building_id == building_id)

        return query.limit(limit).all()

    # -------------------------------------------------------------------------
    # GET
    # -------------------------------------------------------------------------

    @with_request_id
    def get(
        self,
        site_location_id: int,
        session: Optional[Session] = None,
    ) -> Optional[SiteLocation]:

        if session:
            return session.get(SiteLocation, site_location_id)

        with self.db_config.main_session() as s:
            return s.get(SiteLocation, site_location_id)

    # -------------------------------------------------------------------------
    # SAVE
    # -------------------------------------------------------------------------

    @with_request_id
    def save(
        self,
        title: str,
        room_number: str,
        site_area: str,
        building_id: int,
        site_location_id: Optional[int] = None,
        session: Optional[Session] = None,
    ) -> SiteLocation:

        if session:
            return self._save_internal(
                session, title, room_number, site_area, building_id, site_location_id
            )

        with self.db_config.main_session() as s:
            try:
                sl = self._save_internal(
                    s, title, room_number, site_area, building_id, site_location_id
                )
                s.commit()
                return sl
            except:
                s.rollback()
                raise

    def _save_internal(
        self, session, title, room_number, site_area, building_id, site_location_id
    ):

        if site_location_id:
            sl = session.get(SiteLocation, site_location_id)
            if not sl:
                raise ValueError(f"SiteLocation id={site_location_id} not found")

            sl.title = title
            sl.room_number = room_number
            sl.site_area = site_area
            sl.building_id = building_id
        else:
            sl = SiteLocation(
                title=title,
                room_number=room_number,
                site_area=site_area,
                building_id=building_id,
            )
            session.add(sl)

        session.flush()
        return sl

    # -------------------------------------------------------------------------
    # REMOVE
    # -------------------------------------------------------------------------

    @with_request_id
    def remove(
        self,
        site_location_id: int,
        session: Optional[Session] = None,
    ) -> bool:

        if session:
            return self._remove_internal(session, site_location_id)

        with self.db_config.main_session() as s:
            try:
                result = self._remove_internal(s, site_location_id)
                s.commit()
                return result
            except:
                s.rollback()
                raise

    def _remove_internal(self, session, site_location_id):

        sl = session.get(SiteLocation, site_location_id)
        if not sl:
            return False

        if sl.position:
            return False

        session.delete(sl)
        return True

    # -------------------------------------------------------------------------
    # FIND RELATED
    # -------------------------------------------------------------------------

    @with_request_id
    def find_related(
        self,
        identifier: Union[int, str],
        is_id: bool = True,
        session: Optional[Session] = None,
    ) -> Optional[Dict[str, Any]]:

        if session:
            return self._find_related_internal(session, identifier, is_id)

        with self.db_config.main_session() as s:
            return self._find_related_internal(s, identifier, is_id)

    def _find_related_internal(self, session, identifier, is_id):

        if is_id:
            sl = session.get(SiteLocation, identifier)
        else:
            sl = session.query(SiteLocation).filter_by(title=identifier).first()

        if not sl:
            return None

        return {
            "site_location": sl,
            "upward": {
                "building": sl.building,
                "campus": sl.building.campus if sl.building else None,
            },
            "downward": {
                "positions": sl.position,
            },
        }

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
        session: Optional[Session] = None,
    ) -> SiteLocation:

        if session:
            return self._find_or_create_internal(
                session, title, building_id, room_number, site_area
            )

        with self.db_config.main_session() as s:
            try:
                sl = self._find_or_create_internal(
                    s, title, building_id, room_number, site_area
                )
                s.commit()
                return sl
            except:
                s.rollback()
                raise

    def _find_or_create_internal(
        self, session, title, building_id, room_number, site_area
    ):

        sl = (
            session.query(SiteLocation)
            .filter_by(title=title, building_id=building_id)
            .first()
        )

        if sl:
            return sl

        sl = SiteLocation(
            title=title,
            room_number=room_number,
            site_area=site_area,
            building_id=building_id,
        )

        session.add(sl)
        session.flush()
        return sl