from typing import Optional, List, Dict, Any
from sqlalchemy.orm import Session
from sqlalchemy.exc import SQLAlchemyError

from modules.emtacdb.emtacdb_fts import Area
from modules.configuration.config_env import DatabaseConfig
from modules.configuration.log_config import with_request_id


class AreaService:
    """
    Transaction-aware Area service.

    RULE:
    - If session provided → NEVER commit
    - If session not provided → fallback standalone mode
    """

    def __init__(self, db_config: DatabaseConfig = None):
        self.db_config = db_config or DatabaseConfig()

    # ----------------------------------------------------
    # INTERNAL SESSION HANDLER
    # ----------------------------------------------------

    def _standalone_session(self):
        return self.db_config.main_session()

    # ----------------------------------------------------
    # FIND
    # ----------------------------------------------------

    @with_request_id
    def find(
        self,
        name: Optional[str] = None,
        description: Optional[str] = None,
        limit: int = 100,
        session: Session = None,
    ) -> List[Area]:

        if session:
            query = session.query(Area)
            if name:
                query = query.filter(Area.name.ilike(f"%{name}%"))
            if description:
                query = query.filter(Area.description.ilike(f"%{description}%"))
            return query.limit(limit).all()

        with self._standalone_session() as s:
            return self.find(name=name, description=description, limit=limit, session=s)

    # ----------------------------------------------------
    # GET
    # ----------------------------------------------------

    @with_request_id
    def get(self, area_id: int, session: Session = None) -> Optional[Area]:

        if session:
            return session.get(Area, area_id)

        with self._standalone_session() as s:
            return s.get(Area, area_id)

    # ----------------------------------------------------
    # SAVE
    # ----------------------------------------------------

    @with_request_id
    def save(
        self,
        name: str,
        description: Optional[str] = None,
        area_id: Optional[int] = None,
        session: Session = None,
    ) -> Area:

        if session:
            return self._save_internal(session, name, description, area_id)

        with self._standalone_session() as s:
            try:
                area = self._save_internal(s, name, description, area_id)
                s.commit()
                return area
            except:
                s.rollback()
                raise

    def _save_internal(self, session, name, description, area_id):

        if area_id:
            area = session.get(Area, area_id)
            if not area:
                raise ValueError(f"Area with id {area_id} not found")

            area.name = name
            area.description = description
        else:
            area = Area(name=name, description=description)
            session.add(area)

        session.flush()
        return area

    # ----------------------------------------------------
    # REMOVE (SAFE DELETE)
    # ----------------------------------------------------

    @with_request_id
    def remove(self, area_id: int, session: Session = None) -> bool:

        if session:
            return self._remove_internal(session, area_id)

        with self._standalone_session() as s:
            try:
                result = self._remove_internal(s, area_id)
                s.commit()
                return result
            except:
                s.rollback()
                raise

    def _remove_internal(self, session, area_id):

        area = session.get(Area, area_id)
        if not area:
            return False

        if area.equipment_group:
            return False

        if area.position:
            return False

        session.delete(area)
        return True

    # ----------------------------------------------------
    # FIND OR CREATE
    # ----------------------------------------------------

    @with_request_id
    def find_or_create(
        self,
        name: str,
        description: Optional[str] = None,
        session: Session = None,
    ) -> Area:

        if session:
            return self._find_or_create_internal(session, name, description)

        with self._standalone_session() as s:
            try:
                area = self._find_or_create_internal(s, name, description)
                s.commit()
                return area
            except:
                s.rollback()
                raise

    def _find_or_create_internal(self, session, name, description):

        area = session.query(Area).filter_by(name=name).first()
        if area:
            return area

        area = Area(name=name, description=description)
        session.add(area)
        session.flush()
        return area

    # ----------------------------------------------------
    # FIND RELATED
    # ----------------------------------------------------

    @with_request_id
    def find_related(self, area_id: int, session: Session = None):

        if session:
            area = session.get(Area, area_id)
        else:
            with self._standalone_session() as s:
                area = s.get(Area, area_id)

        if not area:
            return None

        return {
            "area": area,
            "downward": {
                "equipment_groups": area.equipment_group,
                "positions": area.position,
            },
        }