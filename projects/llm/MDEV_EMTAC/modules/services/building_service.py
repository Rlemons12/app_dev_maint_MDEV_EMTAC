from typing import Optional, List
from sqlalchemy.orm import Session
from sqlalchemy.exc import SQLAlchemyError

from modules.emtacdb.emtacdb_fts import Building
from modules.configuration.config_env import DatabaseConfig
from modules.configuration.log_config import with_request_id


class BuildingService:
    """
    Transaction-aware Building service.

    RULE:
    - If session provided → NEVER commit
    - If session not provided → standalone mode allowed
    """

    def __init__(self, db_config: DatabaseConfig = None):
        self.db_config = db_config or DatabaseConfig()

    # ----------------------------------------------------
    # GET
    # ----------------------------------------------------

    @with_request_id
    def get(self, building_id: int, session: Optional[Session] = None) -> Optional[Building]:

        if session:
            return session.get(Building, building_id)

        with self.db_config.main_session() as s:
            return s.get(Building, building_id)

    # ----------------------------------------------------
    # FIND
    # ----------------------------------------------------

    @with_request_id
    def find(
        self,
        name: Optional[str] = None,
        campus_id: Optional[int] = None,
        limit: int = 100,
        session: Optional[Session] = None,
    ) -> List[Building]:

        if session:
            return self._find_internal(session, name, campus_id, limit)

        with self.db_config.main_session() as s:
            return self._find_internal(s, name, campus_id, limit)

    def _find_internal(self, session, name, campus_id, limit):

        query = session.query(Building)

        if name:
            query = query.filter(Building.name.ilike(f"%{name}%"))

        if campus_id:
            query = query.filter(Building.campus_id == campus_id)

        return query.limit(limit).all()

    # ----------------------------------------------------
    # SAVE
    # ----------------------------------------------------

    @with_request_id
    def save(
        self,
        name: str,
        campus_id: int,
        description: Optional[str] = None,
        address: Optional[str] = None,
        building_id: Optional[int] = None,
        session: Optional[Session] = None,
    ) -> Building:

        if session:
            return self._save_internal(
                session, name, campus_id, description, address, building_id
            )

        with self.db_config.main_session() as s:
            try:
                building = self._save_internal(
                    s, name, campus_id, description, address, building_id
                )
                s.commit()
                return building
            except:
                s.rollback()
                raise

    def _save_internal(
        self, session, name, campus_id, description, address, building_id
    ):

        if building_id:
            building = session.get(Building, building_id)
            if not building:
                raise ValueError(f"Building id={building_id} not found")

            building.name = name
            building.description = description
            building.address = address
            building.campus_id = campus_id

        else:
            building = Building(
                name=name,
                campus_id=campus_id,
                description=description,
                address=address,
            )
            session.add(building)

        session.flush()
        return building

    # ----------------------------------------------------
    # REMOVE
    # ----------------------------------------------------

    @with_request_id
    def remove(self, building_id: int, session: Optional[Session] = None) -> bool:

        if session:
            return self._remove_internal(session, building_id)

        with self.db_config.main_session() as s:
            try:
                result = self._remove_internal(s, building_id)
                s.commit()
                return result
            except:
                s.rollback()
                raise

    def _remove_internal(self, session, building_id):

        building = session.get(Building, building_id)
        if not building:
            return False

        if building.site_location or building.position:
            return False

        session.delete(building)
        return True

    # ----------------------------------------------------
    # FIND OR CREATE
    # ----------------------------------------------------

    @with_request_id
    def find_or_create(
        self,
        name: str,
        campus_id: int,
        description: Optional[str] = None,
        address: Optional[str] = None,
        session: Optional[Session] = None,
    ) -> Building:

        if session:
            return self._find_or_create_internal(
                session, name, campus_id, description, address
            )

        with self.db_config.main_session() as s:
            try:
                building = self._find_or_create_internal(
                    s, name, campus_id, description, address
                )
                s.commit()
                return building
            except:
                s.rollback()
                raise

    def _find_or_create_internal(
        self, session, name, campus_id, description, address
    ):

        building = (
            session.query(Building)
            .filter_by(name=name, campus_id=campus_id)
            .first()
        )

        if building:
            return building

        building = Building(
            name=name,
            campus_id=campus_id,
            description=description,
            address=address,
        )

        session.add(building)
        session.flush()
        return building

    # ----------------------------------------------------
    # HIERARCHY
    # ----------------------------------------------------

    @with_request_id
    def get_related(self, building_id: int, session: Optional[Session] = None):

        if session:
            building = session.get(Building, building_id)
        else:
            with self.db_config.main_session() as s:
                building = s.get(Building, building_id)

        if not building:
            return None

        return {
            "building": building,
            "campus": building.campus,
            "site_locations": building.site_location,
            "positions": building.position,
        }