# services/location_service.py

from typing import Optional, List, Union, Dict, Any
from sqlalchemy.orm import Session
from sqlalchemy.exc import SQLAlchemyError

from modules.emtacdb.emtacdb_fts import Location
from modules.configuration.config_env import DatabaseConfig
from modules.configuration.log_config import with_request_id


class LocationService:
    """
    Transaction-aware Location service.

    RULE:
    - If session provided → NEVER commit
    - If session not provided → standalone mode allowed
    """

    def __init__(self, db_config: DatabaseConfig = None):
        self.db_config = db_config or DatabaseConfig()

    # -----------------------------------------------------
    # FIND
    # -----------------------------------------------------

    @with_request_id
    def find(
        self,
        name: Optional[str] = None,
        model_id: Optional[int] = None,
        limit: int = 100,
        session: Optional[Session] = None,
    ) -> List[Location]:

        if session:
            return self._find_internal(session, name, model_id, limit)

        with self.db_config.main_session() as s:
            return self._find_internal(s, name, model_id, limit)

    def _find_internal(self, session, name, model_id, limit):

        query = session.query(Location)

        if name:
            query = query.filter(Location.name.ilike(f"%{name}%"))

        if model_id:
            query = query.filter(Location.model_id == model_id)

        return query.limit(limit).all()

    # -----------------------------------------------------
    # GET
    # -----------------------------------------------------

    @with_request_id
    def get(
        self,
        location_id: int,
        session: Optional[Session] = None,
    ) -> Optional[Location]:

        if session:
            return session.get(Location, location_id)

        with self.db_config.main_session() as s:
            return s.get(Location, location_id)

    # -----------------------------------------------------
    # SAVE
    # -----------------------------------------------------

    @with_request_id
    def save(
        self,
        name: str,
        model_id: int,
        description: Optional[str] = None,
        location_id: Optional[int] = None,
        session: Optional[Session] = None,
    ) -> Location:

        if session:
            return self._save_internal(
                session, name, model_id, description, location_id
            )

        with self.db_config.main_session() as s:
            try:
                location = self._save_internal(
                    s, name, model_id, description, location_id
                )
                s.commit()
                return location
            except:
                s.rollback()
                raise

    def _save_internal(
        self, session, name, model_id, description, location_id
    ):

        if location_id:
            location = session.get(Location, location_id)
            if not location:
                raise ValueError(f"Location id={location_id} not found")

            location.name = name
            location.description = description
            location.model_id = model_id
        else:
            location = Location(
                name=name,
                model_id=model_id,
                description=description,
            )
            session.add(location)

        session.flush()
        return location

    # -----------------------------------------------------
    # REMOVE
    # -----------------------------------------------------

    @with_request_id
    def remove(
        self,
        location_id: int,
        session: Optional[Session] = None,
    ) -> bool:

        if session:
            return self._remove_internal(session, location_id)

        with self.db_config.main_session() as s:
            try:
                result = self._remove_internal(s, location_id)
                s.commit()
                return result
            except:
                s.rollback()
                raise

    def _remove_internal(self, session, location_id):

        location = session.get(Location, location_id)
        if not location:
            return False

        session.delete(location)
        return True

    # -----------------------------------------------------
    # FIND RELATED
    # -----------------------------------------------------

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
            location = session.get(Location, identifier)
        else:
            location = session.query(Location).filter_by(name=identifier).first()

        if not location:
            return None

        upward = {
            "model": location.model,
            "equipment_group": location.model.equipment_group if location.model else None,
            "area": location.model.equipment_group.area
            if location.model and location.model.equipment_group else None,
        }

        downward = {
            "positions": location.position,
            "subassemblies": location.subassembly,
        }

        return {
            "location": location,
            "upward": upward,
            "downward": downward,
        }

    # -----------------------------------------------------
    # FIND OR CREATE
    # -----------------------------------------------------

    @with_request_id
    def find_or_create(
        self,
        name: str,
        model_id: int,
        description: Optional[str] = None,
        session: Optional[Session] = None,
    ) -> Location:

        if session:
            return self._find_or_create_internal(
                session, name, model_id, description
            )

        with self.db_config.main_session() as s:
            try:
                location = self._find_or_create_internal(
                    s, name, model_id, description
                )
                s.commit()
                return location
            except:
                s.rollback()
                raise

    def _find_or_create_internal(
        self, session, name, model_id, description
    ):

        location = (
            session.query(Location)
            .filter_by(name=name, model_id=model_id)
            .first()
        )

        if location:
            return location

        location = Location(
            name=name,
            model_id=model_id,
            description=description,
        )

        session.add(location)
        session.flush()
        return location