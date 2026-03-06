from typing import Optional, List
from sqlalchemy.orm import Session
from sqlalchemy.exc import SQLAlchemyError

from modules.emtacdb.emtacdb_fts import Campus
from modules.configuration.config_env import DatabaseConfig
from modules.configuration.log_config import info_id, error_id, debug_id, with_request_id


class CampusService:
    """
    Transaction-aware Campus service.

    RULE:
    - If session provided → NEVER commit
    - If session not provided → fallback standalone mode
    """

    def __init__(self, db_config: DatabaseConfig = None):
        self.db_config = db_config or DatabaseConfig()

    # ----------------------------------------------------
    # INTERNAL SESSION HANDLER
    # ----------------------------------------------------

    def _get_session(self, session: Optional[Session]):
        if session:
            return session, True
        return self.db_config.main_session().__enter__(), False

    # ----------------------------------------------------
    # GET
    # ----------------------------------------------------

    @with_request_id
    def get(self, campus_id: int, session: Session = None) -> Optional[Campus]:

        session, external = self._get_session(session)

        try:
            campus = session.get(Campus, campus_id)
            return campus
        finally:
            if not external:
                session.close()

    # ----------------------------------------------------
    # FIND
    # ----------------------------------------------------

    @with_request_id
    def find(
        self,
        session: Session = None,
        name=None,
        city=None,
        state=None,
        country=None,
        limit=100,
    ) -> List[Campus]:

        session, external = self._get_session(session)

        try:
            query = session.query(Campus)

            if name:
                query = query.filter(Campus.name.ilike(f"%{name}%"))
            if city:
                query = query.filter(Campus.city.ilike(f"%{city}%"))
            if state:
                query = query.filter(Campus.state.ilike(f"%{state}%"))
            if country:
                query = query.filter(Campus.country.ilike(f"%{country}%"))

            return query.limit(limit).all()

        finally:
            if not external:
                session.close()

    # ----------------------------------------------------
    # SAVE (create/update)
    # ----------------------------------------------------

    @with_request_id
    def save(
        self,
        session: Session = None,
        name=None,
        description=None,
        city=None,
        state=None,
        country=None,
        campus_id=None,
    ) -> Campus:

        session, external = self._get_session(session)

        try:
            if campus_id:
                campus = session.get(Campus, campus_id)
                if not campus:
                    raise ValueError(f"Campus id={campus_id} not found")

                campus.name = name
                campus.description = description
                campus.city = city
                campus.state = state
                campus.country = country

            else:
                campus = Campus(
                    name=name,
                    description=description,
                    city=city,
                    state=state,
                    country=country,
                )
                session.add(campus)

            session.flush()

            if not external:
                session.commit()

            return campus

        except SQLAlchemyError:
            if not external:
                session.rollback()
            raise
        finally:
            if not external:
                session.close()

    # ----------------------------------------------------
    # REMOVE
    # ----------------------------------------------------

    @with_request_id
    def remove(self, campus_id: int, session: Session = None) -> bool:

        session, external = self._get_session(session)

        try:
            campus = session.get(Campus, campus_id)
            if not campus:
                return False

            if campus.building or campus.position:
                return False

            session.delete(campus)

            if not external:
                session.commit()

            return True

        finally:
            if not external:
                session.close()

    # ----------------------------------------------------
    # FIND OR CREATE (transaction-safe)
    # ----------------------------------------------------

    @with_request_id
    def find_or_create(
        self,
        session: Session,
        name,
        description=None,
        city=None,
        state=None,
        country=None,
    ) -> Campus:

        campus = session.query(Campus).filter_by(name=name).first()

        if campus:
            return campus

        campus = Campus(
            name=name,
            description=description,
            city=city,
            state=state,
            country=country,
        )

        session.add(campus)
        session.flush()

        return campus

    # ----------------------------------------------------
    # HIERARCHY
    # ----------------------------------------------------

    @with_request_id
    def get_related(self, campus_id: int, session: Session):

        campus = session.get(Campus, campus_id)
        if not campus:
            return None

        return {
            "campus": campus,
            "buildings": campus.building,
            "positions": campus.position,
        }
