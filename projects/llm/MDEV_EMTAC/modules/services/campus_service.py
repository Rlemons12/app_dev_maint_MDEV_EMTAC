from typing import Optional, List
from sqlalchemy.exc import SQLAlchemyError

from modules.emtacdb.emtacdb_fts import Campus
from modules.configuration.config_env import DatabaseConfig
from modules.configuration.log_config import info_id, error_id, debug_id, with_request_id


class CampusService:
    """Service layer for Campus entities."""

    def __init__(self, db_config: DatabaseConfig = None):
        self.db_config = db_config or DatabaseConfig()

    # ------------------------- GET -------------------------
    @with_request_id
    def get(self, campus_id: int, request_id=None) -> Optional[Campus]:
        with self.db_config.main_session() as session:
            try:
                campus = session.query(Campus).get(campus_id)
                debug_id(f"Retrieved Campus id={campus_id}", request_id)
                return campus
            except SQLAlchemyError as e:
                error_id(f"CampusService.get failed: {e}", request_id)
                raise

    # ------------------------- FIND -------------------------
    @with_request_id
    def find(self, name=None, city=None, state=None, country=None, limit=100, request_id=None) -> List[Campus]:
        with self.db_config.main_session() as session:
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

                results = query.limit(limit).all()
                info_id(f"Found {len(results)} campuses", request_id)
                return results

            except SQLAlchemyError as e:
                error_id(f"CampusService.find failed: {e}", request_id)
                raise

    # ------------------------- SAVE (create/update) -------------------------
    @with_request_id
    def save(self, name, description=None, city=None, state=None, country=None, campus_id=None, request_id=None) -> Campus:
        with self.db_config.main_session() as session:
            try:
                if campus_id:
                    campus = session.query(Campus).get(campus_id)
                    if not campus:
                        raise ValueError(f"Campus id={campus_id} not found")

                    campus.name = name
                    campus.description = description
                    campus.city = city
                    campus.state = state
                    campus.country = country

                    info_id(f"Updated Campus id={campus_id}", request_id)

                else:
                    campus = Campus(
                        name=name,
                        description=description,
                        city=city,
                        state=state,
                        country=country
                    )
                    session.add(campus)
                    info_id(f"Created new Campus '{name}'", request_id)

                session.commit()
                session.refresh(campus)
                return campus

            except SQLAlchemyError as e:
                session.rollback()
                error_id(f"CampusService.save failed: {e}", request_id)
                raise

    # ------------------------- REMOVE -------------------------
    @with_request_id
    def remove(self, campus_id: int, request_id=None) -> bool:
        with self.db_config.main_session() as session:
            try:
                campus = session.query(Campus).get(campus_id)
                if not campus:
                    return False

                if (campus.building or campus.position):
                    error_id(f"Cannot delete Campus id={campus_id}: child entities exist", request_id)
                    return False

                session.delete(campus)
                session.commit()
                info_id(f"Deleted Campus id={campus_id}", request_id)
                return True

            except SQLAlchemyError as e:
                session.rollback()
                error_id(f"CampusService.remove failed: {e}", request_id)
                raise

    # ------------------------- FIND OR CREATE -------------------------
    @with_request_id
    def find_or_create(self, name, description=None, city=None, state=None, country=None, request_id=None) -> Campus:
        with self.db_config.main_session() as session:
            try:
                campus = session.query(Campus).filter_by(name=name).first()
                if campus:
                    return campus

                campus = Campus(
                    name=name,
                    description=description,
                    city=city,
                    state=state,
                    country=country
                )
                session.add(campus)
                session.commit()
                session.refresh(campus)
                info_id(f"Created new Campus '{name}'", request_id)
                return campus

            except SQLAlchemyError as e:
                session.rollback()
                error_id(f"CampusService.find_or_create failed: {e}", request_id)
                raise

    # ------------------------- HIERARCHY -------------------------
    @with_request_id
    def get_related(self, campus_id: int, request_id=None):
        with self.db_config.main_session() as session:
            campus = session.query(Campus).get(campus_id)
            if not campus:
                return None

            return {
                "campus": campus,
                "buildings": campus.building,
                "positions": campus.position
            }

