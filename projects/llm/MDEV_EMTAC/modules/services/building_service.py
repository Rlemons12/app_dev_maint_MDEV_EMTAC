from typing import Optional, List
from sqlalchemy.exc import SQLAlchemyError

from modules.emtacdb.emtacdb_fts import Building
from modules.configuration.config_env import DatabaseConfig
from modules.configuration.log_config import info_id, error_id, debug_id, with_request_id


class BuildingService:
    """Service layer for Building entities."""

    def __init__(self, db_config: DatabaseConfig = None):
        self.db_config = db_config or DatabaseConfig()

    @with_request_id
    def get(self, building_id: int, request_id=None) -> Optional[Building]:
        with self.db_config.main_session() as session:
            return session.query(Building).get(building_id)

    @with_request_id
    def find(self, name=None, campus_id=None, limit=100, request_id=None) -> List[Building]:
        with self.db_config.main_session() as session:
            query = session.query(Building)
            if name:
                query = query.filter(Building.name.ilike(f"%{name}%"))
            if campus_id:
                query = query.filter(Building.campus_id == campus_id)
            return query.limit(limit).all()

    @with_request_id
    def save(self, name, campus_id, description=None, address=None, building_id=None, request_id=None) -> Building:
        with self.db_config.main_session() as session:
            try:
                if building_id:
                    building = session.query(Building).get(building_id)
                    if not building:
                        raise ValueError(f"Building id={building_id} not found")

                    building.name = name
                    building.description = description
                    building.address = address
                    building.campus_id = campus_id

                    info_id(f"Updated Building id={building_id}", request_id)

                else:
                    building = Building(
                        name=name,
                        campus_id=campus_id,
                        description=description,
                        address=address
                    )
                    session.add(building)
                    info_id(f"Created Building '{name}' for campus {campus_id}", request_id)

                session.commit()
                session.refresh(building)
                return building

            except SQLAlchemyError as e:
                session.rollback()
                error_id(f"BuildingService.save failed: {e}", request_id)
                raise

    @with_request_id
    def remove(self, building_id: int, request_id=None) -> bool:
        with self.db_config.main_session() as session:
            try:
                building = session.query(Building).get(building_id)
                if not building:
                    return False

                if (building.site_location or building.position):
                    error_id(f"Cannot delete Building id={building_id}: child entities exist", request_id)
                    return False

                session.delete(building)
                session.commit()
                return True

            except SQLAlchemyError as e:
                session.rollback()
                error_id(f"BuildingService.remove failed: {e}", request_id)
                raise

    @with_request_id
    def find_or_create(self, name, campus_id, description=None, address=None, request_id=None):
        with self.db_config.main_session() as session:
            building = session.query(Building).filter_by(name=name, campus_id=campus_id).first()
            if building:
                return building

            building = Building(
                name=name,
                campus_id=campus_id,
                description=description,
                address=address
            )
            session.add(building)
            session.commit()
            session.refresh(building)
            return building

    @with_request_id
    def get_related(self, building_id: int, request_id=None):
        with self.db_config.main_session() as session:
            building = session.query(Building).get(building_id)
            if not building:
                return None
            return {
                "building": building,
                "campus": building.campus,
                "site_locations": building.site_location,
                "positions": building.position
            }
