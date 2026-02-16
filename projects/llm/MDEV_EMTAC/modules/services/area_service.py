#MDEV_EMTAC\modules\services\area_service.py
from typing import Optional, List, Dict, Any
from sqlalchemy.exc import SQLAlchemyError

from modules.emtacdb.emtacdb_fts import Area
from modules.configuration.config_env import DatabaseConfig
from modules.configuration.log_config import info_id, warning_id, error_id, with_request_id


class AreaService:
    """
    Service layer for managing Area entities.

    Responsibilities:
    ------------------
    CRUD:
        - find
        - get
        - save (create or update)
        - remove (safe delete with dependency check)
        - find_or_create

    RELATIONSHIPS:
        - find_related → returns EquipmentGroups and Positions under the Area
    """

    def __init__(self, db_config: DatabaseConfig = None):
        self.db_config = db_config or DatabaseConfig()

    # ----------------------------------------------------------------------
    # FIND
    # ----------------------------------------------------------------------
    @with_request_id
    def find(self,
             name: Optional[str] = None,
             description: Optional[str] = None,
             limit: int = 100) -> List[Area]:

        with self.db_config.main_session() as session:
            try:
                query = session.query(Area)

                if name:
                    query = query.filter(Area.name.ilike(f"%{name}%"))
                if description:
                    query = query.filter(Area.description.ilike(f"%{description}%"))

                results = query.limit(limit).all()
                info_id(f"Found {len(results)} Areas", None)
                return results

            except SQLAlchemyError as e:
                error_id(f"AreaService.find failed: {e}", None)
                raise

    # ----------------------------------------------------------------------
    # GET
    # ----------------------------------------------------------------------
    @with_request_id
    def get(self, area_id: int) -> Optional[Area]:

        with self.db_config.main_session() as session:
            try:
                return session.query(Area).filter_by(id=area_id).first()
            except SQLAlchemyError as e:
                error_id(f"AreaService.get failed: {e}", None)
                raise

    # ----------------------------------------------------------------------
    # SAVE (CREATE OR UPDATE)
    # ----------------------------------------------------------------------
    @with_request_id
    def save(self,
             name: str,
             description: Optional[str] = None,
             area_id: Optional[int] = None) -> Area:

        with self.db_config.main_session() as session:
            try:
                if area_id:
                    area = session.query(Area).filter_by(id=area_id).first()
                    if not area:
                        raise ValueError(f"Area with id {area_id} not found")

                    area.name = name
                    area.description = description

                    session.commit()
                    info_id(f"Updated Area id={area_id}", None)

                else:
                    area = Area(name=name, description=description)
                    session.add(area)
                    session.commit()
                    info_id(f"Created Area '{name}'", None)

                return area

            except SQLAlchemyError as e:
                error_id(f"AreaService.save failed: {e}", None)
                raise

    # ----------------------------------------------------------------------
    # REMOVE (SAFE DELETE)
    # ----------------------------------------------------------------------
    @with_request_id
    def remove(self, area_id: int) -> bool:

        with self.db_config.main_session() as session:
            try:
                area = session.query(Area).filter_by(id=area_id).first()
                if not area:
                    return False

                # Prevent deleting areas that still have dependent children
                if area.equipment_group and len(area.equipment_group) > 0:
                    warning_id(
                        f"Cannot delete Area id={area_id}: "
                        f"{len(area.equipment_group)} EquipmentGroups depend on it",
                        None
                    )
                    return False

                if area.position and len(area.position) > 0:
                    warning_id(
                        f"Cannot delete Area id={area_id}: "
                        f"{len(area.position)} Positions depend on it",
                        None
                    )
                    return False

                session.delete(area)
                session.commit()
                info_id(f"Deleted Area id={area_id}", None)
                return True

            except SQLAlchemyError as e:
                error_id(f"AreaService.remove failed: {e}", None)
                raise

    # ----------------------------------------------------------------------
    # FIND OR CREATE
    # ----------------------------------------------------------------------
    @with_request_id
    def find_or_create(self,
                       name: str,
                       description: Optional[str] = None) -> Area:

        with self.db_config.main_session() as session:
            try:
                area = session.query(Area).filter_by(name=name).first()

                if area:
                    info_id(f"Found existing Area '{name}'", None)
                    return area

                area = Area(name=name, description=description)
                session.add(area)
                session.commit()
                info_id(f"Created new Area '{name}'", None)
                return area

            except SQLAlchemyError as e:
                error_id(f"AreaService.find_or_create failed: {e}", None)
                raise

    # ----------------------------------------------------------------------
    # FIND RELATED (NEW)
    # ----------------------------------------------------------------------
    @with_request_id
    def find_related(self, area_id: int) -> Optional[Dict[str, Any]]:
        """
        Return all related entities under this Area.

        Returns:
            {
                "area": Area,
                "downward": {
                    "equipment_groups": [...],
                    "positions": [...]
                }
            }
        """
        with self.db_config.main_session() as session:
            try:
                area = session.query(Area).filter_by(id=area_id).first()
                if not area:
                    return None

                downward = {
                    "equipment_groups": area.equipment_group,
                    "positions": area.position
                }

                return {"area": area, "downward": downward}

            except SQLAlchemyError as e:
                error_id(f"AreaService.find_related failed: {e}", None)
                raise
