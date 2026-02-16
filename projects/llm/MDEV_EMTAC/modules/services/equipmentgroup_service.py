# modules/services/equipment_group_service.py

from typing import Optional, List
from sqlalchemy.exc import SQLAlchemyError
from modules.emtacdb.emtacdb_fts import EquipmentGroup
from modules.configuration.config_env import DatabaseConfig
from modules.configuration.log_config import (
    info_id, error_id, with_request_id
)


class EquipmentGroupService:
    """Service layer for managing EquipmentGroup entities."""

    def __init__(self, db_config: Optional[DatabaseConfig] = None):
        self.db_config = db_config or DatabaseConfig()

    # ------------------------------------------------------------
    # FIND / SEARCH
    # ------------------------------------------------------------
    @with_request_id
    def find(self,
             name: Optional[str] = None,
             area_id: Optional[int] = None,
             request_id: Optional[str] = None) -> List[EquipmentGroup]:
        """
        Search for EquipmentGroups by name and/or area_id.
        Model remains unchanged.
        """
        with self.db_config.main_session() as session:
            try:
                query = session.query(EquipmentGroup)

                if name:
                    query = query.filter(EquipmentGroup.name.ilike(f"%{name}%"))

                if area_id:
                    query = query.filter(EquipmentGroup.area_id == area_id)

                results = query.all()
                info_id(f"EquipmentGroupService.find → {len(results)} results", request_id)
                return results

            except SQLAlchemyError as e:
                error_id(f"EquipmentGroupService.find failed: {e}", request_id)
                raise

    # ------------------------------------------------------------
    # GET BY ID
    # ------------------------------------------------------------
    @with_request_id
    def get(self, equipment_group_id: int,
            request_id: Optional[str] = None) -> Optional[EquipmentGroup]:

        with self.db_config.main_session() as session:
            try:
                return (
                    session.query(EquipmentGroup)
                    .filter_by(id=equipment_group_id)
                    .first()
                )
            except SQLAlchemyError as e:
                error_id(f"EquipmentGroupService.get failed: {e}", request_id)
                raise

    # ------------------------------------------------------------
    # CREATE / UPDATE
    # ------------------------------------------------------------
    @with_request_id
    def save(self,
             name: str,
             area_id: int,
             description: Optional[str] = None,
             equipment_group_id: Optional[int] = None,
             request_id: Optional[str] = None) -> EquipmentGroup:
        """
        Save or update WITHOUT editing the model.
        Uses raw SQLAlchemy session operations only.
        """
        with self.db_config.main_session() as session:
            try:
                if equipment_group_id:
                    # Update existing
                    group = (
                        session.query(EquipmentGroup)
                        .filter_by(id=equipment_group_id)
                        .first()
                    )
                    if not group:
                        raise ValueError(f"EquipmentGroup id={equipment_group_id} not found")

                    group.name = name
                    group.area_id = area_id
                    group.description = description
                    session.commit()

                    info_id(f"Updated EquipmentGroup id={equipment_group_id}", request_id)
                    return group

                # Create new
                group = EquipmentGroup(
                    name=name,
                    area_id=area_id,
                    description=description
                )
                session.add(group)
                session.commit()

                info_id(f"Created EquipmentGroup '{name}'", request_id)
                return group

            except SQLAlchemyError as e:
                session.rollback()
                error_id(f"EquipmentGroupService.save failed: {e}", request_id)
                raise

    # ------------------------------------------------------------
    # DELETE
    # ------------------------------------------------------------
    @with_request_id
    def remove(self,
               equipment_group_id: int,
               request_id: Optional[str] = None) -> bool:

        with self.db_config.main_session() as session:
            try:
                group = (
                    session.query(EquipmentGroup)
                    .filter_by(id=equipment_group_id)
                    .first()
                )
                if not group:
                    return False

                session.delete(group)
                session.commit()
                info_id(f"Deleted EquipmentGroup id={equipment_group_id}", request_id)
                return True

            except SQLAlchemyError as e:
                session.rollback()
                error_id(f"EquipmentGroupService.remove failed: {e}", request_id)
                raise

    # ------------------------------------------------------------
    # RELATIONSHIP TRAVERSAL
    # ------------------------------------------------------------
    @with_request_id
    def find_related(self,
                     identifier,
                     is_id: bool = True,
                     request_id: Optional[str] = None):
        """
        Calls the EXISTING model method without modifications.
        """
        with self.db_config.main_session() as session:
            try:
                return EquipmentGroup.find_related_entities(
                    session=session,
                    identifier=identifier,
                    is_id=is_id,
                    request_id=request_id
                )
            except SQLAlchemyError as e:
                error_id(f"EquipmentGroupService.find_related failed: {e}", request_id)
                raise
