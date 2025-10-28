# services/equipment_group_service.py
from typing import Optional, List, Union, Dict, Any
from sqlalchemy.exc import SQLAlchemyError
from modules.emtacdb.emtacdb_fts import EquipmentGroup
from modules.configuration.config_env import DatabaseConfig
from modules.configuration.log_config import info_id, error_id, with_request_id


class EquipmentGroupService:
    """Service layer for managing EquipmentGroup entities with a slim API."""

    def __init__(self, db_config: DatabaseConfig = None):
        # Initialize with a DatabaseConfig (fallback to default if not provided)
        self.db_config = db_config or DatabaseConfig()

    @with_request_id
    def find(self, name: Optional[str] = None, area_id: Optional[int] = None) -> List[EquipmentGroup]:
        """
        Search for EquipmentGroups with flexible filters.
        - If `name` is provided → returns groups whose name contains it (case-insensitive).
        - If `area_id` is provided → returns groups belonging to that area.
        Returns a list of matching EquipmentGroup objects.
        """
        with self.db_config.main_session() as session:
            try:
                query = session.query(EquipmentGroup)
                if name:
                    query = query.filter(EquipmentGroup.name.ilike(f"%{name}%"))
                if area_id:
                    query = query.filter_by(area_id=area_id)
                results = query.all()
                info_id(f"Found {len(results)} EquipmentGroups", None)
                return results
            except SQLAlchemyError as e:
                error_id(f"EquipmentGroupService.find failed: {e}", None)
                raise

    @with_request_id
    def get(self, equipment_group_id: int) -> Optional[EquipmentGroup]:
        """
        Retrieve a single EquipmentGroup by its ID.
        Returns the EquipmentGroup object if found, otherwise None.
        """
        with self.db_config.main_session() as session:
            try:
                return session.query(EquipmentGroup).filter_by(id=equipment_group_id).first()
            except SQLAlchemyError as e:
                error_id(f"EquipmentGroupService.get failed: {e}", None)
                raise

    @with_request_id
    def save(self, name: str, area_id: int, description: Optional[str] = None,
             equipment_group_id: Optional[int] = None) -> EquipmentGroup:
        """
        Create or update an EquipmentGroup.
        - If `equipment_group_id` is given → updates that group with new values.
        - If not → creates a new EquipmentGroup with the given name, area_id, and description.
        Returns the created or updated EquipmentGroup object.
        """
        with self.db_config.main_session() as session:
            try:
                if equipment_group_id:
                    group = session.query(EquipmentGroup).filter_by(id=equipment_group_id).first()
                    if not group:
                        raise ValueError(f"EquipmentGroup with id {equipment_group_id} not found")
                    group.name = name
                    group.area_id = area_id
                    group.description = description
                    info_id(f"Updated EquipmentGroup id={equipment_group_id}", None)
                else:
                    group = EquipmentGroup(name=name, area_id=area_id, description=description)
                    session.add(group)
                    info_id(f"Created EquipmentGroup '{name}'", None)
                return group
            except SQLAlchemyError as e:
                error_id(f"EquipmentGroupService.save failed: {e}", None)
                raise

    @with_request_id
    def remove(self, equipment_group_id: int) -> bool:
        """
        Delete an EquipmentGroup by its ID.
        Returns True if deletion succeeded, False if no such group exists.
        """
        with self.db_config.main_session() as session:
            try:
                group = session.query(EquipmentGroup).filter_by(id=equipment_group_id).first()
                if group:
                    session.delete(group)
                    info_id(f"Deleted EquipmentGroup id={equipment_group_id}", None)
                    return True
                return False
            except SQLAlchemyError as e:
                error_id(f"EquipmentGroupService.remove failed: {e}", None)
                raise

    @with_request_id
    def find_or_create(self, name: str, area_id: int, description: Optional[str] = None) -> EquipmentGroup:
        """
        Find an EquipmentGroup by name + area, or create it if missing.

        Args:
            name (str): Name of the equipment group.
            area_id (int): ID of the area this group belongs to.
            description (str, optional): Description of the group.

        Returns:
            EquipmentGroup: Existing or newly created equipment group.
        """
        with self.db_config.main_session() as session:
            try:
                group = (
                    session.query(EquipmentGroup)
                    .filter_by(name=name, area_id=area_id)
                    .first()
                )
                if group:
                    info_id(f"Found existing EquipmentGroup '{name}' in area_id={area_id}", None)
                else:
                    group = EquipmentGroup(name=name, area_id=area_id, description=description)
                    session.add(group)
                    session.commit()
                    info_id(f"Created new EquipmentGroup '{name}' in area_id={area_id}", None)
                return group
            except SQLAlchemyError as e:
                error_id(f"EquipmentGroupService.find_or_create failed: {e}", None)
                raise
