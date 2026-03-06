from typing import Optional, List
from sqlalchemy.orm import Session
from sqlalchemy.exc import SQLAlchemyError

from modules.emtacdb.emtacdb_fts import EquipmentGroup
from modules.configuration.config_env import DatabaseConfig
from modules.configuration.log_config import with_request_id


class EquipmentGroupService:
    """
    Transaction-aware EquipmentGroup service.

    RULE:
    - If session provided → NEVER commit
    - If session not provided → standalone mode allowed
    """

    def __init__(self, db_config: Optional[DatabaseConfig] = None):
        self.db_config = db_config or DatabaseConfig()

    # ------------------------------------------------------------
    # FIND
    # ------------------------------------------------------------

    @with_request_id
    def find(
        self,
        name: Optional[str] = None,
        area_id: Optional[int] = None,
        session: Optional[Session] = None,
    ) -> List[EquipmentGroup]:

        if session:
            return self._find_internal(session, name, area_id)

        with self.db_config.main_session() as s:
            return self._find_internal(s, name, area_id)

    def _find_internal(self, session, name, area_id):

        query = session.query(EquipmentGroup)

        if name:
            query = query.filter(EquipmentGroup.name.ilike(f"%{name}%"))

        if area_id:
            query = query.filter(EquipmentGroup.area_id == area_id)

        return query.all()

    # ------------------------------------------------------------
    # GET
    # ------------------------------------------------------------

    @with_request_id
    def get(
        self,
        equipment_group_id: int,
        session: Optional[Session] = None,
    ) -> Optional[EquipmentGroup]:

        if session:
            return session.get(EquipmentGroup, equipment_group_id)

        with self.db_config.main_session() as s:
            return s.get(EquipmentGroup, equipment_group_id)

    # ------------------------------------------------------------
    # SAVE
    # ------------------------------------------------------------

    @with_request_id
    def save(
        self,
        name: str,
        area_id: int,
        description: Optional[str] = None,
        equipment_group_id: Optional[int] = None,
        session: Optional[Session] = None,
    ) -> EquipmentGroup:

        if session:
            return self._save_internal(
                session, name, area_id, description, equipment_group_id
            )

        with self.db_config.main_session() as s:
            try:
                group = self._save_internal(
                    s, name, area_id, description, equipment_group_id
                )
                s.commit()
                return group
            except:
                s.rollback()
                raise

    def _save_internal(
        self, session, name, area_id, description, equipment_group_id
    ):

        if equipment_group_id:
            group = session.get(EquipmentGroup, equipment_group_id)
            if not group:
                raise ValueError(
                    f"EquipmentGroup id={equipment_group_id} not found"
                )

            group.name = name
            group.area_id = area_id
            group.description = description

        else:
            group = EquipmentGroup(
                name=name,
                area_id=area_id,
                description=description,
            )
            session.add(group)

        session.flush()
        return group

    # ------------------------------------------------------------
    # REMOVE
    # ------------------------------------------------------------

    @with_request_id
    def remove(
        self,
        equipment_group_id: int,
        session: Optional[Session] = None,
    ) -> bool:

        if session:
            return self._remove_internal(session, equipment_group_id)

        with self.db_config.main_session() as s:
            try:
                result = self._remove_internal(s, equipment_group_id)
                s.commit()
                return result
            except:
                s.rollback()
                raise

    def _remove_internal(self, session, equipment_group_id):

        group = session.get(EquipmentGroup, equipment_group_id)
        if not group:
            return False

        session.delete(group)
        return True

    # ------------------------------------------------------------
    # FIND RELATED
    # ------------------------------------------------------------

    @with_request_id
    def find_related(
        self,
        identifier,
        is_id: bool = True,
        session: Optional[Session] = None,
    ):

        if session:
            return EquipmentGroup.find_related_entities(
                session=session,
                identifier=identifier,
                is_id=is_id,
            )

        with self.db_config.main_session() as s:
            return EquipmentGroup.find_related_entities(
                session=s,
                identifier=identifier,
                is_id=is_id,
            )

    # ------------------------------------------------------------
    # FIND OR CREATE
    # ------------------------------------------------------------

    @with_request_id
    def find_or_create(
        self,
        name: str,
        area_id: int,
        description: Optional[str] = None,
        session: Optional[Session] = None,
    ) -> EquipmentGroup:

        if session:
            return self._find_or_create_internal(
                session, name, area_id, description
            )

        with self.db_config.main_session() as s:
            try:
                group = self._find_or_create_internal(
                    s, name, area_id, description
                )
                s.commit()
                return group
            except:
                s.rollback()
                raise

    def _find_or_create_internal(
        self, session, name, area_id, description
    ):

        group = (
            session.query(EquipmentGroup)
            .filter_by(name=name, area_id=area_id)
            .first()
        )

        if group:
            return group

        group = EquipmentGroup(
            name=name,
            area_id=area_id,
            description=description,
        )

        session.add(group)
        session.flush()
        return group