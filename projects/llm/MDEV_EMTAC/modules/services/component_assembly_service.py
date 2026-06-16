# modules/services/component_assembly_service.py

from typing import List, Optional, Dict, Any, Union
from sqlalchemy.orm import Session
from sqlalchemy.exc import SQLAlchemyError

from modules.emtacdb.emtacdb_fts import ComponentAssembly
from modules.configuration.config_env import DatabaseConfig
from modules.configuration.log_config import with_request_id


class ComponentAssemblyService:
    """
    Transaction-aware ComponentAssembly service.

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
        subassembly_id: Optional[int] = None,
        limit: int = 100,
        session: Optional[Session] = None,
    ) -> List[ComponentAssembly]:

        if session:
            return self._find_internal(session, name, subassembly_id, limit)

        with self.db_config.main_session() as s:
            return self._find_internal(s, name, subassembly_id, limit)

    def _find_internal(self, session, name, subassembly_id, limit):

        query = session.query(ComponentAssembly)

        if name:
            query = query.filter(ComponentAssembly.name.ilike(f"%{name}%"))

        if subassembly_id:
            query = query.filter(ComponentAssembly.subassembly_id == subassembly_id)

        return query.limit(limit).all()

    # -----------------------------------------------------
    # GET
    # -----------------------------------------------------

    @with_request_id
    def get(
        self,
        component_assembly_id: int,
        session: Optional[Session] = None,
    ) -> Optional[ComponentAssembly]:

        if session:
            return session.get(ComponentAssembly, component_assembly_id)

        with self.db_config.main_session() as s:
            return s.get(ComponentAssembly, component_assembly_id)

    # -----------------------------------------------------
    # SAVE
    # -----------------------------------------------------

    @with_request_id
    def save(
        self,
        name: str,
        subassembly_id: int,
        description: Optional[str] = None,
        component_assembly_id: Optional[int] = None,
        session: Optional[Session] = None,
    ) -> ComponentAssembly:

        if session:
            return self._save_internal(
                session,
                name,
                subassembly_id,
                description,
                component_assembly_id,
            )

        with self.db_config.main_session() as s:
            try:
                ca = self._save_internal(
                    s,
                    name,
                    subassembly_id,
                    description,
                    component_assembly_id,
                )
                s.commit()
                return ca
            except:
                s.rollback()
                raise

    def _save_internal(
        self,
        session,
        name,
        subassembly_id,
        description,
        component_assembly_id,
    ):

        if component_assembly_id:
            ca = session.get(ComponentAssembly, component_assembly_id)
            if not ca:
                raise ValueError(
                    f"ComponentAssembly id={component_assembly_id} not found"
                )

            ca.name = name
            ca.description = description
            ca.subassembly_id = subassembly_id

        else:
            ca = ComponentAssembly(
                name=name,
                description=description,
                subassembly_id=subassembly_id,
            )
            session.add(ca)

        session.flush()
        return ca

    # -----------------------------------------------------
    # REMOVE (SAFE DELETE)
    # -----------------------------------------------------

    @with_request_id
    def remove(
        self,
        component_assembly_id: int,
        session: Optional[Session] = None,
    ) -> bool:

        if session:
            return self._remove_internal(session, component_assembly_id)

        with self.db_config.main_session() as s:
            try:
                result = self._remove_internal(s, component_assembly_id)
                s.commit()
                return result
            except:
                s.rollback()
                raise

    def _remove_internal(self, session, component_assembly_id):

        ca = session.get(ComponentAssembly, component_assembly_id)
        if not ca:
            return False

        # Safe delete guard
        if ca.position or ca.assembly_view:
            return False

        session.delete(ca)
        return True

    # -----------------------------------------------------
    # FIND OR CREATE
    # -----------------------------------------------------

    @with_request_id
    def find_or_create(
        self,
        name: str,
        subassembly_id: int,
        description: Optional[str] = None,
        session: Optional[Session] = None,
    ) -> ComponentAssembly:

        if session:
            return self._find_or_create_internal(
                session,
                name,
                subassembly_id,
                description,
            )

        with self.db_config.main_session() as s:
            try:
                ca = self._find_or_create_internal(
                    s,
                    name,
                    subassembly_id,
                    description,
                )
                s.commit()
                return ca
            except:
                s.rollback()
                raise

    def _find_or_create_internal(
        self,
        session,
        name,
        subassembly_id,
        description,
    ):

        ca = (
            session.query(ComponentAssembly)
            .filter_by(name=name, subassembly_id=subassembly_id)
            .first()
        )

        if ca:
            return ca

        ca = ComponentAssembly(
            name=name,
            description=description,
            subassembly_id=subassembly_id,
        )

        session.add(ca)
        session.flush()
        return ca

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
            ca = session.get(ComponentAssembly, identifier)
        else:
            ca = session.query(ComponentAssembly).filter_by(name=identifier).first()

        if not ca:
            return None

        return {
            "component_assembly": ca,
            "upward": {
                "subassembly": ca.subassembly,
            },
            "downward": {
                "assembly_views": ca.assembly_view,
                "positions": ca.position,
            },
        }