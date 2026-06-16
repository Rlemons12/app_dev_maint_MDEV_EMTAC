# modules/services/subassembly_service.py

from typing import List, Optional, Dict, Any
from sqlalchemy.orm import Session
from sqlalchemy.exc import SQLAlchemyError

from modules.configuration.log_config import with_request_id
from modules.configuration.config_env import DatabaseConfig
from modules.emtacdb.emtacdb_fts import Subassembly


class SubassemblyService:
    """
    Transaction-aware Subassembly service.

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
        location_id: Optional[int] = None,
        limit: int = 100,
        session: Optional[Session] = None,
    ) -> List[Subassembly]:

        if session:
            return self._find_internal(session, name, location_id, limit)

        with self.db_config.main_session() as s:
            return self._find_internal(s, name, location_id, limit)

    def _find_internal(self, session, name, location_id, limit):

        query = session.query(Subassembly)

        if name:
            query = query.filter(Subassembly.name.ilike(f"%{name}%"))

        if location_id:
            query = query.filter(Subassembly.location_id == location_id)

        return query.limit(limit).all()

    # -----------------------------------------------------
    # GET
    # -----------------------------------------------------

    @with_request_id
    def get(
        self,
        subassembly_id: int,
        session: Optional[Session] = None,
    ) -> Optional[Subassembly]:

        if session:
            return session.get(Subassembly, subassembly_id)

        with self.db_config.main_session() as s:
            return s.get(Subassembly, subassembly_id)

    # -----------------------------------------------------
    # SAVE
    # -----------------------------------------------------

    @with_request_id
    def save(
        self,
        name: str,
        location_id: int,
        description: Optional[str] = None,
        subassembly_id: Optional[int] = None,
        session: Optional[Session] = None,
    ) -> Subassembly:

        if session:
            return self._save_internal(
                session, name, location_id, description, subassembly_id
            )

        with self.db_config.main_session() as s:
            try:
                sub = self._save_internal(
                    s, name, location_id, description, subassembly_id
                )
                s.commit()
                return sub
            except:
                s.rollback()
                raise

    def _save_internal(
        self, session, name, location_id, description, subassembly_id
    ):

        if subassembly_id:
            sub = session.get(Subassembly, subassembly_id)
            if not sub:
                raise ValueError(f"Subassembly id={subassembly_id} not found")

            sub.name = name
            sub.location_id = location_id
            sub.description = description
        else:
            sub = Subassembly(
                name=name,
                location_id=location_id,
                description=description,
            )
            session.add(sub)

        session.flush()
        return sub

    # -----------------------------------------------------
    # REMOVE
    # -----------------------------------------------------

    @with_request_id
    def remove(
        self,
        subassembly_id: int,
        session: Optional[Session] = None,
    ) -> bool:

        if session:
            return self._remove_internal(session, subassembly_id)

        with self.db_config.main_session() as s:
            try:
                result = self._remove_internal(s, subassembly_id)
                s.commit()
                return result
            except:
                s.rollback()
                raise

    def _remove_internal(self, session, subassembly_id):

        sub = session.get(Subassembly, subassembly_id)
        if not sub:
            return False

        session.delete(sub)
        return True

    # -----------------------------------------------------
    # FIND OR CREATE
    # -----------------------------------------------------

    @with_request_id
    def find_or_create(
        self,
        name: str,
        location_id: int,
        description: Optional[str] = None,
        session: Optional[Session] = None,
    ) -> Subassembly:

        if session:
            return self._find_or_create_internal(
                session, name, location_id, description
            )

        with self.db_config.main_session() as s:
            try:
                sub = self._find_or_create_internal(
                    s, name, location_id, description
                )
                s.commit()
                return sub
            except:
                s.rollback()
                raise

    def _find_or_create_internal(
        self, session, name, location_id, description
    ):

        sub = (
            session.query(Subassembly)
            .filter_by(name=name, location_id=location_id)
            .first()
        )

        if sub:
            return sub

        sub = Subassembly(
            name=name,
            location_id=location_id,
            description=description,
        )

        session.add(sub)
        session.flush()
        return sub

    # -----------------------------------------------------
    # FIND RELATED
    # -----------------------------------------------------

    @with_request_id
    def find_related(
        self,
        identifier: int,
        is_id: bool = True,
        session: Optional[Session] = None,
    ) -> Optional[Dict[str, Any]]:

        if session:
            return self._find_related_internal(session, identifier, is_id)

        with self.db_config.main_session() as s:
            return self._find_related_internal(s, identifier, is_id)

    def _find_related_internal(self, session, identifier, is_id):

        if is_id:
            sub = session.get(Subassembly, identifier)
        else:
            sub = session.query(Subassembly).filter_by(name=identifier).first()

        if not sub:
            return None

        upward = {"location": sub.location}
        downward = {
            "component_assemblies": sub.component_assembly,
            "positions": sub.position,
        }

        return {
            "subassembly": sub,
            "upward": upward,
            "downward": downward,
        }