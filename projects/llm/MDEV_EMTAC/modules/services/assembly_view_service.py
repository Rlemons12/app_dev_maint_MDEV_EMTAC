from typing import List, Optional, Dict, Any
from sqlalchemy.orm import Session
from sqlalchemy.exc import SQLAlchemyError

from modules.configuration.log_config import warning_id, with_request_id
from modules.emtacdb.emtacdb_fts import AssemblyView
from modules.configuration.config_env import DatabaseConfig


class AssemblyViewService:
    """
    Transaction-aware AssemblyView service.

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
        component_assembly_id: Optional[int] = None,
        asset_number_id: Optional[int] = None,
        limit: int = 100,
        session: Optional[Session] = None,
    ) -> List[AssemblyView]:

        if session:
            return self._find_internal(
                session, name, component_assembly_id, asset_number_id, limit
            )

        with self.db_config.main_session() as s:
            return self._find_internal(
                s, name, component_assembly_id, asset_number_id, limit
            )

    def _find_internal(
        self, session, name, component_assembly_id, asset_number_id, limit
    ):
        query = session.query(AssemblyView)

        if name:
            query = query.filter(AssemblyView.name.ilike(f"%{name}%"))

        if component_assembly_id:
            query = query.filter(
                AssemblyView.component_assembly_id == component_assembly_id
            )

        if asset_number_id:
            query = query.filter(
                AssemblyView.asset_number_id == asset_number_id
            )

        return query.limit(limit).all()

    # -----------------------------------------------------
    # GET
    # -----------------------------------------------------

    @with_request_id
    def get(
        self,
        assembly_view_id: int,
        session: Optional[Session] = None,
    ) -> Optional[AssemblyView]:

        if session:
            return session.get(AssemblyView, assembly_view_id)

        with self.db_config.main_session() as s:
            return s.get(AssemblyView, assembly_view_id)

    # -----------------------------------------------------
    # SAVE
    # -----------------------------------------------------

    @with_request_id
    def save(
        self,
        name: str,
        component_assembly_id: int,
        description: Optional[str] = None,
        asset_number_id: Optional[int] = None,
        assembly_view_id: Optional[int] = None,
        session: Optional[Session] = None,
    ) -> AssemblyView:

        if session:
            return self._save_internal(
                session, name, component_assembly_id,
                description, asset_number_id, assembly_view_id
            )

        with self.db_config.main_session() as s:
            try:
                av = self._save_internal(
                    s, name, component_assembly_id,
                    description, asset_number_id, assembly_view_id
                )
                s.commit()
                return av
            except:
                s.rollback()
                raise

    def _save_internal(
        self, session, name, component_assembly_id,
        description, asset_number_id, assembly_view_id
    ):

        if assembly_view_id:
            av = session.get(AssemblyView, assembly_view_id)
            if not av:
                raise ValueError(
                    f"AssemblyView with id {assembly_view_id} not found"
                )

            av.name = name
            av.description = description
            av.component_assembly_id = component_assembly_id
            av.asset_number_id = asset_number_id
        else:
            av = AssemblyView(
                name=name,
                description=description,
                component_assembly_id=component_assembly_id,
                asset_number_id=asset_number_id,
            )
            session.add(av)

        session.flush()
        return av

    # -----------------------------------------------------
    # REMOVE
    # -----------------------------------------------------

    @with_request_id
    def remove(
        self,
        assembly_view_id: int,
        session: Optional[Session] = None,
    ) -> bool:

        if session:
            return self._remove_internal(session, assembly_view_id)

        with self.db_config.main_session() as s:
            try:
                result = self._remove_internal(s, assembly_view_id)
                s.commit()
                return result
            except:
                s.rollback()
                raise

    def _remove_internal(self, session, assembly_view_id):

        av = session.get(AssemblyView, assembly_view_id)
        if not av:
            return False

        if av.position:
            warning_id(
                f"Cannot delete AssemblyView id={assembly_view_id} "
                f"because {len(av.position)} Positions depend on it",
                None,
            )
            return False

        session.delete(av)
        return True

    # -----------------------------------------------------
    # FIND OR CREATE
    # -----------------------------------------------------

    @with_request_id
    def find_or_create(
        self,
        name: str,
        component_assembly_id: int,
        asset_number_id: Optional[int] = None,
        description: Optional[str] = None,
        session: Optional[Session] = None,
    ) -> AssemblyView:

        if session:
            return self._find_or_create_internal(
                session, name, component_assembly_id,
                asset_number_id, description
            )

        with self.db_config.main_session() as s:
            try:
                av = self._find_or_create_internal(
                    s, name, component_assembly_id,
                    asset_number_id, description
                )
                s.commit()
                return av
            except:
                s.rollback()
                raise

    def _find_or_create_internal(
        self, session, name, component_assembly_id,
        asset_number_id, description
    ):

        av = (
            session.query(AssemblyView)
            .filter_by(
                name=name,
                component_assembly_id=component_assembly_id,
                asset_number_id=asset_number_id,
            )
            .first()
        )

        if av:
            return av

        av = AssemblyView(
            name=name,
            description=description,
            component_assembly_id=component_assembly_id,
            asset_number_id=asset_number_id,
        )

        session.add(av)
        session.flush()
        return av