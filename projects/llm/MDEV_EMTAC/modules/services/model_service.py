# modules/services/model_service.py

from typing import List, Optional, Dict, Any, Union
from sqlalchemy.orm import Session
from sqlalchemy.exc import SQLAlchemyError

from modules.emtacdb.emtacdb_fts import Model
from modules.configuration.log_config import with_request_id
from modules.configuration.config_env import DatabaseConfig


class ModelService:
    """
    Transaction-aware Model service.

    RULE:
    - If session provided → NEVER commit
    - If session not provided → standalone mode allowed
    """

    def __init__(self, db_config: Optional[DatabaseConfig] = None):
        self.db_config = db_config or DatabaseConfig()

    # -----------------------------------------------------
    # FIND
    # -----------------------------------------------------

    @with_request_id
    def find(
        self,
        name: Optional[str] = None,
        description: Optional[str] = None,
        equipment_group_id: Optional[int] = None,
        limit: int = 100,
        session: Optional[Session] = None,
    ) -> List[Model]:

        if session:
            return self._find_internal(
                session, name, description, equipment_group_id, limit
            )

        with self.db_config.main_session() as s:
            return self._find_internal(
                s, name, description, equipment_group_id, limit
            )

    def _find_internal(
        self, session, name, description, equipment_group_id, limit
    ):

        query = session.query(Model)

        if name:
            query = query.filter(Model.name.ilike(f"%{name}%"))

        if description:
            query = query.filter(Model.description.ilike(f"%{description}%"))

        if equipment_group_id:
            query = query.filter(Model.equipment_group_id == equipment_group_id)

        return query.limit(limit).all()

    # -----------------------------------------------------
    # SEARCH (Autocomplete)
    # -----------------------------------------------------

    @with_request_id
    def search(
        self,
        query: str,
        limit: int = 10,
        session: Optional[Session] = None,
    ) -> List[Dict[str, Any]]:

        if session:
            return Model.search_models(session, query, limit)

        with self.db_config.main_session() as s:
            return Model.search_models(s, query, limit)

    # -----------------------------------------------------
    # GET
    # -----------------------------------------------------

    @with_request_id
    def get(
        self,
        model_id: int,
        session: Optional[Session] = None,
    ) -> Optional[Model]:

        if session:
            return session.get(Model, model_id)

        with self.db_config.main_session() as s:
            return s.get(Model, model_id)

    # -----------------------------------------------------
    # SAVE
    # -----------------------------------------------------

    @with_request_id
    def save(
        self,
        name: str,
        equipment_group_id: int,
        description: Optional[str] = None,
        model_id: Optional[int] = None,
        session: Optional[Session] = None,
    ) -> Model:

        if session:
            return self._save_internal(
                session, name, equipment_group_id, description, model_id
            )

        with self.db_config.main_session() as s:
            try:
                model = self._save_internal(
                    s, name, equipment_group_id, description, model_id
                )
                s.commit()
                return model
            except:
                s.rollback()
                raise

    def _save_internal(
        self, session, name, equipment_group_id, description, model_id
    ):

        if model_id:
            model = session.get(Model, model_id)
            if not model:
                raise ValueError(f"Model id={model_id} not found")

            model.name = name
            model.description = description
            model.equipment_group_id = equipment_group_id

        else:
            model = Model(
                name=name,
                equipment_group_id=equipment_group_id,
                description=description,
            )
            session.add(model)

        session.flush()
        return model

    # -----------------------------------------------------
    # REMOVE
    # -----------------------------------------------------

    @with_request_id
    def remove(
        self,
        model_id: int,
        session: Optional[Session] = None,
    ) -> bool:

        if session:
            return self._remove_internal(session, model_id)

        with self.db_config.main_session() as s:
            try:
                result = self._remove_internal(s, model_id)
                s.commit()
                return result
            except:
                s.rollback()
                raise

    def _remove_internal(self, session, model_id):

        model = session.get(Model, model_id)
        if not model:
            return False

        session.delete(model)
        return True

    # -----------------------------------------------------
    # FIND OR CREATE
    # -----------------------------------------------------

    @with_request_id
    def find_or_create(
        self,
        name: str,
        equipment_group_id: int,
        description: Optional[str] = None,
        session: Optional[Session] = None,
    ) -> Model:

        if session:
            return self._find_or_create_internal(
                session, name, equipment_group_id, description
            )

        with self.db_config.main_session() as s:
            try:
                model = self._find_or_create_internal(
                    s, name, equipment_group_id, description
                )
                s.commit()
                return model
            except:
                s.rollback()
                raise

    def _find_or_create_internal(
        self, session, name, equipment_group_id, description
    ):

        model = (
            session.query(Model)
            .filter_by(name=name, equipment_group_id=equipment_group_id)
            .first()
        )

        if model:
            return model

        model = Model(
            name=name,
            equipment_group_id=equipment_group_id,
            description=description,
        )

        session.add(model)
        session.flush()
        return model

    # -----------------------------------------------------
    # FIND RELATED
    # -----------------------------------------------------

    @with_request_id
    def find_related(
        self,
        identifier: Union[int, str],
        is_id: bool = True,
        session: Optional[Session] = None,
    ):

        if session:
            return Model.find_related_entities(
                session=session,
                identifier=identifier,
                is_id=is_id,
            )

        with self.db_config.main_session() as s:
            return Model.find_related_entities(
                session=s,
                identifier=identifier,
                is_id=is_id,
            )