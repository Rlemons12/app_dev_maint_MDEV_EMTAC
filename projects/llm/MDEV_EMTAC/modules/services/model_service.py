# services/model_service.py

from typing import List, Optional, Dict, Any, Union
from sqlalchemy.exc import SQLAlchemyError

from modules.emtacdb.emtacdb_fts import Model
from modules.configuration.log_config import (
    info_id, error_id, debug_id, with_request_id
)
from modules.configuration.config_env import DatabaseConfig


class ModelService:
    """Service layer for Model entities."""

    def __init__(self, db_config: Optional[DatabaseConfig] = None):
        self.db_config = db_config or DatabaseConfig()

    # -----------------------------------------------------
    # FIND
    # -----------------------------------------------------
    @with_request_id
    def find(self,
             name: Optional[str] = None,
             description: Optional[str] = None,
             equipment_group_id: Optional[int] = None,
             limit: int = 100,
             request_id: Optional[str] = None) -> List[Model]:

        with self.db_config.main_session() as session:
            try:
                query = session.query(Model)

                if name:
                    query = query.filter(Model.name.ilike(f"%{name}%"))

                if description:
                    query = query.filter(Model.description.ilike(f"%{description}%"))

                if equipment_group_id:
                    query = query.filter(Model.equipment_group_id == equipment_group_id)

                results = query.limit(limit).all()
                info_id(f"ModelService.find → {len(results)} results", request_id)
                return results

            except SQLAlchemyError as e:
                error_id(f"ModelService.find failed: {e}", request_id)
                raise

    # -----------------------------------------------------
    # AUTOCOMPLETE SEARCH
    # -----------------------------------------------------
    @with_request_id
    def search(self,
               query: str,
               limit: int = 10,
               request_id: Optional[str] = None) -> List[Dict[str, Any]]:

        with self.db_config.main_session() as session:
            try:
                debug_id(f"ModelService.search query='{query}'", request_id)
                return Model.search_models(session, query, limit)

            except SQLAlchemyError as e:
                error_id(f"ModelService.search failed: {e}", request_id)
                raise

    # -----------------------------------------------------
    # GET
    # -----------------------------------------------------
    @with_request_id
    def get(self,
            model_id: int,
            request_id: Optional[str] = None) -> Optional[Model]:

        with self.db_config.main_session() as session:
            try:
                model = session.query(Model).filter_by(id=model_id).first()
                debug_id(f"ModelService.get returned {model}", request_id)
                return model

            except SQLAlchemyError as e:
                error_id(f"ModelService.get failed: {e}", request_id)
                raise

    # -----------------------------------------------------
    # SAVE (CREATE + UPDATE)
    # -----------------------------------------------------
    @with_request_id
    def save(self,
             name: str,
             equipment_group_id: int,
             description: Optional[str] = None,
             model_id: Optional[int] = None,
             request_id: Optional[str] = None) -> Model:

        with self.db_config.main_session() as session:
            try:
                if model_id:
                    # UPDATE
                    m = session.query(Model).filter_by(id=model_id).first()
                    if not m:
                        raise ValueError(f"Model id={model_id} not found")

                    m.name = name
                    m.description = description
                    m.equipment_group_id = equipment_group_id

                    session.commit()
                    info_id(f"Updated Model id={model_id}", request_id)
                    return m

                # CREATE
                m = Model.add_model(
                    session=session,
                    name=name,
                    equipment_group_id=equipment_group_id,
                    description=description
                )
                session.commit()

                info_id(f"Created Model '{name}'", request_id)
                return m

            except SQLAlchemyError as e:
                session.rollback()
                error_id(f"ModelService.save failed: {e}", request_id)
                raise

    # -----------------------------------------------------
    # REMOVE
    # -----------------------------------------------------
    @with_request_id
    def remove(self, model_id: int, request_id=None) -> bool:

        with self.db_config.main_session() as session:
            try:
                removed = Model.delete_model(session, model_id)
                session.commit()

                if removed:
                    info_id(f"Deleted Model id={model_id}", request_id)

                return removed

            except SQLAlchemyError as e:
                session.rollback()
                error_id(f"ModelService.remove failed: {e}", request_id)
                raise

    # -----------------------------------------------------
    # RELATIONSHIPS
    # -----------------------------------------------------
    @with_request_id
    def find_related(self,
                     identifier: Union[int, str],
                     is_id: bool = True,
                     request_id: Optional[str] = None):

        with self.db_config.main_session() as session:
            try:
                data = Model.find_related_entities(
                    session=session,
                    identifier=identifier,
                    is_id=is_id,
                    request_id=request_id
                )
                debug_id(f"ModelService.find_related returned {data}", request_id)
                return data

            except SQLAlchemyError as e:
                error_id(f"ModelService.find_related failed: {e}", request_id)
                raise
