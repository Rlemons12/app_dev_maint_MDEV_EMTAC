# services/model_service.py
from typing import List, Optional, Dict, Any, Union
from sqlalchemy.exc import SQLAlchemyError

from modules.emtacdb.emtacdb_fts import Model, EquipmentGroup, Area
from modules.configuration.log_config import info_id, error_id, with_request_id
from modules.database.config_env import DatabaseConfig


class ModelService:
    """
    Service layer for managing Model entities.

    Provides CRUD operations and relationship traversal:
      - `find`         → Broad search by name/description.
      - `get`          → Retrieve a Model by ID.
      - `save`         → Create or update a Model.
      - `remove`       → Delete a Model by ID.
      - `search`       → Autocomplete search for models.
      - `find_related` → Get related entities (upward/downward hierarchy).
    """

    def __init__(self, db_config: DatabaseConfig = None):
        self.db_config = db_config or DatabaseConfig()

    # -----------------
    # SEARCH OPERATIONS
    # -----------------

    @with_request_id
    def find(self, name: Optional[str] = None,
             description: Optional[str] = None,
             equipment_group_id: Optional[int] = None,
             limit: int = 100) -> List[Model]:
        """
        Broad search for models by filters.

        Args:
            name (str, optional): Partial match on model name.
            description (str, optional): Partial match on description.
            equipment_group_id (int, optional): Filter by equipment group.
            limit (int): Max results (default 100).

        Returns:
            List[Model]: Matching model objects.
        """
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
                info_id(f"Found {len(results)} Models", None)
                return results
            except SQLAlchemyError as e:
                error_id(f"ModelService.find failed: {e}", None)
                raise

    @with_request_id
    def search(self, query: str, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Autocomplete search for models.

        Args:
            query (str): Partial name of model.
            limit (int): Max results (default 10).

        Returns:
            List[dict]: Lightweight model details for autocomplete.
        """
        with self.db_config.main_session() as session:
            try:
                return Model.search_models(session, query, limit=limit)
            except SQLAlchemyError as e:
                error_id(f"ModelService.search failed: {e}", None)
                raise

    # -----------------
    # CRUD OPERATIONS
    # -----------------

    @with_request_id
    def get(self, model_id: int) -> Optional[Model]:
        """Retrieve a Model by ID."""
        with self.db_config.main_session() as session:
            try:
                return session.query(Model).filter_by(id=model_id).first()
            except SQLAlchemyError as e:
                error_id(f"ModelService.get failed: {e}", None)
                raise

    @with_request_id
    def save(self, name: str, equipment_group_id: int,
             description: Optional[str] = None,
             model_id: Optional[int] = None) -> Model:
        """
        Create or update a Model.

        Args:
            name (str): Model name.
            equipment_group_id (int): FK to equipment group.
            description (str, optional): Model description.
            model_id (int, optional): If provided, update existing record.

        Returns:
            Model: Created or updated object.
        """
        with self.db_config.main_session() as session:
            try:
                if model_id:
                    m = session.query(Model).filter_by(id=model_id).first()
                    if not m:
                        raise ValueError(f"Model with id {model_id} not found")
                    m.name = name
                    m.description = description
                    m.equipment_group_id = equipment_group_id
                    info_id(f"Updated Model id={model_id}", None)
                else:
                    m = Model.add_model(session, name, equipment_group_id, description)
                    info_id(f"Created Model '{name}'", None)
                return m
            except SQLAlchemyError as e:
                error_id(f"ModelService.save failed: {e}", None)
                raise

    @with_request_id
    def remove(self, model_id: int) -> bool:
        """Delete a Model by ID."""
        with self.db_config.main_session() as session:
            try:
                return Model.delete_model(session, model_id)
            except SQLAlchemyError as e:
                error_id(f"ModelService.remove failed: {e}", None)
                raise

    # -----------------
    # RELATIONSHIP OPS
    # -----------------

    @with_request_id
    def find_related(self, identifier: Union[int, str], is_id: bool = True) -> Optional[Dict[str, Any]]:
        """
        Traverse hierarchy for a Model.

        Upward:
          - EquipmentGroup
          - Area

        Downward:
          - AssetNumbers
          - Locations
          - Positions
        """
        with self.db_config.main_session() as session:
            try:
                return Model.find_related_entities(session, identifier, is_id=is_id)
            except SQLAlchemyError as e:
                error_id(f"ModelService.find_related failed: {e}", None)
                raise

