# services/asset_number_service.py
from typing import Optional, List, Union, Dict, Any
from sqlalchemy.exc import SQLAlchemyError
from modules.emtacdb.emtacdb_fts import AssetNumber
from modules.configuration.config_env import DatabaseConfig
from modules.configuration.log_config import info_id, error_id, with_request_id


class AssetNumberService:
    """
    Service layer for managing AssetNumber entities.

    Provides:
      - `find`            → Search AssetNumbers by number/description.
      - `get`             → Retrieve AssetNumber by ID.
      - `save`            → Create or update an AssetNumber.
      - `remove`          → Delete an AssetNumber by ID.
      - `find_related`    → Get related entities (upward/downward).
      - Wrappers for ORM helpers:
          get_ids_by_number,
          get_model_id,
          get_equipment_group_id,
          get_area_id,
          get_position_ids,
          search_numbers
    """

    def __init__(self, db_config: DatabaseConfig = None):
        self.db_config = db_config or DatabaseConfig()

    # ------------------------
    # BASIC CRUD
    # ------------------------

    @with_request_id
    def save(self, number: str, model_id: int, description: Optional[str] = None,
             asset_number_id: Optional[int] = None) -> AssetNumber:
        """
        Create or update an AssetNumber.

        Args:
            number (str): Asset number string.
            model_id (int): FK to Model.
            description (str, optional): Description.
            asset_number_id (int, optional): Existing record ID (update).

        Returns:
            AssetNumber: Created or updated object.
        """
        with self.db_config.main_session() as session:
            try:
                if asset_number_id:
                    asset = session.query(AssetNumber).filter_by(id=asset_number_id).first()
                    if not asset:
                        raise ValueError(f"AssetNumber id={asset_number_id} not found")
                    asset.number = number
                    asset.model_id = model_id
                    asset.description = description
                    info_id(f"Updated AssetNumber id={asset_number_id}", None)
                else:
                    asset = AssetNumber.add_asset_number(
                        session, number=number, model_id=model_id, description=description
                    )
                    info_id(f"Created AssetNumber '{number}'", None)
                return asset
            except SQLAlchemyError as e:
                error_id(f"AssetNumberService.save failed: {e}", None)
                raise

    @with_request_id
    def remove(self, asset_number_id: int) -> bool:
        """
        Delete an AssetNumber by ID.

        Args:
            asset_number_id (int): ID of the AssetNumber.

        Returns:
            bool: True if deleted, False if not found.
        """
        with self.db_config.main_session() as session:
            try:
                return AssetNumber.delete_asset_number(session, asset_number_id)
            except SQLAlchemyError as e:
                error_id(f"AssetNumberService.remove failed: {e}", None)
                raise

    @with_request_id
    def find_related(self, identifier: Union[int, str], is_id: bool = True) -> Optional[Dict[str, Any]]:
        """
        Find related entities for an AssetNumber.

        Traverses upward (Model, EquipmentGroup, Area) and downward (Positions).

        Args:
            identifier (int | str): Either ID or number.
            is_id (bool): If True → identifier is treated as ID, else as number.

        Returns:
            dict | None
        """
        with self.db_config.main_session() as session:
            try:
                return AssetNumber.find_related_entities(session, identifier, is_id=is_id)
            except SQLAlchemyError as e:
                error_id(f"AssetNumberService.find_related failed: {e}", None)
                raise

    # ------------------------
    # EXISTING WRAPPERS
    # ------------------------

    def get_ids_by_number(self, number: str) -> List[int]:
        with self.db_config.main_session() as session:
            return AssetNumber.get_ids_by_number(session, number)

    def get_model_id(self, asset_number_id: int) -> Optional[int]:
        with self.db_config.main_session() as session:
            return AssetNumber.get_model_id_by_asset_number_id(session, asset_number_id)

    def get_equipment_group_id(self, asset_number_id: int) -> Optional[int]:
        with self.db_config.main_session() as session:
            return AssetNumber.get_equipment_group_id_by_asset_number_id(session, asset_number_id)

    def get_area_id(self, asset_number_id: int) -> Optional[int]:
        with self.db_config.main_session() as session:
            return AssetNumber.get_area_id_by_asset_number_id(session, asset_number_id)

    def get_position_ids(self, asset_number_id: int) -> List[int]:
        with self.db_config.main_session() as session:
            return AssetNumber.get_position_ids_by_asset_number_id(session, asset_number_id)

    def search_numbers(self, query: str, limit: int = 10) -> List[Dict[str, Any]]:
        with self.db_config.main_session() as session:
            return AssetNumber.search_asset_numbers(session, query, limit=limit)

