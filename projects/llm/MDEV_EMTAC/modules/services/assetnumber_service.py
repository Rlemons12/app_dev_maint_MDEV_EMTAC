# services/asset_number_service.py
from typing import Optional, List, Union, Dict, Any
from sqlalchemy.exc import SQLAlchemyError

from modules.emtacdb.emtacdb_fts import AssetNumber
from modules.configuration.config_env import DatabaseConfig
from modules.configuration.log_config import info_id, warning_id, error_id, with_request_id


class AssetNumberService:
    """
    Service layer for managing AssetNumber entities.

    Responsibilities:
    ------------------
    CRUD:
        - find
        - get
        - save (create/update)
        - remove (safe delete)

    RELATIONSHIPS:
        - find_related → upward/downward traversal

    WRAPPERS around ORM helpers:
        - get_ids_by_number
        - get_model_id
        - get_equipment_group_id
        - get_area_id
        - get_position_ids
        - search_numbers
    """

    def __init__(self, db_config: DatabaseConfig = None):
        self.db_config = db_config or DatabaseConfig()

    # ----------------------------------------------------------------------
    # FIND
    # ----------------------------------------------------------------------
    @with_request_id
    def find(self,
             number: Optional[str] = None,
             description: Optional[str] = None,
             model_id: Optional[int] = None,
             limit: int = 100) -> List[AssetNumber]:

        with self.db_config.main_session() as session:
            try:
                query = session.query(AssetNumber)

                if number:
                    query = query.filter(AssetNumber.number.ilike(f"%{number}%"))
                if description:
                    query = query.filter(AssetNumber.description.ilike(f"%{description}%"))
                if model_id:
                    query = query.filter(AssetNumber.model_id == model_id)

                results = query.limit(limit).all()
                info_id(f"Found {len(results)} AssetNumbers", None)
                return results
            except SQLAlchemyError as e:
                error_id(f"AssetNumberService.find failed: {e}", None)
                raise

    # ----------------------------------------------------------------------
    # GET
    # ----------------------------------------------------------------------
    @with_request_id
    def get(self, asset_number_id: int) -> Optional[AssetNumber]:

        with self.db_config.main_session() as session:
            try:
                return session.query(AssetNumber).filter_by(id=asset_number_id).first()
            except SQLAlchemyError as e:
                error_id(f"AssetNumberService.get failed: {e}", None)
                raise

    # ----------------------------------------------------------------------
    # SAVE (CREATE / UPDATE)
    # ----------------------------------------------------------------------
    @with_request_id
    def save(self,
             number: str,
             model_id: int,
             description: Optional[str] = None,
             asset_number_id: Optional[int] = None) -> AssetNumber:

        with self.db_config.main_session() as session:
            try:
                if asset_number_id:
                    asset = session.query(AssetNumber).filter_by(id=asset_number_id).first()
                    if not asset:
                        raise ValueError(f"AssetNumber id={asset_number_id} not found")

                    asset.number = number
                    asset.model_id = model_id
                    asset.description = description

                    session.commit()
                    info_id(f"Updated AssetNumber id={asset_number_id}", None)

                else:
                    asset = AssetNumber(
                        number=number,
                        model_id=model_id,
                        description=description
                    )
                    session.add(asset)
                    session.commit()

                    info_id(f"Created new AssetNumber '{number}'", None)

                return asset

            except SQLAlchemyError as e:
                error_id(f"AssetNumberService.save failed: {e}", None)
                raise

    # ----------------------------------------------------------------------
    # SAFE DELETE
    # ----------------------------------------------------------------------
    @with_request_id
    def remove(self, asset_number_id: int) -> bool:
        """
        Prevent deletion if Positions reference this AssetNumber.
        """

        with self.db_config.main_session() as session:
            try:
                asset = session.query(AssetNumber).filter_by(id=asset_number_id).first()
                if not asset:
                    return False

                # SAFE DELETE CHECKS
                if asset.position and len(asset.position) > 0:
                    warning_id(
                        f"Cannot delete AssetNumber id={asset_number_id}: "
                        f"{len(asset.position)} Positions depend on it",
                        None
                    )
                    return False

                session.delete(asset)
                session.commit()
                info_id(f"Deleted AssetNumber id={asset_number_id}", None)
                return True

            except SQLAlchemyError as e:
                error_id(f"AssetNumberService.remove failed: {e}", None)
                raise

    # ----------------------------------------------------------------------
    # FIND RELATED (UPWARD + DOWNWARD)
    # ----------------------------------------------------------------------
    @with_request_id
    def find_related(self, identifier: Union[int, str], is_id: bool = True) -> Optional[Dict[str, Any]]:

        with self.db_config.main_session() as session:
            try:
                asset = (
                    session.query(AssetNumber)
                    .filter(AssetNumber.id == identifier if is_id else AssetNumber.number == identifier)
                    .first()
                )

                if not asset:
                    warning_id(f"AssetNumber not found: {identifier}", None)
                    return None

                upward = {
                    "model": asset.model,
                    "equipment_group": asset.model.equipment_group if asset.model else None,
                    "area": asset.model.equipment_group.area if asset.model and asset.model.equipment_group else None,
                }

                downward = {
                    "positions": asset.position
                }

                return {
                    "asset_number": asset,
                    "upward": upward,
                    "downward": downward
                }

            except SQLAlchemyError as e:
                error_id(f"AssetNumberService.find_related failed: {e}", None)
                raise

    # ----------------------------------------------------------------------
    # WRAPPERS TO ORM HELPERS
    # ----------------------------------------------------------------------
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
