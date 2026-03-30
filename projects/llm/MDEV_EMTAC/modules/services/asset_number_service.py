from typing import Optional, List, Union, Dict, Any
from sqlalchemy.orm import Session
from sqlalchemy.exc import SQLAlchemyError

from modules.emtacdb.emtacdb_fts import AssetNumber
from modules.configuration.config_env import DatabaseConfig
from modules.configuration.log_config import warning_id, with_request_id


class AssetNumberService:

    def __init__(self, db_config: DatabaseConfig = None):
        self.db_config = db_config or DatabaseConfig()

    # -----------------------------------------------------
    # FIND
    # -----------------------------------------------------

    @with_request_id
    def find(
        self,
        number: Optional[str] = None,
        description: Optional[str] = None,
        model_id: Optional[int] = None,
        limit: int = 100,
        session: Optional[Session] = None,
    ) -> List[AssetNumber]:

        if session:
            return self._find_internal(session, number, description, model_id, limit)

        with self.db_config.main_session() as s:
            return self._find_internal(s, number, description, model_id, limit)

    def _find_internal(self, session, number, description, model_id, limit):
        query = session.query(AssetNumber)

        if number:
            query = query.filter(AssetNumber.number.ilike(f"%{number}%"))

        if description:
            query = query.filter(AssetNumber.description.ilike(f"%{description}%"))

        if model_id:
            query = query.filter(AssetNumber.model_id == model_id)

        return query.limit(limit).all()

    # -----------------------------------------------------
    # GET
    # -----------------------------------------------------

    @with_request_id
    def get(
        self,
        asset_number_id: int,
        session: Optional[Session] = None,
    ) -> Optional[AssetNumber]:

        if session:
            return session.get(AssetNumber, asset_number_id)

        with self.db_config.main_session() as s:
            return s.get(AssetNumber, asset_number_id)

    # -----------------------------------------------------
    # SAVE
    # -----------------------------------------------------

    @with_request_id
    def save(
        self,
        number: str,
        model_id: int,
        description: Optional[str] = None,
        asset_number_id: Optional[int] = None,
        session: Optional[Session] = None,
    ) -> AssetNumber:

        if session:
            return self._save_internal(
                session, number, model_id, description, asset_number_id
            )

        with self.db_config.main_session() as s:
            try:
                asset = self._save_internal(
                    s, number, model_id, description, asset_number_id
                )
                s.commit()
                return asset
            except:
                s.rollback()
                raise

    def _save_internal(
        self, session, number, model_id, description, asset_number_id
    ):

        if asset_number_id:
            asset = session.get(AssetNumber, asset_number_id)
            if not asset:
                raise ValueError(f"AssetNumber id={asset_number_id} not found")

            asset.number = number
            asset.model_id = model_id
            asset.description = description
        else:
            asset = AssetNumber(
                number=number,
                model_id=model_id,
                description=description,
            )
            session.add(asset)

        session.flush()
        return asset

    # -----------------------------------------------------
    # REMOVE
    # -----------------------------------------------------

    @with_request_id
    def remove(
        self,
        asset_number_id: int,
        session: Optional[Session] = None,
    ) -> bool:

        if session:
            return self._remove_internal(session, asset_number_id)

        with self.db_config.main_session() as s:
            try:
                result = self._remove_internal(s, asset_number_id)
                s.commit()
                return result
            except:
                s.rollback()
                raise

    def _remove_internal(self, session, asset_number_id):

        asset = session.get(AssetNumber, asset_number_id)
        if not asset:
            return False

        if asset.position:
            warning_id(
                f"Cannot delete AssetNumber id={asset_number_id}: "
                f"{len(asset.position)} Positions depend on it",
                None,
            )
            return False

        session.delete(asset)
        return True

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
            asset = self._find_related_internal(session, identifier, is_id)
        else:
            with self.db_config.main_session() as s:
                asset = self._find_related_internal(s, identifier, is_id)

        return asset

    def _find_related_internal(self, session, identifier, is_id):

        if is_id:
            asset = session.get(AssetNumber, identifier)
        else:
            asset = session.query(AssetNumber).filter_by(number=identifier).first()

        if not asset:
            return None

        upward = {
            "model": asset.model,
            "equipment_group": asset.model.equipment_group if asset.model else None,
            "area": asset.model.equipment_group.area
            if asset.model and asset.model.equipment_group else None,
        }

        downward = {
            "positions": asset.position
        }

        return {
            "asset_number": asset,
            "upward": upward,
            "downward": downward,
        }

    # -----------------------------------------------------
    # ORM WRAPPERS (unchanged)
    # -----------------------------------------------------

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