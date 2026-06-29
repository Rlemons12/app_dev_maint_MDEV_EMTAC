from __future__ import annotations
from typing import Optional
from sqlalchemy.orm import Session
from modules.emtacdb.emtacdb_fts import AssetNumber, Model, EquipmentGroup, Position
from ._service_utils import apply_non_none_updates, apply_string_search_filters, get_by_id, delete_by_id

class AssetNumberService:
    @staticmethod
    def create(session: Session, **kwargs) -> AssetNumber:
        obj = AssetNumber(**kwargs)
        session.add(obj)
        session.flush()
        return obj

    @staticmethod
    def get_by_id(session: Session, entity_id: int) -> Optional[AssetNumber]:
        return get_by_id(session, AssetNumber, entity_id)

    @staticmethod
    def search(session: Session, **filters) -> list[AssetNumber]:
        query = session.query(AssetNumber)
        query = apply_string_search_filters(query, AssetNumber, filters, ilike_fields=["number","description"])
        return query.order_by(AssetNumber.number, AssetNumber.id).all()

    @staticmethod
    def update(session: Session, entity_id: int, **kwargs) -> Optional[AssetNumber]:
        obj = AssetNumberService.get_by_id(session, entity_id)
        if not obj:
            return None
        apply_non_none_updates(obj, kwargs)
        session.flush()
        return obj

    @staticmethod
    def delete(session: Session, entity_id: int) -> bool:
        return delete_by_id(session, AssetNumber, entity_id)

    @staticmethod
    def find_or_create(session: Session, **kwargs):
        query = session.query(AssetNumber)
        for key, value in kwargs.items():
            if value is not None and hasattr(AssetNumber, key):
                query = query.filter(getattr(AssetNumber, key) == value)
        obj = query.first()
        if obj:
            return obj
        return AssetNumberService.create(session, **kwargs)

    @staticmethod
    def get_ids_by_number(session: Session, number: str) -> list[int]:
        rows = session.query(AssetNumber.id).filter(AssetNumber.number == number).all()
        return [x for (x,) in rows]

    @staticmethod
    def get_model_id_by_asset_number_id(session: Session, asset_number_id: int):
        return session.query(AssetNumber.model_id).filter(AssetNumber.id == asset_number_id).scalar()

    @staticmethod
    def get_equipment_group_id_by_asset_number_id(session: Session, asset_number_id: int):
        model_id = AssetNumberService.get_model_id_by_asset_number_id(session, asset_number_id)
        if model_id is None:
            return None
        return session.query(Model.equipment_group_id).filter(Model.id == model_id).scalar()

    @staticmethod
    def get_area_id_by_asset_number_id(session: Session, asset_number_id: int):
        equipment_group_id = AssetNumberService.get_equipment_group_id_by_asset_number_id(session, asset_number_id)
        if equipment_group_id is None:
            return None
        return session.query(EquipmentGroup.area_id).filter(EquipmentGroup.id == equipment_group_id).scalar()

    @staticmethod
    def get_position_ids_by_asset_number_id(session: Session, asset_number_id: int) -> list[int]:
        rows = session.query(Position.id).filter(Position.asset_number_id == asset_number_id).all()
        return [x for (x,) in rows]
