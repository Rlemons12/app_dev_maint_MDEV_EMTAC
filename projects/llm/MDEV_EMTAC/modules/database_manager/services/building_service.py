from __future__ import annotations
from typing import Optional
from sqlalchemy.orm import Session
from modules.emtacdb.emtacdb_fts import Building
from ._service_utils import apply_non_none_updates, apply_string_search_filters, get_by_id, delete_by_id

class BuildingService:
    @staticmethod
    def create(session: Session, **kwargs) -> Building:
        obj = Building(**kwargs)
        session.add(obj)
        session.flush()
        return obj

    @staticmethod
    def get_by_id(session: Session, entity_id: int) -> Optional[Building]:
        return get_by_id(session, Building, entity_id)

    @staticmethod
    def search(session: Session, **filters) -> list[Building]:
        query = session.query(Building)
        query = apply_string_search_filters(query, Building, filters, ilike_fields=['name', 'description', 'address'])
        return query.order_by(Building.id).all()

    @staticmethod
    def update(session: Session, entity_id: int, **kwargs) -> Optional[Building]:
        obj = BuildingService.get_by_id(session, entity_id)
        if not obj:
            return None
        apply_non_none_updates(obj, kwargs)
        session.flush()
        return obj

    @staticmethod
    def delete(session: Session, entity_id: int) -> bool:
        return delete_by_id(session, Building, entity_id)

    @staticmethod
    def find_or_create(session: Session, **kwargs):
        query = session.query(Building)
        for key, value in kwargs.items():
            if value is not None and hasattr(Building, key):
                query = query.filter(getattr(Building, key) == value)
        obj = query.first()
        if obj:
            return obj
        return BuildingService.create(session, **kwargs)
