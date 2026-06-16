from __future__ import annotations
from typing import Optional
from sqlalchemy.orm import Session
from modules.emtacdb.emtacdb_fts import Area
from ._service_utils import apply_non_none_updates, apply_string_search_filters, get_by_id, delete_by_id

class AreaService:
    @staticmethod
    def create(session: Session, **kwargs) -> Area:
        obj = Area(**kwargs)
        session.add(obj)
        session.flush()
        return obj

    @staticmethod
    def get_by_id(session: Session, entity_id: int) -> Optional[Area]:
        return get_by_id(session, Area, entity_id)

    @staticmethod
    def search(session: Session, **filters) -> list[Area]:
        query = session.query(Area)
        query = apply_string_search_filters(query, Area, filters, ilike_fields=['name', 'description'])
        return query.order_by(Area.id).all()

    @staticmethod
    def update(session: Session, entity_id: int, **kwargs) -> Optional[Area]:
        obj = AreaService.get_by_id(session, entity_id)
        if not obj:
            return None
        apply_non_none_updates(obj, kwargs)
        session.flush()
        return obj

    @staticmethod
    def delete(session: Session, entity_id: int) -> bool:
        return delete_by_id(session, Area, entity_id)

    @staticmethod
    def find_or_create(session: Session, **kwargs):
        query = session.query(Area)
        for key, value in kwargs.items():
            if value is not None and hasattr(Area, key):
                query = query.filter(getattr(Area, key) == value)
        obj = query.first()
        if obj:
            return obj
        return AreaService.create(session, **kwargs)
