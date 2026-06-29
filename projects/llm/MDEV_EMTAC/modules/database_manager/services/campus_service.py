from __future__ import annotations
from typing import Optional
from sqlalchemy.orm import Session
from modules.emtacdb.emtacdb_fts import Campus
from ._service_utils import apply_non_none_updates, apply_string_search_filters, get_by_id, delete_by_id

class CampusService:
    @staticmethod
    def create(session: Session, **kwargs) -> Campus:
        obj = Campus(**kwargs)
        session.add(obj)
        session.flush()
        return obj

    @staticmethod
    def get_by_id(session: Session, entity_id: int) -> Optional[Campus]:
        return get_by_id(session, Campus, entity_id)

    @staticmethod
    def search(session: Session, **filters) -> list[Campus]:
        query = session.query(Campus)
        query = apply_string_search_filters(query, Campus, filters, ilike_fields=['name', 'description', 'city', 'state', 'country'])
        return query.order_by(Campus.id).all()

    @staticmethod
    def update(session: Session, entity_id: int, **kwargs) -> Optional[Campus]:
        obj = CampusService.get_by_id(session, entity_id)
        if not obj:
            return None
        apply_non_none_updates(obj, kwargs)
        session.flush()
        return obj

    @staticmethod
    def delete(session: Session, entity_id: int) -> bool:
        return delete_by_id(session, Campus, entity_id)

    @staticmethod
    def find_or_create(session: Session, **kwargs):
        query = session.query(Campus)
        for key, value in kwargs.items():
            if value is not None and hasattr(Campus, key):
                query = query.filter(getattr(Campus, key) == value)
        obj = query.first()
        if obj:
            return obj
        return CampusService.create(session, **kwargs)
