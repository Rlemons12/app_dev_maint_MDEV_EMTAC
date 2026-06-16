from __future__ import annotations
from typing import Optional
from sqlalchemy.orm import Session
from modules.emtacdb.emtacdb_fts import ComponentAssembly
from ._service_utils import apply_non_none_updates, apply_string_search_filters, get_by_id, delete_by_id

class ComponentAssemblyService:
    @staticmethod
    def create(session: Session, **kwargs) -> ComponentAssembly:
        obj = ComponentAssembly(**kwargs)
        session.add(obj)
        session.flush()
        return obj

    @staticmethod
    def get_by_id(session: Session, entity_id: int) -> Optional[ComponentAssembly]:
        return get_by_id(session, ComponentAssembly, entity_id)

    @staticmethod
    def search(session: Session, **filters) -> list[ComponentAssembly]:
        query = session.query(ComponentAssembly)
        query = apply_string_search_filters(query, ComponentAssembly, filters, ilike_fields=['name', 'description'])
        return query.order_by(ComponentAssembly.id).all()

    @staticmethod
    def update(session: Session, entity_id: int, **kwargs) -> Optional[ComponentAssembly]:
        obj = ComponentAssemblyService.get_by_id(session, entity_id)
        if not obj:
            return None
        apply_non_none_updates(obj, kwargs)
        session.flush()
        return obj

    @staticmethod
    def delete(session: Session, entity_id: int) -> bool:
        return delete_by_id(session, ComponentAssembly, entity_id)

    @staticmethod
    def find_or_create(session: Session, **kwargs):
        query = session.query(ComponentAssembly)
        for key, value in kwargs.items():
            if value is not None and hasattr(ComponentAssembly, key):
                query = query.filter(getattr(ComponentAssembly, key) == value)
        obj = query.first()
        if obj:
            return obj
        return ComponentAssemblyService.create(session, **kwargs)
