from __future__ import annotations
from typing import Optional
from sqlalchemy.orm import Session
from modules.emtacdb.emtacdb_fts import AssemblyView
from ._service_utils import apply_non_none_updates, apply_string_search_filters, get_by_id, delete_by_id

class AssemblyViewService:
    @staticmethod
    def create(session: Session, **kwargs) -> AssemblyView:
        obj = AssemblyView(**kwargs)
        session.add(obj)
        session.flush()
        return obj

    @staticmethod
    def get_by_id(session: Session, entity_id: int) -> Optional[AssemblyView]:
        return get_by_id(session, AssemblyView, entity_id)

    @staticmethod
    def search(session: Session, **filters) -> list[AssemblyView]:
        query = session.query(AssemblyView)
        query = apply_string_search_filters(query, AssemblyView, filters, ilike_fields=['name', 'description'])
        return query.order_by(AssemblyView.id).all()

    @staticmethod
    def update(session: Session, entity_id: int, **kwargs) -> Optional[AssemblyView]:
        obj = AssemblyViewService.get_by_id(session, entity_id)
        if not obj:
            return None
        apply_non_none_updates(obj, kwargs)
        session.flush()
        return obj

    @staticmethod
    def delete(session: Session, entity_id: int) -> bool:
        return delete_by_id(session, AssemblyView, entity_id)

    @staticmethod
    def find_or_create(session: Session, **kwargs):
        query = session.query(AssemblyView)
        for key, value in kwargs.items():
            if value is not None and hasattr(AssemblyView, key):
                query = query.filter(getattr(AssemblyView, key) == value)
        obj = query.first()
        if obj:
            return obj
        return AssemblyViewService.create(session, **kwargs)
