from __future__ import annotations
from typing import Optional
from sqlalchemy.orm import Session
from modules.emtacdb.emtacdb_fts import Subassembly
from ._service_utils import apply_non_none_updates, apply_string_search_filters, get_by_id, delete_by_id

class SubassemblyService:
    @staticmethod
    def create(session: Session, **kwargs) -> Subassembly:
        obj = Subassembly(**kwargs)
        session.add(obj)
        session.flush()
        return obj

    @staticmethod
    def get_by_id(session: Session, entity_id: int) -> Optional[Subassembly]:
        return get_by_id(session, Subassembly, entity_id)

    @staticmethod
    def search(session: Session, **filters) -> list[Subassembly]:
        query = session.query(Subassembly)
        query = apply_string_search_filters(query, Subassembly, filters, ilike_fields=['name', 'description'])
        return query.order_by(Subassembly.id).all()

    @staticmethod
    def update(session: Session, entity_id: int, **kwargs) -> Optional[Subassembly]:
        obj = SubassemblyService.get_by_id(session, entity_id)
        if not obj:
            return None
        apply_non_none_updates(obj, kwargs)
        session.flush()
        return obj

    @staticmethod
    def delete(session: Session, entity_id: int) -> bool:
        return delete_by_id(session, Subassembly, entity_id)

    @staticmethod
    def find_or_create(session: Session, **kwargs):
        query = session.query(Subassembly)
        for key, value in kwargs.items():
            if value is not None and hasattr(Subassembly, key):
                query = query.filter(getattr(Subassembly, key) == value)
        obj = query.first()
        if obj:
            return obj
        return SubassemblyService.create(session, **kwargs)
