from __future__ import annotations
from typing import Optional
from sqlalchemy.orm import Session
from modules.emtacdb.emtacdb_fts import Model
from ._service_utils import apply_non_none_updates, apply_string_search_filters, get_by_id, delete_by_id

class ModelService:
    @staticmethod
    def create(session: Session, **kwargs) -> Model:
        obj = Model(**kwargs)
        session.add(obj)
        session.flush()
        return obj

    @staticmethod
    def get_by_id(session: Session, entity_id: int) -> Optional[Model]:
        return get_by_id(session, Model, entity_id)

    @staticmethod
    def search(session: Session, **filters) -> list[Model]:
        query = session.query(Model)
        query = apply_string_search_filters(query, Model, filters, ilike_fields=['name', 'description'])
        return query.order_by(Model.id).all()

    @staticmethod
    def update(session: Session, entity_id: int, **kwargs) -> Optional[Model]:
        obj = ModelService.get_by_id(session, entity_id)
        if not obj:
            return None
        apply_non_none_updates(obj, kwargs)
        session.flush()
        return obj

    @staticmethod
    def delete(session: Session, entity_id: int) -> bool:
        return delete_by_id(session, Model, entity_id)

    @staticmethod
    def find_or_create(session: Session, **kwargs):
        query = session.query(Model)
        for key, value in kwargs.items():
            if value is not None and hasattr(Model, key):
                query = query.filter(getattr(Model, key) == value)
        obj = query.first()
        if obj:
            return obj
        return ModelService.create(session, **kwargs)
