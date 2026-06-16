from __future__ import annotations
from typing import Any, Sequence
from sqlalchemy.orm import Session

def apply_non_none_updates(instance: Any, updates: dict[str, Any]) -> Any:
    for key, value in updates.items():
        if value is not None and hasattr(instance, key):
            setattr(instance, key, value)
    return instance

def apply_string_search_filters(query, model_cls, filters: dict[str, Any], ilike_fields: Sequence[str] | None = None):
    ilike_fields = set(ilike_fields or [])
    for field_name, value in filters.items():
        if value is None or value == "" or not hasattr(model_cls, field_name):
            continue
        column = getattr(model_cls, field_name)
        if field_name in ilike_fields and isinstance(value, str):
            query = query.filter(column.ilike(f"%{value}%"))
        else:
            query = query.filter(column == value)
    return query

def get_by_id(session: Session, model_cls: Any, entity_id: int):
    return session.query(model_cls).filter(model_cls.id == entity_id).first()

def delete_by_id(session: Session, model_cls: Any, entity_id: int) -> bool:
    instance = get_by_id(session, model_cls, entity_id)
    if not instance:
        return False
    session.delete(instance)
    session.flush()
    return True
