from __future__ import annotations
from typing import Optional
from sqlalchemy.orm import Session
from modules.emtacdb.emtacdb_fts import SiteLocation
from ._service_utils import apply_non_none_updates, apply_string_search_filters, get_by_id, delete_by_id

class SiteLocationService:
    @staticmethod
    def create(session: Session, **kwargs) -> SiteLocation:
        obj = SiteLocation(**kwargs)
        session.add(obj)
        session.flush()
        return obj

    @staticmethod
    def get_by_id(session: Session, entity_id: int) -> Optional[SiteLocation]:
        return get_by_id(session, SiteLocation, entity_id)

    @staticmethod
    def search(session: Session, **filters) -> list[SiteLocation]:
        query = session.query(SiteLocation)
        query = apply_string_search_filters(query, SiteLocation, filters, ilike_fields=['title', 'room_number', 'site_area'])
        return query.order_by(SiteLocation.id).all()

    @staticmethod
    def update(session: Session, entity_id: int, **kwargs) -> Optional[SiteLocation]:
        obj = SiteLocationService.get_by_id(session, entity_id)
        if not obj:
            return None
        apply_non_none_updates(obj, kwargs)
        session.flush()
        return obj

    @staticmethod
    def delete(session: Session, entity_id: int) -> bool:
        return delete_by_id(session, SiteLocation, entity_id)

    @staticmethod
    def find_or_create(session: Session, **kwargs):
        query = session.query(SiteLocation)
        for key, value in kwargs.items():
            if value is not None and hasattr(SiteLocation, key):
                query = query.filter(getattr(SiteLocation, key) == value)
        obj = query.first()
        if obj:
            return obj
        return SiteLocationService.create(session, **kwargs)
