from __future__ import annotations

from typing import Optional

from sqlalchemy.orm import Session

from modules.emtacdb.emtacdb_fts import Location
from ._service_utils import (
    apply_non_none_updates,
    apply_string_search_filters,
    get_by_id,
    delete_by_id,
)


class LocationService:
    @staticmethod
    def create(session: Session, **kwargs) -> Location:
        obj = Location(**kwargs)
        session.add(obj)
        session.flush()
        return obj

    @staticmethod
    def get_by_id(session: Session, entity_id: int) -> Optional[Location]:
        return get_by_id(session, Location, entity_id)

    @staticmethod
    def search(session: Session, **filters) -> list[Location]:
        query = session.query(Location)
        query = apply_string_search_filters(
            query,
            Location,
            filters,
            ilike_fields=["name", "description"],
        )

        # Exact-match filters for FK fields
        for key in ("model_id", "asset_number_id"):
            if key in filters:
                value = filters[key]
                column = getattr(Location, key)
                if value is None:
                    query = query.filter(column.is_(None))
                else:
                    query = query.filter(column == value)

        return query.order_by(Location.id).all()

    @staticmethod
    def update(session: Session, entity_id: int, **kwargs) -> Optional[Location]:
        obj = LocationService.get_by_id(session, entity_id)
        if not obj:
            return None
        apply_non_none_updates(obj, kwargs)
        session.flush()
        return obj

    @staticmethod
    def delete(session: Session, entity_id: int) -> bool:
        return delete_by_id(session, Location, entity_id)

    @staticmethod
    def find_or_create(
        session: Session,
        *,
        name: str,
        model_id: int,
        asset_number_id: Optional[int] = None,
        description: Optional[str] = None,
        **extra_kwargs,
    ) -> Location:
        """
        Location uniqueness rule:
        - name
        - model_id
        - asset_number_id

        This means these are different rows:
        - Station 1 / model 4 / asset_number_id NULL
        - Station 1 / model 4 / asset_number_id 27
        """
        query = (
            session.query(Location)
            .filter(Location.name == name)
            .filter(Location.model_id == model_id)
        )

        if asset_number_id is None:
            query = query.filter(Location.asset_number_id.is_(None))
        else:
            query = query.filter(Location.asset_number_id == asset_number_id)

        obj = query.first()
        if obj:
            if description is not None:
                obj.description = description
                session.flush()
            return obj

        payload = {
            "name": name,
            "model_id": model_id,
            "asset_number_id": asset_number_id,
            "description": description,
        }
        payload.update(extra_kwargs)

        return LocationService.create(session, **payload)

    @staticmethod
    def get_asset_specific_locations(session: Session, asset_number_id: int) -> list[Location]:
        return (
            session.query(Location)
            .filter(Location.asset_number_id == asset_number_id)
            .order_by(Location.id)
            .all()
        )

    @staticmethod
    def get_model_level_locations(session: Session, model_id: int) -> list[Location]:
        return (
            session.query(Location)
            .filter(Location.model_id == model_id)
            .filter(Location.asset_number_id.is_(None))
            .order_by(Location.id)
            .all()
        )