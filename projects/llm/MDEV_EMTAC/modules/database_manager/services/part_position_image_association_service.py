from __future__ import annotations

from typing import Optional

from sqlalchemy.orm import Session

from modules.emtacdb.emtacdb_fts import (
    PartsPositionImageAssociation,
    Position,
)


class PartPositionImageAssociationService:
    """
    Service layer for PartsPositionImageAssociation.

    Rules:
    - accepts an active SQLAlchemy session
    - does not open/close sessions
    - does not commit/rollback
    """

    # -------------------------------------------------------------------------
    # Basic CRUD / association
    # -------------------------------------------------------------------------
    @staticmethod
    def create_association(
        session: Session,
        *,
        part_id: int,
        position_id: int | None,
        image_id: int | None,
    ) -> PartsPositionImageAssociation:
        association = PartsPositionImageAssociation(
            part_id=part_id,
            position_id=position_id,
            image_id=image_id,
        )
        session.add(association)
        session.flush()
        return association

    @staticmethod
    def get_association(
        session: Session,
        *,
        part_id: int,
        position_id: int | None = None,
        image_id: int | None = None,
    ) -> Optional[PartsPositionImageAssociation]:
        query = session.query(PartsPositionImageAssociation).filter(
            PartsPositionImageAssociation.part_id == part_id
        )

        if position_id is not None:
            query = query.filter(
                PartsPositionImageAssociation.position_id == position_id
            )

        if image_id is not None:
            query = query.filter(
                PartsPositionImageAssociation.image_id == image_id
            )

        return query.first()

    @staticmethod
    def add(
        session: Session,
        *,
        part_id: int,
        position_id: int | None,
        image_id: int | None,
    ) -> int:
        """
        Create association if not exists.
        Returns association ID either way.
        """
        existing = PartPositionImageAssociationService.get_association(
            session,
            part_id=part_id,
            position_id=position_id,
            image_id=image_id,
        )

        if existing:
            return existing.id

        association = PartPositionImageAssociationService.create_association(
            session,
            part_id=part_id,
            position_id=position_id,
            image_id=image_id,
        )

        return association.id

    @staticmethod
    def delete(
        session: Session,
        *,
        association_id: int,
    ) -> bool:
        assoc = (
            session.query(PartsPositionImageAssociation)
            .filter(PartsPositionImageAssociation.id == association_id)
            .first()
        )

        if not assoc:
            return False

        session.delete(assoc)
        session.flush()
        return True

    # -------------------------------------------------------------------------
    # Search
    # -------------------------------------------------------------------------
    @staticmethod
    def search(
        session: Session,
        *,
        part_id: int | None = None,
        position_id: int | None = None,
        image_id: int | None = None,
    ) -> list[PartsPositionImageAssociation]:
        query = session.query(PartsPositionImageAssociation)

        if part_id is not None:
            query = query.filter(
                PartsPositionImageAssociation.part_id == part_id
            )

        if position_id is not None:
            query = query.filter(
                PartsPositionImageAssociation.position_id == position_id
            )

        if image_id is not None:
            query = query.filter(
                PartsPositionImageAssociation.image_id == image_id
            )

        return query.all()

    # -------------------------------------------------------------------------
    # Bulk operations
    # -------------------------------------------------------------------------
    @staticmethod
    def bulk_link(
        session: Session,
        *,
        part_ids: list[int] | int,
        position_ids: list[int] | int | None = None,
        image_ids: list[int] | int | None = None,
    ) -> int:
        """
        Bulk create associations.

        Supports:
        - part -> image
        - part -> position
        - part -> position -> image

        Returns number of new associations created
        """

        if isinstance(part_ids, int):
            part_ids = [part_ids]

        if isinstance(position_ids, int):
            position_ids = [position_ids]

        if isinstance(image_ids, int):
            image_ids = [image_ids]

        created_count = 0

        for part_id in part_ids:
            for position_id in (position_ids or [None]):
                for image_id in (image_ids or [None]):

                    existing = PartPositionImageAssociationService.get_association(
                        session,
                        part_id=part_id,
                        position_id=position_id,
                        image_id=image_id,
                    )

                    if existing:
                        continue

                    PartPositionImageAssociationService.create_association(
                        session,
                        part_id=part_id,
                        position_id=position_id,
                        image_id=image_id,
                    )

                    created_count += 1

        session.flush()
        return created_count

    # -------------------------------------------------------------------------
    # Position lookup (replaces your model hierarchy logic)
    # -------------------------------------------------------------------------
    @staticmethod
    def get_corresponding_position_ids(
        session: Session,
        *,
        area_id: int | None = None,
        equipment_group_id: int | None = None,
        model_id: int | None = None,
        asset_number_id: int | None = None,
        location_id: int | None = None,
    ) -> list[int]:
        positions = PartPositionImageAssociationService._get_positions_by_hierarchy(
            session,
            area_id=area_id,
            equipment_group_id=equipment_group_id,
            model_id=model_id,
            asset_number_id=asset_number_id,
            location_id=location_id,
        )

        return [p.id for p in positions]

    @staticmethod
    def _get_positions_by_hierarchy(
        session: Session,
        *,
        area_id: int | None = None,
        equipment_group_id: int | None = None,
        model_id: int | None = None,
        asset_number_id: int | None = None,
        location_id: int | None = None,
    ) -> list[Position]:

        filters = {}

        if area_id:
            filters["area_id"] = area_id
        if equipment_group_id:
            filters["equipment_group_id"] = equipment_group_id
        if model_id:
            filters["model_id"] = model_id
        if asset_number_id:
            filters["asset_number_id"] = asset_number_id
        if location_id:
            filters["location_id"] = location_id

        query = session.query(Position)

        if filters:
            query = query.filter_by(**filters)

        return query.all()

    # -------------------------------------------------------------------------
    # Convenience helpers
    # -------------------------------------------------------------------------
    @staticmethod
    def list_by_part_id(
        session: Session,
        *,
        part_id: int,
    ) -> list[PartsPositionImageAssociation]:
        return (
            session.query(PartsPositionImageAssociation)
            .filter(PartsPositionImageAssociation.part_id == part_id)
            .all()
        )

    @staticmethod
    def list_by_image_id(
        session: Session,
        *,
        image_id: int,
    ) -> list[PartsPositionImageAssociation]:
        return (
            session.query(PartsPositionImageAssociation)
            .filter(PartsPositionImageAssociation.image_id == image_id)
            .all()
        )

    @staticmethod
    def list_by_position_id(
        session: Session,
        *,
        position_id: int,
    ) -> list[PartsPositionImageAssociation]:
        return (
            session.query(PartsPositionImageAssociation)
            .filter(PartsPositionImageAssociation.position_id == position_id)
            .all()
        )