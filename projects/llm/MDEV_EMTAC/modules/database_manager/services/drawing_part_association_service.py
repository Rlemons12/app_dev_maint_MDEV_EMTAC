from __future__ import annotations

from typing import Optional

from sqlalchemy.orm import Session

from modules.emtacdb.emtacdb_fts import (
    Drawing,
    DrawingPartAssociation,
    Part,
)


class DrawingPartAssociationService:
    """
    Service layer for DrawingPartAssociation.

    Rules:
    - accepts an active SQLAlchemy session
    - does not open/close sessions
    - does not commit/rollback
    """

    # -------------------------------------------------------------------------
    # Single association operations
    # -------------------------------------------------------------------------
    @staticmethod
    def create_association(
        session: Session,
        *,
        drawing_id: int,
        part_id: int,
    ) -> DrawingPartAssociation:
        association = DrawingPartAssociation(
            drawing_id=drawing_id,
            part_id=part_id,
        )
        session.add(association)
        session.flush()
        return association

    @staticmethod
    def get_association(
        session: Session,
        *,
        drawing_id: int,
        part_id: int,
    ) -> Optional[DrawingPartAssociation]:
        return (
            session.query(DrawingPartAssociation)
            .filter(
                DrawingPartAssociation.drawing_id == drawing_id,
                DrawingPartAssociation.part_id == part_id,
            )
            .first()
        )

    @staticmethod
    def add(
        session: Session,
        *,
        drawing_id: int,
        part_id: int,
    ) -> int:
        """
        Create a drawing-part association if it does not already exist.
        Returns the association ID either way.
        """
        existing = DrawingPartAssociationService.get_association(
            session,
            drawing_id=drawing_id,
            part_id=part_id,
        )
        if existing:
            return existing.id

        association = DrawingPartAssociationService.create_association(
            session,
            drawing_id=drawing_id,
            part_id=part_id,
        )
        return association.id

    @staticmethod
    def delete_association(
        session: Session,
        *,
        drawing_id: int,
        part_id: int,
    ) -> bool:
        association = DrawingPartAssociationService.get_association(
            session,
            drawing_id=drawing_id,
            part_id=part_id,
        )
        if not association:
            return False

        session.delete(association)
        session.flush()
        return True

    # -------------------------------------------------------------------------
    # Query: parts by drawing
    # -------------------------------------------------------------------------
    @staticmethod
    def get_parts_by_drawing(
        session: Session,
        *,
        drawing_id: int | None = None,
        drw_equipment_name: str | None = None,
        drw_number: str | None = None,
        drw_name: str | None = None,
        drw_revision: str | None = None,
        drw_spare_part_number: str | None = None,
        part_id: int | None = None,
        part_number: str | None = None,
        part_name: str | None = None,
        oem_mfg: str | None = None,
        model: str | None = None,
        class_flag: str | None = None,
        exact_match: bool = False,
        limit: int = 100,
    ) -> list[Part]:
        query = (
            session.query(Part)
            .select_from(Part)
            .join(
                DrawingPartAssociation,
                DrawingPartAssociation.part_id == Part.id,
            )
            .join(
                Drawing,
                Drawing.id == DrawingPartAssociation.drawing_id,
            )
        )

        # Drawing filters
        if drawing_id is not None:
            query = query.filter(Drawing.id == drawing_id)

        if drw_equipment_name is not None:
            query = query.filter(
                Drawing.drw_equipment_name == drw_equipment_name
                if exact_match
                else Drawing.drw_equipment_name.ilike(f"%{drw_equipment_name}%")
            )

        if drw_number is not None:
            query = query.filter(
                Drawing.drw_number == drw_number
                if exact_match
                else Drawing.drw_number.ilike(f"%{drw_number}%")
            )

        if drw_name is not None:
            query = query.filter(
                Drawing.drw_name == drw_name
                if exact_match
                else Drawing.drw_name.ilike(f"%{drw_name}%")
            )

        if drw_revision is not None:
            query = query.filter(
                Drawing.drw_revision == drw_revision
                if exact_match
                else Drawing.drw_revision.ilike(f"%{drw_revision}%")
            )

        if drw_spare_part_number is not None:
            query = query.filter(
                Drawing.drw_spare_part_number == drw_spare_part_number
                if exact_match
                else Drawing.drw_spare_part_number.ilike(f"%{drw_spare_part_number}%")
            )

        # Part filters
        if part_id is not None:
            query = query.filter(Part.id == part_id)

        if part_number is not None:
            query = query.filter(
                Part.part_number == part_number
                if exact_match
                else Part.part_number.ilike(f"%{part_number}%")
            )

        if part_name is not None:
            query = query.filter(
                Part.name == part_name
                if exact_match
                else Part.name.ilike(f"%{part_name}%")
            )

        if oem_mfg is not None:
            query = query.filter(
                Part.oem_mfg == oem_mfg
                if exact_match
                else Part.oem_mfg.ilike(f"%{oem_mfg}%")
            )

        if model is not None:
            query = query.filter(
                Part.model == model
                if exact_match
                else Part.model.ilike(f"%{model}%")
            )

        if class_flag is not None:
            query = query.filter(
                Part.class_flag == class_flag
                if exact_match
                else Part.class_flag.ilike(f"%{class_flag}%")
            )

        return query.distinct().limit(limit).all()

    # -------------------------------------------------------------------------
    # Query: drawings by part
    # -------------------------------------------------------------------------
    @staticmethod
    def get_drawings_by_part(
        session: Session,
        *,
        part_id: int | None = None,
        part_number: str | None = None,
        part_name: str | None = None,
        oem_mfg: str | None = None,
        model: str | None = None,
        class_flag: str | None = None,
        drawing_id: int | None = None,
        drw_equipment_name: str | None = None,
        drw_number: str | None = None,
        drw_name: str | None = None,
        drw_revision: str | None = None,
        drw_spare_part_number: str | None = None,
        exact_match: bool = False,
        limit: int = 100,
    ) -> list[Drawing]:
        query = (
            session.query(Drawing)
            .select_from(Drawing)
            .join(
                DrawingPartAssociation,
                DrawingPartAssociation.drawing_id == Drawing.id,
            )
            .join(
                Part,
                Part.id == DrawingPartAssociation.part_id,
            )
        )

        # Part filters
        if part_id is not None:
            query = query.filter(Part.id == part_id)

        if part_number is not None:
            query = query.filter(
                Part.part_number == part_number
                if exact_match
                else Part.part_number.ilike(f"%{part_number}%")
            )

        if part_name is not None:
            query = query.filter(
                Part.name == part_name
                if exact_match
                else Part.name.ilike(f"%{part_name}%")
            )

        if oem_mfg is not None:
            query = query.filter(
                Part.oem_mfg == oem_mfg
                if exact_match
                else Part.oem_mfg.ilike(f"%{oem_mfg}%")
            )

        if model is not None:
            query = query.filter(
                Part.model == model
                if exact_match
                else Part.model.ilike(f"%{model}%")
            )

        if class_flag is not None:
            query = query.filter(
                Part.class_flag == class_flag
                if exact_match
                else Part.class_flag.ilike(f"%{class_flag}%")
            )

        # Drawing filters
        if drawing_id is not None:
            query = query.filter(Drawing.id == drawing_id)

        if drw_equipment_name is not None:
            query = query.filter(
                Drawing.drw_equipment_name == drw_equipment_name
                if exact_match
                else Drawing.drw_equipment_name.ilike(f"%{drw_equipment_name}%")
            )

        if drw_number is not None:
            query = query.filter(
                Drawing.drw_number == drw_number
                if exact_match
                else Drawing.drw_number.ilike(f"%{drw_number}%")
            )

        if drw_name is not None:
            query = query.filter(
                Drawing.drw_name == drw_name
                if exact_match
                else Drawing.drw_name.ilike(f"%{drw_name}%")
            )

        if drw_revision is not None:
            query = query.filter(
                Drawing.drw_revision == drw_revision
                if exact_match
                else Drawing.drw_revision.ilike(f"%{drw_revision}%")
            )

        if drw_spare_part_number is not None:
            query = query.filter(
                Drawing.drw_spare_part_number == drw_spare_part_number
                if exact_match
                else Drawing.drw_spare_part_number.ilike(f"%{drw_spare_part_number}%")
            )

        return query.distinct().limit(limit).all()

    # -------------------------------------------------------------------------
    # Bulk operations
    # -------------------------------------------------------------------------
    @staticmethod
    def bulk_link(
        session: Session,
        *,
        drawing_ids: list[int] | int,
        part_ids: list[int] | int,
    ) -> int:
        """
        Bulk link parts and drawings in one session.

        You can:
        - pass one drawing_id and many part_ids
        - pass one part_id and many drawing_ids
        - pass many of both and all combinations will be created if missing

        Returns:
            number of newly created associations
        """
        if not drawing_ids or not part_ids:
            return 0

        if isinstance(drawing_ids, int):
            drawing_ids = [drawing_ids]

        if isinstance(part_ids, int):
            part_ids = [part_ids]

        created_count = 0

        for drawing_id in drawing_ids:
            for part_id in part_ids:
                existing = DrawingPartAssociationService.get_association(
                    session,
                    drawing_id=drawing_id,
                    part_id=part_id,
                )
                if existing:
                    continue

                DrawingPartAssociationService.create_association(
                    session,
                    drawing_id=drawing_id,
                    part_id=part_id,
                )
                created_count += 1

        session.flush()
        return created_count

    # -------------------------------------------------------------------------
    # Utility helpers
    # -------------------------------------------------------------------------
    @staticmethod
    def list_by_drawing_id(
        session: Session,
        *,
        drawing_id: int,
    ) -> list[DrawingPartAssociation]:
        return (
            session.query(DrawingPartAssociation)
            .filter(DrawingPartAssociation.drawing_id == drawing_id)
            .all()
        )

    @staticmethod
    def list_by_part_id(
        session: Session,
        *,
        part_id: int,
    ) -> list[DrawingPartAssociation]:
        return (
            session.query(DrawingPartAssociation)
            .filter(DrawingPartAssociation.part_id == part_id)
            .all()
        )