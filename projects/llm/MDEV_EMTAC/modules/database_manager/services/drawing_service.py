from __future__ import annotations

from typing import Optional

from sqlalchemy import and_, or_
from sqlalchemy.orm import Session

from modules.emtacdb.emtacdb_fts import (
    AssetNumber,
    Drawing,
    DrawingPositionAssociation,
    DrawingType,
)
from ._service_utils import apply_non_none_updates, get_by_id, delete_by_id


class DrawingService:
    """
    Service layer for Drawing.

    Rules:
    - accepts an active SQLAlchemy session
    - does not open/close sessions
    - does not commit/rollback
    - returns ORM objects or formatted dictionaries depending on method
    """

    DEFAULT_SEARCH_FIELDS = (
        "drw_number",
        "drw_name",
        "drw_equipment_name",
        "drw_spare_part_number",
        "drw_type",
    )

    # -------------------------------------------------------------------------
    # CRUD
    # -------------------------------------------------------------------------
    @staticmethod
    def create(
        session: Session,
        *,
        drw_equipment_name: str | None = None,
        drw_number: str | None = None,
        drw_name: str | None = None,
        drw_revision: str | None = None,
        drw_spare_part_number: str | None = None,
        drw_type: str | None = "Other",
        file_path: str,
    ) -> Drawing:
        drawing = Drawing(
            drw_equipment_name=drw_equipment_name,
            drw_number=drw_number,
            drw_name=drw_name,
            drw_revision=drw_revision,
            drw_spare_part_number=drw_spare_part_number,
            drw_type=drw_type or "Other",
            file_path=file_path,
        )
        session.add(drawing)
        session.flush()
        return drawing

    @staticmethod
    def get_by_id(session: Session, drawing_id: int) -> Optional[Drawing]:
        return get_by_id(session, Drawing, drawing_id)

    @staticmethod
    def get_by_number(
        session: Session,
        *,
        drw_number: str,
    ) -> Optional[Drawing]:
        return (
            session.query(Drawing)
            .filter(Drawing.drw_number == drw_number)
            .first()
        )

    @staticmethod
    def update(
        session: Session,
        drawing_id: int,
        *,
        drw_equipment_name: str | None = None,
        drw_number: str | None = None,
        drw_name: str | None = None,
        drw_revision: str | None = None,
        drw_spare_part_number: str | None = None,
        drw_type: str | None = None,
        file_path: str | None = None,
    ) -> Optional[Drawing]:
        drawing = DrawingService.get_by_id(session, drawing_id)
        if not drawing:
            return None

        apply_non_none_updates(
            drawing,
            {
                "drw_equipment_name": drw_equipment_name,
                "drw_number": drw_number,
                "drw_name": drw_name,
                "drw_revision": drw_revision,
                "drw_spare_part_number": drw_spare_part_number,
                "drw_type": drw_type,
                "file_path": file_path,
            },
        )
        session.flush()
        return drawing

    @staticmethod
    def delete(session: Session, drawing_id: int) -> bool:
        return delete_by_id(session, Drawing, drawing_id)

    @staticmethod
    def find_or_create(
        session: Session,
        *,
        drw_equipment_name: str | None = None,
        drw_number: str | None = None,
        drw_name: str | None = None,
        drw_revision: str | None = None,
        drw_spare_part_number: str | None = None,
        drw_type: str | None = "Other",
        file_path: str,
    ) -> Drawing:
        query = session.query(Drawing)

        if drw_number is not None:
            query = query.filter(Drawing.drw_number == drw_number)

        if drw_revision is not None:
            query = query.filter(Drawing.drw_revision == drw_revision)

        existing = query.first()
        if existing:
            return existing

        return DrawingService.create(
            session,
            drw_equipment_name=drw_equipment_name,
            drw_number=drw_number,
            drw_name=drw_name,
            drw_revision=drw_revision,
            drw_spare_part_number=drw_spare_part_number,
            drw_type=drw_type,
            file_path=file_path,
        )

    # -------------------------------------------------------------------------
    # Search
    # -------------------------------------------------------------------------
    @staticmethod
    def search(
        session: Session,
        *,
        search_text: str | None = None,
        fields: list[str] | tuple[str, ...] | None = None,
        exact_match: bool = False,
        drawing_id: int | None = None,
        drw_equipment_name: str | None = None,
        drw_number: str | None = None,
        drw_name: str | None = None,
        drw_revision: str | None = None,
        drw_spare_part_number: str | None = None,
        drw_type: str | None = None,
        file_path: str | None = None,
        limit: int = 100,
    ) -> list[Drawing]:
        """
        Comprehensive search method for Drawing objects with flexible search options.
        """
        query = session.query(Drawing)
        filters = []

        # -------------------------------------------------------------
        # search_text across selected/default fields
        # -------------------------------------------------------------
        if search_text:
            search_text = search_text.strip()
            if search_text:
                search_fields = fields or DrawingService.DEFAULT_SEARCH_FIELDS
                text_filters = []

                for field_name in search_fields:
                    if hasattr(Drawing, field_name):
                        field = getattr(Drawing, field_name)
                        if exact_match:
                            text_filters.append(field == search_text)
                        else:
                            text_filters.append(field.ilike(f"%{search_text}%"))

                if text_filters:
                    filters.append(or_(*text_filters))

        # -------------------------------------------------------------
        # field-specific filters
        # -------------------------------------------------------------
        if drawing_id is not None:
            filters.append(Drawing.id == drawing_id)

        if drw_equipment_name is not None:
            filters.append(
                Drawing.drw_equipment_name == drw_equipment_name
                if exact_match
                else Drawing.drw_equipment_name.ilike(f"%{drw_equipment_name}%")
            )

        if drw_number is not None:
            filters.append(
                Drawing.drw_number == drw_number
                if exact_match
                else Drawing.drw_number.ilike(f"%{drw_number}%")
            )

        if drw_name is not None:
            filters.append(
                Drawing.drw_name == drw_name
                if exact_match
                else Drawing.drw_name.ilike(f"%{drw_name}%")
            )

        if drw_revision is not None:
            filters.append(
                Drawing.drw_revision == drw_revision
                if exact_match
                else Drawing.drw_revision.ilike(f"%{drw_revision}%")
            )

        if drw_spare_part_number is not None:
            filters.append(
                Drawing.drw_spare_part_number == drw_spare_part_number
                if exact_match
                else Drawing.drw_spare_part_number.ilike(f"%{drw_spare_part_number}%")
            )

        if drw_type is not None:
            filters.append(
                Drawing.drw_type == drw_type
                if exact_match
                else Drawing.drw_type.ilike(f"%{drw_type}%")
            )

        if file_path is not None:
            filters.append(
                Drawing.file_path == file_path
                if exact_match
                else Drawing.file_path.ilike(f"%{file_path}%")
            )

        if filters:
            query = query.filter(and_(*filters))

        query = query.limit(limit)
        return query.all()

    @staticmethod
    def search_and_format(
        session: Session,
        *,
        search_text: str | None = None,
        fields: list[str] | tuple[str, ...] | None = None,
        exact_match: bool = False,
        drawing_id: int | None = None,
        drw_equipment_name: str | None = None,
        drw_number: str | None = None,
        drw_name: str | None = None,
        drw_revision: str | None = None,
        drw_spare_part_number: str | None = None,
        drw_type: str | None = None,
        file_path: str | None = None,
        limit: int = 100,
    ) -> dict:
        """
        Search for drawings and return formatted results ready for API response.
        """
        results = DrawingService.search(
            session,
            search_text=search_text,
            fields=fields,
            exact_match=exact_match,
            drawing_id=drawing_id,
            drw_equipment_name=drw_equipment_name,
            drw_number=drw_number,
            drw_name=drw_name,
            drw_revision=drw_revision,
            drw_spare_part_number=drw_spare_part_number,
            drw_type=drw_type,
            file_path=file_path,
            limit=limit,
        )

        if results:
            drawing_results = []
            for drawing in results:
                drawing_results.append(
                    {
                        "id": drawing.id,
                        "number": drawing.drw_number,
                        "name": drawing.drw_name,
                        "equipment_name": drawing.drw_equipment_name,
                        "revision": drawing.drw_revision,
                        "spare_part_number": drawing.drw_spare_part_number,
                        "type": drawing.drw_type,
                        "file_path": drawing.file_path,
                        "url": f"/drawings/view/{drawing.id}",
                    }
                )

            return {
                "entity_type": "drawing",
                "results": drawing_results,
            }

        return {
            "entity_type": "response",
            "results": [{"text": "No drawings found matching your criteria."}],
        }

    @staticmethod
    def search_by_asset_number(
        session: Session,
        *,
        asset_number_value: str,
    ) -> list[Drawing]:
        """
        Search for drawings related to a specific asset number.
        """
        asset_number_ids = AssetNumber.get_ids_by_number(session, asset_number_value)
        if not asset_number_ids:
            return []

        position_ids: list[int] = []
        for asset_id in asset_number_ids:
            pos_ids = AssetNumber.get_position_ids_by_asset_number_id(session, asset_id)
            position_ids.extend(pos_ids)

        if not position_ids:
            return []

        drawings: list[Drawing] = []
        for pos_id in position_ids:
            drawing_results = DrawingPositionAssociation.get_drawings_by_position(
                session=session,
                position_id=pos_id,
            )
            drawings.extend(drawing_results)

        unique_drawings = {drawing.id: drawing for drawing in drawings}
        return list(unique_drawings.values())

    @staticmethod
    def search_by_type(
        session: Session,
        *,
        drawing_type: str,
        limit: int = 100,
    ) -> list[Drawing]:
        return DrawingService.search(
            session,
            drw_type=drawing_type,
            limit=limit,
        )

    # -------------------------------------------------------------------------
    # Utility / formatting helpers
    # -------------------------------------------------------------------------
    @staticmethod
    def get_available_types() -> list[str]:
        return [dtype.value for dtype in DrawingType]

    @staticmethod
    def format_drawing(drawing: Drawing) -> dict:
        return {
            "id": drawing.id,
            "number": drawing.drw_number,
            "name": drawing.drw_name,
            "equipment_name": drawing.drw_equipment_name,
            "revision": drawing.drw_revision,
            "spare_part_number": drawing.drw_spare_part_number,
            "type": drawing.drw_type,
            "file_path": drawing.file_path,
            "url": f"/drawings/view/{drawing.id}",
        }

    @staticmethod
    def list_all(session: Session, limit: int = 1000) -> list[Drawing]:
        return (
            session.query(Drawing)
            .order_by(Drawing.drw_number, Drawing.id)
            .limit(limit)
            .all()
        )