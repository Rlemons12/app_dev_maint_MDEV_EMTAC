# modules/services/drawing_part_association_service.py

from typing import List, Optional, Dict
from sqlalchemy.orm import Session
from modules.configuration.config_env import DatabaseConfig
from modules.emtacdb.emtacdb_fts import DrawingPartAssociation, Part, Drawing
from modules.configuration.log_config import (
    info_id, debug_id, error_id, warning_id, with_request_id, get_request_id
)


class DrawingPartAssociationService:
    """Service layer for managing Drawing ↔ Part associations."""

    def __init__(self, db_config: Optional[DatabaseConfig] = None):
        self.db_config = db_config or DatabaseConfig()

    # --------------------------
    # CREATE
    # --------------------------
    @with_request_id
    def add_association(self, drawing_id: int, part_id: int, request_id: Optional[str] = None) -> Optional[int]:
        """Add a single drawing ↔ part association (wraps ORM add)."""
        return DrawingPartAssociation.add(drawing_id=drawing_id, part_id=part_id, request_id=request_id)

    @with_request_id
    def bulk_link(self,
                  drawing_ids: List[int],
                  part_ids: List[int],
                  request_id: Optional[str] = None) -> int:
        """Bulk link parts and drawings (wraps ORM bulk_link)."""
        return DrawingPartAssociation.bulk_link(
            drawing_ids=drawing_ids, part_ids=part_ids, request_id=request_id
        )

    # --------------------------
    # READ
    # --------------------------
    @with_request_id
    def get_parts_by_drawing(self,
                             drawing_id: Optional[int] = None,
                             drw_equipment_name: Optional[str] = None,
                             drw_number: Optional[str] = None,
                             drw_name: Optional[str] = None,
                             drw_revision: Optional[str] = None,
                             drw_spare_part_number: Optional[str] = None,
                             part_id: Optional[int] = None,
                             part_number: Optional[str] = None,
                             part_name: Optional[str] = None,
                             oem_mfg: Optional[str] = None,
                             model: Optional[str] = None,
                             class_flag: Optional[str] = None,
                             exact_match: bool = False,
                             limit: int = 100,
                             request_id: Optional[str] = None) -> List[Part]:
        """Get all parts associated with a drawing (wraps ORM get_parts_by_drawing)."""
        return DrawingPartAssociation.get_parts_by_drawing(
            drawing_id=drawing_id,
            drw_equipment_name=drw_equipment_name,
            drw_number=drw_number,
            drw_name=drw_name,
            drw_revision=drw_revision,
            drw_spare_part_number=drw_spare_part_number,
            part_id=part_id,
            part_number=part_number,
            part_name=part_name,
            oem_mfg=oem_mfg,
            model=model,
            class_flag=class_flag,
            exact_match=exact_match,
            limit=limit,
            request_id=request_id
        )

    @with_request_id
    def get_drawings_by_part(self,
                             part_id: Optional[int] = None,
                             part_number: Optional[str] = None,
                             part_name: Optional[str] = None,
                             oem_mfg: Optional[str] = None,
                             model: Optional[str] = None,
                             class_flag: Optional[str] = None,
                             drawing_id: Optional[int] = None,
                             drw_equipment_name: Optional[str] = None,
                             drw_number: Optional[str] = None,
                             drw_name: Optional[str] = None,
                             drw_revision: Optional[str] = None,
                             drw_spare_part_number: Optional[str] = None,
                             exact_match: bool = False,
                             limit: int = 100,
                             request_id: Optional[str] = None) -> List[Drawing]:
        """Get all drawings associated with a part (wraps ORM get_drawings_by_part)."""
        return DrawingPartAssociation.get_drawings_by_part(
            part_id=part_id,
            part_number=part_number,
            part_name=part_name,
            oem_mfg=oem_mfg,
            model=model,
            class_flag=class_flag,
            drawing_id=drawing_id,
            drw_equipment_name=drw_equipment_name,
            drw_number=drw_number,
            drw_name=drw_name,
            drw_revision=drw_revision,
            drw_spare_part_number=drw_spare_part_number,
            exact_match=exact_match,
            limit=limit,
            request_id=request_id
        )

    @with_request_id
    def get_drawing_numbers_by_part_ids(
            self,
            part_ids: List[int],
            session: Session,
    ) -> Dict[int, List[str]]:
        """
        Return drawing numbers grouped by part_id.

        Example:
        {
            1: ["DRW-1001", "DRW-1002"],
            5: ["DRW-2204"]
        }
        """
        rid = get_request_id()

        if not part_ids:
            debug_id("[DrawingPartAssociationService] No part_ids provided", rid)
            return {}

        rows = (
            session.query(
                DrawingPartAssociation.part_id,
                Drawing.drw_number,
            )
            .join(Drawing, Drawing.id == DrawingPartAssociation.drawing_id)
            .filter(DrawingPartAssociation.part_id.in_(part_ids))
            .distinct()
            .all()
        )

        drawing_map: Dict[int, List[str]] = {}

        for part_id, drw_number in rows:
            if not drw_number:
                continue
            drawing_map.setdefault(part_id, []).append(drw_number)

        debug_id(
            f"[DrawingPartAssociationService] Resolved drawings for {len(drawing_map)} parts",
            rid,
        )

        return drawing_map

    # --------------------------
    # HIGH-LEVEL RESOLUTION
    # --------------------------
    @with_request_id
    def resolve_parts_with_drawings(
            self,
            session: Optional[Session] = None,
            part_id: Optional[int] = None,
            part_number: Optional[str] = None,
            limit: int = 500,
            request_id: Optional[str] = None,
    ) -> Dict[int, Dict]:
        """
        Resolve parts and their associated drawings.

        Mirrors SQL:
        part → drawing_part → drawing

        Returns:
        {
            part_id: {
                "part_number": str,
                "name": str,
                "drawings": [
                    {
                        "drawing_id": int,
                        "drw_number": str,
                        "drw_name": str,
                        "drw_revision": str,
                    }
                ]
            }
        }
        """

        rid = request_id or get_request_id()

        session_provided = session is not None
        if not session_provided:
            session = self.db_config.get_main_session()

        try:
            # ---------------------------------
            # 1. Resolve Parts
            # ---------------------------------
            if part_id:
                parts = [session.query(Part).filter(Part.id == part_id).first()]
                parts = [p for p in parts if p]
            else:
                q = session.query(Part)

                if part_number:
                    q = q.filter(Part.part_number.ilike(f"%{part_number}%"))

                parts = q.order_by(Part.part_number).limit(limit).all()

            if not parts:
                debug_id("[resolve_parts_with_drawings] No parts found", rid)
                return {}

            part_ids = [p.id for p in parts]

            # ---------------------------------
            # 2. Join drawing_part → drawing
            # ---------------------------------
            rows = (
                session.query(
                    DrawingPartAssociation.part_id,
                    Drawing.id,
                    Drawing.drw_number,
                    Drawing.drw_name,
                    Drawing.drw_revision,
                )
                .join(Drawing, Drawing.id == DrawingPartAssociation.drawing_id)
                .filter(DrawingPartAssociation.part_id.in_(part_ids))
                .order_by(DrawingPartAssociation.part_id, Drawing.drw_number)
                .all()
            )

            # ---------------------------------
            # 3. Assemble result
            # ---------------------------------
            result: Dict[int, Dict] = {
                p.id: {
                    "part_number": p.part_number,
                    "name": p.name,
                    "drawings": [],
                }
                for p in parts
            }

            for part_id, drw_id, drw_number, drw_name, drw_revision in rows:
                result[part_id]["drawings"].append({
                    "drawing_id": drw_id,
                    "drw_number": drw_number,
                    "drw_name": drw_name,
                    "drw_revision": drw_revision,
                })

            # Remove parts with no drawings
            result = {k: v for k, v in result.items() if v["drawings"]}

            debug_id(
                f"[resolve_parts_with_drawings] Resolved drawings for {len(result)} parts",
                rid,
            )

            return result

        except Exception as e:
            error_id(
                f"[resolve_parts_with_drawings] Failed: {e}",
                rid,
                exc_info=True,
            )
            return {}

        finally:
            if not session_provided:
                session.close()

    def get_drawings_for_part(self, *, part_id: int, session):
        return (
            session.query(Drawing)
            .join(DrawingPartAssociation,
                  DrawingPartAssociation.drawing_id == Drawing.id)
            .filter(DrawingPartAssociation.part_id == part_id)
            .all()
        )


