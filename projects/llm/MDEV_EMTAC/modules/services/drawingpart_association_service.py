# modules/services/drawing_part_association_service.py

from typing import List, Optional

from modules.configuration.config_env import DatabaseConfig
from modules.emtacdb_fts import DrawingPartAssociation, Part, Drawing
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
