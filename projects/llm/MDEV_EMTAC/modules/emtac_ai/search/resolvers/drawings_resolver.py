from typing import List, Dict, Any, Optional

from modules.configuration.log_config import (
    debug_id,
    info_id,
    warning_id,
    with_request_id,
    get_request_id,
)

from modules.services.drawing_service import DrawingService


class DrawingsResolver:
    """
    DRAWINGS RESOLVER

    Resolves Drawing IDs from query + drawing-focused NER output.

    Pipeline:
        Intent → NER (drawings) → DrawingsResolver → DrawingsSearchExpander
    """

    def __init__(self, drawing_service: Optional[DrawingService] = None):
        self.drawing_service = drawing_service or DrawingService()

    # --------------------------------------------------
    # PRIMARY RESOLUTION ENTRY POINT
    # --------------------------------------------------
    @with_request_id
    def resolve(
        self,
        query: Optional[str],
        entities: Dict[str, Any],
        limit: int = 10,
        request_id: Optional[str] = None,
    ) -> List[int]:

        request_id = request_id or get_request_id()
        entities = entities or {}

        debug_id(
            f"[DrawingsResolver] resolve | query='{query}' | entity_keys={list(entities.keys())}",
            request_id,
        )

        resolved_ids: List[int] = []

        # --------------------------------------------------
        # 0. EXPLICIT DRAWING ID (highest precision)
        # --------------------------------------------------
        explicit_ids = entities.get("DRAWING_ID", [])
        if explicit_ids:
            try:
                resolved_ids.extend(
                    int(i) for i in explicit_ids if str(i).isdigit()
                )
                info_id(
                    f"[DrawingsResolver] resolved via explicit DRAWING_ID ({len(resolved_ids)})",
                    request_id,
                )
                return sorted(set(resolved_ids))
            except Exception as e:
                warning_id(
                    f"[DrawingsResolver] DRAWING_ID parsing failed: {e}",
                    request_id,
                )

        # --------------------------------------------------
        # 1. ENTITY-DRIVEN LOOKUPS (structured, high precision)
        # --------------------------------------------------
        for number in entities.get("DRAWING_NUMBER", []):
            try:
                drawings = self.drawing_service.find(
                    drw_number=number,
                    limit=limit,
                )
                resolved_ids.extend(d.id for d in drawings)
            except Exception as e:
                warning_id(f"[DrawingsResolver] DRAWING_NUMBER lookup failed: {e}", request_id)

        for name in entities.get("DRAWING_NAME", []):
            try:
                drawings = self.drawing_service.find(
                    drw_name=name,
                    limit=limit,
                )
                resolved_ids.extend(d.id for d in drawings)
            except Exception as e:
                warning_id(f"[DrawingsResolver] DRAWING_NAME lookup failed: {e}", request_id)

        for equip in entities.get("EQUIPMENT", []):
            try:
                drawings = self.drawing_service.find(
                    drw_equipment_name=equip,
                    limit=limit,
                )
                resolved_ids.extend(d.id for d in drawings)
            except Exception as e:
                warning_id(f"[DrawingsResolver] EQUIPMENT lookup failed: {e}", request_id)

        for dtype in entities.get("DRAWING_TYPE", []):
            try:
                drawings = self.drawing_service.find(
                    drw_type=dtype,
                    limit=limit,
                )
                resolved_ids.extend(d.id for d in drawings)
            except Exception as e:
                warning_id(f"[DrawingsResolver] DRAWING_TYPE lookup failed: {e}", request_id)

        for spn in entities.get("SPARE_PART_NUMBER", []):
            try:
                drawings = self.drawing_service.find(
                    drw_spare_part_number=spn,
                    limit=limit,
                )
                resolved_ids.extend(d.id for d in drawings)
            except Exception as e:
                warning_id(
                    f"[DrawingsResolver] SPARE_PART_NUMBER lookup failed: {e}",
                    request_id,
                )

        # --------------------------------------------------
        # 2. QUERY-BASED TEXT SEARCH
        # --------------------------------------------------
        if query:
            try:
                drawings = self.drawing_service.find(
                    search_text=query,
                    limit=limit,
                )
                resolved_ids.extend(d.id for d in drawings)
            except Exception as e:
                warning_id(f"[DrawingsResolver] text search failed: {e}", request_id)

        # --------------------------------------------------
        # 3. DE-DUPLICATE & RETURN
        # --------------------------------------------------
        unique_ids = sorted(set(resolved_ids))

        info_id(
            f"[DrawingsResolver] resolved {len(unique_ids)} Drawing IDs",
            request_id,
        )

        return unique_ids

    # --------------------------------------------------
    # DIRECT-ID PASS THROUGH
    # --------------------------------------------------
    def resolve_from_ids(self, ids: List[int]) -> List[int]:
        return sorted({i for i in ids if isinstance(i, int) and i > 0})
