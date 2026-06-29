from typing import List, Dict, Any, Optional

from modules.configuration.log_config import (
    debug_id,
    info_id,
    warning_id,
    with_request_id,
    get_request_id,
)

from modules.services.part_service import PartService


class PartsResolver:
    """
    PARTS RESOLVER

    Resolves Part IDs from query + parts NER output.

    Pipeline:
        Intent → NER (parts) → PartsResolver → SearchExpansion
    """

    def __init__(self, part_service: Optional[PartService] = None):
        if part_service and not isinstance(part_service, PartService):
            raise TypeError(
                f"PartsResolver expected PartService, got {type(part_service)}"
            )
        self.part_service = part_service or PartService()

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
            f"[PartsResolver] resolve | query='{query}' | entity_keys={list(entities.keys())}",
            request_id,
        )

        resolved_ids: List[int] = []

        # --------------------------------------------------
        # 0. EXPLICIT PART ID (highest precision)
        # --------------------------------------------------
        explicit_ids = entities.get("PART_ID", [])
        if explicit_ids:
            try:
                resolved_ids = [
                    int(i) for i in explicit_ids if str(i).isdigit()
                ]
                info_id(
                    f"[PartsResolver] resolved via explicit PART_ID ({len(resolved_ids)})",
                    request_id,
                )
                return sorted(set(resolved_ids))
            except Exception as e:
                warning_id(
                    f"[PartsResolver] PART_ID parsing failed: {e}",
                    request_id,
                )

        # --------------------------------------------------
        # 1. ENTITY-DRIVEN LOOKUPS (structured, precise)
        # --------------------------------------------------
        for key in ("ITEMNUM", "PARTNUM", "PART_NUMBER"):
            for value in entities.get(key, []):
                try:
                    parts = self.part_service.find(
                        part_number=value,
                        exact_match=False,
                        limit=limit,
                    )
                    resolved_ids.extend(p.id for p in parts)
                except Exception as e:
                    warning_id(
                        f"[PartsResolver] {key} lookup failed ({value}): {e}",
                        request_id,
                    )

        for key in ("MANUFACTURER", "OEMMFG"):
            for value in entities.get(key, []):
                try:
                    parts = self.part_service.find(
                        oem_mfg=value,
                        limit=limit,
                    )
                    resolved_ids.extend(p.id for p in parts)
                except Exception as e:
                    warning_id(
                        f"[PartsResolver] {key} lookup failed ({value}): {e}",
                        request_id,
                    )

        for key in ("MODEL", "MODELNUM"):
            for value in entities.get(key, []):
                try:
                    parts = self.part_service.find(
                        model=value,
                        limit=limit,
                    )
                    resolved_ids.extend(p.id for p in parts)
                except Exception as e:
                    warning_id(
                        f"[PartsResolver] {key} lookup failed ({value}): {e}",
                        request_id,
                    )

        for key in ("DESCRIPTION", "PARTDESC", "DESCRIPTION_FORMAL"):
            for value in entities.get(key, []):
                try:
                    parts = self.part_service.find(
                        search_text=value,
                        use_fts=True,
                        limit=limit,
                    )
                    resolved_ids.extend(p.id for p in parts)
                except Exception as e:
                    warning_id(
                        f"[PartsResolver] {key} lookup failed ({value}): {e}",
                        request_id,
                    )

        # --------------------------------------------------
        # 2. QUERY-BASED SEARCH (single FTS pass)
        # --------------------------------------------------
        if query and not resolved_ids:
            try:
                parts = self.part_service.find(
                    search_text=query,
                    use_fts=True,
                    limit=limit,
                )
                resolved_ids.extend(p.id for p in parts)
            except Exception as e:
                warning_id(
                    f"[PartsResolver] query FTS search failed: {e}",
                    request_id,
                )

        # --------------------------------------------------
        # 3. DE-DUPLICATE & RETURN
        # --------------------------------------------------
        unique_ids = sorted(set(resolved_ids))

        info_id(
            f"[PartsResolver] resolved {len(unique_ids)} Part IDs",
            request_id,
        )

        return unique_ids

    # --------------------------------------------------
    # DIRECT-ID PASS THROUGH (UI / API)
    # --------------------------------------------------
    def resolve_from_ids(self, ids: List[int]) -> List[int]:
        return sorted({i for i in ids if isinstance(i, int) and i > 0})
