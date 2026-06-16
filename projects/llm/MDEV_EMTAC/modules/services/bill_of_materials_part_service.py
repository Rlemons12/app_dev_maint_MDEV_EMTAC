from __future__ import annotations

from typing import Any, Dict, List, Optional

from modules.configuration.log_config import logger, with_request_id
from modules.emtacdb.emtacdb_fts import Part


class BillOfMaterialsPartService:
    """
    Domain service for Part read/update operations.

    HARD RULES:
    - no session creation
    - no session closing
    - no commit
    - no rollback
    """

    @with_request_id
    def get_part_by_id(self, *, session, part_id: int) -> Optional[Part]:
        logger.debug("Fetching part by id=%s", part_id)
        return Part.get_by_id(part_id=part_id, session=session)

    @with_request_id
    def search_parts(
        self,
        *,
        session,
        search_text: str,
        limit: int = 10,
        use_fts: bool = False,
    ) -> List[Part]:
        normalized_search_text = (search_text or "").strip()

        logger.debug(
            "Searching parts with search_text=%s limit=%s use_fts=%s",
            normalized_search_text,
            limit,
            use_fts,
        )

        return Part.search(
            search_text=normalized_search_text,
            session=session,
            limit=limit,
            use_fts=use_fts,
        )

    @with_request_id
    def update_part_fields(
        self,
        *,
        part: Part,
        part_fields: Dict[str, Any],
    ) -> None:
        old_values = {
            "part_number": part.part_number,
            "name": part.name,
            "oem_mfg": part.oem_mfg,
            "model": part.model,
            "class_flag": part.class_flag,
            "ud6": part.ud6,
            "type": part.type,
            "notes": part.notes,
            "documentation": part.documentation,
        }

        part.part_number = part_fields.get("part_number")
        part.name = part_fields.get("name")
        part.oem_mfg = part_fields.get("oem_mfg")
        part.model = part_fields.get("model")
        part.class_flag = part_fields.get("class_flag")
        part.ud6 = part_fields.get("ud6")
        part.type = part_fields.get("type")
        part.notes = part_fields.get("notes")
        part.documentation = part_fields.get("documentation")

        new_values = {
            "part_number": part.part_number,
            "name": part.name,
            "oem_mfg": part.oem_mfg,
            "model": part.model,
            "class_flag": part.class_flag,
            "ud6": part.ud6,
            "type": part.type,
            "notes": part.notes,
            "documentation": part.documentation,
        }

        for key, old_value in old_values.items():
            new_value = new_values.get(key)
            if old_value != new_value:
                logger.info(
                    'Updated part %s field %s: "%s" -> "%s"',
                    part.id,
                    key,
                    old_value,
                    new_value,
                )