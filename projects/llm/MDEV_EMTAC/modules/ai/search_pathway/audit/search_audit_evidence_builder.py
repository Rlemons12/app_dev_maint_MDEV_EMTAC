from __future__ import annotations

from typing import Any


class SearchAuditEvidenceBuilder:
    """
    Builds relationship/evidence data explaining why a payload item was returned.

    This starts simple and can grow as the RAG relationship map becomes more formal.
    """

    @staticmethod
    def build_basic_evidence(
        *,
        item: dict[str, Any],
        item_type: str,
        relationship_map: dict[str, Any] | None = None,
        used_chunk_ids: list[int] | None = None,
    ) -> dict[str, Any]:
        return {
            "item_type": item_type,
            "payload_item": item,
            "used_chunk_ids": used_chunk_ids or [],
            "relationship_map": relationship_map or {},
        }
