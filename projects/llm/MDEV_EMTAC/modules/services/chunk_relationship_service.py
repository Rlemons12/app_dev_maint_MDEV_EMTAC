from __future__ import annotations

from typing import Dict, Any, List, Optional

from sqlalchemy.orm import Session

from modules.configuration.log_config import debug_id


class ChunkRelationshipService:
    """
    Resolves relationships for RAG chunks.

    Returns a normalized relationship_map structure:

    {
        "forward": {
            "1st_tier": {
                "images": [...]
            },
            "2nd_tier": {
                "positions": [...],
                "parts": [...],
                "drawings": []
            }
        },
        "summary": {}
    }
    """

    def resolve(
        self,
        *,
        session: Session,
        chunk_ids: List[int],
        document_ids: List[int],
        complete_document_ids: List[int],
        request_id: Optional[str] = None,
    ) -> Dict[str, Any]:

        # --------------------------------------------
        # TODO: Replace with real queries
        # --------------------------------------------

        images = []
        positions = []
        parts = []

        # --------------------------------------------------
        # Example: Fetch positions (basic)
        # --------------------------------------------------
        try:
            from modules.emtacdb.emtacdb_fts import Position

            if complete_document_ids:
                positions = (
                    session.query(Position)
                    .filter(Position.id.in_(complete_document_ids))
                    .all()
                )

                positions = [
                    {
                        "id": p.id,
                        "area_id": p.area_id,
                        "equipment_group_id": p.equipment_group_id,
                        "model_id": p.model_id,
                        "asset_number_id": p.asset_number_id,
                        "location_id": p.location_id,
                    }
                    for p in positions
                ]

        except Exception:
            positions = []

        # --------------------------------------------------
        # Example: Images (placeholder)
        # --------------------------------------------------
        try:
            from modules.emtacdb.emtacdb_fts import Image

            if complete_document_ids:
                imgs = (
                    session.query(Image)
                    .limit(10)
                    .all()
                )

                images = [
                    {
                        "id": i.id,
                        "title": i.title,
                        "description": i.description,
                        "src": f"/images/{i.id}",
                    }
                    for i in imgs
                ]

        except Exception:
            images = []

        # --------------------------------------------------
        # Example: Parts (placeholder)
        # --------------------------------------------------
        try:
            from modules.emtacdb.emtacdb_fts import Part

            parts = (
                session.query(Part)
                .limit(10)
                .all()
            )

            parts = [
                {
                    "id": p.id,
                    "part_number": p.part_number,
                    "name": p.name,
                }
                for p in parts
            ]

        except Exception:
            parts = []

        debug_id(
            f"[ChunkRelationshipService] resolved "
            f"(images={len(images)}, positions={len(positions)}, parts={len(parts)})",
            request_id,
        )

        return {
            "forward": {
                "1st_tier": {
                    "images": images,
                },
                "2nd_tier": {
                    "positions": positions,
                    "parts": parts,
                    "drawings": [],
                },
            },
            "summary": {
                "images": len(images),
                "positions": len(positions),
                "parts": len(parts),
            },
        }