from typing import Dict, Any, List

from modules.emtac_ai.search.rag_core.document_ui_payload import DocumentUIPayload
from modules.services.drawing_navigation_projection import DrawingNavigationProjection



class ChunkRelationshipProjection:

    def __init__(self, session):
        self.session = session

    def project_chunks_for_ui(
        self,
        chunks: List[Dict[str, Any]],
        relationship_map: Dict[str, Any],
    ) -> Dict[str, Any]:

        if not chunks:
            return {"documents-container": []}

        forward = relationship_map.get("forward", {})
        first = forward.get("1st_tier", {})
        second = forward.get("2nd_tier", {})

        payload = DocumentUIPayload()
        payload.aggregate_from_chunks(chunks)
        drawing_nav = DrawingNavigationProjection(session=self.session)

        # --------------------------------------------------
        # IMAGES (already serialized → pass-through)
        # --------------------------------------------------
        images = first.get("images", [])
        if images:
            for doc in payload._documents.values():
                doc["images"] = images

        # --------------------------------------------------
        # POSITIONS (already serialized → pass-through)
        # --------------------------------------------------
        positions = second.get("positions", [])
        if positions:
            for doc in payload._documents.values():
                doc["positions"] = positions
                doc["position_ids"] = [
                    p["id"] for p in positions if isinstance(p, dict) and "id" in p
                ]
        # --------------------------------------------------
        # DRAWING NAVIGATION (Area → Model → Asset)
        # --------------------------------------------------
        for doc in payload._documents.values():
            positions = doc.get("positions", [])
            position_ids = [
                p["id"] for p in positions
                if isinstance(p, dict) and p.get("id")
            ]

            if not position_ids:
                debug_id("[UI PROJECTION] No valid position_ids for drawing navigation", None)
                continue

            doc["drawing_navigation"] = drawing_nav.build_navigation(
                complete_document_id=doc["complete_document_id"],
                position_ids=position_ids,
            )

        # --------------------------------------------------
        # PARTS (already serialized → pass-through)
        # --------------------------------------------------
        parts = second.get("parts", [])
        if parts:
            for doc in payload._documents.values():
                doc["parts"] = parts

        # --------------------------------------------------
        # OPTIONAL PANELS
        # --------------------------------------------------
        for key in ["drawings", "tasks", "tools", "problems"]:
            items = second.get(key, [])
            if items:
                for doc in payload._documents.values():
                    doc[key] = items

        return {
            "documents-container": payload.build(),
            "summary": relationship_map.get("summary", {}),
        }
