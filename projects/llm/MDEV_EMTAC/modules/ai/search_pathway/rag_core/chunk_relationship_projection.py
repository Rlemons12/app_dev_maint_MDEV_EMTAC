from typing import Dict, Any, List, Optional

from modules.ai.search_pathway.rag_core.document_ui_payload import DocumentUIPayload
from modules.services.drawing_navigation_projection import DrawingNavigationProjection
from modules.configuration.log_config import debug_id, warning_id


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

        self._enrich_document_names(payload=payload, chunks=chunks)

        drawing_nav = DrawingNavigationProjection(session=self.session)

        images = first.get("images", [])
        if images:
            for doc in payload._documents.values():
                doc["images"] = images

        positions = second.get("positions", [])
        if positions:
            for doc in payload._documents.values():
                doc["positions"] = positions
                doc["position_ids"] = [
                    p["id"] for p in positions
                    if isinstance(p, dict) and "id" in p
                ]

        for doc in payload._documents.values():
            positions = doc.get("positions", [])
            position_ids = [
                p["id"] for p in positions
                if isinstance(p, dict) and p.get("id")
            ]

            if not position_ids:
                debug_id("[UI PROJECTION] No valid position_ids for drawing navigation", None)
                continue

            complete_document_id = doc.get("complete_document_id")

            if not complete_document_id:
                debug_id("[UI PROJECTION] Missing complete_document_id for drawing navigation", None)
                continue

            doc["drawing_navigation"] = drawing_nav.build_navigation(
                complete_document_id=complete_document_id,
                position_ids=position_ids,
            )

        parts = second.get("parts", [])
        if parts:
            for doc in payload._documents.values():
                doc["parts"] = parts

        for key in ["drawings", "tasks", "tools", "problems"]:
            items = second.get(key, [])
            if items:
                for doc in payload._documents.values():
                    doc[key] = items

        return {
            "documents-container": payload.build(),
            "summary": relationship_map.get("summary", {}),
        }

    def _enrich_document_names(
        self,
        *,
        payload: DocumentUIPayload,
        chunks: List[Dict[str, Any]],
    ) -> None:
        """
        Ensures document cards display the complete document name/title instead
        of only the chunk/document id.

        This patches payload._documents after DocumentUIPayload aggregates chunks.
        """

        complete_document_ids = set()

        for doc in payload._documents.values():
            complete_document_id = self._safe_int(doc.get("complete_document_id"))

            if complete_document_id is not None:
                complete_document_ids.add(complete_document_id)

        for chunk in chunks or []:
            if not isinstance(chunk, dict):
                continue

            complete_document_id = self._safe_int(
                chunk.get("complete_document_id")
                or chunk.get("completed_document_id")
                or chunk.get("completeDocumentId")
            )

            if complete_document_id is not None:
                complete_document_ids.add(complete_document_id)

        if not complete_document_ids:
            return

        title_lookup = self._load_complete_document_titles(
            complete_document_ids=sorted(complete_document_ids),
        )

        for doc in payload._documents.values():
            complete_document_id = self._safe_int(doc.get("complete_document_id"))

            if complete_document_id is None:
                continue

            title = title_lookup.get(complete_document_id)

            if not title:
                continue

            doc["complete_document_title"] = title
            doc["document_title"] = title
            doc["document_name"] = title
            doc["title"] = title
            doc["name"] = title
            doc["display_name"] = title

    def _load_complete_document_titles(
        self,
        *,
        complete_document_ids: List[int],
    ) -> Dict[int, str]:

        if not complete_document_ids:
            return {}

        try:
            from modules.emtacdb.emtacdb_fts import CompleteDocument

            rows = (
                self.session.query(CompleteDocument)
                .filter(CompleteDocument.id.in_(complete_document_ids))
                .all()
            )

            lookup: Dict[int, str] = {}

            for complete_doc in rows:
                complete_document_id = self._safe_int(getattr(complete_doc, "id", None))

                if complete_document_id is None:
                    continue

                title = self._complete_document_title(
                    complete_doc=complete_doc,
                    fallback=f"Document #{complete_document_id}",
                )

                lookup[complete_document_id] = title

            return lookup

        except Exception as exc:
            warning_id(
                f"[ChunkRelationshipProjection] Failed to load CompleteDocument titles: {exc}",
                None,
            )
            return {}

    @staticmethod
    def _complete_document_title(
        *,
        complete_doc: Any,
        fallback: str,
    ) -> str:

        if complete_doc is None:
            return fallback

        for attr_name in (
            "title",
            "name",
            "document_name",
            "file_name",
            "filename",
            "original_filename",
        ):
            value = getattr(complete_doc, attr_name, None)

            if value:
                return str(value)

        file_path = getattr(complete_doc, "file_path", None)

        if file_path:
            normalized_path = str(file_path).replace("\\", "/")
            return normalized_path.split("/")[-1] or fallback

        return fallback

    @staticmethod
    def _safe_int(value: Any) -> Optional[int]:
        if value in (None, "", "None"):
            return None

        if isinstance(value, bool):
            return None

        try:
            return int(value)
        except (TypeError, ValueError):
            return None