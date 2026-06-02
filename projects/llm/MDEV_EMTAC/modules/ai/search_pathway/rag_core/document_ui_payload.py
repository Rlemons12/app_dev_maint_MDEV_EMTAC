from __future__ import annotations
from typing import List, Dict, Any, Optional

from modules.configuration.log_config import debug_id, with_request_id
from modules.services.drawing_part_association_service import DrawingPartAssociationService


class DocumentUIPayload:
    """
    Orchestrates construction of UI-ready CompleteDocument payloads.
    """

    def __init__(self, session=None):
        self.session = session
        self._documents: Dict[int, Dict[str, Any]] = {}

    # ---------------------------------------------------------
    # Serialization helpers (STATIC)
    # ---------------------------------------------------------
    @staticmethod
    def _serialize_image(i):
        """
        UI-safe image serializer.

        - NEVER exposes filesystem paths
        - ALWAYS emits media-route URL
        """
        return {
            "id": i.id,
            "title": i.title,
            "description": i.description,
            "src": f"/images/{i.id}",
        }

    @staticmethod
    def _serialize_position(pos):
        return {
            "id": pos.id,
            "area_id": pos.area_id,
            "equipment_group_id": pos.equipment_group_id,
            "model_id": pos.model_id,
            "asset_number_id": pos.asset_number_id,
            "location_id": pos.location_id,
            "site_location_id": pos.site_location_id,
        }

    @staticmethod
    def _serialize_part(p):
        return {
            "id": p.id,
            "part_number": p.part_number,
            "name": p.name,
            "oem_mfg": p.oem_mfg,
            "model": p.model,
        }

    # ---------------------------------------------------------
    # Step 1: Aggregate chunks
    # ---------------------------------------------------------
    @with_request_id
    def aggregate_from_chunks(
            self,
            chunks: List[Dict[str, Any]],
            request_id: Optional[str] = None,
    ) -> "DocumentUIPayload":

        for ch in chunks:
            if not isinstance(ch, dict):
                continue

            complete_document_id = ch.get("complete_document_id")
            if not complete_document_id:
                continue

            document_title = (
                    ch.get("complete_document_title")
                    or ch.get("document_title")
                    or ch.get("document_name")
                    or ch.get("name")
                    or ch.get("display_name")
                    or ch.get("title")
                    or self._filename_from_path(ch.get("file_path"))
                    or f"Document #{complete_document_id}"
            )

            doc = self._documents.setdefault(
                complete_document_id,
                {
                    "document_id": complete_document_id,
                    "complete_document_id": complete_document_id,

                    # Main display fields for frontend compatibility
                    "title": document_title,
                    "document_title": document_title,
                    "document_name": document_title,
                    "complete_document_title": document_title,
                    "display_name": document_title,
                    "name": document_title,

                    "url": ch.get("url"),
                    "file_path": ch.get("file_path"),
                    "chunks": [],
                },
            )

            # If the document already existed but had a fallback title,
            # upgrade it when a better title appears later.
            if (
                    document_title
                    and not str(document_title).startswith("Document #")
                    and str(doc.get("title", "")).startswith("Document #")
            ):
                doc["title"] = document_title
                doc["document_title"] = document_title
                doc["document_name"] = document_title
                doc["complete_document_title"] = document_title
                doc["display_name"] = document_title
                doc["name"] = document_title

            doc["chunks"].append(
                {
                    "chunk_id": ch.get("chunk_id") or ch.get("id"),
                    "chunk_document_id": ch.get("document_id"),
                    "complete_document_id": complete_document_id,
                    "text": (
                            ch.get("content")
                            or ch.get("text")
                            or ch.get("chunk_text")
                            or ch.get("page_content")
                    ),
                    "score": ch.get("distance") or ch.get("score"),
                }
            )

        debug_id(
            f"[DocumentUIPayload] Aggregated {len(self._documents)} documents",
            request_id,
        )
        return self

    # ---------------------------------------------------------
    # Step 2: Images
    # ---------------------------------------------------------
    @with_request_id
    def enrich_with_images(
            self,
            image_assoc_service,
            request_id: Optional[str] = None,
    ) -> "DocumentUIPayload":

        for doc in self._documents.values():
            if not self.session:
                raise RuntimeError("DocumentUIPayload requires session for image enrichment")

            resolved = image_assoc_service.resolve_related_entities(
                session=self.session,
                complete_document_id=doc["complete_document_id"],
            )

            images = resolved.get("images", [])

            # ✅ EXPLICIT SERIALIZATION (FIX)
            doc["images"] = [self._serialize_image(i) for i in images]

            debug_id(
                f"[DocumentUIPayload] Document {doc['complete_document_id']} "
                f"images={len(images)}",
                request_id,
            )

        return self

    # ---------------------------------------------------------
    # Step 3: Positions
    # ---------------------------------------------------------
    @with_request_id
    def enrich_with_positions(
        self,
        position_service,
        session,
        request_id: Optional[str] = None,
    ) -> "DocumentUIPayload":

        for doc in self._documents.values():
            positions, position_ids = position_service.get_positions_for_complete_document(
                complete_document_id=doc["complete_document_id"],
                session=session,
            )

            doc["positions"] = positions
            doc["position_ids"] = position_ids

        return self

    # ---------------------------------------------------------
    # Step 4: Parts
    # ---------------------------------------------------------
    @with_request_id
    def enrich_with_parts(
            self,
            parts_position_image_service,
            session,
            request_id: Optional[str] = None,
    ) -> "DocumentUIPayload":

        drawing_part_service = DrawingPartAssociationService()

        for doc in self._documents.values():
            position_ids = doc.get("position_ids", [])
            if not position_ids:
                continue

            # ----------------------------------
            # Fetch ALL part ↔ image associations
            # ----------------------------------
            associations = parts_position_image_service.search_by_positions(
                session=session,
                position_ids=position_ids,
            )

            parts_by_id = {}
            images_by_id = {}

            for assoc in associations:
                if assoc.part:
                    parts_by_id[assoc.part.id] = assoc.part
                if assoc.image:
                    images_by_id[assoc.image.id] = assoc.image

            if not parts_by_id:
                continue

            # ----------------------------------
            # 🔑 Fetch drawings for ALL parts (single query)
            # ----------------------------------
            part_ids = list(parts_by_id.keys())

            drawing_map = drawing_part_service.get_drawing_numbers_by_part_ids(
                part_ids=part_ids,
                session=session,
            )
            # drawing_map example:
            # { part_id: ["DRW-1001", "DRW-1002"] }

            # ----------------------------------
            # Serialize parts WITH drawings
            # ----------------------------------
            serialized_parts = []

            for part in parts_by_id.values():
                part_payload = self._serialize_part(part)
                part_payload["drawing_numbers"] = drawing_map.get(part.id, [])
                serialized_parts.append(part_payload)

            doc["parts"] = serialized_parts

            # ----------------------------------
            # Serialize images
            # ----------------------------------
            doc["part_images"] = [
                self._serialize_image(i)
                for i in images_by_id.values()
            ]

            # ----------------------------------
            # UI panel contract (unchanged)
            # ----------------------------------
            doc["parts_panel"] = {
                "parts": doc["parts"],
                "images": doc["part_images"],
            }

        return self

    # ---------------------------------------------------------
    # Final output
    # ---------------------------------------------------------
    def build(self) -> List[Dict[str, Any]]:
        return list(self._documents.values())

    # ---------------------------------------------------------
    # Step 4b: Drawings (Document → Position → Drawings)
    # ---------------------------------------------------------
    # ---------------------------------------------------------
    # Step 4b: Drawings (Position → Drawings)
    # ---------------------------------------------------------
    @with_request_id
    def enrich_with_drawings(
            self,
            session,
            request_id: Optional[str] = None,
    ) -> "DocumentUIPayload":

        from modules.services.drawing_navigation_projection import (
            DrawingNavigationProjection,
        )

        for doc in self._documents.values():
            position_ids = doc.get("position_ids", [])
            if not position_ids:
                doc["drawing_navigation"] = {
                    "areas": [],
                    "meta": {
                        "area_count": 0,
                        "model_count": 0,
                        "asset_count": 0,
                        "drawing_count": 0,
                    },
                }
                continue

            projection = DrawingNavigationProjection(session=session)

            nav = projection.build_navigation(
                complete_document_id=doc.get("complete_document_id"),
                position_ids=position_ids,
            )

            doc["drawing_navigation"] = nav

            debug_id(
                f"[DocumentUIPayload] Document {doc['complete_document_id']} "
                f"drawing_navigation areas={len(nav.get('areas', []))}",
                request_id,
            )

        return self

    @staticmethod
    def _filename_from_path(file_path: Optional[str]) -> Optional[str]:
        if not file_path:
            return None

        filename = str(file_path).replace("\\", "/").split("/")[-1].strip()
        return filename or None
