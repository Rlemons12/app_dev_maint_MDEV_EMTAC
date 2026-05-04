# modules/services/chunk_relationship_service.py

from __future__ import annotations

from typing import Dict, Any, List, Optional, Set

from sqlalchemy.orm import Session

from modules.configuration.log_config import debug_id, warning_id


class ChunkRelationshipService:
    """
    Resolves relationships for RAG chunks.

    Relationship paths used:

    1. Document / chunk -> CompleteDocument
        document.complete_document_id

    2. CompleteDocument -> Positions
        completed_document_position_association.complete_document_id
            -> position_id
            -> position

    3. Positions -> Drawings
        drawing_position.position_id
            -> drawing

    4. CompleteDocument -> Images
        image_completed_document_association.complete_document_id
            -> image

    5. Positions -> Images
        image_position_association.position_id
            -> image

    6. Positions -> Parts
        part_position_image.position_id
            -> part

    Returns:

    {
        "forward": {
            "1st_tier": {
                "images": [...]
            },
            "2nd_tier": {
                "positions": [...],
                "parts": [...],
                "drawings": [...]
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

        chunk_ids = self._clean_ids(chunk_ids)
        document_ids = self._clean_ids(document_ids)
        complete_document_ids = self._clean_ids(complete_document_ids)

        positions: List[Dict[str, Any]] = []
        drawings: List[Dict[str, Any]] = []
        images: List[Dict[str, Any]] = []
        parts: List[Dict[str, Any]] = []

        resolved_document_ids: Set[int] = set(document_ids)
        resolved_complete_document_ids: Set[int] = set(complete_document_ids)

        # Shared image dedupe across complete-document images and position images.
        seen_images: Set[int] = set()

        # --------------------------------------------------
        # 1. Resolve complete_document_ids from document_ids
        # --------------------------------------------------
        try:
            from modules.emtacdb.emtacdb_fts import Document

            if document_ids:
                rows = (
                    session.query(Document.id, Document.complete_document_id)
                    .filter(Document.id.in_(document_ids))
                    .all()
                )

                for document_id, complete_document_id in rows:
                    if document_id:
                        resolved_document_ids.add(int(document_id))

                    if complete_document_id:
                        resolved_complete_document_ids.add(int(complete_document_id))

            if complete_document_ids:
                rows = (
                    session.query(Document.id, Document.complete_document_id)
                    .filter(Document.complete_document_id.in_(complete_document_ids))
                    .all()
                )

                for document_id, complete_document_id in rows:
                    if document_id:
                        resolved_document_ids.add(int(document_id))

                    if complete_document_id:
                        resolved_complete_document_ids.add(int(complete_document_id))

        except Exception as e:
            warning_id(
                f"[ChunkRelationshipService] Could not resolve document/complete_document ids: {e}",
                request_id,
            )

        debug_id(
            "[ChunkRelationshipService] ID resolution "
            f"chunk_ids={chunk_ids} "
            f"document_ids={sorted(resolved_document_ids)} "
            f"complete_document_ids={sorted(resolved_complete_document_ids)}",
            request_id,
        )

        # --------------------------------------------------
        # 2. Resolve positions from completed_document_position_association
        # --------------------------------------------------
        position_ids: Set[int] = set()

        try:
            from modules.emtacdb.emtacdb_fts import (
                CompletedDocumentPositionAssociation,
                Position,
            )

            if resolved_complete_document_ids:
                position_rows = (
                    session.query(Position)
                    .join(
                        CompletedDocumentPositionAssociation,
                        CompletedDocumentPositionAssociation.position_id == Position.id,
                    )
                    .filter(
                        CompletedDocumentPositionAssociation.complete_document_id.in_(
                            resolved_complete_document_ids
                        )
                    )
                    .distinct()
                    .all()
                )

                for p in position_rows:
                    position_ids.add(int(p.id))

                positions = [
                    {
                        "id": p.id,
                        "area_id": p.area_id,
                        "equipment_group_id": p.equipment_group_id,
                        "model_id": p.model_id,
                        "asset_number_id": p.asset_number_id,
                        "location_id": p.location_id,
                        "site_location_id": getattr(p, "site_location_id", None),
                        "subassembly_id": getattr(p, "subassembly_id", None),
                        "component_assembly_id": getattr(p, "component_assembly_id", None),
                        "assembly_view_id": getattr(p, "assembly_view_id", None),
                    }
                    for p in position_rows
                ]

        except Exception as e:
            warning_id(
                f"[ChunkRelationshipService] Position lookup failed: {e}",
                request_id,
            )
            positions = []
            position_ids = set()

        debug_id(
            "[ChunkRelationshipService] Position lookup "
            f"positions={len(positions)} "
            f"position_ids={sorted(position_ids)}",
            request_id,
        )

        # --------------------------------------------------
        # 3. Resolve drawings from drawing_position by position_id
        # --------------------------------------------------
        try:
            from modules.emtacdb.emtacdb_fts import (
                Drawing,
                DrawingPositionAssociation,
            )

            if position_ids:
                drawing_rows = (
                    session.query(Drawing, DrawingPositionAssociation.position_id)
                    .join(
                        DrawingPositionAssociation,
                        DrawingPositionAssociation.drawing_id == Drawing.id,
                    )
                    .filter(DrawingPositionAssociation.position_id.in_(position_ids))
                    .all()
                )

                seen_drawings: Set[int] = set()

                for drawing, position_id in drawing_rows:
                    if drawing.id in seen_drawings:
                        continue

                    seen_drawings.add(int(drawing.id))

                    drawings.append(
                        {
                            "id": drawing.id,
                            "drw_number": drawing.drw_number,
                            "drw_name": drawing.drw_name,
                            "drw_revision": drawing.drw_revision,
                            "drw_equipment_name": getattr(drawing, "drw_equipment_name", None),
                            "drw_spare_part_number": getattr(drawing, "drw_spare_part_number", None),
                            "file_path": drawing.file_path,
                            "position_id": position_id,
                        }
                    )

        except Exception as e:
            warning_id(
                f"[ChunkRelationshipService] Drawing lookup failed: {e}",
                request_id,
            )
            drawings = []

        debug_id(
            f"[ChunkRelationshipService] Drawing lookup drawings={len(drawings)}",
            request_id,
        )

        # --------------------------------------------------
        # 4A. Resolve images directly from complete_document
        #     complete_document -> image_completed_document_association -> image
        # --------------------------------------------------
        try:
            from modules.emtacdb.emtacdb_fts import (
                Image,
                ImageCompletedDocumentAssociation,
            )

            if resolved_complete_document_ids:
                complete_document_image_rows = (
                    session.query(
                        Image,
                        ImageCompletedDocumentAssociation.complete_document_id,
                    )
                    .join(
                        ImageCompletedDocumentAssociation,
                        ImageCompletedDocumentAssociation.image_id == Image.id,
                    )
                    .filter(
                        ImageCompletedDocumentAssociation.complete_document_id.in_(
                            resolved_complete_document_ids
                        )
                    )
                    .all()
                )

                debug_id(
                    "[ChunkRelationshipService] CompleteDocument image lookup "
                    f"complete_document_ids={len(resolved_complete_document_ids)} "
                    f"rows={len(complete_document_image_rows)}",
                    request_id,
                )

                for image, complete_document_id in complete_document_image_rows:
                    self._append_image_payload(
                        images=images,
                        seen_images=seen_images,
                        image=image,
                        source="complete_document",
                        complete_document_id=complete_document_id,
                        position_id=None,
                    )

        except Exception as e:
            warning_id(
                f"[ChunkRelationshipService] CompleteDocument image lookup failed: {e}",
                request_id,
            )

        # --------------------------------------------------
        # 4B. Resolve images from image_position_association by position_id
        #     position -> image_position_association -> image
        # --------------------------------------------------
        try:
            from modules.emtacdb.emtacdb_fts import Image, ImagePositionAssociation

            if position_ids:
                position_image_rows = (
                    session.query(Image, ImagePositionAssociation.position_id)
                    .join(
                        ImagePositionAssociation,
                        ImagePositionAssociation.image_id == Image.id,
                    )
                    .filter(ImagePositionAssociation.position_id.in_(position_ids))
                    .all()
                )

                debug_id(
                    "[ChunkRelationshipService] Position image lookup "
                    f"position_ids={len(position_ids)} "
                    f"rows={len(position_image_rows)}",
                    request_id,
                )

                for image, position_id in position_image_rows:
                    self._append_image_payload(
                        images=images,
                        seen_images=seen_images,
                        image=image,
                        source="position",
                        complete_document_id=None,
                        position_id=position_id,
                    )

        except Exception as e:
            warning_id(
                f"[ChunkRelationshipService] Position image lookup failed: {e}",
                request_id,
            )

        debug_id(
            f"[ChunkRelationshipService] Image lookup total images={len(images)}",
            request_id,
        )

        # --------------------------------------------------
        # 5. Resolve parts from part_position_image by position_id
        # --------------------------------------------------
        try:
            from modules.emtacdb.emtacdb_fts import Part, PartsPositionImageAssociation

            if position_ids:
                part_rows = (
                    session.query(Part, PartsPositionImageAssociation.position_id)
                    .join(
                        PartsPositionImageAssociation,
                        PartsPositionImageAssociation.part_id == Part.id,
                    )
                    .filter(PartsPositionImageAssociation.position_id.in_(position_ids))
                    .all()
                )

                seen_parts: Set[int] = set()

                for part, position_id in part_rows:
                    if part.id in seen_parts:
                        continue

                    seen_parts.add(int(part.id))

                    parts.append(
                        {
                            "id": part.id,
                            "part_number": part.part_number,
                            "name": part.name,
                            "oem_mfg": getattr(part, "oem_mfg", None),
                            "model": getattr(part, "model", None),
                            "position_id": position_id,
                        }
                    )

        except Exception as e:
            warning_id(
                f"[ChunkRelationshipService] Part lookup failed: {e}",
                request_id,
            )
            parts = []

        debug_id(
            f"[ChunkRelationshipService] resolved "
            f"(chunks={len(chunk_ids)}, "
            f"documents={len(resolved_document_ids)}, "
            f"complete_documents={len(resolved_complete_document_ids)}, "
            f"positions={len(positions)}, "
            f"drawings={len(drawings)}, "
            f"images={len(images)}, "
            f"parts={len(parts)})",
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
                    "drawings": drawings,
                },
            },
            "summary": {
                "chunks": len(chunk_ids),
                "documents": len(resolved_document_ids),
                "complete_documents": len(resolved_complete_document_ids),
                "positions": len(positions),
                "drawings": len(drawings),
                "images": len(images),
                "parts": len(parts),
            },
        }

    # --------------------------------------------------
    # Serialization helpers
    # --------------------------------------------------

    @staticmethod
    def _append_image_payload(
        *,
        images: List[Dict[str, Any]],
        seen_images: Set[int],
        image: Any,
        source: str,
        complete_document_id: Optional[int],
        position_id: Optional[int],
    ) -> None:
        """
        Appends a UI-safe image payload once.

        Important:
            src/url must be the Flask image-serving route,
            not the Windows filesystem path.
        """

        image_id = getattr(image, "id", None)

        if image_id is None:
            return

        image_id = int(image_id)

        if image_id in seen_images:
            return

        seen_images.add(image_id)

        image_url = f"/serve_image/{image_id}"

        images.append(
            {
                "id": image_id,
                "title": getattr(image, "title", None),
                "description": getattr(image, "description", None),

                # Frontend-safe route fields
                "src": image_url,
                "url": image_url,
                "href": image_url,

                # Backward compatibility:
                # If old frontend code uses file_path as a clickable link,
                # this prevents it from requesting /DB_IMAGES/...
                "file_path": image_url,

                "source": source,
                "complete_document_id": complete_document_id,
                "position_id": position_id,
            }
        )

    @staticmethod
    def _clean_ids(values: Optional[List[int]]) -> List[int]:
        cleaned: List[int] = []

        if not values:
            return cleaned

        for value in values:
            try:
                if value is not None:
                    cleaned.append(int(value))
            except Exception:
                continue

        return list(dict.fromkeys(cleaned))