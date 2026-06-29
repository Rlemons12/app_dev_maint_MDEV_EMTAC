# modules/services/chunk_relationship_service.py

from __future__ import annotations

import time
from typing import Dict, Any, List, Optional, Set, Tuple

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

    Performance notes:
    - This service returns UI-safe dictionaries.
    - It queries only the columns needed for payload projection.
    - It avoids loading full ORM objects where possible.
    """

    # Keeping this false avoids a potentially expensive query:
    #
    #   SELECT document.id
    #   FROM document
    #   WHERE complete_document_id IN (...)
    #
    # For relationship payload building, complete_document_ids are enough.
    # resolved_document_ids are currently used only for summary/debug counts.
    EXPAND_COMPLETE_DOCUMENTS_TO_DOCUMENT_IDS = False

    def resolve(
        self,
        *,
        session: Session,
        chunk_ids: List[int],
        document_ids: List[int],
        complete_document_ids: List[int],
        request_id: Optional[str] = None,
    ) -> Dict[str, Any]:

        start_total = time.perf_counter()

        chunk_ids = self._clean_ids(chunk_ids)
        document_ids = self._clean_ids(document_ids)
        complete_document_ids = self._clean_ids(complete_document_ids)

        positions: List[Dict[str, Any]] = []
        drawings: List[Dict[str, Any]] = []
        images: List[Dict[str, Any]] = []
        parts: List[Dict[str, Any]] = []

        resolved_document_ids: Set[int] = set(document_ids)
        resolved_complete_document_ids: Set[int] = set(complete_document_ids)

        seen_images: Set[int] = set()

        # --------------------------------------------------
        # 1. Resolve complete_document_ids from document/chunk IDs
        # --------------------------------------------------
        document_lookup_start = time.perf_counter()

        try:
            from modules.emtacdb.emtacdb_fts import Document

            # In this project, chunk_id and document_id may both point back to
            # Document.id depending on where the seed came from.
            candidate_document_ids = self._clean_ids(chunk_ids + document_ids)

            if candidate_document_ids:
                rows = (
                    session.query(
                        Document.id.label("document_id"),
                        Document.complete_document_id.label("complete_document_id"),
                    )
                    .filter(Document.id.in_(candidate_document_ids))
                    .all()
                )

                for row in rows:
                    document_id = getattr(row, "document_id", None)
                    complete_document_id = getattr(row, "complete_document_id", None)

                    if document_id is not None:
                        resolved_document_ids.add(int(document_id))

                    if complete_document_id is not None:
                        resolved_complete_document_ids.add(int(complete_document_id))

            if (
                self.EXPAND_COMPLETE_DOCUMENTS_TO_DOCUMENT_IDS
                and resolved_complete_document_ids
            ):
                rows = (
                    session.query(
                        Document.id.label("document_id"),
                        Document.complete_document_id.label("complete_document_id"),
                    )
                    .filter(Document.complete_document_id.in_(resolved_complete_document_ids))
                    .all()
                )

                for row in rows:
                    document_id = getattr(row, "document_id", None)
                    complete_document_id = getattr(row, "complete_document_id", None)

                    if document_id is not None:
                        resolved_document_ids.add(int(document_id))

                    if complete_document_id is not None:
                        resolved_complete_document_ids.add(int(complete_document_id))

        except Exception as exc:
            warning_id(
                f"[ChunkRelationshipService] Could not resolve document/complete_document ids: {exc}",
                request_id,
            )

        document_lookup_time = time.perf_counter() - document_lookup_start

        debug_id(
            "[ChunkRelationshipService] ID resolution "
            f"chunk_ids={chunk_ids} "
            f"document_ids={sorted(resolved_document_ids)} "
            f"complete_document_ids={sorted(resolved_complete_document_ids)} "
            f"time={document_lookup_time:.3f}s",
            request_id,
        )

        # --------------------------------------------------
        # 2. Resolve positions from completed_document_position_association
        # --------------------------------------------------
        position_lookup_start = time.perf_counter()
        position_ids: Set[int] = set()

        try:
            from modules.emtacdb.emtacdb_fts import (
                CompletedDocumentPositionAssociation,
                Position,
            )

            if resolved_complete_document_ids:
                position_columns = self._position_columns(Position)

                rows = (
                    session.query(*position_columns)
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

                for row in rows:
                    payload = self._row_to_position_payload(row)
                    position_id = payload.get("id")

                    if position_id is None:
                        continue

                    position_id = int(position_id)
                    position_ids.add(position_id)
                    positions.append(payload)

        except Exception as exc:
            warning_id(
                f"[ChunkRelationshipService] Position lookup failed: {exc}",
                request_id,
            )
            positions = []
            position_ids = set()

        position_lookup_time = time.perf_counter() - position_lookup_start

        debug_id(
            "[ChunkRelationshipService] Position lookup "
            f"positions={len(positions)} "
            f"position_ids={sorted(position_ids)} "
            f"time={position_lookup_time:.3f}s",
            request_id,
        )

        # --------------------------------------------------
        # 3. Resolve drawings from drawing_position by position_id
        # --------------------------------------------------
        drawing_lookup_start = time.perf_counter()

        try:
            from modules.emtacdb.emtacdb_fts import (
                Drawing,
                DrawingPositionAssociation,
            )

            if position_ids:
                drawing_columns = self._drawing_columns(
                    Drawing=Drawing,
                    DrawingPositionAssociation=DrawingPositionAssociation,
                )

                rows = (
                    session.query(*drawing_columns)
                    .join(
                        DrawingPositionAssociation,
                        DrawingPositionAssociation.drawing_id == Drawing.id,
                    )
                    .filter(DrawingPositionAssociation.position_id.in_(position_ids))
                    .all()
                )

                seen_drawings: Set[int] = set()

                for row in rows:
                    drawing_id = getattr(row, "id", None)

                    if drawing_id is None:
                        continue

                    drawing_id = int(drawing_id)

                    if drawing_id in seen_drawings:
                        continue

                    seen_drawings.add(drawing_id)
                    drawings.append(self._row_to_drawing_payload(row))

        except Exception as exc:
            warning_id(
                f"[ChunkRelationshipService] Drawing lookup failed: {exc}",
                request_id,
            )
            drawings = []

        drawing_lookup_time = time.perf_counter() - drawing_lookup_start

        debug_id(
            "[ChunkRelationshipService] Drawing lookup "
            f"drawings={len(drawings)} "
            f"time={drawing_lookup_time:.3f}s",
            request_id,
        )

        # --------------------------------------------------
        # 4A. Resolve images directly from complete_document
        # --------------------------------------------------
        complete_document_image_start = time.perf_counter()

        try:
            from modules.emtacdb.emtacdb_fts import (
                Image,
                ImageCompletedDocumentAssociation,
            )

            if resolved_complete_document_ids:
                image_columns = self._complete_document_image_columns(
                    Image=Image,
                    ImageCompletedDocumentAssociation=ImageCompletedDocumentAssociation,
                )

                rows = (
                    session.query(*image_columns)
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
                    f"rows={len(rows)}",
                    request_id,
                )

                for row in rows:
                    self._append_image_payload_from_row(
                        images=images,
                        seen_images=seen_images,
                        row=row,
                        source="complete_document",
                        complete_document_id=getattr(row, "complete_document_id", None),
                        position_id=None,
                    )

        except Exception as exc:
            warning_id(
                f"[ChunkRelationshipService] CompleteDocument image lookup failed: {exc}",
                request_id,
            )

        complete_document_image_time = time.perf_counter() - complete_document_image_start

        # --------------------------------------------------
        # 4B. Resolve images from image_position_association by position_id
        # --------------------------------------------------
        position_image_start = time.perf_counter()

        try:
            from modules.emtacdb.emtacdb_fts import Image, ImagePositionAssociation

            if position_ids:
                image_columns = self._position_image_columns(
                    Image=Image,
                    ImagePositionAssociation=ImagePositionAssociation,
                )

                rows = (
                    session.query(*image_columns)
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
                    f"rows={len(rows)}",
                    request_id,
                )

                for row in rows:
                    self._append_image_payload_from_row(
                        images=images,
                        seen_images=seen_images,
                        row=row,
                        source="position",
                        complete_document_id=None,
                        position_id=getattr(row, "position_id", None),
                    )

        except Exception as exc:
            warning_id(
                f"[ChunkRelationshipService] Position image lookup failed: {exc}",
                request_id,
            )

        position_image_time = time.perf_counter() - position_image_start

        debug_id(
            "[ChunkRelationshipService] Image lookup "
            f"images={len(images)} "
            f"complete_document_image_time={complete_document_image_time:.3f}s "
            f"position_image_time={position_image_time:.3f}s",
            request_id,
        )

        # --------------------------------------------------
        # 5. Resolve parts from part_position_image by position_id
        # --------------------------------------------------
        part_lookup_start = time.perf_counter()

        try:
            from modules.emtacdb.emtacdb_fts import Part, PartsPositionImageAssociation

            if position_ids:
                rows = (
                    session.query(
                        Part.id.label("id"),
                        Part.part_number.label("part_number"),
                        Part.name.label("name"),
                        Part.oem_mfg.label("oem_mfg"),
                        Part.model.label("model"),
                        PartsPositionImageAssociation.position_id.label("position_id"),
                    )
                    .join(
                        PartsPositionImageAssociation,
                        PartsPositionImageAssociation.part_id == Part.id,
                    )
                    .filter(PartsPositionImageAssociation.position_id.in_(position_ids))
                    .all()
                )

                seen_parts: Set[int] = set()

                for row in rows:
                    part_id = getattr(row, "id", None)

                    if part_id is None:
                        continue

                    part_id = int(part_id)

                    if part_id in seen_parts:
                        continue

                    seen_parts.add(part_id)

                    parts.append(
                        {
                            "id": part_id,
                            "part_number": getattr(row, "part_number", None),
                            "name": getattr(row, "name", None),
                            "oem_mfg": getattr(row, "oem_mfg", None),
                            "model": getattr(row, "model", None),
                            "position_id": getattr(row, "position_id", None),
                        }
                    )

        except Exception as exc:
            warning_id(
                f"[ChunkRelationshipService] Part lookup failed: {exc}",
                request_id,
            )
            parts = []

        part_lookup_time = time.perf_counter() - part_lookup_start
        total_time = time.perf_counter() - start_total

        debug_id(
            "[ChunkRelationshipService] resolved "
            f"(chunks={len(chunk_ids)}, "
            f"documents={len(resolved_document_ids)}, "
            f"complete_documents={len(resolved_complete_document_ids)}, "
            f"positions={len(positions)}, "
            f"drawings={len(drawings)}, "
            f"images={len(images)}, "
            f"parts={len(parts)}, "
            f"times={{document={document_lookup_time:.3f}s, "
            f"positions={position_lookup_time:.3f}s, "
            f"drawings={drawing_lookup_time:.3f}s, "
            f"complete_doc_images={complete_document_image_time:.3f}s, "
            f"position_images={position_image_time:.3f}s, "
            f"parts={part_lookup_time:.3f}s, "
            f"total={total_time:.3f}s}})",
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
                "timings": {
                    "document_lookup_time": document_lookup_time,
                    "position_lookup_time": position_lookup_time,
                    "drawing_lookup_time": drawing_lookup_time,
                    "complete_document_image_time": complete_document_image_time,
                    "position_image_time": position_image_time,
                    "part_lookup_time": part_lookup_time,
                    "total_time": total_time,
                },
            },
        }

    # --------------------------------------------------
    # Column helpers
    # --------------------------------------------------

    @staticmethod
    def _position_columns(Position: Any) -> List[Any]:
        columns = [
            Position.id.label("id"),
            Position.area_id.label("area_id"),
            Position.equipment_group_id.label("equipment_group_id"),
            Position.model_id.label("model_id"),
            Position.asset_number_id.label("asset_number_id"),
            Position.location_id.label("location_id"),
        ]

        for optional_name in (
            "site_location_id",
            "subassembly_id",
            "component_assembly_id",
            "assembly_view_id",
        ):
            column = getattr(Position, optional_name, None)
            if column is not None:
                columns.append(column.label(optional_name))

        return columns

    @staticmethod
    def _drawing_columns(
        *,
        Drawing: Any,
        DrawingPositionAssociation: Any,
    ) -> List[Any]:
        columns = [
            Drawing.id.label("id"),
            Drawing.drw_number.label("drw_number"),
            Drawing.drw_name.label("drw_name"),
            Drawing.drw_revision.label("drw_revision"),
            Drawing.file_path.label("file_path"),
            DrawingPositionAssociation.position_id.label("position_id"),
        ]

        for optional_name in (
            "drw_equipment_name",
            "drw_spare_part_number",
        ):
            column = getattr(Drawing, optional_name, None)
            if column is not None:
                columns.append(column.label(optional_name))

        return columns

    @staticmethod
    def _complete_document_image_columns(
        *,
        Image: Any,
        ImageCompletedDocumentAssociation: Any,
    ) -> List[Any]:
        return [
            Image.id.label("id"),
            Image.title.label("title"),
            Image.description.label("description"),
            ImageCompletedDocumentAssociation.complete_document_id.label(
                "complete_document_id"
            ),
        ]

    @staticmethod
    def _position_image_columns(
        *,
        Image: Any,
        ImagePositionAssociation: Any,
    ) -> List[Any]:
        return [
            Image.id.label("id"),
            Image.title.label("title"),
            Image.description.label("description"),
            ImagePositionAssociation.position_id.label("position_id"),
        ]

    # --------------------------------------------------
    # Row serialization helpers
    # --------------------------------------------------

    @staticmethod
    def _row_to_position_payload(row: Any) -> Dict[str, Any]:
        return {
            "id": getattr(row, "id", None),
            "area_id": getattr(row, "area_id", None),
            "equipment_group_id": getattr(row, "equipment_group_id", None),
            "model_id": getattr(row, "model_id", None),
            "asset_number_id": getattr(row, "asset_number_id", None),
            "location_id": getattr(row, "location_id", None),
            "site_location_id": getattr(row, "site_location_id", None),
            "subassembly_id": getattr(row, "subassembly_id", None),
            "component_assembly_id": getattr(row, "component_assembly_id", None),
            "assembly_view_id": getattr(row, "assembly_view_id", None),
        }

    @staticmethod
    def _row_to_drawing_payload(row: Any) -> Dict[str, Any]:
        return {
            "id": getattr(row, "id", None),
            "drw_number": getattr(row, "drw_number", None),
            "drw_name": getattr(row, "drw_name", None),
            "drw_revision": getattr(row, "drw_revision", None),
            "drw_equipment_name": getattr(row, "drw_equipment_name", None),
            "drw_spare_part_number": getattr(row, "drw_spare_part_number", None),
            "file_path": getattr(row, "file_path", None),
            "position_id": getattr(row, "position_id", None),
        }

    @staticmethod
    def _append_image_payload_from_row(
        *,
        images: List[Dict[str, Any]],
        seen_images: Set[int],
        row: Any,
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

        image_id = getattr(row, "id", None)

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
                "title": getattr(row, "title", None),
                "description": getattr(row, "description", None),

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

    # --------------------------------------------------
    # General helpers
    # --------------------------------------------------

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