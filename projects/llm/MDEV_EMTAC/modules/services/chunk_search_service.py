# MDEV_EMTAC\modules\services\chunk_search_service.py

"""
ChunkAssociationSearch

Comprehensive retrieval of all entities associated with a Document chunk.

This module implements a two-tier traversal model starting from a Document
(chunk) ID and resolving all directly and indirectly related entities.

Tier 1 — Direct Chunk Associations:
    - Parent CompleteDocument
    - Images via ImageCompletedDocumentAssociation
    - DocumentEmbedding records

Tier 2 — Position-Based Associations:
    - Positions via CompletedDocumentPositionAssociation
    - Full hierarchy resolution (Area, EquipmentGroup, Model, etc.)
    - Parts, Drawings, Problems, Tasks, Tools via position associations
    - Images via ImagePositionAssociation and PartsPositionImageAssociation

Extended capabilities (ChunkAssociationSearchExtended):
    - Reverse lookups (entity → chunks)
    - Batch processing
    - Embedding similarity search (pgvector)
    - Lightweight association summaries

"""


# ===============================
# Standard Library
# ===============================
from sqlalchemy import select
from sqlalchemy import inspect as sa_inspect

from typing import Dict, List, Optional, Any, Set, Tuple
import logging
# ===============================
# SQLAlchemy
# ===============================
from sqlalchemy.orm import Session
from sqlalchemy.sql import text

# ===============================
# Logging / Configuration
# ===============================
from modules.configuration.config_env import DatabaseConfig
from modules.configuration.log_config import (
    debug_id,
    info_id,
    warning_id,
    error_id,
    get_request_id,
    with_request_id,
    log_timed_operation,
)

# ===============================
# EMTAC Core ORM Models (READ-ONLY)
# ===============================
from modules.emtacdb.emtacdb_fts import (Document,CompleteDocument,DocumentEmbedding,
                                        Position,Part,Drawing,Problem,Task,Tool,Image,)

# ===============================
# Service Layer (PRIMARY DEPENDENCY)
# ===============================
from modules.services.complete_document_service import CompleteDocumentService
from modules.services.completed_document_position_service import (
    CompletedDocumentPositionService,
)
from modules.services.image_completed_document_association_service import (
    ImageCompletedDocumentAssociationService,
)

from modules.services.parts_position_image_service import (
    PartsPositionImageService,
)
from modules.services.part_service import PartService

from modules.services.drawing_position_association_service import (
    DrawingPositionAssociationService,
)

from modules.services.problem_position_association_service import (
    ProblemPositionAssociationService,
)

from modules.services.task_position_association_service import (
    TaskPositionAssociationService,
)

from modules.services.tool_position_association_service import (
    ToolPositionAssociationService,
)

from modules.services.image_completed_document_association_service import (
    ImageCompletedDocumentAssociationService,
)

from modules.services.image_position_association_service import (
    ImagePositionAssociationService,
)

logger = logging.getLogger(__name__)



class ChunkAssociationSearchService:
    """
    Orchestrates retrieval of all entities related to a Document chunk.

    HARD RULE:
    - This class NEVER trusts ORM objects from other services
    - Everything is serialized defensively
    """

    def __init__(
        self,
        session: Optional[Session] = None,
        request_id: Optional[str] = None,
    ):
        self.db_config = DatabaseConfig()
        self._external_session = session
        self.request_id = request_id or get_request_id()

        # ---------------- Core services ----------------
        self.complete_doc_service = CompleteDocumentService(self.db_config)
        self.doc_position_service = CompletedDocumentPositionService(self.db_config)

        self.parts_position_image_service = PartsPositionImageService(self.db_config)
        self.part_service = PartService(self.db_config)

        self.drawing_position_service = DrawingPositionAssociationService(self.db_config)
        self.problem_position_service = ProblemPositionAssociationService(self.db_config)
        self.task_position_service = TaskPositionAssociationService(self.db_config)
        self.tool_position_service = ToolPositionAssociationService(self.db_config)
        self.image_completed_document_association_service = (ImageCompletedDocumentAssociationService(self.db_config)
        )

    # =========================================================================
    # SESSION HANDLING
    # =========================================================================

    def _get_session(self) -> tuple[Session, bool]:
        if self._external_session:
            return self._external_session, False
        return self.db_config.get_main_session(), True

    # =========================================================================
    # MAIN ENTRY
    # =========================================================================

    @with_request_id
    def search_from_chunk(
            self,
            chunk_id: int,
            include_embeddings: bool = True,
            include_2nd_tier: bool = True,
            request_id: Optional[str] = None,
    ) -> dict[str, Any]:

        rid = request_id or self.request_id
        session, created_here = self._get_session()

        try:
            with log_timed_operation("ChunkAssociationSearch.search_from_chunk", rid):
                info_id(f"Starting chunk search chunk_id={chunk_id}", rid)

                # --------------------------------------------------
                # Fetch chunk
                # --------------------------------------------------
                chunk = session.query(Document).filter_by(id=chunk_id).first()
                if not chunk:
                    warning_id(f"Chunk id={chunk_id} not found", rid)
                    return {"error": "Chunk not found", "chunk_id": chunk_id}

                result = {
                    "chunk_id": chunk_id,
                    "chunk": self._serialize_chunk(chunk),
                    "documents": [],
                    "1st_tier": {},
                    "2nd_tier": {},
                    "summary": {},
                }

                # --------------------------------------------------
                # Tier 1 — Document / Images / Embeddings
                # --------------------------------------------------
                result["1st_tier"] = self._resolve_1st_tier(
                    session=session,
                    chunk=chunk,
                    include_embeddings=include_embeddings,
                    request_id=rid,
                )

                # --------------------------------------------------
                # DOCUMENT-CENTRIC UI PAYLOAD (ADAPTER)
                # --------------------------------------------------
                complete_doc = result["1st_tier"].get("complete_document")

                result["documents"] = [{
                    "complete_document_id": chunk.complete_document_id,
                    "title": (
                        complete_doc.get("title")
                        if isinstance(complete_doc, dict)
                        else None
                    ),
                    "chunks": [{
                        "chunk_id": chunk.id,
                        "text": chunk.content,
                    }],
                }]

                # --------------------------------------------------
                # Tier 2 — Parts via Images (Path A ONLY)
                # --------------------------------------------------
                if include_2nd_tier:
                    tier1_images = result["1st_tier"].get("images", {})

                    has_images = (
                            tier1_images.get("chunk_level")
                            or tier1_images.get("document_level")
                    )

                    if has_images:
                        try:
                            result["2nd_tier"] = self._resolve_2nd_tier_parts_from_images(
                                session=session,
                                tier1_images=tier1_images,
                                request_id=rid,
                            )
                        except Exception as e:
                            error_id(
                                f"[Tier2] Failed but isolated safely: {e}",
                                rid,
                                exc_info=True,
                            )
                            session.rollback()
                            result["2nd_tier"] = {
                                "parts": [],
                                "position_ids": [],
                            }
                    else:
                        debug_id("[Tier2] Skipped — no images returned from Tier 1", rid)
                        result["2nd_tier"] = {
                            "parts": [],
                            "position_ids": [],
                        }

                # --------------------------------------------------
                # Tier 2 — Positions via CompleteDocument (Path B)
                # --------------------------------------------------
                complete_document_id = chunk.complete_document_id

                try:
                    position_data = self._resolve_2nd_tier_positions_from_document(
                        session=session,
                        complete_document_id=complete_document_id,
                        request_id=rid,
                    )
                except Exception as e:
                    error_id(
                        f"[Tier2] Position resolution failed: {e}",
                        rid,
                        exc_info=True,
                    )
                    session.rollback()
                    position_data = {"positions": [], "position_ids": []}

                # Merge into 2nd tier (do NOT overwrite parts)
                result["2nd_tier"].update(position_data)

                position_ids = result["2nd_tier"].get("position_ids", [])

                drawing_data = self._resolve_2nd_tier_drawings_from_positions(
                    session=session,
                    position_ids=position_ids,
                    request_id=rid,
                )

                result["2nd_tier"].update(drawing_data)

                # --------------------------------------------------
                # Summary
                # --------------------------------------------------
                result["summary"] = self._generate_summary(result)
                return result

        except Exception as e:
            error_id(
                f"ChunkAssociationSearch failed: {e}",
                rid,
                exc_info=True,
            )
            session.rollback()
            return {"error": str(e), "chunk_id": chunk_id}

        finally:
            if created_here:
                session.close()

    # =========================================================================
    # 1ST TIER
    # =========================================================================

    def _resolve_1st_tier(
            self,
            *,
            session: Session,
            chunk: Document,
            include_embeddings: bool,
            request_id: str,
    ) -> dict[str, Any]:
        """
        1ST TIER (Path A only)

        Directly reachable from a Document chunk:
        - CompleteDocument
        - Images (chunk-level + document-level)
        - DocumentEmbeddings

        NO positions
        NO parts
        NO drawings
        NO tools
        """

        result = {
            "complete_document": None,
            "images": {
                "chunk_level": [],
                "document_level": [],
            },
            "embeddings": [],
        }

        # --------------------------------------------------
        # Guard: chunk must belong to a complete document
        # --------------------------------------------------
        if not chunk.complete_document_id:
            return result

        cd_id = chunk.complete_document_id

        # --------------------------------------------------
        # 1️⃣ CompleteDocument
        # --------------------------------------------------
        complete_doc = self.complete_doc_service.get(
            complete_document_id=cd_id,
            request_id=request_id,
        )

        result["complete_document"] = self._safe_serialize(complete_doc)

        # --------------------------------------------------
        # 2️⃣ Images — CHUNK LEVEL (highest precision)
        # Document.id → ImageCompletedDocumentAssociation.document_id
        # --------------------------------------------------
        chunk_image_payload = (
            self.image_completed_document_association_service
            .resolve_related_entities(
                document_id=chunk.id,
                session=session,
            )
        )

        # 🚨 ENFORCE SHAPE: ORM Image → UI-safe dict with id
        result["images"]["chunk_level"] = [
            {
                "id": img.id,
                "file_path": img.file_path,
                "description": img.description,
                "source": "chunk",
            }
            for img in chunk_image_payload.get("images", [])
            if hasattr(img, "id")
        ]

        # --------------------------------------------------
        # 3️⃣ Images — DOCUMENT LEVEL (lower precision)
        # CompleteDocument.id → ImageCompletedDocumentAssociation.complete_document_id
        # --------------------------------------------------
        document_image_payload = (
            self.image_completed_document_association_service
            .resolve_related_entities(
                complete_document_id=cd_id,
                session=session,
            )
        )

        result["images"]["document_level"] = [
            {
                "id": img.id,
                "file_path": img.file_path,
                "description": img.description,
                "source": "complete_document",
            }
            for img in document_image_payload.get("images", [])
            if hasattr(img, "id")
        ]

        # --------------------------------------------------
        # 4️⃣ Document embeddings (chunk-scoped)
        # --------------------------------------------------
        if include_embeddings:
            embeddings = (
                session.query(DocumentEmbedding)
                .filter(DocumentEmbedding.document_id == chunk.id)
                .all()
            )

            result["embeddings"] = [
                self._serialize_document_embedding(e)
                for e in embeddings
            ]

        return result

    # =========================================================================
    # 2ND TIER
    # =========================================================================

    def _resolve_2nd_tier_parts_from_images(
            self,
            *,
            session: Session,
            tier1_images: dict,
            request_id: str,
    ) -> dict[str, Any]:

        # --------------------------------------------------
        # Flatten Tier-1 images (ONLY SOURCE OF TRUTH)
        # --------------------------------------------------
        images = (
                tier1_images.get("chunk_level", []) +
                tier1_images.get("document_level", [])
        )

        image_ids = [img["id"] for img in images if isinstance(img, dict) and "id" in img]

        if not image_ids:
            debug_id("[Tier2] No valid image IDs found", request_id)
            return {"parts": [], "positions": []}

        debug_id(
            f"[Tier2] Resolving parts from {len(image_ids)} image IDs",
            request_id,
        )

        # --------------------------------------------------
        # Image → Position → Part
        # --------------------------------------------------
        associations = self.parts_position_image_service.search(
            session=session,
            image_ids=image_ids,
            request_id=request_id,
        )

        position_ids = {assoc.position_id for assoc in associations}
        part_ids = {assoc.part_id for assoc in associations}

        parts = (
            session.query(Part)
            .filter(Part.id.in_(part_ids))
            .all()
        )

        return {
            "parts": [self._serialize_part_safe(p) for p in parts],
            "position_ids": list(position_ids),
        }

    def _resolve_2nd_tier_positions_from_document(
            self,
            session,
            complete_document_id: int,
            request_id: Optional[str] = None,
    ) -> dict[str, Any]:
        """
        Resolve positions directly associated with a CompleteDocument.
        """

        if not complete_document_id:
            return {"positions": [], "position_ids": []}

        from modules.emtacdb.emtacdb_fts import (
            CompletedDocumentPositionAssociation,
            Position,
        )

        rows = (
            session.query(Position)
            .join(
                CompletedDocumentPositionAssociation,
                CompletedDocumentPositionAssociation.position_id == Position.id,
            )
            .filter(
                CompletedDocumentPositionAssociation.complete_document_id
                == complete_document_id
            )
            .all()
        )

        positions = []
        position_ids = []

        for pos in rows:
            positions.append({
                "id": pos.id,
                "area_id": pos.area_id,
                "equipment_group_id": pos.equipment_group_id,
                "model_id": pos.model_id,
                "asset_number_id": pos.asset_number_id,
                "location_id": pos.location_id,
                "site_location_id": pos.site_location_id,
            })
            position_ids.append(pos.id)

        debug_id(
            f"[Tier2] Resolved {len(position_ids)} positions from document",
            request_id,
        )

        return {
            "positions": positions,
            "position_ids": position_ids,
        }

    def _resolve_2nd_tier_drawings_from_positions(
            self,
            *,
            session: Session,
            position_ids: list[int],
            request_id: Optional[str] = None,
    ) -> dict[str, Any]:
        """
        Resolve drawings associated with Positions.

        Tier-2C Path:
            Position → DrawingPositionAssociation → Drawing

        Rules:
        - Requires position_ids (already resolved by Tier-2A / Tier-2B)
        - Never trusts ORM state from other services
        - Returns UI-safe serialized payload
        """

        if not position_ids:
            debug_id("[Tier2] No position_ids provided for drawing resolution", request_id)
            return {
                "drawings": [],
                "drawing_ids": [],
            }

        debug_id(
            f"[Tier2] Resolving drawings for {len(position_ids)} positions",
            request_id,
        )

        try:
            drawings = self.drawing_position_service.get_drawings_for_positions(
                position_ids=position_ids,
                session=session,
                request_id=request_id,
            )

        except Exception as e:
            error_id(
                f"[Tier2] Drawing resolution failed: {e}",
                request_id,
                exc_info=True,
            )
            session.rollback()
            return {
                "drawings": [],
                "drawing_ids": [],
            }

        serialized_drawings = []
        drawing_ids = set()

        for drawing in drawings:
            if drawing is None:
                continue

            # Defensive serialization (NO lazy loads)
            if isinstance(drawing, dict):
                serialized = drawing
                did = drawing.get("id")
            else:
                serialized = {
                    "id": getattr(drawing, "id", None),
                    "drw_number": getattr(drawing, "drw_number", None),
                    "drw_name": getattr(drawing, "drw_name", None),
                    "drw_revision": getattr(drawing, "drw_revision", None),
                    "file_path": getattr(drawing, "file_path", None),
                }
                did = serialized["id"]

            if did:
                drawing_ids.add(did)

            serialized_drawings.append(serialized)

        debug_id(
            f"[Tier2] Resolved {len(drawing_ids)} drawings from positions",
            request_id,
        )

        return {
            "drawings": serialized_drawings,
            "drawing_ids": list(drawing_ids),
        }

    # =========================================================================
    # SAFE SERIALIZATION HELPERS
    # =========================================================================

    def _safe_serialize(self, obj):
        """Pass through dicts, serialize ORM safely, ignore None."""
        if obj is None:
            return None
        if isinstance(obj, dict):
            return obj
        return self._serialize_generic(obj)

    def _safe_list(self, items):
        return [i for i in (items or []) if isinstance(i, dict)]

    def _serialize_generic(self, obj):
        """Last-resort ORM serializer (ID only)."""
        return {"id": getattr(obj, "id", None)}

    def _serialize_chunk(self, chunk) -> dict:
        return {
            "id": chunk.id,
            "name": chunk.name,
            "content": chunk.content,
            "complete_document_id": chunk.complete_document_id,
        }

    def _serialize_image_safe(self, image, source: str, session: Optional[Session] = None) -> dict:
        """
        Serialize an Image WITHOUT triggering lazy loads on detached/expired instances.

        - If `image` is already a dict: pass through.
        - If `image` is an ORM instance from another session: pull its PK via inspect().identity
          (does NOT hit the DB), then reload minimal columns using *our* session.
        """

        # Pass-through if already serialized
        if image is None:
            return {"id": None, "file_path": None, "description": None, "source": source}

        if isinstance(image, dict):
            image.setdefault("source", source)
            return image

        # Try to get PK without touching instrumented attributes
        pk = None
        try:
            insp = sa_inspect(image)
            if insp is not None and insp.identity:
                # identity is a tuple (pk,) for single-column PKs
                pk = insp.identity[0]
        except Exception:
            pk = None

        # If we can't determine PK, return minimal safe shape
        if pk is None:
            return {"id": None, "file_path": None, "description": None, "source": source}

        # If we have a session, reload minimal fields safely (no ORM attribute access needed)
        if session is not None:
            row = session.execute(
                select(Image.id, Image.file_path, Image.description).where(Image.id == pk)
            ).first()

            if row:
                return {
                    "id": row.id,
                    "file_path": row.file_path,
                    "description": row.description,
                    "source": source,
                }

        # Fallback: at least return the PK and source
        return {"id": pk, "file_path": None, "description": None, "source": source}

    def _serialize_document_embedding(self, embedding) -> dict:
        return {
            "id": embedding.id,
            "model_name": embedding.model_name,
            "document_id": embedding.document_id,
        }

    def _serialize_position_safe(self, position) -> dict:
        if isinstance(position, dict):
            return position

        return {
            "id": position.id,
            "area_id": position.area_id,
            "equipment_group_id": position.equipment_group_id,
            "model_id": position.model_id,
            "asset_number_id": position.asset_number_id,
            "location_id": position.location_id,
            "site_location_id": position.site_location_id,
        }

    # =========================================================================
    # SERVICE ADAPTER HELPERS (SESSION SAFE)
    # =========================================================================

    def _get_parts_for_positions(self, positions):
        session, _ = self._get_session()
        return self.part_service.get_parts_for_positions(
            position_ids=[p.id for p in positions],
            session=session,
        )

    def _get_drawings_for_positions(self, positions):
        session, _ = self._get_session()
        return self.drawing_position_service.get_drawings_for_positions(
            position_ids=[p.id for p in positions],
            session=session,
        )

    def _get_problems_for_positions(self, positions):
        session, _ = self._get_session()
        return self.problem_position_service.get_problems_for_positions(
            position_ids=[p.id for p in positions],
            session=session,
        )

    def _get_tasks_for_positions(self, positions):
        session, _ = self._get_session()
        return self.task_position_service.get_tasks_for_positions(
            position_ids=[p.id for p in positions],
            session=session,
        )

    def _get_tools_for_positions(self, positions):
        session, _ = self._get_session()
        return self.tool_position_service.get_tools_for_positions(
            position_ids=[p.id for p in positions],
            session=session,
        )

    def _serialize_part_safe(self, part) -> dict:
        """
        Serialize a Part safely without triggering lazy loads.
        """
        if part is None:
            return {"id": None}

        if isinstance(part, dict):
            return part

        return {
            "id": part.id,
            "part_number": part.part_number,
            "name": part.name,
            "oem_mfg": part.oem_mfg,
            "model": part.model,
        }

    # =========================================================================
    # SUMMARY
    # =========================================================================

    def _generate_summary(self, result: dict) -> dict:
        return {
            "image_count": len(result["1st_tier"].get("images", [])),
            "embedding_count": len(result["1st_tier"].get("embeddings", [])),
            "position_count": len(result["2nd_tier"].get("positions", [])),
            "part_count": len(result["2nd_tier"].get("parts", [])),
            "drawing_count": len(result["2nd_tier"].get("drawings", [])),
            "problem_count": len(result["2nd_tier"].get("problems", [])),
            "task_count": len(result["2nd_tier"].get("tasks", [])),
            "tool_count": len(result["2nd_tier"].get("tools", [])),
        }

class ChunkAssociationSearchExtendedService:
    """
    Extended search capabilities for chunk-entity relationships.

    ✔ Reverse lookups (entity → chunks)
    ✔ Batch processing
    ✔ Similarity search integration
    ✔ Cross-entity traversal
    """

    def __init__(
        self,
        session: Optional[Session] = None,
        request_id: Optional[str] = None,
    ):
        self.db_config = DatabaseConfig()
        self._external_session = session
        self.request_id = request_id or get_request_id()

        # -------------------------------------------------
        # Service dependencies
        # -------------------------------------------------
        self.image_completed_doc_service = ImageCompletedDocumentAssociationService(self.db_config)
        self.image_position_service = ImagePositionAssociationService(self.db_config)

        self.document_position_service = CompletedDocumentPositionService(self.db_config)
        self.parts_position_service = PartsPositionImageService(self.db_config)

        self.drawing_position_service = DrawingPositionAssociationService(self.db_config)
        self.problem_position_service = ProblemPositionAssociationService(self.db_config)
        self.task_position_service = TaskPositionAssociationService(self.db_config)
        self.tool_position_service = ToolPositionAssociationService(self.db_config)

    def _get_session(self) -> tuple[Session, bool]:
        if self._external_session:
            return self._external_session, False
        return self.db_config.get_main_session(), True

    # =========================================================================
    # REVERSE LOOKUPS
    # =========================================================================

    @with_request_id
    def find_chunks_by_image(
        self,
        image_id: int,
        request_id: Optional[str] = None,
    ) -> list[dict[str, Any]]:
        """
        Image → Chunks
        """
        rid = request_id or self.request_id
        session, created = self._get_session()

        try:
            seen: set[int] = set()
            results: list[dict[str, Any]] = []

            resolved = self.image_completed_doc_service.resolve_related_entities(
                image_id=image_id,
                session=session,
            )

            for doc in resolved.get("documents", []):
                if doc.id in seen:
                    continue
                seen.add(doc.id)

                results.append({
                    "chunk_id": doc.id,
                    "chunk_name": doc.name,
                    "content_preview": doc.content[:200] if doc.content else None,
                    "complete_document_id": doc.complete_document_id,
                    "association_method": "image",
                })

            info_id(f"Found {len(results)} chunks for image_id={image_id}", rid)
            return results

        except Exception as e:
            error_id(f"find_chunks_by_image failed: {e}", rid, exc_info=True)
            return []

        finally:
            if created:
                session.close()

    @with_request_id
    def find_chunks_by_part(
        self,
        part_id: int,
        request_id: Optional[str] = None,
    ) -> list[dict[str, Any]]:
        """
        Part → Position → CompleteDocument → Chunks
        """
        rid = request_id or self.request_id
        session, created = self._get_session()

        try:
            position_ids = self.parts_position_service.get_position_ids_for_part(
                part_id=part_id,
                session=session,
            )

            return self._chunks_from_positions(position_ids, session, rid)

        except Exception as e:
            error_id(f"find_chunks_by_part failed: {e}", rid, exc_info=True)
            return []

        finally:
            if created:
                session.close()

    @with_request_id
    def find_chunks_by_position(
        self,
        position_id: int,
        request_id: Optional[str] = None,
    ) -> list[dict[str, Any]]:
        rid = request_id or self.request_id
        session, created = self._get_session()

        try:
            return self._chunks_from_positions([position_id], session, rid)
        except Exception as e:
            error_id(f"find_chunks_by_position failed: {e}", rid, exc_info=True)
            return []
        finally:
            if created:
                session.close()

    @with_request_id
    def find_chunks_by_hierarchy(self, **filters) -> list[dict[str, Any]]:
        """
        Hierarchy → Positions → Documents → Chunks
        """
        rid = self.request_id
        session, created = self._get_session()

        try:
            position_ids = self.document_position_service.find_position_ids_by_hierarchy(
                session=session,
                **filters,
            )
            return self._chunks_from_positions(position_ids, session, rid)

        except Exception as e:
            error_id(f"find_chunks_by_hierarchy failed: {e}", rid, exc_info=True)
            return []

        finally:
            if created:
                session.close()

    # =========================================================================
    # BATCH PROCESSING
    # =========================================================================

    @with_request_id
    def batch_search_from_chunks(
        self,
        chunk_ids: list[int],
        include_embeddings: bool = False,
        request_id: Optional[str] = None,
    ) -> dict[int, dict[str, Any]]:
        rid = request_id or self.request_id
        session, created = self._get_session()

        try:
            with log_timed_operation("batch_search_from_chunks", rid):
                chunks = (
                    session.query(Document)
                    .filter(Document.id.in_(chunk_ids))
                    .all()
                )

                result: dict[int, dict[str, Any]] = {}

                for chunk in chunks:
                    result[chunk.id] = {
                        "chunk_id": chunk.id,
                        "chunk": {
                            "id": chunk.id,
                            "name": chunk.name,
                            "content": chunk.content,
                            "complete_document_id": chunk.complete_document_id,
                        },
                        "associations": self.lightweight_counts(
                            session,
                            chunk.id,
                            chunk.complete_document_id,
                            rid,
                        ),
                    }

                return result

        except Exception as e:
            error_id(f"batch_search_from_chunks failed: {e}", rid, exc_info=True)
            return {}

        finally:
            if created:
                session.close()

    # =========================================================================
    # SIMILARITY SEARCH
    # =========================================================================

    @with_request_id
    def find_similar_chunks_with_associations(
        self,
        query_embedding: list[float],
        model_name: str,
        limit: int = 10,
        similarity_threshold: float = 0.5,
        include_associations: bool = True,
        request_id: Optional[str] = None,
    ) -> list[dict[str, Any]]:
        rid = request_id or self.request_id
        session, created = self._get_session()

        try:
            vector = "[" + ",".join(map(str, query_embedding)) + "]"

            sql = text("""
                SELECT
                    d.id,
                    d.name,
                    d.content,
                    d.complete_document_id,
                    1 - (e.embedding_vector <=> :vec) AS score
                FROM document_embedding e
                JOIN document d ON d.id = e.document_id
                WHERE e.model_name = :model
                  AND (1 - (e.embedding_vector <=> :vec)) >= :threshold
                ORDER BY e.embedding_vector <=> :vec
                LIMIT :limit
            """)

            rows = session.execute(sql, {
                "vec": vector,
                "model": model_name,
                "threshold": similarity_threshold,
                "limit": limit,
            }).fetchall()

            output: list[dict[str, Any]] = []

            for chunk_id, name, content, cd_id, score in rows:
                item = {
                    "chunk_id": chunk_id,
                    "chunk_name": name,
                    "content_preview": content[:300] if content else None,
                    "complete_document_id": cd_id,
                    "similarity": float(score),
                }

                if include_associations:
                    item["associations"] = self.lightweight_counts(
                        session,
                        chunk_id,
                        cd_id,
                        rid,
                    )

                output.append(item)

            return output

        except Exception as e:
            error_id(f"find_similar_chunks_with_associations failed: {e}", rid, exc_info=True)
            return []

        finally:
            if created:
                session.close()

    # =========================================================================
    # INTERNAL HELPERS
    # =========================================================================

    def _chunks_from_positions(
        self,
        position_ids: list[int],
        session: Session,
        request_id: str,
    ) -> list[dict[str, Any]]:
        if not position_ids:
            return []

        complete_doc_ids = self.document_position_service.get_complete_document_ids_for_positions(
            position_ids=position_ids,
            session=session,
        )

        if not complete_doc_ids:
            return []

        chunks = (
            session.query(Document)
            .filter(Document.complete_document_id.in_(complete_doc_ids))
            .all()
        )

        return [{
            "chunk_id": c.id,
            "chunk_name": c.name,
            "content_preview": c.content[:200] if c.content else None,
            "complete_document_id": c.complete_document_id,
        } for c in chunks]

    def lightweight_counts(
            self,
            session: Session,
            complete_document_id: Optional[int],
            request_id: str,
    ) -> dict[str, int]:

        """
        Service-backed association counts (NO ORM traversal).
        """
        counts = {
            "image_count": 0,
            "position_count": 0,
            "part_count": 0,
            "drawing_count": 0,
            "problem_count": 0,
            "task_count": 0,
            "tool_count": 0,
        }

        if not complete_document_id:
            return counts

        position_ids = self.document_position_service.get_position_ids_for_document(
            complete_document_id=complete_document_id,
            session=session,
        )

        counts["position_count"] = len(position_ids)
        counts["part_count"] = self.parts_position_service.count_parts(position_ids, session)
        counts["drawing_count"] = self.drawing_position_service.count_drawings(position_ids, session)
        counts["problem_count"] = self.problem_position_service.count_problems(position_ids, session)
        counts["task_count"] = self.task_position_service.count_tasks(position_ids, session)
        counts["tool_count"] = self.tool_position_service.count_tools(position_ids, session)

        return counts

@with_request_id
def build_chunk_relationship_map(
    chunk_id: int,
    *,
    session: Optional[Session] = None,
    include_embeddings: bool = True,
    include_reverse: bool = True,
    include_similarity: bool = False,
    similarity_embedding: Optional[List[float]] = None,
    similarity_model: Optional[str] = None,
    request_id: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Build a COMPLETE relationship map for a chunk.

    ✔ Forward associations (1st + 2nd tier)
    ✔ Reverse associations (entity → chunk)
    ✔ Lightweight counts
    ✔ Optional similarity neighbors

    This is the SINGLE source of truth for chunk-related UI & RAG context.
    """

    forward = ChunkAssociationSearchService(
        session=session,
        request_id=request_id,
    )

    extended = ChunkAssociationSearchExtendedService(
        session=session,
        request_id=request_id,
    )

    info_id(f"Building full relationship map for chunk_id={chunk_id}", request_id)

    # --------------------------------------------------
    # FORWARD ASSOCIATIONS
    # --------------------------------------------------
    forward_result = forward.search_from_chunk(
        chunk_id=chunk_id,
        include_embeddings=include_embeddings,
        include_2nd_tier=True,
        request_id=request_id,
    )

    if "error" in forward_result:
        return forward_result

    complete_document_id = forward_result["chunk"].get("complete_document_id")

    # --------------------------------------------------
    # REVERSE ASSOCIATIONS
    # --------------------------------------------------
    reverse: Dict[str, Any] = {}

    if include_reverse and complete_document_id:
        reverse["by_position"] = []
        for pos in forward_result["2nd_tier"].get("positions", []):
            reverse["by_position"].extend(
                extended.find_chunks_by_position(
                    position_id=pos["id"],
                    request_id=request_id,
                )
            )

        # Images → chunks
        reverse["by_images"] = []
        for img in forward_result["1st_tier"].get("images", []):
            reverse["by_images"].extend(
                extended.find_chunks_by_image(
                    image_id=img["id"],
                    request_id=request_id,
                )
            )

        # Parts → chunks
        reverse["by_parts"] = []
        for part in forward_result["2nd_tier"].get("parts", []):
            reverse["by_parts"].extend(
                extended.find_chunks_by_part(
                    part_id=part["id"],
                    request_id=request_id,
                )
            )

    # --------------------------------------------------
    # LIGHTWEIGHT SUMMARY
    # --------------------------------------------------
    summary = extended.lightweight_counts(
        session=forward._get_session()[0],
        complete_document_id=complete_document_id,
        request_id=request_id,
    )

    # --------------------------------------------------
    # SIMILARITY (OPTIONAL)
    # --------------------------------------------------
    similarity = []
    if include_similarity and similarity_embedding and similarity_model:
        similarity = extended.find_similar_chunks_with_associations(
            query_embedding=similarity_embedding,
            model_name=similarity_model,
            request_id=request_id,
        )

    # --------------------------------------------------
    # FINAL MAP
    # --------------------------------------------------
    return {
        "chunk_id": chunk_id,
        "forward": forward_result,
        "reverse": reverse,
        "summary": summary,
        "similar_chunks": similarity,
    }
# ============================================================================
# PRIMARY CHUNK → ASSOCIATION ENTRY POINTS
# ============================================================================



def search_chunk_associations(
    chunk_id: int,
    request_id: Optional[str] = None,
    **kwargs,
) -> Dict[str, Any]:
    """
    Full association search starting from a chunk.

    Thin convenience wrapper around ChunkAssociationSearchService.
    """
    service = ChunkAssociationSearchService(request_id=request_id)
    return service.search_from_chunk(chunk_id, **kwargs)


def get_chunk_association_context(
    chunk_id: int,
    request_id: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Alias for search_from_chunk with default options.

    Kept for backward compatibility / semantic clarity.
    """
    service = ChunkAssociationSearchService(request_id=request_id)
    return service.search_from_chunk(chunk_id)


# ============================================================================
# REVERSE LOOKUPS (ENTITY → CHUNKS)
# ============================================================================

def find_chunks_for_image(
    image_id: int,
    request_id: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """
    Reverse lookup: Image → Chunks
    """
    service = ChunkAssociationSearchExtendedService(request_id=request_id)
    return service.find_chunks_by_image(image_id)


def find_chunks_for_part(
    part_id: int,
    request_id: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """
    Reverse lookup: Part → Position → Document → Chunks
    """
    service = ChunkAssociationSearchExtendedService(request_id=request_id)
    return service.find_chunks_by_part(part_id)



