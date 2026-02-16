from typing import Optional

from sqlalchemy.orm import Session

from modules.configuration.log_config import debug_id, error_id
from modules.emtacdb.emtacdb_fts import Document
from modules.services.db_services import DBServices
from modules.services.drawing_part_association_service import DrawingPartAssociationService

def find_trigger_chunk(
    session: Session,
    services: DBServices,
    request_id: Optional[str] = None,
) -> Optional[Document]:
    """
    Returns ONE Document (chunk) that can trigger UI enrichment.

    Trigger conditions:
    - Tier 1: Images (chunk-level OR document-level)
    - Tier 2: Parts (direct OR via drawings)

    SAFE:
    - Rolls back poisoned transactions
    - Skips bad chunks
    - Only returns real enrichment candidates
    """

    # --------------------------------------------------
    # Session-bound services (IMPORTANT)
    # --------------------------------------------------
    chunk_search = services.chunk_search(
        session=session,
        request_id=request_id,
    )

    drawing_part_service = services.drawing_part_associations

    # --------------------------------------------------
    # Iterate chunks safely
    # --------------------------------------------------
    for chunk in session.query(Document).yield_per(50):
        try:
            result = chunk_search.search_from_chunk(
                chunk_id=chunk.id,
                include_embeddings=False,
                include_2nd_tier=True,
                request_id=request_id,
            )

        except Exception as e:
            error_id(
                f"[TRIGGER_PROBE] search_from_chunk failed for chunk {chunk.id}: {e}",
                request_id,
                exc_info=True,
            )
            session.rollback()
            continue

        # -----------------------------
        # Defensive guards
        # -----------------------------
        if not isinstance(result, dict):
            session.rollback()
            continue

        if "error" in result:
            session.rollback()
            continue

        first = result.get("1st_tier") or {}
        second = result.get("2nd_tier") or {}

        # -----------------------------
        # Tier 1: Images
        # -----------------------------
        images = first.get("images") or {}

        has_images = bool(
            images.get("chunk_level") or images.get("document_level")
        )

        # -----------------------------
        # Tier 2: Parts (direct)
        # -----------------------------
        parts = second.get("parts") or []
        has_parts = bool(parts)

        # -----------------------------
        # Tier 2b: Parts → Drawings
        # -----------------------------
        has_part_drawings = False

        if parts:
            part_ids = [
                p["id"] for p in parts
                if isinstance(p, dict) and p.get("id")
            ]

            if part_ids:
                try:
                    drawing_map = drawing_part_service.get_drawing_numbers_by_part_ids(
                        part_ids=part_ids,
                        session=session,
                    )
                    has_part_drawings = bool(drawing_map)

                except Exception as e:
                    error_id(
                        f"[TRIGGER_PROBE] drawing lookup failed for chunk {chunk.id}: {e}",
                        request_id,
                        exc_info=True,
                    )
                    session.rollback()

        # -----------------------------
        # FINAL TRIGGER CONDITION
        # -----------------------------
        if has_images or has_parts or has_part_drawings:
            debug_id(
                f"[TRIGGER] chunk_id={chunk.id} "
                f"images={has_images} "
                f"parts={has_parts} "
                f"drawings={has_part_drawings}",
                request_id,
            )
            return chunk

    return None

# ============================================================================
# STANDALONE DIAGNOSTIC (SLIM + CORRECT)
# ============================================================================

if __name__ == "__main__":
    services = DBServices()

    with services.db_config.get_main_session() as session:
        # --------------------------------------------------
        # 1. Find trigger chunk
        # --------------------------------------------------
        chunk = find_trigger_chunk(session, services)

        if not chunk:
            print("NO TRIGGER CHUNK FOUND")
            raise SystemExit(1)

        print("TRIGGER CHUNK FOUND")
        print(f"Chunk ID: {chunk.id}")
        print(f"Chunk name: {chunk.name}")
        print("-" * 60)
        print((chunk.content or "")[:300])
        print("-" * 60)

        # --------------------------------------------------
        # 2. Build UI payload (SINGLE, CORRECT CALL)
        # --------------------------------------------------
        ui_service = services.ui_projection(session)
        ui_payload = ui_service.build_from_chunk(chunk.id)

        # --------------------------------------------------
        # 3. Diagnostics
        # --------------------------------------------------
        documents = ui_payload.get("documents-container", [])

        print("\nUI PAYLOAD SUMMARY")
        print(f" Documents returned: {len(documents)}")

        for doc in documents:
            print(
                f" - Document {doc['complete_document_id']}: "
                f"chunks={len(doc.get('chunks', []))} "
                f"images={len(doc.get('images', []))} "
                f"positions={len(doc.get('positions', [])) if 'positions' in doc else 0} "
                f"parts={len(doc.get('parts', [])) if 'parts' in doc else 0} "
                f"drawing_nav={'YES' if 'drawing_navigation' in doc else 'NO'}"
            )

        print("\n✔ UI payload built successfully")
