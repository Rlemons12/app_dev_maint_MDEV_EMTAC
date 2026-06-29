from __future__ import annotations

import os
import sys
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv

# ============================================================================
# BOOTSTRAP: MAKE DIRECT SCRIPT EXECUTION WORK
# ============================================================================
CURRENT_FILE = Path(__file__).resolve()
PROJECT_ROOT = CURRENT_FILE.parents[1]  # ...\MDEV_EMTAC


def bootstrap_project() -> Path:
    """
    Ensure the project root is importable when this file is run directly.

    Example:
        python E:\emtac\projects\llm\MDEV_EMTAC\scripts\enrichment_probe.py
    """
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))

    env_path = PROJECT_ROOT / ".env"
    if env_path.exists():
        load_dotenv(env_path)
        print(f"[BOOTSTRAP] Loaded environment from {env_path}")
    else:
        print(f"[BOOTSTRAP] No .env found at {env_path}")

    print(f"[BOOTSTRAP] PROJECT_ROOT added to sys.path: {PROJECT_ROOT}")
    return PROJECT_ROOT


bootstrap_project()

# ============================================================================
# IMPORTS THAT DEPEND ON PROJECT ROOT
# ============================================================================
from sqlalchemy.orm import Session

from modules.configuration.config import BASE_DIR
from modules.configuration.log_config import (
    get_request_id,
    debug_id,
    info_id,
    error_id,
)
from modules.emtacdb.emtacdb_fts import Document
from modules.services.db_services import DBServices

# ============================================================================
# HELPERS
# ============================================================================


def safe_rollback(session: Session, request_id: Optional[str] = None) -> None:
    """
    Safely rollback a poisoned or failed SQLAlchemy transaction.
    """
    try:
        session.rollback()
        debug_id("[TRIGGER_PROBE] Session rollback completed", request_id)
    except Exception as rollback_error:
        error_id(
            f"[TRIGGER_PROBE] Session rollback failed: {rollback_error}",
            request_id,
            exc_info=True,
        )


def summarize_result(result: Dict[str, Any]) -> Dict[str, Any]:
    """
    Build a lightweight summary of the chunk-search result.
    """
    first = result.get("1st_tier") or {}
    second = result.get("2nd_tier") or {}

    images = first.get("images") or {}
    parts = second.get("parts") or []

    summary = {
        "has_chunk_images": bool(images.get("chunk_level")),
        "has_document_images": bool(images.get("document_level")),
        "has_parts": bool(parts),
        "part_count": len(parts) if isinstance(parts, list) else 0,
    }
    return summary


def get_part_ids(parts: Any) -> List[int]:
    """
    Extract valid integer part IDs from a second-tier parts payload.
    """
    part_ids: List[int] = []

    if not isinstance(parts, list):
        return part_ids

    for part in parts:
        if not isinstance(part, dict):
            continue

        part_id = part.get("id")
        if isinstance(part_id, int):
            part_ids.append(part_id)
            continue

        try:
            if part_id is not None:
                part_ids.append(int(part_id))
        except (TypeError, ValueError):
            continue

    return part_ids


# ============================================================================
# CORE LOGIC
# ============================================================================


def find_trigger_chunk(
    session: Session,
    services: DBServices,
    request_id: Optional[str] = None,
) -> Optional[Document]:
    """
    Returns ONE Document chunk that can trigger UI enrichment.

    Trigger conditions:
    - Tier 1: Images (chunk-level OR document-level)
    - Tier 2: Parts (direct OR via drawings)

    Safety behavior:
    - Rolls back poisoned transactions
    - Skips bad chunks
    - Only returns real enrichment candidates
    """

    info_id("[TRIGGER_PROBE] Starting trigger chunk search", request_id)

    # ----------------------------------------------------------------------
    # Session-bound chunk search service
    # ----------------------------------------------------------------------
    chunk_search = services.chunk_search(
        session=session,
        request_id=request_id,
    )

    # Some DBServices implementations expose this as a property-like service.
    drawing_part_service = services.drawing_part_associations

    processed = 0
    trigger_count = 0

    query = session.query(Document).yield_per(50)

    for chunk in query:
        processed += 1

        try:
            result = chunk_search.search_from_chunk(
                chunk_id=chunk.id,
                include_embeddings=False,
                include_2nd_tier=True,
                request_id=request_id,
            )
        except Exception as exc:
            error_id(
                f"[TRIGGER_PROBE] search_from_chunk failed for chunk_id={getattr(chunk, 'id', None)}: {exc}",
                request_id,
                exc_info=True,
            )
            safe_rollback(session, request_id)
            continue

        # ------------------------------------------------------------------
        # Defensive result validation
        # ------------------------------------------------------------------
        if not isinstance(result, dict):
            debug_id(
                f"[TRIGGER_PROBE] Skipping chunk_id={chunk.id}: result was not a dict",
                request_id,
            )
            safe_rollback(session, request_id)
            continue

        if result.get("error"):
            debug_id(
                f"[TRIGGER_PROBE] Skipping chunk_id={chunk.id}: result contained error={result.get('error')}",
                request_id,
            )
            safe_rollback(session, request_id)
            continue

        first = result.get("1st_tier") or {}
        second = result.get("2nd_tier") or {}

        # ------------------------------------------------------------------
        # Tier 1: Images
        # ------------------------------------------------------------------
        images = first.get("images") or {}
        has_images = bool(images.get("chunk_level") or images.get("document_level"))

        # ------------------------------------------------------------------
        # Tier 2: Parts (direct)
        # ------------------------------------------------------------------
        parts = second.get("parts") or []
        has_parts = bool(parts)

        # ------------------------------------------------------------------
        # Tier 2b: Parts -> Drawings
        # ------------------------------------------------------------------
        has_part_drawings = False

        if has_parts:
            part_ids = get_part_ids(parts)

            if part_ids:
                try:
                    drawing_map = drawing_part_service.get_drawing_numbers_by_part_ids(
                        part_ids=part_ids,
                        session=session,
                    )
                    has_part_drawings = bool(drawing_map)
                except Exception as exc:
                    error_id(
                        f"[TRIGGER_PROBE] drawing lookup failed for chunk_id={chunk.id}: {exc}",
                        request_id,
                        exc_info=True,
                    )
                    safe_rollback(session, request_id)

        # ------------------------------------------------------------------
        # Final trigger condition
        # ------------------------------------------------------------------
        if has_images or has_parts or has_part_drawings:
            trigger_count += 1

            debug_id(
                (
                    f"[TRIGGER_PROBE] Trigger chunk found: "
                    f"chunk_id={chunk.id} "
                    f"images={has_images} "
                    f"parts={has_parts} "
                    f"drawings={has_part_drawings}"
                ),
                request_id,
            )

            return chunk

        if processed % 100 == 0:
            debug_id(
                f"[TRIGGER_PROBE] Processed {processed} chunks so far; triggers found={trigger_count}",
                request_id,
            )

    info_id(
        f"[TRIGGER_PROBE] Completed search. Processed={processed}, triggers_found={trigger_count}",
        request_id,
    )
    return None


def print_chunk_preview(chunk: Document) -> None:
    """
    Print a short preview of the trigger chunk.
    """
    content = (getattr(chunk, "content", None) or "").strip()
    preview = content[:300]

    print("TRIGGER CHUNK FOUND")
    print(f"Chunk ID: {getattr(chunk, 'id', None)}")
    print(f"Chunk name: {getattr(chunk, 'name', None)}")
    print("-" * 60)
    print(preview)
    print("-" * 60)


def print_ui_payload_summary(ui_payload: Dict[str, Any]) -> None:
    """
    Print a compact summary of the UI payload structure.
    """
    documents = ui_payload.get("documents-container", [])

    print("\nUI PAYLOAD SUMMARY")
    print(f"Documents returned: {len(documents)}")

    for doc in documents:
        complete_document_id = doc.get("complete_document_id")
        chunks = doc.get("chunks", []) or []
        images = doc.get("images", []) or []
        positions = doc.get("positions", []) or []
        parts = doc.get("parts", []) or []
        drawing_nav = "YES" if doc.get("drawing_navigation") else "NO"

        print(
            f" - Document {complete_document_id}: "
            f"chunks={len(chunks)} "
            f"images={len(images)} "
            f"positions={len(positions)} "
            f"parts={len(parts)} "
            f"drawing_nav={drawing_nav}"
        )


def run_probe(
    session: Session,
    services: DBServices,
    request_id: Optional[str] = None,
) -> int:
    """
    Execute the enrichment probe workflow.

    Returns:
        0 on success
        1 if no trigger chunk found
        2 on fatal failure
    """
    info_id("[TRIGGER_PROBE] Probe started", request_id)

    # ----------------------------------------------------------------------
    # 1. Find a trigger chunk
    # ----------------------------------------------------------------------
    chunk = find_trigger_chunk(
        session=session,
        services=services,
        request_id=request_id,
    )

    if not chunk:
        print("NO TRIGGER CHUNK FOUND")
        info_id("[TRIGGER_PROBE] No trigger chunk found", request_id)
        return 1

    print_chunk_preview(chunk)

    # ----------------------------------------------------------------------
    # 2. Build UI payload
    # ----------------------------------------------------------------------
    try:
        ui_service = services.ui_projection(session)
        ui_payload = ui_service.build_from_chunk(chunk.id)
    except Exception as exc:
        error_id(
            f"[TRIGGER_PROBE] ui_projection.build_from_chunk failed for chunk_id={chunk.id}: {exc}",
            request_id,
            exc_info=True,
        )
        safe_rollback(session, request_id)
        print("\nFAILED TO BUILD UI PAYLOAD")
        return 2

    # ----------------------------------------------------------------------
    # 3. Diagnostics
    # ----------------------------------------------------------------------
    if not isinstance(ui_payload, dict):
        print("\nUI PAYLOAD WAS NOT A DICTIONARY")
        error_id(
            f"[TRIGGER_PROBE] Unexpected ui_payload type: {type(ui_payload).__name__}",
            request_id,
        )
        return 2

    print_ui_payload_summary(ui_payload)

    print("\n✔ UI payload built successfully")
    info_id("[TRIGGER_PROBE] Probe completed successfully", request_id)
    return 0


# ============================================================================
# ENTRYPOINT
# ============================================================================


def main() -> int:
    request_id: Optional[str] = None

    try:
        request_id = get_request_id()

        print("=" * 80)
        print("ENRICHMENT PROBE")
        print("=" * 80)
        print(f"PROJECT_ROOT : {PROJECT_ROOT}")
        print(f"BASE_DIR     : {BASE_DIR}")
        print(f"PYTHON       : {sys.executable}")
        print(f"REQUEST_ID   : {request_id}")
        print("=" * 80)

        services = DBServices()

        with services.db_config.get_main_session() as session:
            return run_probe(
                session=session,
                services=services,
                request_id=request_id,
            )

    except KeyboardInterrupt:
        print("\nInterrupted by user.")
        return 130

    except Exception as exc:
        error_id(
            f"[TRIGGER_PROBE] Fatal error: {exc}",
            request_id,
            exc_info=True,
        )
        print("\nFATAL ERROR")
        print(str(exc))
        print("\nTRACEBACK")
        print(traceback.format_exc())
        return 2


if __name__ == "__main__":
    raise SystemExit(main())