"""
verify_ingestion_pathway.py

Ingests files from a folder through the full production pathway
(FileProcessingCoordinator → CompleteDocumentOrchestrator → Services → DB)
then immediately reports every ID created so the pathway can be verified.

Output per file:
    CompleteDocument id
      └── Document (chunk) ids
              └── DocumentEmbedding ids
      └── Position ids
      └── ImageCompletedDocumentAssociation ids
              └── Image ids

Usage:
    # Ingest test folder and show results
    python verify_ingestion_pathway.py

    # Custom folder
    python verify_ingestion_pathway.py --folder "E:\\emtac\\data\\raw_documention\\test_doc"

    # Dry run — discover files only, no ingestion
    python verify_ingestion_pathway.py --dry-run

    # Inspect only — skip ingestion, show last N ingested
    python verify_ingestion_pathway.py --inspect-only --last 5

    # Inspect a specific document
    python verify_ingestion_pathway.py --inspect-only --doc-id 42
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from sqlalchemy import text

from modules.configuration.config_env import get_db_config
from modules.coordinators.file_processing_coordinator import FileProcessingCoordinator

# Services — for inspect phase
from modules.services.document_service import DocumentService
from modules.services.document_embedding_service import DocumentEmbeddingService
from modules.services.completed_document_position_service import CompletedDocumentPositionService
from modules.services.image_completed_document_association_service import (
    ImageCompletedDocumentAssociationService,
)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

DEFAULT_FOLDER = r"E:\emtac\data\raw_documention\test_doc"
SUPPORTED_EXTS = {".pdf", ".docx", ".doc", ".xlsx", ".xls", ".csv", ".txt"}

SEP = "─" * 70
SEP2 = "┄" * 55


# ---------------------------------------------------------------------------
# DiskFile — mimics Flask FileStorage for the coordinator
# ---------------------------------------------------------------------------

class DiskFile:
    """Wraps a Path so FileProcessingCoordinator treats it like an upload."""

    def __init__(self, path: Path):
        self._path = path
        self.filename = path.name
        self._stream = None

    def read(self, size: int = -1) -> bytes:
        return self._path.read_bytes() if size == -1 else self._path.open("rb").read(size)

    def save(self, dst):
        import shutil
        shutil.copy2(self._path, dst)

    @property
    def stream(self):
        if self._stream is None:
            self._stream = self._path.open("rb")
        return self._stream

    def __repr__(self):
        return f"<DiskFile {self.filename}>"


# ---------------------------------------------------------------------------
# Result models
# ---------------------------------------------------------------------------

@dataclass
class FileIngestResult:
    file_name: str
    state: str  # persisted | skipped | rolled_back | failed | exception
    status_code: Optional[int] = None
    status: Optional[str] = None
    reason: Optional[str] = None
    source_type: Optional[str] = None
    documents_processed: int = 0
    chunks_created: int = 0
    embeddings_created: int = 0
    images_extracted: int = 0
    deduped: bool = False
    returned_ids: List[int] = field(default_factory=list)
    persisted_ids: List[int] = field(default_factory=list)
    result_keys: List[str] = field(default_factory=list)
    raw_result: Optional[Dict[str, Any]] = None


# ---------------------------------------------------------------------------
# Raw SQL — aggregate summary (cross-domain, not in any service)
# ---------------------------------------------------------------------------

def get_db_summary(session) -> object:
    sql = text("""
        SELECT
            (SELECT COUNT(*) FROM complete_document)                        AS complete_docs,
            (SELECT COUNT(*) FROM document)                                 AS chunks,
            (SELECT COUNT(*) FROM document_embedding)                       AS embeddings,
            (SELECT COUNT(*) FROM document_embedding
             WHERE embedding_vector IS NOT NULL)                            AS pgvector_embeddings,
            (SELECT COUNT(*) FROM image_completed_document_association)     AS image_assocs,
            (SELECT COUNT(*) FROM completed_document_position_association)  AS position_assocs,
            (SELECT COUNT(*) FROM position)                                 AS positions
    """)
    return session.execute(sql).fetchone()


def get_documents_overview(
    session,
    doc_ids: Optional[List[int]] = None,
    last: Optional[int] = None,
    doc_id: Optional[int] = None,
) -> list:
    """Aggregate per-document counts. Filtered by doc_ids list, last N, or single doc_id."""
    where = ""
    params = {}

    if doc_ids:
        id_list = ",".join(str(i) for i in doc_ids)
        where = f"WHERE cd.id IN ({id_list})"
    elif doc_id:
        where = "WHERE cd.id = :doc_id"
        params = {"doc_id": doc_id}

    limit = f"ORDER BY cd.id DESC LIMIT {int(last)}" if last else "ORDER BY cd.id DESC"

    sql = text(f"""
        SELECT
            cd.id,
            cd.title,
            cd.file_path,
            cd.rev,
            COUNT(DISTINCT d.id)    AS chunk_count,
            COUNT(DISTINCT de.id)   AS embedding_count,
            COUNT(DISTINCT ia.id)   AS image_count,
            COUNT(DISTINCT pa.id)   AS position_count
        FROM complete_document cd
        LEFT JOIN document d
            ON d.complete_document_id = cd.id
        LEFT JOIN document_embedding de
            ON de.document_id = d.id
        LEFT JOIN image_completed_document_association ia
            ON ia.complete_document_id = cd.id
        LEFT JOIN completed_document_position_association pa
            ON pa.complete_document_id = cd.id
        {where}
        GROUP BY cd.id, cd.title, cd.file_path, cd.rev
        {limit}
    """)

    return session.execute(sql, params).fetchall()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _verify_complete_document_ids(session, ids: List[int]) -> List[int]:
    """Return only IDs that truly exist in complete_document."""
    verified: List[int] = []
    for cid in ids:
        row = session.execute(
            text("SELECT id FROM complete_document WHERE id = :id"),
            {"id": cid},
        ).fetchone()
        if row:
            verified.append(cid)
    return verified


def _extract_id_from_result(result: dict) -> Optional[int]:
    """Try common singular key names for the complete_document id."""
    for key in ("complete_document_id", "id", "document_id", "doc_id"):
        val = result.get(key)
        if val and isinstance(val, int):
            return val

    for item in result.get("results", []):
        if isinstance(item, dict):
            for key in ("complete_document_id", "id"):
                val = item.get(key)
                if val and isinstance(val, int):
                    return val
    return None


def _extract_returned_ids(result: Dict[str, Any]) -> List[int]:
    returned_ids: List[int] = []

    doc_ids_from_result = result.get("document_ids")
    if isinstance(doc_ids_from_result, list):
        returned_ids.extend(i for i in doc_ids_from_result if isinstance(i, int))

    singular_id = _extract_id_from_result(result)
    if singular_id and singular_id not in returned_ids:
        returned_ids.append(singular_id)

    return returned_ids


def _classify_success_result(
    file_name: str,
    result: Dict[str, Any],
    status_code: int,
    db_config,
) -> FileIngestResult:
    status = result.get("status")
    documents_processed = int(result.get("documents_processed") or 0)
    chunks_created = int(result.get("chunks_created") or 0)
    embeddings_created = int(result.get("embeddings_created") or 0)
    images_extracted = int(result.get("images_extracted") or 0)
    deduped = bool(result.get("deduped") or False)
    source_type = result.get("source_type")

    returned_ids = _extract_returned_ids(result)

    verified_ids: List[int] = []
    if returned_ids:
        with db_config.main_session() as session:
            verified_ids = _verify_complete_document_ids(session, returned_ids)

    # ----------------------------
    # Skipped / no-op classifications
    # ----------------------------
    if not returned_ids and not verified_ids:
        reason = None

        if status in {"skipped", "no_extractable_text"}:
            reason = status
        elif documents_processed == 0 and chunks_created == 0 and embeddings_created == 0:
            reason = "no_documents_processed"
        elif source_type in {"doc-windows-skipped", "unsupported"}:
            reason = source_type
        elif status:
            reason = status
        else:
            reason = "no_id_returned"

        return FileIngestResult(
            file_name=file_name,
            state="skipped",
            status_code=status_code,
            status=status,
            reason=reason,
            source_type=source_type,
            documents_processed=documents_processed,
            chunks_created=chunks_created,
            embeddings_created=embeddings_created,
            images_extracted=images_extracted,
            deduped=deduped,
            returned_ids=returned_ids,
            persisted_ids=[],
            result_keys=list(result.keys()),
            raw_result=result,
        )

    # ----------------------------
    # Returned IDs but not persisted = rollback or partial failure
    # ----------------------------
    if returned_ids and not verified_ids:
        return FileIngestResult(
            file_name=file_name,
            state="rolled_back",
            status_code=status_code,
            status=status,
            reason="returned_ids_not_persisted",
            source_type=source_type,
            documents_processed=documents_processed,
            chunks_created=chunks_created,
            embeddings_created=embeddings_created,
            images_extracted=images_extracted,
            deduped=deduped,
            returned_ids=returned_ids,
            persisted_ids=[],
            result_keys=list(result.keys()),
            raw_result=result,
        )

    # ----------------------------
    # Persisted
    # ----------------------------
    return FileIngestResult(
        file_name=file_name,
        state="persisted",
        status_code=status_code,
        status=status,
        reason=None,
        source_type=source_type,
        documents_processed=documents_processed,
        chunks_created=chunks_created,
        embeddings_created=embeddings_created,
        images_extracted=images_extracted,
        deduped=deduped,
        returned_ids=returned_ids,
        persisted_ids=verified_ids,
        result_keys=list(result.keys()),
        raw_result=result,
    )


# ---------------------------------------------------------------------------
# Ingestion
# ---------------------------------------------------------------------------

def discover_files(folder: Path) -> List[Path]:
    files = sorted(
        f for f in folder.iterdir()
        if f.is_file() and f.suffix.lower() in SUPPORTED_EXTS
    )
    return files


def ingest_files(files: List[Path], metadata: dict, db_config) -> List[FileIngestResult]:
    """
    Runs each file through FileProcessingCoordinator.
    Returns per-file result records with persisted ID verification.
    """
    coordinator = FileProcessingCoordinator()
    ingest_results: List[FileIngestResult] = []

    print(f"\n{'=' * 55}")
    print(f"  INGESTION  ({len(files)} file(s))")
    print(f"{'=' * 55}")

    for file_path in files:
        disk_file = DiskFile(file_path)
        print(f"\n  → {file_path.name}")

        try:
            success, result, status_code = coordinator.process_upload(
                files=[disk_file],
                metadata=metadata,
            )

            if not isinstance(result, dict):
                ingest_result = FileIngestResult(
                    file_name=file_path.name,
                    state="failed",
                    status_code=status_code,
                    status=None,
                    reason=f"unexpected result type: {type(result).__name__}",
                    result_keys=[],
                    raw_result=None,
                )
                ingest_results.append(ingest_result)
                print(f"     ✗ FAILED [{status_code}] unexpected result type: {type(result).__name__}")
                continue

            if success:
                ingest_result = _classify_success_result(
                    file_name=file_path.name,
                    result=result,
                    status_code=status_code,
                    db_config=db_config,
                )
            else:
                ingest_result = FileIngestResult(
                    file_name=file_path.name,
                    state="failed",
                    status_code=status_code,
                    status=result.get("status"),
                    reason=result.get("error", str(result)),
                    source_type=result.get("source_type"),
                    documents_processed=int(result.get("documents_processed") or 0),
                    chunks_created=int(result.get("chunks_created") or 0),
                    embeddings_created=int(result.get("embeddings_created") or 0),
                    images_extracted=int(result.get("images_extracted") or 0),
                    deduped=bool(result.get("deduped") or False),
                    returned_ids=_extract_returned_ids(result),
                    persisted_ids=[],
                    result_keys=list(result.keys()),
                    raw_result=result,
                )

            ingest_results.append(ingest_result)

            if ingest_result.state == "persisted":
                print(
                    f"     ✓ PERSISTED complete_document_id(s)={ingest_result.persisted_ids}"
                    f"  status={ingest_result.status}"
                    f"  chunks={ingest_result.chunks_created}"
                    f"  embeddings={ingest_result.embeddings_created}"
                    f"  images={ingest_result.images_extracted}"
                )
            elif ingest_result.state == "skipped":
                print(
                    f"     - SKIPPED"
                    f"  reason={ingest_result.reason}"
                    f"  status={ingest_result.status}"
                    f"  processed={ingest_result.documents_processed}"
                    f"  chunks={ingest_result.chunks_created}"
                    f"  embeddings={ingest_result.embeddings_created}"
                )
            elif ingest_result.state == "rolled_back":
                print(
                    f"     ✗ ROLLED BACK"
                    f"  returned_ids={ingest_result.returned_ids}"
                    f"  status={ingest_result.status}"
                    f"  chunks={ingest_result.chunks_created}"
                    f"  embeddings={ingest_result.embeddings_created}"
                )
            else:
                print(
                    f"     ✗ FAILED [{ingest_result.status_code}]"
                    f"  {ingest_result.reason}"
                )

        except Exception as e:
            ingest_result = FileIngestResult(
                file_name=file_path.name,
                state="exception",
                status_code=None,
                status=None,
                reason=str(e),
            )
            ingest_results.append(ingest_result)
            print(f"     ✗ EXCEPTION: {e}")

    print()
    return ingest_results


def print_ingestion_summary(results: List[FileIngestResult]) -> List[int]:
    persisted_ids: List[int] = []
    persisted_count = 0
    skipped_count = 0
    rolled_back_count = 0
    failed_count = 0
    exception_count = 0

    for r in results:
        if r.state == "persisted":
            persisted_count += 1
            persisted_ids.extend(r.persisted_ids)
        elif r.state == "skipped":
            skipped_count += 1
        elif r.state == "rolled_back":
            rolled_back_count += 1
        elif r.state == "failed":
            failed_count += 1
        elif r.state == "exception":
            exception_count += 1

    print(f"\n{'=' * 55}")
    print(f"  INGESTION COMPLETE")
    print(f"{'=' * 55}")
    print(f"  persisted files : {persisted_count}")
    print(f"  skipped files   : {skipped_count}")
    print(f"  rolled back     : {rolled_back_count}")
    print(f"  failed files    : {failed_count}")
    print(f"  exceptions      : {exception_count}")
    if persisted_ids:
        print(f"  persisted IDs   : {persisted_ids}")
    print(f"{'=' * 55}")

    return persisted_ids


# ---------------------------------------------------------------------------
# Inspect / display
# ---------------------------------------------------------------------------

def print_db_summary(session) -> None:
    row = get_db_summary(session)
    print(f"\n{'=' * 55}")
    print(f"  DATABASE TOTALS")
    print(f"{'=' * 55}")
    print(f"  complete_document                   : {row[0]}")
    print(f"  document (chunks)                   : {row[1]}")
    print(f"  document_embedding (total)          : {row[2]}")
    print(f"  document_embedding (pgvector)       : {row[3]}")
    print(f"  image_completed_document_assoc      : {row[4]}")
    print(f"  completed_document_position_assoc   : {row[5]}")
    print(f"  position                            : {row[6]}")
    print(f"{'=' * 55}\n")


def print_document_tree(
    session,
    overview_row,
    *,
    document_service: DocumentService,
    embedding_service: DocumentEmbeddingService,
    position_service: CompletedDocumentPositionService,
    image_assoc_service: ImageCompletedDocumentAssociationService,
) -> None:
    cdid = overview_row[0]
    title = overview_row[1] or "(no title)"
    file_path = overview_row[2] or ""
    rev = overview_row[3]
    chunks = overview_row[4]
    embeddings = overview_row[5]
    images = overview_row[6]
    positions = overview_row[7]

    print(f"\n{SEP}")
    print(f"  CompleteDocument  id={cdid}  rev={rev}")
    print(f"  title     : {title}")
    print(f"  file_path : {file_path}")
    print(f"  chunks={chunks}  embeddings={embeddings}  images={images}  positions={positions}")
    print(SEP)

    # ── Positions ──────────────────────────────────────────────────────────
    try:
        position_objs = position_service.get_positions_for_document(
            session=session,
            complete_document_id=cdid,
        )
        if position_objs:
            print(f"\n  POSITIONS  ({len(position_objs)})")
            for pos in position_objs:
                print(
                    f"    position_id={pos.id}  area={pos.area_id}  "
                    f"equip_group={pos.equipment_group_id}  model={pos.model_id}  "
                    f"location={pos.location_id}  asset={pos.asset_number_id}"
                )
    except Exception as e:
        print(f"\n  POSITIONS  [error: {e}]")

    # ── Chunks + Embeddings ────────────────────────────────────────────────
    try:
        chunk_objs = document_service.find(
            session=session,
            complete_document_id=cdid,
            limit=10000,
        )
        if chunk_objs:
            print(f"\n  CHUNKS  ({len(chunk_objs)} total)")
            for chunk in chunk_objs:
                meta = chunk.doc_metadata or {}
                page_label = ""
                if isinstance(meta, dict):
                    pn = meta.get("page_number")
                    ci = meta.get("chunk_index")
                    if pn is not None:
                        page_label += f"  page={pn}"
                    if ci is not None:
                        page_label += f"  chunk_index={ci}"

                print(f"    {SEP2}")
                print(f"    Document id={chunk.id}  rev={chunk.rev}{page_label}")
                print(f"    name : {chunk.name}")

                try:
                    emb_objs = embedding_service.get_by_document(
                        session=session,
                        document_id=chunk.id,
                    )
                    if emb_objs:
                        for emb in emb_objs:
                            storage = emb.get_storage_type()
                            dims = emb.actual_dimensions or "?"
                            created = str(emb.created_at)[:19] if emb.created_at else "?"
                            print(
                                f"      └─ Embedding id={emb.id}  "
                                f"model={emb.model_name}  dims={dims}  "
                                f"storage={storage}  created={created}"
                            )
                    else:
                        print(f"      └─ (no embeddings)")
                except Exception as e:
                    print(f"      └─ [embedding error: {e}]")

    except Exception as e:
        print(f"\n  CHUNKS  [error: {e}]")

    # ── Images ─────────────────────────────────────────────────────────────
    try:
        image_assocs = image_assoc_service.get_by_complete_document(
            session=session,
            complete_document_id=cdid,
        )
        if image_assocs:
            print(f"\n  IMAGES  ({len(image_assocs)} association(s))")
            for ia in image_assocs:
                img = getattr(ia, "image", None)
                print(
                    f"    ImageAssoc id={ia.id}  image_id={ia.image_id}  "
                    f"chunk_id={ia.document_id}  page={ia.page_number}  "
                    f"method={ia.association_method}  confidence={ia.confidence_score}"
                )
                if img:
                    print(f"      └─ title : {img.title}")
                    print(f"         path  : {img.file_path}")
        else:
            print(f"\n  IMAGES  (none)")
    except Exception as e:
        print(f"\n  IMAGES  [error: {e}]")


def run_inspect(session, doc_ids=None, last=None, doc_id=None):
    """Fetch overview rows and print full trees."""
    document_service = DocumentService()
    embedding_service = DocumentEmbeddingService()
    position_service = CompletedDocumentPositionService()
    image_assoc_service = ImageCompletedDocumentAssociationService()

    print_db_summary(session)

    rows = get_documents_overview(session, doc_ids=doc_ids, last=last, doc_id=doc_id)

    if not rows:
        print("  No documents found.")
        return

    print(f"\n  Showing {len(rows)} document(s)\n")

    for row in rows:
        print_document_tree(
            session,
            row,
            document_service=document_service,
            embedding_service=embedding_service,
            position_service=position_service,
            image_assoc_service=image_assoc_service,
        )

    print()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Ingest files through the production pathway then verify all IDs."
    )
    parser.add_argument(
        "--folder",
        default=DEFAULT_FOLDER,
        help=f"Folder to ingest (default: {DEFAULT_FOLDER})",
    )
    parser.add_argument("--dry-run", action="store_true", help="Discover files only, no ingestion")
    parser.add_argument("--inspect-only", action="store_true", help="Skip ingestion, inspect existing records")
    parser.add_argument("--last", type=int, default=None, help="With --inspect-only: show last N documents")
    parser.add_argument("--doc-id", type=int, default=None, help="With --inspect-only: show specific document")
    parser.add_argument("--area", type=int, default=None, help="area_id metadata for ingestion")
    parser.add_argument("--location", type=int, default=None, help="location_id metadata for ingestion")
    args = parser.parse_args()

    db_config = get_db_config()

    # ── Inspect-only mode ─────────────────────────────────────────────────
    if args.inspect_only:
        with db_config.main_session() as session:
            run_inspect(session, last=args.last, doc_id=args.doc_id)
        return

    # ── Discover files ────────────────────────────────────────────────────
    folder = Path(args.folder)
    if not folder.exists():
        print(f"  ERROR: Folder not found: {folder}")
        sys.exit(1)

    files = discover_files(folder)
    if not files:
        print(f"  No supported files found in: {folder}")
        sys.exit(0)

    print(f"\n  Folder : {folder}")
    print(f"  Files  : {len(files)}")
    for f in files:
        print(f"    • {f.name}")

    if args.dry_run:
        print("\n  [dry-run] No files ingested.")
        return

    # ── Ingest ────────────────────────────────────────────────────────────
    metadata = {}
    if args.area:
        metadata["area_id"] = args.area
    if args.location:
        metadata["location_id"] = args.location

    ingest_results = ingest_files(files, metadata, db_config)
    persisted_ids = print_ingestion_summary(ingest_results)

    # ── Inspect results ───────────────────────────────────────────────────
    with db_config.main_session() as session:
        if persisted_ids:
            run_inspect(session, doc_ids=persisted_ids)
        else:
            # Fallback: show last N matching number of files processed
            run_inspect(session, last=max(len(files), 5))


if __name__ == "__main__":
    main()