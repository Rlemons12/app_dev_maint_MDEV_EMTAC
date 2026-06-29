from __future__ import annotations

import argparse
import json
import shutil
import sys
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional

from sqlalchemy import text


# ---------------------------------------------------------------------
# Project root bootstrap
# ---------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[3]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from modules.configuration.config import DATABASE_DIR
from modules.configuration.config_env import DatabaseConfig
from modules.emtacdb.emtacdb_fts import Image


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Restore image rows/files from an EMTAC image artifact cleanup report."
        )
    )

    mode_group = parser.add_mutually_exclusive_group()

    mode_group.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview restore actions only. This is the default.",
    )

    mode_group.add_argument(
        "--apply",
        action="store_true",
        help="Apply restore actions.",
    )

    parser.add_argument(
        "--report-path",
        required=True,
        help="Path to image_artifact_cleanup_*.json report.",
    )

    parser.add_argument(
        "--quarantine-dir",
        required=True,
        help="Folder where cleanup quarantined the image files.",
    )

    parser.add_argument(
        "--move-files",
        action="store_true",
        help=(
            "Move files out of quarantine instead of copying them. "
            "Default is copy, which is safer."
        ),
    )

    parser.add_argument(
        "--update-existing",
        action="store_true",
        help="Update existing image rows if the same image id already exists.",
    )

    parser.add_argument(
        "--allow-missing-files",
        action="store_true",
        help=(
            "Restore DB rows even if the physical file cannot be found. "
            "Not recommended unless you know the file already exists elsewhere."
        ),
    )

    parser.add_argument(
        "--debug",
        action="store_true",
        help="Print full traceback on failure.",
    )

    return parser


def load_report(report_path: str) -> Dict[str, Any]:
    path = Path(report_path)

    if not path.exists():
        raise FileNotFoundError(f"Report not found: {path}")

    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, dict):
        raise ValueError("Report JSON must be an object.")

    candidates = data.get("candidates")

    if not isinstance(candidates, list):
        raise ValueError("Report JSON does not contain a candidates list.")

    return data


def resolve_original_file_path(candidate: Dict[str, Any]) -> Optional[Path]:
    resolved_file_path = candidate.get("resolved_file_path")

    if resolved_file_path:
        return Path(resolved_file_path)

    db_file_path = candidate.get("db_file_path")

    if db_file_path:
        db_path = Path(db_file_path)

        if db_path.is_absolute():
            return db_path

        return Path(DATABASE_DIR) / db_path

    return None


def find_quarantined_file(
    *,
    candidate: Dict[str, Any],
    quarantine_dir: Path,
) -> Optional[Path]:
    image_id = candidate.get("image_id")
    original_path = resolve_original_file_path(candidate)

    if image_id is None:
        return None

    if original_path is not None:
        expected = quarantine_dir / f"image_{image_id}_{original_path.name}"

        if expected.exists():
            return expected

    matches = sorted(quarantine_dir.glob(f"image_{image_id}_*"))

    if matches:
        return matches[0]

    return None


def normalize_text(value: Any, fallback: str) -> str:
    if value is None:
        return fallback

    text_value = str(value).strip()

    if not text_value:
        return fallback

    return text_value


def build_restored_metadata(
    *,
    candidate: Dict[str, Any],
    report_path: str,
) -> Dict[str, Any]:
    return {
        "restored_from_cleanup": True,
        "restore_source": "image_artifact_cleanup_report",
        "cleanup_report_path": str(report_path),
        "cleanup_reason_codes": candidate.get("reason_codes", []),
        "cleanup_file_size_bytes": candidate.get("file_size_bytes"),
        "cleanup_width": candidate.get("width"),
        "cleanup_height": candidate.get("height"),
        "cleanup_association_counts": candidate.get("association_counts", {}),
        "note": (
            "This image row was restored from the cleanup report. "
            "Original embeddings and exact deleted associations may need to be regenerated/restored separately."
        ),
    }


def restore_candidate_file(
    *,
    candidate: Dict[str, Any],
    quarantine_dir: Path,
    move_files: bool,
    dry_run: bool,
    allow_missing_files: bool,
) -> Dict[str, Any]:
    image_id = candidate.get("image_id")
    original_path = resolve_original_file_path(candidate)
    quarantined_file = find_quarantined_file(
        candidate=candidate,
        quarantine_dir=quarantine_dir,
    )

    if original_path is None:
        return {
            "image_id": image_id,
            "status": "error",
            "message": "Could not resolve original file path from report candidate.",
            "restored_path": None,
            "quarantined_path": str(quarantined_file) if quarantined_file else None,
        }

    if original_path.exists():
        return {
            "image_id": image_id,
            "status": "already_exists",
            "message": "Original file already exists. No file copy/move needed.",
            "restored_path": str(original_path),
            "quarantined_path": str(quarantined_file) if quarantined_file else None,
        }

    if quarantined_file is None or not quarantined_file.exists():
        if allow_missing_files:
            return {
                "image_id": image_id,
                "status": "missing_allowed",
                "message": "Quarantined file not found, but allow_missing_files=True.",
                "restored_path": str(original_path),
                "quarantined_path": None,
            }

        return {
            "image_id": image_id,
            "status": "error",
            "message": "Quarantined file not found.",
            "restored_path": str(original_path),
            "quarantined_path": None,
        }

    if dry_run:
        return {
            "image_id": image_id,
            "status": "would_restore",
            "message": "Dry-run: file would be restored.",
            "restored_path": str(original_path),
            "quarantined_path": str(quarantined_file),
        }

    original_path.parent.mkdir(parents=True, exist_ok=True)

    if move_files:
        shutil.move(str(quarantined_file), str(original_path))
        action = "moved"
    else:
        shutil.copy2(str(quarantined_file), str(original_path))
        action = "copied"

    return {
        "image_id": image_id,
        "status": action,
        "message": f"File {action} back to original location.",
        "restored_path": str(original_path),
        "quarantined_path": str(quarantined_file),
    }


def restore_candidate_db_row(
    *,
    session,
    candidate: Dict[str, Any],
    report_path: str,
    update_existing: bool,
    dry_run: bool,
) -> Dict[str, Any]:
    image_id = int(candidate["image_id"])

    title = normalize_text(
        candidate.get("title"),
        fallback=f"Restored Image {image_id}",
    )

    description = normalize_text(
        candidate.get("description"),
        fallback="Restored image from image artifact cleanup report.",
    )

    db_file_path = normalize_text(
        candidate.get("db_file_path"),
        fallback="",
    )

    if not db_file_path:
        original_path = resolve_original_file_path(candidate)

        if original_path is not None:
            try:
                db_file_path = str(original_path.relative_to(Path(DATABASE_DIR)))
            except Exception:
                db_file_path = str(original_path)

    if not db_file_path:
        return {
            "image_id": image_id,
            "status": "error",
            "message": "Could not restore DB row because db_file_path is missing.",
        }

    existing = (
        session.query(Image)
        .filter(Image.id == image_id)
        .first()
    )

    metadata = build_restored_metadata(
        candidate=candidate,
        report_path=report_path,
    )

    if existing is not None:
        if not update_existing:
            return {
                "image_id": image_id,
                "status": "already_exists",
                "message": "Image row already exists. Use --update-existing to update it.",
            }

        if dry_run:
            return {
                "image_id": image_id,
                "status": "would_update",
                "message": "Dry-run: existing image row would be updated.",
            }

        existing.title = title
        existing.description = description
        existing.file_path = db_file_path
        existing.img_metadata = metadata

        session.add(existing)
        session.flush()

        return {
            "image_id": image_id,
            "status": "updated",
            "message": "Existing image row updated.",
        }

    if dry_run:
        return {
            "image_id": image_id,
            "status": "would_insert",
            "message": "Dry-run: image row would be inserted.",
        }

    restored = Image(
        id=image_id,
        title=title,
        description=description,
        file_path=db_file_path,
        img_metadata=metadata,
    )

    session.add(restored)
    session.flush()

    return {
        "image_id": image_id,
        "status": "inserted",
        "message": "Image row inserted.",
    }


def fix_postgres_image_sequence(session) -> None:
    """
    If PostgreSQL is used and explicit Image.id values were inserted,
    reset the image.id sequence so future inserts do not collide.

    Safe to ignore on databases that do not support pg_get_serial_sequence.
    """
    try:
        session.execute(
            text(
                """
                SELECT setval(
                    pg_get_serial_sequence('image', 'id'),
                    GREATEST((SELECT COALESCE(MAX(id), 1) FROM image), 1),
                    true
                );
                """
            )
        )
        session.flush()
    except Exception:
        # SQLite or non-Postgres fallback. Ignore safely.
        pass


def summarize_statuses(items: List[Dict[str, Any]]) -> Dict[str, int]:
    summary: Dict[str, int] = {}

    for item in items:
        status = str(item.get("status", "unknown"))
        summary[status] = summary.get(status, 0) + 1

    return summary


def print_summary(result: Dict[str, Any]) -> None:
    print("\n" + "=" * 80)
    print("EMTAC Image Artifact Restore Summary")
    print("=" * 80)
    print(f"Dry run:           {result['dry_run']}")
    print(f"Report path:       {result['report_path']}")
    print(f"Quarantine dir:    {result['quarantine_dir']}")
    print(f"Candidates:        {result['candidate_count']}")
    print(f"DB status:         {result['db_status_summary']}")
    print(f"File status:       {result['file_status_summary']}")
    print("=" * 80)
    print()


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    dry_run = not args.apply

    try:
        report = load_report(args.report_path)
        candidates = report.get("candidates", [])

        quarantine_dir = Path(args.quarantine_dir)

        if not quarantine_dir.exists():
            raise FileNotFoundError(f"Quarantine directory not found: {quarantine_dir}")

        file_results: List[Dict[str, Any]] = []
        db_results: List[Dict[str, Any]] = []

        db = DatabaseConfig()

        with db.main_session() as session:
            try:
                for candidate in candidates:
                    file_result = restore_candidate_file(
                        candidate=candidate,
                        quarantine_dir=quarantine_dir,
                        move_files=args.move_files,
                        dry_run=dry_run,
                        allow_missing_files=args.allow_missing_files,
                    )
                    file_results.append(file_result)

                    if file_result["status"] == "error":
                        db_results.append(
                            {
                                "image_id": candidate.get("image_id"),
                                "status": "skipped",
                                "message": (
                                    "Skipped DB restore because file restore failed. "
                                    "Use --allow-missing-files to override."
                                ),
                            }
                        )
                        continue

                    db_result = restore_candidate_db_row(
                        session=session,
                        candidate=candidate,
                        report_path=args.report_path,
                        update_existing=args.update_existing,
                        dry_run=dry_run,
                    )
                    db_results.append(db_result)

                if dry_run:
                    session.rollback()
                else:
                    fix_postgres_image_sequence(session)
                    session.commit()

            except Exception:
                session.rollback()
                raise

        result = {
            "dry_run": dry_run,
            "report_path": args.report_path,
            "quarantine_dir": str(quarantine_dir),
            "candidate_count": len(candidates),
            "db_status_summary": summarize_statuses(db_results),
            "file_status_summary": summarize_statuses(file_results),
            "db_results": db_results,
            "file_results": file_results,
        }

        print_summary(result)

        print(json.dumps(result, indent=2, ensure_ascii=False, default=str))

        return 0

    except Exception as exc:
        print("\n" + "=" * 80)
        print("Image artifact restore failed")
        print("=" * 80)
        print(str(exc))
        print("=" * 80)

        if args.debug:
            traceback.print_exc()

        return 1


if __name__ == "__main__":
    raise SystemExit(main())