"""
modules/database_manager/runner_scripts/run_drawing_file_sync.py

Runner script for folder-first drawing file sync.

Purpose:
    - Walk DATABASE_DRAWING and subfolders.
    - Match physical drawing files to Drawing rows in the database.
    - Update Drawing.file_path when a safe match is found.
    - Default to dry-run for safety.
    - Optionally apply database updates.
    - Optionally save a JSON report.
    - Optionally scan all files without filtering by extension.

Recommended commands:

    Dry run:
        python -m modules.database_manager.runner_scripts.run_drawing_file_sync --show-results

    Dry run, scan all files:
        python -m modules.database_manager.runner_scripts.run_drawing_file_sync --all-files --show-results

    Apply safe updates, scan all files:
        python -m modules.database_manager.runner_scripts.run_drawing_file_sync --all-files --apply --show-results

    Apply and save report:
        python -m modules.database_manager.runner_scripts.run_drawing_file_sync --all-files --apply --report-json "E:\\emtac\\logs\\drawing_file_sync_report.json"

Important:
    Dry run is the default.
    Do not use --create-missing until you have reviewed dry-run results.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional

from modules.database_manager.services.drawing_file_sync_service import (
    DrawingFileSyncService,
    print_console_summary,
    write_json_report,
)

try:
    from modules.configuration.log_config import error_id, info_id
except Exception:
    def info_id(message: str, request_id: Optional[str] = None) -> None:
        print(f"[INFO] {message}")

    def error_id(message: str, request_id: Optional[str] = None) -> None:
        print(f"[ERROR] {message}", file=sys.stderr)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Folder-first drawing file sync. "
            "Walks the drawing folder, matches files to database Drawing rows, "
            "and optionally updates Drawing.file_path."
        )
    )

    parser.add_argument(
        "--apply",
        action="store_true",
        help="Apply database updates. Without this flag, the runner performs a dry run.",
    )

    parser.add_argument(
        "--drawing-root",
        default=None,
        help=(
            "Optional drawing folder override. "
            "Defaults to DATABASE_DRAWING from config/.env."
        ),
    )

    parser.add_argument(
        "--no-recursive",
        action="store_true",
        help="Only scan the top level of the drawing folder. By default, all subfolders are walked.",
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional max number of physical files to process.",
    )

    parser.add_argument(
        "--commit-every",
        type=int,
        default=100,
        help="Commit after this many writes when --apply is used.",
    )

    parser.add_argument(
        "--extensions",
        nargs="*",
        default=None,
        help="Allowed file extensions. Example: --extensions .dwg .dxf .slddrw .sldprt .pdf",
    )

    parser.add_argument(
        "--all-files",
        action="store_true",
        help=(
            "Scan all files regardless of extension. "
            "This disables the extension filter."
        ),
    )

    parser.add_argument(
        "--no-compact-match",
        action="store_true",
        help="Disable loose matching that ignores spaces, dashes, underscores, and punctuation.",
    )

    parser.add_argument(
        "--prefer-first-on-ambiguous",
        action="store_true",
        help=(
            "Use the first DB match when multiple Drawing rows match one physical file. "
            "Default is safer: mark ambiguous and do not update."
        ),
    )

    parser.add_argument(
        "--create-missing",
        action="store_true",
        help=(
            "Create Drawing rows for files that do not match any DB record. "
            "Use with --apply to actually create records."
        ),
    )

    parser.add_argument(
        "--default-drw-type",
        default="Other",
        help="Default drw_type used when --create-missing is enabled.",
    )

    parser.add_argument(
        "--report-json",
        default=None,
        help="Optional path to save a JSON sync report.",
    )

    parser.add_argument(
        "--show-results",
        action="store_true",
        help="Print detailed per-file results to the console.",
    )

    return parser


def validate_args(args: argparse.Namespace) -> None:
    if args.limit is not None and args.limit < 1:
        raise ValueError("--limit must be at least 1 when provided.")

    if args.commit_every < 1:
        raise ValueError("--commit-every must be at least 1.")

    if args.default_drw_type is None or not str(args.default_drw_type).strip():
        raise ValueError("--default-drw-type cannot be blank.")

    if args.all_files and args.extensions:
        raise ValueError("Use either --all-files or --extensions, not both.")

    if args.extensions:
        for ext in args.extensions:
            if ext is None or not str(ext).strip():
                raise ValueError("--extensions cannot contain blank values.")


def run_sync(args: argparse.Namespace):
    """
    Build the service and execute folder-first sync.
    """
    validate_args(args)

    dry_run = not args.apply
    recursive = not args.no_recursive

    if args.create_missing and dry_run:
        print(
            "[WARNING] --create-missing was provided without --apply. "
            "This will only report would_create records."
        )

    service = DrawingFileSyncService(
        drawing_root=args.drawing_root,
        allowed_extensions=args.extensions,
        recursive=recursive,
        all_files=args.all_files,
    )

    info_id(
        (
            "Starting folder-first drawing file sync runner. "
            f"dry_run={dry_run}, "
            f"drawing_root={args.drawing_root}, "
            f"recursive={recursive}, "
            f"limit={args.limit}, "
            f"all_files={args.all_files}, "
            f"extensions={args.extensions}, "
            f"create_missing={args.create_missing}"
        ),
        None,
    )

    summary = service.sync_folder_to_database(
        dry_run=dry_run,
        limit=args.limit,
        commit_every=args.commit_every,
        use_compact_match=not args.no_compact_match,
        create_missing=args.create_missing,
        prefer_first_on_ambiguous=args.prefer_first_on_ambiguous,
        default_drw_type=args.default_drw_type,
    )

    return summary


def main() -> int:
    parser = build_arg_parser()
    args = parser.parse_args()

    try:
        summary = run_sync(args)

        print_console_summary(
            summary=summary,
            show_results=args.show_results,
        )

        if args.report_json:
            report_path = Path(args.report_json).expanduser().resolve()

            write_json_report(
                summary=summary,
                report_path=str(report_path),
                include_results=True,
            )

            print()
            print(f"JSON report saved to: {report_path}")

        errors = getattr(summary, "errors", 0)
        ambiguous = getattr(summary, "ambiguous", 0)

        if errors:
            return 2

        if ambiguous:
            return 1

        return 0

    except KeyboardInterrupt:
        print()
        print("Drawing file sync interrupted by user.")
        return 130

    except Exception as exc:
        error_id(f"Drawing file sync runner failed: {exc}", None)
        print(f"[ERROR] Drawing file sync runner failed: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())