from __future__ import annotations

import argparse
import json
import sys
import traceback
from pathlib import Path
from typing import Any, Dict, Optional


# ---------------------------------------------------------------------
# Project root bootstrap
# ---------------------------------------------------------------------
# File location:
#   modules/database_manager/runner_scripts/run_image_artifact_cleanup.py
#
# parents[0] = runner_scripts
# parents[1] = database_manager
# parents[2] = modules
# parents[3] = MDEV_EMTAC project root
# ---------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[3]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from modules.database_manager.coordinators.image_artifact_cleanup_coordinator import (
    ImageArtifactCleanupCoordinator,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run EMTAC image artifact cleanup. "
            "Default mode is dry-run. Use --apply to delete database rows."
        )
    )

    # ------------------------------------------------------------
    # Mode
    # ------------------------------------------------------------
    mode_group = parser.add_mutually_exclusive_group()

    mode_group.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview cleanup candidates only. This is the default.",
    )

    mode_group.add_argument(
        "--apply",
        action="store_true",
        help="Apply cleanup and delete matching database rows.",
    )

    # ------------------------------------------------------------
    # Scan controls
    # ------------------------------------------------------------
    parser.add_argument(
        "--batch-size",
        type=int,
        default=500,
        help="Number of image rows to scan per batch. Default: 500.",
    )

    parser.add_argument(
        "--max-images",
        type=int,
        default=None,
        help="Maximum number of image rows to scan. Default: no limit.",
    )

    # ------------------------------------------------------------
    # Cleanup thresholds
    # ------------------------------------------------------------
    parser.add_argument(
        "--min-file-bytes",
        type=int,
        default=5_000,
        help="Files smaller than this are cleanup candidates. Default: 5000.",
    )

    parser.add_argument(
        "--min-width",
        type=int,
        default=80,
        help="Images narrower than this are cleanup candidates. Default: 80.",
    )

    parser.add_argument(
        "--min-height",
        type=int,
        default=80,
        help="Images shorter than this are cleanup candidates. Default: 80.",
    )

    parser.add_argument(
        "--min-area",
        type=int,
        default=10_000,
        help="Images with width * height below this are cleanup candidates. Default: 10000.",
    )

    # ------------------------------------------------------------
    # Safety flags
    # ------------------------------------------------------------
    parser.add_argument(
        "--include-missing-files",
        action="store_true",
        help="Include image DB rows where the physical file is missing.",
    )

    parser.add_argument(
        "--allow-delete-protected",
        action="store_true",
        help=(
            "Allow deleting images with protected associations. "
            "Protected associations include position/tool/task/problem/parts-position links."
        ),
    )

    parser.add_argument(
        "--protect-document-images",
        action="store_true",
        help=(
            "Treat complete-document image associations as protected. "
            "By default, complete-document associations do not protect extracted artifacts."
        ),
    )

    # ------------------------------------------------------------
    # Physical file actions
    # ------------------------------------------------------------
    file_group = parser.add_mutually_exclusive_group()

    file_group.add_argument(
        "--quarantine-files",
        action="store_true",
        help="After DB cleanup commits, move physical files to a quarantine folder.",
    )

    file_group.add_argument(
        "--delete-files",
        action="store_true",
        help="After DB cleanup commits, permanently delete physical files.",
    )

    parser.add_argument(
        "--quarantine-dir",
        type=str,
        default=None,
        help="Folder used when --quarantine-files is enabled.",
    )

    # ------------------------------------------------------------
    # Reporting / output
    # ------------------------------------------------------------
    parser.add_argument(
        "--report-path",
        type=str,
        default=None,
        help="Optional path for JSON cleanup report.",
    )

    parser.add_argument(
        "--request-id",
        type=str,
        default=None,
        help="Optional request id for logs/tracing.",
    )

    parser.add_argument(
        "--print-json",
        action="store_true",
        help="Print the full result JSON to the console.",
    )

    parser.add_argument(
        "--debug",
        action="store_true",
        help="Print full traceback on failure.",
    )

    return parser


def print_summary(result: Dict[str, Any]) -> None:
    print("\n" + "=" * 80)
    print("EMTAC Image Artifact Cleanup Summary")
    print("=" * 80)

    print(f"Request ID:              {result.get('request_id')}")
    print(f"Dry run:                 {result.get('dry_run')}")
    print(f"Scanned images:          {result.get('scanned_images')}")
    print(f"Cleanup candidates:      {result.get('candidate_count')}")
    print(f"Deleted database rows:   {result.get('deleted_database_rows')}")
    print(f"File action:             {result.get('file_action')}")
    print(f"File action successes:   {result.get('file_action_success_count')}")
    print(f"File action errors:      {result.get('file_action_error_count')}")
    print(f"Report path:             {result.get('report_path')}")

    print("=" * 80)

    if result.get("dry_run"):
        print(
            "\nDry-run only. No database rows or files were deleted. "
            "Review the report before running with --apply."
        )
    else:
        print("\nCleanup apply run completed. Review the report for details.")

    print()


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    dry_run = not args.apply

    try:
        coordinator = ImageArtifactCleanupCoordinator()

        result = coordinator.run(
            dry_run=dry_run,
            batch_size=args.batch_size,
            max_images=args.max_images,
            min_file_bytes=args.min_file_bytes,
            min_width=args.min_width,
            min_height=args.min_height,
            min_area=args.min_area,
            include_missing_files=args.include_missing_files,
            allow_delete_protected=args.allow_delete_protected,
            protect_document_images=args.protect_document_images,
            quarantine_files=args.quarantine_files,
            delete_files=args.delete_files,
            quarantine_dir=args.quarantine_dir,
            report_path=args.report_path,
            request_id=args.request_id,
        )

        print_summary(result)

        if args.print_json:
            print(json.dumps(result, indent=2, ensure_ascii=False, default=str))

        return 0

    except Exception as exc:
        print("\n" + "=" * 80)
        print("Image artifact cleanup failed")
        print("=" * 80)
        print(str(exc))
        print("=" * 80)

        if args.debug:
            traceback.print_exc()

        return 1


if __name__ == "__main__":
    raise SystemExit(main())