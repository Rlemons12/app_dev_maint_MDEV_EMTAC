from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

from modules.configuration.config import LOGS_DIR
from modules.configuration.config_env import get_db_config
from modules.database_manager.coordinators.maintenance_coordinator import (
    MaintenanceCoordinator,
)


def print_banner() -> None:
    """Print a simple banner for the maintenance tool."""
    print("=" * 60)
    print("DATABASE MAINTENANCE TOOL")
    print("=" * 60)
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run database maintenance tasks through the coordinator/orchestrator pipeline.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Available Tasks:
  build-positions           Build Position rows from existing hierarchy data
  associate-asset-parts     Load asset/part BOM workbook and create part_position_image associations
  associate-images          Associate parts with matching images
  associate-drawings        Associate drawings with matching parts
  validate-embeddings       Validate image embeddings
  find-duplicates           Find potential duplicate parts
  validate-data-integrity   Run integrity checks
  run-all                   Run the main maintenance workflow

Examples:
  python -m modules.database_manager.maintenance.run_maintenance --task build-positions
  python -m modules.database_manager.maintenance.run_maintenance --task build-positions --no-model-level-locations
  python -m modules.database_manager.maintenance.run_maintenance --task associate-asset-parts
  python -m modules.database_manager.maintenance.run_maintenance --task associate-asset-parts --dry-run
  python -m modules.database_manager.maintenance.run_maintenance --task associate-asset-parts --workbook-path "E:\\emtac\\Database\\DB_LOADSHEETS\\boms.xlsx"
  python -m modules.database_manager.maintenance.run_maintenance --task associate-images
  python -m modules.database_manager.maintenance.run_maintenance --task associate-images --propagation-progress-interval 1000
  python -m modules.database_manager.maintenance.run_maintenance --task associate-images --no-propagation-progress
  python -m modules.database_manager.maintenance.run_maintenance --task associate-drawings --report-dir ./reports
  python -m modules.database_manager.maintenance.run_maintenance --task find-duplicates --threshold 0.92
  python -m modules.database_manager.maintenance.run_maintenance --task run-all --include-embedding-validation
  python -m modules.database_manager.maintenance.run_maintenance --task run-all --include-duplicate-check --include-integrity-validation
        """.strip(),
    )

    parser.add_argument(
        "--task",
        choices=[
            "build-positions",
            "associate-asset-parts",
            "associate-images",
            "associate-drawings",
            "validate-embeddings",
            "find-duplicates",
            "validate-data-integrity",
            "run-all",
        ],
        default="run-all",
        help="Maintenance task to run (default: run-all)",
    )

    parser.add_argument(
        "--report-dir",
        type=str,
        default=None,
        help="Directory to save reports (default: <LOGS_DIR>/database_maintenance)",
    )

    parser.add_argument(
        "--no-report",
        action="store_true",
        help="Do not generate report files",
    )

    parser.add_argument(
        "--quick",
        action="store_true",
        help="Minimal console output",
    )

    parser.add_argument(
        "--log-to-console",
        action="store_true",
        help="Mirror maintenance logger output to console",
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=1000,
        help="Batch/progress interval for associate-images title-match phase (default: 1000)",
    )

    parser.add_argument(
        "--threshold",
        type=float,
        default=0.9,
        help="Similarity threshold for duplicate detection (default: 0.9)",
    )

    parser.add_argument(
        "--propagation-progress-interval",
        type=int,
        default=5000,
        help="Propagation progress log interval in candidate pairs for associate-images/run-all (default: 5000)",
    )

    parser.add_argument(
        "--no-propagation-progress",
        action="store_true",
        help="Disable propagation progress logging for associate-images/run-all",
    )

    parser.add_argument(
        "--include-embedding-validation",
        action="store_true",
        help="Include embedding validation when running run-all",
    )

    parser.add_argument(
        "--include-duplicate-check",
        action="store_true",
        help="Include duplicate detection when running run-all",
    )

    parser.add_argument(
        "--include-integrity-validation",
        action="store_true",
        help="Include integrity validation when running run-all",
    )

    # ------------------------------------------------------------------
    # build-positions options
    # ------------------------------------------------------------------
    parser.add_argument(
        "--no-asset-only-positions",
        action="store_true",
        help="When running build-positions, do not create asset-only positions with location_id=NULL",
    )

    parser.add_argument(
        "--no-model-level-locations",
        action="store_true",
        help="When running build-positions, do not create generic model-level location positions where Location.asset_number_id is NULL",
    )

    # ------------------------------------------------------------------
    # associate-asset-parts options
    # ------------------------------------------------------------------
    parser.add_argument(
        "--workbook-path",
        type=str,
        default=r"E:\emtac\Database\DB_LOADSHEETS\boms.xlsx",
        help="When running associate-asset-parts, path to the BOM workbook "
             "(default: E:\\emtac\\Database\\DB_LOADSHEETS\\boms.xlsx)",
    )

    parser.add_argument(
        "--sheets",
        nargs="+",
        default=["bom_1", "bom_2"],
        help="When running associate-asset-parts, list of sheet names to process "
             "(default: bom_1 bom_2)",
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="When running associate-asset-parts, process rows but roll back database changes",
    )

    parser.add_argument(
        "--log-progress-every",
        type=int,
        default=1000,
        help="When running associate-asset-parts, log progress every N rows (default: 1000)",
    )

    parser.add_argument(
        "--json",
        action="store_true",
        help="Print final result as JSON",
    )

    return parser


def print_task_summary(task_result: dict) -> None:
    """Print a readable summary for the task result."""
    task_name = task_result.get("task_name", "unknown")
    success = task_result.get("success", False)
    summary = task_result.get("summary", {})
    report_files = task_result.get("report_files", [])
    errors = task_result.get("errors", [])

    print()
    print("=" * 60)
    print(f"TASK: {task_name}")
    print(f"STATUS: {'SUCCESS' if success else 'FAILED'}")
    print("=" * 60)

    if summary:
        print("Summary:")
        if isinstance(summary, dict):
            for key, value in summary.items():
                if isinstance(value, dict):
                    print(f"  {key}:")
                    for sub_key, sub_value in value.items():
                        print(f"    {sub_key}: {sub_value}")
                elif isinstance(value, list):
                    print(f"  {key}: {len(value)} item(s)")
                else:
                    print(f"  {key}: {value}")

    # Some tasks return totals/details instead of summary/report_files
    totals = task_result.get("data", {}).get("totals", {})
    if totals:
        print()
        print("Totals:")
        for key, value in totals.items():
            print(f"  {key}: {value}")

    report_files_from_data = task_result.get("data", {}).get("report_files", [])
    all_report_files = report_files or report_files_from_data
    if all_report_files:
        print()
        print("Report files:")
        for file_path in all_report_files:
            print(f"  {file_path}")

    if errors:
        print()
        print("Errors:")
        for error in errors:
            print(f"  - {error}")


def main() -> int:
    print_banner()

    parser = build_parser()
    args = parser.parse_args()

    export_reports = not args.no_report
    report_dir = args.report_dir or str(
        Path(LOGS_DIR or "logs") / "database_maintenance"
    )
    show_propagation_progress = not args.no_propagation_progress
    create_asset_only_positions = not args.no_asset_only_positions
    include_model_level_locations = not args.no_model_level_locations

    if not args.quick:
        print(f"Selected task: {args.task}")
        print(f"Report directory: {os.path.abspath(report_dir)}")
        print(f"Export reports: {'Yes' if export_reports else 'No'}")
        print(f"Log to console: {'Yes' if args.log_to_console else 'No'}")

        if args.task == "build-positions":
            print(
                f"Create asset-only positions: "
                f"{'Yes' if create_asset_only_positions else 'No'}"
            )
            print(
                f"Include model-level locations: "
                f"{'Yes' if include_model_level_locations else 'No'}"
            )

        elif args.task == "associate-asset-parts":
            print(f"Workbook path: {args.workbook_path}")
            print(f"Sheets: {args.sheets}")
            print(f"Dry run: {'Yes' if args.dry_run else 'No'}")
            print(f"Log progress every: {args.log_progress_every}")

        elif args.task == "run-all":
            print(
                f"Include embedding validation: "
                f"{'Yes' if args.include_embedding_validation else 'No'}"
            )
            print(
                f"Include duplicate check: "
                f"{'Yes' if args.include_duplicate_check else 'No'}"
            )
            print(
                f"Include integrity validation: "
                f"{'Yes' if args.include_integrity_validation else 'No'}"
            )
            print(
                f"Propagation progress interval: {args.propagation_progress_interval}"
            )
            print(
                f"Show propagation progress: "
                f"{'Yes' if show_propagation_progress else 'No'}"
            )

        elif args.task == "find-duplicates":
            print(f"Duplicate threshold: {args.threshold}")

        elif args.task == "associate-images":
            print(f"Batch size: {args.batch_size}")
            print(
                f"Propagation progress interval: {args.propagation_progress_interval}"
            )
            print(
                f"Show propagation progress: "
                f"{'Yes' if show_propagation_progress else 'No'}"
            )

        print()

    coordinator = MaintenanceCoordinator(
        db_config=get_db_config(),
    )

    start_time = time.time()

    try:
        result = coordinator.run_task(
            args.task,
            report_dir=report_dir,
            export_reports=export_reports,
            quick=args.quick,
            log_to_console=args.log_to_console,
            batch_size=args.batch_size,
            threshold=args.threshold,
            propagation_progress_interval=args.propagation_progress_interval,
            show_propagation_progress=show_propagation_progress,
            include_embedding_validation=args.include_embedding_validation,
            include_duplicate_check=args.include_duplicate_check,
            duplicate_threshold=args.threshold,
            include_integrity_validation=args.include_integrity_validation,
            create_asset_only_positions=create_asset_only_positions,
            include_model_level_locations=include_model_level_locations,
            workbook_path=args.workbook_path,
            sheets=tuple(args.sheets),
            dry_run=args.dry_run,
            log_progress_every=args.log_progress_every,
        )

        duration = time.time() - start_time

        if args.json:
            print(json.dumps(result, indent=2, default=str))
        else:
            task_result = result.get("data", {}).get("task_result", {})
            if task_result:
                print_task_summary(task_result)
            else:
                print_task_summary(result)

            print()
            print(f"Duration: {duration:.2f} seconds")

        if result.get("success", False):
            print()
            print("MAINTENANCE COMPLETED SUCCESSFULLY")
            print("=" * 60)
            return 0

        print()
        print("MAINTENANCE COMPLETED WITH ERRORS")
        print("=" * 60)
        return 1

    except KeyboardInterrupt:
        print("\nMaintenance interrupted by user")
        return 1
    except Exception as exc:
        print(f"\nUnexpected error: {exc}")
        return 1


if __name__ == "__main__":
    sys.exit(main())