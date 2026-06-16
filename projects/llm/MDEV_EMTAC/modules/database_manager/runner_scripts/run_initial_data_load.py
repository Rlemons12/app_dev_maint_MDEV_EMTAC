from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from modules.database_manager.coordinators.initial_data_load_coordinator import (
    InitialDataLoadCoordinator,
)


DEFAULT_EXCEL_PATH = (
    r"E:\emtac\projects\llm\MDEV_EMTAC\Database\DB_LOADSHEETS"
    r"\xau_mech_locations_to_assets_setup_20260415_110055.xlsx"
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run initial data load workflows for the EMTAC database."
    )

    parser.add_argument(
        "task",
        nargs="?",
        default="import-asset-locations",
        help="Initial data load task to run. Default: import-asset-locations",
    )

    parser.add_argument(
        "--excel-path",
        default=DEFAULT_EXCEL_PATH,
        help=f"Full path to the Excel file to import. Default: {DEFAULT_EXCEL_PATH}",
    )

    parser.add_argument(
        "--sheet-name",
        default=0,
        help="Excel sheet name or zero-based sheet index. Default: 0",
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run validation and processing without committing database changes.",
    )

    parser.add_argument(
        "--log-to-console",
        action="store_true",
        help="Enable orchestrator logging to the console.",
    )

    parser.add_argument(
        "--pretty",
        action="store_true",
        help="Pretty-print a safe JSON summary only.",
    )

    parser.add_argument(
        "--show-error-preview",
        action="store_true",
        help="Show a short preview of row-level errors at the end.",
    )

    parser.add_argument(
        "--error-preview-limit",
        type=int,
        default=10,
        help="How many row-level errors to preview. Default: 10",
    )

    return parser


def normalize_sheet_name(value: str | int) -> str | int:
    if isinstance(value, int):
        return value

    text = str(value).strip()
    if text.isdigit():
        return int(text)
    return text


def print_summary(
    result: dict,
    *,
    show_error_preview: bool = False,
    error_preview_limit: int = 10,
) -> None:
    print("\n" + "=" * 80)
    print("INITIAL DATA LOAD RESULT")
    print("=" * 80)
    print(f"Success : {result.get('success')}")
    print(f"Message : {result.get('message')}")

    data = result.get("data", {})
    task_result = data.get("task_result", {})

    if task_result:
        print("\nTask Result Message:")
        print(task_result.get("message"))

        task_data = task_result.get("data", {})
        totals = task_data.get("totals", {})

        if totals:
            print("\nTOTALS")
            print("-" * 80)
            print(f"Rows in sheet                 : {totals.get('rows_in_sheet', 0)}")
            print(f"Rows processed                : {totals.get('rows_processed', 0)}")
            print(f"Rows skipped blank            : {totals.get('rows_skipped_blank', 0)}")
            print(f"Assets not found              : {totals.get('assets_not_found', 0)}")
            print(f"Assets missing model          : {totals.get('assets_missing_model', 0)}")
            print(f"Locations created             : {totals.get('locations_created', 0)}")
            print(f"Locations updated             : {totals.get('locations_updated', 0)}")
            print(f"Locations reused              : {totals.get('locations_reused', 0)}")
            print(f"Position rows updated         : {totals.get('position_rows_updated', 0)}")
            print(f"Associations created          : {totals.get('associations_created', 0)}")
            print(f"Conflicts                     : {totals.get('conflicts', 0)}")
            print(f"Errors counted in summary     : {totals.get('errors_counted_in_summary', 0)}")
            print(f"Detail rows count             : {totals.get('detail_rows_count', 0)}")
            print(f"Error messages count          : {totals.get('error_messages_count', 0)}")
            print("-" * 80)

            status_counts = totals.get("status_counts", {})
            if status_counts:
                print("\nSTATUS COUNTS")
                print("-" * 80)
                for status, count in sorted(status_counts.items()):
                    print(f"{status:<35} {count}")
                print("-" * 80)

            top_missing_assets = totals.get("top_missing_assets", {})
            if top_missing_assets:
                print("\nTOP MISSING ASSETS")
                print("-" * 80)
                for asset_number, count in top_missing_assets.items():
                    print(f"{asset_number:<30} {count}")
                print("-" * 80)

        errors = task_result.get("errors", [])
        if show_error_preview and errors:
            preview = errors[:max(error_preview_limit, 0)]
            print(f"\nERROR PREVIEW (showing {len(preview)} of {len(errors)})")
            print("-" * 80)
            for error in preview:
                print(f"  - {error}")
            print("-" * 80)

    top_errors = result.get("errors", [])
    if top_errors:
        print("\nCoordinator Errors:")
        for error in top_errors:
            print(f"  - {error}")

    print("=" * 80 + "\n")


def build_safe_pretty_result(result: dict) -> dict:
    """
    Return a compact JSON-safe version of the result.
    Strips row-level details and full error lists.
    """
    safe_result = {
        "success": result.get("success"),
        "message": result.get("message"),
        "errors": result.get("errors", []),
        "data": {},
    }

    data = result.get("data", {})
    task_result = data.get("task_result", {})

    safe_task_result = {
        "success": task_result.get("success"),
        "message": task_result.get("message"),
        "errors": [],
        "data": {},
    }

    task_data = task_result.get("data", {})

    safe_task_result["data"] = {
        "summary": task_data.get("summary", {}),
        "totals": task_data.get("totals", {}),
        "dry_run": task_data.get("dry_run"),
        "excel_path": task_data.get("excel_path"),
    }

    full_error_count = len(task_result.get("errors", []))
    if full_error_count:
        safe_task_result["errors"] = [
            f"{full_error_count} row-level error message(s) suppressed. "
            f"Use --show-error-preview to view a sample."
        ]

    safe_result["data"] = {"task_result": safe_task_result}
    return safe_result


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    excel_path = Path(args.excel_path)
    sheet_name = normalize_sheet_name(args.sheet_name)

    coordinator = InitialDataLoadCoordinator()

    result = coordinator.run_task(
        args.task,
        excel_path=str(excel_path),
        sheet_name=sheet_name,
        dry_run=args.dry_run,
        log_to_console=args.log_to_console,
    )

    print_summary(
        result,
        show_error_preview=args.show_error_preview,
        error_preview_limit=args.error_preview_limit,
    )

    if args.pretty:
        print(json.dumps(build_safe_pretty_result(result), indent=2, default=str))

    return 0 if result.get("success") else 1


if __name__ == "__main__":
    sys.exit(main())