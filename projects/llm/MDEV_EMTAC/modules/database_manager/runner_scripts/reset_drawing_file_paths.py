"""
modules/database_manager/runner_scripts/reset_drawing_file_paths.py

Reset Drawing.file_path values back to the drawing-list placeholder.

Purpose:
    You ran drawing file sync with the wrong file type/path results.
    This script resets any Drawing.file_path value that is NOT already:

        active_drawing_list_import

    back to:

        active_drawing_list_import

Safety:
    - Dry run is default.
    - Use --apply to actually update the database.
    - Use --show-rows to preview affected rows.

Example dry run:
    python -m modules.database_manager.runner_scripts.reset_drawing_file_paths --show-rows

Example apply:
    python -m modules.database_manager.runner_scripts.reset_drawing_file_paths --apply --show-rows
"""

from __future__ import annotations

import argparse
import sys
from typing import Optional

from sqlalchemy.exc import SQLAlchemyError

from modules.configuration.config_env import DatabaseConfig
from modules.emtacdb.emtacdb_fts import Drawing

try:
    from modules.configuration.log_config import info_id, error_id
except Exception:
    def info_id(message: str, request_id: Optional[str] = None) -> None:
        print(f"[INFO] {message}")

    def error_id(message: str, request_id: Optional[str] = None) -> None:
        print(f"[ERROR] {message}", file=sys.stderr)


PLACEHOLDER_FILE_PATH = "active_drawing_list_import"


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Reset Drawing.file_path values back to active_drawing_list_import "
            "for rows that currently have a different value."
        )
    )

    parser.add_argument(
        "--apply",
        action="store_true",
        help="Actually update the database. Without this flag, dry-run only.",
    )

    parser.add_argument(
        "--show-rows",
        action="store_true",
        help="Print affected drawing rows.",
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional max number of rows to process. Useful for testing.",
    )

    parser.add_argument(
        "--placeholder",
        default=PLACEHOLDER_FILE_PATH,
        help="Placeholder value to reset file_path to.",
    )

    return parser


def reset_drawing_file_paths(
    *,
    apply_changes: bool,
    show_rows: bool,
    limit: Optional[int],
    placeholder: str,
) -> int:
    if not placeholder or not str(placeholder).strip():
        raise ValueError("placeholder cannot be blank")

    placeholder = str(placeholder).strip()

    if limit is not None and limit < 1:
        raise ValueError("--limit must be at least 1 when provided")

    db_config = DatabaseConfig()
    session = None

    try:
        session = db_config.get_main_session()

        query = (
            session.query(Drawing)
            .filter(Drawing.file_path.isnot(None))
            .filter(Drawing.file_path != placeholder)
            .order_by(Drawing.id)
        )

        if limit is not None:
            query = query.limit(limit)

        drawings = query.all()

        print()
        print("Reset Drawing.file_path Summary")
        print("-------------------------------")
        print(f"Apply changes:       {apply_changes}")
        print(f"Placeholder value:   {placeholder}")
        print(f"Rows affected:       {len(drawings)}")

        if show_rows:
            print()
            print("Rows to reset")
            print("-------------")

            for drawing in drawings:
                print(
                    f"id={drawing.id} | "
                    f"number={getattr(drawing, 'drw_number', None)!r} | "
                    f"name={getattr(drawing, 'drw_name', None)!r} | "
                    f"old_file_path={getattr(drawing, 'file_path', None)!r}"
                )

        if not apply_changes:
            session.rollback()
            print()
            print("Dry run only. No database changes were made.")
            return 0

        for drawing in drawings:
            drawing.file_path = placeholder

        session.commit()

        info_id(
            f"Reset {len(drawings)} Drawing.file_path value(s) to {placeholder}",
            None,
        )

        print()
        print(f"Updated {len(drawings)} row(s).")
        return 0

    except SQLAlchemyError as exc:
        if session is not None:
            session.rollback()

        error_id(f"Database error while resetting drawing file paths: {exc}", None)
        print(f"[ERROR] Database error: {exc}", file=sys.stderr)
        return 2

    except Exception as exc:
        if session is not None:
            session.rollback()

        error_id(f"Error while resetting drawing file paths: {exc}", None)
        print(f"[ERROR] {exc}", file=sys.stderr)
        return 2

    finally:
        if session is not None:
            session.close()


def main() -> int:
    parser = build_arg_parser()
    args = parser.parse_args()

    return reset_drawing_file_paths(
        apply_changes=args.apply,
        show_rows=args.show_rows,
        limit=args.limit,
        placeholder=args.placeholder,
    )


if __name__ == "__main__":
    raise SystemExit(main())