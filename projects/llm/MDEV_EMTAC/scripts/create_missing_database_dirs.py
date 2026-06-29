from __future__ import annotations

import os
import sys
from pathlib import Path


def main() -> int:
    project_root = Path(__file__).resolve().parents[1]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

    from modules.configuration.config import (
        DATABASE_DIR,
        directories_to_check,
        DB_LOADSHEET_BOMS,
        DRAWING_IMPORT_DATA_DIR,
    )

    extra_paths = [
        DB_LOADSHEET_BOMS,
        DRAWING_IMPORT_DATA_DIR,
    ]

    all_paths: list[str] = []
    seen: set[str] = set()

    for raw_path in list(directories_to_check) + extra_paths:
        normalized = os.path.normpath(raw_path)
        if normalized not in seen:
            seen.add(normalized)
            all_paths.append(normalized)

    print("=" * 100)
    print("CREATE MISSING DIRECTORIES")
    print("=" * 100)
    print(f"DATABASE_DIR: {os.path.normpath(DATABASE_DIR)}")
    print("-" * 100)

    created_count = 0
    exists_count = 0
    failed_count = 0

    for path in all_paths:
        try:
            if os.path.exists(path):
                exists_count += 1
                print(f"EXISTS  | {path}")
            else:
                os.makedirs(path, exist_ok=True)
                created_count += 1
                print(f"CREATED | {path}")
        except Exception as exc:
            failed_count += 1
            print(f"FAILED  | {path} | {exc}")

    print("-" * 100)
    print(f"Total   : {len(all_paths)}")
    print(f"Created : {created_count}")
    print(f"Exists  : {exists_count}")
    print(f"Failed  : {failed_count}")
    print("=" * 100)

    return 0 if failed_count == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())