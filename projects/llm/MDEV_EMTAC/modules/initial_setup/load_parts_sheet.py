import os
import sys
import time
from typing import Optional, Tuple

from modules.configuration.config import DB_LOADSHEET
from modules.configuration.log_config import (
    info_id,
    warning_id,
    error_id,
    set_request_id,
)
from modules.initial_setup.initializer_logger import close_initializer_logger
from modules.coordinators.parts_import_coordinator import PartsImportCoordinator


class PartsSheetLoader:
    """
    Initializer wrapper for parts Excel import.

    This script is intentionally thin now:
    - gathers CLI / prompt input
    - routes work to PartsImportCoordinator
    - logs a summary
    - does NOT open DB sessions
    - does NOT commit / rollback directly
    """

    DEFAULT_FILENAME = "load_MP2_ITEMS_BOMS.xlsx"

    def __init__(self) -> None:
        self.request_id = set_request_id()
        self.coordinator = PartsImportCoordinator()

        self.stats = {
            "started_at": None,
            "finished_at": None,
            "duration_seconds": 0.0,
            "success": False,
            "status_code": None,
            "rows_loaded": 0,
            "rows_cleaned": 0,
            "new_parts": 0,
            "duplicates_skipped": 0,
            "associations_created": 0,
        }

        info_id("Initialized PartsSheetLoader", self.request_id)

    def resolve_file_path(self, file_path: Optional[str] = None) -> str:
        """
        Resolve the workbook path.
        """
        if file_path:
            return str(file_path).strip().strip('"').strip("'")

        return os.path.join(DB_LOADSHEET, self.DEFAULT_FILENAME)

    def run_import(
        self,
        *,
        file_path: Optional[str] = None,
        create_associations: bool = True,
        create_backup: bool = False,
    ) -> bool:
        """
        Run the parts import through the coordinator.
        """
        resolved_file_path = self.resolve_file_path(file_path)

        info_id(
            f"Starting parts sheet import | file_path={resolved_file_path} "
            f"| create_associations={create_associations} "
            f"| create_backup={create_backup}",
            self.request_id,
        )

        self.stats["started_at"] = time.time()

        try:
            success, response, status_code = self.coordinator.import_parts_from_excel(
                file_path=resolved_file_path,
                create_associations=create_associations,
                create_backup=create_backup,
                request_id=self.request_id,
            )

            self.stats["success"] = success
            self.stats["status_code"] = status_code

            data = response.get("data", {}) if isinstance(response, dict) else {}

            self.stats["rows_loaded"] = int(data.get("rows_loaded", 0) or 0)
            self.stats["rows_cleaned"] = int(data.get("rows_cleaned", 0) or 0)
            self.stats["new_parts"] = int(data.get("new_parts", 0) or 0)
            self.stats["duplicates_skipped"] = int(data.get("duplicates_skipped", 0) or 0)
            self.stats["associations_created"] = int(data.get("associations_created", 0) or 0)

            if success:
                info_id(
                    f"Parts import succeeded | status_code={status_code} "
                    f"| message={response.get('message', '')}",
                    self.request_id,
                )
            else:
                warning_id(
                    f"Parts import completed with issues | status_code={status_code} "
                    f"| message={response.get('message', '')}",
                    self.request_id,
                )

            self._display_summary(response=response)
            return success

        except Exception as exc:
            error_id(
                f"Unhandled error during parts import: {exc}",
                self.request_id,
                exc_info=True,
            )
            return False

        finally:
            self.stats["finished_at"] = time.time()
            self.stats["duration_seconds"] = (
                self.stats["finished_at"] - self.stats["started_at"]
                if self.stats["started_at"] is not None
                else 0.0
            )

    def _display_summary(self, *, response: Optional[dict] = None) -> None:
        """
        Log a concise final summary.
        """
        info_id("Parts import summary", self.request_id)
        info_id(f"Success: {self.stats['success']}", self.request_id)
        info_id(f"Status code: {self.stats['status_code']}", self.request_id)
        info_id(f"Rows loaded: {self.stats['rows_loaded']:,}", self.request_id)
        info_id(f"Rows cleaned: {self.stats['rows_cleaned']:,}", self.request_id)
        info_id(f"New parts: {self.stats['new_parts']:,}", self.request_id)
        info_id(f"Duplicates skipped: {self.stats['duplicates_skipped']:,}", self.request_id)
        info_id(f"Associations created: {self.stats['associations_created']:,}", self.request_id)
        info_id(
            f"Duration: {self._format_time(self.stats['duration_seconds'])}",
            self.request_id,
        )

        if response and isinstance(response, dict):
            message = response.get("message")
            if message:
                info_id(f"Response message: {message}", self.request_id)

    def _format_time(self, seconds: float) -> str:
        """
        Format seconds into a readable time string.
        """
        total_seconds = int(seconds)

        if total_seconds < 60:
            return f"{total_seconds}s"
        if total_seconds < 3600:
            return f"{total_seconds // 60}m {total_seconds % 60}s"

        hours = total_seconds // 3600
        minutes = (total_seconds % 3600) // 60
        return f"{hours}h {minutes}m"


def _prompt_yes_no(prompt: str, default: bool = True) -> bool:
    """
    Prompt helper for yes/no values.
    """
    suffix = "Y/n" if default else "y/N"

    while True:
        raw = input(f"{prompt} ({suffix}): ").strip().lower()

        if raw == "":
            return default
        if raw in {"y", "yes"}:
            return True
        if raw in {"n", "no"}:
            return False

        print("Please enter y or n.")


def _get_cli_args() -> Tuple[Optional[str], bool, bool]:
    """
    Minimal CLI parsing without adding argparse.
    Supported:
      python -m modules.initial_setup.<script_name>
      python -m modules.initial_setup.<script_name> "E:\\path\\file.xlsx"
      python -m modules.initial_setup.<script_name> "E:\\path\\file.xlsx" --no-associations
      python -m modules.initial_setup.<script_name> --backup
    """
    args = sys.argv[1:]

    file_path = None
    create_associations = True
    create_backup = False

    for arg in args:
        normalized = str(arg).strip()

        if normalized == "--no-associations":
            create_associations = False
        elif normalized == "--associations":
            create_associations = True
        elif normalized == "--backup":
            create_backup = True
        elif normalized.startswith("--"):
            # ignore unknown flags for now, but keep it visible in logs later if needed
            continue
        elif file_path is None:
            file_path = normalized

    return file_path, create_associations, create_backup


def main() -> None:
    """
    Main entrypoint for parts sheet import.
    """
    info_id("Starting parts sheet import", request_id=None)

    loader = None

    try:
        loader = PartsSheetLoader()

        file_path, create_associations, create_backup = _get_cli_args()

        if not file_path:
            use_default = _prompt_yes_no(
                f"Use default parts workbook at {os.path.join(DB_LOADSHEET, loader.DEFAULT_FILENAME)}?",
                default=True,
            )
            if not use_default:
                entered = input("Enter parts workbook path: ").strip().strip('"').strip("'")
                file_path = entered or None

        if len(sys.argv) == 1:
            create_associations = _prompt_yes_no(
                "Create part-image associations?",
                default=True,
            )
            create_backup = _prompt_yes_no(
                "Create backup before import?",
                default=False,
            )

        success = loader.run_import(
            file_path=file_path,
            create_associations=create_associations,
            create_backup=create_backup,
        )

        if success:
            info_id("Parts sheet import completed successfully", loader.request_id)
        else:
            warning_id("Parts sheet import completed with issues", loader.request_id)

    except KeyboardInterrupt:
        error_id("Import interrupted by user", loader.request_id if loader else None)

    except Exception as exc:
        error_id(
            f"Import failed: {exc}",
            loader.request_id if loader else None,
            exc_info=True,
        )

    finally:
        try:
            close_initializer_logger()
        except Exception:
            pass


if __name__ == "__main__":
    main()