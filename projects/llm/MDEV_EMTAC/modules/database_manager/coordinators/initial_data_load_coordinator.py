from __future__ import annotations

from typing import Optional

from modules.database_manager.orchestrators.asset_location_import_orchestrator import (
    AssetLocationImportOrchestrator,
)


class InitialDataLoadCoordinator:
    """
    Coordinator for initial data load / import workflows.

    Responsibilities:
    - normalize incoming task names and options
    - instantiate the correct orchestrator
    - dispatch the requested load/import task
    - return a consistent result payload

    Does NOT:
    - open database sessions directly
    - commit or rollback transactions
    - perform row-level business/domain logic
    """

    TASK_ALIASES = {
        "import-asset-locations": "import_asset_locations",
        "import_asset_locations": "import_asset_locations",
        "asset-locations": "import_asset_locations",
        "asset_locations": "import_asset_locations",
        "locations-to-assets": "import_asset_locations",
        "locations_to_assets": "import_asset_locations",
        "load-asset-locations": "import_asset_locations",
        "load_asset_locations": "import_asset_locations",
    }

    def __init__(self, db_config=None):
        self.db_config = db_config

    @staticmethod
    def _build_result(
        success: bool,
        message: str,
        data: Optional[dict] = None,
        errors: Optional[list[str]] = None,
    ) -> dict:
        return {
            "success": success,
            "message": message,
            "data": data or {},
            "errors": errors or [],
        }

    def _normalize_task_name(self, task_name: str) -> str | None:
        if not task_name:
            return None
        return self.TASK_ALIASES.get(task_name.strip().lower())

    def run_task(
        self,
        task_name: str,
        **kwargs,
    ) -> dict:
        """
        Dispatch an initial data load task to the appropriate orchestrator.

        Supported kwargs:
        - log_to_console
        - db_log_manager
        - log_run_dir

        Asset location import kwargs:
        - excel_path
        - sheet_name
        - dry_run
        """
        normalized_task = self._normalize_task_name(task_name)

        if not normalized_task:
            return self._build_result(
                False,
                f"Unknown initial data load task: {task_name}",
                errors=[
                    f"Unsupported task '{task_name}'. "
                    f"Supported tasks: {self.get_supported_tasks()}"
                ],
            )

        if normalized_task == "import_asset_locations":
            orchestrator = AssetLocationImportOrchestrator(
                db_config=self.db_config,
                db_log_manager=kwargs.get("db_log_manager"),
                log_run_dir=kwargs.get("log_run_dir"),
                log_to_console=kwargs.get("log_to_console", False),
            )

            try:
                excel_path = kwargs.get("excel_path")
                if not excel_path:
                    return self._build_result(
                        False,
                        "Initial data load task 'import_asset_locations' requires 'excel_path'.",
                        errors=["Missing required kwarg: excel_path"],
                    )

                result = orchestrator.import_asset_locations_from_excel(
                    excel_path=excel_path,
                    sheet_name=kwargs.get("sheet_name", 0),
                    dry_run=kwargs.get("dry_run", False),
                )

                task_success = bool(result.get("success", False))

                return self._build_result(
                    task_success,
                    f"Initial data load task '{normalized_task}' completed."
                    if task_success
                    else f"Initial data load task '{normalized_task}' failed.",
                    data={"task_result": result},
                    errors=[],  # IMPORTANT: do not bubble row-level task errors up here
                )

            except Exception as exc:
                return self._build_result(
                    False,
                    f"Initial data load task '{normalized_task}' failed.",
                    errors=[str(exc)],
                )
            finally:
                orchestrator.close()

        return self._build_result(
            False,
            f"Task normalization failed for: {task_name}",
            errors=[f"No dispatch handler found for task '{task_name}'."],
        )

    def get_supported_tasks(self) -> list[str]:
        """
        Return canonical public task names.
        """
        return [
            "import-asset-locations",
        ]