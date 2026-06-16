from __future__ import annotations

from typing import Optional

from modules.database_manager.orchestrators.asset_part_bom_association_orchestrator import (
    AssetPartBomAssociationOrchestrator,
)
from modules.database_manager.orchestrators.database_maintenance_orchestrator import (
    DatabaseMaintenanceOrchestrator,
)
from modules.database_manager.orchestrators.position_orchestrator import (
    PositionOrchestrator,
)


class MaintenanceCoordinator:
    """
    Coordinator for database maintenance workflows.

    Responsibilities:
    - normalize incoming task names and options
    - instantiate the appropriate orchestrator
    - dispatch the requested task
    - return a consistent result payload

    Does NOT:
    - open database sessions directly
    - commit or rollback transactions directly
    - perform business/domain logic
    """

    TASK_ALIASES = {
        # ------------------------------------------------------------------
        # Position build
        # ------------------------------------------------------------------
        "build-positions": "build_positions",
        "build_positions": "build_positions",
        "positions": "build_positions",

        # ------------------------------------------------------------------
        # Asset/part BOM association from workbook
        # ------------------------------------------------------------------
        "associate-asset-parts": "associate_asset_parts",
        "associate_asset_parts": "associate_asset_parts",
        "asset-parts": "associate_asset_parts",
        "asset_parts": "associate_asset_parts",
        "bom-assets": "associate_asset_parts",
        "bom_asset_parts": "associate_asset_parts",
        "associate-bom-asset-parts": "associate_asset_parts",

        # ------------------------------------------------------------------
        # Existing maintenance tasks
        # ------------------------------------------------------------------
        "associate-images": "associate_images",
        "associate_images": "associate_images",
        "images": "associate_images",

        "associate-drawings": "associate_drawings",
        "associate_drawings": "associate_drawings",
        "drawings": "associate_drawings",

        "validate-embeddings": "validate_embeddings",
        "validate_embeddings": "validate_embeddings",
        "embeddings": "validate_embeddings",

        "find-duplicates": "find_duplicates",
        "find_duplicates": "find_duplicates",
        "duplicates": "find_duplicates",

        "validate-data-integrity": "validate_data_integrity",
        "validate_data_integrity": "validate_data_integrity",
        "validate-integrity": "validate_data_integrity",
        "integrity": "validate_data_integrity",
        "validate": "validate_data_integrity",

        "run-all": "run_all",
        "run_all": "run_all",
        "all": "run_all",
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
        Dispatch a maintenance task to the appropriate orchestrator.

        Shared kwargs:
        - log_to_console
        - db_log_manager

        DatabaseMaintenanceOrchestrator kwargs:
        - report_dir
        - export_reports
        - quick
        - batch_size
        - threshold
        - include_embedding_validation
        - include_duplicate_check
        - duplicate_threshold
        - include_integrity_validation
        - propagation_progress_interval
        - show_propagation_progress

        Position build task kwargs:
        - create_asset_only_positions
        - include_model_level_locations

        Asset/part BOM association kwargs:
        - workbook_path
        - sheets
        - dry_run
        - log_progress_every
        """
        normalized_task = self._normalize_task_name(task_name)

        if not normalized_task:
            return self._build_result(
                False,
                f"Unknown maintenance task: {task_name}",
                errors=[
                    f"Unsupported task '{task_name}'. "
                    f"Supported tasks: {self.get_supported_tasks()}"
                ],
            )

        try:
            # ------------------------------------------------------------------
            # BUILD POSITIONS
            # ------------------------------------------------------------------
            if normalized_task == "build_positions":
                orchestrator = PositionOrchestrator(
                    db_config=self.db_config,
                    db_log_manager=kwargs.get("db_log_manager"),
                    log_to_console=kwargs.get("log_to_console", False),
                )

                try:
                    result = orchestrator.build_positions_from_existing_data(
                        create_asset_only_positions=kwargs.get(
                            "create_asset_only_positions",
                            True,
                        ),
                        include_model_level_locations=kwargs.get(
                            "include_model_level_locations",
                            True,
                        ),
                    )
                finally:
                    orchestrator.close()

            # ------------------------------------------------------------------
            # ASSOCIATE ASSET PARTS FROM BOM WORKBOOK
            # ------------------------------------------------------------------
            elif normalized_task == "associate_asset_parts":
                orchestrator = AssetPartBomAssociationOrchestrator(
                    db_config=self.db_config,
                    workbook_path=kwargs.get("workbook_path"),
                    sheets=kwargs.get("sheets"),
                    log_progress_every=kwargs.get("log_progress_every", 1000),
                )

                result = orchestrator.run(
                    workbook_path=kwargs.get("workbook_path"),
                    sheets=kwargs.get("sheets"),
                    dry_run=kwargs.get("dry_run", False),
                )

            # ------------------------------------------------------------------
            # STANDARD DATABASE MAINTENANCE TASKS
            # ------------------------------------------------------------------
            else:
                orchestrator = DatabaseMaintenanceOrchestrator(
                    db_config=self.db_config,
                    db_log_manager=kwargs.get("db_log_manager"),
                    report_dir=kwargs.get("report_dir"),
                    export_reports=kwargs.get("export_reports", True),
                    quick=kwargs.get("quick", False),
                    log_to_console=kwargs.get("log_to_console", False),
                )

                try:
                    if normalized_task == "associate_images":
                        result = orchestrator.associate_images(
                            batch_size=kwargs.get("batch_size", 1000),
                            propagation_progress_interval=kwargs.get(
                                "propagation_progress_interval",
                                5000,
                            ),
                            show_propagation_progress=kwargs.get(
                                "show_propagation_progress",
                                True,
                            ),
                        )

                    elif normalized_task == "associate_drawings":
                        result = orchestrator.associate_drawings()

                    elif normalized_task == "validate_embeddings":
                        result = orchestrator.validate_embeddings()

                    elif normalized_task == "find_duplicates":
                        result = orchestrator.find_duplicates(
                            threshold=kwargs.get("threshold", 0.9),
                        )

                    elif normalized_task == "validate_data_integrity":
                        result = orchestrator.validate_data_integrity()

                    elif normalized_task == "run_all":
                        result = orchestrator.run_all(
                            include_embedding_validation=kwargs.get(
                                "include_embedding_validation",
                                False,
                            ),
                            include_duplicate_check=kwargs.get(
                                "include_duplicate_check",
                                False,
                            ),
                            duplicate_threshold=kwargs.get(
                                "duplicate_threshold",
                                kwargs.get("threshold", 0.9),
                            ),
                            include_integrity_validation=kwargs.get(
                                "include_integrity_validation",
                                False,
                            ),
                            batch_size=kwargs.get("batch_size", 1000),
                            propagation_progress_interval=kwargs.get(
                                "propagation_progress_interval",
                                5000,
                            ),
                            show_propagation_progress=kwargs.get(
                                "show_propagation_progress",
                                True,
                            ),
                        )

                    else:
                        return self._build_result(
                            False,
                            f"Task normalization failed for: {task_name}",
                            errors=[f"No dispatch handler found for task '{task_name}'."],
                        )
                finally:
                    orchestrator.close()

            task_success = bool(result.get("success", False))

            return self._build_result(
                task_success,
                f"Maintenance task '{normalized_task}' completed."
                if task_success
                else f"Maintenance task '{normalized_task}' failed.",
                data={"task_result": result},
                errors=result.get("errors", []),
            )

        except Exception as exc:
            return self._build_result(
                False,
                f"Maintenance task '{normalized_task}' failed.",
                errors=[str(exc)],
            )

    def get_supported_tasks(self) -> list[str]:
        """
        Return canonical public task names.
        """
        return [
            "build-positions",
            "associate-asset-parts",
            "associate-images",
            "associate-drawings",
            "validate-embeddings",
            "find-duplicates",
            "validate-data-integrity",
            "run-all",
        ]