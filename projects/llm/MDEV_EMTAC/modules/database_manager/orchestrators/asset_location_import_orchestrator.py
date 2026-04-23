from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any, Optional

import pandas as pd
from sqlalchemy.exc import SQLAlchemyError

from modules.configuration.config_env import get_db_config
from modules.configuration.log_config import DatabaseMaintLogManager
from modules.database_manager.services.location_service import LocationService
from modules.emtacdb.emtacdb_fts import AssetNumber, Location, Position


class AssetLocationImportOrchestrator:
    """
    Owns session lifecycle and transaction boundaries for importing/updating
    locations from a spreadsheet that maps ASSET_NUMBER -> LOCATION_DESCRIPTION.

    Rules:
    - Services do NOT commit/rollback
    - This orchestrator DOES commit/rollback
    - This orchestrator handles file parsing, normalization, row processing,
      summary creation, and optional report output

    Business rule for this import:
    - ASSET_NUMBER is used to find the AssetNumber row
    - asset.model_id is then used to resolve/create the Location
    - LOCATION_DESCRIPTION from the spreadsheet maps to Location.name
    - Location.description must remain empty / None
    - Locations are MODEL-based, not ASSET-based
    """

    REQUIRED_COLUMNS = ["ASSET_NUMBER", "LOCATION_DESCRIPTION"]

    def __init__(
        self,
        db_config=None,
        db_log_manager: DatabaseMaintLogManager | None = None,
        log_run_dir=None,
        log_to_console: bool = False,
    ):
        self.db_config = db_config or get_db_config()
        self.db_log_manager = db_log_manager or DatabaseMaintLogManager(
            run_dir=log_run_dir,
            run_name="asset_location_import",
            to_console=log_to_console,
        )
        self.logger = self.db_log_manager.logger

    # -------------------------------------------------------------------------
    # Helpers
    # -------------------------------------------------------------------------
    @staticmethod
    def _clean_str(value: Any) -> Optional[str]:
        if value is None:
            return None

        value = str(value).strip()

        if not value:
            return None

        lowered = value.lower()
        if lowered in {"nan", "none", "null"}:
            return None

        # Normalize Excel numbers like 12345.0 -> 12345
        if value.endswith(".0"):
            numeric_part = value[:-2]
            if numeric_part.isdigit():
                return numeric_part

        return value

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

    @staticmethod
    def _build_totals(
        *,
        summary: dict[str, int],
        status_counts: dict[str, int],
        missing_asset_counts: dict[str, int],
        details_count: int,
        errors_count: int,
    ) -> dict:
        """
        Dedicated totals block so the runner does not need to derive totals.
        """
        return {
            "rows_in_sheet": summary.get("rows_in_sheet", 0),
            "rows_processed": summary.get("rows_processed", 0),
            "rows_skipped_blank": summary.get("rows_skipped_blank", 0),
            "assets_not_found": summary.get("assets_not_found", 0),
            "assets_missing_model": summary.get("assets_missing_model", 0),
            "locations_created": summary.get("locations_created", 0),
            "locations_updated": summary.get("locations_updated", 0),
            "locations_reused": summary.get("locations_reused", 0),
            "position_rows_updated": summary.get("position_rows_updated", 0),
            "associations_created": summary.get("position_rows_updated", 0),
            "conflicts": summary.get("conflicts", 0),
            "errors_counted_in_summary": summary.get("errors", 0),
            "detail_rows_count": details_count,
            "error_messages_count": errors_count,
            "status_counts": status_counts,
            "top_missing_assets": missing_asset_counts,
        }

    def close(self) -> None:
        if self.db_log_manager:
            self.db_log_manager.close()

    def _validate_columns(self, df: pd.DataFrame) -> list[str]:
        normalized_columns = {str(col).strip().upper(): col for col in df.columns}
        missing = [col for col in self.REQUIRED_COLUMNS if col not in normalized_columns]
        return missing

    @staticmethod
    def _normalize_dataframe_columns(df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df.columns = [str(col).strip().upper() for col in df.columns]
        return df

    # -------------------------------------------------------------------------
    # Core processing helpers
    # -------------------------------------------------------------------------
    def _resolve_asset(self, session, asset_number_value: str) -> AssetNumber | None:
        return (
            session.query(AssetNumber)
            .filter(AssetNumber.number == asset_number_value)
            .first()
        )

    def _resolve_or_create_location(
        self,
        session,
        *,
        asset: AssetNumber,
        location_name: str,
    ) -> tuple[Location | None, str, str | None]:
        """
        Resolve or create a Location using MODEL-based logic.

        Rules:
        - AssetNumber is ONLY used to find model_id
        - Location is resolved by (model_id + location_name)
        - Multiple assets can share the same location under the same model
        - Location.description remains None
        """

        existing_model_location = (
            session.query(Location)
            .filter(
                Location.model_id == asset.model_id,
                Location.name == location_name,
            )
            .first()
        )

        if existing_model_location:
            return existing_model_location, "existing_model_location_reused", None

        created = LocationService.create(
            session,
            name=location_name,
            description=None,
            model_id=asset.model_id,
            asset_number_id=None,
        )

        return created, "created_new_location", None

    def _update_positions_for_asset(
        self,
        session,
        *,
        asset_number_id: int,
        location_id: int,
    ) -> int:
        positions = (
            session.query(Position)
            .filter(Position.asset_number_id == asset_number_id)
            .all()
        )

        updated_count = 0
        for position in positions:
            if position.location_id != location_id:
                position.location_id = location_id
                updated_count += 1

        session.flush()
        return updated_count

    # -------------------------------------------------------------------------
    # Main workflow
    # -------------------------------------------------------------------------
    def import_asset_locations_from_excel(
        self,
        *,
        excel_path: str | Path,
        sheet_name: str | int = 0,
        dry_run: bool = False,
    ) -> dict:
        session = self.db_config.get_main_session()

        try:
            excel_path = Path(excel_path)

            if not excel_path.exists():
                return self._build_result(
                    False,
                    f"Excel file not found: {excel_path}",
                    errors=[f"Path does not exist: {excel_path}"],
                )

            self.logger.info("Loading asset/location spreadsheet: %s", excel_path)

            df = pd.read_excel(excel_path, sheet_name=sheet_name)
            df = self._normalize_dataframe_columns(df)

            missing_columns = self._validate_columns(df)
            if missing_columns:
                return self._build_result(
                    False,
                    "Spreadsheet is missing required columns.",
                    errors=[f"Missing required columns: {missing_columns}"],
                )

            summary = {
                "rows_in_sheet": len(df),
                "rows_processed": 0,
                "rows_skipped_blank": 0,
                "assets_not_found": 0,
                "assets_missing_model": 0,
                "locations_created": 0,
                "locations_updated": 0,
                "locations_reused": 0,
                "position_rows_updated": 0,
                "conflicts": 0,
                "errors": 0,
            }

            detail_rows: list[dict[str, Any]] = []
            errors: list[str] = []
            status_counter: Counter[str] = Counter()
            missing_asset_counter: Counter[str] = Counter()

            for index, row in df.iterrows():
                excel_row_num = index + 2  # header row assumed row 1

                asset_number_value = self._clean_str(row.get("ASSET_NUMBER"))
                location_description = self._clean_str(row.get("LOCATION_DESCRIPTION"))

                if not asset_number_value and not location_description:
                    summary["rows_skipped_blank"] += 1
                    status_counter["skipped_blank"] += 1
                    detail_rows.append(
                        {
                            "excel_row": excel_row_num,
                            "asset_number": None,
                            "location_description": None,
                            "status": "skipped_blank",
                            "message": "Blank row skipped.",
                        }
                    )
                    continue

                summary["rows_processed"] += 1

                if not asset_number_value:
                    summary["errors"] += 1
                    status_counter["error"] += 1
                    msg = f"Row {excel_row_num}: missing ASSET_NUMBER."
                    errors.append(msg)
                    detail_rows.append(
                        {
                            "excel_row": excel_row_num,
                            "asset_number": None,
                            "location_description": location_description,
                            "status": "error",
                            "message": msg,
                        }
                    )
                    continue

                if not location_description:
                    summary["errors"] += 1
                    status_counter["error"] += 1
                    msg = (
                        f"Row {excel_row_num}: missing LOCATION_DESCRIPTION "
                        f"for asset '{asset_number_value}'."
                    )
                    errors.append(msg)
                    detail_rows.append(
                        {
                            "excel_row": excel_row_num,
                            "asset_number": asset_number_value,
                            "location_description": None,
                            "status": "error",
                            "message": msg,
                        }
                    )
                    continue

                asset = self._resolve_asset(session, asset_number_value)

                if not asset:
                    summary["assets_not_found"] += 1
                    status_counter["asset_not_found"] += 1
                    missing_asset_counter[asset_number_value] += 1
                    msg = f"Row {excel_row_num}: asset '{asset_number_value}' not found."
                    errors.append(msg)
                    detail_rows.append(
                        {
                            "excel_row": excel_row_num,
                            "asset_number": asset_number_value,
                            "location_description": location_description,
                            "status": "asset_not_found",
                            "message": msg,
                        }
                    )
                    continue

                if not asset.model_id:
                    summary["assets_missing_model"] += 1
                    status_counter["asset_missing_model"] += 1
                    msg = f"Row {excel_row_num}: asset '{asset_number_value}' has no model_id."
                    errors.append(msg)
                    detail_rows.append(
                        {
                            "excel_row": excel_row_num,
                            "asset_number": asset_number_value,
                            "location_description": location_description,
                            "status": "asset_missing_model",
                            "message": msg,
                        }
                    )
                    continue

                location, action, location_error = self._resolve_or_create_location(
                    session,
                    asset=asset,
                    location_name=location_description,
                )

                if location_error or not location:
                    if action == "conflict_multiple_asset_locations":
                        summary["conflicts"] += 1
                        status_counter["conflict_multiple_asset_locations"] += 1
                    else:
                        summary["errors"] += 1
                        status_counter[action or "location_resolution_error"] += 1

                    msg = (
                        f"Row {excel_row_num}: failed to resolve location for asset "
                        f"'{asset_number_value}'. {location_error or 'Unknown error.'}"
                    )
                    errors.append(msg)
                    detail_rows.append(
                        {
                            "excel_row": excel_row_num,
                            "asset_number": asset_number_value,
                            "location_description": location_description,
                            "status": action,
                            "message": msg,
                        }
                    )
                    continue

                if action == "created_new_location":
                    summary["locations_created"] += 1
                elif action == "existing_asset_location_updated":
                    summary["locations_updated"] += 1
                else:
                    summary["locations_reused"] += 1

                updated_positions = self._update_positions_for_asset(
                    session,
                    asset_number_id=asset.id,
                    location_id=location.id,
                )
                summary["position_rows_updated"] += updated_positions
                status_counter["success"] += 1

                detail_rows.append(
                    {
                        "excel_row": excel_row_num,
                        "asset_number": asset.number,
                        "asset_number_id": asset.id,
                        "model_id": asset.model_id,
                        "location_description": location_description,
                        "location_id": location.id,
                        "location_name": location.name,
                        "location_description_db": location.description,
                        "status": "success",
                        "action": action,
                        "positions_updated": updated_positions,
                    }
                )

            totals = self._build_totals(
                summary=summary,
                status_counts=dict(status_counter),
                missing_asset_counts=dict(missing_asset_counter.most_common(10)),
                details_count=len(detail_rows),
                errors_count=len(errors),
            )

            result_data = {
                "summary": summary,
                "totals": totals,
                "details": detail_rows,
                "dry_run": dry_run,
                "excel_path": str(excel_path),
            }

            if dry_run:
                session.rollback()
                return self._build_result(
                    True,
                    "Dry run completed. No database changes were committed.",
                    data=result_data,
                    errors=errors,
                )

            session.commit()

            return self._build_result(
                True,
                "Asset location import completed successfully.",
                data=result_data,
                errors=errors,
            )

        except SQLAlchemyError as exc:
            session.rollback()
            self.logger.exception("Database error during asset location import.")
            return self._build_result(
                False,
                "Database error during asset location import.",
                errors=[str(exc)],
            )
        except Exception as exc:
            session.rollback()
            self.logger.exception("Unexpected error during asset location import.")
            return self._build_result(
                False,
                "Unexpected error during asset location import.",
                errors=[str(exc)],
            )
        finally:
            session.close()