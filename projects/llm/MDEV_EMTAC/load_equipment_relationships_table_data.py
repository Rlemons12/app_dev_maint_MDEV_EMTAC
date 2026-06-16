from __future__ import annotations

import os
from datetime import datetime
from typing import Any, Optional

import numpy as np
import pandas as pd
from sqlalchemy import func, text
from sqlalchemy.exc import SQLAlchemyError

from modules.configuration.config import BASE_DIR
from modules.configuration.config_env import DatabaseConfig
from modules.configuration.log_config import (
    debug_id,
    error_id,
    info_id,
    log_timed_operation,
    set_request_id,
    warning_id,
)
from modules.emtacdb.emtacdb_fts import (
    Area,
    AssetNumber,
    Base,
    Building,
    Campus,
    EquipmentGroup,
    Location,
    Model,
    SiteLocation,
)
from modules.emtacdb.emtac_revision_control_db import (
    AreaSnapshot,
    AssetNumberSnapshot,
    EquipmentGroupSnapshot,
    LocationSnapshot,
    ModelSnapshot,
    RevisionControlSession,
    VersionInfo,
)
from modules.emtacdb.utlity.revision_database.snapshot_utils import create_snapshot
from modules.initial_setup.initializer_logger import close_initializer_logger


class PostgreSQLEquipmentRelationshipsLoader:
    """
    Bootstrap loader for equipment/site relationship tables.

    Expected workbook sheets and columns:

    - campus:
        required: id, name
        optional: description, city, state, country

    - building:
        required: id, name, campus_id
        optional: description, address

    - site_location:
        required: id, title
        optional: room_number, site_area, building_id

    - area:
        required: area_id, area

    - equipment_group:
        required: equipment_group_id, area_id, equipment_group

    - model:
        required: model_id, equipment_group_id, model

    - asset_number:
        required: asset_number_id, model_id, asset_number
        optional: asset_description

    - location:
        required: location_id, model_id, location
        optional: location_description, asset_number_id

    Important:
    - This loader is destructive by design.
    - It truncates the target tables before importing.
    - TRUNCATE uses RESTART IDENTITY CASCADE so PostgreSQL sequences reset.
    - Workbook IDs are treated as workbook-local IDs only.
    - Parent-child relationships are rebuilt using workbook ID -> DB object mappings.
    """

    DEFAULT_FILE_PATH = r"E:\emtac\Database\DB_LOADSHEETS\emtac_db_export.xlsx"

    def __init__(self) -> None:
        self.request_id = set_request_id()
        self.db_config = DatabaseConfig()

        info_id("Initialized PostgreSQL Equipment Relationships Loader", self.request_id)

        self.table_order = [
            ("campus", Campus, ["id", "name"], ["description", "city", "state", "country"]),
            ("building", Building, ["id", "name", "campus_id"], ["description", "address"]),
            ("site_location", SiteLocation, ["id", "title"], ["room_number", "site_area", "building_id"]),
            ("area", Area, ["area_id", "area"], []),
            ("equipment_group", EquipmentGroup, ["equipment_group_id", "area_id", "equipment_group"], []),
            ("model", Model, ["model_id", "equipment_group_id", "model"], []),
            ("asset_number", AssetNumber, ["asset_number_id", "model_id", "asset_number"], ["asset_description"]),
            ("location", Location, ["location_id", "model_id", "location"], ["location_description", "asset_number_id"]),
        ]

        self.reset_runtime_state()

    # ------------------------------------------------------------------
    # runtime state
    # ------------------------------------------------------------------
    def reset_runtime_state(self) -> None:
        self.stats: dict[str, int] = {
            "campuses_processed": 0,
            "buildings_processed": 0,
            "site_locations_processed": 0,
            "areas_processed": 0,
            "equipment_groups_processed": 0,
            "models_processed": 0,
            "asset_numbers_processed": 0,
            "locations_processed": 0,
            "duplicates_removed": 0,
            "snapshots_created": 0,
            "errors_encountered": 0,
            "fk_validation_failures": 0,
            "rows_skipped_fk": 0,
            "rows_deleted_before_load": 0,
        }

        self.excel_id_map: dict[str, dict[int, Any]] = {
            "campus": {},
            "building": {},
            "site_location": {},
            "area": {},
            "equipment_group": {},
            "model": {},
            "asset_number": {},
            "location": {},
        }

        self.cleaned_sheets: dict[str, pd.DataFrame] = {}
        self.workbook_fk_errors: list[dict[str, Any]] = []

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _normalize_col_name(value: Any) -> str:
        if value is None:
            return ""
        return str(value).strip().lower()

    @staticmethod
    def _normalize_sheet_name(value: str) -> str:
        return str(value).strip().lower()

    @staticmethod
    def _to_none_if_blank(value: Any) -> Any:
        if pd.isna(value):
            return None
        if isinstance(value, str):
            cleaned = value.strip()
            return cleaned if cleaned else None
        return value

    @staticmethod
    def _safe_int(value: Any) -> Optional[int]:
        if value is None or pd.isna(value):
            return None
        if isinstance(value, str) and value.strip() == "":
            return None
        try:
            return int(float(value))
        except (TypeError, ValueError):
            return None

    def _map_excel_id(self, sheet_name: str, excel_id: Any, db_obj: Any) -> None:
        excel_id_int = self._safe_int(excel_id)
        if excel_id_int is None:
            return
        self.excel_id_map[sheet_name][excel_id_int] = db_obj

    def _resolve_db_id(self, parent_sheet_name: str, excel_id: Any) -> Optional[int]:
        excel_id_int = self._safe_int(excel_id)
        if excel_id_int is None:
            return None

        db_obj = self.excel_id_map[parent_sheet_name].get(excel_id_int)
        if db_obj is None:
            warning_id(
                f"Workbook FK value {excel_id_int} for parent '{parent_sheet_name}' was not found in mapping",
                self.request_id,
            )
            return None

        return getattr(db_obj, "id", None)

    def _resolve_sheet_name_map(self, file_path: str) -> dict[str, str]:
        excel_file = pd.ExcelFile(file_path)
        return {self._normalize_sheet_name(name): name for name in excel_file.sheet_names}

    def _log_fk_issue(
        self,
        sheet_name: str,
        row_index: int,
        row_id: Any,
        field_name: str,
        missing_value: Any,
        parent_sheet: str,
        label: str,
    ) -> None:
        self.stats["fk_validation_failures"] += 1
        issue = {
            "sheet_name": sheet_name,
            "row_index": row_index,
            "row_id": row_id,
            "field_name": field_name,
            "missing_value": missing_value,
            "parent_sheet": parent_sheet,
            "label": label,
        }
        self.workbook_fk_errors.append(issue)

        warning_id(
            f"[FK VALIDATION] {sheet_name} row={row_index} id={row_id} "
            f"has {field_name}={missing_value!r} that does not exist in parent sheet "
            f"'{parent_sheet}' ({label})",
            self.request_id,
        )

    def _normalize_text_columns(self, df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
        for col in columns:
            if col in df.columns:
                df[col] = df[col].apply(self._to_none_if_blank)
        return df

    def _normalize_int_columns(self, df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
        for col in columns:
            if col in df.columns:
                df[col] = df[col].apply(self._safe_int)
        return df

    # ------------------------------------------------------------------
    # db setup
    # ------------------------------------------------------------------
    def create_database_tables(self, session) -> None:
        try:
            info_id("Creating database tables if they don't exist", self.request_id)
            engine = session.bind
            Base.metadata.create_all(engine)
            info_id("Database tables created/verified", self.request_id)
        except Exception as exc:
            error_id(f"Error creating database tables: {exc}", self.request_id)
            raise

    # ------------------------------------------------------------------
    # destructive bootstrap wipe
    # ------------------------------------------------------------------
    def wipe_target_tables(self, session) -> None:
        """
        Wipe the target relationship tables using PostgreSQL TRUNCATE.

        RESTART IDENTITY resets serial/identity sequences.
        CASCADE removes dependent rows if PostgreSQL requires it.
        """
        try:
            info_id(
                "Wiping target relationship tables before import using TRUNCATE RESTART IDENTITY CASCADE",
                self.request_id,
            )

            counts_before = {
                "image_position_association": session.execute(
                    text("SELECT COUNT(*) FROM image_position_association")
                ).scalar() or 0,
                "part_position_image": session.execute(
                    text("SELECT COUNT(*) FROM part_position_image")
                ).scalar() or 0,
                "drawing_position": session.execute(
                    text("SELECT COUNT(*) FROM drawing_position")
                ).scalar() or 0,
                "site_location": session.query(SiteLocation).count(),
                "building": session.query(Building).count(),
                "campus": session.query(Campus).count(),
                "location": session.query(Location).count(),
                "asset_number": session.query(AssetNumber).count(),
                "model": session.query(Model).count(),
                "equipment_group": session.query(EquipmentGroup).count(),
                "area": session.query(Area).count(),
            }

            total_deleted = sum(counts_before.values())

            session.execute(
                text(
                    """
                    TRUNCATE TABLE
                        image_position_association,
                        part_position_image,
                        drawing_position,
                        site_location,
                        building,
                        campus,
                        location,
                        asset_number,
                        model,
                        equipment_group,
                        area
                    RESTART IDENTITY CASCADE
                    """
                )
            )

            session.flush()

            for table_name, count_before in counts_before.items():
                info_id(f"Truncated {table_name} (previous rows: {count_before})", self.request_id)

            self.stats["rows_deleted_before_load"] = total_deleted
            info_id(
                f"Finished truncating target tables. Total rows removed before load: {total_deleted}",
                self.request_id,
            )

        except Exception as exc:
            error_id(f"Error truncating target tables: {exc}", self.request_id)
            raise

    # ------------------------------------------------------------------
    # workbook validation / cleaning
    # ------------------------------------------------------------------
    def validate_excel_file(self, file_path: str) -> tuple[bool, str]:
        info_id(f"Validating Excel file: {file_path}", self.request_id)

        if not os.path.exists(file_path):
            return False, f"Excel file not found: {file_path}"

        try:
            normalized_sheet_map = self._resolve_sheet_name_map(file_path)
            required_sheets = [name for name, _, _, _ in self.table_order]
            missing = [name for name in required_sheets if name not in normalized_sheet_map]
            if missing:
                return False, f"Missing required sheets: {missing}"

            for required in required_sheets:
                actual_name = normalized_sheet_map[required]
                df = pd.read_excel(file_path, sheet_name=actual_name)
                if df.empty:
                    warning_id(f"Sheet '{actual_name}' is empty", self.request_id)

            return True, "Valid"

        except Exception as exc:
            return False, f"Error reading Excel file: {exc}"

    def clean_dataframe(
            self,
            df: pd.DataFrame,
            required_columns: list[str],
            optional_columns: list[str],
            sheet_name: str,
    ) -> pd.DataFrame:
        info_id(f"Cleaning DataFrame for sheet: {sheet_name}", self.request_id)

        try:
            original_rows = len(df)

            df = df.copy()
            df.columns = [self._normalize_col_name(col) for col in df.columns]

            if df.columns.duplicated().any():
                dupes = list(df.columns[df.columns.duplicated()])
                warning_id(f"Duplicate normalized columns in {sheet_name}: {dupes}", self.request_id)
                df = df.loc[:, ~df.columns.duplicated()]

            df = df.dropna(axis=1, how="all")
            df = df.replace({np.nan: None})

            missing_required = [col for col in required_columns if col not in df.columns]
            if missing_required:
                raise ValueError(f"Missing required columns in {sheet_name}: {missing_required}")

            for col in optional_columns:
                if col not in df.columns:
                    df[col] = None

            final_columns = required_columns + [col for col in optional_columns if col not in required_columns]
            cleaned_df = df[final_columns].copy()

            cleaned_df = self._normalize_int_columns(
                cleaned_df,
                [
                    "id",
                    "campus_id",
                    "building_id",
                    "area_id",
                    "equipment_group_id",
                    "model_id",
                    "asset_number_id",
                    "location_id",
                ],
            )

            cleaned_df = self._normalize_text_columns(
                cleaned_df,
                [
                    "name",
                    "description",
                    "city",
                    "state",
                    "country",
                    "address",
                    "title",
                    "site_area",
                    "area",
                    "equipment_group",
                    "model",
                    "asset_number",
                    "asset_description",
                    "location",
                    "location_description",
                ],
            )

            if "room_number" in cleaned_df.columns:
                cleaned_df["room_number"] = cleaned_df["room_number"].apply(
                    lambda value: None if pd.isna(value) else str(value).strip()
                )
                cleaned_df["room_number"] = cleaned_df["room_number"].replace({"": None})

                def _normalize_room_number(value: Any) -> Any:
                    if value is None:
                        return None
                    try:
                        numeric = float(value)
                        if numeric.is_integer():
                            return str(int(numeric))
                    except (TypeError, ValueError):
                        pass
                    return str(value).strip()

                cleaned_df["room_number"] = cleaned_df["room_number"].apply(_normalize_room_number)

            key_column_map = {
                "campus": "name",
                "building": "name",
                "site_location": "title",
                "area": "area",
                "equipment_group": "equipment_group",
                "model": "model",
                "asset_number": "asset_number",
                "location": "location",
            }

            key_field = key_column_map.get(sheet_name)
            if key_field and key_field in cleaned_df.columns:
                before = len(cleaned_df)
                cleaned_df = cleaned_df[cleaned_df[key_field].notna()]
                removed = before - len(cleaned_df)
                if removed > 0:
                    warning_id(
                        f"Removed {removed} rows with blank '{key_field}' in sheet '{sheet_name}'",
                        self.request_id,
                    )

            cleaned_df = cleaned_df.reset_index(drop=True)

            info_id(f"Cleaned {sheet_name}: {original_rows} -> {len(cleaned_df)} rows", self.request_id)
            return cleaned_df

        except Exception as exc:
            error_id(f"Error cleaning DataFrame for {sheet_name}: {exc}", self.request_id)
            raise

    def preload_cleaned_sheets(self, file_path: str, normalized_sheet_map: dict[str, str]) -> None:
        self.cleaned_sheets = {}

        for expected_sheet_name, _model_class, required_columns, optional_columns in self.table_order:
            actual_sheet_name = normalized_sheet_map[expected_sheet_name]

            with log_timed_operation(f"preload_{expected_sheet_name}_sheet", self.request_id):
                df_raw = pd.read_excel(file_path, sheet_name=actual_sheet_name)

                df_cleaned = self.clean_dataframe(
                    df=df_raw,
                    required_columns=required_columns,
                    optional_columns=optional_columns,
                    sheet_name=expected_sheet_name,
                )

                self.cleaned_sheets[expected_sheet_name] = df_cleaned

    def validate_workbook_foreign_keys(self) -> None:
        info_id("Validating workbook foreign keys before database load", self.request_id)
        self.workbook_fk_errors = []

        campus_ids = set()
        building_ids = set()
        area_ids = set()
        equipment_group_ids = set()
        model_ids = set()
        asset_number_ids = set()

        if "campus" in self.cleaned_sheets:
            campus_ids = {
                self._safe_int(v)
                for v in self.cleaned_sheets["campus"]["id"].tolist()
                if self._safe_int(v) is not None
            }

        if "building" in self.cleaned_sheets:
            building_ids = {
                self._safe_int(v)
                for v in self.cleaned_sheets["building"]["id"].tolist()
                if self._safe_int(v) is not None
            }

        if "area" in self.cleaned_sheets:
            area_ids = {
                self._safe_int(v)
                for v in self.cleaned_sheets["area"]["area_id"].tolist()
                if self._safe_int(v) is not None
            }

        if "equipment_group" in self.cleaned_sheets:
            equipment_group_ids = {
                self._safe_int(v)
                for v in self.cleaned_sheets["equipment_group"]["equipment_group_id"].tolist()
                if self._safe_int(v) is not None
            }

        if "model" in self.cleaned_sheets:
            model_ids = {
                self._safe_int(v)
                for v in self.cleaned_sheets["model"]["model_id"].tolist()
                if self._safe_int(v) is not None
            }

        if "asset_number" in self.cleaned_sheets:
            asset_number_ids = {
                self._safe_int(v)
                for v in self.cleaned_sheets["asset_number"]["asset_number_id"].tolist()
                if self._safe_int(v) is not None
            }

        if "building" in self.cleaned_sheets:
            df = self.cleaned_sheets["building"]
            for idx, row in df.iterrows():
                campus_id = self._safe_int(row.get("campus_id"))
                if campus_id is not None and campus_id not in campus_ids:
                    self._log_fk_issue(
                        sheet_name="building",
                        row_index=idx,
                        row_id=row.get("id"),
                        field_name="campus_id",
                        missing_value=campus_id,
                        parent_sheet="campus",
                        label=row.get("name") or "<blank building>",
                    )

        if "site_location" in self.cleaned_sheets:
            df = self.cleaned_sheets["site_location"]
            for idx, row in df.iterrows():
                building_id = self._safe_int(row.get("building_id"))
                if building_id is not None and building_id not in building_ids:
                    self._log_fk_issue(
                        sheet_name="site_location",
                        row_index=idx,
                        row_id=row.get("id"),
                        field_name="building_id",
                        missing_value=building_id,
                        parent_sheet="building",
                        label=row.get("title") or "<blank site_location>",
                    )

        if "equipment_group" in self.cleaned_sheets:
            df = self.cleaned_sheets["equipment_group"]
            for idx, row in df.iterrows():
                area_id = self._safe_int(row.get("area_id"))
                if area_id is not None and area_id not in area_ids:
                    self._log_fk_issue(
                        sheet_name="equipment_group",
                        row_index=idx,
                        row_id=row.get("equipment_group_id"),
                        field_name="area_id",
                        missing_value=area_id,
                        parent_sheet="area",
                        label=row.get("equipment_group") or "<blank equipment_group>",
                    )

        if "model" in self.cleaned_sheets:
            df = self.cleaned_sheets["model"]
            for idx, row in df.iterrows():
                equipment_group_id = self._safe_int(row.get("equipment_group_id"))
                if equipment_group_id is not None and equipment_group_id not in equipment_group_ids:
                    self._log_fk_issue(
                        sheet_name="model",
                        row_index=idx,
                        row_id=row.get("model_id"),
                        field_name="equipment_group_id",
                        missing_value=equipment_group_id,
                        parent_sheet="equipment_group",
                        label=row.get("model") or "<blank model>",
                    )

        if "asset_number" in self.cleaned_sheets:
            df = self.cleaned_sheets["asset_number"]
            for idx, row in df.iterrows():
                model_id = self._safe_int(row.get("model_id"))
                if model_id is not None and model_id not in model_ids:
                    self._log_fk_issue(
                        sheet_name="asset_number",
                        row_index=idx,
                        row_id=row.get("asset_number_id"),
                        field_name="model_id",
                        missing_value=model_id,
                        parent_sheet="model",
                        label=row.get("asset_number") or "<blank asset_number>",
                    )

        if "location" in self.cleaned_sheets:
            df = self.cleaned_sheets["location"]
            for idx, row in df.iterrows():
                model_id = self._safe_int(row.get("model_id"))
                if model_id is not None and model_id not in model_ids:
                    self._log_fk_issue(
                        sheet_name="location",
                        row_index=idx,
                        row_id=row.get("location_id"),
                        field_name="model_id",
                        missing_value=model_id,
                        parent_sheet="model",
                        label=row.get("location") or "<blank location>",
                    )

                asset_number_id = self._safe_int(row.get("asset_number_id"))
                if asset_number_id is not None and asset_number_id not in asset_number_ids:
                    self._log_fk_issue(
                        sheet_name="location",
                        row_index=idx,
                        row_id=row.get("location_id"),
                        field_name="asset_number_id",
                        missing_value=asset_number_id,
                        parent_sheet="asset_number",
                        label=row.get("location") or "<blank location>",
                    )

        if self.workbook_fk_errors:
            warning_id(
                f"Workbook FK validation found {len(self.workbook_fk_errors)} broken references",
                self.request_id,
            )
        else:
            info_id("Workbook FK validation passed with no broken references", self.request_id)

    def export_fk_validation_report(self) -> Optional[str]:
        if not self.workbook_fk_errors:
            return None

        try:
            report_directory = os.path.join(BASE_DIR, "Database", "DB_LOADSHEETS_BACKUP")
            os.makedirs(report_directory, exist_ok=True)

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            report_path = os.path.join(
                report_directory,
                f"equipment_relationships_fk_validation_{timestamp}.csv",
            )

            pd.DataFrame(self.workbook_fk_errors).to_csv(report_path, index=False)

            warning_id(f"FK validation report written to: {report_path}", self.request_id)
            return report_path

        except Exception as exc:
            warning_id(f"Failed to write FK validation report: {exc}", self.request_id)
            return None

    # ------------------------------------------------------------------
    # backup
    # ------------------------------------------------------------------
    def create_database_backup(self, session) -> None:
        try:
            info_id("Creating database backup", self.request_id)

            with log_timed_operation("create_database_backup", self.request_id):
                backup_directory = os.path.join(BASE_DIR, "Database", "DB_LOADSHEETS_BACKUP")
                os.makedirs(backup_directory, exist_ok=True)

                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                excel_file_name = f"equipment_relationships_backup_{timestamp}.xlsx"
                excel_file_path = os.path.join(backup_directory, excel_file_name)

                backup_data: dict[str, pd.DataFrame] = {}

                backup_data["campus"] = pd.DataFrame(
                    [
                        (obj.id, obj.name, obj.description, obj.city, obj.state, obj.country)
                        for obj in session.query(Campus).all()
                    ],
                    columns=["id", "name", "description", "city", "state", "country"],
                )

                backup_data["building"] = pd.DataFrame(
                    [
                        (obj.id, obj.name, obj.description, obj.address, obj.campus_id)
                        for obj in session.query(Building).all()
                    ],
                    columns=["id", "name", "description", "address", "campus_id"],
                )

                backup_data["site_location"] = pd.DataFrame(
                    [
                        (obj.id, obj.title, obj.room_number, obj.site_area, obj.building_id)
                        for obj in session.query(SiteLocation).all()
                    ],
                    columns=["id", "title", "room_number", "site_area", "building_id"],
                )

                backup_data["area"] = pd.DataFrame(
                    [(obj.id, obj.name) for obj in session.query(Area).all()],
                    columns=["area_id", "area"],
                )

                backup_data["equipment_group"] = pd.DataFrame(
                    [(obj.id, obj.area_id, obj.name) for obj in session.query(EquipmentGroup).all()],
                    columns=["equipment_group_id", "area_id", "equipment_group"],
                )

                backup_data["model"] = pd.DataFrame(
                    [(obj.id, obj.equipment_group_id, obj.name) for obj in session.query(Model).all()],
                    columns=["model_id", "equipment_group_id", "model"],
                )

                backup_data["asset_number"] = pd.DataFrame(
                    [(obj.id, obj.model_id, obj.number, obj.description) for obj in session.query(AssetNumber).all()],
                    columns=["asset_number_id", "model_id", "asset_number", "asset_description"],
                )

                backup_data["location"] = pd.DataFrame(
                    [
                        (obj.id, obj.model_id, obj.name, obj.description, obj.asset_number_id)
                        for obj in session.query(Location).all()
                    ],
                    columns=["location_id", "model_id", "location", "location_description", "asset_number_id"],
                )

                with pd.ExcelWriter(excel_file_path, engine="openpyxl") as writer:
                    for sheet_name, df in backup_data.items():
                        df.to_excel(writer, sheet_name=sheet_name, index=False)

                info_id(f"Database backup created: {excel_file_name}", self.request_id)

        except Exception as exc:
            error_id(f"Error creating database backup: {exc}", self.request_id)
            warning_id(f"Backup failed: {exc}", self.request_id)

    # ------------------------------------------------------------------
    # duplicate handling
    # ------------------------------------------------------------------
    def delete_duplicates_enhanced(self, session, model, unique_columns: list[str], sheet_name: str) -> None:
        try:
            info_id(
                f"Removing duplicates from {sheet_name} based on columns {unique_columns}",
                self.request_id,
            )

            query_columns = [getattr(model, col) for col in unique_columns]
            duplicates = (
                session.query(*query_columns, func.count().label("count"))
                .group_by(*query_columns)
                .having(func.count() > 1)
                .all()
            )

            duplicates_removed = 0

            for dup in duplicates:
                values = dup[:-1]
                filters = {col_name: value for col_name, value in zip(unique_columns, values)}

                records = (
                    session.query(model)
                    .filter_by(**filters)
                    .order_by(model.id.asc())
                    .all()
                )

                for record in records[1:]:
                    session.delete(record)
                    duplicates_removed += 1

            if duplicates_removed:
                info_id(f"Removed {duplicates_removed} duplicates from {sheet_name}", self.request_id)
                self.stats["duplicates_removed"] += duplicates_removed

        except Exception as exc:
            error_id(f"Error removing duplicates from {sheet_name}: {exc}", self.request_id)
            raise

    # ------------------------------------------------------------------
    # row processors
    # ------------------------------------------------------------------
    def _get_or_create_campus(self, session, row: pd.Series) -> Campus:
        name = self._to_none_if_blank(row["name"])
        if not name:
            raise ValueError("Campus row missing name")

        obj = session.query(Campus).filter_by(name=name).first()
        if not obj:
            obj = Campus(
                name=name,
                description=self._to_none_if_blank(row.get("description")),
                city=self._to_none_if_blank(row.get("city")),
                state=self._to_none_if_blank(row.get("state")),
                country=self._to_none_if_blank(row.get("country")),
            )
            session.add(obj)
        else:
            obj.description = self._to_none_if_blank(row.get("description"))
            obj.city = self._to_none_if_blank(row.get("city"))
            obj.state = self._to_none_if_blank(row.get("state"))
            obj.country = self._to_none_if_blank(row.get("country"))

        session.flush()
        return obj

    def _get_or_create_building(self, session, row: pd.Series) -> Building:
        name = self._to_none_if_blank(row["name"])
        if not name:
            raise ValueError("Building row missing name")

        workbook_campus_id = self._safe_int(row.get("campus_id"))
        campus_db_id = self._resolve_db_id("campus", workbook_campus_id)

        if workbook_campus_id is not None and campus_db_id is None:
            raise ValueError(
                f"Building mapping failed: workbook campus_id={workbook_campus_id} "
                f"for building id={row.get('id')} name={name!r}"
            )

        obj = session.query(Building).filter_by(name=name, campus_id=campus_db_id).first()
        if not obj:
            obj = Building(
                name=name,
                description=self._to_none_if_blank(row.get("description")),
                address=self._to_none_if_blank(row.get("address")),
                campus_id=campus_db_id,
            )
            session.add(obj)
        else:
            obj.description = self._to_none_if_blank(row.get("description"))
            obj.address = self._to_none_if_blank(row.get("address"))
            obj.campus_id = campus_db_id

        session.flush()
        return obj

    @staticmethod
    def _safe_str(value: Any) -> Optional[str]:
        if value is None or pd.isna(value):
            return None
        text = str(value).strip()
        return text if text else None

    def _get_or_create_site_location(self, session, row: pd.Series) -> SiteLocation:
        title = self._to_none_if_blank(row["title"])
        if not title:
            raise ValueError("SiteLocation row missing title")

        workbook_building_id = self._safe_int(row.get("building_id"))
        building_db_id = self._resolve_db_id("building", workbook_building_id)

        if workbook_building_id is not None and building_db_id is None:
            raise ValueError(
                f"SiteLocation mapping failed: workbook building_id={workbook_building_id} "
                f"for site_location id={row.get('id')} title={title!r}"
            )

        raw_room_number = self._to_none_if_blank(row.get("room_number"))
        room_number = str(raw_room_number).strip() if raw_room_number is not None else "Unknown"
        if not room_number:
            room_number = "Unknown"

        raw_site_area = self._to_none_if_blank(row.get("site_area"))
        site_area = str(raw_site_area).strip() if raw_site_area is not None else "General"
        if not site_area:
            site_area = "General"

        obj = (
            session.query(SiteLocation)
            .filter_by(
                title=title,
                room_number=room_number,
                building_id=building_db_id,
            )
            .first()
        )

        if not obj:
            obj = SiteLocation(
                title=title,
                room_number=room_number,
                site_area=site_area,
                building_id=building_db_id,
            )
            session.add(obj)
        else:
            obj.room_number = room_number
            obj.site_area = site_area
            obj.building_id = building_db_id

        session.flush()
        return obj

    def _get_or_create_area(self, session, row: pd.Series) -> Area:
        name = self._to_none_if_blank(row["area"])
        if not name:
            raise ValueError("Area row missing area value")

        obj = session.query(Area).filter_by(name=name).first()
        if not obj:
            obj = Area(name=name, description=None)
            session.add(obj)

        session.flush()
        return obj

    def _get_or_create_equipment_group(self, session, row: pd.Series) -> EquipmentGroup:
        name = self._to_none_if_blank(row["equipment_group"])
        if not name:
            raise ValueError("EquipmentGroup row missing equipment_group value")

        workbook_area_id = self._safe_int(row.get("area_id"))
        area_db_id = self._resolve_db_id("area", workbook_area_id)

        if workbook_area_id is not None and area_db_id is None:
            raise ValueError(
                f"EquipmentGroup mapping failed: workbook area_id={workbook_area_id} "
                f"for equipment_group_id={row.get('equipment_group_id')} name={name!r}"
            )

        obj = session.query(EquipmentGroup).filter_by(name=name, area_id=area_db_id).first()
        if not obj:
            obj = EquipmentGroup(name=name, area_id=area_db_id, description=None)
            session.add(obj)
        else:
            obj.area_id = area_db_id

        session.flush()
        return obj

    def _get_or_create_model(self, session, row: pd.Series) -> Model:
        name = self._to_none_if_blank(row["model"])
        if not name:
            raise ValueError("Model row missing model value")

        workbook_equipment_group_id = self._safe_int(row.get("equipment_group_id"))
        equipment_group_db_id = self._resolve_db_id("equipment_group", workbook_equipment_group_id)

        if workbook_equipment_group_id is not None and equipment_group_db_id is None:
            raise ValueError(
                f"Model mapping failed: workbook equipment_group_id={workbook_equipment_group_id} "
                f"for model_id={row.get('model_id')} name={name!r}"
            )

        obj = session.query(Model).filter_by(name=name, equipment_group_id=equipment_group_db_id).first()
        if not obj:
            obj = Model(name=name, equipment_group_id=equipment_group_db_id, description=None)
            session.add(obj)
        else:
            obj.equipment_group_id = equipment_group_db_id

        session.flush()
        return obj

    def _get_or_create_asset_number(self, session, row: pd.Series) -> AssetNumber:
        number = self._to_none_if_blank(row["asset_number"])
        description = self._to_none_if_blank(row.get("asset_description"))

        if not number:
            raise ValueError("AssetNumber row missing asset_number value")

        workbook_model_id = self._safe_int(row.get("model_id"))
        model_db_id = self._resolve_db_id("model", workbook_model_id)

        if workbook_model_id is not None and model_db_id is None:
            raise ValueError(
                f"AssetNumber mapping failed: workbook model_id={workbook_model_id} "
                f"for asset_number_id={row.get('asset_number_id')} number={number!r}"
            )

        obj = session.query(AssetNumber).filter_by(number=number, model_id=model_db_id).first()
        if not obj:
            obj = AssetNumber(number=number, model_id=model_db_id, description=description)
            session.add(obj)
        else:
            obj.model_id = model_db_id
            obj.description = description

        session.flush()
        return obj

    def _get_or_create_location(self, session, row: pd.Series) -> Location:
        name = self._to_none_if_blank(row["location"])
        if not name:
            raise ValueError("Location row missing location value")

        description = self._to_none_if_blank(row.get("location_description"))
        workbook_model_id = self._safe_int(row.get("model_id"))
        model_db_id = self._resolve_db_id("model", workbook_model_id)

        if workbook_model_id is not None and model_db_id is None:
            raise ValueError(
                f"Location mapping failed: workbook model_id={workbook_model_id} "
                f"for location_id={row.get('location_id')} location={name!r}"
            )

        workbook_asset_number_id = self._safe_int(row.get("asset_number_id"))
        asset_number_db_id = None
        if workbook_asset_number_id is not None:
            asset_number_db_id = self._resolve_db_id("asset_number", workbook_asset_number_id)
            if asset_number_db_id is None:
                raise ValueError(
                    f"Location mapping failed: workbook asset_number_id={workbook_asset_number_id} "
                    f"for location_id={row.get('location_id')} location={name!r}"
                )

        obj = (
            session.query(Location)
            .filter_by(name=name, model_id=model_db_id, asset_number_id=asset_number_db_id)
            .first()
        )
        if not obj:
            obj = Location(
                name=name,
                model_id=model_db_id,
                asset_number_id=asset_number_db_id,
                description=description,
            )
            session.add(obj)
        else:
            obj.model_id = model_db_id
            obj.asset_number_id = asset_number_db_id
            obj.description = description

        session.flush()
        return obj

    def process_single_table(self, session, df: pd.DataFrame, model_class: type, sheet_name: str) -> None:
        info_id(f"Processing {sheet_name} table", self.request_id)

        processed_count = 0

        for index, row in df.iterrows():
            try:
                if sheet_name == "campus":
                    obj = self._get_or_create_campus(session, row)
                    workbook_id = row.get("id")

                elif sheet_name == "building":
                    workbook_campus_id = self._safe_int(row.get("campus_id"))
                    if workbook_campus_id is not None and self._resolve_db_id("campus", workbook_campus_id) is None:
                        self.stats["rows_skipped_fk"] += 1
                        warning_id(
                            f"Skipping building row {index} due to unresolved campus_id={workbook_campus_id}",
                            self.request_id,
                        )
                        continue
                    obj = self._get_or_create_building(session, row)
                    workbook_id = row.get("id")

                elif sheet_name == "site_location":
                    workbook_building_id = self._safe_int(row.get("building_id"))
                    if workbook_building_id is not None and self._resolve_db_id("building", workbook_building_id) is None:
                        self.stats["rows_skipped_fk"] += 1
                        warning_id(
                            f"Skipping site_location row {index} due to unresolved building_id={workbook_building_id}",
                            self.request_id,
                        )
                        continue
                    obj = self._get_or_create_site_location(session, row)
                    workbook_id = row.get("id")

                elif sheet_name == "area":
                    obj = self._get_or_create_area(session, row)
                    workbook_id = row.get("area_id")

                elif sheet_name == "equipment_group":
                    workbook_area_id = self._safe_int(row.get("area_id"))
                    if workbook_area_id is not None and self._resolve_db_id("area", workbook_area_id) is None:
                        self.stats["rows_skipped_fk"] += 1
                        warning_id(
                            f"Skipping equipment_group row {index} due to unresolved area_id={workbook_area_id}",
                            self.request_id,
                        )
                        continue
                    obj = self._get_or_create_equipment_group(session, row)
                    workbook_id = row.get("equipment_group_id")

                elif sheet_name == "model":
                    workbook_equipment_group_id = self._safe_int(row.get("equipment_group_id"))
                    if (
                        workbook_equipment_group_id is not None
                        and self._resolve_db_id("equipment_group", workbook_equipment_group_id) is None
                    ):
                        self.stats["rows_skipped_fk"] += 1
                        warning_id(
                            f"Skipping model row {index} due to unresolved equipment_group_id={workbook_equipment_group_id}",
                            self.request_id,
                        )
                        continue
                    obj = self._get_or_create_model(session, row)
                    workbook_id = row.get("model_id")

                elif sheet_name == "asset_number":
                    workbook_model_id = self._safe_int(row.get("model_id"))
                    if workbook_model_id is not None and self._resolve_db_id("model", workbook_model_id) is None:
                        self.stats["rows_skipped_fk"] += 1
                        warning_id(
                            f"Skipping asset_number row {index} due to unresolved model_id={workbook_model_id}",
                            self.request_id,
                        )
                        continue
                    obj = self._get_or_create_asset_number(session, row)
                    workbook_id = row.get("asset_number_id")

                elif sheet_name == "location":
                    workbook_model_id = self._safe_int(row.get("model_id"))
                    if workbook_model_id is not None and self._resolve_db_id("model", workbook_model_id) is None:
                        self.stats["rows_skipped_fk"] += 1
                        warning_id(
                            f"Skipping location row {index} due to unresolved model_id={workbook_model_id}",
                            self.request_id,
                        )
                        continue

                    workbook_asset_number_id = self._safe_int(row.get("asset_number_id"))
                    if (
                        workbook_asset_number_id is not None
                        and self._resolve_db_id("asset_number", workbook_asset_number_id) is None
                    ):
                        self.stats["rows_skipped_fk"] += 1
                        warning_id(
                            f"Skipping location row {index} due to unresolved asset_number_id={workbook_asset_number_id}",
                            self.request_id,
                        )
                        continue

                    obj = self._get_or_create_location(session, row)
                    workbook_id = row.get("location_id")

                else:
                    continue

                self._map_excel_id(sheet_name, workbook_id, obj)

                debug_id(
                    f"MAPPED {sheet_name} workbook id={workbook_id} -> DB id={obj.id}",
                    self.request_id,
                )

                processed_count += 1

                if processed_count % 100 == 0:
                    debug_id(f"Processed {processed_count} {sheet_name} records", self.request_id)

            except Exception as exc:
                error_id(f"Error processing {sheet_name} row {index}: {exc}", self.request_id)
                self.stats["errors_encountered"] += 1
                raise

        stats_map = {
            "campus": "campuses_processed",
            "building": "buildings_processed",
            "site_location": "site_locations_processed",
            "area": "areas_processed",
            "equipment_group": "equipment_groups_processed",
            "model": "models_processed",
            "asset_number": "asset_numbers_processed",
            "location": "locations_processed",
        }
        stats_key = stats_map.get(sheet_name)
        if stats_key:
            self.stats[stats_key] = processed_count

        info_id(f"Processed {processed_count} {sheet_name} records", self.request_id)

    # ------------------------------------------------------------------
    # snapshots
    # ------------------------------------------------------------------
    def create_revision_snapshots(self, main_session) -> None:
        try:
            info_id("Creating revision control snapshots", self.request_id)

            with log_timed_operation("create_revision_snapshots", self.request_id):
                rev_session = RevisionControlSession()

                try:
                    current_max_version = rev_session.query(func.max(VersionInfo.version_number)).scalar()
                    next_version = (current_max_version or 0) + 1

                    new_version = VersionInfo(
                        version_number=next_version,
                        description=(
                            "Bootstrap relationship import after TRUNCATE RESTART IDENTITY CASCADE "
                            "including campus/building/site_location"
                        ),
                    )
                    rev_session.add(new_version)
                    rev_session.commit()

                    snapshots_created = 0
                    snapshot_mapping = [
                        (Area, AreaSnapshot, "areas"),
                        (EquipmentGroup, EquipmentGroupSnapshot, "equipment groups"),
                        (Model, ModelSnapshot, "models"),
                        (AssetNumber, AssetNumberSnapshot, "asset numbers"),
                        (Location, LocationSnapshot, "locations"),
                    ]

                    for entity_class, snapshot_class, label in snapshot_mapping:
                        try:
                            entities = main_session.query(entity_class).all()
                            for entity in entities:
                                create_snapshot(entity, rev_session, snapshot_class)
                                snapshots_created += 1
                            info_id(f"Created {len(entities)} {label} snapshots", self.request_id)
                        except Exception as exc:
                            warning_id(f"Error creating {label} snapshots: {exc}", self.request_id)

                    rev_session.commit()
                    self.stats["snapshots_created"] = snapshots_created

                finally:
                    rev_session.close()

        except Exception as exc:
            warning_id(f"Snapshot creation failed: {exc}", self.request_id)

    # ------------------------------------------------------------------
    # summary
    # ------------------------------------------------------------------
    def display_processing_summary(self) -> None:
        info_id("Processing Summary", self.request_id)

        summary_order = [
            "campuses_processed",
            "buildings_processed",
            "site_locations_processed",
            "areas_processed",
            "equipment_groups_processed",
            "models_processed",
            "asset_numbers_processed",
            "locations_processed",
            "duplicates_removed",
            "snapshots_created",
            "errors_encountered",
            "fk_validation_failures",
            "rows_skipped_fk",
            "rows_deleted_before_load",
        ]

        for key in summary_order:
            info_id(f"{key.replace('_', ' ').title()}: {self.stats.get(key, 0)}", self.request_id)

        if self.excel_id_map["campus"]:
            info_id(f"Campus workbook-ID mappings created: {len(self.excel_id_map['campus'])}", self.request_id)
        else:
            warning_id("No Campus workbook-ID mappings were created", self.request_id)

        if self.excel_id_map["building"]:
            info_id(f"Building workbook-ID mappings created: {len(self.excel_id_map['building'])}", self.request_id)
        else:
            warning_id("No Building workbook-ID mappings were created", self.request_id)

        if self.excel_id_map["site_location"]:
            info_id(
                f"SiteLocation workbook-ID mappings created: {len(self.excel_id_map['site_location'])}",
                self.request_id,
            )
        else:
            warning_id("No SiteLocation workbook-ID mappings were created", self.request_id)

        if self.excel_id_map["model"]:
            info_id(f"Model workbook-ID mappings created: {len(self.excel_id_map['model'])}", self.request_id)
        else:
            warning_id("No Model workbook-ID mappings were created", self.request_id)

    # ------------------------------------------------------------------
    # main loader
    # ------------------------------------------------------------------
    def load_equipment_relationships(self, file_path: Optional[str] = None) -> bool:
        try:
            self.reset_runtime_state()
            info_id("Equipment Relationships Data Import", self.request_id)

            if not file_path:
                file_path = self.DEFAULT_FILE_PATH

            info_id(f"REAL FILE PATH: {file_path}", self.request_id)

            is_valid, message = self.validate_excel_file(file_path)
            if not is_valid:
                raise ValueError(f"Invalid Excel file: {message}")

            normalized_sheet_map = self._resolve_sheet_name_map(file_path)

            self.preload_cleaned_sheets(file_path, normalized_sheet_map)
            self.validate_workbook_foreign_keys()
            self.export_fk_validation_report()

            with self.db_config.main_session() as session:
                self.create_database_tables(session)

                self.create_database_backup(session)

                self.wipe_target_tables(session)
                session.flush()

                info_id("Processing tables in dependency order...", self.request_id)

                try:
                    for expected_sheet_name, model_class, _required_columns, _optional_columns in self.table_order:
                        df_cleaned = self.cleaned_sheets[expected_sheet_name]
                        self.process_single_table(session, df_cleaned, model_class, expected_sheet_name)

                    duplicate_mappings = [
                        (Campus, ["name"], "campus"),
                        (Building, ["name", "campus_id"], "building"),
                        (SiteLocation, ["title", "room_number", "building_id"], "site_location"),
                        (Area, ["name"], "area"),
                        (EquipmentGroup, ["name", "area_id"], "equipment_group"),
                        (Model, ["name", "equipment_group_id"], "model"),
                        (AssetNumber, ["number", "model_id"], "asset_number"),
                        (Location, ["name", "model_id", "asset_number_id"], "location"),
                    ]

                    for model_class, unique_columns, sheet_name in duplicate_mappings:
                        self.delete_duplicates_enhanced(session, model_class, unique_columns, sheet_name)

                    session.commit()
                    info_id("All database changes committed successfully", self.request_id)

                except Exception:
                    session.rollback()
                    error_id(
                        "Database load failed. Transaction rolled back; no partial changes were kept.",
                        self.request_id,
                    )
                    raise

                self.create_revision_snapshots(session)

            self.display_processing_summary()
            info_id("Equipment Relationships Import Completed Successfully!", self.request_id)
            return True

        except SQLAlchemyError as exc:
            error_id(f"Equipment relationships import failed: {exc}", self.request_id, exc_info=True)
            return False
        except Exception as exc:
            error_id(f"Equipment relationships import failed: {exc}", self.request_id, exc_info=True)
            return False


def main() -> None:
    info_id("Starting Equipment Relationships Data Import", request_id=None)
    loader = None

    try:
        loader = PostgreSQLEquipmentRelationshipsLoader()
        success = loader.load_equipment_relationships()

        if success:
            info_id("Equipment Relationships Import Completed Successfully!", request_id=loader.request_id)
        else:
            warning_id("Equipment Relationships Import Completed with Issues", request_id=loader.request_id)

    except KeyboardInterrupt:
        warning_id("Import interrupted by user", request_id=loader.request_id if loader else None)
    except Exception as exc:
        error_id(f"Import failed: {exc}", request_id=loader.request_id if loader else None, exc_info=True)
    finally:
        try:
            close_initializer_logger()
        except Exception:
            pass


if __name__ == "__main__":
    main()