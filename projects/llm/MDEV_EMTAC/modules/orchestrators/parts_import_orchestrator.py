from __future__ import annotations

import os
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from modules.configuration.config import DB_LOADSHEET
from modules.configuration.config_env import get_db_config
from modules.configuration.log_config import logger, with_request_id
from modules.database_manager.db_manager import RelationshipManager
from modules.services.part_service import PartService


class PartsImportOrchestrator:
    """
    Orchestrator for bulk part import from Excel.

    Responsibilities:
    - Own session lifecycle
    - Own transaction boundaries
    - Validate workbook input
    - Normalize and clean rows
    - Insert new parts
    - Create part-image associations
    """

    SHEET_NAME = "EQUIP_BOMS"

    COLUMN_MAPPING = {
        "part_number": "ITEMNUM",
        "name": "DESCRIPTION",
        "oem_mfg": "OEMMFG",
        "model": "MODEL",
        "class_flag": "Class Flag",
        "ud6": "UD6",
        "type": "TYPE",
        "notes": "Notes",
        "documentation": "Specifications",
    }

    REQUIRED_COLUMNS = list(COLUMN_MAPPING.values())

    def __init__(
        self,
        shared_db_config=None,
        part_service: Optional[PartService] = None,
    ) -> None:
        self.db_config = shared_db_config or get_db_config()
        self.part_service = part_service or PartService()

    @with_request_id
    def import_parts_from_excel(
        self,
        *,
        file_path: Optional[str] = None,
        create_associations: bool = True,
        create_backup: bool = False,
    ) -> Dict[str, Any]:
        """
        Import parts from the configured Excel file.

        Args:
            file_path: Optional explicit path to workbook.
            create_associations: Whether to auto-link parts to images by title.
            create_backup: Reserved for future use. Present for interface stability.

        Returns:
            Dict[str, Any]: Standard orchestrator response payload.
        """
        session = self.db_config.get_main_session()

        try:
            resolved_file_path = file_path or os.path.join(
                DB_LOADSHEET,
                "load_MP2_ITEMS_BOMS.xlsx",
            )

            validation = self._validate_excel_file(file_path=resolved_file_path)
            if not validation["success"]:
                return validation

            df = pd.read_excel(resolved_file_path, sheet_name=self.SHEET_NAME)
            cleaned_df = self._clean_dataframe(df)

            if cleaned_df.empty:
                return {
                    "success": False,
                    "message": "No valid part rows found after cleaning.",
                    "status_code": 400,
                    "data": {
                        "file_path": resolved_file_path,
                        "rows_loaded": len(df),
                        "rows_cleaned": 0,
                        "new_parts": 0,
                        "duplicates_skipped": 0,
                        "associations_created": 0,
                    },
                }

            existing_part_numbers = self.part_service.get_existing_part_numbers(
                session=session
            )

            prepared = self._prepare_new_part_rows(
                cleaned_df=cleaned_df,
                existing_part_numbers=existing_part_numbers,
            )

            new_part_rows = prepared["new_part_rows"]
            duplicates_skipped = prepared["duplicates_skipped"]

            if not new_part_rows:
                session.rollback()
                return {
                    "success": True,
                    "message": "No new parts to import.",
                    "status_code": 200,
                    "data": {
                        "file_path": resolved_file_path,
                        "rows_loaded": len(df),
                        "rows_cleaned": len(cleaned_df),
                        "new_parts": 0,
                        "duplicates_skipped": duplicates_skipped,
                        "associations_created": 0,
                    },
                }

            self.part_service.create_parts_bulk(
                session=session,
                part_rows=new_part_rows,
            )

            inserted_part_numbers = [
                row["part_number"]
                for row in new_part_rows
                if row.get("part_number")
            ]

            inserted_parts = self.part_service.fetch_parts_by_numbers(
                session=session,
                part_numbers=inserted_part_numbers,
            )

            associations_created = 0
            if create_associations and inserted_parts:
                part_ids = [part.id for part in inserted_parts]
                associations_created = self._create_part_image_associations(
                    session=session,
                    part_ids=part_ids,
                )

            session.commit()

            return {
                "success": True,
                "message": "Parts imported successfully.",
                "status_code": 200,
                "data": {
                    "file_path": resolved_file_path,
                    "rows_loaded": len(df),
                    "rows_cleaned": len(cleaned_df),
                    "new_parts": len(inserted_parts),
                    "duplicates_skipped": duplicates_skipped,
                    "associations_created": associations_created,
                    "inserted_part_numbers": inserted_part_numbers,
                },
            }

        except ValueError as exc:
            session.rollback()
            logger.warning("Part import validation error: %s", exc, exc_info=True)
            return {
                "success": False,
                "message": str(exc),
                "status_code": 400,
            }

        except Exception as exc:
            session.rollback()
            logger.error(
                "Unexpected error importing parts: %s",
                exc,
                exc_info=True,
            )
            return {
                "success": False,
                "message": f"An error occurred during part import: {exc}",
                "status_code": 500,
            }

        finally:
            session.close()

    def _validate_excel_file(self, *, file_path: str) -> Dict[str, Any]:
        """
        Validate workbook existence, sheet presence, and required columns.
        """
        if not os.path.exists(file_path):
            return {
                "success": False,
                "message": f"Excel file not found: {file_path}",
                "status_code": 404,
            }

        try:
            excel_file = pd.ExcelFile(file_path)

            if self.SHEET_NAME not in excel_file.sheet_names:
                return {
                    "success": False,
                    "message": (
                        f"Required sheet '{self.SHEET_NAME}' not found. "
                        f"Available sheets: {', '.join(excel_file.sheet_names)}"
                    ),
                    "status_code": 400,
                }

            df_sample = pd.read_excel(
                file_path,
                sheet_name=self.SHEET_NAME,
                nrows=5,
            )

            missing_columns = [
                col for col in self.REQUIRED_COLUMNS if col not in df_sample.columns
            ]

            if missing_columns:
                return {
                    "success": False,
                    "message": f"Missing required columns: {missing_columns}",
                    "status_code": 400,
                }

            return {
                "success": True,
                "message": "Excel file is valid.",
                "status_code": 200,
            }

        except Exception as exc:
            return {
                "success": False,
                "message": f"Error validating Excel file: {exc}",
                "status_code": 400,
            }

    def _clean_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean the incoming sheet data and remove invalid rows.
        """
        df = df.copy()
        df = df.replace({np.nan: None, "": None})

        for excel_col in self.COLUMN_MAPPING.values():
            if excel_col in df.columns:
                df[excel_col] = df[excel_col].astype(str).str.strip()
                df[excel_col] = df[excel_col].replace(["nan", ""], None)

        for required_field in ("ITEMNUM", "DESCRIPTION"):
            if required_field in df.columns:
                df = df[
                    df[required_field].notna()
                    & (df[required_field] != "")
                    & (df[required_field] != "nan")
                ]

        if "ITEMNUM" in df.columns:
            df = df[df["ITEMNUM"].astype(str).str.len() >= 3]

        return df

    def _prepare_new_part_rows(
        self,
        *,
        cleaned_df: pd.DataFrame,
        existing_part_numbers: set[str],
    ) -> Dict[str, Any]:
        """
        Remove internal duplicates and rows already present in the database.
        """
        initial_count = len(cleaned_df)

        deduped_df = cleaned_df.drop_duplicates(subset=["ITEMNUM"], keep="last")
        internal_duplicates = initial_count - len(deduped_df)

        if existing_part_numbers:
            new_df = deduped_df[~deduped_df["ITEMNUM"].isin(existing_part_numbers)]
            database_duplicates = len(deduped_df) - len(new_df)
        else:
            new_df = deduped_df
            database_duplicates = 0

        new_df = new_df.copy()

        for db_col, excel_col in self.COLUMN_MAPPING.items():
            if excel_col not in new_df.columns:
                new_df[excel_col] = None

        new_part_rows = (
            new_df[list(self.COLUMN_MAPPING.values())]
            .rename(columns={excel: db for db, excel in self.COLUMN_MAPPING.items()})
            .to_dict("records")
        )

        return {
            "new_part_rows": new_part_rows,
            "duplicates_skipped": internal_duplicates + database_duplicates,
            "internal_duplicates": internal_duplicates,
            "database_duplicates": database_duplicates,
        }

    def _create_part_image_associations(
        self,
        *,
        session,
        part_ids: List[int],
    ) -> int:
        """
        Attempt to associate newly inserted parts with images by title.
        Association failures do not fail the entire import.
        """
        if not part_ids:
            return 0

        try:
            with RelationshipManager(session=session) as manager:
                result = manager.associate_parts_with_images_by_title(
                    part_ids=part_ids
                )
                total_associations = sum(len(assocs) for assocs in result.values())
                return total_associations
        except Exception as exc:
            logger.warning(
                "Part import succeeded but part-image association step failed: %s",
                exc,
                exc_info=True,
            )
            return 0