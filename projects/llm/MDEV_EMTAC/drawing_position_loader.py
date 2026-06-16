from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Optional

import pandas as pd
from sqlalchemy import text
from sqlalchemy.exc import SQLAlchemyError

from modules.configuration.config_env import get_db_config
from modules.emtacdb.emtacdb_fts import (
    AssetNumber,
    Drawing,
    DrawingPositionAssociation,
    Position,
)

logger = logging.getLogger("drawing_position_association_loader")


WORKBOOK_PATH = r"E:\emtac\Database\DB_LOADSHEETS\Active Drawing List.xlsx"
SHEET_NAME = "drawings_data"
TARGET_TABLE_NAME = "drawing_position"


def clean(value: Any) -> Optional[Any]:
    if pd.isna(value):
        return None

    if isinstance(value, str):
        value = value.strip()
        if value == "":
            return None

    return value


def normalize_text(value: Any) -> Optional[str]:
    value = clean(value)
    if value is None:
        return None
    return str(value).strip()


def normalize_excel_headers(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(col).strip().lower().replace(" ", "_") for col in df.columns]
    return df


def validate_required_columns(df: pd.DataFrame) -> None:
    required_columns = {
        "equipment_number",
        "drawing_number",
    }
    missing = required_columns - set(df.columns)

    if missing:
        raise ValueError(
            f"Missing required columns in sheet '{SHEET_NAME}': {sorted(missing)}"
        )


def build_target_asset_number(excel_equipment_number: Optional[str]) -> Optional[str]:
    """
    Business rule:
    Excel contains base equipment number like:
        AFL12700

    We only want to match:
        AFL12700-00
    """
    equipment_number = normalize_text(excel_equipment_number)
    if not equipment_number:
        return None

    base_number = equipment_number.split("-")[0].strip()
    if not base_number:
        return None

    return f"{base_number}-00"


def truncate_drawing_position_table(session) -> None:
    logger.warning("Truncating table '%s' before reload", TARGET_TABLE_NAME)
    session.execute(text(f"TRUNCATE TABLE {TARGET_TABLE_NAME} RESTART IDENTITY"))
    session.commit()


def build_asset_lookup(session) -> dict[str, dict[str, Optional[int]]]:
    """
    Build lookup by asset text value.

    Returns:
        {
            'AFL12700-00': {
                'asset_number_id': 123,
                'model_id': 45,
            },
            ...
        }
    """
    lookup: dict[str, dict[str, Optional[int]]] = {}

    asset_rows = session.query(AssetNumber).all()

    for row in asset_rows:
        candidate_values = []

        for attr_name in ("number", "asset_number", "name"):
            if hasattr(row, attr_name):
                candidate_values.append(getattr(row, attr_name))

        for value in candidate_values:
            norm = normalize_text(value)
            if norm:
                lookup[norm] = {
                    "asset_number_id": row.id,
                    "model_id": getattr(row, "model_id", None),
                }

    return lookup

def build_position_lookup_by_model_id(session) -> dict[int, list[int]]:
    """
    Build lookup:
        {
            model_id: [position_id_1, position_id_2, ...]
        }

    Only includes model-level positions:
    - model_id is not None
    - asset_number_id is None
    - location_id is None
    """
    lookup: dict[int, list[int]] = {}

    rows = (
        session.query(Position.id, Position.model_id)
        .filter(Position.model_id.isnot(None))
        .filter(Position.asset_number_id.is_(None))
        .filter(Position.location_id.is_(None))
        .all()
    )

    for position_id, model_id in rows:
        lookup.setdefault(model_id, []).append(position_id)

    return lookup


def build_unique_drawing_lookup_by_number(
    session,
) -> tuple[dict[str, int], dict[str, list[int]]]:
    grouped: dict[str, list[int]] = {}

    rows = session.query(Drawing.id, Drawing.drw_number).all()

    for drawing_id, drawing_number in rows:
        norm = normalize_text(drawing_number)
        if not norm:
            continue
        grouped.setdefault(norm, []).append(drawing_id)

    unique_lookup: dict[str, int] = {}
    duplicate_lookup: dict[str, list[int]] = {}

    for drawing_number, ids in grouped.items():
        if len(ids) == 1:
            unique_lookup[drawing_number] = ids[0]
        else:
            duplicate_lookup[drawing_number] = ids

    return unique_lookup, duplicate_lookup


def associate_existing_drawings_to_positions(
    workbook_path: str = WORKBOOK_PATH,
    truncate_first: bool = True,
) -> dict[str, Any]:
    """
    Rebuild drawing_position using:
    - workbook: Active Drawing List.xlsx
    - sheet: drawings_data
    - Excel EQUIPMENT NUMBER -> asset_number '-00' exact match only
    - target asset -> resolve model_id
    - model_id -> Position.model_id
    - only positions where Position.asset_number_id is not None
    - Excel DRAWING NUMBER -> Drawing.drw_number exact match only
    """
    workbook_path_obj = Path(workbook_path)
    if not workbook_path_obj.exists():
        raise FileNotFoundError(f"Workbook not found: {workbook_path_obj}")

    logger.info("Loading workbook: %s", workbook_path_obj)
    logger.info("Using sheet: %s", SHEET_NAME)

    drawing_df = pd.read_excel(workbook_path_obj, sheet_name=SHEET_NAME)
    drawing_df = normalize_excel_headers(drawing_df)
    validate_required_columns(drawing_df)

    db_config = get_db_config()
    session = db_config.get_main_session()

    stats = {
        "rows_seen": 0,
        "rows_skipped_blank": 0,
        "rows_failed": 0,
        "asset_numbers_not_found": 0,
        "assets_missing_model": 0,
        "positions_not_found_for_model": 0,
        "drawing_numbers_not_found": 0,
        "drawing_numbers_ambiguous": 0,
        "associations_created": 0,
    }

    try:
        if truncate_first:
            truncate_drawing_position_table(session)

        logger.info("Preloading asset lookup...")
        asset_lookup = build_asset_lookup(session)

        logger.info("Preloading position lookup by model id (asset-backed positions only)...")
        position_lookup = build_position_lookup_by_model_id(session)

        logger.info("Preloading drawing lookup by drawing number...")
        drawing_lookup, drawing_duplicates = build_unique_drawing_lookup_by_number(session)

        logger.info(
            "Preload summary | asset_lookup=%s | position_lookup=%s | unique_drawings=%s | duplicate_drawings=%s",
            len(asset_lookup),
            len(position_lookup),
            len(drawing_lookup),
            len(drawing_duplicates),
        )

        for excel_row_num, row in enumerate(drawing_df.itertuples(index=False), start=2):
            stats["rows_seen"] += 1

            try:
                raw_equipment_number = normalize_text(
                    getattr(row, "equipment_number", None)
                )
                drawing_number = normalize_text(getattr(row, "drawing_number", None))

                if not raw_equipment_number and not drawing_number:
                    stats["rows_skipped_blank"] += 1
                    continue

                if not raw_equipment_number:
                    logger.warning(
                        "Skipping row %s because EQUIPMENT NUMBER is blank",
                        excel_row_num,
                    )
                    stats["rows_failed"] += 1
                    continue

                if not drawing_number:
                    logger.warning(
                        "Skipping row %s because DRAWING NUMBER is blank",
                        excel_row_num,
                    )
                    stats["rows_failed"] += 1
                    continue

                target_asset_number = build_target_asset_number(raw_equipment_number)
                if not target_asset_number:
                    logger.warning(
                        "Skipping row %s because target asset number could not be built from EQUIPMENT NUMBER=%s",
                        excel_row_num,
                        raw_equipment_number,
                    )
                    stats["rows_failed"] += 1
                    continue

                asset_info = asset_lookup.get(target_asset_number)
                if asset_info is None:
                    logger.warning(
                        "Asset number not found | row=%s | excel_equipment_number=%s | expected_asset_number=%s",
                        excel_row_num,
                        raw_equipment_number,
                        target_asset_number,
                    )
                    stats["asset_numbers_not_found"] += 1
                    continue

                asset_number_id = asset_info.get("asset_number_id")
                model_id = asset_info.get("model_id")

                if model_id is None:
                    logger.warning(
                        "Asset missing model_id | row=%s | expected_asset_number=%s | asset_number_id=%s",
                        excel_row_num,
                        target_asset_number,
                        asset_number_id,
                    )
                    stats["assets_missing_model"] += 1
                    continue

                position_ids = position_lookup.get(model_id, [])
                if not position_ids:
                    logger.warning(
                        "No asset-backed positions found for model | row=%s | expected_asset_number=%s | asset_number_id=%s | model_id=%s",
                        excel_row_num,
                        target_asset_number,
                        asset_number_id,
                        model_id,
                    )
                    stats["positions_not_found_for_model"] += 1
                    continue

                if drawing_number in drawing_duplicates:
                    logger.warning(
                        "Ambiguous drawing number | row=%s | drawing_number=%s | matching_ids=%s",
                        excel_row_num,
                        drawing_number,
                        drawing_duplicates[drawing_number],
                    )
                    stats["drawing_numbers_ambiguous"] += 1
                    continue

                drawing_id = drawing_lookup.get(drawing_number)
                if drawing_id is None:
                    logger.warning(
                        "Drawing number not found | row=%s | drawing_number=%s",
                        excel_row_num,
                        drawing_number,
                    )
                    stats["drawing_numbers_not_found"] += 1
                    continue

                created_for_row = 0

                for position_id in position_ids:
                    DrawingPositionAssociation.associate_drawing_position(
                        drawing_id=drawing_id,
                        position_id=position_id,
                        session=session,
                    )
                    stats["associations_created"] += 1
                    created_for_row += 1

                logger.info(
                    "Row %s processed | excel_equipment_number=%s | expected_asset_number=%s | asset_number_id=%s | model_id=%s | drawing_number=%s | drawing_id=%s | asset_backed_positions_found=%s | associations_created=%s",
                    excel_row_num,
                    raw_equipment_number,
                    target_asset_number,
                    asset_number_id,
                    model_id,
                    drawing_number,
                    drawing_id,
                    len(position_ids),
                    created_for_row,
                )

                if stats["rows_seen"] % 250 == 0:
                    session.commit()
                    logger.info(
                        "Progress | rows_seen=%s | associations_created=%s",
                        stats["rows_seen"],
                        stats["associations_created"],
                    )

            except Exception as exc:
                session.rollback()
                stats["rows_failed"] += 1
                logger.exception(
                    "Failed processing row %s: %s",
                    excel_row_num,
                    exc,
                )

        session.commit()
        logger.info("Drawing-position rebuild complete: %s", stats)
        return stats

    except SQLAlchemyError:
        session.rollback()
        logger.exception("Drawing-position rebuild failed")
        raise
    finally:
        session.close()


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )

    result = associate_existing_drawings_to_positions(
        workbook_path=WORKBOOK_PATH,
        truncate_first=True,
    )
    print(result)