import logging
from pathlib import Path
from typing import Any, Optional

import pandas as pd
from sqlalchemy import text
from sqlalchemy.exc import SQLAlchemyError

from modules.configuration.config_env import get_db_config
from modules.emtacdb.emtacdb_fts import PartsPositionImageAssociation


logger = logging.getLogger("part_position_loader")

BATCH_SIZE = 10000


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
    value = str(value).strip()

    # Common Excel cleanup
    if value.endswith(".0"):
        value = value[:-2]

    return value


def to_int(value: Any) -> Optional[int]:
    value = clean(value)
    if value is None:
        return None
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return None


def load_part_map(session) -> dict[str, int]:
    logger.info("Loading part map from database...")
    rows = session.execute(
        text("SELECT id, part_number FROM part WHERE part_number IS NOT NULL")
    ).fetchall()

    part_map: dict[str, int] = {}
    for row in rows:
        part_number = str(row.part_number).strip()
        if part_number.endswith(".0"):
            part_number = part_number[:-2]
        part_map[part_number] = row.id

    logger.info("Loaded %s parts into memory", len(part_map))
    return part_map


def load_position_set(session) -> set[int]:
    logger.info("Loading position ids from database...")
    rows = session.execute(text("SELECT id FROM position")).fetchall()
    position_ids = {int(row.id) for row in rows}
    logger.info("Loaded %s positions into memory", len(position_ids))
    return position_ids


def load_existing_association_set(session) -> set[tuple[int, int]]:
    logger.info("Loading existing part-position associations with image_id IS NULL...")
    rows = session.execute(
        text("""
            SELECT part_id, position_id
            FROM part_position_image
            WHERE image_id IS NULL
        """)
    ).fetchall()

    assoc_set = {(int(row.part_id), int(row.position_id)) for row in rows}
    logger.info("Loaded %s existing part-position associations into memory", len(assoc_set))
    return assoc_set


def bulk_insert_associations(session, rows_to_insert: list[dict]) -> int:
    if not rows_to_insert:
        return 0

    session.execute(
        text("""
            INSERT INTO part_position_image (part_id, position_id, image_id)
            VALUES (:part_id, :position_id, :image_id)
        """),
        rows_to_insert,
    )
    return len(rows_to_insert)


def process_dataframe(
    df: pd.DataFrame,
    session,
    part_map: dict[str, int],
    valid_position_ids: set[int],
    existing_assoc_set: set[tuple[int, int]],
    stats: dict,
    sheet_name: str,
):
    pending_inserts: list[dict] = []

    logger.info("Processing sheet '%s' with %s rows", sheet_name, len(df))

    for excel_row_num, row in enumerate(df.itertuples(index=False), start=2):
        position_id = to_int(getattr(row, "position_id", None))
        part_number = normalize_text(getattr(row, "part_number", None))

        if position_id is None and not part_number:
            stats["rows_skipped"] += 1
            continue

        if position_id is None:
            stats["rows_skipped"] += 1
            continue

        if not part_number:
            stats["rows_skipped"] += 1
            continue

        if position_id not in valid_position_ids:
            stats["positions_not_found"] += 1
            stats["rows_skipped"] += 1
            continue

        part_id = part_map.get(part_number)
        if not part_id:
            stats["parts_not_found"] += 1
            stats["rows_skipped"] += 1
            continue

        assoc_key = (part_id, position_id)
        if assoc_key in existing_assoc_set:
            stats["associations_created_or_found"] += 1
            stats["rows_processed"] += 1
            continue

        pending_inserts.append(
            {
                "part_id": part_id,
                "position_id": position_id,
                "image_id": None,
            }
        )

        existing_assoc_set.add(assoc_key)
        stats["associations_created_or_found"] += 1
        stats["rows_processed"] += 1

        if len(pending_inserts) >= BATCH_SIZE:
            inserted = bulk_insert_associations(session, pending_inserts)
            session.commit()
            stats["rows_inserted"] += inserted
            logger.info(
                "Sheet=%s | rows_processed=%s | rows_inserted=%s | rows_skipped=%s | parts_not_found=%s | positions_not_found=%s",
                sheet_name,
                stats["rows_processed"],
                stats["rows_inserted"],
                stats["rows_skipped"],
                stats["parts_not_found"],
                stats["positions_not_found"],
            )
            pending_inserts.clear()

    if pending_inserts:
        inserted = bulk_insert_associations(session, pending_inserts)
        session.commit()
        stats["rows_inserted"] += inserted
        logger.info(
            "Final batch for sheet=%s | rows_processed=%s | rows_inserted=%s",
            sheet_name,
            stats["rows_processed"],
            stats["rows_inserted"],
        )


def load_part_position_associations(workbook_path: str):
    workbook_path = Path(workbook_path)
    if not workbook_path.exists():
        raise FileNotFoundError(f"Workbook not found: {workbook_path}")

    logger.info("Loading workbook: %s", workbook_path)

    logger.info("Reading bom_1 sheet...")
    bom_1_df = pd.read_excel(
        workbook_path,
        sheet_name="bom_1",
        usecols=["position_id", "part_number"],
    )
    logger.info("Finished bom_1 sheet. Rows=%s", len(bom_1_df))

    logger.info("Reading bom_2 sheet...")
    bom_2_df = pd.read_excel(
        workbook_path,
        sheet_name="bom_2",
        usecols=["position_id", "part_number"],
    )
    logger.info("Finished bom_2 sheet. Rows=%s", len(bom_2_df))

    db_config = get_db_config()
    session = db_config.get_main_session()

    stats = {
        "rows_processed": 0,
        "rows_skipped": 0,
        "rows_failed": 0,
        "parts_not_found": 0,
        "positions_not_found": 0,
        "associations_created_or_found": 0,
        "rows_inserted": 0,
    }

    try:
        part_map = load_part_map(session)
        valid_position_ids = load_position_set(session)
        existing_assoc_set = load_existing_association_set(session)

        process_dataframe(
            df=bom_1_df,
            session=session,
            part_map=part_map,
            valid_position_ids=valid_position_ids,
            existing_assoc_set=existing_assoc_set,
            stats=stats,
            sheet_name="bom_1",
        )

        process_dataframe(
            df=bom_2_df,
            session=session,
            part_map=part_map,
            valid_position_ids=valid_position_ids,
            existing_assoc_set=existing_assoc_set,
            stats=stats,
            sheet_name="bom_2",
        )

        logger.info("Part-position load complete: %s", stats)
        return stats

    except SQLAlchemyError:
        session.rollback()
        logger.exception("Part-position load failed")
        raise
    except Exception:
        session.rollback()
        logger.exception("Unexpected error during part-position load")
        raise
    finally:
        session.close()


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )

    workbook = r"E:\emtac\Database\DB_LOADSHEETS\position_load_template_with_drawing.xlsx"
    result = load_part_position_associations(workbook)
    print(result)