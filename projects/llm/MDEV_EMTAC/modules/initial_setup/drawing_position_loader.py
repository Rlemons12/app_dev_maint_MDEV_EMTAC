import logging
from pathlib import Path
from typing import Optional, Any

import pandas as pd
from sqlalchemy.exc import SQLAlchemyError

from modules.configuration.config_env import get_db_config
from modules.emtacdb.emtacdb_fts import Drawing, DrawingPositionAssociation, Position


logger = logging.getLogger("drawing_position_association_loader")


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


def to_int(value: Any) -> Optional[int]:
    value = clean(value)
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def find_existing_drawing(
    session,
    drawing_number: Optional[str],
    revision: Optional[str],
    drawing_name: Optional[str],
    equipment_name: Optional[str],
) -> Optional[Drawing]:
    """
    Find an already-loaded drawing using a conservative match order.
    """

    if drawing_number and revision:
        row = (
            session.query(Drawing)
            .filter(
                Drawing.drw_number == drawing_number,
                Drawing.drw_revision == revision,
            )
            .first()
        )
        if row:
            return row

    if drawing_number:
        row = (
            session.query(Drawing)
            .filter(Drawing.drw_number == drawing_number)
            .first()
        )
        if row:
            return row

    if drawing_name:
        row = (
            session.query(Drawing)
            .filter(Drawing.drw_name == drawing_name)
            .first()
        )
        if row:
            return row

    if equipment_name and drawing_name:
        row = (
            session.query(Drawing)
            .filter(
                Drawing.drw_equipment_name == equipment_name,
                Drawing.drw_name == drawing_name,
            )
            .first()
        )
        if row:
            return row

    return None


def associate_existing_drawings_to_positions(workbook_path: str):
    workbook_path = Path(workbook_path)
    if not workbook_path.exists():
        raise FileNotFoundError(f"Workbook not found: {workbook_path}")

    logger.info("Loading drawing sheet from: %s", workbook_path)

    drawing_df = pd.read_excel(workbook_path, sheet_name="drawing")

    db_config = get_db_config()
    session = db_config.get_main_session()

    stats = {
        "rows_processed": 0,
        "associations_created_or_found": 0,
        "rows_skipped": 0,
        "rows_failed": 0,
        "drawings_not_found": 0,
        "positions_not_found": 0,
    }

    try:
        for excel_row_num, row in enumerate(drawing_df.itertuples(index=False), start=2):
            try:
                position_id = to_int(getattr(row, "position_id", None))
                equipment_name = normalize_text(getattr(row, "equipment_name", None))
                drawing_number = normalize_text(getattr(row, "drawing_number", None))
                drawing_name = normalize_text(getattr(row, "drawing_name", None))
                revision = normalize_text(getattr(row, "revision", None))

                if not any([position_id, drawing_number, drawing_name, equipment_name, revision]):
                    stats["rows_skipped"] += 1
                    continue

                if position_id is None:
                    logger.warning(
                        "Skipping row %s because position_id is missing",
                        excel_row_num,
                    )
                    stats["rows_skipped"] += 1
                    continue

                position = session.query(Position).filter(Position.id == position_id).first()
                if not position:
                    logger.warning(
                        "Skipping row %s because Position id=%s was not found",
                        excel_row_num,
                        position_id,
                    )
                    stats["positions_not_found"] += 1
                    stats["rows_skipped"] += 1
                    continue

                drawing = find_existing_drawing(
                    session=session,
                    drawing_number=drawing_number,
                    revision=revision,
                    drawing_name=drawing_name,
                    equipment_name=equipment_name,
                )

                if not drawing:
                    logger.warning(
                        "Drawing not found for row %s | drawing_number=%s | revision=%s | drawing_name=%s | equipment_name=%s",
                        excel_row_num,
                        drawing_number,
                        revision,
                        drawing_name,
                        equipment_name,
                    )
                    stats["drawings_not_found"] += 1
                    stats["rows_skipped"] += 1
                    continue

                assoc = DrawingPositionAssociation.associate_drawing_position(
                    drawing_id=drawing.id,
                    position_id=position_id,
                    session=session,
                )

                if assoc:
                    stats["associations_created_or_found"] += 1

                stats["rows_processed"] += 1

                if stats["rows_processed"] % 250 == 0:
                    session.commit()
                    logger.info(
                        "Drawing-position progress: %s rows processed",
                        stats["rows_processed"],
                    )

            except Exception as exc:
                session.rollback()
                stats["rows_failed"] += 1
                logger.exception(
                    "Failed processing drawing-position row %s: %s",
                    excel_row_num,
                    exc,
                )

        session.commit()
        logger.info("Drawing-position association load complete: %s", stats)
        return stats

    except SQLAlchemyError:
        session.rollback()
        logger.exception("Drawing-position association load failed")
        raise
    finally:
        session.close()


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )

    workbook = r"E:\emtac\Database\DB_LOADSHEETS\position_load_template_with_drawing.xlsx"
    result = associate_existing_drawings_to_positions(workbook)
    print(result)