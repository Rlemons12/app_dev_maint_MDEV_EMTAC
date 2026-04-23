from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any, Optional

import pandas as pd
from sqlalchemy import text

from modules.configuration.config_env import get_db_config
from modules.emtacdb.emtacdb_fts import AssetNumber, Part, Position


logger = logging.getLogger("asset_part_bom_orchestrator")


class AssetPartBomAssociationOrchestrator:
    """
    Orchestrator to load asset/part rows from an Excel workbook and create
    part_position_image associations using asset-linked positions.

    Optimized for large workbooks:
    - preloads part map
    - preloads asset number map
    - preloads asset_number_id -> position_ids map
    - preloads existing part_position_image associations
    - uses bulk insert batches
    - avoids row-by-row service lookups

    Expected workbook structure:
    - workbook path: E:\\emtac\\Database\\DB_LOADSHEETS\\boms.xlsx
    - sheets: bom_1, bom_2
    - headers:
        asset_number
        asset_description
        part_number
        part_description
    """

    TASK_NAME = "associate_asset_parts"

    DEFAULT_WORKBOOK_PATH = r"E:\emtac\Database\DB_LOADSHEETS\boms.xlsx"
    DEFAULT_SHEETS = ("bom_1", "bom_2")
    REQUIRED_COLUMNS = (
        "asset_number",
        "asset_description",
        "part_number",
        "part_description",
    )
    DEFAULT_INSERT_BATCH_SIZE = 10000

    def __init__(
        self,
        db_config=None,
        workbook_path: str | Path | None = None,
        sheets: tuple[str, ...] | list[str] | None = None,
        log_progress_every: int = 1000,
        insert_batch_size: int = DEFAULT_INSERT_BATCH_SIZE,
    ) -> None:
        self.db_config = db_config or get_db_config()
        self.workbook_path = Path(workbook_path or self.DEFAULT_WORKBOOK_PATH)
        self.sheets = tuple(sheets or self.DEFAULT_SHEETS)
        self.log_progress_every = max(1, int(log_progress_every))
        self.insert_batch_size = max(1, int(insert_batch_size))

    # ---------------------------------------------------------------------
    # Timing / logging helpers
    # ---------------------------------------------------------------------
    @staticmethod
    def _elapsed(started: float) -> float:
        return round(time.time() - started, 4)

    def _log_timed_step_start(self, step_name: str) -> float:
        logger.info("START: %s", step_name)
        return time.time()

    def _log_timed_step_end(self, step_name: str, started: float, **extra: Any) -> None:
        duration = self._elapsed(started)
        if extra:
            logger.info("DONE: %s | duration_seconds=%s | %s", step_name, duration, extra)
        else:
            logger.info("DONE: %s | duration_seconds=%s", step_name, duration)

    # ---------------------------------------------------------------------
    # Normalization helpers
    # ---------------------------------------------------------------------
    @staticmethod
    def _clean(value: Any) -> Optional[Any]:
        if value is None:
            return None
        if pd.isna(value):
            return None
        if isinstance(value, str):
            value = value.strip()
            if value == "":
                return None
        return value

    @classmethod
    def _normalize_text(cls, value: Any) -> Optional[str]:
        value = cls._clean(value)
        if value is None:
            return None

        normalized = str(value).strip()

        if normalized.endswith(".0"):
            normalized = normalized[:-2]

        return normalized or None

    @classmethod
    def _normalize_header(cls, value: Any) -> str:
        return str(value).strip().lower()

    @classmethod
    def _validate_columns(cls, df: pd.DataFrame, sheet_name: str) -> None:
        actual = {cls._normalize_header(col) for col in df.columns}
        required = set(cls.REQUIRED_COLUMNS)
        missing = sorted(required - actual)
        if missing:
            raise ValueError(
                f"Sheet '{sheet_name}' is missing required columns: {missing}. "
                f"Found columns: {list(df.columns)}"
            )

    def _read_sheet(self, workbook_path: Path, sheet_name: str) -> pd.DataFrame:
        step_started = self._log_timed_step_start(f"read_sheet[{sheet_name}]")

        df = pd.read_excel(workbook_path, sheet_name=sheet_name)
        df.columns = [self._normalize_header(col) for col in df.columns]

        self._validate_columns(df, sheet_name)

        df = df[list(self.REQUIRED_COLUMNS)].copy()

        self._log_timed_step_end(
            f"read_sheet[{sheet_name}]",
            step_started,
            rows=len(df),
            columns=list(df.columns),
        )
        return df

    # ---------------------------------------------------------------------
    # Preload helpers
    # ---------------------------------------------------------------------
    def _load_part_map(self, session) -> dict[str, int]:
        step_started = self._log_timed_step_start("load_part_map")

        rows = session.query(Part.id, Part.part_number).order_by(Part.id).all()

        part_map: dict[str, int] = {}
        for part_id, part_number in rows:
            normalized = self._normalize_text(part_number)
            if normalized and normalized not in part_map:
                part_map[normalized] = int(part_id)

        self._log_timed_step_end(
            "load_part_map",
            step_started,
            db_rows=len(rows),
            unique_part_numbers=len(part_map),
        )
        return part_map

    def _load_asset_number_map(self, session) -> dict[str, list[int]]:
        step_started = self._log_timed_step_start("load_asset_number_map")

        rows = (
            session.query(AssetNumber.id, AssetNumber.number)
            .order_by(AssetNumber.id)
            .all()
        )

        asset_map: dict[str, list[int]] = {}
        for asset_number_id, asset_number in rows:
            normalized = self._normalize_text(asset_number)
            if not normalized:
                continue
            asset_map.setdefault(normalized, []).append(int(asset_number_id))

        distinct_asset_ids = sum(len(v) for v in asset_map.values())

        self._log_timed_step_end(
            "load_asset_number_map",
            step_started,
            db_rows=len(rows),
            unique_asset_numbers=len(asset_map),
            total_asset_ids=distinct_asset_ids,
        )
        return asset_map

    def _load_positions_by_asset_number_id(self, session) -> dict[int, list[int]]:
        step_started = self._log_timed_step_start("load_positions_by_asset_number_id")

        rows = (
            session.query(Position.id, Position.asset_number_id)
            .filter(Position.asset_number_id.isnot(None))
            .order_by(Position.asset_number_id, Position.id)
            .all()
        )

        positions_map: dict[int, list[int]] = {}
        for position_id, asset_number_id in rows:
            asset_id = int(asset_number_id)
            positions_map.setdefault(asset_id, []).append(int(position_id))

        total_position_links = sum(len(v) for v in positions_map.values())

        self._log_timed_step_end(
            "load_positions_by_asset_number_id",
            step_started,
            db_rows=len(rows),
            asset_ids_with_positions=len(positions_map),
            total_position_links=total_position_links,
        )
        return positions_map

    def _load_existing_assoc_set(self, session) -> set[tuple[int, int, Optional[int]]]:
        step_started = self._log_timed_step_start("load_existing_assoc_set")

        rows = session.execute(
            text(
                """
                SELECT part_id, position_id, image_id
                FROM part_position_image
                WHERE part_id IS NOT NULL
                  AND position_id IS NOT NULL
                """
            )
        ).fetchall()

        existing: set[tuple[int, int, Optional[int]]] = set()
        for row in rows:
            existing.add((int(row.part_id), int(row.position_id), row.image_id))

        self._log_timed_step_end(
            "load_existing_assoc_set",
            step_started,
            db_rows=len(rows),
            unique_existing_associations=len(existing),
        )
        return existing

    @staticmethod
    def _bulk_insert_associations(session, rows_to_insert: list[dict[str, Any]]) -> int:
        if not rows_to_insert:
            return 0

        session.execute(
            text(
                """
                INSERT INTO part_position_image (part_id, position_id, image_id)
                VALUES (:part_id, :position_id, :image_id)
                """
            ),
            rows_to_insert,
        )
        return len(rows_to_insert)

    # ---------------------------------------------------------------------
    # Row processing
    # ---------------------------------------------------------------------
    def _ensure_sheet_stats(self, stats: dict[str, Any], sheet_name: str) -> dict[str, Any]:
        sheet_stats = stats["sheet_stats"].setdefault(
            sheet_name,
            {
                "rows_seen": 0,
                "rows_skipped_blank": 0,
                "rows_skipped_missing_asset_number": 0,
                "rows_skipped_missing_part_number": 0,
                "assets_not_found": 0,
                "assets_with_no_positions": 0,
                "parts_not_found": 0,
                "existing_associations_found": 0,
                "associations_created": 0,
                "rows_with_association_created": 0,
                "rows_with_no_new_association": 0,
            },
        )
        return sheet_stats

    def _process_dataframe(
        self,
        *,
        session,
        df: pd.DataFrame,
        sheet_name: str,
        part_map: dict[str, int],
        asset_number_map: dict[str, list[int]],
        positions_by_asset_number_id: dict[int, list[int]],
        existing_assoc_set: set[tuple[int, int, Optional[int]]],
        stats: dict[str, Any],
        detail_rows: list[dict[str, Any]],
        assets_with_no_positions_rows: list[dict[str, Any]],
        no_position_asset_ids: set[int],
        missing_asset_numbers: set[str],
        missing_part_numbers: set[str],
        pending_inserts: list[dict[str, Any]],
    ) -> None:
        step_started = self._log_timed_step_start(f"process_dataframe[{sheet_name}]")
        logger.info("Processing sheet '%s' with %s rows", sheet_name, len(df))

        sheet_stats = self._ensure_sheet_stats(stats, sheet_name)

        for excel_row_num, row in enumerate(df.itertuples(index=False), start=2):
            stats["rows_seen"] += 1
            sheet_stats["rows_seen"] += 1

            asset_number = self._normalize_text(getattr(row, "asset_number", None))
            asset_description = self._normalize_text(
                getattr(row, "asset_description", None)
            )
            part_number = self._normalize_text(getattr(row, "part_number", None))
            part_description = self._normalize_text(
                getattr(row, "part_description", None)
            )

            if not asset_number and not part_number:
                stats["rows_skipped_blank"] += 1
                sheet_stats["rows_skipped_blank"] += 1
                continue

            if not asset_number:
                stats["rows_skipped_missing_asset_number"] += 1
                sheet_stats["rows_skipped_missing_asset_number"] += 1
                detail_rows.append(
                    {
                        "sheet_name": sheet_name,
                        "excel_row_num": excel_row_num,
                        "asset_number": "",
                        "asset_description": asset_description or "",
                        "part_number": part_number or "",
                        "part_description": part_description or "",
                        "status": "skipped",
                        "reason": "missing_asset_number",
                        "asset_number_id": None,
                        "position_id": None,
                        "part_id": None,
                        "association_id": None,
                    }
                )
                continue

            if not part_number:
                stats["rows_skipped_missing_part_number"] += 1
                sheet_stats["rows_skipped_missing_part_number"] += 1
                detail_rows.append(
                    {
                        "sheet_name": sheet_name,
                        "excel_row_num": excel_row_num,
                        "asset_number": asset_number,
                        "asset_description": asset_description or "",
                        "part_number": "",
                        "part_description": part_description or "",
                        "status": "skipped",
                        "reason": "missing_part_number",
                        "asset_number_id": None,
                        "position_id": None,
                        "part_id": None,
                        "association_id": None,
                    }
                )
                continue

            part_id = part_map.get(part_number)
            if part_id is None:
                stats["parts_not_found"] += 1
                sheet_stats["parts_not_found"] += 1
                missing_part_numbers.add(part_number)

                detail_rows.append(
                    {
                        "sheet_name": sheet_name,
                        "excel_row_num": excel_row_num,
                        "asset_number": asset_number,
                        "asset_description": asset_description or "",
                        "part_number": part_number,
                        "part_description": part_description or "",
                        "status": "skipped",
                        "reason": "part_not_found",
                        "asset_number_id": None,
                        "position_id": None,
                        "part_id": None,
                        "association_id": None,
                    }
                )
                continue

            asset_number_ids = asset_number_map.get(asset_number, [])
            if not asset_number_ids:
                stats["assets_not_found"] += 1
                sheet_stats["assets_not_found"] += 1
                missing_asset_numbers.add(asset_number)

                detail_rows.append(
                    {
                        "sheet_name": sheet_name,
                        "excel_row_num": excel_row_num,
                        "asset_number": asset_number,
                        "asset_description": asset_description or "",
                        "part_number": part_number,
                        "part_description": part_description or "",
                        "status": "skipped",
                        "reason": "asset_not_found",
                        "asset_number_id": None,
                        "position_id": None,
                        "part_id": part_id,
                        "association_id": None,
                    }
                )
                continue

            row_created_any = False

            for asset_number_id in asset_number_ids:
                position_ids = positions_by_asset_number_id.get(asset_number_id, [])

                if not position_ids:
                    stats["assets_with_no_positions"] += 1
                    sheet_stats["assets_with_no_positions"] += 1
                    no_position_asset_ids.add(asset_number_id)

                    no_position_row = {
                        "sheet_name": sheet_name,
                        "excel_row_num": excel_row_num,
                        "asset_number": asset_number,
                        "asset_description": asset_description or "",
                        "part_number": part_number,
                        "part_description": part_description or "",
                        "status": "skipped",
                        "reason": "no_positions_for_asset",
                        "asset_number_id": asset_number_id,
                        "position_id": None,
                        "part_id": part_id,
                        "association_id": None,
                    }

                    detail_rows.append(no_position_row)
                    assets_with_no_positions_rows.append(no_position_row.copy())
                    continue

                for position_id in position_ids:
                    assoc_key = (part_id, position_id, None)

                    if assoc_key in existing_assoc_set:
                        stats["existing_associations_found"] += 1
                        sheet_stats["existing_associations_found"] += 1
                        detail_rows.append(
                            {
                                "sheet_name": sheet_name,
                                "excel_row_num": excel_row_num,
                                "asset_number": asset_number,
                                "asset_description": asset_description or "",
                                "part_number": part_number,
                                "part_description": part_description or "",
                                "status": "existing",
                                "reason": "association_already_exists",
                                "asset_number_id": asset_number_id,
                                "position_id": position_id,
                                "part_id": part_id,
                                "association_id": None,
                            }
                        )
                        continue

                    pending_inserts.append(
                        {
                            "part_id": part_id,
                            "position_id": position_id,
                            "image_id": None,
                        }
                    )
                    existing_assoc_set.add(assoc_key)
                    stats["associations_created"] += 1
                    sheet_stats["associations_created"] += 1
                    row_created_any = True

                    detail_rows.append(
                        {
                            "sheet_name": sheet_name,
                            "excel_row_num": excel_row_num,
                            "asset_number": asset_number,
                            "asset_description": asset_description or "",
                            "part_number": part_number,
                            "part_description": part_description or "",
                            "status": "created",
                            "reason": "linked_part_to_asset_position",
                            "asset_number_id": asset_number_id,
                            "position_id": position_id,
                            "part_id": part_id,
                            "association_id": None,
                        }
                    )

                    if len(pending_inserts) >= self.insert_batch_size:
                        flush_started = self._log_timed_step_start(
                            f"bulk_insert_flush[{sheet_name}]"
                        )
                        inserted = self._bulk_insert_associations(session, pending_inserts)
                        session.flush()
                        stats["rows_inserted"] += inserted
                        logger.info(
                            "Flushed insert batch | sheet=%s | inserted=%s | pending_before_clear=%s",
                            sheet_name,
                            inserted,
                            len(pending_inserts),
                        )
                        pending_inserts.clear()
                        self._log_timed_step_end(
                            f"bulk_insert_flush[{sheet_name}]",
                            flush_started,
                            rows_inserted=inserted,
                            total_rows_inserted=stats["rows_inserted"],
                        )

            if row_created_any:
                stats["rows_with_association_created"] += 1
                sheet_stats["rows_with_association_created"] += 1
            else:
                stats["rows_with_no_new_association"] += 1
                sheet_stats["rows_with_no_new_association"] += 1

            if stats["rows_seen"] % self.log_progress_every == 0:
                logger.info(
                    (
                        "Progress | sheet=%s | rows_seen=%s | created=%s | "
                        "inserted=%s | existing=%s | assets_not_found=%s | "
                        "parts_not_found=%s | assets_with_no_positions=%s | "
                        "distinct_assets_with_no_positions=%s | "
                        "distinct_missing_assets=%s | distinct_missing_parts=%s | "
                        "pending_inserts=%s"
                    ),
                    sheet_name,
                    stats["rows_seen"],
                    stats["associations_created"],
                    stats["rows_inserted"],
                    stats["existing_associations_found"],
                    stats["assets_not_found"],
                    stats["parts_not_found"],
                    stats["assets_with_no_positions"],
                    len(no_position_asset_ids),
                    len(missing_asset_numbers),
                    len(missing_part_numbers),
                    len(pending_inserts),
                )

        self._log_timed_step_end(
            f"process_dataframe[{sheet_name}]",
            step_started,
            rows_seen=sheet_stats["rows_seen"],
            associations_created=sheet_stats["associations_created"],
            existing_associations_found=sheet_stats["existing_associations_found"],
            assets_not_found=sheet_stats["assets_not_found"],
            parts_not_found=sheet_stats["parts_not_found"],
            assets_with_no_positions=sheet_stats["assets_with_no_positions"],
        )

    # ---------------------------------------------------------------------
    # Public API
    # ---------------------------------------------------------------------
    def run(
        self,
        *,
        workbook_path: str | Path | None = None,
        sheets: tuple[str, ...] | list[str] | None = None,
        dry_run: bool = False,
    ) -> dict[str, Any]:
        workbook = Path(workbook_path) if workbook_path else self.workbook_path
        selected_sheets = tuple(sheets or self.sheets)

        if not workbook.exists():
            raise FileNotFoundError(f"Workbook not found: {workbook}")

        overall_started = time.time()
        detail_rows: list[dict[str, Any]] = []
        assets_with_no_positions_rows: list[dict[str, Any]] = []
        no_position_asset_ids: set[int] = set()
        missing_asset_numbers: set[str] = set()
        missing_part_numbers: set[str] = set()
        pending_inserts: list[dict[str, Any]] = []

        stats: dict[str, Any] = {
            "workbook_path": str(workbook),
            "sheets_processed": [],
            "sheet_stats": {},
            "rows_seen": 0,
            "rows_skipped_blank": 0,
            "rows_skipped_missing_asset_number": 0,
            "rows_skipped_missing_part_number": 0,
            "assets_not_found": 0,
            "distinct_assets_not_found": 0,
            "assets_with_no_positions": 0,
            "distinct_assets_with_no_positions": 0,
            "parts_not_found": 0,
            "distinct_parts_not_found": 0,
            "existing_associations_found": 0,
            "associations_created": 0,
            "rows_inserted": 0,
            "rows_with_association_created": 0,
            "rows_with_no_new_association": 0,
            "dry_run": dry_run,
            "insert_batch_size": self.insert_batch_size,
            "duration_seconds": 0.0,
        }

        logger.info("Starting asset-part BOM association load")
        logger.info("Task name: %s", self.TASK_NAME)
        logger.info("Workbook: %s", workbook)
        logger.info("Sheets: %s", list(selected_sheets))
        logger.info("Dry run: %s", dry_run)
        logger.info("Insert batch size: %s", self.insert_batch_size)
        logger.info("Log progress every: %s", self.log_progress_every)

        with self.db_config.main_session() as session:
            try:
                preload_started = self._log_timed_step_start("preload_phase")

                part_map = self._load_part_map(session)
                asset_number_map = self._load_asset_number_map(session)
                positions_by_asset_number_id = self._load_positions_by_asset_number_id(session)
                existing_assoc_set = self._load_existing_assoc_set(session)

                self._log_timed_step_end(
                    "preload_phase",
                    preload_started,
                    parts=len(part_map),
                    asset_numbers=len(asset_number_map),
                    asset_position_maps=len(positions_by_asset_number_id),
                    existing_associations=len(existing_assoc_set),
                )

                for sheet_name in selected_sheets:
                    df = self._read_sheet(workbook, sheet_name)
                    stats["sheets_processed"].append(sheet_name)

                    self._process_dataframe(
                        session=session,
                        df=df,
                        sheet_name=sheet_name,
                        part_map=part_map,
                        asset_number_map=asset_number_map,
                        positions_by_asset_number_id=positions_by_asset_number_id,
                        existing_assoc_set=existing_assoc_set,
                        stats=stats,
                        detail_rows=detail_rows,
                        assets_with_no_positions_rows=assets_with_no_positions_rows,
                        no_position_asset_ids=no_position_asset_ids,
                        missing_asset_numbers=missing_asset_numbers,
                        missing_part_numbers=missing_part_numbers,
                        pending_inserts=pending_inserts,
                    )

                if pending_inserts:
                    final_flush_started = self._log_timed_step_start("final_bulk_insert_flush")
                    inserted = self._bulk_insert_associations(session, pending_inserts)
                    session.flush()
                    stats["rows_inserted"] += inserted
                    logger.info(
                        "Final flush complete | inserted=%s | pending_before_clear=%s",
                        inserted,
                        len(pending_inserts),
                    )
                    pending_inserts.clear()
                    self._log_timed_step_end(
                        "final_bulk_insert_flush",
                        final_flush_started,
                        rows_inserted=inserted,
                        total_rows_inserted=stats["rows_inserted"],
                    )

                stats["distinct_assets_not_found"] = len(missing_asset_numbers)
                stats["distinct_assets_with_no_positions"] = len(no_position_asset_ids)
                stats["distinct_parts_not_found"] = len(missing_part_numbers)

                tx_started = self._log_timed_step_start(
                    "transaction_finalize[dry_run]" if dry_run else "transaction_finalize[commit]"
                )
                if dry_run:
                    session.rollback()
                    logger.info("Dry run enabled - rolled back all changes")
                else:
                    session.commit()
                    logger.info("Committed all association changes")
                self._log_timed_step_end(
                    "transaction_finalize[dry_run]" if dry_run else "transaction_finalize[commit]",
                    tx_started,
                )

                stats["duration_seconds"] = round(time.time() - overall_started, 4)

                result = {
                    "task_name": self.TASK_NAME,
                    "success": True,
                    "message": (
                        "Dry run completed successfully"
                        if dry_run
                        else "Asset-part BOM association load completed successfully"
                    ),
                    "summary": stats,
                    "errors": [],
                    "data": {
                        "detail_rows": detail_rows,
                        "assets_with_no_positions_rows": assets_with_no_positions_rows,
                        "no_position_asset_ids": sorted(no_position_asset_ids),
                        "missing_asset_numbers": sorted(missing_asset_numbers),
                        "missing_part_numbers": sorted(missing_part_numbers),
                    },
                }

                logger.info("Run complete | summary=%s", stats)
                return result

            except Exception as exc:
                session.rollback()
                stats["distinct_assets_not_found"] = len(missing_asset_numbers)
                stats["distinct_assets_with_no_positions"] = len(no_position_asset_ids)
                stats["distinct_parts_not_found"] = len(missing_part_numbers)
                stats["duration_seconds"] = round(time.time() - overall_started, 4)
                logger.exception("Asset-part BOM association load failed: %s", exc)

                return {
                    "task_name": self.TASK_NAME,
                    "success": False,
                    "message": "Asset-part BOM association load failed",
                    "summary": stats,
                    "errors": [str(exc)],
                    "data": {
                        "detail_rows": detail_rows,
                        "assets_with_no_positions_rows": assets_with_no_positions_rows,
                        "no_position_asset_ids": sorted(no_position_asset_ids),
                        "missing_asset_numbers": sorted(missing_asset_numbers),
                        "missing_part_numbers": sorted(missing_part_numbers),
                    },
                }


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )

    orchestrator = AssetPartBomAssociationOrchestrator(
        workbook_path=r"E:\emtac\Database\DB_LOADSHEETS\boms.xlsx",
        sheets=("bom_1", "bom_2"),
        log_progress_every=1000,
        insert_batch_size=10000,
    )

    result = orchestrator.run(dry_run=False)
    print(result["task_name"])
    print(result["success"])
    print(result["message"])
    print(result["summary"])