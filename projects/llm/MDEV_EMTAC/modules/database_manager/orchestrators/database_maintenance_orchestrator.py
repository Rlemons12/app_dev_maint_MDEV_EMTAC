from __future__ import annotations

import csv
import os
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

from sqlalchemy import func, or_

from modules.configuration.config_env import get_db_config
from modules.configuration.log_config import DatabaseMaintLogManager
from modules.database_manager.db_manager import DuplicateManager
from modules.database_manager.services.part_service import PartService
from modules.database_manager.services.drawing_service import DrawingService
from modules.database_manager.services.image_embedding_service import ImageEmbeddingService
from modules.database_manager.services.drawing_part_association_service import (
    DrawingPartAssociationService,
)
from modules.database_manager.services.part_position_image_association_service import (
    PartPositionImageAssociationService,
)
from modules.emtacdb.emtacdb_fts import (
    Drawing,
    DrawingPartAssociation,
    Image,
    ImageEmbedding,
    Part,
    PartsPositionImageAssociation,
)


class DatabaseMaintenanceOrchestrator:
    """
    Database maintenance orchestrator.

    Responsibilities:
    - owns the database session lifecycle
    - owns commit / rollback through db_config.main_session()
    - coordinates maintenance workflows
    - writes reports
    - logs through DatabaseMaintLogManager

    Public tasks:
    - associate_images
    - associate_drawings
    - validate_embeddings
    - find_duplicates
    - validate_data_integrity
    - run_all
    """

    def __init__(
        self,
        db_config=None,
        db_log_manager: DatabaseMaintLogManager | None = None,
        report_dir: str | os.PathLike | None = None,
        export_reports: bool = True,
        quick: bool = False,
        log_to_console: bool = False,
    ):
        self.db_config = db_config or get_db_config()
        self.export_reports = export_reports
        self.quick = quick

        self.report_dir = Path(report_dir) if report_dir else Path("db_maint_logs")
        self.report_dir.mkdir(parents=True, exist_ok=True)

        self.db_log_manager = db_log_manager or DatabaseMaintLogManager(
            run_dir=self.report_dir,
            run_name="database_maintenance",
            to_console=log_to_console,
        )
        self.logger = self.db_log_manager.logger

    # -------------------------------------------------------------------------
    # Helpers
    # -------------------------------------------------------------------------
    @staticmethod
    def _ts() -> str:
        return datetime.now().strftime("%Y%m%d_%H%M%S")

    @staticmethod
    def _safe_str(value: Any) -> str:
        if value is None:
            return ""
        return str(value).strip()

    @staticmethod
    def _normalize_part_number(value: Any) -> str:
        value = "" if value is None else str(value).strip()
        if value.endswith(".0"):
            value = value[:-2]
        return value

    @staticmethod
    def _split_spare_part_numbers(value: Any) -> list[str]:
        """
        Drawings may contain multiple comma-separated spare part numbers.
        Preserve order and remove duplicates.
        """
        if value is None:
            return []

        raw = str(value).strip()
        if not raw:
            return []

        tokens: list[str] = []
        for piece in raw.split(","):
            cleaned = piece.strip()
            if not cleaned:
                continue

            normalized = DatabaseMaintenanceOrchestrator._normalize_part_number(cleaned)
            if normalized:
                tokens.append(normalized)

        seen = set()
        ordered: list[str] = []
        for token in tokens:
            if token not in seen:
                seen.add(token)
                ordered.append(token)

        return ordered

    @staticmethod
    def _build_result(
        task_name: str,
        success: bool,
        summary: dict | None = None,
        report_files: list[str] | None = None,
        errors: list[str] | None = None,
        data: dict | None = None,
    ) -> dict:
        return {
            "task_name": task_name,
            "success": success,
            "summary": summary or {},
            "report_files": report_files or [],
            "errors": errors or [],
            "data": data or {},
        }

    def close(self) -> None:
        if self.db_log_manager:
            self.db_log_manager.close()

    def _write_csv(
        self,
        *,
        filename: str,
        rows: list[dict],
        fieldnames: list[str],
    ) -> str:
        path = self.report_dir / filename
        with path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for row in rows:
                writer.writerow(row)
        return str(path)

    def _write_txt(self, *, filename: str, content: str) -> str:
        path = self.report_dir / filename
        path.write_text(content, encoding="utf-8")
        return str(path)

    def _report_or_empty(
        self,
        writers: list[tuple[str, list[dict], list[str]]],
    ) -> list[str]:
        if not self.export_reports:
            return []

        report_files: list[str] = []
        for filename, rows, fieldnames in writers:
            report_files.append(
                self._write_csv(
                    filename=filename,
                    rows=rows,
                    fieldnames=fieldnames,
                )
            )
        return report_files

    def _load_lightweight_parts(self, session) -> list[dict]:
        """
        Load only the columns needed for associate_images.
        """
        rows = (
            session.query(Part.id, Part.part_number, Part.name)
            .order_by(Part.id)
            .all()
        )

        return [
            {
                "id": part_id,
                "part_number": part_number or "",
                "part_name": name or "",
                "normalized_part_number": self._normalize_part_number(part_number),
            }
            for part_id, part_number, name in rows
        ]

    def _load_lightweight_images(self, session) -> list[dict]:
        """
        Load only the columns needed for associate_images.
        """
        rows = (
            session.query(Image.id, Image.title, Image.file_path)
            .order_by(Image.id)
            .all()
        )

        return [
            {
                "image_id": image_id,
                "title": title or "",
                "title_lower": (title or "").lower(),
                "file_path": file_path or "",
            }
            for image_id, title, file_path in rows
        ]

    def _load_lightweight_parts_for_drawings(self, session) -> dict[str, dict]:
        """
        Load only the columns needed for associate_drawings and return
        a normalized lookup keyed by part_number.
        """
        rows = (
            session.query(Part.id, Part.part_number, Part.name)
            .order_by(Part.id)
            .all()
        )

        part_by_number: dict[str, dict] = {}
        for part_id, part_number, part_name in rows:
            normalized = self._normalize_part_number(part_number)
            if not normalized:
                continue

            # Preserve first seen part for parity with earlier behavior
            if normalized not in part_by_number:
                part_by_number[normalized] = {
                    "id": part_id,
                    "part_number": part_number or "",
                    "part_name": part_name or "",
                }

        return part_by_number

    def _load_lightweight_drawings(self, session) -> list[dict]:
        """
        Load only the columns needed for associate_drawings.
        """
        rows = (
            session.query(
                Drawing.id,
                Drawing.drw_number,
                Drawing.drw_name,
                Drawing.drw_spare_part_number,
            )
            .order_by(Drawing.id)
            .all()
        )

        return [
            {
                "id": drawing_id,
                "drawing_number": drw_number or "",
                "drawing_name": drw_name or "",
                "spare_part_number": drw_spare_part_number or "",
            }
            for drawing_id, drw_number, drw_name, drw_spare_part_number in rows
        ]

    def _build_image_title_index(self, images: list[dict]) -> dict[str, list[dict]]:
        """
        Build a coarse first-character index to reduce full scans.

        This preserves your current substring matching behavior while
        avoiding scanning all images for every part in many cases.
        """
        index: dict[str, list[dict]] = defaultdict(list)

        for image in images:
            title_lower = image["title_lower"]
            if not title_lower:
                continue

            first_char = title_lower[0]
            index[first_char].append(image)

            # Also index all digits seen in the title as a cheap narrowing key.
            seen_digits = {ch for ch in title_lower if ch.isdigit()}
            for digit in seen_digits:
                index[digit].append(image)

        return index

    def _candidate_images_for_part(
        self,
        normalized_part_number: str,
        image_index: dict[str, list[dict]],
        all_images: list[dict],
    ) -> list[dict]:
        """
        Return a narrowed candidate image list for a part number.
        Falls back to all images if no safe narrowing key exists.
        """
        if not normalized_part_number:
            return []

        keys: list[str] = []
        first_char = normalized_part_number[0].lower()
        keys.append(first_char)

        first_digit = next((ch for ch in normalized_part_number if ch.isdigit()), None)
        if first_digit and first_digit not in keys:
            keys.append(first_digit)

        candidates: list[dict] = []
        seen_ids: set[int] = set()

        for key in keys:
            for image in image_index.get(key, []):
                image_id = image["image_id"]
                if image_id not in seen_ids:
                    seen_ids.add(image_id)
                    candidates.append(image)

        return candidates if candidates else all_images

    def _propagate_existing_part_images_to_null_rows(
        self,
        *,
        session,
        progress_interval: int = 5000,
        log_progress: bool = True,
    ) -> tuple[int, list[dict], dict]:
        """
        For each existing (part_id, image_id) pair where image_id is not null,
        find all rows for that same part_id that currently have image_id NULL,
        and create the missing (part_id, position_id, image_id) association.

        Important:
        - NEVER overwrite an existing non-null image_id row
        - ONLY use null-image rows as propagation targets
        - position_id=None target rows are skipped because the base part/image
          association should already exist or be created separately

        Performance:
        - loads association rows once
        - builds in-memory lookup maps
        - avoids repeated DB existence queries for each candidate pair

        Returns:
            (
                created_count,
                propagated_detail_rows,
                propagation_stats,
            )
        """
        started = time.time()

        if progress_interval <= 0:
            progress_interval = 5000

        all_rows = (
            session.query(
                PartsPositionImageAssociation.id,
                PartsPositionImageAssociation.part_id,
                PartsPositionImageAssociation.position_id,
                PartsPositionImageAssociation.image_id,
            )
            .order_by(
                PartsPositionImageAssociation.part_id,
                PartsPositionImageAssociation.position_id,
                PartsPositionImageAssociation.id,
            )
            .all()
        )

        total_rows_scanned = len(all_rows)

        image_ids_by_part: dict[int, set[int]] = {}
        null_position_ids_by_part: dict[int, set[int]] = {}
        existing_triples: set[tuple[int, int, int]] = set()

        for _, part_id, position_id, image_id in all_rows:
            if part_id is None:
                continue

            if image_id is not None:
                image_ids_by_part.setdefault(part_id, set()).add(image_id)
                if position_id is not None:
                    existing_triples.add((part_id, position_id, image_id))
            else:
                if position_id is not None:
                    null_position_ids_by_part.setdefault(part_id, set()).add(position_id)

        candidate_part_ids = sorted(
            set(image_ids_by_part.keys()).intersection(null_position_ids_by_part.keys())
        )

        total_candidate_parts = len(candidate_part_ids)
        estimated_candidate_pairs = 0
        for part_id in candidate_part_ids:
            estimated_candidate_pairs += (
                len(image_ids_by_part.get(part_id, set()))
                * len(null_position_ids_by_part.get(part_id, set()))
            )

        created_count = 0
        propagated_rows: list[dict] = []
        scanned_candidate_pairs = 0

        if log_progress and not self.quick:
            self.logger.info(
                "Propagation scan starting: rows_scanned=%s candidate_parts=%s estimated_candidate_pairs=%s",
                total_rows_scanned,
                total_candidate_parts,
                estimated_candidate_pairs,
            )

        for part_index, part_id in enumerate(candidate_part_ids, start=1):
            image_ids = sorted(image_ids_by_part.get(part_id, set()))
            null_position_ids = sorted(null_position_ids_by_part.get(part_id, set()))

            for position_id in null_position_ids:
                for image_id in image_ids:
                    scanned_candidate_pairs += 1

                    triple = (part_id, position_id, image_id)
                    if triple in existing_triples:
                        continue

                    assoc_id = PartPositionImageAssociationService.add(
                        session,
                        part_id=part_id,
                        position_id=position_id,
                        image_id=image_id,
                    )

                    if assoc_id:
                        created_count += 1
                        existing_triples.add(triple)
                        propagated_rows.append(
                            {
                                "part_id": part_id,
                                "position_id": position_id,
                                "image_id": image_id,
                                "association_id": assoc_id,
                                "match_method": "propagated_from_existing_part_image_to_null_row",
                            }
                        )

                    if (
                        log_progress
                        and not self.quick
                        and scanned_candidate_pairs % progress_interval == 0
                    ):
                        pct = (
                            (scanned_candidate_pairs / estimated_candidate_pairs) * 100
                            if estimated_candidate_pairs > 0
                            else 100.0
                        )
                        self.logger.info(
                            "Propagation progress: scanned_pairs=%s/%s (%.2f%%) created=%s current_part_index=%s/%s",
                            scanned_candidate_pairs,
                            estimated_candidate_pairs,
                            pct,
                            created_count,
                            part_index,
                            total_candidate_parts,
                        )

        duration = time.time() - started

        propagation_stats = {
            "total_rows_scanned": total_rows_scanned,
            "candidate_parts": total_candidate_parts,
            "estimated_candidate_pairs": estimated_candidate_pairs,
            "scanned_candidate_pairs": scanned_candidate_pairs,
            "propagated_associations_created": created_count,
            "propagation_duration_seconds": round(duration, 4),
        }

        if log_progress and not self.quick:
            self.logger.info("Propagation complete: %s", propagation_stats)

        return created_count, propagated_rows, propagation_stats

    # -------------------------------------------------------------------------
    # Task 1: Associate parts with images
    # -------------------------------------------------------------------------
    def associate_images(
        self,
        batch_size: int = 1000,
        propagation_progress_interval: int = 5000,
        show_propagation_progress: bool = True,
    ) -> dict:
        """
        Associate parts with images based on part number appearing in image title.

        Behavior target:
        - optimized maintenance utility style
        - duplicate-safe association creation
        - after matching, propagate existing part/image pairs to rows for the same
          part where image_id is NULL
        - NEVER overwrite a row that already has a non-null image_id
        - batch progress logging
        - optimized report naming
        """
        task_name = "associate_images"
        timestamp = self._ts()
        started = time.time()

        if batch_size <= 0:
            batch_size = 1000
        if propagation_progress_interval <= 0:
            propagation_progress_interval = 5000

        with self.db_log_manager.timed_operation(task_name):
            try:
                with self.db_config.main_session() as session:
                    self.logger.info(
                        "Loading parts and images for image association workflow"
                    )

                    parts = self._load_lightweight_parts(session)
                    images = self._load_lightweight_images(session)
                    image_index = self._build_image_title_index(images)

                    self.logger.info(
                        "Loaded %s parts and %s images",
                        len(parts),
                        len(images),
                    )

                    matched_rows: list[dict] = []
                    unmatched_rows: list[dict] = []
                    detail_rows: list[dict] = []

                    total_parts_processed = 0
                    parts_with_matching_images = 0
                    parts_with_no_matches = 0
                    total_associations_created = 0

                    part_lookup = {part["id"]: part for part in parts}
                    image_lookup = {image["image_id"]: image for image in images}

                    total_parts = len(parts)

                    for idx, part in enumerate(parts, start=1):
                        total_parts_processed += 1

                        part_id = part["id"]
                        part_number = part["part_number"]
                        part_name = part["part_name"]
                        normalized_part_number = part["normalized_part_number"]

                        if not normalized_part_number:
                            parts_with_no_matches += 1
                            unmatched_rows.append(
                                {
                                    "part_id": part_id,
                                    "part_number": "",
                                    "part_name": part_name,
                                    "reason": "missing_part_number",
                                }
                            )
                            continue

                        token = normalized_part_number.lower()
                        part_match_count = 0

                        candidate_images = self._candidate_images_for_part(
                            normalized_part_number=token,
                            image_index=image_index,
                            all_images=images,
                        )

                        for image_row in candidate_images:
                            if token and token in image_row["title_lower"]:
                                before_existing = PartPositionImageAssociationService.get_association(
                                    session,
                                    part_id=part_id,
                                    position_id=None,
                                    image_id=image_row["image_id"],
                                )

                                assoc_id = PartPositionImageAssociationService.add(
                                    session,
                                    part_id=part_id,
                                    position_id=None,
                                    image_id=image_row["image_id"],
                                )

                                base_created = before_existing is None and assoc_id is not None
                                if base_created:
                                    part_match_count += 1
                                    total_associations_created += 1

                                detail_rows.append(
                                    {
                                        "part_id": part_id,
                                        "part_number": part_number,
                                        "part_name": part_name,
                                        "position_id": None,
                                        "image_id": image_row["image_id"],
                                        "image_title": image_row["title"],
                                        "image_file_path": image_row["file_path"],
                                        "association_id": assoc_id,
                                        "match_method": "title_contains_part_number",
                                    }
                                )

                        if part_match_count > 0:
                            parts_with_matching_images += 1
                            matched_rows.append(
                                {
                                    "part_id": part_id,
                                    "part_number": part_number,
                                    "part_name": part_name,
                                    "match_count": part_match_count,
                                }
                            )
                        else:
                            parts_with_no_matches += 1
                            unmatched_rows.append(
                                {
                                    "part_id": part_id,
                                    "part_number": part_number,
                                    "part_name": part_name,
                                    "reason": "no_image_title_match",
                                }
                            )

                        if not self.quick and idx % batch_size == 0:
                            pct = (idx / total_parts * 100) if total_parts > 0 else 100.0
                            self.logger.info(
                                "associate_images progress: processed=%s/%s (%.2f%%) matched=%s unmatched=%s created=%s",
                                idx,
                                total_parts,
                                pct,
                                parts_with_matching_images,
                                parts_with_no_matches,
                                total_associations_created,
                            )

                    propagated_count, propagation_rows, propagation_stats = (
                        self._propagate_existing_part_images_to_null_rows(
                            session=session,
                            progress_interval=propagation_progress_interval,
                            log_progress=show_propagation_progress,
                        )
                    )

                    if propagated_count > 0:
                        total_associations_created += propagated_count

                        for row in propagation_rows:
                            part_obj = part_lookup.get(row["part_id"])
                            image_obj = image_lookup.get(row["image_id"])

                            row["part_number"] = part_obj["part_number"] if part_obj else ""
                            row["part_name"] = part_obj["part_name"] if part_obj else ""
                            row["image_title"] = image_obj["title"] if image_obj else ""
                            row["image_file_path"] = image_obj["file_path"] if image_obj else ""

                    duration = time.time() - started
                    all_detail_rows = detail_rows + propagation_rows

                    summary = {
                        "total_parts_processed": total_parts_processed,
                        "parts_with_matching_images": parts_with_matching_images,
                        "total_associations_created": total_associations_created,
                        "propagated_associations_created": propagated_count,
                        "parts_with_no_matches": parts_with_no_matches,
                        "total_rows_scanned_for_propagation": propagation_stats.get("total_rows_scanned", 0),
                        "candidate_parts_for_propagation": propagation_stats.get("candidate_parts", 0),
                        "estimated_candidate_pairs_for_propagation": propagation_stats.get("estimated_candidate_pairs", 0),
                        "scanned_candidate_pairs_for_propagation": propagation_stats.get("scanned_candidate_pairs", 0),
                        "propagation_duration_seconds": propagation_stats.get("propagation_duration_seconds", 0.0),
                        "duration_seconds": round(duration, 4),
                    }

                    report_files: list[str] = []
                    if self.export_reports:
                        report_files.extend(
                            self._report_or_empty(
                                [
                                    (
                                        f"optimized_part_image_summary_{timestamp}.csv",
                                        [summary],
                                        list(summary.keys()),
                                    ),
                                    (
                                        f"optimized_part_image_details_{timestamp}.csv",
                                        all_detail_rows,
                                        [
                                            "part_id",
                                            "part_number",
                                            "part_name",
                                            "position_id",
                                            "image_id",
                                            "image_title",
                                            "image_file_path",
                                            "association_id",
                                            "match_method",
                                        ],
                                    ),
                                    (
                                        f"optimized_unmatched_parts_{timestamp}.csv",
                                        unmatched_rows,
                                        [
                                            "part_id",
                                            "part_number",
                                            "part_name",
                                            "reason",
                                        ],
                                    ),
                                ]
                            )
                        )

                    self.logger.info("associate_images summary=%s", summary)

                    return self._build_result(
                        task_name=task_name,
                        success=True,
                        summary=summary,
                        report_files=report_files,
                        data={
                            "matched_rows": matched_rows,
                            "detail_rows": all_detail_rows,
                            "base_detail_rows": detail_rows,
                            "propagation_rows": propagation_rows,
                            "propagation_stats": propagation_stats,
                            "unmatched_rows": unmatched_rows,
                        },
                    )

            except Exception as exc:
                self.logger.exception("associate_images failed: %s", exc)
                return self._build_result(
                    task_name=task_name,
                    success=False,
                    errors=[str(exc)],
                )

    # -------------------------------------------------------------------------
    # Task 2: Associate drawings with parts
    # -------------------------------------------------------------------------
    def associate_drawings(self) -> dict:
        """
        Associate drawings with parts based on drawing spare part numbers.

        Optimized behavior:
        - lightweight drawing load
        - lightweight part lookup by normalized part_number
        - multiple comma-separated spare part numbers supported
        - duplicate-safe add behavior
        - optimized report naming
        """
        task_name = "associate_drawings"
        timestamp = self._ts()
        started = time.time()

        with self.db_log_manager.timed_operation(task_name):
            try:
                with self.db_config.main_session() as session:
                    self.logger.info(
                        "Loading lightweight drawings and part lookup for drawing-part association workflow"
                    )

                    drawings = self._load_lightweight_drawings(session)
                    part_by_number = self._load_lightweight_parts_for_drawings(session)

                    self.logger.info(
                        "Loaded %s drawings and %s normalized parts",
                        len(drawings),
                        len(part_by_number),
                    )

                    matched_rows: list[dict] = []
                    unmatched_rows: list[dict] = []

                    total_drawings_processed = 0
                    drawings_with_matching_parts = 0
                    drawings_with_no_matches = 0
                    drawings_without_spare_part_number = 0
                    multiple_part_numbers_count = 0
                    total_associations_created = 0

                    total_drawings = len(drawings)

                    for idx, drawing in enumerate(drawings, start=1):
                        total_drawings_processed += 1

                        drawing_id = drawing["id"]
                        drawing_number = drawing["drawing_number"]
                        drawing_name = drawing["drawing_name"]
                        raw_spare = drawing["spare_part_number"]

                        parsed_part_numbers = self._split_spare_part_numbers(raw_spare)

                        if not parsed_part_numbers:
                            drawings_without_spare_part_number += 1
                            drawings_with_no_matches += 1

                            unmatched_rows.append(
                                {
                                    "drawing_id": drawing_id,
                                    "drawing_number": drawing_number,
                                    "drawing_name": drawing_name,
                                    "spare_part_number": raw_spare,
                                    "parsed_part_numbers": "",
                                    "reason": "missing_spare_part_number",
                                }
                            )
                            continue

                        if len(parsed_part_numbers) > 1:
                            multiple_part_numbers_count += 1

                        drawing_match_count = 0

                        for parsed_number in parsed_part_numbers:
                            part = part_by_number.get(parsed_number)
                            if not part:
                                continue

                            assoc_id = DrawingPartAssociationService.add(
                                session,
                                drawing_id=drawing_id,
                                part_id=part["id"],
                            )

                            if assoc_id:
                                drawing_match_count += 1
                                total_associations_created += 1

                                matched_rows.append(
                                    {
                                        "drawing_id": drawing_id,
                                        "drawing_number": drawing_number,
                                        "drawing_name": drawing_name,
                                        "drawing_spare_part_numbers": raw_spare,
                                        "matched_part_number": part["part_number"],
                                        "part_id": part["id"],
                                        "part_name": part["part_name"],
                                        "association_id": assoc_id,
                                    }
                                )

                        if drawing_match_count > 0:
                            drawings_with_matching_parts += 1
                        else:
                            drawings_with_no_matches += 1
                            unmatched_rows.append(
                                {
                                    "drawing_id": drawing_id,
                                    "drawing_number": drawing_number,
                                    "drawing_name": drawing_name,
                                    "spare_part_number": raw_spare,
                                    "parsed_part_numbers": ", ".join(parsed_part_numbers),
                                    "reason": "no_matching_part",
                                }
                            )

                        if not self.quick and idx % 1000 == 0:
                            pct = (idx / total_drawings * 100) if total_drawings > 0 else 100.0
                            self.logger.info(
                                "associate_drawings progress: processed=%s/%s (%.2f%%) matched=%s unmatched=%s created=%s",
                                idx,
                                total_drawings,
                                pct,
                                drawings_with_matching_parts,
                                drawings_with_no_matches,
                                total_associations_created,
                            )

                    duration = time.time() - started

                    summary = {
                        "total_drawings_processed": total_drawings_processed,
                        "drawings_with_matching_parts": drawings_with_matching_parts,
                        "drawings_with_no_matches": drawings_with_no_matches,
                        "drawings_without_spare_part_number": drawings_without_spare_part_number,
                        "multiple_part_numbers_count": multiple_part_numbers_count,
                        "total_associations_created": total_associations_created,
                        "duration_seconds": round(duration, 4),
                    }

                    report_files: list[str] = []
                    if self.export_reports:
                        summary_text = "\n".join(f"{k}: {v}" for k, v in summary.items())
                        report_files.append(
                            self._write_txt(
                                filename=f"optimized_drawing_part_summary_{timestamp}.txt",
                                content=summary_text,
                            )
                        )
                        report_files.extend(
                            self._report_or_empty(
                                [
                                    (
                                        f"optimized_drawing_part_matches_{timestamp}.csv",
                                        matched_rows,
                                        [
                                            "drawing_id",
                                            "drawing_number",
                                            "drawing_name",
                                            "drawing_spare_part_numbers",
                                            "matched_part_number",
                                            "part_id",
                                            "part_name",
                                            "association_id",
                                        ],
                                    ),
                                    (
                                        f"optimized_drawing_part_unmatched_{timestamp}.csv",
                                        unmatched_rows,
                                        [
                                            "drawing_id",
                                            "drawing_number",
                                            "drawing_name",
                                            "spare_part_number",
                                            "parsed_part_numbers",
                                            "reason",
                                        ],
                                    ),
                                ]
                            )
                        )

                    self.logger.info("associate_drawings summary=%s", summary)

                    return self._build_result(
                        task_name=task_name,
                        success=True,
                        summary=summary,
                        report_files=report_files,
                        data={
                            "matched_rows": matched_rows,
                            "unmatched_rows": unmatched_rows,
                        },
                    )

            except Exception as exc:
                self.logger.exception("associate_drawings failed: %s", exc)
                return self._build_result(
                    task_name=task_name,
                    success=False,
                    errors=[str(exc)],
                )

    # -------------------------------------------------------------------------
    # Task 3: Validate image embeddings
    # -------------------------------------------------------------------------
    def validate_embeddings(self) -> dict:
        """
        Validate:
        - images missing embeddings
        - orphan embeddings with missing images
        """
        task_name = "validate_embeddings"
        timestamp = self._ts()
        started = time.time()

        with self.db_log_manager.timed_operation(task_name):
            try:
                with self.db_config.main_session() as session:
                    self.logger.info("Running embedding validation workflow")

                    all_images = session.query(Image).order_by(Image.id).all()
                    all_embeddings = (
                        session.query(ImageEmbedding).order_by(ImageEmbedding.id).all()
                    )

                    image_ids = {img.id for img in all_images}
                    embedded_image_ids = {
                        emb.image_id for emb in all_embeddings if emb.image_id is not None
                    }

                    images_missing_embeddings = [
                        {
                            "image_id": img.id,
                            "image_title": img.title or "",
                            "file_path": img.file_path or "",
                        }
                        for img in all_images
                        if img.id not in embedded_image_ids
                    ]

                    orphan_embeddings = [
                        {
                            "embedding_id": emb.id,
                            "image_id": emb.image_id,
                            "model_name": emb.model_name or "",
                            "storage_type": (
                                emb.get_storage_type()
                                if hasattr(emb, "get_storage_type")
                                else ""
                            ),
                        }
                        for emb in all_embeddings
                        if emb.image_id not in image_ids
                    ]

                    stats = ImageEmbeddingService.get_statistics(session)
                    duration = time.time() - started

                    summary = {
                        "total_images": len(all_images),
                        "total_embeddings": len(all_embeddings),
                        "images_missing_embeddings": len(images_missing_embeddings),
                        "orphan_embeddings": len(orphan_embeddings),
                        "pgvector_embeddings": stats.get("pgvector_embeddings", 0),
                        "legacy_embeddings": stats.get("legacy_embeddings", 0),
                        "both_formats": stats.get("both_formats", 0),
                        "duration_seconds": round(duration, 4),
                    }

                    report_files: list[str] = []
                    if self.export_reports:
                        report_files.extend(
                            self._report_or_empty(
                                [
                                    (
                                        f"image_embedding_summary_{timestamp}.csv",
                                        [summary],
                                        list(summary.keys()),
                                    ),
                                    (
                                        f"images_missing_embeddings_{timestamp}.csv",
                                        images_missing_embeddings,
                                        ["image_id", "image_title", "file_path"],
                                    ),
                                    (
                                        f"orphan_embeddings_{timestamp}.csv",
                                        orphan_embeddings,
                                        [
                                            "embedding_id",
                                            "image_id",
                                            "model_name",
                                            "storage_type",
                                        ],
                                    ),
                                ]
                            )
                        )

                    self.logger.info("validate_embeddings summary=%s", summary)

                    return self._build_result(
                        task_name=task_name,
                        success=True,
                        summary=summary,
                        report_files=report_files,
                        data={
                            "images_missing_embeddings": images_missing_embeddings,
                            "orphan_embeddings": orphan_embeddings,
                        },
                    )

            except Exception as exc:
                self.logger.exception("validate_embeddings failed: %s", exc)
                return self._build_result(
                    task_name=task_name,
                    success=False,
                    errors=[str(exc)],
                )

    # -------------------------------------------------------------------------
    # Task 4: Find duplicate parts
    # -------------------------------------------------------------------------
    def find_duplicates(self, threshold: float = 0.9) -> dict:
        """
        Find duplicate parts using DuplicateManager for closer compatibility
        with older behavior.
        """
        task_name = "find_duplicates"
        timestamp = self._ts()
        started = time.time()

        with self.db_log_manager.timed_operation(task_name):
            try:
                with self.db_config.main_session() as session:
                    self.logger.info(
                        "Running duplicate detection workflow with threshold=%s",
                        threshold,
                    )

                    with DuplicateManager(session=session, request_id=None) as manager:
                        duplicates = manager.find_duplicate_parts(threshold=threshold)

                    duration = time.time() - started

                    duplicate_rows = [
                        {
                            "source_id": source_id,
                            "target_id": target_id,
                            "similarity": similarity,
                        }
                        for source_id, target_id, similarity in duplicates
                    ]

                    summary = {
                        "duplicate_pairs_found": len(duplicate_rows),
                        "threshold_used": threshold,
                        "duration_seconds": round(duration, 4),
                    }

                    report_files: list[str] = []
                    if self.export_reports:
                        report_files.extend(
                            self._report_or_empty(
                                [
                                    (
                                        f"duplicate_parts_{timestamp}.csv",
                                        duplicate_rows,
                                        ["source_id", "target_id", "similarity"],
                                    ),
                                ]
                            )
                        )

                    self.logger.info("find_duplicates summary=%s", summary)

                    return self._build_result(
                        task_name=task_name,
                        success=True,
                        summary=summary,
                        report_files=report_files,
                        data={"duplicates": duplicate_rows},
                    )

            except Exception as exc:
                self.logger.exception("find_duplicates failed: %s", exc)
                return self._build_result(
                    task_name=task_name,
                    success=False,
                    errors=[str(exc)],
                )

    # -------------------------------------------------------------------------
    # Task 5: Validate data integrity
    # -------------------------------------------------------------------------
    def validate_data_integrity(self) -> dict:
        """
        Run lightweight integrity checks.

        This task remains available for the new pipeline even though the main
        user guide focuses on the optimized core tasks.
        """
        task_name = "validate_data_integrity"
        timestamp = self._ts()
        started = time.time()

        with self.db_log_manager.timed_operation(task_name):
            try:
                with self.db_config.main_session() as session:
                    self.logger.info("Running data integrity validation workflow")

                    images_missing_title = session.query(Image).filter(
                        or_(Image.title.is_(None), func.trim(Image.title) == "")
                    ).all()

                    parts_missing_number = session.query(Part).filter(
                        or_(Part.part_number.is_(None), func.trim(Part.part_number) == "")
                    ).all()

                    drawings_missing_spare = session.query(Drawing).filter(
                        or_(
                            Drawing.drw_spare_part_number.is_(None),
                            func.trim(Drawing.drw_spare_part_number) == "",
                        )
                    ).all()

                    broken_part_image_associations = session.query(
                        PartsPositionImageAssociation
                    ).filter(
                        or_(
                            PartsPositionImageAssociation.part_id.is_(None),
                            PartsPositionImageAssociation.image_id.is_(None),
                        )
                    ).all()

                    broken_drawing_part_associations = session.query(
                        DrawingPartAssociation
                    ).filter(
                        or_(
                            DrawingPartAssociation.drawing_id.is_(None),
                            DrawingPartAssociation.part_id.is_(None),
                        )
                    ).all()

                    image_rows = [
                        {
                            "image_id": row.id,
                            "image_title": row.title or "",
                            "file_path": row.file_path or "",
                            "reason": "missing_title",
                        }
                        for row in images_missing_title
                    ]

                    part_rows = [
                        {
                            "part_id": row.id,
                            "part_number": row.part_number or "",
                            "part_name": row.name or "",
                            "reason": "missing_part_number",
                        }
                        for row in parts_missing_number
                    ]

                    drawing_rows = [
                        {
                            "drawing_id": row.id,
                            "drawing_number": row.drw_number or "",
                            "drawing_name": row.drw_name or "",
                            "spare_part_number": row.drw_spare_part_number or "",
                            "reason": "missing_spare_part_number",
                        }
                        for row in drawings_missing_spare
                    ]

                    part_image_assoc_rows = [
                        {
                            "association_id": row.id,
                            "part_id": row.part_id,
                            "position_id": row.position_id,
                            "image_id": row.image_id,
                            "reason": "missing_required_foreign_key",
                        }
                        for row in broken_part_image_associations
                    ]

                    drawing_part_assoc_rows = [
                        {
                            "association_id": row.id,
                            "drawing_id": row.drawing_id,
                            "part_id": row.part_id,
                            "reason": "missing_required_foreign_key",
                        }
                        for row in broken_drawing_part_associations
                    ]

                    total_issues = (
                        len(image_rows)
                        + len(part_rows)
                        + len(drawing_rows)
                        + len(part_image_assoc_rows)
                        + len(drawing_part_assoc_rows)
                    )

                    duration = time.time() - started

                    summary = {
                        "images_missing_title": len(image_rows),
                        "parts_missing_part_number": len(part_rows),
                        "drawings_missing_spare_part_number": len(drawing_rows),
                        "broken_part_image_associations": len(part_image_assoc_rows),
                        "broken_drawing_part_associations": len(drawing_part_assoc_rows),
                        "total_integrity_issues": total_issues,
                        "duration_seconds": round(duration, 4),
                    }

                    report_files: list[str] = []
                    if self.export_reports:
                        report_files.extend(
                            self._report_or_empty(
                                [
                                    (
                                        f"data_integrity_summary_{timestamp}.csv",
                                        [summary],
                                        list(summary.keys()),
                                    ),
                                    (
                                        f"integrity_images_missing_title_{timestamp}.csv",
                                        image_rows,
                                        [
                                            "image_id",
                                            "image_title",
                                            "file_path",
                                            "reason",
                                        ],
                                    ),
                                    (
                                        f"integrity_parts_missing_number_{timestamp}.csv",
                                        part_rows,
                                        [
                                            "part_id",
                                            "part_number",
                                            "part_name",
                                            "reason",
                                        ],
                                    ),
                                    (
                                        f"integrity_drawings_missing_spare_{timestamp}.csv",
                                        drawing_rows,
                                        [
                                            "drawing_id",
                                            "drawing_number",
                                            "drawing_name",
                                            "spare_part_number",
                                            "reason",
                                        ],
                                    ),
                                    (
                                        f"integrity_part_image_associations_{timestamp}.csv",
                                        part_image_assoc_rows,
                                        [
                                            "association_id",
                                            "part_id",
                                            "position_id",
                                            "image_id",
                                            "reason",
                                        ],
                                    ),
                                    (
                                        f"integrity_drawing_part_associations_{timestamp}.csv",
                                        drawing_part_assoc_rows,
                                        [
                                            "association_id",
                                            "drawing_id",
                                            "part_id",
                                            "reason",
                                        ],
                                    ),
                                ]
                            )
                        )

                    self.logger.info("validate_data_integrity summary=%s", summary)

                    return self._build_result(
                        task_name=task_name,
                        success=True,
                        summary=summary,
                        report_files=report_files,
                        data={
                            "image_rows": image_rows,
                            "part_rows": part_rows,
                            "drawing_rows": drawing_rows,
                            "part_image_assoc_rows": part_image_assoc_rows,
                            "drawing_part_assoc_rows": drawing_part_assoc_rows,
                        },
                    )

            except Exception as exc:
                self.logger.exception("validate_data_integrity failed: %s", exc)
                return self._build_result(
                    task_name=task_name,
                    success=False,
                    errors=[str(exc)],
                )

    # -------------------------------------------------------------------------
    # Task 6: Run all
    # -------------------------------------------------------------------------
    def run_all(
        self,
        *,
        include_embedding_validation: bool = False,
        include_duplicate_check: bool = False,
        duplicate_threshold: float = 0.9,
        include_integrity_validation: bool = False,
        batch_size: int = 1000,
        propagation_progress_interval: int = 5000,
        show_propagation_progress: bool = True,
    ) -> dict:
        """
        Run all maintenance tasks in sequence.

        Base optimized workflow:
        - part-image association
        - drawing-part association

        Optional:
        - embedding validation
        - duplicate detection
        - integrity validation
        """
        task_name = "run_all"

        with self.db_log_manager.timed_operation(task_name):
            results: list[dict] = []
            errors: list[str] = []

            image_result = self.associate_images(
                batch_size=batch_size,
                propagation_progress_interval=propagation_progress_interval,
                show_propagation_progress=show_propagation_progress,
            )
            results.append(image_result)
            if not image_result["success"]:
                errors.extend(image_result.get("errors", []))

            drawing_result = self.associate_drawings()
            results.append(drawing_result)
            if not drawing_result["success"]:
                errors.extend(drawing_result.get("errors", []))

            if include_embedding_validation:
                embedding_result = self.validate_embeddings()
                results.append(embedding_result)
                if not embedding_result["success"]:
                    errors.extend(embedding_result.get("errors", []))

            if include_duplicate_check:
                duplicate_result = self.find_duplicates(
                    threshold=duplicate_threshold
                )
                results.append(duplicate_result)
                if not duplicate_result["success"]:
                    errors.extend(duplicate_result.get("errors", []))

            if include_integrity_validation:
                integrity_result = self.validate_data_integrity()
                results.append(integrity_result)
                if not integrity_result["success"]:
                    errors.extend(integrity_result.get("errors", []))

            overall_success = all(result["success"] for result in results)

            summary = {
                "tasks_run": [r["task_name"] for r in results],
                "tasks_succeeded": sum(1 for r in results if r["success"]),
                "tasks_failed": sum(1 for r in results if not r["success"]),
                "include_embedding_validation": include_embedding_validation,
                "include_duplicate_check": include_duplicate_check,
                "duplicate_threshold": duplicate_threshold,
                "include_integrity_validation": include_integrity_validation,
                "batch_size": batch_size,
                "propagation_progress_interval": propagation_progress_interval,
                "show_propagation_progress": show_propagation_progress,
            }

            all_reports: list[str] = []
            for result in results:
                all_reports.extend(result.get("report_files", []))

            self.logger.info("run_all summary=%s", summary)

            return self._build_result(
                task_name=task_name,
                success=overall_success,
                summary={
                    "overall": summary,
                    "task_results": results,
                },
                report_files=all_reports,
                errors=errors,
            )