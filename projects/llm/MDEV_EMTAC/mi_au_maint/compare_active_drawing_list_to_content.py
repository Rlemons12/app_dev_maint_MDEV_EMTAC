#!/usr/bin/env python
"""
compare_active_drawing_list_to_content.py

DROP-IN REPLACEMENT
-------------------
Purpose:
    Compare a trusted equipment workbook against a generated content map
    and produce a clean equipment-document relationship staging sheet.

Preserved CLI shape:
    python compare_active_drawing_list_to_content.py \
        --equipment-workbook "E:\\emtac\\Database\\DB_LOADSHEETS\\Active Drawing List.xlsx" \
        --content-map "E:\\emtac\\projects\\llm\\MDEV_EMTAC\\mi_au_maint\\migration_analysis\\2026-03-25_032802\\equipment_map\\document_equipment_associations_v2.csv" \
        --output-dir "E:\\emtac\\projects\\llm\\MDEV_EMTAC\\mi_au_maint\\migration_analysis\\2026-03-25_032802\\compare" \
        --verbose

Primary outputs:
    equipment_match_coverage.csv
    equipment_list_comparison_summary.json

Additional outputs:
    equipment_document_relation_all.csv
    equipment_document_relation_matched.csv
    equipment_document_relation_manual_review.csv
    equipment_document_relation_orphan_content.csv
    equipment_document_relation_missing_content.csv
    workbook_equipment_normalized.csv
    content_equipment_rollup.csv

Notes:
    - This script is intentionally defensive and heuristic-based because
      legacy workbook columns may vary.
    - It focuses on generating a clean staging relation sheet, not a final DB import.
    - It preserves "equipment_list_only" and "content_only" visibility because
      those are useful for manual review and migration planning.
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import pandas as pd


# ---------------------------------------------------------------------
# LOGGING
# ---------------------------------------------------------------------

LOGGER_NAME = "compare_equipment_list_to_content"
logger = logging.getLogger(LOGGER_NAME)


def configure_logging(verbose: bool = False) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    if logger.handlers:
        logger.setLevel(level)
        return

    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(
        logging.Formatter(
            "%(asctime)s - %(levelname)s - %(name)s - %(message)s"
        )
    )
    logger.addHandler(handler)
    logger.setLevel(level)
    logger.propagate = False


# ---------------------------------------------------------------------
# CONSTANTS / HEURISTICS
# ---------------------------------------------------------------------

DOCUMENT_LIKE_CATEGORIES = {
    "html",
    "pdf",
    "word",
    "excel",
    "powerpoint",
    "text",
}

EXCLUDED_RELATION_CATEGORIES = {
    "xml",
    "css",
    "javascript",
    "json",
    "archive",
}

LOW_VALUE_FILENAMES = {
    "thumbs.db",
    "filelist.xml",
    "cachedata.xml",
}

PREFERRED_EQUIPMENT_NUMBER_COLUMNS = [
    "equipment_number",
    "equipment no",
    "equipment_no",
    "equipment #",
    "equipment#",
    "asset_number",
    "asset number",
    "asset_no",
    "asset no",
    "drawing equipment number",
    "equipment id",
    "eq_no",
    "eq number",
    "eq_number",
    "tag",
    "tag number",
    "equipment",
]

PREFERRED_EQUIPMENT_DESCRIPTION_COLUMNS = [
    "description",
    "equipment description",
    "name",
    "equipment name",
    "title",
    "desc",
    "asset description",
    "drawing description",
]

CONTENT_EQUIPMENT_COLUMNS_PRIORITY = [
    "equipment_number_final",
    "equipment_number_direct",
    "inherited_equipment_number",
]

CONTENT_CONFIDENCE_ORDER = {"low": 1, "medium": 2, "high": 3}
MATCH_REASON_PRIORITY = {
    "high_value_matched": 5,
    "matched": 4,
    "review_required": 3,
    "orphan_content": 2,
    "missing_content": 1,
}


# ---------------------------------------------------------------------
# DATA CLASSES
# ---------------------------------------------------------------------

@dataclass
class WorkbookEquipmentRecord:
    canonical_equipment_number: str
    raw_equipment_number: str
    equipment_description: str
    source_sheet: str
    source_row_index: int


@dataclass
class ContentRelationRecord:
    canonical_equipment_number: str
    raw_equipment_number: str
    root_path: str
    relative_path: str
    file_name: str
    stem: str
    extension: str
    file_category: str
    is_document_like: bool
    parent_folder: str
    ancestor_path: str
    page_title: str
    hierarchy_level: int
    guessed_page_role: str
    side_final: str
    area_final: str
    equipment_number_final: str
    equipment_number_final_source: str
    nearest_equipment_folder: str
    nearest_launcher_ancestor: str
    nearest_launcher_ancestor_title: str
    nearest_content_ancestor: str
    nearest_content_ancestor_title: str
    association_confidence: str
    notes: str
    content_inclusion_reason: str
    content_exclusion_reason: str


# ---------------------------------------------------------------------
# UTILS
# ---------------------------------------------------------------------

def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def normalize_bool(value: object) -> bool:
    if pd.isna(value):
        return False
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    return text in {"true", "1", "yes", "y"}


def safe_str(value: object) -> str:
    if pd.isna(value):
        return ""
    return str(value).strip()


def safe_lower(value: object) -> str:
    return safe_str(value).lower()


def canonicalize_column_name(name: str) -> str:
    return re.sub(r"\s+", " ", safe_lower(name.replace("_", " ").replace("-", " ")))


def normalize_equipment_number(raw_value: object) -> str:
    """
    Normalize equipment identifiers without being overly destructive.

    Examples:
        " aco 10195 "   -> "ACO10195"
        "aco-10195"     -> "ACO10195"
        "ACS 4025"      -> "ACS4025"
        "ACO20070-06"   -> "ACO20070-06"  (keeps meaningful internal hyphen)
    """
    text = safe_str(raw_value).upper()
    if not text:
        return ""

    text = text.replace("_", "-")
    text = re.sub(r"\s+", "", text)

    # Remove obvious wrapper punctuation
    text = text.strip(".,;:()[]{}")

    # Remove hyphen only between leading alpha block and first digit block: ACO-10195 -> ACO10195
    text = re.sub(r"^([A-Z]+)-(?=\d)", r"\1", text)

    # Collapse duplicate hyphens
    text = re.sub(r"-{2,}", "-", text)

    return text


def unique_preserve_order(values: Iterable[str]) -> List[str]:
    seen = set()
    result = []
    for value in values:
        if value not in seen:
            seen.add(value)
            result.append(value)
    return result


def write_csv(df: pd.DataFrame, path: Path) -> None:
    logger.info("Writing CSV: %s", path)
    df.to_csv(path, index=False, encoding="utf-8-sig")


def write_json(data: object, path: Path) -> None:
    logger.info("Writing JSON: %s", path)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def choose_best_confidence(values: Sequence[str]) -> str:
    best = ""
    best_score = -1
    for value in values:
        score = CONTENT_CONFIDENCE_ORDER.get(safe_lower(value), 0)
        if score > best_score:
            best_score = score
            best = safe_lower(value)
    return best or "low"


def deduce_content_types(values: Iterable[str]) -> str:
    cleaned = [safe_lower(v) for v in values if safe_str(v)]
    return " | ".join(sorted(unique_preserve_order(cleaned)))


def is_low_value_support_file(file_name: str, file_category: str) -> bool:
    if safe_lower(file_name) in LOW_VALUE_FILENAMES:
        return True
    if safe_lower(file_category) in EXCLUDED_RELATION_CATEGORIES:
        return True
    return False


# ---------------------------------------------------------------------
# WORKBOOK LOADING
# ---------------------------------------------------------------------

def detect_column(columns: Sequence[str], preferred_names: Sequence[str]) -> Optional[str]:
    canonical_map = {canonicalize_column_name(col): col for col in columns}

    for preferred in preferred_names:
        key = canonicalize_column_name(preferred)
        if key in canonical_map:
            return canonical_map[key]

    # Partial contains fallback
    normalized_columns = [(canonicalize_column_name(col), col) for col in columns]
    for preferred in preferred_names:
        preferred_key = canonicalize_column_name(preferred)
        for normalized, original in normalized_columns:
            if preferred_key == normalized or preferred_key in normalized:
                return original

    return None


def load_equipment_workbook(workbook_path: Path) -> Tuple[pd.DataFrame, List[WorkbookEquipmentRecord]]:
    logger.info("Loading equipment workbook: %s", workbook_path)

    excel_file = pd.ExcelFile(workbook_path)
    records: List[WorkbookEquipmentRecord] = []
    workbook_frames: List[pd.DataFrame] = []

    for sheet_name in excel_file.sheet_names:
        try:
            df = excel_file.parse(sheet_name=sheet_name)
        except Exception as exc:
            logger.warning("Skipping sheet '%s' due to parse error: %s", sheet_name, exc)
            continue

        if df.empty:
            continue

        original_columns = list(df.columns)
        equipment_col = detect_column(original_columns, PREFERRED_EQUIPMENT_NUMBER_COLUMNS)
        description_col = detect_column(original_columns, PREFERRED_EQUIPMENT_DESCRIPTION_COLUMNS)

        if equipment_col is None:
            logger.debug("Skipping sheet '%s' because no equipment-number-like column was detected.", sheet_name)
            continue

        logger.debug(
            "Sheet '%s' detected columns: equipment='%s', description='%s'",
            sheet_name,
            equipment_col,
            description_col,
        )

        for idx, row in df.iterrows():
            raw_equipment = safe_str(row.get(equipment_col, ""))
            canonical_equipment = normalize_equipment_number(raw_equipment)
            if not canonical_equipment:
                continue

            description = safe_str(row.get(description_col, "")) if description_col else ""

            record = WorkbookEquipmentRecord(
                canonical_equipment_number=canonical_equipment,
                raw_equipment_number=raw_equipment,
                equipment_description=description,
                source_sheet=sheet_name,
                source_row_index=int(idx) + 2,  # +2 to approximate Excel row number with header
            )
            records.append(record)

        temp = df.copy()
        temp["__source_sheet"] = sheet_name
        workbook_frames.append(temp)

    if not records:
        raise ValueError(
            f"No equipment-number-like column could be resolved from workbook: {workbook_path}"
        )

    logger.info("Loaded equipment rows: %s", len(records))
    combined_df = pd.concat(workbook_frames, ignore_index=True) if workbook_frames else pd.DataFrame()
    return combined_df, records


def build_workbook_equipment_df(records: List[WorkbookEquipmentRecord]) -> pd.DataFrame:
    rows = [asdict(r) for r in records]
    df = pd.DataFrame(rows)

    if df.empty:
        return df

    grouped = (
        df.groupby("canonical_equipment_number", as_index=False)
        .agg(
            raw_equipment_number=("raw_equipment_number", "first"),
            equipment_description=("equipment_description", "first"),
            workbook_row_count=("canonical_equipment_number", "size"),
            source_sheets=("source_sheet", lambda x: " | ".join(sorted(unique_preserve_order(map(str, x))))),
            source_rows=("source_row_index", lambda x: " | ".join(map(str, sorted(set(map(int, x)))))),
        )
        .sort_values("canonical_equipment_number")
        .reset_index(drop=True)
    )

    return grouped


# ---------------------------------------------------------------------
# CONTENT MAP LOADING
# ---------------------------------------------------------------------

def resolve_content_equipment_number(row: pd.Series) -> Tuple[str, str]:
    for col in CONTENT_EQUIPMENT_COLUMNS_PRIORITY:
        if col in row.index:
            raw_value = safe_str(row[col])
            canonical = normalize_equipment_number(raw_value)
            if canonical:
                return canonical, raw_value
    return "", ""


def content_row_is_relation_candidate(row: pd.Series) -> Tuple[bool, str]:
    file_name = safe_str(row.get("file_name", ""))
    file_category = safe_lower(row.get("file_category", ""))
    is_document_like = normalize_bool(row.get("is_document_like", False))
    relative_path = safe_lower(row.get("relative_path", ""))

    if file_name.lower() in LOW_VALUE_FILENAMES:
        return False, "low_value_support_file"

    if file_category in EXCLUDED_RELATION_CATEGORIES:
        return False, "excluded_category"

    if relative_path.endswith("/thumbs.db") or relative_path.endswith("\\thumbs.db"):
        return False, "thumbs_db"

    # Strong inclusion
    if is_document_like:
        return True, "document_like"

    # Images can still be useful if tied to equipment, but keep them as candidates.
    if file_category == "image":
        return True, "image_candidate"

    # A few non-document categories are still worth staging when mapped to equipment.
    if file_category in {"other"}:
        return True, "other_candidate"

    return False, "non_document_low_value"


def load_content_map(content_map_path: Path) -> pd.DataFrame:
    logger.info("Loading content map CSV: %s", content_map_path)
    df = pd.read_csv(content_map_path, dtype=str, keep_default_na=False)

    required_columns = {
        "root_path",
        "relative_path",
        "file_name",
        "extension",
        "file_category",
        "is_document_like",
        "association_confidence",
    }
    missing = [col for col in required_columns if col not in df.columns]
    if missing:
        raise ValueError(
            f"Content map is missing required columns: {missing}. "
            f"Found columns: {list(df.columns)}"
        )

    logger.info("Loaded content rows: %s", len(df))
    return df


def build_content_relation_records(content_df: pd.DataFrame) -> List[ContentRelationRecord]:
    records: List[ContentRelationRecord] = []

    for _, row in content_df.iterrows():
        canonical_equipment, raw_equipment = resolve_content_equipment_number(row)
        include, reason = content_row_is_relation_candidate(row)

        record = ContentRelationRecord(
            canonical_equipment_number=canonical_equipment,
            raw_equipment_number=raw_equipment,
            root_path=safe_str(row.get("root_path", "")),
            relative_path=safe_str(row.get("relative_path", "")),
            file_name=safe_str(row.get("file_name", "")),
            stem=safe_str(row.get("stem", "")),
            extension=safe_str(row.get("extension", "")),
            file_category=safe_str(row.get("file_category", "")),
            is_document_like=normalize_bool(row.get("is_document_like", False)),
            parent_folder=safe_str(row.get("parent_folder", "")),
            ancestor_path=safe_str(row.get("ancestor_path", "")),
            page_title=safe_str(row.get("page_title", "")),
            hierarchy_level=int(safe_str(row.get("hierarchy_level", "-1")) or -1),
            guessed_page_role=safe_str(row.get("guessed_page_role", "")),
            side_final=safe_str(row.get("side_final", "")),
            area_final=safe_str(row.get("area_final", "")),
            equipment_number_final=safe_str(row.get("equipment_number_final", "")),
            equipment_number_final_source=safe_str(row.get("equipment_number_final_source", "")),
            nearest_equipment_folder=safe_str(row.get("nearest_equipment_folder", "")),
            nearest_launcher_ancestor=safe_str(row.get("nearest_launcher_ancestor", "")),
            nearest_launcher_ancestor_title=safe_str(row.get("nearest_launcher_ancestor_title", "")),
            nearest_content_ancestor=safe_str(row.get("nearest_content_ancestor", "")),
            nearest_content_ancestor_title=safe_str(row.get("nearest_content_ancestor_title", "")),
            association_confidence=safe_lower(row.get("association_confidence", "")) or "low",
            notes=safe_str(row.get("notes", "")),
            content_inclusion_reason=reason if include else "",
            content_exclusion_reason="" if include else reason,
        )
        records.append(record)

    return records


# ---------------------------------------------------------------------
# RELATION BUILD
# ---------------------------------------------------------------------

def build_content_df(records: List[ContentRelationRecord]) -> pd.DataFrame:
    df = pd.DataFrame([asdict(r) for r in records])
    if df.empty:
        return df

    df["has_equipment_number"] = df["canonical_equipment_number"].astype(str).str.len() > 0
    return df


def rollup_content_by_equipment(content_df: pd.DataFrame) -> pd.DataFrame:
    if content_df.empty:
        return pd.DataFrame()

    included = content_df[content_df["content_exclusion_reason"] == ""].copy()

    if included.empty:
        return pd.DataFrame(
            columns=[
                "canonical_equipment_number",
                "content_row_count",
                "document_like_count",
                "content_types",
                "best_confidence",
                "sides",
                "areas",
            ]
        )

    rollup = (
        included[included["canonical_equipment_number"] != ""]
        .groupby("canonical_equipment_number", as_index=False)
        .agg(
            content_row_count=("canonical_equipment_number", "size"),
            document_like_count=("is_document_like", lambda x: int(sum(bool(v) for v in x))),
            content_types=("file_category", deduce_content_types),
            best_confidence=("association_confidence", choose_best_confidence),
            sides=("side_final", lambda x: " | ".join(sorted(v for v in unique_preserve_order(map(safe_str, x)) if v))),
            areas=("area_final", lambda x: " | ".join(sorted(v for v in unique_preserve_order(map(safe_str, x)) if v))),
        )
        .sort_values(["content_row_count", "canonical_equipment_number"], ascending=[False, True])
        .reset_index(drop=True)
    )

    return rollup


def classify_equipment_coverage(
    workbook_df: pd.DataFrame,
    content_rollup_df: pd.DataFrame,
) -> pd.DataFrame:
    workbook_keys = set(workbook_df["canonical_equipment_number"]) if not workbook_df.empty else set()
    content_keys = set(content_rollup_df["canonical_equipment_number"]) if not content_rollup_df.empty else set()
    all_keys = sorted(workbook_keys | content_keys)

    workbook_map = (
        workbook_df.set_index("canonical_equipment_number").to_dict(orient="index")
        if not workbook_df.empty else {}
    )
    content_map = (
        content_rollup_df.set_index("canonical_equipment_number").to_dict(orient="index")
        if not content_rollup_df.empty else {}
    )

    rows = []
    for equipment_number in all_keys:
        in_workbook = equipment_number in workbook_keys
        in_content = equipment_number in content_keys

        wb = workbook_map.get(equipment_number, {})
        ct = content_map.get(equipment_number, {})

        if in_workbook and in_content:
            match_status = "matched"
            if int(ct.get("content_row_count", 0) or 0) >= 100:
                match_reason = "high_value_matched"
            else:
                match_reason = "matched"
        elif in_workbook and not in_content:
            match_status = "equipment_list_only"
            match_reason = "missing_content"
        else:
            match_status = "content_only"
            match_reason = "orphan_content"

        rows.append(
            {
                "equipment_number": equipment_number,
                "equipment_description": wb.get("equipment_description", ""),
                "exists_in_workbook": in_workbook,
                "exists_in_content": in_content,
                "content_row_count": int(ct.get("content_row_count", 0) or 0),
                "content_types": ct.get("content_types", ""),
                "best_confidence": ct.get("best_confidence", ""),
                "match_status": match_status,
                "match_reason": match_reason,
                "sides": ct.get("sides", ""),
                "areas": ct.get("areas", ""),
                "workbook_row_count": int(wb.get("workbook_row_count", 0) or 0),
                "source_sheets": wb.get("source_sheets", ""),
                "source_rows": wb.get("source_rows", ""),
            }
        )

    coverage_df = pd.DataFrame(rows)
    if not coverage_df.empty:
        coverage_df = coverage_df.sort_values(
            by=["match_status", "content_row_count", "equipment_number"],
            ascending=[True, False, True],
        ).reset_index(drop=True)

    return coverage_df


def build_equipment_document_relation_all(
    workbook_df: pd.DataFrame,
    content_df: pd.DataFrame,
    coverage_df: pd.DataFrame,
) -> pd.DataFrame:
    workbook_map = (
        workbook_df.set_index("canonical_equipment_number").to_dict(orient="index")
        if not workbook_df.empty else {}
    )
    coverage_map = (
        coverage_df.set_index("equipment_number").to_dict(orient="index")
        if not coverage_df.empty else {}
    )

    included = content_df[content_df["content_exclusion_reason"] == ""].copy()

    relation_rows: List[Dict[str, object]] = []

    for _, row in included.iterrows():
        canonical_equipment = safe_str(row.get("canonical_equipment_number", ""))
        coverage = coverage_map.get(canonical_equipment, {})
        workbook = workbook_map.get(canonical_equipment, {})

        # Content row has an equipment number
        if canonical_equipment:
            if coverage.get("match_status") == "matched":
                review_status = "auto_matched"
            elif coverage.get("match_status") == "content_only":
                review_status = "manual_review"
            else:
                review_status = "manual_review"
        else:
            review_status = "manual_review"

        relation_rows.append(
            {
                "equipment_number": canonical_equipment,
                "raw_equipment_number": safe_str(row.get("raw_equipment_number", "")),
                "canonical_equipment_number": canonical_equipment,
                "equipment_description": workbook.get("equipment_description", ""),
                "exists_in_workbook": bool(workbook),
                "exists_in_content": True,
                "match_status": coverage.get("match_status", "content_only" if canonical_equipment else "needs_review"),
                "match_reason": coverage.get("match_reason", "unresolved_content"),
                "review_status": review_status,
                "association_confidence": safe_str(row.get("association_confidence", "")),
                "equipment_number_final_source": safe_str(row.get("equipment_number_final_source", "")),
                "root_path": safe_str(row.get("root_path", "")),
                "relative_path": safe_str(row.get("relative_path", "")),
                "file_name": safe_str(row.get("file_name", "")),
                "stem": safe_str(row.get("stem", "")),
                "extension": safe_str(row.get("extension", "")),
                "file_category": safe_str(row.get("file_category", "")),
                "is_document_like": bool(row.get("is_document_like", False)),
                "parent_folder": safe_str(row.get("parent_folder", "")),
                "ancestor_path": safe_str(row.get("ancestor_path", "")),
                "page_title": safe_str(row.get("page_title", "")),
                "hierarchy_level": int(row.get("hierarchy_level", -1)),
                "guessed_page_role": safe_str(row.get("guessed_page_role", "")),
                "side_final": safe_str(row.get("side_final", "")),
                "area_final": safe_str(row.get("area_final", "")),
                "nearest_equipment_folder": safe_str(row.get("nearest_equipment_folder", "")),
                "nearest_launcher_ancestor": safe_str(row.get("nearest_launcher_ancestor", "")),
                "nearest_launcher_ancestor_title": safe_str(row.get("nearest_launcher_ancestor_title", "")),
                "nearest_content_ancestor": safe_str(row.get("nearest_content_ancestor", "")),
                "nearest_content_ancestor_title": safe_str(row.get("nearest_content_ancestor_title", "")),
                "notes": safe_str(row.get("notes", "")),
                "content_inclusion_reason": safe_str(row.get("content_inclusion_reason", "")),
            }
        )

    # Also add explicit workbook-only missing content rows so the sheet is complete.
    for _, coverage in coverage_df[coverage_df["match_status"] == "equipment_list_only"].iterrows():
        relation_rows.append(
            {
                "equipment_number": safe_str(coverage.get("equipment_number", "")),
                "raw_equipment_number": safe_str(coverage.get("equipment_number", "")),
                "canonical_equipment_number": safe_str(coverage.get("equipment_number", "")),
                "equipment_description": safe_str(coverage.get("equipment_description", "")),
                "exists_in_workbook": True,
                "exists_in_content": False,
                "match_status": "equipment_list_only",
                "match_reason": "missing_content",
                "review_status": "missing_content",
                "association_confidence": "",
                "equipment_number_final_source": "",
                "root_path": "",
                "relative_path": "",
                "file_name": "",
                "stem": "",
                "extension": "",
                "file_category": "",
                "is_document_like": False,
                "parent_folder": "",
                "ancestor_path": "",
                "page_title": "",
                "hierarchy_level": -1,
                "guessed_page_role": "",
                "side_final": safe_str(coverage.get("sides", "")),
                "area_final": safe_str(coverage.get("areas", "")),
                "nearest_equipment_folder": "",
                "nearest_launcher_ancestor": "",
                "nearest_launcher_ancestor_title": "",
                "nearest_content_ancestor": "",
                "nearest_content_ancestor_title": "",
                "notes": "",
                "content_inclusion_reason": "workbook_only_equipment",
            }
        )

    relation_df = pd.DataFrame(relation_rows)

    if relation_df.empty:
        return relation_df

    relation_df = relation_df.sort_values(
        by=[
            "match_status",
            "equipment_number",
            "relative_path",
            "file_name",
        ],
        ascending=[True, True, True, True],
    ).reset_index(drop=True)

    return relation_df


# ---------------------------------------------------------------------
# SUMMARY
# ---------------------------------------------------------------------

def build_summary(
    workbook_path: Path,
    content_map_path: Path,
    output_dir: Path,
    workbook_df: pd.DataFrame,
    content_df: pd.DataFrame,
    content_rollup_df: pd.DataFrame,
    coverage_df: pd.DataFrame,
    relation_df: pd.DataFrame,
) -> Dict[str, object]:
    matched_count = int((coverage_df["match_status"] == "matched").sum()) if not coverage_df.empty else 0
    workbook_only_count = int((coverage_df["match_status"] == "equipment_list_only").sum()) if not coverage_df.empty else 0
    content_only_count = int((coverage_df["match_status"] == "content_only").sum()) if not coverage_df.empty else 0

    summary = {
        "equipment_workbook": str(workbook_path),
        "content_map": str(content_map_path),
        "output_dir": str(output_dir),
        "workbook_equipment_rows": int(len(workbook_df)),
        "content_rows_loaded": int(len(content_df)),
        "content_rows_with_equipment_number": int((content_df["canonical_equipment_number"] != "").sum()) if not content_df.empty else 0,
        "content_relation_candidates": int((content_df["content_exclusion_reason"] == "").sum()) if not content_df.empty else 0,
        "content_excluded_rows": int((content_df["content_exclusion_reason"] != "").sum()) if not content_df.empty else 0,
        "unique_content_equipment_groups": int(len(content_rollup_df)),
        "total_equipment_groups_compared": int(len(coverage_df)),
        "matched_equipment_groups": matched_count,
        "equipment_list_only_groups": workbook_only_count,
        "content_only_groups": content_only_count,
        "relation_all_rows": int(len(relation_df)),
        "relation_auto_matched_rows": int((relation_df["review_status"] == "auto_matched").sum()) if not relation_df.empty else 0,
        "relation_manual_review_rows": int((relation_df["review_status"] == "manual_review").sum()) if not relation_df.empty else 0,
        "relation_missing_content_rows": int((relation_df["review_status"] == "missing_content").sum()) if not relation_df.empty else 0,
        "top_matched_equipment": coverage_df[coverage_df["match_status"] == "matched"]
        .sort_values(["content_row_count", "equipment_number"], ascending=[False, True])
        .head(25)[["equipment_number", "content_row_count", "content_types", "best_confidence"]]
        .to_dict(orient="records")
        if not coverage_df.empty else [],
    }
    return summary


# ---------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------

def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare equipment workbook to content map and generate an equipment-document relationship sheet."
    )
    parser.add_argument(
        "--equipment-workbook",
        required=True,
        help="Path to trusted workbook containing equipment rows.",
    )
    parser.add_argument(
        "--content-map",
        required=True,
        help="Path to document_equipment_associations_v2.csv or equivalent content-map CSV.",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory where outputs should be written.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging.",
    )
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    configure_logging(verbose=args.verbose)

    workbook_path = Path(args.equipment_workbook)
    content_map_path = Path(args.content_map)
    output_dir = Path(args.output_dir)

    ensure_dir(output_dir)

    if not workbook_path.exists():
        raise FileNotFoundError(f"Equipment workbook not found: {workbook_path}")
    if not content_map_path.exists():
        raise FileNotFoundError(f"Content map CSV not found: {content_map_path}")

    logger.info("Loading equipment workbook: %s", workbook_path)
    _, workbook_records = load_equipment_workbook(workbook_path)
    workbook_df = build_workbook_equipment_df(workbook_records)

    logger.info("Loading content map CSV: %s", content_map_path)
    raw_content_df = load_content_map(content_map_path)
    content_records = build_content_relation_records(raw_content_df)
    content_df = build_content_df(content_records)

    logger.info("Rolling up content by equipment.")
    content_rollup_df = rollup_content_by_equipment(content_df)

    logger.info("Building equipment coverage comparison.")
    coverage_df = classify_equipment_coverage(workbook_df, content_rollup_df)

    logger.info("Building equipment-document relation output.")
    relation_df = build_equipment_document_relation_all(
        workbook_df=workbook_df,
        content_df=content_df,
        coverage_df=coverage_df,
    )

    matched_df = relation_df[relation_df["review_status"] == "auto_matched"].copy()
    manual_review_df = relation_df[relation_df["review_status"] == "manual_review"].copy()
    orphan_content_df = relation_df[relation_df["match_status"] == "content_only"].copy()
    missing_content_df = relation_df[relation_df["match_status"] == "equipment_list_only"].copy()

    summary = build_summary(
        workbook_path=workbook_path,
        content_map_path=content_map_path,
        output_dir=output_dir,
        workbook_df=workbook_df,
        content_df=content_df,
        content_rollup_df=content_rollup_df,
        coverage_df=coverage_df,
        relation_df=relation_df,
    )

    # Preserved / expected outputs
    coverage_csv = output_dir / "equipment_match_coverage.csv"
    summary_json = output_dir / "equipment_list_comparison_summary.json"

    # New cleaner staging outputs
    workbook_csv = output_dir / "workbook_equipment_normalized.csv"
    content_rollup_csv = output_dir / "content_equipment_rollup.csv"
    relation_all_csv = output_dir / "equipment_document_relation_all.csv"
    relation_matched_csv = output_dir / "equipment_document_relation_matched.csv"
    relation_manual_csv = output_dir / "equipment_document_relation_manual_review.csv"
    relation_orphan_csv = output_dir / "equipment_document_relation_orphan_content.csv"
    relation_missing_csv = output_dir / "equipment_document_relation_missing_content.csv"

    write_csv(workbook_df, workbook_csv)
    write_csv(content_rollup_df, content_rollup_csv)
    write_csv(coverage_df, coverage_csv)
    write_csv(relation_df, relation_all_csv)
    write_csv(matched_df, relation_matched_csv)
    write_csv(manual_review_df, relation_manual_csv)
    write_csv(orphan_content_df, relation_orphan_csv)
    write_csv(missing_content_df, relation_missing_csv)
    write_json(summary, summary_json)

    print("Equipment list comparison complete.")
    print(f"Output directory: {output_dir}")
    print(f"Coverage CSV: {coverage_csv}")
    print(f"Summary JSON: {summary_json}")
    print(f"Relation All CSV: {relation_all_csv}")
    print(f"Relation Matched CSV: {relation_matched_csv}")
    print(f"Relation Manual Review CSV: {relation_manual_csv}")
    print(f"Relation Orphan Content CSV: {relation_orphan_csv}")
    print(f"Relation Missing Content CSV: {relation_missing_csv}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())