from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
from collections import defaultdict
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple


# ---------------------------------------------------------
# Ensure project root is on sys.path when running script directly
# ---------------------------------------------------------
CURRENT_FILE = Path(__file__).resolve()
PACKAGE_DIR = CURRENT_FILE.parent.resolve()
PROJECT_ROOT = CURRENT_FILE.parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


logger = logging.getLogger("build_migration_worklist")


# =========================================================
# Logging
# =========================================================
def configure_logging(verbose: bool = False) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    )


# =========================================================
# Data models
# =========================================================
@dataclass
class WorklistRow:
    equipment_number: str
    equipment_name: str
    migration_bucket: str
    migration_wave: str
    priority_hint: str
    content_file_count: int
    file_categories: str
    hierarchy_file_count: int
    html_file_count: int
    document_file_count: int
    image_file_count: int
    top_launcher_ancestors: str
    top_content_ancestors: str
    top_areas: str
    top_sides: str
    highest_confidence: str
    workbook_match_status: str
    recommended_target_type: str
    recommended_action: str
    manual_review_reason: str
    notes: str = ""


@dataclass
class ManualReviewRow:
    equipment_number: str
    equipment_name: str
    reason: str
    priority_hint: str
    content_file_count: int
    file_categories: str
    top_launcher_ancestors: str
    notes: str = ""


# =========================================================
# Helpers
# =========================================================
def safe_load_json(path: Optional[Path]) -> Dict:
    if not path or not path.exists():
        return {}

    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        logger.warning("Could not load JSON %s | reason=%s", path, exc)
        return {}


def load_csv(path: Path) -> List[Dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(f"CSV not found: {path}")

    rows: List[Dict[str, str]] = []

    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(
                {
                    (k.strip() if isinstance(k, str) else k): (
                        v.strip() if isinstance(v, str) else v
                    )
                    for k, v in row.items()
                }
            )

    return rows


def normalize_text(value) -> str:
    if value is None:
        return ""
    return str(value).strip()


def normalize_equipment_number(value: str) -> str:
    return normalize_text(value).upper()


def safe_int(value, default: int = 0) -> int:
    try:
        if value is None or value == "":
            return default
        return int(float(value))
    except Exception:
        return default


def unique_join(
    values: List[str],
    sep: str = " | ",
    max_items: int = 10,
) -> str:
    cleaned: List[str] = []
    seen = set()

    for value in values:
        v = normalize_text(value)
        if not v:
            continue
        if v in seen:
            continue

        seen.add(v)
        cleaned.append(v)

        if len(cleaned) >= max_items:
            break

    return sep.join(cleaned)


def write_csv(
    rows: List[object],
    output_path: Path,
    fallback_headers: List[str],
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    headers = list(asdict(rows[0]).keys()) if rows else fallback_headers

    with output_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        for row in rows:
            writer.writerow(asdict(row))


def write_json(payload: Dict, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


# =========================================================
# Core logic
# =========================================================
def categorize_file_category(file_category: str) -> str:
    category = normalize_text(file_category).lower()

    if category == "html":
        return "html"
    if category in {"pdf", "word", "excel", "powerpoint", "text"}:
        return "document"
    if category == "image":
        return "image"
    return "other"


def recommend_target_type(
    *,
    html_file_count: int,
    document_file_count: int,
    image_file_count: int,
    launcher_ancestors: List[str],
) -> str:
    if html_file_count >= 10 and launcher_ancestors:
        return "equipment_module"
    if document_file_count >= 10 and html_file_count < 5:
        return "document_library"
    if image_file_count > document_file_count and html_file_count < 3:
        return "media_gallery"
    return "mixed_equipment_module"


def recommend_action(
    *,
    bucket: str,
    target_type: str,
    workbook_match_status: str,
) -> str:
    if bucket == "Wave 1":
        if target_type == "equipment_module":
            return "rebuild_as_equipment_page"
        if target_type == "document_library":
            return "create_equipment_document_library"
        return "rebuild_as_mixed_module"

    if bucket == "Wave 2":
        if workbook_match_status == "matched":
            return "migrate_after_wave_1"
        return "review_then_migrate"

    return "manual_review_required"


def determine_manual_review_reason(
    *,
    workbook_match_status: str,
    priority_hint: str,
    content_file_count: int,
    highest_confidence: str,
) -> str:
    if workbook_match_status == "content_only":
        if priority_hint == "orphan_content_high_volume":
            return "high_volume_content_not_in_workbook"
        return "content_not_in_workbook"

    if workbook_match_status == "equipment_list_only":
        return "workbook_equipment_missing_content"

    if highest_confidence in {"low", "low-medium"}:
        return "weak_equipment_association_confidence"

    if content_file_count == 0:
        return "no_content_files"

    return ""


def assign_bucket(
    *,
    workbook_match_status: str,
    priority_hint: str,
    content_file_count: int,
    highest_confidence: str,
) -> Tuple[str, str]:
    if workbook_match_status == "matched":
        if priority_hint == "high_value_matched":
            return "Wave 1", "Wave 1"
        if priority_hint == "matched_mid_volume":
            return "Wave 2", "Wave 2"
        if priority_hint == "matched_low_volume":
            return "Manual Review", ""
        if highest_confidence in {"high", "medium"} and content_file_count >= 25:
            return "Wave 2", "Wave 2"
        return "Manual Review", ""

    if workbook_match_status == "content_only":
        return "Manual Review", ""

    if workbook_match_status == "equipment_list_only":
        return "Manual Review", ""

    return "Manual Review", ""


def build_equipment_aggregates(
    equipment_map_rows: List[Dict[str, str]],
) -> Dict[str, Dict[str, object]]:
    grouped: Dict[str, Dict[str, object]] = {}

    for row in equipment_map_rows:
        eq = normalize_equipment_number(row.get("equipment_number_final", ""))
        if not eq:
            continue

        if eq not in grouped:
            grouped[eq] = {
                "row_count": 0,
                "file_categories": [],
                "top_launcher_ancestors": [],
                "top_content_ancestors": [],
                "areas": [],
                "sides": [],
                "confidences": [],
                "hierarchy_file_count": 0,
                "html_file_count": 0,
                "document_file_count": 0,
                "image_file_count": 0,
            }

        grouped[eq]["row_count"] += 1

        file_category = normalize_text(row.get("file_category", ""))
        grouped[eq]["file_categories"].append(file_category)

        launcher_title = normalize_text(
            row.get("nearest_launcher_ancestor_title", "")
        )
        if not launcher_title:
            launcher_title = normalize_text(
                row.get("nearest_launcher_ancestor", "")
            )
        grouped[eq]["top_launcher_ancestors"].append(launcher_title)

        content_title = normalize_text(
            row.get("nearest_content_ancestor_title", "")
        )
        if not content_title:
            content_title = normalize_text(
                row.get("nearest_content_ancestor", "")
            )
        grouped[eq]["top_content_ancestors"].append(content_title)

        grouped[eq]["areas"].append(normalize_text(row.get("area_final", "")))
        grouped[eq]["sides"].append(normalize_text(row.get("side_final", "")))
        grouped[eq]["confidences"].append(
            normalize_text(row.get("association_confidence", ""))
        )

        hierarchy_level = safe_int(row.get("hierarchy_level", -1), default=-1)
        if hierarchy_level >= 0:
            grouped[eq]["hierarchy_file_count"] += 1

        category_bucket = categorize_file_category(file_category)
        if category_bucket == "html":
            grouped[eq]["html_file_count"] += 1
        elif category_bucket == "document":
            grouped[eq]["document_file_count"] += 1
        elif category_bucket == "image":
            grouped[eq]["image_file_count"] += 1

    return grouped


def choose_highest_confidence(confidences: List[str]) -> str:
    rank = {
        "high": 4,
        "medium": 3,
        "low-medium": 2,
        "low": 1,
        "": 0,
    }

    best = ""
    best_score = -1

    for confidence in confidences:
        c = normalize_text(confidence)
        score = rank.get(c, 0)
        if score > best_score:
            best_score = score
            best = c

    return best


def build_worklist(
    coverage_rows: List[Dict[str, str]],
    equipment_map_rows: List[Dict[str, str]],
) -> Tuple[List[WorklistRow], List[ManualReviewRow], Dict]:
    equipment_aggregates = build_equipment_aggregates(equipment_map_rows)

    worklist_rows: List[WorklistRow] = []
    manual_review_rows: List[ManualReviewRow] = []

    bucket_counts: Dict[str, int] = defaultdict(int)
    wave_counts: Dict[str, int] = defaultdict(int)

    for coverage in coverage_rows:
        equipment_number = normalize_equipment_number(
            coverage.get("equipment_number", "")
        )
        if not equipment_number:
            continue

        equipment_name = normalize_text(coverage.get("equipment_name", ""))
        priority_hint = normalize_text(coverage.get("priority_hint", ""))
        workbook_match_status = normalize_text(coverage.get("match_status", ""))
        content_file_count = safe_int(
            coverage.get("content_file_count", 0),
            default=0,
        )
        coverage_categories = normalize_text(
            coverage.get("content_file_categories", "")
        )

        agg = equipment_aggregates.get(
            equipment_number,
            {
                "row_count": 0,
                "file_categories": [],
                "top_launcher_ancestors": [],
                "top_content_ancestors": [],
                "areas": [],
                "sides": [],
                "confidences": [],
                "hierarchy_file_count": 0,
                "html_file_count": 0,
                "document_file_count": 0,
                "image_file_count": 0,
            },
        )

        top_launcher_ancestors = unique_join(agg["top_launcher_ancestors"])
        top_content_ancestors = unique_join(agg["top_content_ancestors"])
        top_areas = unique_join(agg["areas"])
        top_sides = unique_join(agg["sides"])
        highest_confidence = choose_highest_confidence(agg["confidences"])

        migration_bucket, migration_wave = assign_bucket(
            workbook_match_status=workbook_match_status,
            priority_hint=priority_hint,
            content_file_count=content_file_count,
            highest_confidence=highest_confidence,
        )

        target_type = recommend_target_type(
            html_file_count=agg["html_file_count"],
            document_file_count=agg["document_file_count"],
            image_file_count=agg["image_file_count"],
            launcher_ancestors=agg["top_launcher_ancestors"],
        )

        manual_review_reason = (
            determine_manual_review_reason(
                workbook_match_status=workbook_match_status,
                priority_hint=priority_hint,
                content_file_count=content_file_count,
                highest_confidence=highest_confidence,
            )
            if migration_bucket == "Manual Review"
            else ""
        )

        recommended_action = recommend_action(
            bucket=migration_bucket,
            target_type=target_type,
            workbook_match_status=workbook_match_status,
        )

        row = WorklistRow(
            equipment_number=equipment_number,
            equipment_name=equipment_name,
            migration_bucket=migration_bucket,
            migration_wave=migration_wave,
            priority_hint=priority_hint,
            content_file_count=content_file_count,
            file_categories=coverage_categories or unique_join(
                agg["file_categories"]
            ),
            hierarchy_file_count=agg["hierarchy_file_count"],
            html_file_count=agg["html_file_count"],
            document_file_count=agg["document_file_count"],
            image_file_count=agg["image_file_count"],
            top_launcher_ancestors=top_launcher_ancestors,
            top_content_ancestors=top_content_ancestors,
            top_areas=top_areas,
            top_sides=top_sides,
            highest_confidence=highest_confidence,
            workbook_match_status=workbook_match_status,
            recommended_target_type=target_type,
            recommended_action=recommended_action,
            manual_review_reason=manual_review_reason,
            notes="",
        )
        worklist_rows.append(row)

        bucket_counts[migration_bucket] += 1
        if migration_wave:
            wave_counts[migration_wave] += 1

        if migration_bucket == "Manual Review":
            manual_review_rows.append(
                ManualReviewRow(
                    equipment_number=equipment_number,
                    equipment_name=equipment_name,
                    reason=manual_review_reason or "manual_review_required",
                    priority_hint=priority_hint,
                    content_file_count=content_file_count,
                    file_categories=coverage_categories or unique_join(
                        agg["file_categories"]
                    ),
                    top_launcher_ancestors=top_launcher_ancestors,
                    notes="",
                )
            )

    def sort_key(row: WorklistRow):
        bucket_rank = {"Wave 1": 1, "Wave 2": 2, "Manual Review": 3}
        return (
            bucket_rank.get(row.migration_bucket, 99),
            -row.content_file_count,
            row.equipment_number,
        )

    worklist_rows.sort(key=sort_key)
    manual_review_rows.sort(
        key=lambda r: (-r.content_file_count, r.equipment_number)
    )

    summary = {
        "total_worklist_rows": len(worklist_rows),
        "bucket_counts": dict(sorted(bucket_counts.items(), key=lambda x: x[0])),
        "wave_counts": dict(sorted(wave_counts.items(), key=lambda x: x[0])),
    }

    return worklist_rows, manual_review_rows, summary


# =========================================================
# Validation and paths
# =========================================================
def resolve_run_root(run_root_arg: str) -> Path:
    run_root = Path(run_root_arg).resolve()
    if not run_root.exists():
        raise FileNotFoundError(f"Run root not found: {run_root}")
    if not run_root.is_dir():
        raise NotADirectoryError(f"Run root is not a directory: {run_root}")
    return run_root


def validate_run_structure(run_root: Path) -> Dict[str, Optional[Path]]:
    compare_coverage = run_root / "compare" / "equipment_match_coverage.csv"
    equipment_map = (
        run_root / "equipment_map" / "document_equipment_associations_v2.csv"
    )
    hierarchy_pages = run_root / "hierarchy" / "site_hierarchy_pages.csv"
    master_summary = run_root / "reports" / "migration_master_summary.json"
    reports_dir = run_root / "reports"

    required_files = {
        "compare coverage": compare_coverage,
        "equipment map": equipment_map,
        "hierarchy pages": hierarchy_pages,
    }

    missing = {
        label: path
        for label, path in required_files.items()
        if not path.exists()
    }

    if missing:
        lines = [
            "Run folder is missing required canonical files:",
            f"Run root: {run_root}",
            f"Package dir: {PACKAGE_DIR}",
            "",
        ]

        for label, path in missing.items():
            lines.append(f"- {label}: {path}")

        lines.extend(
            [
                "",
                "Expected canonical layout:",
                f"- {compare_coverage}",
                f"- {equipment_map}",
                f"- {hierarchy_pages}",
            ]
        )

        raise FileNotFoundError("\n".join(lines))

    reports_dir.mkdir(parents=True, exist_ok=True)

    return {
        "compare_coverage": compare_coverage,
        "equipment_map": equipment_map,
        "hierarchy_pages": hierarchy_pages,
        "master_summary": master_summary if master_summary.exists() else None,
        "reports_dir": reports_dir,
    }


# =========================================================
# CLI
# =========================================================
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build migration worklist CSVs from a migration_analysis run folder."
    )
    parser.add_argument(
        "--run-root",
        type=str,
        required=True,
        help="Path to one migration_analysis/<run_name> folder.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="",
        help="Optional output directory. Defaults to <run-root>/reports/worklist",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    configure_logging(verbose=args.verbose)

    run_root = resolve_run_root(args.run_root)
    paths = validate_run_structure(run_root)

    output_dir = (
        Path(args.output_dir).resolve()
        if args.output_dir.strip()
        else paths["reports_dir"] / "worklist"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Project root: %s", PROJECT_ROOT)
    logger.info("Package dir: %s", PACKAGE_DIR)
    logger.info("Run root: %s", run_root)
    logger.info("Output dir: %s", output_dir)
    logger.info("Resolved compare coverage: %s", paths["compare_coverage"])
    logger.info("Resolved equipment map: %s", paths["equipment_map"])
    logger.info("Resolved hierarchy pages: %s", paths["hierarchy_pages"])
    logger.info("Resolved master summary: %s", paths["master_summary"])

    coverage_rows = load_csv(paths["compare_coverage"])
    equipment_map_rows = load_csv(paths["equipment_map"])
    master_summary = (
        safe_load_json(paths["master_summary"])
        if paths.get("master_summary")
        else {}
    )

    worklist_rows, manual_review_rows, worklist_summary = build_worklist(
        coverage_rows=coverage_rows,
        equipment_map_rows=equipment_map_rows,
    )

    wave_1_rows = [r for r in worklist_rows if r.migration_bucket == "Wave 1"]
    wave_2_rows = [r for r in worklist_rows if r.migration_bucket == "Wave 2"]

    all_worklist_csv = output_dir / "migration_worklist_all.csv"
    wave_1_csv = output_dir / "migration_worklist_wave_1.csv"
    wave_2_csv = output_dir / "migration_worklist_wave_2.csv"
    manual_review_csv = output_dir / "migration_worklist_manual_review.csv"
    summary_json = output_dir / "migration_worklist_summary.json"

    write_csv(
        worklist_rows,
        all_worklist_csv,
        fallback_headers=list(WorklistRow.__dataclass_fields__.keys()),
    )
    write_csv(
        wave_1_rows,
        wave_1_csv,
        fallback_headers=list(WorklistRow.__dataclass_fields__.keys()),
    )
    write_csv(
        wave_2_rows,
        wave_2_csv,
        fallback_headers=list(WorklistRow.__dataclass_fields__.keys()),
    )
    write_csv(
        manual_review_rows,
        manual_review_csv,
        fallback_headers=list(ManualReviewRow.__dataclass_fields__.keys()),
    )

    payload = {
        "project_root": str(PROJECT_ROOT),
        "package_dir": str(PACKAGE_DIR),
        "run_root": str(run_root),
        "output_dir": str(output_dir),
        "source_files": {
            "coverage_comparison_csv": str(paths["compare_coverage"]),
            "document_equipment_associations_v2": str(paths["equipment_map"]),
            "site_hierarchy_pages": str(paths["hierarchy_pages"]),
            "migration_master_summary": (
                str(paths["master_summary"]) if paths.get("master_summary") else ""
            ),
        },
        "worklist_summary": worklist_summary,
        "master_summary_metrics": master_summary.get("high_level_metrics", {}),
    }
    write_json(payload, summary_json)

    print("\nMigration worklist build complete.")
    print(f"Run root: {run_root}")
    print(f"Output dir: {output_dir}")
    print(f"All worklist: {all_worklist_csv}")
    print(f"Wave 1: {wave_1_csv}")
    print(f"Wave 2: {wave_2_csv}")
    print(f"Manual review: {manual_review_csv}")
    print(f"Summary: {summary_json}")


if __name__ == "__main__":
    main()