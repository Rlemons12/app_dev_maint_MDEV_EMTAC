from __future__ import annotations

import argparse
import csv
import json
import logging
import re
import sys
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional, Set


# ---------------------------------------------------------
# Ensure project root is on sys.path when running script directly
# ---------------------------------------------------------
CURRENT_FILE = Path(__file__).resolve()
PROJECT_ROOT = CURRENT_FILE.parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from modules.configuration.config import SCRIPTS_OUTPUT


logger = logging.getLogger("build_wave1_route_plan")


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
class RoutePlanRow:
    equipment_number: str
    equipment_name: str
    route_path: str
    route_name: str
    page_slug: str
    template_type: str
    template_file: str
    blueprint_name: str
    module_group: str
    side: str
    area: str
    launcher_context: str
    target_page_title: str
    recommended_sections: str
    content_file_count: int
    html_file_count: int
    document_file_count: int
    image_file_count: int
    implementation_priority: str
    implementation_notes: str
    notes: str = ""


@dataclass
class RoutePlanSummaryRow:
    module_group: str
    template_type: str
    equipment_count: int
    route_count: int
    notes: str = ""


# =========================================================
# Helpers
# =========================================================
def normalize_text(value) -> str:
    if value is None:
        return ""
    return str(value).strip()


def safe_int(value, default: int = 0) -> int:
    try:
        if value is None or value == "":
            return default
        return int(float(value))
    except Exception:
        return default


def load_csv(path: Path) -> List[Dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(f"CSV not found: {path}")

    rows: List[Dict[str, str]] = []
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(
                {
                    (k.strip() if isinstance(k, str) else k): (v.strip() if isinstance(v, str) else v)
                    for k, v in row.items()
                }
            )
    return rows


def write_csv(rows: List[object], output_path: Path, fallback_headers: List[str]) -> None:
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


def slugify(value: str) -> str:
    value = normalize_text(value).lower()
    value = value.replace("&", " and ")
    value = re.sub(r"[^a-z0-9]+", "-", value)
    value = re.sub(r"-+", "-", value).strip("-")
    return value


def first_non_blank(*values: str) -> str:
    for value in values:
        v = normalize_text(value)
        if v:
            return v
    return ""


# =========================================================
# Planning rules
# =========================================================
def infer_module_group(side: str, area: str, launcher_context: str) -> str:
    launcher = normalize_text(launcher_context).lower()
    side_norm = normalize_text(side).lower()
    area_norm = normalize_text(area).lower()

    if area_norm:
        return area_norm
    if side_norm:
        return side_norm
    if "wet" in launcher:
        return "wetside"
    if "dry" in launcher:
        return "dryside"
    return "general"


def infer_template_type(
    html_file_count: int,
    document_file_count: int,
    image_file_count: int,
) -> str:
    if html_file_count >= 10 and document_file_count >= 5:
        return "equipment_dashboard"
    if document_file_count >= 10 and html_file_count < 10:
        return "equipment_document_library"
    if image_file_count >= 10 and document_file_count < 5:
        return "equipment_media_page"
    return "equipment_dashboard"


def infer_template_file(template_type: str) -> str:
    mapping = {
        "equipment_dashboard": "equipment/dashboard.html",
        "equipment_document_library": "equipment/document_library.html",
        "equipment_media_page": "equipment/media_page.html",
    }
    return mapping.get(template_type, "equipment/dashboard.html")


def infer_blueprint_name(module_group: str) -> str:
    return f"{slugify(module_group)}_equipment_bp"


def infer_route_name(module_group: str, equipment_number: str) -> str:
    return f"{slugify(module_group)}_{slugify(equipment_number)}"


def infer_route_path(module_group: str, equipment_number: str) -> str:
    return f"/equipment/{slugify(module_group)}/{slugify(equipment_number)}"


def infer_target_page_title(equipment_number: str, equipment_name: str) -> str:
    if equipment_name:
        return f"{equipment_number} - {equipment_name}"
    return equipment_number


def infer_sections(
    template_type: str,
    html_file_count: int,
    document_file_count: int,
    image_file_count: int,
) -> List[str]:
    sections = ["overview"]

    if template_type == "equipment_dashboard":
        sections.extend(["drawings", "documents", "media", "troubleshooting"])
    elif template_type == "equipment_document_library":
        sections.extend(["documents", "drawings"])
    elif template_type == "equipment_media_page":
        sections.extend(["media", "documents"])

    if html_file_count > 0 and "training" not in sections:
        sections.append("training")

    if image_file_count > 0 and "images" not in sections:
        sections.append("images")

    if document_file_count > 0 and "references" not in sections:
        sections.append("references")

    # de-dupe preserving order
    final_sections: List[str] = []
    seen: Set[str] = set()
    for section in sections:
        if section not in seen:
            seen.add(section)
            final_sections.append(section)

    return final_sections


def infer_priority(content_file_count: int, html_file_count: int, document_file_count: int) -> str:
    if content_file_count >= 250 or html_file_count >= 25:
        return "P1"
    if content_file_count >= 100 or document_file_count >= 25:
        return "P2"
    return "P3"


def build_implementation_notes(
    template_type: str,
    launcher_context: str,
    content_file_count: int,
    html_file_count: int,
) -> str:
    notes: List[str] = []

    if launcher_context:
        notes.append(f"Use launcher context '{launcher_context}' for navigation breadcrumbs.")

    if template_type == "equipment_dashboard":
        notes.append("Build as primary equipment landing page with modular sections.")
    elif template_type == "equipment_document_library":
        notes.append("Focus on structured document listings and drawing references.")
    elif template_type == "equipment_media_page":
        notes.append("Focus on image/media-heavy layout with supporting documents.")

    if html_file_count >= 10:
        notes.append("High HTML count suggests reusable legacy page content should be collapsed into tabs/panels.")

    if content_file_count >= 100:
        notes.append("High content volume; validate deduplication before migration.")

    return " ".join(notes)


# =========================================================
# Main transformation
# =========================================================
def build_route_plan(
    wave1_rows: List[Dict[str, str]],
) -> tuple[List[RoutePlanRow], List[RoutePlanSummaryRow], Dict]:
    route_rows: List[RoutePlanRow] = []

    for row in wave1_rows:
        equipment_number = normalize_text(row.get("equipment_number", ""))
        equipment_name = normalize_text(row.get("equipment_name", ""))
        content_file_count = safe_int(row.get("content_file_count", 0))
        html_file_count = safe_int(row.get("html_file_count", 0))
        document_file_count = safe_int(row.get("document_file_count", 0))
        image_file_count = safe_int(row.get("image_file_count", 0))

        side = normalize_text(row.get("top_sides", ""))
        area = normalize_text(row.get("top_areas", ""))
        launcher_context = normalize_text(row.get("top_launcher_ancestors", ""))

        module_group = infer_module_group(side, area, launcher_context)
        template_type = infer_template_type(
            html_file_count=html_file_count,
            document_file_count=document_file_count,
            image_file_count=image_file_count,
        )
        template_file = infer_template_file(template_type)
        blueprint_name = infer_blueprint_name(module_group)
        route_name = infer_route_name(module_group, equipment_number)
        route_path = infer_route_path(module_group, equipment_number)
        page_slug = slugify(equipment_number)
        target_page_title = infer_target_page_title(equipment_number, equipment_name)
        sections = infer_sections(
            template_type=template_type,
            html_file_count=html_file_count,
            document_file_count=document_file_count,
            image_file_count=image_file_count,
        )
        implementation_priority = infer_priority(
            content_file_count=content_file_count,
            html_file_count=html_file_count,
            document_file_count=document_file_count,
        )
        implementation_notes = build_implementation_notes(
            template_type=template_type,
            launcher_context=launcher_context,
            content_file_count=content_file_count,
            html_file_count=html_file_count,
        )

        route_rows.append(
            RoutePlanRow(
                equipment_number=equipment_number,
                equipment_name=equipment_name,
                route_path=route_path,
                route_name=route_name,
                page_slug=page_slug,
                template_type=template_type,
                template_file=template_file,
                blueprint_name=blueprint_name,
                module_group=module_group,
                side=side,
                area=area,
                launcher_context=launcher_context,
                target_page_title=target_page_title,
                recommended_sections=" | ".join(sections),
                content_file_count=content_file_count,
                html_file_count=html_file_count,
                document_file_count=document_file_count,
                image_file_count=image_file_count,
                implementation_priority=implementation_priority,
                implementation_notes=implementation_notes,
                notes="",
            )
        )

    route_rows.sort(key=lambda r: (r.implementation_priority, -r.content_file_count, r.equipment_number))

    # summary rows
    grouped: Dict[tuple[str, str], Dict[str, object]] = {}
    for row in route_rows:
        key = (row.module_group, row.template_type)
        if key not in grouped:
            grouped[key] = {
                "equipment_numbers": set(),
                "route_paths": set(),
            }
        grouped[key]["equipment_numbers"].add(row.equipment_number)
        grouped[key]["route_paths"].add(row.route_path)

    summary_rows: List[RoutePlanSummaryRow] = []
    for (module_group, template_type), data in sorted(grouped.items()):
        summary_rows.append(
            RoutePlanSummaryRow(
                module_group=module_group,
                template_type=template_type,
                equipment_count=len(data["equipment_numbers"]),
                route_count=len(data["route_paths"]),
                notes="",
            )
        )

    summary = {
        "total_wave1_equipment": len(route_rows),
        "template_type_counts": {},
        "module_group_counts": {},
        "priority_counts": {},
    }

    for row in route_rows:
        summary["template_type_counts"][row.template_type] = summary["template_type_counts"].get(row.template_type, 0) + 1
        summary["module_group_counts"][row.module_group] = summary["module_group_counts"].get(row.module_group, 0) + 1
        summary["priority_counts"][row.implementation_priority] = summary["priority_counts"].get(row.implementation_priority, 0) + 1

    return route_rows, summary_rows, summary


# =========================================================
# Path resolution
# =========================================================
def resolve_input_paths(run_root: Optional[Path], wave1_csv: Optional[Path]) -> tuple[Path, Path]:
    if wave1_csv:
        if not wave1_csv.exists():
            raise FileNotFoundError(f"Wave 1 CSV not found: {wave1_csv}")
        base_output = wave1_csv.parent
        return wave1_csv, base_output

    if not run_root:
        raise ValueError("Either --run-root or --wave1-csv must be provided.")

    if not run_root.exists():
        raise FileNotFoundError(f"Run root not found: {run_root}")

    candidates = [
        run_root / "reports" / "worklist" / "migration_worklist_wave_1.csv",
        run_root / "worklist" / "migration_worklist_wave_1.csv",
    ]

    for candidate in candidates:
        if candidate.exists():
            return candidate, candidate.parent

    raise FileNotFoundError(
        "Could not find migration_worklist_wave_1.csv. Tried:\n" + "\n".join(str(c) for c in candidates)
    )


# =========================================================
# CLI
# =========================================================
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Turn Wave 1 migration worklist into a route/template migration plan."
    )
    parser.add_argument(
        "--run-root",
        type=str,
        default="",
        help="Path to one migration_analysis/<run_name> folder.",
    )
    parser.add_argument(
        "--wave1-csv",
        type=str,
        default="",
        help="Optional direct path to migration_worklist_wave_1.csv",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="",
        help="Optional output directory. Defaults next to the Wave 1 CSV.",
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

    run_root = Path(args.run_root).resolve() if args.run_root.strip() else None
    wave1_csv = Path(args.wave1_csv).resolve() if args.wave1_csv.strip() else None

    wave1_input_csv, default_output_dir = resolve_input_paths(run_root, wave1_csv)

    output_dir = (
        Path(args.output_dir).resolve()
        if args.output_dir.strip()
        else default_output_dir / "route_plan"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Wave 1 CSV: %s", wave1_input_csv)
    logger.info("Output dir: %s", output_dir)

    wave1_rows = load_csv(wave1_input_csv)

    route_rows, summary_rows, summary = build_route_plan(wave1_rows)

    route_plan_csv = output_dir / "wave1_route_template_plan.csv"
    route_plan_summary_csv = output_dir / "wave1_route_template_summary.csv"
    route_plan_json = output_dir / "wave1_route_template_plan.json"

    write_csv(
        route_rows,
        route_plan_csv,
        fallback_headers=list(RoutePlanRow.__dataclass_fields__.keys()),
    )
    write_csv(
        summary_rows,
        route_plan_summary_csv,
        fallback_headers=list(RoutePlanSummaryRow.__dataclass_fields__.keys()),
    )
    write_json(
        {
            "input_wave1_csv": str(wave1_input_csv),
            "output_dir": str(output_dir),
            "summary": summary,
            "routes": [asdict(r) for r in route_rows],
            "module_summary": [asdict(r) for r in summary_rows],
        },
        route_plan_json,
    )

    print("\nWave 1 route/template plan complete.")
    print(f"Input Wave 1 CSV: {wave1_input_csv}")
    print(f"Output dir: {output_dir}")
    print(f"Route plan CSV: {route_plan_csv}")
    print(f"Summary CSV: {route_plan_summary_csv}")
    print(f"Plan JSON: {route_plan_json}")


if __name__ == "__main__":
    main()