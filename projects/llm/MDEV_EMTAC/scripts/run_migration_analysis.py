from __future__ import annotations

import argparse
import json
import logging
import os
import subprocess
import sys
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional


# ---------------------------------------------------------
# Ensure project root is on sys.path when running script directly
# ---------------------------------------------------------
CURRENT_FILE = Path(__file__).resolve()
PROJECT_ROOT = CURRENT_FILE.parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from modules.configuration.config import DB_LOADSHEET, SCRIPTS_OUTPUT


logger = logging.getLogger("run_migration_analysis")


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
class PhaseResult:
    phase_name: str
    script_path: str
    command: List[str]
    success: bool
    return_code: int
    stdout: str
    stderr: str
    output_dir: str
    notes: str = ""


# =========================================================
# Helpers
# =========================================================
def now_timestamp() -> str:
    return datetime.now().strftime("%Y-%m-%d_%H%M%S")


def safe_load_json(json_path: Path) -> Dict:
    if not json_path.exists():
        return {}
    try:
        return json.loads(json_path.read_text(encoding="utf-8"))
    except Exception as exc:
        logger.warning("Could not read JSON file %s | reason=%s", json_path, exc)
        return {}


def ensure_file_exists(path: Path, label: str) -> None:
    if not path.exists():
        raise FileNotFoundError(f"{label} not found: {path}")
    if not path.is_file():
        raise FileNotFoundError(f"{label} is not a file: {path}")


def ensure_dir_exists(path: Path, label: str) -> None:
    if not path.exists():
        raise FileNotFoundError(f"{label} not found: {path}")
    if not path.is_dir():
        raise NotADirectoryError(f"{label} is not a directory: {path}")


def script_path(script_name: str) -> Path:
    path = PROJECT_ROOT / "scripts" / script_name
    ensure_file_exists(path, f"Required script '{script_name}'")
    return path


def run_subprocess(
    *,
    phase_name: str,
    command: List[str],
    script: Path,
    output_dir: Path,
) -> PhaseResult:
    logger.info("Starting phase: %s", phase_name)
    logger.info("Command: %s", " ".join(command))

    completed = subprocess.run(
        command,
        cwd=str(PROJECT_ROOT),
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )

    success = completed.returncode == 0

    if success:
        logger.info("Phase completed successfully: %s", phase_name)
    else:
        logger.error("Phase failed: %s | return_code=%s", phase_name, completed.returncode)
        logger.error("stderr:\n%s", completed.stderr)

    return PhaseResult(
        phase_name=phase_name,
        script_path=str(script),
        command=command,
        success=success,
        return_code=completed.returncode,
        stdout=completed.stdout,
        stderr=completed.stderr,
        output_dir=str(output_dir),
        notes="",
    )


def write_json(payload: Dict, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def choose_python_executable() -> str:
    return sys.executable


# =========================================================
# Orchestration
# =========================================================
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run the full migration analysis pipeline:\n"
            "1) source inventory\n"
            "2) site hierarchy\n"
            "3) hierarchy-aware equipment mapping\n"
            "4) workbook-to-content comparison"
        )
    )
    parser.add_argument(
        "--root",
        type=str,
        required=True,
        help="Root folder to analyze.",
    )
    parser.add_argument(
        "--seed",
        type=str,
        required=True,
        help="Seed HTML page for hierarchy crawl.",
    )
    parser.add_argument(
        "--equipment-workbook",
        type=str,
        default=os.path.join(DB_LOADSHEET, "Active Drawing List.xlsx"),
        help="Path to Active Drawing List.xlsx",
    )
    parser.add_argument(
        "--run-name",
        type=str,
        default="",
        help="Optional custom run name. If omitted, a timestamp is used.",
    )

    # Inventory options
    parser.add_argument(
        "--inventory-hash",
        action="store_true",
        help="Enable SHA256 hashing during source inventory.",
    )
    parser.add_argument(
        "--inventory-exclude-hidden",
        action="store_true",
        help="Exclude hidden files during source inventory.",
    )
    parser.add_argument(
        "--inventory-skip-html-map",
        action="store_true",
        help="Skip HTML page/link map during source inventory.",
    )

    # Hierarchy options
    parser.add_argument(
        "--max-depth",
        type=int,
        default=None,
        help="Optional max depth for hierarchy crawl.",
    )
    parser.add_argument(
        "--max-pages",
        type=int,
        default=None,
        help="Optional max number of HTML pages for hierarchy crawl.",
    )
    parser.add_argument(
        "--progress-every",
        type=int,
        default=100,
        help="Progress log interval for hierarchy crawl.",
    )

    # Phase controls
    parser.add_argument(
        "--skip-inventory",
        action="store_true",
        help="Skip source inventory phase.",
    )
    parser.add_argument(
        "--skip-hierarchy",
        action="store_true",
        help="Skip site hierarchy phase.",
    )
    parser.add_argument(
        "--skip-equipment-map",
        action="store_true",
        help="Skip hierarchy-aware equipment mapping phase.",
    )
    parser.add_argument(
        "--skip-compare",
        action="store_true",
        help="Skip equipment workbook comparison phase.",
    )

    parser.add_argument(
        "--stop-on-failure",
        action="store_true",
        help="Stop pipeline immediately if any phase fails.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging.",
    )

    return parser.parse_args()


def build_run_directories(base_output: Path, run_name: str) -> Dict[str, Path]:
    run_root = base_output / "migration_analysis" / run_name
    dirs = {
        "run_root": run_root,
        "inventory": run_root / "inventory",
        "hierarchy": run_root / "hierarchy",
        "equipment_map": run_root / "equipment_map",
        "compare": run_root / "compare",
        "reports": run_root / "reports",
    }
    for d in dirs.values():
        d.mkdir(parents=True, exist_ok=True)
    return dirs


def build_inventory_phase(
    python_exe: str,
    dirs: Dict[str, Path],
    args: argparse.Namespace,
) -> PhaseResult:
    script = script_path("build_source_inventory.py")

    command = [
        python_exe,
        str(script),
        args.root,
        "--output-dir",
        str(dirs["inventory"]),
    ]

    if args.inventory_hash:
        command.append("--hash")
    if args.inventory_exclude_hidden:
        command.append("--exclude-hidden")
    if args.inventory_skip_html_map:
        command.append("--skip-html-map")
    if args.verbose:
        command.append("--verbose")

    return run_subprocess(
        phase_name="source_inventory",
        command=command,
        script=script,
        output_dir=dirs["inventory"],
    )


def build_hierarchy_phase(
    python_exe: str,
    dirs: Dict[str, Path],
    args: argparse.Namespace,
) -> PhaseResult:
    script = script_path("build_site_hierarchy.py")

    command = [
        python_exe,
        str(script),
        args.seed,
        "--root",
        args.root,
        "--output-dir",
        str(dirs["hierarchy"]),
        "--progress-every",
        str(args.progress_every),
    ]

    if args.max_depth is not None:
        command.extend(["--max-depth", str(args.max_depth)])
    if args.max_pages is not None:
        command.extend(["--max-pages", str(args.max_pages)])
    if args.verbose:
        command.append("--verbose")

    return run_subprocess(
        phase_name="site_hierarchy",
        command=command,
        script=script,
        output_dir=dirs["hierarchy"],
    )


def build_equipment_map_phase(
    python_exe: str,
    dirs: Dict[str, Path],
    args: argparse.Namespace,
) -> PhaseResult:
    script = script_path("build_document_equipment_map_v2.py")
    hierarchy_json = dirs["hierarchy"] / "site_hierarchy.json"

    ensure_file_exists(hierarchy_json, "Required hierarchy JSON for equipment map")

    command = [
        python_exe,
        str(script),
        args.root,
        "--hierarchy-json",
        str(hierarchy_json),
        "--output-dir",
        str(dirs["equipment_map"]),
    ]

    if args.verbose:
        command.append("--verbose")

    return run_subprocess(
        phase_name="document_equipment_map_v2",
        command=command,
        script=script,
        output_dir=dirs["equipment_map"],
    )


def build_compare_phase(
    python_exe: str,
    dirs: Dict[str, Path],
    args: argparse.Namespace,
) -> PhaseResult:
    script = script_path("compare_active_drawing_list_to_content.py")
    content_map_csv = dirs["equipment_map"] / "document_equipment_associations_v2.csv"

    ensure_file_exists(content_map_csv, "Required equipment association CSV for compare phase")
    ensure_file_exists(Path(args.equipment_workbook), "Equipment workbook")

    command = [
        python_exe,
        str(script),
        "--equipment-workbook",
        args.equipment_workbook,
        "--content-map",
        str(content_map_csv),
        "--output-dir",
        str(dirs["compare"]),
    ]

    if args.verbose:
        command.append("--verbose")

    return run_subprocess(
        phase_name="equipment_workbook_compare",
        command=command,
        script=script,
        output_dir=dirs["compare"],
    )


def build_master_summary(
    *,
    args: argparse.Namespace,
    dirs: Dict[str, Path],
    phase_results: List[PhaseResult],
) -> Dict:
    inventory_summary = safe_load_json(dirs["inventory"] / "source_inventory_summary.json")
    html_summary = safe_load_json(dirs["inventory"] / "html_map_summary.json")
    hierarchy_summary = safe_load_json(dirs["hierarchy"] / "site_hierarchy_summary.json")
    equipment_map_summary = safe_load_json(dirs["equipment_map"] / "document_equipment_summary_v2.json")
    compare_summary = safe_load_json(dirs["compare"] / "equipment_list_comparison_summary.json")

    # Some of your scripts write summary JSON under {"summary": {...}}
    equipment_map_summary_inner = equipment_map_summary.get("summary", equipment_map_summary)
    compare_summary_inner = compare_summary.get("summary", compare_summary)

    return {
        "generated_at": datetime.now().isoformat(),
        "project_root": str(PROJECT_ROOT),
        "root_analyzed": str(Path(args.root).resolve()),
        "seed_page": str(Path(args.seed).resolve()),
        "equipment_workbook": str(Path(args.equipment_workbook).resolve()),
        "run_directories": {k: str(v) for k, v in dirs.items()},
        "phase_results": [asdict(r) for r in phase_results],
        "high_level_metrics": {
            "inventory_total_files": inventory_summary.get("total_files"),
            "inventory_total_size_mb": inventory_summary.get("total_size_mb"),
            "html_total_pages": html_summary.get("total_html_pages"),
            "html_total_relationships": html_summary.get("total_html_relationships"),
            "hierarchy_total_pages_crawled": hierarchy_summary.get("total_pages_crawled"),
            "hierarchy_total_edges": hierarchy_summary.get("total_edges"),
            "hierarchy_total_missing_local_references": hierarchy_summary.get("total_missing_local_references"),
            "equipment_map_total_records": equipment_map_summary_inner.get("total_records"),
            "equipment_map_records_with_hierarchy_context": equipment_map_summary_inner.get("records_with_hierarchy_context"),
            "compare_total_equipment_groups_compared": compare_summary_inner.get("total_equipment_groups_compared"),
            "compare_matched_equipment_groups": compare_summary_inner.get("matched_equipment_groups"),
            "compare_equipment_list_only_groups": compare_summary_inner.get("equipment_list_only_groups"),
            "compare_content_only_groups": compare_summary_inner.get("content_only_groups"),
        },
        "inventory_summary": inventory_summary,
        "html_summary": html_summary,
        "hierarchy_summary": hierarchy_summary,
        "equipment_map_summary": equipment_map_summary,
        "compare_summary": compare_summary,
    }


def write_run_readme(
    output_path: Path,
    dirs: Dict[str, Path],
    master_summary: Dict,
) -> None:
    metrics = master_summary.get("high_level_metrics", {})

    lines = [
        "Migration Analysis Run",
        "======================",
        "",
        f"Generated at: {master_summary.get('generated_at', '')}",
        f"Root analyzed: {master_summary.get('root_analyzed', '')}",
        f"Seed page: {master_summary.get('seed_page', '')}",
        f"Equipment workbook: {master_summary.get('equipment_workbook', '')}",
        "",
        "Output folders",
        "--------------",
        f"Inventory:      {dirs['inventory']}",
        f"Hierarchy:      {dirs['hierarchy']}",
        f"Equipment map:  {dirs['equipment_map']}",
        f"Comparison:     {dirs['compare']}",
        f"Reports:        {dirs['reports']}",
        "",
        "High-level metrics",
        "------------------",
        f"Total files inventoried:           {metrics.get('inventory_total_files')}",
        f"HTML pages discovered:             {metrics.get('html_total_pages')}",
        f"Hierarchy pages crawled:           {metrics.get('hierarchy_total_pages_crawled')}",
        f"Hierarchy edges recorded:          {metrics.get('hierarchy_total_edges')}",
        f"Missing local references:          {metrics.get('hierarchy_total_missing_local_references')}",
        f"Equipment map total records:       {metrics.get('equipment_map_total_records')}",
        f"Records with hierarchy context:    {metrics.get('equipment_map_records_with_hierarchy_context')}",
        f"Equipment groups compared:         {metrics.get('compare_total_equipment_groups_compared')}",
        f"Matched equipment groups:          {metrics.get('compare_matched_equipment_groups')}",
        f"Equipment-list-only groups:        {metrics.get('compare_equipment_list_only_groups')}",
        f"Content-only groups:               {metrics.get('compare_content_only_groups')}",
        "",
        "Key files to review first",
        "-------------------------",
        f"{dirs['reports'] / 'migration_master_summary.json'}",
        f"{dirs['compare'] / 'equipment_match_coverage.csv'}",
        f"{dirs['compare'] / 'matched_equipment.csv'}",
        f"{dirs['compare'] / 'equipment_list_missing_content.csv'}",
        f"{dirs['compare'] / 'content_not_in_equipment_list.csv'}",
        f"{dirs['equipment_map'] / 'document_equipment_associations_v2.csv'}",
        f"{dirs['hierarchy'] / 'site_hierarchy_pages.csv'}",
        "",
    ]

    output_path.write_text("\n".join(lines), encoding="utf-8")


def maybe_stop(phase_result: PhaseResult, stop_on_failure: bool) -> None:
    if stop_on_failure and not phase_result.success:
        raise RuntimeError(
            f"Stopping pipeline because phase failed: {phase_result.phase_name}"
        )


def main() -> None:
    args = parse_args()
    configure_logging(verbose=args.verbose)

    root_dir = Path(args.root).resolve()
    seed_page = Path(args.seed).resolve()
    workbook_path = Path(args.equipment_workbook).resolve()

    ensure_dir_exists(root_dir, "Root directory")
    ensure_file_exists(seed_page, "Seed HTML page")

    run_name = args.run_name.strip() or now_timestamp()
    base_output = Path(SCRIPTS_OUTPUT).resolve()
    dirs = build_run_directories(base_output, run_name)
    python_exe = choose_python_executable()

    logger.info("Project root: %s", PROJECT_ROOT)
    logger.info("Root directory: %s", root_dir)
    logger.info("Seed page: %s", seed_page)
    logger.info("Equipment workbook: %s", workbook_path)
    logger.info("Run root: %s", dirs["run_root"])

    phase_results: List[PhaseResult] = []

    # -----------------------------------------------------
    # Phase 1 - Source inventory
    # -----------------------------------------------------
    if not args.skip_inventory:
        result = build_inventory_phase(python_exe, dirs, args)
        phase_results.append(result)
        maybe_stop(result, args.stop_on_failure)

    # -----------------------------------------------------
    # Phase 2 - Site hierarchy
    # -----------------------------------------------------
    if not args.skip_hierarchy:
        result = build_hierarchy_phase(python_exe, dirs, args)
        phase_results.append(result)
        maybe_stop(result, args.stop_on_failure)

    # -----------------------------------------------------
    # Phase 3 - Equipment map v2
    # -----------------------------------------------------
    if not args.skip_equipment_map:
        result = build_equipment_map_phase(python_exe, dirs, args)
        phase_results.append(result)
        maybe_stop(result, args.stop_on_failure)

    # -----------------------------------------------------
    # Phase 4 - Workbook compare
    # -----------------------------------------------------
    if not args.skip_compare:
        result = build_compare_phase(python_exe, dirs, args)
        phase_results.append(result)
        maybe_stop(result, args.stop_on_failure)

    master_summary = build_master_summary(
        args=args,
        dirs=dirs,
        phase_results=phase_results,
    )

    master_summary_path = dirs["reports"] / "migration_master_summary.json"
    readme_path = dirs["reports"] / "README.txt"

    write_json(master_summary, master_summary_path)
    write_run_readme(readme_path, dirs, master_summary)

    print("\nMigration analysis pipeline complete.")
    print(f"Run root: {dirs['run_root']}")
    print(f"Master summary: {master_summary_path}")
    print(f"Readme: {readme_path}")
    print("\nPhase results:")
    for result in phase_results:
        status = "OK" if result.success else "FAILED"
        print(f" - {result.phase_name}: {status} (return_code={result.return_code})")


if __name__ == "__main__":
    main()