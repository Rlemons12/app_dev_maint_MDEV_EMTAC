from __future__ import annotations

import argparse
import json
import logging
import os
import subprocess
import sys
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional


# ---------------------------------------------------------
# Ensure project root is on sys.path when running script directly
# ---------------------------------------------------------
CURRENT_FILE = Path(__file__).resolve()
PACKAGE_DIR = CURRENT_FILE.parent.resolve()
PROJECT_ROOT = CURRENT_FILE.parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from modules.configuration.config import DB_LOADSHEET


logger = logging.getLogger("run_full_migration_pipeline")


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
# Basic helpers
# =========================================================
def now_timestamp() -> str:
    return datetime.now().strftime("%Y-%m-%d_%H%M%S")


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


def safe_load_json(path: Optional[Path]) -> Dict:
    if not path or not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        logger.warning("Could not read JSON %s | reason=%s", path, exc)
        return {}


def write_json(payload: Dict, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def write_text(text: str, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(text, encoding="utf-8")


def script_path(script_name: str) -> Path:
    path = PACKAGE_DIR / script_name
    ensure_file_exists(path, f"Required script '{script_name}'")
    return path


def choose_python_executable() -> str:
    return sys.executable


def run_subprocess(
    *,
    phase_name: str,
    command: List[str],
    script: Path,
    output_dir: Path,
    logs_dir: Path,
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

    stdout_log = logs_dir / f"{phase_name}_stdout.log"
    stderr_log = logs_dir / f"{phase_name}_stderr.log"
    write_text(completed.stdout or "", stdout_log)
    write_text(completed.stderr or "", stderr_log)

    if success:
        logger.info("Phase completed successfully: %s", phase_name)
    else:
        logger.error(
            "Phase failed: %s | return_code=%s",
            phase_name,
            completed.returncode,
        )
        logger.error("stderr log: %s", stderr_log)

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


def maybe_stop(result: PhaseResult, stop_on_failure: bool) -> None:
    if stop_on_failure and not result.success:
        raise RuntimeError(
            f"Stopping pipeline because phase failed: {result.phase_name}"
        )


# =========================================================
# Run directory layout
# =========================================================
def build_run_directories(base_output: Path, run_name: str) -> Dict[str, Path]:
    run_root = base_output / "migration_analysis" / run_name
    dirs = {
        "run_root": run_root,
        "inventory": run_root / "inventory",
        "hierarchy": run_root / "hierarchy",
        "equipment_map": run_root / "equipment_map",
        "compare": run_root / "compare",
        "reports": run_root / "reports",
        "worklist": run_root / "reports" / "worklist",
        "logs": run_root / "logs",
    }
    for d in dirs.values():
        d.mkdir(parents=True, exist_ok=True)
    return dirs


# =========================================================
# Output resolution
# =========================================================
def resolve_phase_outputs(dirs: Dict[str, Path]) -> Dict[str, Optional[Path]]:
    def existing_or_none(path: Path) -> Optional[Path]:
        return path if path.exists() else None

    return {
        # inventory
        "inventory_summary": existing_or_none(
            dirs["inventory"] / "source_inventory_summary.json"
        ),
        "inventory_json": existing_or_none(
            dirs["inventory"] / "source_inventory.json"
        ),
        "source_inventory_csv": existing_or_none(
            dirs["inventory"] / "source_inventory.csv"
        ),
        "html_summary": existing_or_none(
            dirs["inventory"] / "html_map_summary.json"
        ),
        "html_json": existing_or_none(
            dirs["inventory"] / "html_map.json"
        ),

        # hierarchy
        "hierarchy_json": existing_or_none(
            dirs["hierarchy"] / "site_hierarchy.json"
        ),
        "hierarchy_summary": existing_or_none(
            dirs["hierarchy"] / "site_hierarchy_summary.json"
        ),
        "hierarchy_pages_csv": existing_or_none(
            dirs["hierarchy"] / "site_hierarchy_pages.csv"
        ),
        "hierarchy_edges_csv": existing_or_none(
            dirs["hierarchy"] / "site_hierarchy_edges.csv"
        ),

        # equipment map
        "equipment_map_csv": existing_or_none(
            dirs["equipment_map"] / "document_equipment_associations_v2.csv"
        ),
        "equipment_map_summary": existing_or_none(
            dirs["equipment_map"] / "document_equipment_summary_v2.json"
        ),

        # compare
        "compare_coverage_csv": existing_or_none(
            dirs["compare"] / "equipment_match_coverage.csv"
        ),
        "compare_matched_csv": existing_or_none(
            dirs["compare"] / "matched_equipment.csv"
        ),
        "compare_summary": existing_or_none(
            dirs["compare"] / "equipment_list_comparison_summary.json"
        ),

        # reports/worklist
        "migration_master_summary": existing_or_none(
            dirs["reports"] / "migration_master_summary.json"
        ),
        "full_pipeline_summary": existing_or_none(
            dirs["reports"] / "full_migration_pipeline_summary.json"
        ),
        "worklist_summary": existing_or_none(
            dirs["worklist"] / "migration_worklist_summary.json"
        ),
        "wave_1_csv": existing_or_none(
            dirs["worklist"] / "migration_worklist_wave_1.csv"
        ),
        "wave_2_csv": existing_or_none(
            dirs["worklist"] / "migration_worklist_wave_2.csv"
        ),
        "manual_review_csv": existing_or_none(
            dirs["worklist"] / "migration_worklist_manual_review.csv"
        ),
        "all_worklist_csv": existing_or_none(
            dirs["worklist"] / "migration_worklist_all.csv"
        ),
    }


# =========================================================
# Summary builders
# =========================================================
def build_pipeline_summary(
    *,
    args: argparse.Namespace,
    dirs: Dict[str, Path],
    phase_results: List[PhaseResult],
) -> Dict:
    generated = resolve_phase_outputs(dirs)

    inventory_summary = safe_load_json(generated["inventory_summary"])
    html_summary = safe_load_json(generated["html_summary"])
    hierarchy_summary = safe_load_json(generated["hierarchy_summary"])
    equipment_map_summary_raw = safe_load_json(generated["equipment_map_summary"])
    compare_summary_raw = safe_load_json(generated["compare_summary"])
    worklist_summary_raw = safe_load_json(generated["worklist_summary"])

    equipment_map_summary = equipment_map_summary_raw.get(
        "summary",
        equipment_map_summary_raw,
    )
    compare_summary = compare_summary_raw.get(
        "summary",
        compare_summary_raw,
    )
    worklist_summary = worklist_summary_raw.get(
        "worklist_summary",
        worklist_summary_raw,
    )

    return {
        "generated_at": datetime.now().isoformat(),
        "project_root": str(PROJECT_ROOT),
        "package_dir": str(PACKAGE_DIR),
        "root_analyzed": str(Path(args.root).resolve()),
        "seed_page": str(Path(args.seed).resolve()),
        "equipment_workbook": str(Path(args.equipment_workbook).resolve()),
        "run_root": str(dirs["run_root"]),
        "run_directories": {k: str(v) for k, v in dirs.items()},
        "phase_results": [asdict(r) for r in phase_results],
        "generated_files": {k: (str(v) if v else "") for k, v in generated.items()},
        "high_level_metrics": {
            "inventory_total_files": inventory_summary.get("total_files"),
            "inventory_total_size_mb": inventory_summary.get("total_size_mb"),
            "html_total_pages": html_summary.get("total_html_pages"),
            "html_total_relationships": html_summary.get("total_html_relationships"),
            "hierarchy_total_pages_crawled": hierarchy_summary.get(
                "total_pages_crawled"
            ),
            "hierarchy_total_edges": hierarchy_summary.get("total_edges"),
            "hierarchy_total_missing_local_references": hierarchy_summary.get(
                "total_missing_local_references"
            ),
            "equipment_map_total_records": equipment_map_summary.get("total_records"),
            "equipment_map_records_with_hierarchy_context": equipment_map_summary.get(
                "records_with_hierarchy_context"
            ),
            "compare_total_equipment_groups_compared": compare_summary.get(
                "total_equipment_groups_compared"
            ),
            "compare_matched_equipment_groups": compare_summary.get(
                "matched_equipment_groups"
            ),
            "compare_equipment_list_only_groups": compare_summary.get(
                "equipment_list_only_groups"
            ),
            "compare_content_only_groups": compare_summary.get(
                "content_only_groups"
            ),
            "worklist_total_rows": worklist_summary.get("total_worklist_rows"),
            "worklist_bucket_counts": worklist_summary.get("bucket_counts", {}),
            "worklist_wave_counts": worklist_summary.get("wave_counts", {}),
        },
        "inventory_summary": inventory_summary,
        "html_summary": html_summary,
        "hierarchy_summary": hierarchy_summary,
        "equipment_map_summary": equipment_map_summary_raw,
        "compare_summary": compare_summary_raw,
        "worklist_summary": worklist_summary_raw,
    }


def write_run_readme(output_path: Path, dirs: Dict[str, Path], summary: Dict) -> None:
    metrics = summary.get("high_level_metrics", {})
    generated = summary.get("generated_files", {})

    lines = [
        "Full Migration Pipeline Run",
        "===========================",
        "",
        f"Generated at: {summary.get('generated_at', '')}",
        f"Package dir: {summary.get('package_dir', '')}",
        f"Root analyzed: {summary.get('root_analyzed', '')}",
        f"Seed page: {summary.get('seed_page', '')}",
        f"Equipment workbook: {summary.get('equipment_workbook', '')}",
        "",
        "Run folders",
        "-----------",
        f"Run root:       {dirs['run_root']}",
        f"Inventory:      {dirs['inventory']}",
        f"Hierarchy:      {dirs['hierarchy']}",
        f"Equipment map:  {dirs['equipment_map']}",
        f"Compare:        {dirs['compare']}",
        f"Reports:        {dirs['reports']}",
        f"Worklist:       {dirs['worklist']}",
        f"Logs:           {dirs['logs']}",
        "",
        "High-level metrics",
        "------------------",
        f"Total files inventoried:         {metrics.get('inventory_total_files')}",
        f"HTML pages discovered:           {metrics.get('html_total_pages')}",
        f"Hierarchy pages crawled:         {metrics.get('hierarchy_total_pages_crawled')}",
        f"Hierarchy edges recorded:        {metrics.get('hierarchy_total_edges')}",
        f"Missing local refs:              {metrics.get('hierarchy_total_missing_local_references')}",
        f"Equipment map records:           {metrics.get('equipment_map_total_records')}",
        f"Records with hierarchy context:  {metrics.get('equipment_map_records_with_hierarchy_context')}",
        f"Equipment groups compared:       {metrics.get('compare_total_equipment_groups_compared')}",
        f"Matched equipment groups:        {metrics.get('compare_matched_equipment_groups')}",
        f"Equipment-list-only groups:      {metrics.get('compare_equipment_list_only_groups')}",
        f"Content-only groups:             {metrics.get('compare_content_only_groups')}",
        f"Worklist total rows:             {metrics.get('worklist_total_rows')}",
        f"Worklist bucket counts:          {metrics.get('worklist_bucket_counts')}",
        f"Worklist wave counts:            {metrics.get('worklist_wave_counts')}",
        "",
        "Review these first",
        "------------------",
        generated.get("compare_coverage_csv", ""),
        generated.get("compare_matched_csv", ""),
        generated.get("wave_1_csv", ""),
        generated.get("wave_2_csv", ""),
        generated.get("manual_review_csv", ""),
        generated.get("all_worklist_csv", ""),
        "",
        "Logs",
        "----",
        str(dirs["logs"]),
        "",
    ]

    output_path.write_text("\n".join(lines), encoding="utf-8")


def write_pre_worklist_master_summary(
    *,
    args: argparse.Namespace,
    dirs: Dict[str, Path],
    phase_results: List[PhaseResult],
) -> Path:
    summary = build_pipeline_summary(
        args=args,
        dirs=dirs,
        phase_results=phase_results,
    )
    output_path = dirs["reports"] / "migration_master_summary.json"
    write_json(summary, output_path)
    return output_path


# =========================================================
# Phase builders
# =========================================================
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
        logs_dir=dirs["logs"],
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
        logs_dir=dirs["logs"],
    )


def build_equipment_map_phase(
    python_exe: str,
    dirs: Dict[str, Path],
    args: argparse.Namespace,
) -> PhaseResult:
    script = script_path("build_document_equipment_map_v2.py")

    hierarchy_json = dirs["hierarchy"] / "site_hierarchy.json"
    ensure_file_exists(hierarchy_json, "site_hierarchy.json")

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
        logs_dir=dirs["logs"],
    )


def build_compare_phase(
    python_exe: str,
    dirs: Dict[str, Path],
    args: argparse.Namespace,
) -> PhaseResult:
    script = script_path("compare_active_drawing_list_to_content.py")

    content_map_csv = dirs["equipment_map"] / "document_equipment_associations_v2.csv"
    ensure_file_exists(
        content_map_csv,
        "document_equipment_associations_v2.csv",
    )

    workbook_path = Path(args.equipment_workbook).resolve()
    ensure_file_exists(workbook_path, "Equipment workbook")

    command = [
        python_exe,
        str(script),
        "--equipment-workbook",
        str(workbook_path),
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
        logs_dir=dirs["logs"],
    )


def build_worklist_phase(
    python_exe: str,
    dirs: Dict[str, Path],
    args: argparse.Namespace,
) -> PhaseResult:
    script = script_path("build_migration_worklist.py")

    command = [
        python_exe,
        str(script),
        "--run-root",
        str(dirs["run_root"]),
        "--output-dir",
        str(dirs["worklist"]),
    ]

    if args.verbose:
        command.append("--verbose")

    return run_subprocess(
        phase_name="migration_worklist",
        command=command,
        script=script,
        output_dir=dirs["worklist"],
        logs_dir=dirs["logs"],
    )


# =========================================================
# CLI
# =========================================================
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run the full migration pipeline end-to-end:\n"
            "1) source inventory\n"
            "2) site hierarchy\n"
            "3) hierarchy-aware equipment mapping\n"
            "4) equipment workbook comparison\n"
            "5) migration worklist generation"
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

    parser.add_argument(
        "--inventory-hash",
        action="store_true",
        help="Enable SHA256 hashing during inventory.",
    )
    parser.add_argument(
        "--inventory-exclude-hidden",
        action="store_true",
        help="Exclude hidden files during inventory.",
    )
    parser.add_argument(
        "--inventory-skip-html-map",
        action="store_true",
        help="Skip HTML page/link extraction during inventory.",
    )

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
        help="Optional max number of HTML pages to crawl.",
    )
    parser.add_argument(
        "--progress-every",
        type=int,
        default=100,
        help="Progress log interval for hierarchy crawl.",
    )

    parser.add_argument(
        "--skip-inventory",
        action="store_true",
        help="Skip inventory phase.",
    )
    parser.add_argument(
        "--skip-hierarchy",
        action="store_true",
        help="Skip hierarchy phase.",
    )
    parser.add_argument(
        "--skip-equipment-map",
        action="store_true",
        help="Skip equipment mapping phase.",
    )
    parser.add_argument(
        "--skip-compare",
        action="store_true",
        help="Skip compare phase.",
    )
    parser.add_argument(
        "--skip-worklist",
        action="store_true",
        help="Skip worklist phase.",
    )

    parser.add_argument(
        "--stop-on-failure",
        action="store_true",
        help="Stop immediately if any phase fails.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging.",
    )

    return parser.parse_args()


# =========================================================
# Main
# =========================================================
def main() -> None:
    args = parse_args()
    configure_logging(verbose=args.verbose)

    root_dir = Path(args.root).resolve()
    seed_page = Path(args.seed).resolve()
    workbook_path = Path(args.equipment_workbook).resolve()

    ensure_dir_exists(root_dir, "Root directory")
    ensure_file_exists(seed_page, "Seed HTML page")
    ensure_file_exists(workbook_path, "Equipment workbook")

    run_name = args.run_name.strip() or now_timestamp()

    # Outputs live under the package folder
    base_output = PACKAGE_DIR
    dirs = build_run_directories(base_output, run_name)
    python_exe = choose_python_executable()

    logger.info("Project root: %s", PROJECT_ROOT)
    logger.info("Package dir: %s", PACKAGE_DIR)
    logger.info("Root directory: %s", root_dir)
    logger.info("Seed page: %s", seed_page)
    logger.info("Equipment workbook: %s", workbook_path)
    logger.info("Run root: %s", dirs["run_root"])

    phase_results: List[PhaseResult] = []

    if not args.skip_inventory:
        result = build_inventory_phase(python_exe, dirs, args)
        phase_results.append(result)
        maybe_stop(result, args.stop_on_failure)

    if not args.skip_hierarchy:
        result = build_hierarchy_phase(python_exe, dirs, args)
        phase_results.append(result)
        maybe_stop(result, args.stop_on_failure)

    if not args.skip_equipment_map:
        result = build_equipment_map_phase(python_exe, dirs, args)
        phase_results.append(result)
        maybe_stop(result, args.stop_on_failure)

    if not args.skip_compare:
        result = build_compare_phase(python_exe, dirs, args)
        phase_results.append(result)
        maybe_stop(result, args.stop_on_failure)

    pre_worklist_summary_path = write_pre_worklist_master_summary(
        args=args,
        dirs=dirs,
        phase_results=phase_results,
    )
    logger.info(
        "Pre-worklist migration master summary written: %s",
        pre_worklist_summary_path,
    )

    if not args.skip_worklist:
        result = build_worklist_phase(python_exe, dirs, args)
        phase_results.append(result)
        maybe_stop(result, args.stop_on_failure)

    unified_summary = build_pipeline_summary(
        args=args,
        dirs=dirs,
        phase_results=phase_results,
    )

    unified_summary_path = dirs["reports"] / "full_migration_pipeline_summary.json"
    readme_path = dirs["reports"] / "README_full_pipeline.txt"

    write_json(unified_summary, unified_summary_path)
    write_run_readme(readme_path, dirs, unified_summary)

    all_success = all(r.success for r in phase_results)

    if all_success:
        print("\nFull migration pipeline complete.")
    else:
        print("\nFull migration pipeline completed with failures.")

    print(f"Run root: {dirs['run_root']}")
    print(f"Pre-worklist summary: {pre_worklist_summary_path}")
    print(f"Unified summary: {unified_summary_path}")
    print(f"Readme: {readme_path}")
    print("\nPhase results:")
    for result in phase_results:
        status = "OK" if result.success else "FAILED"
        print(
            f" - {result.phase_name}: {status} "
            f"(return_code={result.return_code})"
        )


if __name__ == "__main__":
    main()