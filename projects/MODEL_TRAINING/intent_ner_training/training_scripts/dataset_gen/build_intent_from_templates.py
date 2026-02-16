# modules/emtac_ai/training_scripts/dataset_gen/build_intent_from_templates.py
from __future__ import annotations

import argparse
import json
import os
import random
import re
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import pandas as pd

# ---- Project config paths (your config.py) ----
from modules.configuration.config import (
    ORC_INTENT_TRAIN_DATA_DIR,
    ORC_TRAINING_DATA_LOADSHEET,   # not required at runtime; kept for consistency
    ORC_QUERY_TEMPLATE_PARTS,
    ORC_QUERY_TEMPLATE_DRAWINGS,   # example of another intent dir, optional
)

# ---- Your custom logging helpers ----
from modules.configuration.log_config import (
    debug_id, info_id, warning_id, error_id,
    get_request_id,
)

# --------- Placeholder pattern like {PART_NUMBER} ----------
PLACEHOLDER_RE = re.compile(r"\{([A-Z_]+)\}")

# --------- Default column mapping (edit to match your loadsheet) ----------
# You can also pass a JSON mapping via --column-map
DEFAULT_COLUMN_MAP = {
    "PART_NUMBER": "Item Number",      # e.g., A104317
    "PART_NAME": "Description",        # part short description
    "MANUFACTURER": "Manufacturer",    # OEM/MFG
    "MODEL": "Model",                  # model string
    "OEM_NUMBER": "Mfg Part Number",   # OEM/MPN
}

# --------- File suffixes considered as template files ----------
TEMPLATE_SUFFIXES = {".txt", ".tmpl", ".templates"}


# ------------------------- Utilities -------------------------
def load_inventory(path: Optional[str], req_id: Optional[str]) -> Optional[pd.DataFrame]:
    """Load inventory file if provided. Supports .xlsx/.xls/.csv."""
    if not path:
        info_id("No inventory path provided. Will not fill placeholders.", req_id)
        return None

    p = Path(path)
    if not p.exists():
        error_id(f"Inventory file not found: {path}", req_id)
        return None

    try:
        if p.suffix.lower() in {".xlsx", ".xls"}:
            df = pd.read_excel(p)
        elif p.suffix.lower() == ".csv":
            df = pd.read_csv(p)
        else:
            error_id(f"Unsupported inventory format: {p.suffix}", req_id)
            return None
    except Exception as e:
        error_id(f"Failed to load inventory '{path}': {e}", req_id)
        return None

    # Normalize columns (strip, string)
    df.columns = [str(c).strip() for c in df.columns]
    info_id(f"Inventory loaded with {len(df)} rows and columns: {list(df.columns)}", req_id)
    return df


def get_template_files(root_dir: str, req_id: Optional[str]) -> List[Path]:
    """Recursively find template files in a directory."""
    root = Path(root_dir)
    if not root.exists():
        warning_id(f"Templates root not found: {root_dir}", req_id)
        return []
    files: List[Path] = []
    for path in root.rglob("*"):
        if path.is_file() and path.suffix.lower() in TEMPLATE_SUFFIXES:
            files.append(path)
    files.sort()
    info_id(f"Discovered {len(files)} template files under {root_dir}", req_id)
    return files


def group_template_files_by_label(root_dir: str, req_id: Optional[str]) -> Dict[str, List[Path]]:
    """
    Walk root_dir and group discovered template files by the root's first-level
    subfolder name. Example:
        query_templates/
          parts/*.txt -> label 'parts'
          drawings/*.txt -> label 'drawings'
          foo/bar/*.tmpl -> label 'foo'   (still grouped by the first segment)
    """
    root = Path(root_dir).resolve()
    groups: Dict[str, List[Path]] = {}
    if not root.exists():
        warning_id(f"Templates root not found: {root_dir}", req_id)
        return groups

    for path in root.rglob("*"):
        if not (path.is_file() and path.suffix.lower() in TEMPLATE_SUFFIXES):
            continue
        try:
            rel = path.relative_to(root)
        except Exception:
            # Defensive: if relative_to fails, treat as unlabeled
            label = "unknown"
        else:
            # label = first dir segment under root, or file stem if none
            parts = rel.parts
            label = parts[0] if len(parts) > 1 else path.stem
        groups.setdefault(label, []).append(path)

    # Stable ordering for logging/debuggability
    for label in list(groups.keys()):
        groups[label].sort()
        info_id(f"[root-scan] Label '{label}': {len(groups[label])} files", req_id)
    return groups


def read_templates(files: List[Path], req_id: Optional[str]) -> List[str]:
    """Read lines from template files, skip empty and comment lines."""
    templates: List[str] = []
    for fpath in files:
        try:
            with fpath.open("r", encoding="utf-8") as f:
                for ln in f:
                    line = ln.strip()
                    if not line or line.startswith("#"):
                        continue
                    templates.append(line)
        except Exception as e:
            warning_id(f"Failed reading template file '{fpath}': {e}", req_id)
    info_id(f"Loaded {len(templates)} template lines.", req_id)
    return templates


def find_placeholders(tmpl: str) -> List[str]:
    return list(set(PLACEHOLDER_RE.findall(tmpl)))


def _pick_values(df: pd.DataFrame, fields: Iterable[str], column_map: Dict[str, str]) -> Dict[str, str]:
    out: Dict[str, str] = {}
    for field in fields:
        col = column_map.get(field)
        if col and col in df.columns:
            candidates = df[col].dropna().astype(str)
            candidates = candidates[candidates.str.strip() != ""]
            if not candidates.empty:
                out[field] = candidates.sample(1).iloc[0].strip()
            else:
                out[field] = ""
        else:
            out[field] = ""
    return out


def fill_template(tmpl: str, values: Dict[str, str]) -> str:
    def _repl(m):
        key = m.group(1)
        return values.get(key, "")
    return PLACEHOLDER_RE.sub(_repl, tmpl).strip()


def synthesize_examples(
    templates: List[str],
    label: str,
    n_per_template: int,
    df: Optional[pd.DataFrame],
    column_map: Dict[str, str],
    skip_unknown_placeholders: bool,
    req_id: Optional[str],
) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    for tmpl in templates:
        phs = find_placeholders(tmpl)

        if phs and df is None:
            if skip_unknown_placeholders:
                warning_id(f"Skipping placeholder template (no inventory): {tmpl}", req_id)
                continue
            else:
                # Keep raw template as-is
                rows.append({"text": tmpl, "label": label})
                continue

        # If placeholders include unknown tokens not in column_map
        if phs:
            unknown = [p for p in phs if p not in column_map]
            if unknown and skip_unknown_placeholders:
                warning_id(f"Skipping template with unknown placeholders {unknown}: {tmpl}", req_id)
                continue

        # If there are placeholders and we have inventory: sample repeatedly
        if phs and df is not None:
            for _ in range(max(1, n_per_template)):
                vals = _pick_values(df, phs, column_map)
                text = fill_template(tmpl, vals)
                if text:
                    rows.append({"text": text, "label": label})
        else:
            # No placeholders: single example or replicate if you want
            rows.append({"text": tmpl, "label": label})

    info_id(f"Synthesized {len(rows)} examples for label '{label}'", req_id)
    return rows


def deduplicate(rows: List[Dict[str, str]], req_id: Optional[str]) -> List[Dict[str, str]]:
    seen = set()
    keep: List[Dict[str, str]] = []
    for r in rows:
        key = r["text"].strip().lower()
        if key and key not in seen:
            seen.add(key)
            keep.append(r)
    info_id(f"Deduped dataset: {len(rows)} -> {len(keep)}", req_id)
    return keep


def split_rows(rows: List[Dict[str, str]], train_ratio: float, val_ratio: float, seed: int) -> tuple:
    rnd = random.Random(seed)
    rnd.shuffle(rows)
    n = len(rows)
    train_n = int(n * train_ratio)
    val_n = int(n * val_ratio)
    test_n = n - train_n - val_n
    return rows[:train_n], rows[train_n:train_n+val_n], rows[train_n+val_n:]


def write_jsonl(path: Path, rows: List[Dict[str, str]], req_id: Optional[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    info_id(f"Wrote {len(rows)} rows -> {path}", req_id)


# ---------------- QUICK REPORT ----------------
def report_summary(rows: List[Dict[str, str]], sample_per_label: int, req_id: Optional[str]) -> Dict[str, object]:
    """Return and log per-label counts and a few sample texts per label."""
    from collections import Counter, defaultdict
    counts = Counter(r["label"] for r in rows)

    info_id("=== INTENT DATASET SUMMARY ===", req_id)
    for label, count in sorted(counts.items(), key=lambda x: x[0]):
        info_id(f"Label '{label}': {count} examples", req_id)

    grouped = defaultdict(list)
    for r in rows:
        if len(grouped[r["label"]]) < sample_per_label:
            grouped[r["label"]].append(r["text"])

    info_id("=== SAMPLE EXAMPLES PER LABEL ===", req_id)
    for label, samples in grouped.items():
        info_id(f"Label '{label}':", req_id)
        for ex in samples:
            info_id(f"   {ex}", req_id)

    return {"counts": dict(counts), "samples": dict(grouped)}


# ------------------------- Main -------------------------
def main():
    # Single request ID for the whole run
    req_id = get_request_id()
    info_id("Starting intent dataset build from templates.", req_id)

    parser = argparse.ArgumentParser(description="Build intent dataset from query templates.")
    parser.add_argument("--parts-templates-dir", default=ORC_QUERY_TEMPLATE_PARTS,
                        help="Directory with parts templates (defaults to config.ORC_QUERY_TEMPLATE_PARTS).")
    parser.add_argument("--inventory-path", default=None,
                        help="Excel/CSV loadsheet for filling placeholders. If omitted, templates without placeholders are used as-is.")
    parser.add_argument("--column-map", default=None,
                        help="JSON string mapping placeholders to inventory columns. If omitted, uses DEFAULT_COLUMN_MAP.")
    parser.add_argument("--samples-per-template", type=int, default=20,
                        help="How many samples to generate per template when placeholders exist (default=20).")
    parser.add_argument("--train-out-dir", default=ORC_INTENT_TRAIN_DATA_DIR,
                        help="Output directory for intent files (defaults to config.ORC_INTENT_TRAIN_DATA_DIR).")
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--skip-unknown-placeholders", action="store_true",
                        help="If set, skip templates with placeholders not in the column map.")
    parser.add_argument("--dedup", action="store_true",
                        help="If set, deduplicate final texts (case-insensitive).")
    parser.add_argument("--include-other-intents", action="store_true",
                        help="If set, also read sibling template dirs (e.g., drawings) and label them by folder name.")
    parser.add_argument("--other-template-dirs", nargs="*",
                        default=[ORC_QUERY_TEMPLATE_DRAWINGS],
                        help="List of other template directories to include if --include-other-intents is set.")
    parser.add_argument("--report-samples", type=int, default=3,
                        help="How many sample texts to show per label in the quick report (default=3).")
    parser.add_argument("--save-report", action="store_true",
                        help="If set, save a JSON report (counts+samples + split sizes) to the output directory.")
    parser.add_argument("--root-templates-dir", default=None,
                        help="If set, scan this root and include ALL subfolders as separate intent labels (label = subfolder name).")

    args = parser.parse_args()

    # Column map
    column_map = DEFAULT_COLUMN_MAP.copy()
    if args.column_map:
        try:
            column_map.update(json.loads(args.column_map))
            debug_id(f"Using custom column map: {column_map}", req_id)
        except Exception as e:
            error_id(f"Failed to parse --column-map JSON: {e}", req_id)

    # Load inventory (optional)
    df = load_inventory(args.inventory_path, req_id)

    rows: List[Dict[str, str]] = []

    # ---------- NEW: root-scanning mode (auto include ALL subfolders as labels) ----------
    if args.root_templates_dir:
        root = Path(args.root_templates_dir).resolve()
        if not root.exists():
            error_id(f"No such root: {args.root_templates_dir}", req_id)
            return

        # Group files by first-level subfolder under root (label = subfolder name)
        label_to_files: Dict[str, List[Path]] = {}
        for f in root.rglob("*"):
            if f.is_file() and f.suffix.lower() in TEMPLATE_SUFFIXES:
                try:
                    rel = f.relative_to(root)
                except Exception:
                    label = "unknown"
                else:
                    parts = rel.parts
                    label = parts[0] if len(parts) > 1 else f.stem
                label_to_files.setdefault(label, []).append(f)

        total_files = sum(len(v) for v in label_to_files.values())
        if total_files == 0:
            error_id(f"No template files found under root: {args.root_templates_dir}", req_id)
            return

        info_id(f"Discovered {total_files} template files across {len(label_to_files)} labels under {args.root_templates_dir}", req_id)

        for label, files in sorted(label_to_files.items(), key=lambda kv: kv[0]):
            info_id(f"[root-scan] Label '{label}': {len(files)} files", req_id)
            tpls = read_templates(files, req_id)  # loads non-empty, non-comment lines:contentReference[oaicite:1]{index=1}
            info_id(f"Loaded {len(tpls)} templates for intent '{label}'", req_id)
            rows.extend(
                synthesize_examples(
                    templates=tpls,
                    label=label,
                    n_per_template=args.samples_per_template,
                    df=df,
                    column_map=column_map,
                    skip_unknown_placeholders=args.skip_unknown_placeholders,
                    req_id=req_id,
                )
            )
    else:
        # ---------- ORIGINAL behavior: parts + optional other dirs ----------
        parts_files = get_template_files(args.parts_templates_dir, req_id)  # discovers .txt/.tmpl/.templates recursively:contentReference[oaicite:2]{index=2}
        if not parts_files:
            error_id(f"No template files found under: {args.parts_templates_dir}", req_id)
            return

        parts_templates = read_templates(parts_files, req_id)  # parse template lines:contentReference[oaicite:3]{index=3}
        info_id(f"Loaded {len(parts_templates)} parts templates from {args.parts_templates_dir}", req_id)

        rows = synthesize_examples(  # placeholder fill + synthesis:contentReference[oaicite:4]{index=4}
            templates=parts_templates,
            label="parts",
            n_per_template=args.samples_per_template,
            df=df,
            column_map=column_map,
            skip_unknown_placeholders=args.skip_unknown_placeholders,
            req_id=req_id,
        )

        # Optionally add other intents (label = folder name)
        if args.include_other_intents and args.other_template_dirs:
            for d in args.other_template_dirs:
                if not d:
                    continue
                files = get_template_files(d, req_id)
                if not files:
                    warning_id(f"No templates in other intent dir: {d}", req_id)
                    continue
                tpls = read_templates(files, req_id)
                label = Path(d).name  # e.g., "drawings"
                info_id(f"Loaded {len(tpls)} templates for intent '{label}' from {d}", req_id)
                rows.extend(
                    synthesize_examples(
                        templates=tpls,
                        label=label,
                        n_per_template=args.samples_per_template,
                        df=df,
                        column_map=column_map,
                        skip_unknown_placeholders=args.skip_unknown_placeholders,
                        req_id=req_id,
                    )
                )

    # Deduplicate if requested
    if args.dedup:
        before = len(rows)
        rows = deduplicate(rows, req_id)
        debug_id(f"Deduplicated total: {before} -> {len(rows)}", req_id)

    # --------- QUICK REPORT (before splitting) ---------
    summary = report_summary(rows, sample_per_label=args.report_samples, req_id=req_id)

    # Split and write
    train, val, test = split_rows(rows, args.train_ratio, args.val_ratio, args.seed)
    out_dir = Path(args.train_out_dir)
    write_jsonl(out_dir / "intent_train.jsonl", train, req_id)
    write_jsonl(out_dir / "intent_val.jsonl", val, req_id)
    write_jsonl(out_dir / "intent_test.jsonl", test, req_id)

    info_id(f"Wrote {len(train)} train, {len(val)} val, {len(test)} test to {out_dir}", req_id)

    # Optionally save a JSON report
    if args.save_report:
        report_path = out_dir / "intent_build_report.json"
        try:
            with report_path.open("w", encoding="utf-8") as f:
                json.dump({
                    "counts": summary.get("counts", {}),
                    "samples": summary.get("samples", {}),
                    "train": len(train),
                    "val": len(val),
                    "test": len(test),
                }, f, ensure_ascii=False, indent=2)
            info_id(f"Saved report to {report_path}", req_id)
        except Exception as e:
            error_id(f"Failed to write report JSON: {e}", req_id)

    info_id("Intent dataset build completed.", req_id)



if __name__ == "__main__":
    main()
