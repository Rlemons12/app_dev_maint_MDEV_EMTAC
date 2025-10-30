import os
import argparse
import json
import random
import pandas as pd
import ast
from pathlib import Path
from tqdm import tqdm
import multiprocessing as mp
from collections import Counter

from modules.configuration.config import (
    ORC_INTENT_TRAIN_DATA_DIR,
    ORC_TRAINING_DATA_PARTS_LOADSHEET_PATH,
    ORC_TRAINING_DATA_DRAWINGS_LOADSHEET_PATH,
    ORC_QUERY_TEMPLATE_PARTS,
    ORC_QUERY_TEMPLATE_DRAWINGS,
)

# --- Alias map ---
PLACEHOLDER_ALIASES = {
    "MANUFACTURER": "OEMMFG",
    "MANUFACTURERS": "OEMMFG",
    "OEM": "OEMMFG",
    "ITEMNUM_CONTEXTUAL": "ITEMNUM",
    "ITEMNUM": "ITEMNUM",
    "MODELNUM": "MODEL",
    "MODEL": "MODEL",
    "EQUIPMENT": "EQUIPMENT_NAME",
    "EQUIPMENTNUMBER": "EQUIPMENT_NUMBER",
    "DRAWINGNUMBER": "DRAWING_NUMBER",
    "DRAWINGNAME": "DRAWING_NAME",
}

# --- Helpers ---
def load_templates_from_txt(py_file: Path, var_name: str) -> list[str]:
    if not os.path.exists(py_file):
        raise FileNotFoundError(f"Template file not found: {py_file}")
    with open(py_file, "r", encoding="utf-8") as f:
        text = f.read()
    cleaned = "\n".join([line for line in text.splitlines() if not line.strip().startswith("#")])
    if "=" in cleaned.splitlines()[0]:
        cleaned = cleaned.split("=", 1)[1].strip()
    variations = ast.literal_eval(cleaned)
    templates = []
    for _, categories in variations.items():
        if isinstance(categories, dict):
            for _, phrases in categories.items():
                templates.extend(phrases)
    return templates

def expand_row_with_templates(row: pd.Series, templates: list[str], label: str) -> list[dict]:
    samples = []
    norm_row = {col.strip().upper().replace(" ", "_"): val for col, val in row.items()}
    aliased_row = {}
    for alias, target_col in PLACEHOLDER_ALIASES.items():
        if target_col.upper() in norm_row:
            aliased_row[alias.upper()] = norm_row[target_col.upper()]
    full_row = {**norm_row, **aliased_row}
    for tmpl in templates:
        text = tmpl
        for col, val in full_row.items():
            val_str = str(val) if pd.notna(val) else ""
            text = text.replace(f"[{col}]", val_str)
            text = text.replace(f"{{{col.lower()}}}", val_str)
            text = text.replace(f"{{{col.upper()}}}", val_str)
        if "{" not in text and "[" not in text:
            samples.append({"text": text, "label": label, "source": label})
    return samples

# --- Chunking ---
def chunk_dataframe(df, chunk_size=5000):
    for i in range(0, len(df), chunk_size):
        yield df.iloc[i:i+chunk_size]

def process_chunk(args):
    df, templates, label = args
    out = []
    counts = Counter()
    for _, row in df.iterrows():
        expanded = expand_row_with_templates(row, templates, label)
        out.extend(expanded)
        counts[label] += len(expanded)
    return out, counts, label, len(df)

# --- Build dataset ---
def build_dataset(sample_size: int, workers: int, chunk_size: int = 5000):
    parts_templates = load_templates_from_txt(
        Path(ORC_QUERY_TEMPLATE_PARTS) / "PARTS_NATURAL_LANGUAGE_VARIATIONS.txt",
        "PARTS_NATURAL_LANGUAGE_VARIATIONS"
    )
    drawings_templates = load_templates_from_txt(
        Path(ORC_QUERY_TEMPLATE_DRAWINGS) / "DRAWINGS_NATURAL_LANGUAGE_VARIATIONS.txt",
        "DRAWING_NATURAL_LANGUAGE_VARIATIONS"
    )
    parts_df = pd.read_excel(ORC_TRAINING_DATA_PARTS_LOADSHEET_PATH)
    drawings_df = pd.read_excel(ORC_TRAINING_DATA_DRAWINGS_LOADSHEET_PATH)

    if sample_size > 0:
        parts_df = parts_df.sample(n=min(sample_size, len(parts_df)), random_state=42)
        drawings_df = drawings_df.sample(n=min(sample_size, len(drawings_df)), random_state=42)

    tasks = []
    if not parts_df.empty:
        for chunk in chunk_dataframe(parts_df, chunk_size):
            tasks.append((chunk, parts_templates, "parts"))
    if not drawings_df.empty:
        for chunk in chunk_dataframe(drawings_df, chunk_size):
            tasks.append((chunk, drawings_templates, "drawings"))

    total_rows = sum(len(c[0]) for c in tasks)
    dataset, counts = [], Counter()

    # Progress bars
    overall_bar = tqdm(total=total_rows, desc="Overall", position=0, dynamic_ncols=True)
    parts_bar = tqdm(total=len(parts_df), desc="Parts", position=1, dynamic_ncols=True) if not parts_df.empty else None
    drawings_bar = tqdm(total=len(drawings_df), desc="Drawings", position=2, dynamic_ncols=True) if not drawings_df.empty else None

    with mp.Pool(processes=workers) as pool:
        for rows, label_counts, label, rows_processed in pool.imap_unordered(process_chunk, tasks):
            dataset.extend(rows)
            counts.update(label_counts)
            overall_bar.update(rows_processed)
            if label == "parts" and parts_bar:
                parts_bar.update(rows_processed)
            elif label == "drawings" and drawings_bar:
                drawings_bar.update(rows_processed)

    overall_bar.close()
    if parts_bar: parts_bar.close()
    if drawings_bar: drawings_bar.close()

    random.shuffle(dataset)
    return dataset, counts

def split_and_save(dataset: list[dict], out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    n = len(dataset)
    n_train, n_val = int(n*0.8), int(n*0.1)
    splits = {
        "intent_train.jsonl": dataset[:n_train],
        "intent_val.jsonl": dataset[n_train:n_train+n_val],
        "intent_test.jsonl": dataset[n_train+n_val:],
    }
    for fname, rows in splits.items():
        fpath = out_dir / fname
        with open(fpath, "w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(f"💾 Wrote {len(rows):,} rows → {fpath}")

def summarize_labels(counts: Counter):
    total = sum(counts.values())
    summary = " | ".join([f"{lbl.capitalize()}: {cnt:,}" for lbl, cnt in counts.items()])
    print(f"\n📊 Dataset summary → {summary} | Total: {total:,}\n")

# --- Entrypoint ---
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample_size", type=int, default=50)
    parser.add_argument("--workers", type=int, default=mp.cpu_count())
    parser.add_argument("--chunk_size", type=int, default=5000)
    args = parser.parse_args()

    print(f"⚙️ Building dataset (sample_size={args.sample_size}, workers={args.workers}, chunk_size={args.chunk_size})")
    dataset, counts = build_dataset(args.sample_size, args.workers, args.chunk_size)
    print(f"✅ Generated {len(dataset):,} total samples")
    summarize_labels(counts)
    split_and_save(dataset, Path(ORC_INTENT_TRAIN_DATA_DIR))

    print("\n🔎 Sample rows:")
    for row in random.sample(dataset, min(5, len(dataset))):
        print(json.dumps(row, ensure_ascii=False, indent=2))

    if args.sample_size > 0:
        cont = input("\nProceed with full dataset build (ALL rows)? [y/N]: ").strip().lower()
        if cont == "y":
            print("\n⚙️ Building FULL dataset...")
            dataset, counts = build_dataset(0, args.workers, args.chunk_size)
            print(f"✅ Generated {len(dataset):,} total samples")
            summarize_labels(counts)
            split_and_save(dataset, Path(ORC_INTENT_TRAIN_DATA_DIR))
        else:
            print("❌ Skipped full build — sample dataset only.")

if __name__ == "__main__":
    main()
