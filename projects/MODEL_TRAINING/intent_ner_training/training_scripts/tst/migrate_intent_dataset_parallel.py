#!/usr/bin/env python
"""
Parallel intent dataset migration with pinned progress bars and dataset auto-discovery.
"""

import os
import sys
import json
import random
import shutil
import argparse
import multiprocessing as mp
from collections import Counter
from pathlib import Path
from tqdm import tqdm

# Import dataset dir from your config
try:
    from modules.configuration.config import ORC_INTENT_TRAIN_DATA_DIR
except ImportError:
    ORC_INTENT_TRAIN_DATA_DIR = "modules/emtac_ai/training_data/datasets/intent_classifier"

# ----------------------------
# Worker function
# ----------------------------
def process_chunk(args):
    cid, lines, tmp_dir = args
    out_file = tmp_dir / f"part_{cid}.jsonl"
    counts = Counter()
    with out_file.open("w", encoding="utf-8") as fout:
        for line in lines:
            try:
                obj = json.loads(line)
                # Tag source by label
                label = obj.get("label", "unknown")
                obj["source"] = label
                fout.write(json.dumps(obj, ensure_ascii=False) + "\n")
                counts[label] += 1
            except Exception:
                continue
    return cid, out_file, counts

# ----------------------------
# Migration function
# ----------------------------
def migrate_file_parallel(in_path: Path, workers: int, sample_size: int = 3, inplace: bool = False):
    # Make backup if overwriting
    if inplace:
        backup_path = in_path.with_suffix(in_path.suffix + ".bak")
        if not backup_path.exists():
            shutil.copy2(in_path, backup_path)
            print(f"💾 Backup created: {backup_path}")
        else:
            print(f"⚠️ Backup already exists, skipping: {backup_path}")
        out_path = in_path
    else:
        out_path = in_path.with_name(f"{in_path.stem}_tagged.jsonl")

    # Read all lines
    with in_path.open("r", encoding="utf-8") as f:
        lines = f.readlines()
    total_lines = len(lines)
    if total_lines == 0:
        print(f"⚠️ No lines in {in_path}, skipping.")
        return

    # Auto worker count
    if workers <= 0:
        workers = max(1, os.cpu_count() or 1)
    print(f"⚙️ Using {workers} workers")

    # Dynamic 100k chunking
    chunk_size = 100_000
    chunks = [lines[i:i + chunk_size] for i in range(0, total_lines, chunk_size)]

    tmp_dir = in_path.parent / f"{in_path.stem}_parts"
    tmp_dir.mkdir(exist_ok=True)

    counts_total = Counter()

    # Create progress bars
    global_bar = tqdm(total=total_lines, desc="Overall", position=0, leave=True, dynamic_ncols=True)
    worker_bars = [tqdm(total=len(chunk), desc=f"Worker {i}", position=i+1, leave=True, dynamic_ncols=True)
                   for i, chunk in enumerate(chunks[:workers])]

    # Worker update callback
    def update_progress(result):
        cid, _, counts = result
        counts_total.update(counts)
        global_bar.update(len(chunks[cid]))
        worker_bars[cid % workers].update(len(chunks[cid]))

    with mp.Pool(processes=workers) as pool:
        for cid, chunk in enumerate(chunks):
            pool.apply_async(process_chunk, args=((cid, chunk, tmp_dir),), callback=update_progress)
        pool.close()
        pool.join()

    global_bar.close()
    for bar in worker_bars:
        bar.close()

    # Concatenate parts
    with out_path.open("w", encoding="utf-8") as fout:
        for cid in range(len(chunks)):
            pf = tmp_dir / f"part_{cid}.jsonl"
            if pf.exists():
                with pf.open("r", encoding="utf-8") as f:
                    shutil.copyfileobj(f, fout)
                pf.unlink()
    tmp_dir.rmdir()

    print(f"✅ Tagged dataset written: {out_path} ({total_lines:,} lines)")

    # Print summary
    print("📊 Source distribution:")
    for src, cnt in counts_total.items():
        pct = (cnt / total_lines) * 100 if total_lines else 0
        print(f"  {src}: {cnt:,} rows ({pct:.2f}%)")

    # Show sample rows
    print("\n🔎 Random sample rows:")
    with out_path.open("r", encoding="utf-8") as f:
        sample = random.sample(list(f), min(sample_size, total_lines))
    for row in sample:
        try:
            obj = json.loads(row)
            print(json.dumps(obj, ensure_ascii=False, indent=2))
        except Exception:
            continue
    print("-" * 60)

# ----------------------------
# Main
# ----------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Parallel intent dataset migration")
    parser.add_argument("--workers", type=int, default=0, help="Number of workers (default: auto)")
    parser.add_argument("--samples", type=int, default=3, help="Number of sample rows to print")
    parser.add_argument("--inplace", action="store_true", help="Overwrite originals with backup")
    args = parser.parse_args()

    dataset_dir = Path(ORC_INTENT_TRAIN_DATA_DIR).resolve()
    files = [
        dataset_dir / "intent_train.jsonl",
        dataset_dir / "intent_val.jsonl",
        dataset_dir / "intent_test.jsonl",
    ]

    for f in files:
        if not f.exists():
            print(f"⚠️ File not found: {f}")
        else:
            migrate_file_parallel(f, workers=args.workers, sample_size=args.samples, inplace=args.inplace)
