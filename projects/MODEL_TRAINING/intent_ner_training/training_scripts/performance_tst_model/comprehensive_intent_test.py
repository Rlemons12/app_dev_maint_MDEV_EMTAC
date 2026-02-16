import os
import json
import csv
import argparse
import time
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Iterable
from collections import defaultdict

import torch
import random
import re
import numpy as np

# Optional dependencies
try:
    from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
except Exception:
    classification_report = None
    confusion_matrix = None
    accuracy_score = None

from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline

from configuration.log_config import (
    debug_id, info_id, warning_id, error_id,
    set_request_id, get_request_id, log_timed_operation
)
from configuration.config import (
    MODEL_TRAINING_INTENT_MODEL_DIR,
    MODEL_TRAINING_INTENT_TRAIN_DATA_DIR
)

# =====================================================
# PARAPHRASE BANK
# =====================================================

SYN_SETS = {
    "find": ["locate", "look up", "search for", "pull up", "fetch"],
    "show": ["display", "show me", "bring up", "open"],
    "parts": ["spares", "components", "replacement parts"],
    "drawings": ["prints", "blueprints", "schematics"],
    "manual": ["guide", "handbook", "documentation"],
    "model": ["mdl", "type"],
    "number": ["no.", "num"],
    "by": ["from", "made by"],
    "for": ["regarding", "about"],
    "help": ["assist", "support"],
    "list": ["catalog", "enumerate"],
}

LIGHT_WRAPPERS = [
    "please {q}",
    "{q}, please",
    "could you {q}?",
    "can you {q}?",
    "I need to {q}",
    "I'd like to {q}",
]

MEDIUM_TEMPLATES = [
    "{q} for me",
    "when you can, {q}",
    "if possible, {q}",
    "{q} asap",
    "quickly {q}",
    "just {q}",
]

HEAVY_TEMPLATES = [
    "I’m trying to {q}.",
    "The task is to {q}.",
    "Goal: {q}.",
    "User request: {q}.",
    "Action requested: {q}.",
    "Request: {q}.",
]

# -----------------------------------------------------
# Paraphrase helpers
# -----------------------------------------------------

def small_typos(text: str, rng: random.Random, p: float = 0.06) -> str:
    if not text or len(text) < 5:
        return text
    chars = list(text)
    for i in range(len(chars) - 1):
        if rng.random() < p and chars[i].isalpha() and chars[i + 1].isalpha():
            chars[i], chars[i + 1] = chars[i + 1], chars[i]
    return "".join(chars)


def replace_whole_word(text: str, target: str, repl: str) -> str:
    pattern = r"\b" + re.escape(target) + r"\b"
    return re.sub(pattern, repl, text, flags=re.IGNORECASE)


def apply_synonyms(text: str, rng: random.Random, max_subs: int = 2) -> str:
    candidates = [
        k for k in SYN_SETS
        if re.search(r"\b" + re.escape(k) + r"\b", text, flags=re.IGNORECASE)
    ]
    rng.shuffle(candidates)
    out = text
    for key in candidates[:max_subs]:
        out = replace_whole_word(out, key, rng.choice(SYN_SETS[key]))
    return out


def wrap_text(text: str, rng: random.Random, modes: Iterable[str]) -> str:
    templates = []
    if "light" in modes:
        templates += LIGHT_WRAPPERS
    if "medium" in modes:
        templates += MEDIUM_TEMPLATES
    if "heavy" in modes:
        templates += HEAVY_TEMPLATES
    if not templates:
        return text
    return rng.choice(templates).format(q=text).strip()


def paraphrase_once(text: str, rng: random.Random, modes: Iterable[str]) -> str:
    out = apply_synonyms(text, rng, max_subs=2 if "medium" in modes or "heavy" in modes else 1)
    out = wrap_text(out, rng, modes)
    if "heavy" in modes:
        out = small_typos(out, rng, p=0.03)
    return re.sub(r"\s+", " ", out).strip()


def generate_paraphrases(text: str, n: int, modes: Iterable[str], seed: int) -> List[str]:
    rng = random.Random(seed)
    out = set()
    attempts = max(8, n * 4)
    while len(out) < n and attempts > 0:
        attempts -= 1
        p = paraphrase_once(text, rng, modes)
        if p != text:
            out.add(p)
    return list(out)

# =====================================================
# DATA LOADING
# =====================================================

SUPPORTED_EXTS = {".jsonl", ".json", ".csv", ".tsv"}

def _default_eval_files() -> List[Path]:
    root = Path(MODEL_TRAINING_INTENT_TRAIN_DATA_DIR)
    return [
        root / "intent_val.jsonl",            # canonical eval
        #root / "intent_train_parts.jsonl",     # optional
        #root / "intent_train_drawings.jsonl",  # optional
    ]



def _read_jsonl(fp: Path) -> List[Dict]:
    rows = []
    with fp.open("r", encoding="utf-8") as f:
        for line in f:
            try:
                obj = json.loads(line)
                if "text" in obj and "label" in obj:
                    rows.append({"text": str(obj["text"]), "label": str(obj["label"])})
            except Exception:
                continue
    return rows


def _read_json(fp: Path) -> List[Dict]:
    try:
        data = json.loads(fp.read_text(encoding="utf-8"))
        return [{"text": str(o["text"]), "label": str(o["label"])} for o in data if "text" in o and "label" in o]
    except Exception:
        return []


def _read_tabular(fp: Path, delimiter: str) -> List[Dict]:
    rows = []
    with fp.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f, delimiter=delimiter)
        for r in reader:
            keys = {k.lower(): k for k in r.keys()}
            if "text" in keys and "label" in keys:
                rows.append({"text": str(r[keys["text"]]), "label": str(r[keys["label"]])})
    return rows


def load_eval_data(files: List[Path], max_rows: Optional[int] = None) -> List[Dict]:
    rows = []
    for fp in files:
        if fp.suffix == ".jsonl":
            rows += _read_jsonl(fp)
        elif fp.suffix == ".json":
            rows += _read_json(fp)
        elif fp.suffix == ".csv":
            rows += _read_tabular(fp, ",")
        elif fp.suffix == ".tsv":
            rows += _read_tabular(fp, "\t")

    dedup = {(r["text"], r["label"]): r for r in rows}
    out = list(dedup.values())
    return out[:max_rows] if max_rows else out

# =====================================================
# BALANCING (NEW)
# =====================================================

def balance_examples(rows: List[Dict], seed: int = 42) -> List[Dict]:
    rng = random.Random(seed)
    by_label = defaultdict(list)
    for r in rows:
        by_label[r["label"]].append(r)

    if not by_label:
        return rows

    min_n = min(len(v) for v in by_label.values())
    balanced = []
    for lab, items in by_label.items():
        rng.shuffle(items)
        balanced.extend(items[:min_n])

    rng.shuffle(balanced)
    return balanced

# =====================================================
# EVALUATION
# =====================================================

def eval_model(examples: List[Dict], model_dir: Path, batch_size: int) -> Dict:
    rid = get_request_id()
    device = 0 if torch.cuda.is_available() else -1

    info_id(f"Loading intent model from {model_dir} (device={device})", rid)

    with log_timed_operation("load_model", rid):
        clf = pipeline(
            "text-classification",
            model=AutoModelForSequenceClassification.from_pretrained(model_dir),
            tokenizer=AutoTokenizer.from_pretrained(model_dir),
            device=device,
            batch_size=batch_size,
            truncation=True,
        )

    texts = [e["text"] for e in examples]
    golds = [e["label"] for e in examples]

    y_pred, scores = [], []
    t0 = time.time()

    for i in range(0, len(texts), batch_size):
        outs = clf(texts[i:i + batch_size], top_k=None, return_all_scores=True)
        for o in outs:
            best = max(o, key=lambda d: d["score"])
            y_pred.append(best["label"])
            scores.append(best["score"])

    elapsed = time.time() - t0
    acc = accuracy_score(golds, y_pred) if accuracy_score else float(np.mean(np.array(golds) == np.array(y_pred)))
    labels = sorted(set(golds) | set(y_pred))

    return {
        "summary": {
            "num_examples": len(examples),
            "accuracy": float(acc),
            "avg_confidence": float(np.mean(scores)),
            "labels": labels,
            "elapsed_sec": elapsed,
        },
        "per_class_report_text": classification_report(golds, y_pred, labels=labels, digits=4, zero_division=0)
        if classification_report else None,
        "confusion_matrix": {
            "labels": labels,
            "matrix": confusion_matrix(golds, y_pred, labels=labels).tolist()
            if confusion_matrix else None,
        },
    }

# =====================================================
# CLI
# =====================================================

def parse_args():
    p = argparse.ArgumentParser("Intent model evaluator")
    p.add_argument("--model-dir", default=str(MODEL_TRAINING_INTENT_MODEL_DIR))
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--augment", action="store_true")
    p.add_argument("--n-aug", type=int, default=3)
    p.add_argument("--paraphrase-modes", nargs="*", default=["light", "medium"])
    p.add_argument("--balance", action="store_true", help="Balance eval set by downsampling each label")
    return p.parse_args()

def augment_rows(
    rows: List[Dict],
    n_aug: int,
    modes: List[str],
    seed: int,
    mix_original: bool = True,
) -> List[Dict]:
    augmented: List[Dict] = []
    rng = random.Random(seed)

    for idx, r in enumerate(rows):
        base = r["text"].strip()
        label = r["label"]

        row_seed = seed + idx * 97
        row_rng = random.Random(row_seed)

        if mix_original:
            augmented.append({"text": base, "label": label, "source": "orig"})

        variants = generate_paraphrases(
            base,
            n=n_aug,
            modes=modes,
            seed=row_seed,
        )

        for v in variants:
            augmented.append({"text": v, "label": label, "source": "aug"})

    return augmented


def main():
    set_request_id()
    rid = get_request_id()

    args = parse_args()
    model_dir = Path(args.model_dir)

    files = _default_eval_files()
    rows = load_eval_data(files)

    if args.balance:
        info_id("Applying class-balanced evaluation", rid)
        rows = balance_examples(rows)

    if args.augment:
        rows = augment_rows(
            rows,
            args.n_aug,
            args.paraphrase_modes,
            seed=42,
            mix_original=True,
        )

    report = eval_model(rows, model_dir, args.batch_size)

    print("\n========== INTENT EVAL SUMMARY ==========")
    print(json.dumps(report["summary"], indent=2))

    # 🔍 CONFUSION MATRIX
    print("\nCONFUSION MATRIX:")
    labels = report["confusion_matrix"]["labels"]
    matrix = report["confusion_matrix"]["matrix"]

    print("labels:", labels)
    for row in matrix:
        print(row)

    # Optional: full per-class report
    if report.get("per_class_report_text"):
        print("\nCLASSIFICATION REPORT:")
        print(report["per_class_report_text"])

    print("\nDone.\n")


if __name__ == "__main__":
    main()
