import os
import json
import csv
import argparse
import time
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Iterable

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

# --- Your config + logger (same style as your trainers/testers) ---
from modules.configuration.config import (
    ORC_INTENT_MODEL_DIR,
    ORC_INTENT_TRAIN_DATA_DIR
)
from modules.configuration.log_config import (
    debug_id, info_id, warning_id, error_id,
    set_request_id, get_request_id, log_timed_operation
)

# -----------------------------
# Paraphrase bank
# -----------------------------

# lightweight synonym sets applied conservatively (only whole words)
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

# polite wrappers, directives, and clarifiers
LIGHT_WRAPPERS = [
    "please {q}",
    "{q}, please",
    "could you {q}?",
    "can you {q}?",
    "I need to {q}",
    "I'd like to {q}",
]

# re-order patterns (adds context around same core)
MEDIUM_TEMPLATES = [
    "{q} for me",
    "when you can, {q}",
    "if possible, {q}",
    "{q} asap",
    "quickly {q}",
    "just {q}",
]

# heavier rephrasings (still safe for classification; avoid semantic drift)
HEAVY_TEMPLATES = [
    "I’m trying to {q}.",
    "The task is to {q}.",
    "Goal: {q}.",
    "User request: {q}.",
    "Action requested: {q}.",
    "Request: {q}.",
]

# character-level light noise for robustness without altering semantics much
def small_typos(text: str, rng: random.Random, p: float = 0.06) -> str:
    if not text or len(text) < 5:
        return text
    chars = list(text)
    for i in range(len(chars) - 1):
        if rng.random() < p and chars[i].isalpha() and chars[i+1].isalpha():
            chars[i], chars[i+1] = chars[i+1], chars[i]
    return "".join(chars)

def replace_whole_word(text: str, target: str, repl: str) -> str:
    # replace only whole word occurrences (case-insensitive), keep case simple
    pattern = r"\b" + re.escape(target) + r"\b"
    return re.sub(pattern, repl, text, flags=re.IGNORECASE)

def apply_synonyms(text: str, rng: random.Random, max_subs: int = 2) -> str:
    # choose up to max_subs distinct keys present in the text
    candidates = [k for k in SYN_SETS.keys() if re.search(r"\b" + re.escape(k) + r"\b", text, flags=re.IGNORECASE)]
    rng.shuffle(candidates)
    chosen = candidates[:max_subs]
    out = text
    for key in chosen:
        repl = rng.choice(SYN_SETS[key])
        out = replace_whole_word(out, key, repl)
    return out

def wrap_text(text: str, rng: random.Random, modes: Iterable[str]) -> str:
    q = text
    buckets = []
    if "light" in modes:
        buckets.append(LIGHT_WRAPPERS)
    if "medium" in modes:
        buckets.append(MEDIUM_TEMPLATES)
    if "heavy" in modes:
        buckets.append(HEAVY_TEMPLATES)

    if not buckets:
        return text

    # pick one template overall (simple & diverse)
    all_templates = [t for b in buckets for t in b]
    tmpl = rng.choice(all_templates)
    return tmpl.format(q=q).strip()

def paraphrase_once(text: str, rng: random.Random, modes: Iterable[str]) -> str:
    # pipeline: synonyms -> wrapper -> (optional) small typos for heavy
    out = apply_synonyms(text, rng, max_subs=2 if "medium" in modes or "heavy" in modes else 1)
    out = wrap_text(out, rng, modes)
    if "heavy" in modes:
        # introduce a tiny bit of noise to emulate user typos
        out = small_typos(out, rng, p=0.03)
    # collapse repeated spaces
    out = re.sub(r"\s+", " ", out).strip()
    return out

def generate_paraphrases(text: str, n: int, modes: Iterable[str], seed: int) -> List[str]:
    rng = random.Random(seed)
    out = set()
    # attempt more than n to avoid duplicates
    attempts = max(8, n * 4)
    while len(out) < n and attempts > 0:
        attempts -= 1
        p = paraphrase_once(text, rng, modes)
        if p != text:
            out.add(p)
    return list(out)

# -----------------------------
# Helpers: data loading
# -----------------------------
SUPPORTED_EXTS = {".jsonl", ".json", ".csv", ".tsv"}

def _default_eval_files() -> List[Path]:
    root = Path(ORC_INTENT_TRAIN_DATA_DIR)
    candidates = [
        root / "intent_eval.jsonl",
        root / "intent_dev.jsonl",
        root / "intent_validation.jsonl",
        root / "intent_train.jsonl",
        root / "intent.csv",
        root / "intent.tsv"
    ]
    return [p for p in candidates if p.exists() and p.is_file()]

def _read_jsonl(fp: Path) -> List[Dict]:
    rows = []
    with fp.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f, 1):
            try:
                obj = json.loads(line)
                if "text" in obj and "label" in obj:
                    rows.append({"text": str(obj["text"]), "label": str(obj["label"])})
            except Exception:
                continue
    return rows

def _read_json(fp: Path) -> List[Dict]:
    rows = []
    try:
        data = json.loads(fp.read_text(encoding="utf-8"))
        if isinstance(data, list):
            for obj in data:
                if isinstance(obj, dict) and "text" in obj and "label" in obj:
                    rows.append({"text": str(obj["text"]), "label": str(obj["label"])})
    except Exception:
        pass
    return rows

def _read_tabular(fp: Path, delimiter: str) -> List[Dict]:
    rows = []
    with fp.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f, delimiter=delimiter)
        for r in reader:
            # be tolerant of capitalization
            keys = {k.lower(): k for k in r.keys()}
            tkey = keys.get("text"); lkey = keys.get("label")
            if tkey and lkey:
                rows.append({"text": str(r[tkey]), "label": str(r[lkey])})
    return rows

def load_eval_data(files: List[Path], max_rows: Optional[int] = None) -> List[Dict]:
    all_rows = []
    for fp in files:
        ext = fp.suffix.lower()
        if ext == ".jsonl":
            rows = _read_jsonl(fp)
        elif ext == ".json":
            rows = _read_json(fp)
        elif ext == ".csv":
            rows = _read_tabular(fp, delimiter=",")
        elif ext == ".tsv":
            rows = _read_tabular(fp, delimiter="\t")
        else:
            continue
        all_rows.extend(rows)

    # de-dup
    seen = set()
    deduped = []
    for r in all_rows:
        key = (r["text"], r["label"])
        if key not in seen:
            seen.add(key); deduped.append(r)

    if max_rows and len(deduped) > max_rows:
        deduped = deduped[:max_rows]
    return deduped

# -----------------------------
# Label maps (id2label/label2id)
# -----------------------------
def load_label_maps(model_dir: Path) -> Tuple[Dict[int, str], Dict[str, int]]:
    labels_path = model_dir / "labels.json"
    if labels_path.exists():
        try:
            obj = json.loads(labels_path.read_text(encoding="utf-8"))
            id2label = {int(k): v for k, v in obj.get("id2label", {}).items()}
            label2id = {k: int(v) for k, v in obj.get("label2id", {}).items()}
            if id2label and label2id:
                return id2label, label2id
        except Exception:
            pass
    try:
        model = AutoModelForSequenceClassification.from_pretrained(model_dir)
        cfg = model.config
        id2label = {int(i): str(lab) for i, lab in getattr(cfg, "id2label", {}).items()}
        label2id = {str(lab): int(i) for i, lab in getattr(cfg, "label2id", {}).items()}
        return id2label, label2id
    except Exception:
        return {}, {}

# -----------------------------
# Augmentation
# -----------------------------
def augment_rows(rows: List[Dict], n_aug: int, modes: List[str], seed: int, mix_original: bool) -> List[Dict]:
    rng = random.Random(seed)
    augmented: List[Dict] = []
    for r_idx, r in enumerate(rows):
        base = r["text"].strip()
        label = r["label"]
        # seed per-row for reproducibility but still diverse
        row_seed = seed + r_idx * 97
        variants = generate_paraphrases(base, n=n_aug, modes=modes, seed=row_seed)
        if mix_original:
            augmented.append({"text": base, "label": label, "source": "orig"})
        for v in variants:
            augmented.append({"text": v, "label": label, "source": "aug"})
    return augmented

def export_augmented(rows: List[Dict], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps({"text": r["text"], "label": r["label"]}, ensure_ascii=False) + "\n")

# -----------------------------
# Evaluation + reporting
# -----------------------------
def eval_model(examples: List[Dict], model_dir: Path, batch_size: int = 16) -> Dict:
    rid = get_request_id()
    info_id(f"Loading intent model from: {model_dir}", rid)

    with log_timed_operation("load_model", rid):
        tokenizer = AutoTokenizer.from_pretrained(model_dir)
        model = AutoModelForSequenceClassification.from_pretrained(model_dir)
        clf = pipeline(
            "text-classification",
            model=model,
            tokenizer=tokenizer,
            device=-1,
            truncation=True
        )

    id2label, label2id = load_label_maps(model_dir)
    known_labels = sorted(label2id.keys()) if label2id else None
    if known_labels:
        info_id(f"Known labels (from model): {known_labels}", rid)

    texts = [e["text"] for e in examples]
    golds = [e["label"] for e in examples]

    y_pred = []
    scores = []

    info_id(f"Evaluating {len(examples)} examples...", rid)
    t0 = time.time()

    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i + batch_size]
        with log_timed_operation(f"predict_batch_{i//batch_size}", rid):
            outs = clf(batch_texts, top_k=None, return_all_scores=True)
        for out in outs:
            best = max(out, key=lambda d: float(d["score"]))
            y_pred.append(best["label"])
            scores.append(best["score"])

    elapsed = time.time() - t0
    info_id(f"Inference done in {elapsed:.2f}s ({len(examples)/max(elapsed,1e-6):.1f} samples/s)", rid)

    metrics = {}
    if accuracy_score:
        metrics["accuracy"] = float(accuracy_score(golds, y_pred))
    else:
        metrics["accuracy"] = float(np.mean([1.0 if a == b else 0.0 for a, b in zip(golds, y_pred)]))

    report_str = None
    cm = None
    labels_sorted = sorted(list(set(golds) | set(y_pred)))
    if classification_report:
        report_str = classification_report(golds, y_pred, labels=labels_sorted, digits=4)
    if confusion_matrix:
        cm = confusion_matrix(golds, y_pred, labels=labels_sorted).tolist()

    report = {
        "summary": {
            "num_examples": len(examples),
            "accuracy": metrics["accuracy"],
            "avg_confidence": float(np.mean(scores)) if scores else None,
            "labels": labels_sorted
        },
        "per_class_report_text": report_str,
        "confusion_matrix": {
            "labels": labels_sorted,
            "matrix": cm
        },
        "predictions": [
            {
                "text": ex["text"],
                "gold": ex["label"],
                "pred": yp,
                "correct": bool(ex["label"] == yp),
                "source": ex.get("source", "orig")
            }
            for ex, yp in zip(examples, y_pred)
        ]
    }
    return report

def save_report(report: Dict, out_dir: Path, base_name: str = "intent_eval_report") -> Tuple[Path, Optional[Path]]:
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = time.strftime("%Y%m%d_%H%M%S")
    json_path = out_dir / f"{base_name}_{ts}.json"
    json_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    csv_path = out_dir / f"{base_name}_{ts}.csv"
    try:
        import pandas as pd
        rows = report.get("predictions", [])
        df = pd.DataFrame(rows)
        df.to_csv(csv_path, index=False)
    except Exception:
        csv_path = None

    return json_path, csv_path

# -----------------------------
# CLI
# -----------------------------
def parse_args():
    p = argparse.ArgumentParser("Comprehensive INTENT model tester (with optional paraphrase augmentation)")
    p.add_argument("--model-dir", default=str(ORC_INTENT_MODEL_DIR), help="Trained model directory")
    p.add_argument("--data-files", nargs="*", default=None,
                   help="Eval files (JSONL/JSON/CSV/TSV with columns text,label). Default: auto from ORC_INTENT_TRAIN_DATA_DIR")
    p.add_argument("--max-rows", type=int, default=None, help="Limit examples for a quick pass")
    p.add_argument("--balance", action="store_true",
                   help="Balance classes by downsampling to smallest class count")
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--out-dir", default="intent_test_reports")

    # augmentation params
    p.add_argument("--augment", action="store_true", help="Enable paraphrase-based stress testing")
    p.add_argument("--n-aug", type=int, default=3, help="Number of paraphrases per example when --augment is used")
    p.add_argument("--paraphrase-modes", nargs="*", default=["light", "medium"],
                   choices=["light", "medium", "heavy"], help="Paraphrase intensity buckets to use")
    p.add_argument("--seed", type=int, default=42, help="Random seed for deterministic paraphrases")
    p.add_argument("--mix-original", action="store_true", default=True, help="Include originals with augmentations")
    p.add_argument("--no-mix-original", dest="mix_original", action="store_false")
    p.add_argument("--export-aug", type=str, default=None,
                   help="If set, write the augmented eval set (JSONL) to this path")
    return p.parse_args()

def balance_examples(rows: List[Dict]) -> List[Dict]:
    from collections import defaultdict
    by_label = defaultdict(list)
    for r in rows:
        by_label[r["label"]].append(r)
    if not by_label:
        return rows
    min_n = min(len(v) for v in by_label.values())
    balanced = []
    for lab, arr in by_label.items():
        balanced.extend(arr[:min_n])
    return balanced

def main():
    set_request_id()
    rid = get_request_id()
    info_id("Starting INTENT model comprehensive test...", rid)

    args = parse_args()

    model_dir = Path(args.model_dir).resolve()
    if not model_dir.exists():
        error_id(f"Model dir not found: {model_dir}", rid)
        return

    if args.data_files:
        files = [Path(p).resolve() for p in args.data_files if Path(p).suffix.lower() in SUPPORTED_EXTS]
    else:
        files = _default_eval_files()

    if not files:
        error_id("No evaluation files found. Provide --data-files or place an intent_eval.jsonl under ORC_INTENT_TRAIN_DATA_DIR.", rid)
        return

    info_id(f"Eval files: {[str(p) for p in files]}", rid)

    with log_timed_operation("load_eval_data", rid):
        base_rows = load_eval_data(files, max_rows=args.max_rows)

    info_id(f"Loaded {len(base_rows)} base examples from eval files.", rid)

    if args.balance:
        base_rows = balance_examples(base_rows)
        info_id(f"Balanced base dataset → {len(base_rows)} examples.", rid)

    rows_for_eval = base_rows

    # ---- augmentation path
    if args.augment:
        info_id(f"Augmenting with modes={args.paraphrase_modes}, n_aug={args.n_aug}, mix_original={args.mix_original}, seed={args.seed}", rid)
        with log_timed_operation("augment_rows", rid):
            rows_for_eval = augment_rows(base_rows, n_aug=args.n_aug, modes=args.paraphrase_modes,
                                         seed=args.seed, mix_original=args.mix_original)
        info_id(f"Augmented eval set size: {len(rows_for_eval)} (base={len(base_rows)})", rid)

        if args.export_aug:
            outp = Path(args.export_aug).resolve()
            with log_timed_operation("export_augmented", rid):
                export_augmented(rows_for_eval, outp)
            info_id(f"Wrote augmented eval set to: {outp}", rid)

    with log_timed_operation("eval_model", rid):
        report = eval_model(rows_for_eval, model_dir=model_dir, batch_size=args.batch_size)

    out_dir = Path(args.out_dir)
    with log_timed_operation("save_report", rid):
        json_path, csv_path = save_report(report, out_dir=out_dir)

    info_id(f"Saved JSON report: {json_path}", rid)
    if csv_path:
        info_id(f"Saved CSV predictions: {csv_path}", rid)

    # Pretty print short summary
    summ = report["summary"]
    print("\n================= INTENT EVAL SUMMARY =================")
    print(f"Examples        : {summ['num_examples']}")
    print(f"Accuracy        : {summ['accuracy']:.4f}")
    print(f"Avg confidence  : {summ['avg_confidence']:.4f}" if summ['avg_confidence'] else "Avg confidence  : n/a")
    print(f"Labels          : {', '.join(summ['labels'])}")
    if report["per_class_report_text"]:
        print("\n" + report["per_class_report_text"])
    if report["confusion_matrix"]["matrix"] is not None:
        print("\nConfusion matrix (rows=gold, cols=pred):")
        labs = report["confusion_matrix"]["labels"]
        mat = report["confusion_matrix"]["matrix"]
        header = "gold\\pred," + ",".join(labs)
        print(header)
        for lab, row in zip(labs, mat):
            print(lab + "," + ",".join(str(x) for x in row))

    print("\nDone.\n")

if __name__ == "__main__":
    main()
