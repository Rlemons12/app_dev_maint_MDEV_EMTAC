import os
import json
import csv
import argparse
import math
import time
from pathlib import Path
from typing import List, Dict, Tuple, Optional

import random
import re
import numpy as np

# Optional heavy deps
try:
    import pandas as pd
except Exception:
    pd = None

try:
    from sklearn.metrics import (
        classification_report, confusion_matrix, accuracy_score, f1_score
    )
except Exception:
    classification_report = None
    confusion_matrix = None
    accuracy_score = None
    f1_score = None

try:
    import matplotlib.pyplot as plt
except Exception:
    plt = None

from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline

# --- Project config + logger (mirrors your trainers/testers) ---
from modules.configuration.config import (
    ORC_INTENT_MODEL_DIR,
    ORC_INTENT_TRAIN_DATA_DIR
)
from modules.configuration.log_config import (
    debug_id, info_id, warning_id, error_id,
    set_request_id, get_request_id, log_timed_operation
)

SUPPORTED_EXTS = {".jsonl", ".json", ".csv", ".tsv"}

# -----------------------------
# Robust paraphrase + noise bank (compatible with your v1)
# -----------------------------
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

NOISE_SNIPPETS = [
    "(thanks)", "— urgent", "FYI:", "pls", "thx", "[internal] ",
    "\n\n", "    ", "!!!", " ???", " ..", " [ref 123]"
]

def small_typos(text: str, rng: random.Random, p: float = 0.06) -> str:
    if not text or len(text) < 5:
        return text
    chars = list(text)
    for i in range(len(chars) - 1):
        if rng.random() < p and chars[i].isalpha() and chars[i+1].isalpha():
            chars[i], chars[i+1] = chars[i+1], chars[i]
    return "".join(chars)

_cw = re.compile(r"\s+")

def replace_whole_word(text: str, target: str, repl: str) -> str:
    pattern = r"\\b" + re.escape(target) + r"\\b"
    return re.sub(pattern, repl, text, flags=re.IGNORECASE)

def apply_synonyms(text: str, rng: random.Random, max_subs: int = 2) -> str:
    candidates = [k for k in SYN_SETS.keys() if re.search(r"\\b" + re.escape(k) + r"\\b", text, flags=re.IGNORECASE)]
    rng.shuffle(candidates)
    chosen = candidates[:max_subs]
    out = text
    for key in chosen:
        repl = rng.choice(SYN_SETS[key])
        out = replace_whole_word(out, key, repl)
    return out

def wrap_text(text: str, rng: random.Random, modes):
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
    all_templates = [t for b in buckets for t in b]
    return rng.choice(all_templates).format(q=q).strip()

def add_noise(text: str, rng: random.Random, p: float = 0.25) -> str:
    out = text
    if rng.random() < p:
        out = out + " " + rng.choice(NOISE_SNIPPETS)
    if rng.random() < p:
        out = out.capitalize()
    if rng.random() < p:
        out = out.upper() if rng.random() < 0.5 else out.lower()
    return _cw.sub(" ", out).strip()

def paraphrase_once(text: str, rng: random.Random, modes) -> str:
    out = apply_synonyms(text, rng, max_subs=2 if ("medium" in modes or "heavy" in modes) else 1)
    out = wrap_text(out, rng, modes)
    if "heavy" in modes:
        out = small_typos(out, rng, p=0.03)
    out = add_noise(out, rng, p=0.35)
    return re.sub(r"\s+", " ", out).strip()

def generate_paraphrases(text: str, n: int, modes, seed: int) -> List[str]:
    rng = random.Random(seed)
    out = set()
    attempts = max(8, n * 4)
    while len(out) < n and attempts > 0:
        attempts -= 1
        p = paraphrase_once(text, rng, modes)
        if p != text:
            out.add(p)
    return list(out)

# -----------------------------
# Data loading
# -----------------------------

def _default_eval_files() -> List[Path]:
    root = Path(ORC_INTENT_TRAIN_DATA_DIR)
    candidates = [
        root / "intent_eval.jsonl",
        root / "intent_dev.jsonl",
        root / "intent_validation.jsonl",
        root / "intent_train.jsonl",
        root / "intent.csv",
        root / "intent.tsv",
    ]
    return [p for p in candidates if p.exists() and p.is_file()]


def _read_jsonl(fp: Path) -> List[Dict]:
    rows = []
    with fp.open("r", encoding="utf-8") as f:
        for line in f:
            try:
                obj = json.loads(line)
                if "text" in obj and "label" in obj:
                    rows.append({"text": str(obj["text"]).strip(), "label": str(obj["label"]).strip()})
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
                    rows.append({"text": str(obj["text"]).strip(), "label": str(obj["label"]).strip()})
    except Exception:
        pass
    return rows


def _read_tabular(fp: Path, delimiter: str) -> List[Dict]:
    rows = []
    with fp.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f, delimiter=delimiter)
        for r in reader:
            keys = {k.lower(): k for k in r.keys()}
            tkey = keys.get("text"); lkey = keys.get("label")
            if tkey and lkey:
                rows.append({"text": str(r[tkey]).strip(), "label": str(r[lkey]).strip()})
    return rows


def load_eval_data(files: List[Path], max_rows: Optional[int] = None) -> List[Dict]:
    """Load eval rows with optional early cap at max_rows (streaming stop).
    This prevents reading a 10M+ line file when doing a quick pass.
    """
    rid = get_request_id()
    all_rows = []
    count = 0

    def maybe_stop():
        return (max_rows is not None) and (count >= max_rows)

    for fp in files:
        if maybe_stop():
            break
        ext = fp.suffix.lower()
        if ext not in {".jsonl", ".json", ".csv", ".tsv"}:
            continue
        info_id(f"Loading eval file: {fp}", rid)
        try:
            if ext == ".jsonl":
                with fp.open("r", encoding="utf-8") as f:
                    for line in f:
                        try:
                            obj = json.loads(line)
                            if "text" in obj and "label" in obj:
                                all_rows.append({"text": str(obj["text"]).strip(), "label": str(obj["label"]).strip()})
                                count += 1
                                if count % 100000 == 0:
                                    info_id(f"  loaded {count:,} rows so far...", rid)
                                if maybe_stop():
                                    break
                        except Exception:
                            continue
            elif ext == ".json":
                try:
                    data = json.loads(fp.read_text(encoding="utf-8"))
                except Exception:
                    data = []
                if isinstance(data, list):
                    for obj in data:
                        if isinstance(obj, dict) and "text" in obj and "label" in obj:
                            all_rows.append({"text": str(obj["text"]).strip(), "label": str(obj["label"]).strip()})
                            count += 1
                            if maybe_stop():
                                break
            elif ext in {".csv", ".tsv"}:
                delimiter = "," if ext == ".csv" else "	"
                with fp.open("r", encoding="utf-8", newline="") as f:
                    reader = csv.DictReader(f, delimiter=delimiter)
                    for r in reader:
                        keys = {k.lower(): k for k in r.keys()}
                        tkey = keys.get("text"); lkey = keys.get("label")
                        if tkey and lkey:
                            all_rows.append({"text": str(r[tkey]).strip(), "label": str(r[lkey]).strip()})
                            count += 1
                            if count % 100000 == 0:
                                info_id(f"  loaded {count:,} rows so far...", rid)
                            if maybe_stop():
                                break
        except Exception as e:
            warning_id(f"Failed reading {fp}: {e}", rid)

    # de-dup
    seen = set()
    deduped = []
    for r in all_rows:
        key = (r["text"], r["label"])
        if key not in seen:
            seen.add(key); deduped.append(r)

    return deduped

# -----------------------------
# Label maps
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
# Evaluation core with top-k + confidence
# -----------------------------

def run_inference(texts: List[str], clf, batch_size: int = 16, topk: int = 5, progress: bool = True) -> Tuple[List[str], List[float], List[List[Tuple[str, float]]]]:
    """Batched inference with optional progress logs and controllable top-k.
    For speed, set topk=1 to avoid building large per-class lists.
    """
    rid = get_request_id()
    y_pred: List[str] = []
    scores: List[float] = []
    topk_all: List[List[Tuple[str, float]]] = []

    total_batches = (len(texts) + batch_size - 1) // batch_size
    for bi in range(0, len(texts), batch_size):
        batch_texts = texts[bi:bi + batch_size]
        outs = clf(batch_texts, top_k=None, return_all_scores=True)
        for out in outs:
            # out: list of dicts [{label, score}, ...]
            best = max(out, key=lambda d: float(d["score"]))
            y_pred.append(best["label"])
            scores.append(float(best["score"]))
            if topk and topk > 1:
                ranked = sorted(((d["label"], float(d["score"])) for d in out), key=lambda x: x[1], reverse=True)[:topk]
                topk_all.append(ranked)
            else:
                topk_all.append([(best["label"], float(best["score"]))])
        if progress and ((bi // batch_size) % 100 == 0):
            info_id(f"Inference progress: batch {(bi // batch_size) + 1}/{total_batches}", rid)
    return y_pred, scores, topk_all


def eval_model(examples: List[Dict], model_dir: Path, batch_size: int = 16, device_str: str = "cpu", topk: int = 5, progress: bool = True) -> Dict:
    rid = get_request_id()
    info_id(f"Loading intent model from: {model_dir}", rid)

    with log_timed_operation("load_model", rid):
        tokenizer = AutoTokenizer.from_pretrained(model_dir)
        model = AutoModelForSequenceClassification.from_pretrained(model_dir)
        device_arg = -1
        if device_str.lower().startswith("cuda"):
            # pipeline expects device index for CUDA
            device_arg = 0 if ":" not in device_str else int(device_str.split(":",1)[1])
        clf = pipeline(
            "text-classification",
            model=model,
            tokenizer=tokenizer,
            device=device_arg,
            truncation=True
        )

    id2label, label2id = load_label_maps(model_dir)
    known_labels = sorted(label2id.keys()) if label2id else None
    if known_labels:
        info_id(f"Known labels (from model): {known_labels}", rid)

    texts = [e["text"] for e in examples]
    golds = [e["label"] for e in examples]

    info_id(f"Evaluating {len(examples)} examples...", rid)
    t0 = time.time()
    y_pred, scores, topk = run_inference(texts, clf, batch_size=batch_size, topk=topk, progress=progress)
    elapsed = time.time() - t0
    info_id(f"Inference done in {elapsed:.2f}s ({len(examples)/max(elapsed,1e-6):.1f} samples/s)", rid)

    # Base metrics
    if accuracy_score:
        acc = float(accuracy_score(golds, y_pred))
    else:
        acc = float(np.mean([1.0 if a == b else 0.0 for a, b in zip(golds, y_pred)]))

    macro_f1 = None
    if f1_score:
        try:
            # Derive numeric mapping dynamically
            uniq = sorted(list(set(golds) | set(y_pred)))
            lab2i = {lab: i for i, lab in enumerate(uniq)}
            y_true_i = np.array([lab2i[g] for g in golds])
            y_pred_i = np.array([lab2i[p] for p in y_pred])
            macro_f1 = float(f1_score(y_true_i, y_pred_i, average="macro"))
        except Exception:
            macro_f1 = None

    report_str = None
    cm = None
    labels_sorted = sorted(list(set(golds) | set(y_pred)))
    if classification_report:
        report_str = classification_report(golds, y_pred, labels=labels_sorted, digits=4)
    if confusion_matrix:
        cm = confusion_matrix(golds, y_pred, labels=labels_sorted).tolist()

    # Calibration: Reliability bins (ECE)
    calib = compute_reliability(scores, [1 if a == b else 0 for a, b in zip(golds, y_pred)], n_bins=10)

    # Threshold sweep for abstain (route-to-FTS) policy
    sweep = threshold_sweep(scores, [1 if a == b else 0 for a, b in zip(golds, y_pred)], steps=20)

    report = {
        "summary": {
            "num_examples": len(examples),
            "accuracy": acc,
            "macro_f1": macro_f1,
            "avg_confidence": float(np.mean(scores)) if scores else None,
            "labels": labels_sorted,
        },
        "per_class_report_text": report_str,
        "confusion_matrix": {
            "labels": labels_sorted,
            "matrix": cm,
        },
        "calibration": calib,
        "threshold_sweep": sweep,
        "predictions": [
            {
                "text": ex["text"],
                "gold": ex["label"],
                "pred": yp,
                "confidence": float(sc),
                "correct": bool(ex["label"] == yp),
                "source": ex.get("source", "orig"),
                "topk": topk_i[:5],
            }
            for ex, yp, sc, topk_i in zip(examples, y_pred, scores, topk)
        ],
    }
    return report

# -----------------------------
# Calibration utilities
# -----------------------------

def compute_reliability(confidences: List[float], correctness: List[int], n_bins: int = 10) -> Dict:
    bins = np.linspace(0, 1, n_bins + 1)
    total = len(confidences)
    out = {"bins": [], "bin_acc": [], "bin_conf": [], "bin_count": [], "ece": None}
    ece = 0.0
    for i in range(n_bins):
        lo, hi = bins[i], bins[i+1]
        idx = [j for j, c in enumerate(confidences) if (c > lo and c <= hi) or (i == 0 and c == 0.0)]
        if not idx:
            out["bins"].append([float(lo), float(hi)])
            out["bin_acc"].append(None)
            out["bin_conf"].append(None)
            out["bin_count"].append(0)
            continue
        acc = float(np.mean([correctness[j] for j in idx]))
        conf = float(np.mean([confidences[j] for j in idx]))
        out["bins"].append([float(lo), float(hi)])
        out["bin_acc"].append(acc)
        out["bin_conf"].append(conf)
        out["bin_count"].append(len(idx))
        ece += (len(idx) / total) * abs(acc - conf)
    out["ece"] = float(ece)
    return out


def threshold_sweep(confidences: List[float], correctness: List[int], steps: int = 20) -> List[Dict]:
    # Return coverage/accuracy pairs for thresholds t where we accept pred if conf >= t
    pairs = []
    for k in range(steps + 1):
        t = k / steps
        take = [i for i, c in enumerate(confidences) if c >= t]
        coverage = len(take) / max(1, len(confidences))
        if take:
            acc = float(np.mean([correctness[i] for i in take]))
        else:
            acc = None
        pairs.append({"threshold": t, "coverage": coverage, "accuracy": acc})
    return pairs

# -----------------------------
# Augmentation paths
# -----------------------------

def augment_rows(rows: List[Dict], n_aug: int, modes: List[str], seed: int, mix_original: bool) -> List[Dict]:
    rng = random.Random(seed)
    augmented: List[Dict] = []
    for r_idx, r in enumerate(rows):
        base = r["text"].strip()
        label = r["label"]
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
# Reporting helpers (JSON, CSV, PNG)
# -----------------------------

def save_report(report: Dict, out_dir: Path, base_name: str = "intent_eval_report") -> Tuple[Path, Optional[Path], List[Path]]:
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = time.strftime("%Y%m%d_%H%M%S")
    json_path = out_dir / f"{base_name}_{ts}.json"
    json_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    csv_path = out_dir / f"{base_name}_{ts}.csv"
    try:
        if pd is not None:
            rows = report.get("predictions", [])
            df = pd.DataFrame(rows)
            df.to_csv(csv_path, index=False)
        else:
            csv_path = None
    except Exception:
        csv_path = None

    figs = []
    if plt is not None:
        try:
            # Confusion matrix heatmap
            cm = report.get("confusion_matrix", {}).get("matrix")
            labels = report.get("confusion_matrix", {}).get("labels")
            if cm and labels:
                fig = plt.figure(figsize=(6, 5))
                ax = fig.add_subplot(111)
                im = ax.imshow(np.array(cm), aspect="auto")
                ax.set_xticks(range(len(labels)))
                ax.set_xticklabels(labels, rotation=45, ha="right")
                ax.set_yticks(range(len(labels)))
                ax.set_yticklabels(labels)
                ax.set_xlabel("Pred")
                ax.set_ylabel("Gold")
                ax.set_title("Confusion Matrix")
                fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
                p = out_dir / f"{base_name}_{ts}_cm.png"
                fig.tight_layout()
                fig.savefig(p, dpi=150)
                plt.close(fig)
                figs.append(p)

            # Reliability diagram
            calib = report.get("calibration")
            if calib and calib.get("bin_acc"):
                fig = plt.figure(figsize=(5, 5))
                ax = fig.add_subplot(111)
                acc = [a if a is not None else np.nan for a in calib["bin_acc"]]
                conf = [c if c is not None else np.nan for c in calib["bin_conf"]]
                ax.plot(conf, acc, marker="o")
                ax.plot([0, 1], [0, 1], linestyle="--")
                ax.set_xlim(0, 1)
                ax.set_ylim(0, 1)
                ax.set_xlabel("Confidence")
                ax.set_ylabel("Accuracy")
                ax.set_title(f"Reliability (ECE={calib.get('ece'):.3f})")
                p = out_dir / f"{base_name}_{ts}_reliability.png"
                fig.tight_layout()
                fig.savefig(p, dpi=150)
                plt.close(fig)
                figs.append(p)

            # Coverage–Accuracy curve (threshold sweep)
            sweep = report.get("threshold_sweep", [])
            if sweep:
                fig = plt.figure(figsize=(6, 4))
                ax = fig.add_subplot(111)
                cov = [s["coverage"] for s in sweep if s["accuracy"] is not None]
                acc = [s["accuracy"] for s in sweep if s["accuracy"] is not None]
                ax.plot(cov, acc, marker="o")
                ax.set_xlabel("Coverage (fraction accepted)")
                ax.set_ylabel("Accuracy of accepted")
                ax.set_title("Coverage–Accuracy under confidence threshold")
                ax.grid(True, linestyle=":", linewidth=0.5)
                p = out_dir / f"{base_name}_{ts}_coverage_accuracy.png"
                fig.tight_layout()
                fig.savefig(p, dpi=150)
                plt.close(fig)
                figs.append(p)
        except Exception as e:
            warning_id(f"Plot generation failed: {e}")

    return json_path, csv_path, figs

# -----------------------------
# CLI
# -----------------------------
def parse_args():
    p = argparse.ArgumentParser("Comprehensive INTENT model test (paraphrases + calibration + threshold sweep)")
    p.add_argument("--model-dir", default=str(ORC_INTENT_MODEL_DIR), help="Trained model directory")
    p.add_argument("--use-warmup", action="store_true",
                   help="Test the warmup model instead of production model")
    p.add_argument("--data-files", nargs="*", default=None,
                   help="Eval files (JSONL/JSON/CSV/TSV with columns text,label). Default: auto from ORC_INTENT_TRAIN_DATA_DIR")
    p.add_argument("--max-rows", type=int, default=None, help="Limit examples for a quick pass (streaming early stop)")
    p.add_argument("--balance", action="store_true",
                   help="Balance classes by downsampling to smallest class count")
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--device", type=str, default="cpu", help="cpu or cuda:0, cuda:1, ...")
    p.add_argument("--topk", type=int, default=5, help="How many top classes to keep per example (1 is fastest)")
    p.add_argument("--progress", action="store_true", help="Log periodic inference progress")
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

    # policy knobs
    p.add_argument("--threshold-sweep", action="store_true", help="Print best threshold by accuracy of accepted")

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


# -----------------------------
# Main
# -----------------------------
def main():
    set_request_id()
    rid = get_request_id()
    info_id("Starting INTENT model comprehensive test (v2)...", rid)

    args = parse_args()

    # Handle warmup model flag
    if args.use_warmup:
        warmup_model_dir = Path(ORC_INTENT_MODEL_DIR) / "_warmup_model"
        if not warmup_model_dir.exists():
            error_id(f"Warmup model not found at: {warmup_model_dir}", rid)
            error_id("Train with --max-examples first to create a warmup model", rid)
            return
        model_dir = warmup_model_dir
        info_id(f"Testing WARMUP model from: {model_dir}", rid)
    else:
        model_dir = Path(args.model_dir).resolve()

    if not model_dir.exists():
        error_id(f"Model dir not found: {model_dir}", rid)
        return

    if args.data_files:
        files = [Path(p).resolve() for p in args.data_files if Path(p).suffix.lower() in SUPPORTED_EXTS]
    else:
        files = _default_eval_files()

    if not files:
        error_id(
            "No evaluation files found. Provide --data-files or place an intent_eval.jsonl under ORC_INTENT_TRAIN_DATA_DIR.",
            rid)
        return

    info_id(f"Eval files: {[str(p) for p in files]}", rid)

    with log_timed_operation("load_eval_data", rid):
        base_rows = load_eval_data(files, max_rows=args.max_rows)

    info_id(f"Loaded {len(base_rows)} base examples from eval files.", rid)

    if args.balance:
        base_rows = balance_examples(base_rows)
        info_id(f"Balanced base dataset → {len(base_rows)} examples.", rid)

    rows_for_eval = base_rows

    # augmentation
    if args.augment:
        info_id(
            f"Augmenting with modes={args.paraphrase_modes}, n_aug={args.n_aug}, mix_original={args.mix_original}, seed={args.seed}",
            rid)
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
        report = eval_model(rows_for_eval, model_dir=model_dir, batch_size=args.batch_size, device_str=args.device,
                            topk=args.topk, progress=args.progress)

    out_dir = Path(args.out_dir)
    with log_timed_operation("save_report", rid):
        json_path, csv_path, figs = save_report(report, out_dir=out_dir)

    info_id(f"Saved JSON report: {json_path}", rid)
    if csv_path:
        info_id(f"Saved CSV predictions: {csv_path}", rid)
    if figs:
        for p in figs:
            info_id(f"Saved plot: {p}", rid)

    # Pretty print short summary + optional threshold pick
    summ = report["summary"]
    print("\n================= INTENT EVAL SUMMARY =================")
    print(f"Examples        : {summ['num_examples']}")
    print(f"Accuracy        : {summ['accuracy']:.4f}")
    if summ.get("macro_f1") is not None:
        print(f"Macro-F1       : {summ['macro_f1']:.4f}")
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

    # Best operating point (max accuracy among accepted, preferring >=50% coverage)
    if args.threshold_sweep and report.get("threshold_sweep"):
        best = None
        for pt in report["threshold_sweep"]:
            if pt["accuracy"] is None:
                continue
            cov = pt["coverage"]
            acc = pt["accuracy"]
            score = acc + 0.05 * cov  # small tie-breaker for higher coverage
            if (best is None) or (score > best["score"]):
                best = {"threshold": pt["threshold"], "coverage": cov, "accuracy": acc, "score": score}
        if best:
            print("\nBest threshold (by accepted accuracy with coverage tie-break):")
            print(
                f"  threshold={best['threshold']:.2f} | coverage={best['coverage']:.2%} | accuracy={best['accuracy']:.2%}")

    print("\nDone.\n")


if __name__ == "__main__":
    main()

"""python -m modules.emtac_ai.training_scripts.performance_tst_model.intent_model_comprehensive_test --batch-size 32 --device cpu --progress --max-rows 100000"""