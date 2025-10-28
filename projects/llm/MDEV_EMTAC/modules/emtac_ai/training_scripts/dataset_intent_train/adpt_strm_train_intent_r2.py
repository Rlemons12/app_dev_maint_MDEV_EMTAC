# Adaptive streaming INTENT training (sequence classification)
# Built-in defaults:
#   --train-files C:\data\intent_train.jsonl
#   --eval-files  C:\data\intent_eval.jsonl
#   --audit
#   --early-stop 5
#   --overfit-guard --gap-threshold 0.30 --gap-patience 3 --f1-decline-patience 2 --min-evals 3
#   --mode small

import os, sys, json, argparse, logging, random, shutil, platform, csv, re, hashlib
from pathlib import Path
from typing import List, Dict, Optional, Iterable, Tuple

# Make sure torchvision is not pulled in by transformers
os.environ["TRANSFORMERS_NO_TORCHVISION"] = "1"

import numpy as np
import torch
import psutil

from torch.utils.data import IterableDataset, DataLoader
from datasets import load_dataset
from transformers import (
    DistilBertTokenizerFast,
    DistilBertForSequenceClassification,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
    EarlyStoppingCallback,
)
from transformers.trainer_callback import TrainerCallback, TrainerControl, TrainerState

# Try sklearn for metrics; fall back gracefully if not installed
try:
    from sklearn.metrics import accuracy_score, f1_score
    _HAVE_SK = True
except Exception:
    accuracy_score = None
    f1_score = None
    _HAVE_SK = False

# === Project config paths (yours) ===
from modules.configuration.config import ORC_INTENT_TRAIN_DATA_DIR, ORC_INTENT_MODEL_DIR

# === Your custom logger ===
from modules.configuration.log_config import (
    debug_id, info_id, warning_id, error_id, critical_id,
    with_request_id, set_request_id, get_request_id,
    log_timed_operation
)

logging.basicConfig(level=logging.WARNING)  # keep root quiet; use your logger


# -------------------------
# Utilities
# -------------------------
SUPPORTED = {".jsonl", ".json", ".csv", ".tsv"}

def _normalize_text(s: str) -> str:
    s = s.lower()
    s = re.sub(r"\s+", " ", s)
    s = re.sub(r"[^\w\s-]", "", s)
    return s.strip()

def _hash_text(s: str) -> str:
    return hashlib.sha256(_normalize_text(s).encode("utf-8")).hexdigest()

def _read_label_set(train_files: List[Path]) -> List[str]:
    labels = set()
    for f in train_files:
        try:
            with open(f, "r", encoding="utf-8") as fh:
                for line in fh:
                    try:
                        obj = json.loads(line)
                        if "label" in obj:
                            labels.add(str(obj["label"]))
                    except Exception:
                        continue
        except FileNotFoundError:
            continue
    return sorted(labels)

def _apply_aliases(label_list: List[str], alias_prints_to_drawings: bool, rid=None) -> List[str]:
    if alias_prints_to_drawings and "prints" in label_list:
        info_id("Aliasing label 'prints' -> 'drawings'", rid)
        label_list = ["drawings" if x == "prints" else x for x in label_list]
        # stable unique
        seen, fixed = set(), []
        for x in label_list:
            if x not in seen:
                seen.add(x); fixed.append(x)
        label_list = fixed
    return label_list

def _save_label_maps(model_dir: Path, id2label: Dict[int, str], label2id: Dict[str, int]):
    (model_dir / "labels.json").write_text(
        json.dumps({"id2label": id2label, "label2id": label2id}, indent=2, ensure_ascii=False),
        encoding="utf-8"
    )

def _sorted_id2label_str(id2label: Dict[int, str]) -> str:
    items = sorted(id2label.items(), key=lambda t: t[0])
    return ", ".join(f"{i}:{lab}" for i, lab in items)

def _default_train_files() -> List[str]:
    """
    Detect default training files in ORC_INTENT_TRAIN_DATA_DIR.
    Looks for intent_data.jsonl, intent_train_parts.jsonl, intent_train_drawings.jsonl
    """
    request_id = get_request_id()
    candidates = [
        os.path.join(ORC_INTENT_TRAIN_DATA_DIR, "intent_data.jsonl"),
        os.path.join(ORC_INTENT_TRAIN_DATA_DIR, "intent_train_parts.jsonl"),
        os.path.join(ORC_INTENT_TRAIN_DATA_DIR, "intent_train_drawings.jsonl"),
    ]
    found = [f for f in candidates if os.path.exists(f)]
    if found:
        info_id(f"[TRAIN] Auto-detected {len(found)} intent training files", request_id)
        for f in found:
            info_id(f" - {f}", request_id)
    else:
        warning_id(f"[TRAIN] No default intent training files found in {ORC_INTENT_TRAIN_DATA_DIR}", request_id)
    return found


# -------------------------
# System optimizer
# -------------------------
class SystemOptimizer:
    def __init__(self, request_id=None):
        self.request_id = request_id
        self.system_info = self._get_system_info()
        self.gpu_info = self._get_gpu_info()
        self.optimal_config = self._calculate_optimal_config()

    def _get_system_info(self):
        try:
            memory_gb = psutil.virtual_memory().total / (1024 ** 3)
            cpu_count = psutil.cpu_count(logical=True)
            cpu_count_physical = psutil.cpu_count(logical=False)
            info_id(f"System detected: {memory_gb:.1f}GB RAM, {cpu_count} logical CPUs ({cpu_count_physical} physical)",
                    self.request_id)
            info_id(f"Platform={platform.system()} | Processor={platform.processor()} | Python={platform.python_version()}",
                    self.request_id)
            return {
                "memory_gb": memory_gb, "cpu_count": cpu_count, "cpu_count_physical": cpu_count_physical,
                "platform": platform.system(), "processor": platform.processor(), "python_version": platform.python_version()
            }
        except Exception as e:
            warning_id(f"Failed to collect system info, using defaults: {e}", self.request_id)
            return {"memory_gb": 8, "cpu_count": 4, "cpu_count_physical": 2}

    def _get_gpu_info(self):
        d = {"available": False, "name": None, "memory_gb": 0, "compute_capability": None}
        try:
            if torch.cuda.is_available():
                d["available"] = True
                d["name"] = torch.cuda.get_device_name(0)
                props = torch.cuda.get_device_properties(0)
                d["memory_gb"] = props.total_memory / (1024 ** 3)
                d["compute_capability"] = f"{props.major}.{props.minor}"
                info_id(f"GPU detected: {d['name']} ({d['memory_gb']:.1f}GB, CC {d['compute_capability']})", self.request_id)
            else:
                info_id("No GPU detected - using CPU training", self.request_id)
        except Exception as e:
            warning_id(f"GPU probe failed: {e}", self.request_id)
        return d

    def _calculate_optimal_config(self):
        cfg = {}
        cfg["use_gpu"] = self.gpu_info["available"]
        cfg["fp16"] = self.gpu_info["available"]

        if self.gpu_info["available"]:
            m = self.gpu_info["memory_gb"]
            if m >= 16: cfg["batch_size"]=32; cfg["max_length"]=256
            elif m >= 8: cfg["batch_size"]=16; cfg["max_length"]=256
            elif m >= 4: cfg["batch_size"]=8;  cfg["max_length"]=128
            else:       cfg["batch_size"]=4;  cfg["max_length"]=128
        else:
            r = self.system_info["memory_gb"]
            if r >= 32: cfg["batch_size"]=16; cfg["max_length"]=256
            elif r >= 16: cfg["batch_size"]=8; cfg["max_length"]=256
            elif r >= 8:  cfg["batch_size"]=4; cfg["max_length"]=128
            else:         cfg["batch_size"]=2; cfg["max_length"]=128

        # Keep worker count modest for Windows / CPU
        cfg["num_workers"] = 0 if not self.gpu_info["available"] else min(4, max(1, self.system_info["cpu_count"] // 4))
        cfg["shuffle_buffer_size"] = 2000 if self.system_info["memory_gb"] >= 16 else (1000 if self.system_info["memory_gb"] >= 8 else 500)
        cfg["gradient_accumulation_steps"] = 1
        cfg["learning_rate"] = 5e-5
        return cfg

    def get_mode_cfg(self, max_examples):
        if max_examples is None:
            return {"num_epochs": 2 if self.gpu_info["available"] else 1, "eval_steps": 2000, "save_steps": 2000}
        if max_examples <= 10_000:
            return {"num_epochs": 4, "eval_steps": 200, "save_steps": 200}
        if max_examples <= 100_000:
            return {"num_epochs": 3, "eval_steps": 500, "save_steps": 500}
        return {"num_epochs": 2, "eval_steps": 1000, "save_steps": 1000}

    def print_summary(self):
        print("\n" + "="*60)
        print("SYSTEM OPTIMIZATION SUMMARY")
        print("="*60)
        print(f"Platform: {self.system_info['platform']}")
        print(f"RAM: {self.system_info['memory_gb']:.1f}GB")
        if self.gpu_info["available"]:
            print(f"GPU: {self.gpu_info['name']} ({self.gpu_info['memory_gb']:.1f}GB) CC {self.gpu_info['compute_capability']}")
        else:
            print("GPU: None (CPU training)")
        print("Optimal:", self.optimal_config)
        print("="*60 + "\n")


# -------------------------
# Audit helpers
# -------------------------
def _load_items(path: Path) -> List[Dict]:
    ext = path.suffix.lower()
    rows = []
    if ext == ".jsonl":
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                try:
                    o = json.loads(line)
                    if "text" in o and "label" in o:
                        rows.append({"text": str(o["text"]), "label": str(o["label"])})
                except Exception:
                    pass
    elif ext == ".json":
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(data, list):
                for o in data:
                    if isinstance(o, dict) and "text" in o and "label" in o:
                        rows.append({"text": str(o["text"]), "label": str(o["label"])})
        except Exception:
            pass
    elif ext in {".csv", ".tsv"}:
        delim = "," if ext == ".csv" else "\t"
        with path.open("r", encoding="utf-8", newline="") as f:
            r = csv.DictReader(f, delimiter=delim)
            for row in r:
                kl = {k.lower(): k for k in row}
                tk = kl.get("text"); lk = kl.get("label")
                if tk and lk:
                    rows.append({"text": str(row[tk]), "label": str(row[lk])})
    else:
        raise ValueError(f"Unsupported ext: {ext}")
    return rows

def _label_stats(rows: List[Dict]) -> Dict[str, int]:
    from collections import Counter
    return dict(Counter([r["label"] for r in rows]))

def _jaccard(a: str, b: str) -> float:
    A = set(_normalize_text(a).split())
    B = set(_normalize_text(b).split())
    if not A or not B: return 0.0
    return len(A & B) / len(A | B)

def _key_tokens(text: str) -> Tuple[str, str]:
    toks = _normalize_text(text).split()
    return (toks[0] if toks else "", toks[-1] if toks else "")

def audit_splits(train_files: List[Path], eval_files: List[Path], dev_files: List[Path],
                 near_thresh: float, rid=None) -> None:
    train = []; eval_ = []; dev = []
    for p in train_files: train.extend(_load_items(p))
    for p in eval_files:  eval_.extend(_load_items(p))
    for p in dev_files:   dev.extend(_load_items(p))

    def dedup(rows: List[Dict]) -> List[Dict]:
        seen = set(); out = []
        for r in rows:
            key = (_normalize_text(r["text"]), r["label"])
            if key not in seen:
                seen.add(key); out.append(r)
        return out

    train = dedup(train); eval_ = dedup(eval_); dev = dedup(dev)

    def stats(name, rows):
        cnt = _label_stats(rows)
        total = len(rows)
        info_id(f"[AUDIT] {name}: size={total} | labels={cnt}", rid)
        if total and max(cnt.values()) / total > 0.8:
            warning_id(f"[AUDIT] {name} is highly imbalanced (>80% one label)", rid)
        if name == "EVAL" and total < 300:
            warning_id(f"[AUDIT] EVAL set is small (<300). Metrics may be unstable.", rid)

    stats("TRAIN", train)
    stats("EVAL", eval_)
    if dev: stats("DEV", dev)

    def overlap(A: List[Dict], B: List[Dict], nameA: str, nameB: str):
        H = {_hash_text(r["text"]) for r in A}
        HB = {_hash_text(r["text"]) for r in B}
        exact = len(H & HB)
        from collections import defaultdict
        bucketA = defaultdict(list)
        for r in A: bucketA[_key_tokens(r["text"])].append(r["text"])
        near = 0
        for r in B:
            for cand in bucketA.get(_key_tokens(r["text"]), []):
                if _jaccard(cand, r["text"]) >= near_thresh:
                    near += 1; break
        if exact or near:
            warning_id(f"[AUDIT] Overlap {nameA}∩{nameB}: exact={exact}, near(≥{near_thresh})={near}", rid)
        else:
            info_id(f"[AUDIT] Overlap {nameA}∩{nameB}: none detected", rid)


# -------------------------
# Streaming dataset (intents) – fixed & optimized
# -------------------------
class StreamingIntentDataset(IterableDataset):
    """
    True streaming dataset with:
      - on-the-fly JSONL reading
      - shuffle buffer
      - optional near-duplicate drop via moving window
    """
    def __init__(
        self,
        files: List[Path],
        tokenizer: DistilBertTokenizerFast,
        label2id: Dict[str, int],
        max_length: int = 128,
        max_examples: Optional[int] = None,
        shuffle_buffer_size: int = 1000,
        dedup_window: int = 0,
        epoch: int = 0,
        request_id: Optional[str] = None
    ):
        super().__init__()
        self.files = [Path(f) for f in files]
        self.tokenizer = tokenizer
        self.label2id = label2id
        self.max_length = int(max_length)
        self.max_examples = None if max_examples in (None, 0) else int(max_examples)
        self.shuffle_buffer_size = max(1, int(shuffle_buffer_size))
        self.dedup_window = max(0, int(dedup_window))
        self.epoch = int(epoch)
        self.request_id = request_id

    def _yield_items(self):
        random.seed(42 + self.epoch)
        count = 0
        buffer = []
        recent_hashes: List[str] = []

        for fp in self.files:
            if not fp.exists():
                warning_id(f"Train file missing: {fp}", self.request_id)
                continue
            with fp.open("r", encoding="utf-8") as f:
                for line_num, line in enumerate(f, 1):
                    if self.max_examples is not None and count >= self.max_examples:
                        break
                    try:
                        obj = json.loads(line)
                        text = str(obj.get("text", ""))
                        label = str(obj.get("label", ""))
                        y = self.label2id.get(label)
                        if not text or y is None:
                            continue

                        if self.dedup_window > 0:
                            h = _hash_text(text)
                            if h in recent_hashes:
                                continue
                            recent_hashes.append(h)
                            if len(recent_hashes) > self.dedup_window:
                                recent_hashes.pop(0)

                        tok = self.tokenizer(
                            text,
                            truncation=True,
                            padding="max_length",
                            max_length=self.max_length,
                            return_tensors="pt"
                        )
                        item = {
                            "input_ids": tok["input_ids"].squeeze(0),
                            "attention_mask": tok["attention_mask"].squeeze(0),
                            "labels": torch.tensor(y, dtype=torch.long)
                        }
                        buffer.append(item)
                        if len(buffer) >= self.shuffle_buffer_size:
                            random.shuffle(buffer)
                            for it in buffer:
                                yield it
                            buffer.clear()

                        count += 1
                        if count % 1000 == 0:
                            info_id(f"Streamed {count} intent examples...", self.request_id)
                    except Exception as e:
                        warning_id(f"Skipping bad line in {fp.name} #{line_num}: {e}", self.request_id)
                        continue

        if buffer:
            random.shuffle(buffer)
            for it in buffer:
                yield it

    def __iter__(self):
        return self._yield_items()


# -------------------------
# Validation dataset (fixed signature)
# -------------------------
def build_val_dataset_from_files(
    files: List[Path],
    tokenizer: DistilBertTokenizerFast,
    label2id: Dict[str, int],
    max_length: int = 128,
    val_size: int = 5000,
    request_id: Optional[str] = None
):
    samples = []
    kept = 0
    for f in files:
        if not os.path.exists(f):
            warning_id(f"Eval file missing: {f}", request_id)
            continue
        with open(f, "r", encoding="utf-8") as fh:
            for line in fh:
                if kept >= val_size:
                    break
                try:
                    obj = json.loads(line)
                    text = obj.get("text", None)
                    label_str = obj.get("label", None)
                    if not text or label_str not in label2id:
                        continue
                    label_id = label2id[label_str]
                    tok = tokenizer(
                        text,
                        truncation=True,
                        padding="max_length",
                        max_length=max_length,
                        return_tensors="pt"
                    )
                    samples.append({
                        "input_ids": tok["input_ids"].squeeze(0),
                        "attention_mask": tok["attention_mask"].squeeze(0),
                        "labels": torch.tensor(label_id, dtype=torch.long),
                    })
                    kept += 1
                except Exception:
                    continue

    class ValDS(torch.utils.data.Dataset):
        def __init__(self, samples): self.samples = samples
        def __len__(self): return len(self.samples)
        def __getitem__(self, i): return self.samples[i]

    return ValDS(samples)


# -------------------------
# Metrics
# -------------------------
def compute_metrics(eval_pred, id2label: Dict[int, str]):
    # Hugging Face passes an EvalPrediction or a tuple
    if hasattr(eval_pred, "predictions"):
        logits, labels = eval_pred.predictions, eval_pred.label_ids
    else:
        logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)

    out = {}
    # accuracy
    try:
        out["accuracy"] = float((preds == labels).mean())
    except Exception:
        pass

    # macro_f1 if sklearn exists
    if _HAVE_SK:
        try:
            out["macro_f1"] = float(f1_score(labels, preds, average="macro"))
        except Exception:
            pass
    return out


# -------------------------
# Overfit Guard
# -------------------------
class OverfitGuardCallback(TrainerCallback):
    def __init__(self,
                 request_id,
                 gap_threshold: float = 0.30,
                 gap_patience: int = 3,
                 f1_decline_patience: int = 2,
                 min_evals: int = 3,
                 ema_alpha: float = 0.10):
        self.rid = request_id
        self.gap_threshold = float(gap_threshold)
        self.gap_patience = int(gap_patience)
        self.f1_decline_patience = int(f1_decline_patience)
        self.min_evals = int(min_evals)
        self.ema_alpha = float(ema_alpha)

        self.ema_train_loss = None
        self.eval_count = 0
        self.consec_gap_hits = 0
        self.consec_f1_down = 0
        self.prev_f1 = None
        self.prev_train_loss_for_trend = None

    def on_log(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, logs=None, **kwargs):
        if not logs:
            return
        if "loss" in logs:
            cur = float(logs["loss"])
            if self.ema_train_loss is None:
                self.ema_train_loss = cur
            else:
                self.ema_train_loss = self.ema_train_loss * (1.0 - self.ema_alpha) + cur * self.ema_alpha

    def on_evaluate(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, metrics=None, **kwargs):
        if not metrics:
            return control
        self.eval_count += 1

        eval_loss = float(metrics.get("eval_loss", float("nan")))
        eval_f1 = metrics.get("eval_macro_f1", None)
        if eval_f1 is not None:
            eval_f1 = float(eval_f1)

        if self.ema_train_loss is not None and not np.isnan(eval_loss):
            gap = eval_loss - self.ema_train_loss
            info_id(f"[OverfitGuard] eval_loss={eval_loss:.4f}, EMA(train_loss)={self.ema_train_loss:.4f}, gap={gap:.4f}", self.rid)
            if gap >= self.gap_threshold:
                self.consec_gap_hits += 1
                warning_id(f"[OverfitGuard] gap≥{self.gap_threshold:.2f} for {self.consec_gap_hits}/{self.gap_patience} evals", self.rid)
            else:
                self.consec_gap_hits = 0
        else:
            info_id("[OverfitGuard] Insufficient data for gap check (need train EMA & eval_loss)", self.rid)

        train_improving = (self.prev_train_loss_for_trend is None) or \
                          (self.ema_train_loss is not None and self.ema_train_loss < self.prev_train_loss_for_trend)

        if eval_f1 is not None and self.prev_f1 is not None:
            if eval_f1 + 1e-6 < self.prev_f1 and train_improving:
                self.consec_f1_down += 1
                warning_id(f"[OverfitGuard] macro-F1 down ({self.prev_f1:.4f}→{eval_f1:.4f}) "
                           f"({self.consec_f1_down}/{self.f1_decline_patience}) with improving train loss", self.rid)
            else:
                self.consec_f1_down = 0

        if self.ema_train_loss is not None:
            self.prev_train_loss_for_trend = self.ema_train_loss
        if eval_f1 is not None:
            self.prev_f1 = eval_f1

        if self.eval_count >= self.min_evals:
            if self.consec_gap_hits >= self.gap_patience:
                warning_id("[OverfitGuard] Stopping training due to persistent generalization gap.", self.rid)
                control.should_training_stop = True
            elif self.consec_f1_down >= self.f1_decline_patience:
                warning_id("[OverfitGuard] Stopping training due to consecutive macro-F1 declines with improving train loss.", self.rid)
                control.should_training_stop = True

        return control


# -------------------------
# CLI
# -------------------------
def parse_args():
    p = argparse.ArgumentParser("Adaptive streaming trainer for INTENT classification")

    DEFAULT_TRAIN = [str(Path(ORC_INTENT_TRAIN_DATA_DIR) / "intent_data.jsonl")]
    DEFAULT_EVAL  = [str(Path(ORC_INTENT_TRAIN_DATA_DIR) / "intent_eval.jsonl")]

    p.add_argument("--mode", choices=["fast", "small", "medium", "full"], default="small")
    p.add_argument("--max-examples", type=int, default=None)
    p.add_argument("--force-cpu", action="store_true")
    p.add_argument("--override-batch-size", type=int, default=None)
    p.add_argument("--override-max-length", type=int, default=None)
    p.add_argument("--probe-model-folder-labels", action="store_true")

    # Data sources (defaults baked in)
    p.add_argument("--train-files", nargs="*", default=DEFAULT_TRAIN,
                   help="JSONL/CSV/TSV with columns text,label. Default: C:\\data\\intent_train.jsonl")
    p.add_argument("--eval-files", nargs="*", default=DEFAULT_EVAL,
                   help="Separate eval/validation files. Default: C:\\data\\intent_eval.jsonl")
    p.add_argument("--dev-files", nargs="*", default=None,
                   help="Optional separate dev files (audited only)")
    p.add_argument("--val-size", type=int, default=5000, help="Max validation rows to load")

    p.add_argument("--base-model", default="distilbert-base-uncased")

    # Label handling
    p.add_argument("--labels", nargs="*", default=None,
                   help="Explicit label list (order fixes ids). If omitted, discovered from data.")
    p.add_argument("--alias-prints-to-drawings", action="store_true",
                   help="Map 'prints' → 'drawings' in dataset labels before training.")

    # Audit & de-dup (audit ON by default; add --no-audit to disable)
    p.add_argument("--audit", action="store_true", default=True, help="Run split audit before training")
    p.add_argument("--no-audit", dest="audit", action="store_false", help="Disable audit")
    p.add_argument("--audit-near-thresh", type=float, default=0.85,
                   help="Jaccard threshold for near-duplicate overlap")
    p.add_argument("--dedup-window", type=int, default=0,
                   help="If >0, drop repeated normalized texts in a moving window during streaming")

    # Early stopping (default 5 to match your command)
    p.add_argument("--early-stop", type=int, default=5, help="Early stopping patience (steps)")

    # Overfit guard (ON by default to match your command; add --no-overfit-guard to disable)
    p.add_argument("--overfit-guard", action="store_true", default=True,
                   help="Enable overfitting guard (stops when gap/trend indicates overfit).")
    p.add_argument("--no-overfit-guard", dest="overfit_guard", action="store_false", help="Disable overfit guard")
    p.add_argument("--gap-threshold", type=float, default=0.30,
                   help="Min (eval_loss - EMA(train_loss)) to consider a gap spike (default 0.30).")
    p.add_argument("--gap-patience", type=int, default=3,
                   help="Stop after this many consecutive evals with gap above threshold (default 3).")
    p.add_argument("--f1-decline-patience", type=int, default=2,
                   help="Stop if macro-F1 declines this many consecutive evals while train loss improves.")
    p.add_argument("--min-evals", type=int, default=3,
                   help="Don’t trigger guard until at least this many eval cycles have happened.")

    p.add_argument("--quick1k", action="store_true",
                   help="Run a 1,000-row quick test on CPU with safe defaults")
    p.add_argument("--no-interactive-warmup", action="store_true",
                   help="When using --max-examples warm-up mode, exit after eval (skip interactive prompt)")

    # Power/Memory knobs
    p.add_argument("--gradient-checkpointing", action="store_true",
                   help="Enable gradient checkpointing to save memory at a small compute cost.")
    return p.parse_args()


# -------------------------
# Main
# -------------------------
def main():
    # reproducibility
    try:
        torch.set_float32_matmul_precision("high")  # harmless on CPU
    except Exception:
        pass
    random.seed(42); np.random.seed(42); torch.manual_seed(42)

    set_request_id()
    rid = get_request_id()
    info_id("Starting adaptive streaming training (INTENT)", rid)

    args = parse_args()

    # --- Quick 1k preset ---
    if args.quick1k:
        args.force_cpu = True
        args.mode = args.mode or "small"
        args.max_examples = 1000
        if getattr(args, "override_batch_size", None) in (None, 0):
            args.override_batch_size = 8
        if getattr(args, "override_max_length", None) in (None, 0):
            args.override_max_length = 256
        if not args.labels:
            args.labels = ["drawings", "parts"]
        args.no_interactive_warmup = True

    # Resolve train/eval/dev files
    train_files = [Path(p).resolve() for p in (args.train_files or [])]
    eval_files  = [Path(p).resolve() for p in (args.eval_files  or [])]
    dev_files   = [Path(p).resolve() for p in (args.dev_files   or [])]

    if not train_files:
        fallback = _default_train_files()
        if not fallback:
            error_id("No intent train files found. Pass --train-files path/to/intent_train.jsonl", rid)
            return
        train_files = [Path(p).resolve() for p in fallback]

    info_id(f"Train files: {[str(p) for p in train_files]}", rid)
    if eval_files:
        info_id(f"Eval files: {[str(p) for p in eval_files]}", rid)
    if dev_files:
        info_id(f"Dev files: {[str(p) for p in dev_files]}", rid)

    optimizer = SystemOptimizer(request_id=rid)
    optimizer.print_summary()

    # Determine max_examples from mode if not provided
    if not args.max_examples:
        args.max_examples = {"fast": 2_000, "small": 10_000, "medium": 100_000, "full": None}[args.mode]

    info_id(f"Python executable: {sys.executable}", rid)
    import transformers
    info_id(f"Transformers version: {transformers.__version__}", rid)

    # Discover label set
    if args.labels:
        label_list = list(dict.fromkeys(args.labels))
    else:
        label_list = _read_label_set(train_files + eval_files)
        if not label_list:
            error_id("Could not discover any labels from training/eval files.", rid)
            return

    # Aliases
    label_list = _apply_aliases(label_list, alias_prints_to_drawings=args.alias_prints_to_drawings, rid=rid)
    id2label = {i: lab for i, lab in enumerate(label_list)}
    label2id = {lab: i for i, lab in id2label.items()}
    info_id(f"Label order (ids): {', '.join(f'{i}:{lab}' for i, lab in id2label.items())}", rid)

    # Optional audit
    if args.audit and (eval_files or dev_files):
        info_id("Running dataset split audit...", rid)
        with log_timed_operation("audit_splits", rid):
            audit_splits(train_files, eval_files, dev_files, args.audit_near_thresh, rid)

    # Tokenizer / model
    with log_timed_operation("tokenizer_init", rid):
        try:
            tokenizer = DistilBertTokenizerFast.from_pretrained(args.base_model)
        except Exception as e:
            error_id(f"Failed loading tokenizer from '{args.base_model}': {e}", rid)
            return

    with log_timed_operation("model_init", rid):
        try:
            model = DistilBertForSequenceClassification.from_pretrained(
                args.base_model,
                num_labels=len(label_list),
                id2label=id2label,
                label2id=label2id
            )
        except Exception as e:
            error_id(f"Failed loading model from '{args.base_model}': {e}", rid)
            return

    # Enable gradient checkpointing if requested
    if args.gradient_checkpointing:
        try:
            model.gradient_checkpointing_enable()
            info_id("Enabled gradient checkpointing.", rid)
        except Exception as e:
            warning_id(f"Could not enable gradient checkpointing: {e}", rid)

    cfg = optimizer.optimal_config

    # Runtime overrides
    if args.override_batch_size: cfg["batch_size"] = int(args.override_batch_size)
    if args.override_max_length: cfg["max_length"] = int(args.override_max_length)
    if args.force_cpu:
        cfg["fp16"] = False

    mode_cfg = optimizer.get_mode_cfg(args.max_examples)

    # Validation set
    info_id("Preparing validation dataset...", rid)
    with log_timed_operation("build_validation_dataset", rid):
        if eval_files:
            val_ds = build_val_dataset_from_files(
                eval_files, tokenizer, label2id,
                max_length=cfg["max_length"], val_size=args.val_size, request_id=rid
            )
        else:
            warning_id("No --eval-files provided. Building validation from train files (risk of leakage).", rid)
            val_ds = build_val_dataset_from_files(
                train_files, tokenizer, label2id,
                max_length=cfg["max_length"], val_size=args.val_size, request_id=rid
            )
    info_id(f"Validation size: {len(val_ds)}", rid)

    data_collator = DataCollatorWithPadding(tokenizer)
    steps_per_epoch = max(1, (args.max_examples or 16000) // cfg["batch_size"])
    info_id(f"Estimated steps per epoch: {steps_per_epoch}", rid)

    # Training arguments
    training_args = TrainingArguments(
        output_dir=ORC_INTENT_MODEL_DIR,
        evaluation_strategy="steps",
        save_strategy="steps",
        eval_steps=mode_cfg["eval_steps"],
        save_steps=mode_cfg["save_steps"],
        save_total_limit=3,
        load_best_model_at_end=True,
        metric_for_best_model="macro_f1" if _HAVE_SK else "accuracy",
        greater_is_better=True,

        learning_rate=cfg["learning_rate"],
        weight_decay=0.05,
        warmup_ratio=0.10,
        per_device_train_batch_size=cfg["batch_size"],
        per_device_eval_batch_size=cfg["batch_size"],
        gradient_accumulation_steps=cfg["gradient_accumulation_steps"],
        num_train_epochs=mode_cfg["num_epochs"],

        logging_dir=str(Path(ORC_INTENT_MODEL_DIR) / "logs"),
        logging_steps=50,
        report_to="none",
        seed=42,
        fp16=cfg["fp16"] and torch.cuda.is_available(),
        dataloader_pin_memory=bool(torch.cuda.is_available()),
        dataloader_num_workers=cfg["num_workers"],
        remove_unused_columns=True,
    )

    # Build streaming dataset via Trainer subclass
    streaming_cfg = {
        "files": train_files,
        "tokenizer": tokenizer,
        "label2id": label2id,
        "max_length": cfg["max_length"],
        "max_examples": args.max_examples,
        "shuffle_buffer_size": cfg["shuffle_buffer_size"],
        "dedup_window": max(0, int(args.dedup_window)),
    }

    class StreamingTrainer(Trainer):
        def __init__(self, streaming_dataset_config=None, request_id=None, **kwargs):
            super().__init__(**kwargs)
            self.streaming_dataset_config = streaming_dataset_config or {}
            self.current_epoch = 0
            self.request_id = request_id

        def get_train_dataloader(self):
            ds = StreamingIntentDataset(
                epoch=self.current_epoch,
                request_id=self.request_id,
                **self.streaming_dataset_config
            )
            return DataLoader(
                ds,
                batch_size=self.args.per_device_train_batch_size,
                collate_fn=self.data_collator,
                num_workers=self.args.dataloader_num_workers,
                pin_memory=self.args.dataloader_pin_memory
            )

        def _inner_training_loop(self, **kwargs):
            info_id(f"Starting epoch {self.current_epoch}", self.request_id)
            result = super()._inner_training_loop(**kwargs)
            info_id(f"Completed epoch {self.current_epoch}", self.request_id)
            self.current_epoch += 1
            return result

    callbacks_list = [EarlyStoppingCallback(early_stopping_patience=max(1, int(args.early_stop)))]
    if args.overfit_guard:
        callbacks_list.append(
            OverfitGuardCallback(
                request_id=rid,
                gap_threshold=args.gap_threshold,
                gap_patience=args.gap_patience,
                f1_decline_patience=args.f1_decline_patience,
                min_evals=args.min_evals,
                ema_alpha=0.10,
            )
        )

    trainer = StreamingTrainer(
        model=model,
        args=training_args,
        train_dataset="placeholder",
        eval_dataset=val_ds,
        data_collator=data_collator,
        compute_metrics=lambda p: compute_metrics(p, id2label),
        streaming_dataset_config=streaming_cfg,
        callbacks=callbacks_list,
        request_id=rid,
    )

    info_id("Starting adaptive streaming training...", rid)
    with log_timed_operation("trainer_train", rid):
        trainer.train()

    best_ckpt = trainer.state.best_model_checkpoint or training_args.output_dir
    info_id(f"Best checkpoint: {best_ckpt}", rid)

    with log_timed_operation("load_best_model", rid):
        best_model = DistilBertForSequenceClassification.from_pretrained(best_ckpt)
        best_model.config.id2label = id2label
        best_model.config.label2id = label2id

    dst = Path(ORC_INTENT_MODEL_DIR)
    with log_timed_operation("export_best_checkpoint", rid):
        shutil.rmtree(dst, ignore_errors=True)
        dst.mkdir(parents=True, exist_ok=True)
        best_model.save_pretrained(dst)
        tokenizer.save_pretrained(dst)
        _save_label_maps(dst, id2label, label2id)
        (dst / "_BEST_CHECKPOINT.txt").write_text(f"Exported from: {best_ckpt}\n", encoding="utf-8")
        info_id(f"Exported best checkpoint to: {dst}", rid)

    info_id("Evaluating final (best) model...", rid)
    with log_timed_operation("trainer_evaluate", rid):
        metrics = trainer.evaluate()
    for k, v in metrics.items():
        info_id(f"Final {k}: {v}", rid)

    # quick REPL to sanity-check the model
    try:
        while True:
            q = input("\nType a sentence to classify (Enter to quit): ").strip()
            if not q:
                break
            enc = tokenizer(q, return_tensors="pt", truncation=True, max_length=cfg["max_length"])
            with torch.no_grad():
                out = best_model(**enc)
                pred = out.logits.argmax(dim=-1).item()
                lab = id2label[pred]
                print(f"→ {lab}")
    except KeyboardInterrupt:
        pass

    info_id("Adaptive streaming INTENT training complete!", rid)


if __name__ == "__main__":
    set_request_id()
    main()
