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
os.environ["TRANSFORMERS_NO_TORCHVISION"] = "1"
# ---------------------------------------------------------
# FORCE EMTAC ENV LOAD (required for offline + PyCharm)
# ---------------------------------------------------------
from dotenv import load_dotenv
from pathlib import Path
import os
from sklearn.metrics import f1_score
from collections import defaultdict
ENV_PATH = Path(r"E:\emtac\dev_env\.env")

if ENV_PATH.exists():
    load_dotenv(ENV_PATH)
else:
    raise RuntimeError(f"ENV file not found: {ENV_PATH}")

# Optional sanity log (can remove later)
print("ENV LOADED | MODELS_DISTILBERT_INTENT =", os.getenv("MODELS_DISTILBERT_INTENT"))

import numpy as np
import torch
import psutil

from torch.utils.data import IterableDataset, DataLoader
from datasets import Dataset
from transformers import (
    DistilBertTokenizerFast,
    DistilBertForSequenceClassification,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
    EarlyStoppingCallback,
    TrainerCallback,
)
from transformers.trainer_callback import TrainerControl, TrainerState

# === Config paths (your project) ===
from configuration.config import (MODEL_TRAINING_INTENT_TRAIN_DATA_DIR,MODEL_TRAINING_INTENT_MODEL_DIR,
                                  MODELS_DISTILBERT_INTENT)

# === Your custom logger ===
from configuration.log_config import (
    debug_id, info_id, warning_id, error_id, critical_id,
    with_request_id, set_request_id, get_request_id,
    log_timed_operation
)
from modules.gpu.gpu_training_adapter import  GPUTrainingAdapter


logging.basicConfig(level=logging.WARNING)  # keep root quiet; use your logger


# -------------------------
# Utilities
# -------------------------

def _list_existing(paths: Iterable[Path]) -> List[Path]:
    return [p for p in paths if p.exists() and p.is_file()]

def _default_train_files() -> List[Path]:
    """Pick sensible defaults from MODEL_TRAINING_INTENT_TRAIN_DATA_DIR."""
    root = Path(MODEL_TRAINING_INTENT_TRAIN_DATA_DIR)
    candidates = [
        root / "intent_train.jsonl",
        root / "intent_train_parts.jsonl",
        root / "intent_train_drawings.jsonl",
    ]
    found = _list_existing(candidates)
    if not found:
        found = _list_existing([root / "intent_train.jsonl"])
    return found

def _read_jsonl(p: Path) -> List[Dict]:
    rows = []
    with p.open("r", encoding="utf-8") as f:
        for line in f:
            try:
                o = json.loads(line)
                if "text" in o and "label" in o:
                    rows.append({"text": str(o["text"]), "label": str(o["label"])})
            except Exception:
                pass
    return rows

def _read_json(p: Path) -> List[Dict]:
    rows = []
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
        if isinstance(data, list):
            for o in data:
                if isinstance(o, dict) and "text" in o and "label" in o:
                    rows.append({"text": str(o["text"]), "label": str(o["label"])})
    except Exception:
        pass
    return rows

def _read_tabular(p: Path, delim: str) -> List[Dict]:
    rows = []
    with p.open("r", encoding="utf-8", newline="") as f:
        r = csv.DictReader(f, delimiter=delim)
        for row in r:
            kl = {k.lower(): k for k in row}
            tk = kl.get("text"); lk = kl.get("label")
            if tk and lk:
                rows.append({"text": str(row[tk]), "label": str(row[lk])})
    return rows

def _load_items(path: Path) -> List[Dict]:
    ext = path.suffix.lower()
    if ext == ".jsonl": return _read_jsonl(path)
    if ext == ".json":  return _read_json(path)
    if ext == ".csv":   return _read_tabular(path, ",")
    if ext == ".tsv":   return _read_tabular(path, "\t")
    raise ValueError(f"Unsupported ext: {ext}")

def _normalize_text(s: str) -> str:
    s = s.lower()
    s = re.sub(r"\s+", " ", s)
    s = re.sub(r"[^\w\s-]", "", s)
    return s.strip()

def _hash_text(s: str) -> str:
    return hashlib.sha256(_normalize_text(s).encode("utf-8")).hexdigest()

def _read_label_set(files: List[Path], rid=None) -> List[str]:
    labels = []
    seen = set()
    for fp in files:
        try:
            with fp.open("r", encoding="utf-8") as f:
                for line in f:
                    try:
                        obj = json.loads(line)
                        lab = str(obj.get("label", "")).strip()
                        if lab and lab not in seen:
                            seen.add(lab); labels.append(lab)
                    except Exception:
                        continue
        except Exception as e:
            warning_id(f"Could not scan labels from {fp}: {e}", rid)
    labels = sorted(labels)
    return labels

def _apply_aliases(label_list: List[str], alias_prints_to_drawings: bool, rid=None) -> List[str]:
    if alias_prints_to_drawings and "prints" in label_list:
        info_id("Aliasing label 'prints' -> 'drawings'", rid)
        label_list = ["drawings" if x == "prints" else x for x in label_list]
        seen = set(); fixed = []
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

# -------------------------
# Class-weighted loss helper
# -------------------------
def compute_class_weights_from_files(
    train_files: List[Path],
    label2id: Dict[str, int],
    rid=None,
) -> torch.Tensor:
    """
    Compute inverse-frequency class weights for CrossEntropyLoss.
    """
    counts = {lab: 0 for lab in label2id}
    total = 0

    for fp in train_files:
        try:
            with fp.open("r", encoding="utf-8") as f:
                for line in f:
                    try:
                        obj = json.loads(line)
                        lab = str(obj.get("label"))
                        if lab in counts:
                            counts[lab] += 1
                            total += 1
                    except Exception:
                        continue
        except Exception as e:
            warning_id(f"Failed reading {fp} for class weights: {e}", rid)

    if total == 0:
        raise RuntimeError("No labels found for computing class weights")

    weights = []
    for lab, idx in label2id.items():
        c = counts.get(lab, 0)
        if c == 0:
            warning_id(f"Label '{lab}' has zero samples — weight set to 0", rid)
            weights.append(0.0)
        else:
            weights.append(total / (len(counts) * c))

    w = torch.tensor(weights, dtype=torch.float)
    info_id(f"Class weights: {dict(zip(label2id.keys(), weights))}", rid)
    return w


# -------------------------
# Model wrapper with weighted loss
# -------------------------
class WeightedIntentModel(DistilBertForSequenceClassification):
    def __init__(self, *args, class_weights: Optional[torch.Tensor] = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights

    def forward(self, **kwargs):
        labels = kwargs.get("labels")

        # Let HF compute logits normally
        outputs = super().forward(**kwargs)

        # If labels exist, we MUST return a loss
        if labels is not None:
            if self.class_weights is not None:
                loss_fct = torch.nn.CrossEntropyLoss(
                    weight=self.class_weights.to(outputs.logits.device)
                )
                loss = loss_fct(outputs.logits, labels)
            else:
                # fallback (should never happen, but Trainer-safe)
                loss = torch.nn.functional.cross_entropy(
                    outputs.logits, labels
                )

            outputs.loss = loss

        return outputs


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

def overlap(
    A: List[Dict],
    B: List[Dict],
    nameA: str,
    nameB: str,
    rid: str,
    near_thresh: float = 0.85,
):
    # Exact overlap (hash-based)
    H = {_hash_text(r["text"]) for r in A}
    HB = {_hash_text(r["text"]) for r in B}
    exact = len(H & HB)

    # Near overlap (token bucket + Jaccard)
    bucketA = defaultdict(list)
    for r in A:
        bucketA[_key_tokens(r["text"])].append(r["text"])

    near = 0
    for r in B:
        for cand in bucketA.get(_key_tokens(r["text"]), []):
            if _jaccard(cand, r["text"]) >= near_thresh:
                near += 1
                break

    # Logging (ASCII only)
    if exact or near:
        warning_id(
            f"[AUDIT] Overlap {nameA}&{nameB}: "
            f"exact={exact}, near(>={near_thresh})={near}",
            rid,
        )
    else:
        info_id(
            f"[AUDIT] Overlap {nameA}&{nameB}: none detected",
            rid,
        )

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

    if eval_:
        overlap(train, eval_, "TRAIN", "EVAL", rid)

    if dev:
        overlap(train, dev, "TRAIN", "DEV", rid)
        if eval_:
            overlap(dev, eval_, "DEV", "EVAL", rid)


# -------------------------
# Streaming dataset (intents)
# -------------------------
class StreamingIntentDataset(IterableDataset):
    """
    Streaming dataset for intent classification with rotating exposure windows.

    - Parts: always included
    - Drawings: rotated per epoch via offset + window size
    - Supports deduplication, per-label caps, and bounded shuffle
    """

    def __init__(
        self,
        files: List[Path],
        tokenizer,
        label2id: Dict[str, int],
        max_length=256,
        max_examples=None,
        shuffle_buffer_size=1000,
        skip_examples=0,
        epoch=0,
        request_id=None,
        dedup_window: int = 0,

        # 🔁 ROTATION CONFIG
        drawings_window_size: Optional[int] = None,
        drawings_window_offset: int = 0,

        max_drawings_per_epoch: Optional[int] = None,
        max_parts_per_epoch: Optional[int] = None,
    ):
        super().__init__()
        self.files = files
        self.tokenizer = tokenizer
        self.label2id = label2id
        self.max_length = max_length
        self.max_examples = max_examples
        self.shuffle_buffer_size = shuffle_buffer_size
        self.skip_examples = skip_examples
        self.epoch = epoch
        self.request_id = request_id
        self.dedup_window = dedup_window

        self.drawings_window_size = drawings_window_size
        self.drawings_window_offset = max(0, int(drawings_window_offset))

        self.max_drawings_per_epoch = max_drawings_per_epoch
        self.max_parts_per_epoch = max_parts_per_epoch

    def __iter__(self):
        random.seed(42 + self.epoch)

        count = 0
        buffer = []
        recent: List[str] = []

        per_label_counts = {lab: 0 for lab in self.label2id}

        # 🔁 Rotation counters (drawings only)
        drawings_seen = 0
        drawings_skipped = 0

        for fp in self.files:
            try:
                with fp.open("r", encoding="utf-8") as f:
                    for line_num, line in enumerate(f, 1):

                        if line_num <= self.skip_examples:
                            continue
                        if self.max_examples and count >= self.max_examples:
                            break

                        try:
                            obj = json.loads(line)
                            text = str(obj.get("text", ""))
                            label = str(obj.get("label", ""))
                            y = self.label2id.get(label)

                            if not text or y is None:
                                continue

                            # =====================================================
                            # 🔁 ROTATING DRAWINGS EXPOSURE WINDOW
                            # =====================================================
                            if label == "drawings":

                                # Skip until offset is reached
                                if drawings_skipped < self.drawings_window_offset:
                                    drawings_skipped += 1
                                    continue

                                # Enforce window size
                                if (
                                    self.drawings_window_size is not None
                                    and drawings_seen >= self.drawings_window_size
                                ):
                                    continue

                                drawings_seen += 1

                            # =====================================================
                            # 🔒 PER-LABEL CAPS
                            # =====================================================
                            if (
                                label == "drawings"
                                and self.max_drawings_per_epoch is not None
                                and per_label_counts[label] >= self.max_drawings_per_epoch
                            ):
                                continue

                            if (
                                label == "parts"
                                and self.max_parts_per_epoch is not None
                                and per_label_counts[label] >= self.max_parts_per_epoch
                            ):
                                continue

                            # =====================================================
                            # 🔁 DEDUP WINDOW
                            # =====================================================
                            if self.dedup_window > 0:
                                h = _hash_text(text)
                                if h in recent:
                                    continue
                                recent.append(h)
                                if len(recent) > self.dedup_window:
                                    recent.pop(0)

                            # =====================================================
                            # 🔤 TOKENIZATION
                            # =====================================================
                            tok = self.tokenizer(
                                text,
                                truncation=True,
                                padding="max_length",
                                max_length=self.max_length,
                                return_tensors="pt",
                            )

                            buffer.append({
                                "input_ids": tok["input_ids"].squeeze(0),
                                "attention_mask": tok["attention_mask"].squeeze(0),
                                "labels": torch.tensor(y, dtype=torch.long),
                            })

                            per_label_counts[label] += 1
                            count += 1

                            # =====================================================
                            # 🔀 SHUFFLE + YIELD
                            # =====================================================
                            if len(buffer) >= self.shuffle_buffer_size:
                                random.shuffle(buffer)
                                for it in buffer:
                                    yield it
                                buffer = []

                            if count % 1000 == 0:
                                info_id(
                                    f"Streamed {count} intent examples "
                                    f"(epoch={self.epoch}, drawings_seen={drawings_seen})",
                                    self.request_id,
                                )

                        except Exception as e:
                            warning_id(
                                f"Skipping bad line in {fp.name} #{line_num}: {e}",
                                self.request_id,
                            )

            except FileNotFoundError as e:
                warning_id(f"Train file missing: {e}", self.request_id)

        # =====================================================
        # FLUSH REMAINING BUFFER
        # =====================================================
        if buffer:
            random.shuffle(buffer)
            for it in buffer:
                yield it

        # =====================================================
        # COVERAGE WARNING
        # =====================================================
        if self.drawings_window_size is not None and drawings_seen == 0:
            warning_id(
                f"[ExposureWindow] No drawings seen in epoch={self.epoch} "
                f"(offset={self.drawings_window_offset}, window_size={self.drawings_window_size})",
                self.request_id,
            )

# -------------------------
# Validation dataset
# -------------------------
def build_val_dataset_from_files(
    files: List[Path],
    tokenizer,
    label2id: Dict[str, int],
    max_length=256,
    val_size: int = 5000,
    request_id=None,
) -> Dataset:
    rows = []
    bad = 0

    for fp in files:
        try:
            with fp.open("r", encoding="utf-8") as f:
                for _, line in enumerate(f):
                    if val_size and len(rows) >= val_size:
                        break
                    try:
                        obj = json.loads(line)
                        text = str(obj["text"])
                        lab = str(obj["label"])
                        if lab not in label2id:
                            continue
                        rows.append({
                            "text": text,
                            "labels": int(label2id[lab]),
                        })
                    except Exception:
                        bad += 1
        except Exception as e:
            warning_id(f"Validation read error from {fp}: {e}", request_id)

    info_id(f"Validation collected={len(rows)}, bad_lines={bad}", request_id)

    ds = Dataset.from_list(rows)

    def tok(batch):
        tokens = tokenizer(
            batch["text"],
            truncation=True,
            padding="max_length",
            max_length=max_length,
        )
        tokens["labels"] = batch["labels"]
        return tokens

    # ✅ CRITICAL: remove ONLY 'text'
    return ds.map(
        tok,
        batched=True,
        remove_columns=["text"],
    )


# -------------------------
# Metrics
# -------------------------
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)

    accuracy = float((preds == labels).mean())

    if f1_score is not None:
        try:
            macro_f1 = float(f1_score(labels, preds, average="macro"))
        except Exception:
            macro_f1 = 0.0
    else:
        macro_f1 = 0.0

    return {
        "eval_accuracy": accuracy,
        "eval_macro_f1": macro_f1,
    }


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
                warning_id(f"[OverfitGuard] gap≥={self.gap_threshold:.2f} for {self.consec_gap_hits}/{self.gap_patience} evals", self.rid)
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

    # Built-in defaults to match your desired command
    DEFAULT_TRAIN = []
    DEFAULT_EVAL = []

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
    p.add_argument("--fresh-start",action="store_true",
                   help="Ignore existing intent model and start from base model",)
    p.add_argument("--drawings-window-size", type=int, default=100_000)
    p.add_argument("--drawings-window-offset", type=int, default=0)

    return p.parse_args()

def resolve_intent_base_model(rid, fresh_start: bool = False) -> str:
    """
    Resolves which model directory to load:
    - --fresh-start → clean offline base DistilBERT (immutable)
    - otherwise     → resume from trained intent model if valid
    """

    raw = os.getenv("MODELS_DISTILBERT_INTENT")
    if not raw:
        raise RuntimeError("MODELS_DISTILBERT_INTENT not set")

    base_model_dir = Path(raw.strip()).resolve()
    trained_model_dir = Path(MODEL_TRAINING_INTENT_MODEL_DIR).resolve()

    def _is_valid_model_dir(p: Path) -> bool:
        if not p.exists():
            return False

        # One of these MUST exist
        has_weights = (
            (p / "model.safetensors").exists()
            or (p / "pytorch_model.bin").exists()
        )

        # These MUST exist
        has_meta = all(
            (p / f).exists()
            for f in ["config.json", "tokenizer.json"]
        )

        return has_weights and has_meta

    # --------------------------------------------------
    # FRESH START → always use offline base
    # --------------------------------------------------
    if fresh_start:
        if not _is_valid_model_dir(base_model_dir):
            critical_id(
                f"Offline base intent model missing or incomplete at {base_model_dir}",
                rid,
            )
            info_id(
                f"Base dir contents: {[f.name for f in base_model_dir.iterdir()]}",
                rid,
            )
            raise RuntimeError("Offline base model invalid")

        info_id(
            f"Fresh start requested — using CLEAN base intent model: {base_model_dir}",
            rid,
        )
        return str(base_model_dir)

    # --------------------------------------------------
    # RESUME TRAINING → use trained model if valid
    # --------------------------------------------------
    if _is_valid_model_dir(trained_model_dir):
        info_id(
            f"Resuming training from existing intent model: {trained_model_dir}",
            rid,
        )
        return str(trained_model_dir)

    # --------------------------------------------------
    # FALLBACK → trained missing, base exists
    # --------------------------------------------------
    if _is_valid_model_dir(base_model_dir):
        warning_id(
            "Trained intent model missing — falling back to clean base model.",
            rid,
        )
        return str(base_model_dir)

    # --------------------------------------------------
    # FAILURE
    # --------------------------------------------------
    critical_id(
        "No valid intent model found (neither trained nor base).",
        rid,
    )
    raise RuntimeError("Intent model resolution failed")

# -------------------------
# Main
# -------------------------
def main():
    set_request_id()
    rid = get_request_id()
    info_id("Starting adaptive streaming training (INTENT)", rid)

    args = parse_args()

    args.base_model = resolve_intent_base_model(
        rid,
        fresh_start=args.fresh_start,
    )

    # ---------------------------------------------------------
    # GPU adapter initialization (NEW)
    # ---------------------------------------------------------
    gpu_adapter = GPUTrainingAdapter()
    info_id(f"[GPU-ADAPTER] {gpu_adapter.describe()}", rid)

    # ---------------------------------------------------------
    # Resolve train / eval / dev files (PROJECT-SAFE)
    # ---------------------------------------------------------
    def _resolve_files(cli_files, default_names):
        if cli_files:
            return [Path(p).resolve() for p in cli_files]
        root = Path(MODEL_TRAINING_INTENT_TRAIN_DATA_DIR)
        files = [(root / name) for name in default_names]
        return [p for p in files if p.exists()]

    train_files = _resolve_files(
        args.train_files,
        [
            "intent_train_drawings.jsonl"
            "intent_train_parts.jsonl",
            "intent_train.jsonl",
        ],
    )

    eval_files = _resolve_files(
        args.eval_files,
        [
            "intent_val.jsonl",
            "intent_eval_parts.jsonl",
            "intent_eval_drawings.jsonl",

        ],
    )

    dev_files = [Path(p).resolve() for p in (args.dev_files or [])]

    if not train_files:
        critical_id(
            f"No intent training files found under {MODEL_TRAINING_INTENT_TRAIN_DATA_DIR}",
            rid,
        )
        return

    info_id(f"Train files ({len(train_files)}): {[str(p) for p in train_files]}", rid)
    info_id(f"Eval files  ({len(eval_files)}): {[str(p) for p in eval_files]}", rid)
    if dev_files:
        info_id(f"Dev files   ({len(dev_files)}): {[str(p) for p in dev_files]}", rid)

    # ---------------------------------------------------------
    # System optimization
    # ---------------------------------------------------------
    optimizer = SystemOptimizer(request_id=rid)
    optimizer.print_summary()

    if not args.max_examples:
        args.max_examples = {
            "fast": 2_000,
            "small": 10_000,
            "medium": 100_000,
            "full": None,
        }[args.mode]

    info_id(f"Python executable: {sys.executable}", rid)
    import transformers
    info_id(f"Transformers version: {transformers.__version__}", rid)

    # ---------------------------------------------------------
    # Label discovery
    # ---------------------------------------------------------
    if args.labels:
        label_list = list(dict.fromkeys(args.labels))
    else:
        label_list = _read_label_set(train_files + eval_files, rid)

    label_list = _apply_aliases(
        label_list,
        alias_prints_to_drawings=args.alias_prints_to_drawings,
        rid=rid,
    )

    if len(label_list) < 2:
        critical_id(
            f"Intent training requires ≥=2 labels. Discovered: {label_list}",
            rid,
        )
        return

    id2label = {i: lab for i, lab in enumerate(label_list)}
    label2id = {lab: i for i, lab in id2label.items()}

    info_id(
        f"Label mapping: {', '.join(f'{i}:{l}' for i, l in id2label.items())}",
        rid,
    )

    # ---------------------------------------------------------
    # Optional audit
    # ---------------------------------------------------------
    if args.audit and (eval_files or dev_files):
        info_id("Running dataset split audit...", rid)
        with log_timed_operation("audit_splits", rid):
            audit_splits(
                train_files,
                eval_files,
                dev_files,
                args.audit_near_thresh,
                rid,
            )

    # ---------------------------------------------------------
    # Tokenizer / model
    # ---------------------------------------------------------
    with log_timed_operation("tokenizer_init", rid):
        tokenizer = DistilBertTokenizerFast.from_pretrained(args.base_model)

    # ---------------------------------------------------------
    # Compute class weights (NEW)
    # ---------------------------------------------------------
    class_weights = compute_class_weights_from_files(
        train_files=train_files,
        label2id=label2id,
        rid=rid,
    )

    # ---------------------------------------------------------
    # Model with weighted loss (NEW)
    # ---------------------------------------------------------
    model = WeightedIntentModel.from_pretrained(
        args.base_model,
        num_labels=len(label_list),
        id2label=id2label,
        label2id=label2id,
        ignore_mismatched_sizes=True,  # 🔑 THIS IS THE FIX
    )

    model.class_weights = class_weights

    #  GPU adapter prepares the model
    model = gpu_adapter.prepare_model(model)

    cfg = optimizer.optimal_config
    mode_cfg = optimizer.get_mode_cfg(args.max_examples)

    # ---------------------------------------------------------
    # REQUIRED: max_steps for streaming
    # ---------------------------------------------------------
    if args.max_examples:
        steps_per_epoch = max(1, args.max_examples // cfg["batch_size"])
        max_steps = steps_per_epoch * mode_cfg["num_epochs"]
    else:
        max_steps = 10_000

    info_id(
        f"Streaming config: max_steps={max_steps}, "
        f"batch_size={cfg['batch_size']}, "
        f"epochs={mode_cfg['num_epochs']}",
        rid,
    )

    # ---------------------------------------------------------
    # Validation dataset
    # ---------------------------------------------------------
    info_id("Preparing validation dataset...", rid)
    with log_timed_operation("build_validation_dataset", rid):
        val_source = eval_files if eval_files else train_files
        if not eval_files:
            warning_id(
                "No explicit eval files provided — validation built from training data.",
                rid,
            )

        val_ds = build_val_dataset_from_files(
            val_source,
            tokenizer,
            label2id,
            max_length=cfg["max_length"],
            val_size=args.val_size,
            request_id=rid,
        )

    info_id(f"Validation size: {len(val_ds)}", rid)

    # ---------------------------------------------------------
    # Trainer setup
    # ---------------------------------------------------------
    data_collator = DataCollatorWithPadding(tokenizer)

    training_args = TrainingArguments(
        output_dir=MODEL_TRAINING_INTENT_MODEL_DIR,
        evaluation_strategy="steps",
        save_strategy="steps",
        eval_steps=mode_cfg["eval_steps"],
        save_steps=mode_cfg["save_steps"],
        save_total_limit=3,
        load_best_model_at_end=True,
        metric_for_best_model="eval_macro_f1" if f1_score else "eval_accuracy",
        greater_is_better=True,
        learning_rate=cfg["learning_rate"],
        weight_decay=0.05,
        warmup_ratio=0.10,
        per_device_train_batch_size=cfg["batch_size"],
        per_device_eval_batch_size=cfg["batch_size"],
        gradient_accumulation_steps=cfg["gradient_accumulation_steps"],
        num_train_epochs=mode_cfg["num_epochs"],
        max_steps=max_steps,
        logging_dir=str(Path(MODEL_TRAINING_INTENT_MODEL_DIR) / "logs"),
        logging_steps=50,
        report_to="none",
        seed=42,
        label_names=["labels"],
        # 🔒 fp16 safety
        fp16=cfg["fp16"] and model.device.type == "cuda",

        dataloader_pin_memory=cfg["use_gpu"],
        dataloader_num_workers=cfg["num_workers"],
        remove_unused_columns=False,
    )

    streaming_cfg = {
        "files": train_files,
        "tokenizer": tokenizer,
        "label2id": label2id,
        "max_length": cfg["max_length"],
        "max_examples": args.max_examples,
        "shuffle_buffer_size": cfg["shuffle_buffer_size"],
        "dedup_window": max(0, int(args.dedup_window)),

        # 🔁 ROTATION CONFIG (REQUIRED)
        "drawings_window_size": args.drawings_window_size,
        "max_drawings_per_epoch": args.drawings_window_size,
        "max_parts_per_epoch": None,
    }

    class StreamingTrainer(Trainer):
        """
        Streaming trainer with rotating exposure windows.

        Each epoch represents a NEW drawings exposure window.
        Parts are always fully included.
        """

        def __init__(
                self,
                streaming_dataset_config=None,
                request_id=None,
                **kwargs,
        ):
            super().__init__(**kwargs)
            self.streaming_dataset_config = streaming_dataset_config or {}
            self.request_id = request_id
            self.current_epoch = 0

            # Required for rotation
            self.drawings_window_size = self.streaming_dataset_config.get(
                "drawings_window_size"
            )

            if not self.drawings_window_size:
                raise ValueError(
                    "drawings_window_size must be provided for rotating exposure windows"
                )

        def get_train_dataloader(self):
            """
            Build a NEW dataset per epoch with a shifted drawings window.
            """
            drawings_offset = self.current_epoch * self.drawings_window_size

            info_id(
                f"[ExposureWindow] epoch={self.current_epoch} | "
                f"drawings_offset={drawings_offset} | "
                f"window_size={self.drawings_window_size}",
                self.request_id,
            )

            ds = StreamingIntentDataset(
                epoch=self.current_epoch,
                request_id=self.request_id,
                drawings_window_offset=drawings_offset,
                **self.streaming_dataset_config,
            )

            return DataLoader(
                ds,
                batch_size=self.args.per_device_train_batch_size,
                collate_fn=self.data_collator,
                num_workers=self.args.dataloader_num_workers,
                pin_memory=self.args.dataloader_pin_memory and torch.cuda.is_available(),
            )

        def _inner_training_loop(self, **kwargs):
            """
            One HF training loop == one exposure window.
            """
            info_id(
                f"Starting exposure window {self.current_epoch}",
                self.request_id,
            )

            out = super()._inner_training_loop(**kwargs)

            info_id(
                f"Completed exposure window {self.current_epoch}",
                self.request_id,
            )

            self.current_epoch += 1
            return out

    callbacks = [EarlyStoppingCallback(early_stopping_patience=max(1, int(args.early_stop)))]
    if args.overfit_guard:
        callbacks.append(
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
        compute_metrics=compute_metrics,  #  direct, no lambda
        streaming_dataset_config=streaming_cfg,
        callbacks=callbacks,
        request_id=rid,
    )

    #  Future-proof GPU wrapping
    trainer = gpu_adapter.wrap_trainer(trainer)

    info_id("Starting adaptive streaming training...", rid)
    with log_timed_operation("trainer_train", rid):
        trainer.train()

    # ---------------------------------------------------------
    # Export best model
    # ---------------------------------------------------------
    best_ckpt = trainer.state.best_model_checkpoint or training_args.output_dir
    info_id(f"Best checkpoint: {best_ckpt}", rid)

    best_model = WeightedIntentModel.from_pretrained(best_ckpt)
    best_model.class_weights = class_weights
    best_model.config.id2label = id2label
    best_model.config.label2id = label2id

    dst = Path(MODEL_TRAINING_INTENT_MODEL_DIR)
    base = Path(MODELS_DISTILBERT_INTENT)

    # 🔒 SAFETY GUARD — NEVER overwrite base model
    if dst.resolve() == base.resolve():
        critical_id(
            "Refusing to overwrite base intent model directory!",
            rid,
        )
        raise RuntimeError("Unsafe model export target")

    # Safe to replace trained model directory
    shutil.rmtree(dst, ignore_errors=True)
    dst.mkdir(parents=True, exist_ok=True)

    best_model.save_pretrained(dst)
    tokenizer.save_pretrained(dst)
    _save_label_maps(dst, id2label, label2id)

    info_id(f"Exported best checkpoint to: {dst}", rid)
    info_id("Adaptive streaming INTENT training complete!", rid)


if __name__ == "__main__":
    set_request_id()
    main()
