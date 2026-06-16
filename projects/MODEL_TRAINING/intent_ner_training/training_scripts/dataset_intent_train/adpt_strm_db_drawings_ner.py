"""
Adaptive streaming NER training for DRAWINGS — DB-backed (Postgres).

DROP-IN REPLACEMENT
- FULL parser preserved (and fixed)
- DBStreamingNERDataset (IterableDataset) preserved
- Eval/no-eval logic fixed (defaults to NO eval unless you supply eval_ids_file)
- GPUTrainingAdapter supported
- Offline-safe model loading
- LATEST.txt + best/ handled correctly
- FIX: IterableDataset requires max_steps (Trainer scheduler)
- FIX: eval_ids_file query uses IN(expanding) (Postgres safe)
- FIX: when eval is enabled, Trainer actually evaluates + loads best model
- NEW: --fresh-model support (start from base DistilBERT)
"""

# ============================================================
# Environment bootstrap
# ============================================================
import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv(Path(r"E:\emtac\dev_env\.env"))

BASE_NER_MODEL_ENV = "MODELS_DISTILBERT_BASE_UNCASED"

# ============================================================
# Standard imports
# ============================================================
import re
import json
import math
import random
import shutil
import argparse
import logging
from datetime import datetime
from typing import List

import torch
from torch.utils.data import IterableDataset
from torch.nn import functional as F

from transformers import (
    AutoTokenizer,
    AutoConfig,
    AutoModelForTokenClassification,
    Trainer,
    TrainingArguments,
)

from seqeval.metrics import f1_score, precision_score, recall_score
from sqlalchemy import text, bindparam
from sqlalchemy.orm import sessionmaker

# ============================================================
# EMTAC imports
# ============================================================
from modules.gpu.gpu_training_adapter import GPUTrainingAdapter
from configuration.config_env import TrainingDatabaseConfig
from configuration.config import MODEL_TRAINING_DRAWINGS_MODEL_DIR
from configuration.log_config import (
    get_intent_ner_logger,
    TrainingLogManager,
    maintain_training_logs,
)

# ============================================================
# Logging
# ============================================================
log = get_intent_ner_logger("drawings_ner")
log.setLevel(logging.INFO)

gpu_adapter = GPUTrainingAdapter()
log.info("[GPU] Adapter: %s", gpu_adapter.describe())

# ============================================================
# Label schema
# ============================================================
LABELS = [
    "O",
    "B-EQUIPMENT_NUMBER", "I-EQUIPMENT_NUMBER",
    "B-EQUIPMENT_NAME",   "I-EQUIPMENT_NAME",
    "B-DRAWING_NUMBER",   "I-DRAWING_NUMBER",
    "B-DRAWING_NAME",     "I-DRAWING_NAME",
    "B-SPARE_PART_NUMBER","I-SPARE_PART_NUMBER",
]

LABEL2ID = {l: i for i, l in enumerate(LABELS)}
ID2LABEL = {i: l for i, l in enumerate(LABELS)}

LABEL_ALIASES = {
    "equipment#": "EQUIPMENT_NUMBER",
    "equipnum": "EQUIPMENT_NUMBER",
    "equip_name": "EQUIPMENT_NAME",
    "dwg": "DRAWING_NUMBER",
    "drawing#": "DRAWING_NUMBER",
    "drawingname": "DRAWING_NAME",
    "sparepn": "SPARE_PART_NUMBER",
    "spare_part": "SPARE_PART_NUMBER",
}

def canon_label(raw):
    if not raw:
        return None
    t = str(raw).strip().upper().replace("-", "_").replace(" ", "_")
    return LABEL_ALIASES.get(t.lower(), t)

# ============================================================
# Run directory helpers
# ============================================================
RUN_NAME_PATTERN = re.compile(r"^\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2}_run-(\d{3})$")

def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def next_run_dir(base: Path) -> Path:
    ensure_dir(base)
    ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    idx = 1
    for d in base.iterdir():
        if not d.is_dir():
            continue
        m = RUN_NAME_PATTERN.match(d.name)
        if m and d.name.startswith(ts):
            idx = max(idx, int(m.group(1)) + 1)
    run = base / f"{ts}_run-{idx:03d}"
    ensure_dir(run)
    return run

def write_latest(base: Path, run: Path):
    (base / "LATEST.txt").write_text(run.name, encoding="utf-8")

def prune_old_runs(base: Path, keep=5):
    runs = [d for d in base.iterdir() if d.is_dir() and RUN_NAME_PATTERN.match(d.name)]
    runs.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    for r in runs[keep:]:
        shutil.rmtree(r, ignore_errors=True)
        log.info("[CLEANUP] Removed old run: %s", r)

def save_best(trainer, tokenizer, run: Path):
    best = run / "best"
    ensure_dir(best)
    trainer.model.save_pretrained(str(best))
    tokenizer.save_pretrained(str(best))
    log.info("[SAVE] Best model saved to %s", best)

# ============================================================
# Model resolution
# ============================================================
def resolve_model_dir(model_root: Path) -> Path:
    if (model_root / "config.json").exists():
        return model_root

    latest = model_root / "LATEST.txt"
    if latest.exists():
        run = model_root / latest.read_text().strip()
        best = run / "best"
        if (best / "config.json").exists():
            return best
        if (run / "config.json").exists():
            return run

    raise RuntimeError(f"No valid LOCAL model found in {model_root}")

def resolve_fresh_base_model() -> Path:
    raw = os.getenv(BASE_NER_MODEL_ENV)
    if not raw:
        raise RuntimeError(f"{BASE_NER_MODEL_ENV} not set in .env")

    p = Path(raw).resolve()
    if not (p / "config.json").exists():
        raise RuntimeError(f"Fresh base model invalid at {p}")

    return p

# ============================================================
# DB setup
# ============================================================
db = TrainingDatabaseConfig()
engine = db.get_engine()
SessionLocal = sessionmaker(bind=engine, expire_on_commit=False)

# ============================================================
# Dataset helpers
# ============================================================
def _normalize_entities(val):
    if val is None:
        return []
    if isinstance(val, list):
        return [e for e in val if isinstance(e, dict)]
    if isinstance(val, (str, bytes)):
        try:
            parsed = json.loads(val.decode() if isinstance(val, bytes) else val)
            if isinstance(parsed, list):
                return [e for e in parsed if isinstance(e, dict)]
        except Exception:
            return []
    return []

def convert_example(text_val, entities_val):
    entities = _normalize_entities(entities_val)
    words = text_val.split()

    word_starts, pos = [], 0
    for w in words:
        s = text_val.find(w, pos)
        word_starts.append(s)
        pos = s + len(w)

    labels = ["O"] * len(words)

    for ent in entities:
        e_start, e_end = ent.get("start"), ent.get("end")
        e_type = canon_label(ent.get("label") or ent.get("entity") or ent.get("type"))
        if e_start is None or e_end is None or not e_type:
            continue
        if f"B-{e_type}" not in LABEL2ID:
            continue

        first = last = None
        for i, s0 in enumerate(word_starts):
            if s0 < e_end and (s0 + len(words[i])) > e_start:
                first = i if first is None else first
                last = i

        if first is not None:
            labels[first] = f"B-{e_type}"
            for i in range(first + 1, (last or first) + 1):
                labels[i] = f"I-{e_type}"

    return {"tokens": words, "ner_tags": [LABEL2ID[l] for l in labels]}

def tokenize(tokenizer, example, max_len):
    tok = tokenizer(
        example["tokens"],
        is_split_into_words=True,
        truncation=True,
        padding="max_length",
        max_length=max_len,
        return_tensors="pt",
    )

    labels, prev = [], None
    for wid in tok.word_ids():
        if wid is None:
            labels.append(-100)
        else:
            lab = example["ner_tags"][wid]
            if wid != prev:
                labels.append(lab)
            else:
                if LABELS[lab].startswith("B-"):
                    labels.append(LABEL2ID.get("I-" + LABELS[lab][2:], lab))
                else:
                    labels.append(lab)
            prev = wid

    return {
        "input_ids": tok["input_ids"].squeeze(0),
        "attention_mask": tok["attention_mask"].squeeze(0),
        "labels": torch.tensor(labels, dtype=torch.long),
    }

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = logits.argmax(-1)
    true, pred = [], []

    for p_row, l_row in zip(preds, labels):
        pt, lt = [], []
        for pi, li in zip(p_row, l_row):
            if li == -100:
                continue
            pt.append(ID2LABEL[int(pi)])
            lt.append(ID2LABEL[int(li)])
        if lt:
            true.append(lt)
            pred.append(pt)

    return {
        "precision": precision_score(true, pred),
        "recall": recall_score(true, pred),
        "f1": f1_score(true, pred),
    }

# ============================================================
# Streaming dataset
# ============================================================
class DBStreamingNERDataset(IterableDataset):
    def __init__(self, tokenizer, session_factory, sample_type, max_len, max_examples=None):
        self.tokenizer = tokenizer
        self.session_factory = session_factory
        self.sample_type = sample_type
        self.max_len = max_len
        self.max_examples = max_examples

    def __iter__(self):
        s = self.session_factory()
        try:
            q = "SELECT text, entities FROM training_sample WHERE sample_type=:t"
            params = {"t": self.sample_type}

            if self.max_examples:
                q += " ORDER BY random() LIMIT :n"
                params["n"] = int(self.max_examples)

            for text_val, ent_val in s.execute(text(q), params):
                yield tokenize(self.tokenizer, convert_example(text_val, ent_val), self.max_len)
        finally:
            s.close()

# ============================================================
# Trainer
# ============================================================
class WeightedTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        loss = F.cross_entropy(
            outputs.logits.view(-1, outputs.logits.size(-1)),
            labels.view(-1),
            ignore_index=-100,
        )
        return (loss, outputs) if return_outputs else loss

# ============================================================
# MAIN
# ============================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", default=str(MODEL_TRAINING_DRAWINGS_MODEL_DIR))
    parser.add_argument("--sample_type", default="drawings_ner")
    parser.add_argument("--output_base", default=str(MODEL_TRAINING_DRAWINGS_MODEL_DIR))
    parser.add_argument("--max_len", type=int, default=192)
    parser.add_argument("--train_max_examples", type=int, default=None)
    parser.add_argument("--eval_ids_file", type=str, default=None)
    parser.add_argument("--fresh-model", action="store_true")

    parser.add_argument("--per_device_train_batch_size", type=int, default=8)
    parser.add_argument("--per_device_eval_batch_size", type=int, default=8)
    parser.add_argument("--num_train_epochs", type=float, default=3.0)
    parser.add_argument("--learning_rate", type=float, default=5e-5)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--warmup_ratio", type=float, default=0.0)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--logging_steps", type=int, default=100)
    parser.add_argument("--eval_steps", type=int, default=500)
    parser.add_argument("--save_steps", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    base_model_dir = (
        resolve_fresh_base_model() if args.fresh_model
        else resolve_model_dir(Path(args.model_dir))
    )

    tokenizer = AutoTokenizer.from_pretrained(base_model_dir, local_files_only=True)
    config = AutoConfig.from_pretrained(
        base_model_dir,
        local_files_only=True,
        num_labels=len(LABELS),
        id2label=ID2LABEL,
        label2id=LABEL2ID,
    )

    model = AutoModelForTokenClassification.from_pretrained(
        base_model_dir,
        local_files_only=True,
        config=config,
    )

    model = gpu_adapter.prepare_model(model)

    run_dir = next_run_dir(Path(args.output_base))
    write_latest(Path(args.output_base), run_dir)
    prune_old_runs(Path(args.output_base))
    maintain_training_logs()

    train_ds = DBStreamingNERDataset(
        tokenizer, SessionLocal, args.sample_type, args.max_len, args.train_max_examples
    )

    eval_ds = None
    if args.eval_ids_file and os.path.exists(args.eval_ids_file):
        with open(args.eval_ids_file, "r") as f:
            ids = [int(x) for x in json.load(f)]

        s = SessionLocal()
        stmt = text("SELECT text, entities FROM training_sample WHERE id IN :ids")\
            .bindparams(bindparam("ids", expanding=True))
        rows = s.execute(stmt, {"ids": ids}).fetchall()
        s.close()

        class EvalDataset(torch.utils.data.Dataset):
            def __len__(self): return len(rows)
            def __getitem__(self, i):
                return tokenize(tokenizer, convert_example(*rows[i]), args.max_len)

        eval_ds = EvalDataset()

    if args.train_max_examples:
        total_samples = int(args.train_max_examples)
    else:
        s = SessionLocal()
        try:
            total_samples = s.execute(
                text(
                    "SELECT COUNT(*) FROM training_sample WHERE sample_type=:t"
                ),
                {"t": args.sample_type},
            ).scalar_one()
        finally:
            s.close()

    steps_per_epoch = math.ceil(
        total_samples / (args.per_device_train_batch_size * args.gradient_accumulation_steps)
    )
    max_steps = int(steps_per_epoch * args.num_train_epochs)

    eval_enabled = eval_ds is not None

    targs = TrainingArguments(
        output_dir=str(run_dir / "checkpoints"),
        max_steps=max_steps,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        max_grad_norm=args.max_grad_norm,
        logging_steps=args.logging_steps,
        evaluation_strategy="steps" if eval_enabled else "no",
        eval_steps=args.eval_steps if eval_enabled else None,
        save_strategy="steps",
        save_steps=args.save_steps,
        load_best_model_at_end=eval_enabled,
        metric_for_best_model="f1" if eval_enabled else None,
        greater_is_better=True if eval_enabled else None,
        fp16=torch.cuda.is_available(),
        report_to=[],
        seed=args.seed,
    )

    trainer = WeightedTrainer(
        model=model,
        args=targs,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics if eval_enabled else None,
    )

    with TrainingLogManager(run_dir=run_dir, to_console=False):
        trainer.train()

    save_best(trainer, tokenizer, run_dir)
    log.info("[DONE] Training complete")

if __name__ == "__main__":
    main()
