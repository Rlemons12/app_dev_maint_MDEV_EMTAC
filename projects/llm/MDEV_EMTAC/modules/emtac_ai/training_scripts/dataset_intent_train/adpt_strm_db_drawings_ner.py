# modules/emtac_ai/training_scripts/dataset_intent_train/adpt_strm_db_drawings_ner.py
"""
Adaptive streaming NER training for DRAWINGS — reads directly from Postgres,
mirroring the DB-backed approach used by the parts NER trainer.

Key features:
- Database-backed streaming IterableDataset (short-lived sessions, paging & random sampling)
- Label schema for drawings entities
- Run directory versioning + best checkpoint export
- Early stopping, gradient clipping, eval metrics (seqeval)
- Request-scoped logging integrated with your logger utilities
- DROP-IN UPDATE: backbone/head dropout via AutoConfig, CLI-tunable (defaults 0.30/0.20/0.30)
"""

import os
import re
import json
import math
import random
import shutil
import argparse
import logging
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Dict, Any

import torch
from torch.utils.data import IterableDataset
from transformers import (
    AutoTokenizer,
    AutoModelForTokenClassification,
    AutoConfig,  # <-- added
    Trainer,
    TrainingArguments,
    EarlyStoppingCallback,
)
from seqeval.metrics import f1_score, precision_score, recall_score
from sqlalchemy import text
from sqlalchemy.orm import sessionmaker
from sqlalchemy.exc import OperationalError

# ---- Project config & logging (same sources as parts script) ----
from modules.configuration.config_env import TrainingDatabaseConfig   # DB engine/session  [DB parity]
from modules.configuration.config import ORC_DRAWINGS_MODEL_DIR  # where to save   [Models dir]
from modules.configuration.log_config import (
    info_id, warning_id, error_id, debug_id,
    set_request_id, get_request_id, with_request_id
)  # request-scoped logger

# ---------------- Logging bootstrap ----------------
logger = logging.getLogger("ematac_logger")
if not logger.handlers:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# ---------------- Label schema (DRAWINGS) ----------------
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

# Accept common DB aliases (normalized → canonical)
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
def _canon_entity_type(raw: Optional[str]) -> Optional[str]:
    if not raw:
        return None
    t = str(raw).strip().upper().replace("-", "_").replace(" ", "_")
    return LABEL_ALIASES.get(t.lower(), t)

# ---------------- Model/tokenizer defaults ----------------
MODEL_NAME = "distilbert-base-cased"
MAX_LEN = 192

# ---------------- DB setup (parity with parts script) ----------------
db_conf = TrainingDatabaseConfig()
engine = db_conf.get_engine()
SessionLocal = sessionmaker(bind=engine, expire_on_commit=False)

# ---------------- Helpers: run dirs (matches your pattern) ----------------
RUN_NAME_PATTERN = re.compile(r"^\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2}_run-(\d{3})$")

def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def next_run_dir(base_dir: Path) -> Path:
    ensure_dir(base_dir)
    ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    existing = [d.name for d in base_dir.iterdir() if d.is_dir() and d.name.startswith(ts)]
    idx = 1
    for name in existing:
        m = RUN_NAME_PATTERN.match(name)
        if m:
            idx = max(idx, int(m.group(1)) + 1)
    run_dir = base_dir / f"{ts}_run-{idx:03d}"
    ensure_dir(run_dir)
    return run_dir

def write_latest_pointer(base_dir: Path, run_dir: Path):
    (base_dir / "LATEST.txt").write_text(run_dir.name, encoding="utf-8")

def prune_old_runs(base_dir: Path, keep: int = 5):
    runs = [d for d in base_dir.iterdir() if d.is_dir() and RUN_NAME_PATTERN.match(d.name)]
    runs.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    for old in runs[keep:]:
        try:
            shutil.rmtree(old)
            logger.info(f"[CLEANUP] Removed old run dir: {old}")
        except Exception as e:
            logger.warning(f"[CLEANUP] Could not remove {old}: {e}")

def save_best_artifacts(trainer, tokenizer, run_dir: Path):
    best_dir = run_dir / "best"
    ensure_dir(best_dir)
    trainer.model.save_pretrained(str(best_dir))
    tokenizer.save_pretrained(str(best_dir))
    logger.info(f"[SAVE] Best model + tokenizer saved to: {best_dir}")

# ---------------- SQL helpers (matching the parts script style) ----------------
def _fetch_random_ids(session, n: int, sample_type: str) -> List[int]:
    rows = session.execute(
        text("SELECT id FROM training_sample WHERE sample_type=:t ORDER BY random() LIMIT :n"),
        {"t": sample_type, "n": n},
    ).fetchall()
    return [r[0] for r in rows]

def _fetch_rows_by_ids(session, id_list: List[int]):
    return session.execute(
        text("SELECT id, text, entities FROM training_sample WHERE id = ANY(:ids)"),
        {"ids": id_list},
    ).fetchall()

def _fetch_rows_page(session, last_id: int, limit: int, sample_type: str):
    return session.execute(
        text(
            "SELECT id, text, entities FROM training_sample "
            "WHERE sample_type=:t AND id > :last_id ORDER BY id LIMIT :lim"
        ),
        {"t": sample_type, "last_id": last_id, "lim": limit},
    ).fetchall()

# ---------------- NER utilities ----------------
def convert_example_to_ner_format(text_val: str, entities: Optional[List[Dict[str, Any]]]):
    """
    Convert a DB row into token-level BIO using the drawings schema.
    Unknown/alias labels are canonicalized; anything not in the schema is skipped.
    """
    entities = entities or []
    words = text_val.split()

    # token start offsets in original text
    word_starts, pos = [], 0
    for w in words:
        s = text_val.find(w, pos)
        word_starts.append(s)
        pos = s + len(w)

    labels = ["O"] * len(words)

    for ent in entities:
        e_start = ent.get("start")
        e_end   = ent.get("end")
        raw     = ent.get("label") or ent.get("entity")
        e_type  = _canon_entity_type(raw)
        if e_start is None or e_end is None or not e_type:
            continue
        if f"B-{e_type}" not in LABEL2ID or f"I-{e_type}" not in LABEL2ID:
            continue

        # Which word indices overlap the char span?
        first = last = None
        for i, w_start in enumerate(word_starts):
            w_end = w_start + len(words[i])
            if w_start < e_end and w_end > e_start:
                if first is None:
                    first = i
                last = i
        if first is not None:
            labels[first] = f"B-{e_type}"
            for i in range(first + 1, (last or first) + 1):
                labels[i] = f"I-{e_type}"

    label_ids = [LABEL2ID[l] for l in labels]
    return {"tokens": words, "ner_tags": label_ids}

def tokenize_example(tokenizer, example, max_length: int):
    tok = tokenizer(
        example["tokens"],
        is_split_into_words=True,
        truncation=True,
        padding="max_length",
        max_length=max_length,
        return_tensors="pt",
    )
    word_ids = tok.word_ids()
    labels, prev = [], None
    for wid in word_ids:
        if wid is None:
            labels.append(-100)
        else:
            lab = example["ner_tags"][wid]
            if wid != prev:
                labels.append(lab)
            else:
                # Inside the same wordpiece; switch B- to I-
                if LABELS[lab].startswith("B-"):
                    inside = "I-" + LABELS[lab][2:]
                    labels.append(LABEL2ID.get(inside, lab))
                else:
                    labels.append(lab)
            prev = wid
    return {
        "input_ids": tok["input_ids"].squeeze(),
        "attention_mask": tok["attention_mask"].squeeze(),
        "labels": torch.tensor(labels, dtype=torch.long),
    }

def compute_token_metrics(eval_pred):
    logits, labels = eval_pred
    preds = logits.argmax(-1)
    true_tags, pred_tags = [], []
    for p_row, l_row in zip(preds, labels):
        p_seq, l_seq = [], []
        for p_i, l_i in zip(p_row, l_row):
            li = int(l_i)
            if li == -100:
                continue
            p_seq.append(ID2LABEL[int(p_i)])
            l_seq.append(ID2LABEL[li])
        if l_seq:
            true_tags.append(l_seq)
            pred_tags.append(p_seq)
    return {
        "precision": precision_score(true_tags, pred_tags),
        "recall": recall_score(true_tags, pred_tags),
        "f1": f1_score(true_tags, pred_tags),
    }

# ---------------- Streaming dataset backed by DB ----------------
class DBStreamingNERDataset(IterableDataset):
    """
    Streaming dataset that reads from Postgres with short-lived sessions.
    Mirrors the approach used in the parts NER trainer, but with a configurable sample_type.
    """
    def __init__(
        self,
        tokenizer,
        session_factory,
        sample_type: str,
        max_length: int = MAX_LEN,
        max_examples: Optional[int] = None,
        shuffle_buffer_size: int = 1000,
        skip_examples: int = 0,
        epoch: int = 0,
        request_id: Optional[str] = None,
        exclude_ids: Optional[List[int]] = None,
        include_only_ids: Optional[List[int]] = None,
        page_size: int = 1000,
        id_chunk_size: int = 1000,
    ):
        self.tokenizer = tokenizer
        self.session_factory = session_factory
        self.sample_type = sample_type
        self.max_length = max_length
        self.max_examples = max_examples
        self.shuffle_buffer_size = shuffle_buffer_size
        self.skip_examples = skip_examples
        self.epoch = epoch
        self.request_id = request_id
        self.exclude_ids = set(exclude_ids or [])
        self.include_only_ids = set(include_only_ids or [])
        self.page_size = page_size
        self.id_chunk_size = id_chunk_size

        # Pre-compute a length for Trainer (__len__)
        s = self.session_factory()
        try:
            if self.include_only_ids:
                self._length = s.execute(
                    text("SELECT COUNT(*) FROM training_sample WHERE sample_type=:t AND id = ANY(:ids)"),
                    {"t": self.sample_type, "ids": list(self.include_only_ids)},
                ).scalar_one()
            elif self.max_examples is not None:
                self._length = int(self.max_examples)
            else:
                total = s.execute(
                    text("SELECT COUNT(*) FROM training_sample WHERE sample_type=:t"),
                    {"t": self.sample_type},
                ).scalar_one()
                self._length = total - len(self.exclude_ids)
        finally:
            s.close()

    def __len__(self):
        return max(1, int(self._length))

    def __iter__(self):
        random.seed(42 + self.epoch)
        buffer = []

        def _yield_buffer():
            nonlocal buffer
            random.shuffle(buffer)
            for item in buffer:
                yield item
            buffer = []

        produced = 0

        # Case 1: include-only IDs (eval set)
        if self.include_only_ids:
            id_list = list(self.include_only_ids)
            for i in range(0, len(id_list), self.id_chunk_size):
                chunk_ids = id_list[i:i + self.id_chunk_size]
                retry = 0
                while True:
                    s = None
                    try:
                        s = self.session_factory()
                        rows = _fetch_rows_by_ids(s, chunk_ids)
                        s.close()
                        break
                    except OperationalError:
                        if s is not None:
                            try: s.close()
                            except: pass
                        retry += 1
                        if retry > 3:
                            raise
                for _, text_val, entities_val in rows:
                    ex = convert_example_to_ner_format(text_val, entities_val)
                    tok = tokenize_example(self.tokenizer, ex, self.max_length)
                    buffer.append(tok)
                    if len(buffer) >= self.shuffle_buffer_size:
                        yield from _yield_buffer()
            if buffer:
                yield from _yield_buffer()
            return

        # Case 2: Small / Medium (random sample)
        if self.max_examples:
            target = int(self.max_examples)
            s = None
            try:
                s = self.session_factory()
                sample_ids = _fetch_random_ids(s, target + len(self.exclude_ids), self.sample_type)
            finally:
                if s is not None:
                    try: s.close()
                    except: pass
            if self.exclude_ids:
                sample_ids = [i for i in sample_ids if i not in self.exclude_ids][:target]

            for i in range(0, len(sample_ids), self.id_chunk_size):
                chunk_ids = sample_ids[i:i + self.id_chunk_size]
                retry = 0
                while True:
                    s = None
                    try:
                        s = self.session_factory()
                        rows = _fetch_rows_by_ids(s, chunk_ids)
                        s.close()
                        break
                    except OperationalError:
                        if s is not None:
                            try: s.close()
                            except: pass
                        retry += 1
                        if retry > 3:
                            raise
                for _, text_val, entities_val in rows:
                    ex = convert_example_to_ner_format(text_val, entities_val)
                    tok = tokenize_example(self.tokenizer, ex, self.max_length)
                    buffer.append(tok)
                    produced += 1
                    if len(buffer) >= self.shuffle_buffer_size:
                        yield from _yield_buffer()
                    if produced >= target:
                        if buffer:
                            yield from _yield_buffer()
                        return
            if buffer:
                yield from _yield_buffer()
            return

        # Case 3: Full dataset (paged by id)
        last_id = 0
        while True:
            rows = None
            retry = 0
            while True:
                s = None
                try:
                    s = self.session_factory()
                    rows = _fetch_rows_page(s, last_id, self.page_size, self.sample_type)
                    s.close()
                    break
                except OperationalError:
                    if s is not None:
                        try: s.close()
                        except: pass
                    retry += 1
                    if retry > 3:
                        raise
            if not rows:
                if buffer:
                    yield from _yield_buffer()
                return
            for rid, text_val, entities_val in rows:
                last_id = rid
                if rid in self.exclude_ids:
                    continue
                ex = convert_example_to_ner_format(text_val, entities_val)
                tok = tokenize_example(self.tokenizer, ex, self.max_length)
                buffer.append(tok)
                if len(buffer) >= self.shuffle_buffer_size:
                    yield from _yield_buffer()

# ---------------- Trainer wrapper ----------------
class StreamingTrainer(Trainer):
    def __init__(self, request_id: Optional[str] = None, **kwargs):
        super().__init__(**kwargs)
        self.current_epoch = 0
        self.request_id = request_id or get_request_id()

    def _inner_training_loop(self, **kwargs):
        info_id(f"Starting epoch {self.current_epoch}", self.request_id)
        result = super()._inner_training_loop(**kwargs)
        info_id(f"Completed epoch {self.current_epoch}", self.request_id)
        self.current_epoch += 1
        return result

# ---------------- CLI / main ----------------
@with_request_id
def main():
    parser = argparse.ArgumentParser(
        description="DB-backed adaptive streaming NER training for DRAWINGS"
    )

    parser.add_argument("--model_dir", type=str, default=ORC_DRAWINGS_MODEL_DIR)
    parser.add_argument("--sample_type", type=str, default="drawings_ner")
    parser.add_argument("--output_base", type=str, default=ORC_DRAWINGS_MODEL_DIR)
    parser.add_argument("--max_len", type=int, default=MAX_LEN)
    parser.add_argument("--train_max_examples", type=int, default=None)
    parser.add_argument("--eval_ids_file", type=str, default=None)

    parser.add_argument("--per_device_train_batch_size", type=int, default=8)
    parser.add_argument("--per_device_eval_batch_size", type=int, default=8)
    parser.add_argument("--num_train_epochs", type=float, default=3.0)
    parser.add_argument("--learning_rate", type=float, default=5e-5)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--warmup_ratio", type=float, default=0.0)
    parser.add_argument("--logging_steps", type=int, default=50)
    parser.add_argument("--eval_steps", type=int, default=500)
    parser.add_argument("--save_steps", type=int, default=500)
    parser.add_argument("--early_stopping_patience", type=int, default=5)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)

    # Dropout tuning
    parser.add_argument("--hidden_dropout_prob", type=float, default=0.30)
    parser.add_argument("--attn_dropout_prob", type=float, default=0.20)
    parser.add_argument("--classifier_dropout", type=float, default=0.30)

    # DRY RUN
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--dry_run_samples", type=int, default=4)

    args = parser.parse_args()
    request_id = get_request_id()

    info_id(
        f"[DRAWINGS NER] Starting | sample_type={args.sample_type}",
        request_id,
    )

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    # ------------------------------------------------------------
    # Run directory management
    # ------------------------------------------------------------
    base_dir = Path(args.output_base)
    run_dir = next_run_dir(base_dir)
    write_latest_pointer(base_dir, run_dir)
    prune_old_runs(base_dir, keep=5)
    info_id(f"Run dir: {run_dir}", request_id)

    # ------------------------------------------------------------
    # Load LOCAL base model (offline-safe)
    # ------------------------------------------------------------
    model_dir = Path(args.model_dir)
    if not (model_dir / "config.json").exists():
        raise RuntimeError(
            f"Invalid drawings model directory (missing config.json): {model_dir}"
        )

    info_id(f"Loading base model from {model_dir}", request_id)

    tokenizer = AutoTokenizer.from_pretrained(
        model_dir,
        local_files_only=True,
    )

    config = AutoConfig.from_pretrained(
        model_dir,
        local_files_only=True,
        num_labels=len(LABELS),
        id2label=ID2LABEL,
        label2id=LABEL2ID,
        hidden_dropout_prob=args.hidden_dropout_prob,
        attention_probs_dropout_prob=args.attn_dropout_prob,
        classifier_dropout=args.classifier_dropout,
    )

    model = AutoModelForTokenClassification.from_pretrained(
        model_dir,
        local_files_only=True,
        config=config,
    )

    # ------------------------------------------------------------
    # DRY RUN (single forward pass validation)
    # ------------------------------------------------------------
    if args.dry_run:
        info_id("[DRY RUN] Starting validation pass", request_id)

        model.eval()
        model.to("cpu")

        s = SessionLocal()
        try:
            rows = s.execute(
                text(
                    "SELECT id, text, entities "
                    "FROM training_sample "
                    "WHERE sample_type=:t "
                    "ORDER BY random() "
                    "LIMIT :n"
                ),
                {"t": args.sample_type, "n": args.dry_run_samples},
            ).fetchall()
        finally:
            s.close()

        if not rows:
            raise RuntimeError("[DRY RUN] No samples found in DB")

        with torch.no_grad():
            for rid, text_val, entities_val in rows:
                info_id(f"[DRY RUN] Sample ID={rid}", request_id)
                debug_id(f"TEXT: {text_val}", request_id)
                debug_id(f"ENTITIES: {entities_val}", request_id)

                example = convert_example_to_ner_format(text_val, entities_val)
                encoded = tokenize_example(tokenizer, example, args.max_len)

                outputs = model(
                    input_ids=encoded["input_ids"].unsqueeze(0),
                    attention_mask=encoded["attention_mask"].unsqueeze(0),
                    labels=encoded["labels"].unsqueeze(0),
                )

                info_id(
                    f"[DRY RUN] loss={outputs.loss.item():.4f} "
                    f"tokens={len(example['tokens'])}",
                    request_id,
                )

        info_id("[DRY RUN] Completed successfully — exiting", request_id)
        return

    # ------------------------------------------------------------
    # Build datasets
    # ------------------------------------------------------------
    session_factory = SessionLocal

    train_ds = DBStreamingNERDataset(
        tokenizer=tokenizer,
        session_factory=session_factory,
        sample_type=args.sample_type,
        max_length=args.max_len,
        max_examples=args.train_max_examples,
        shuffle_buffer_size=1000,
        epoch=0,
        request_id=request_id,
    )

    eval_ds = None
    if args.eval_ids_file and os.path.exists(args.eval_ids_file):
        with open(args.eval_ids_file, "r", encoding="utf-8") as f:
            eval_ids = json.load(f)

        eval_ds = DBStreamingNERDataset(
            tokenizer=tokenizer,
            session_factory=session_factory,
            sample_type=args.sample_type,
            max_length=args.max_len,
            include_only_ids=eval_ids,
            shuffle_buffer_size=1000,
            epoch=0,
            request_id=request_id,
        )
        info_id(f"Eval set loaded: {len(eval_ids)} ids", request_id)
    else:
        warning_id(
            "No eval_ids_file provided — eval metrics will reflect streamed windows",
            request_id,
        )

    # ------------------------------------------------------------
    # Training
    # ------------------------------------------------------------
    targs = TrainingArguments(
        output_dir=str(run_dir / "checkpoints"),
        overwrite_output_dir=True,
        evaluation_strategy="steps" if eval_ds is not None else "no",
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        num_train_epochs=args.num_train_epochs,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        logging_steps=args.logging_steps,
        eval_steps=args.eval_steps,
        save_steps=args.save_steps,
        load_best_model_at_end=bool(eval_ds is not None),
        metric_for_best_model="f1",
        greater_is_better=True,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        max_grad_norm=args.max_grad_norm,
        report_to=[],
        seed=args.seed,
    )

    callbacks = []
    if eval_ds is not None:
        callbacks.append(
            EarlyStoppingCallback(
                early_stopping_patience=args.early_stopping_patience
            )
        )

    trainer = StreamingTrainer(
        model=model,
        args=targs,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        tokenizer=tokenizer,
        compute_metrics=compute_token_metrics if eval_ds is not None else None,
        callbacks=callbacks,
    )

    info_id("Beginning training...", request_id)
    train_out = trainer.train()
    info_id(
        f"Training complete. Global step={getattr(train_out, 'global_step', 'n/a')}",
        request_id,
    )

    # ------------------------------------------------------------
    # Save outputs
    # ------------------------------------------------------------
    info_id("Saving final model...", request_id)
    trainer.save_model(str(run_dir / "final"))
    tokenizer.save_pretrained(str(run_dir / "final"))

    if eval_ds is not None:
        save_best_artifacts(trainer, tokenizer, run_dir)

    info_id("Done.", request_id)

if __name__ == "__main__":
    set_request_id()
    try:
        main()
    except Exception as e:
        error_id(f"Fatal error: {e}")
        raise
