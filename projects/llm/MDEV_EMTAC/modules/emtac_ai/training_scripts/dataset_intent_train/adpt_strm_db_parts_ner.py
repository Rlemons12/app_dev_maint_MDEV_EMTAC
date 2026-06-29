import os
import re
import math
import random
import shutil
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
from torch.utils.data import IterableDataset, Dataset
from torch.nn import functional as F
from transformers import (
    AutoTokenizer,
    AutoConfig,
    AutoModelForTokenClassification,
    Trainer,
    TrainingArguments,
    EarlyStoppingCallback,
    TrainerCallback,
)
from sqlalchemy import text
from sqlalchemy.orm import sessionmaker
from sqlalchemy.exc import OperationalError

from modules.configuration.config_env import DatabaseConfig
from modules.configuration.config import ORC_PARTS_MODEL_DIR  # canonical parts model dir
from seqeval.metrics import f1_score, precision_score, recall_score
from modules.configuration.log_config import TrainingLogManager, maintain_training_logs


# ===================== Logging =====================
logger = logging.getLogger("ematac_logger")
if not logger.handlers:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


# ===================== Label schema =================
LABELS = [
    "O",
    "B-PART_NUMBER", "I-PART_NUMBER",
    "B-PART_NAME",   "I-PART_NAME",
    "B-MANUFACTURER","I-MANUFACTURER",
    "B-MODEL",       "I-MODEL",
]
LABEL2ID = {l: i for i, l in enumerate(LABELS)}
ID2LABEL = {i: l for i, l in enumerate(LABELS)}
logger.info("Training with label set: %s", LABELS)

# Accept common aliases from the DB
LABEL_ALIASES = {
    "partnumber": "PART_NUMBER", "part_num": "PART_NUMBER", "pn": "PART_NUMBER",
    "partname": "PART_NAME", "name": "PART_NAME", "desc": "PART_NAME", "description": "PART_NAME",
    "mfg": "MANUFACTURER", "manufacturer": "MANUFACTURER", "oemmfg": "MANUFACTURER",
    "mdl": "MODEL", "model": "MODEL",
    "part": "PART_NAME",  # generic PART -> PART_NAME
}

def _canon_entity_type(raw: Any) -> Optional[str]:
    if raw is None:
        return None
    t = str(raw).strip()
    if not t:
        return None
    # Normalize
    t_norm = t.upper().replace("-", "_").replace(" ", "_")
    # Alias lookup uses lower-case keys
    return LABEL_ALIASES.get(t_norm.lower(), t_norm)


# ================== Run/versioning helpers ==================
RUN_NAME_PATTERN = re.compile(r"^\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2}_run-(\d{3})$")

def ensure_dir(p: Path) -> None:
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

def write_latest_pointer(base_dir: Path, run_dir: Path) -> None:
    (base_dir / "LATEST.txt").write_text(run_dir.name, encoding="utf-8")

def prune_old_runs(base_dir: Path, keep: int = 5) -> None:
    runs = [d for d in base_dir.iterdir() if d.is_dir() and RUN_NAME_PATTERN.match(d.name)]
    runs.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    for old in runs[keep:]:
        try:
            shutil.rmtree(old)
            logger.info("[CLEANUP] Removed old run dir: %s", old)
        except Exception as e:
            logger.warning("[CLEANUP] Could not remove %s: %s", old, e)

def save_best_artifacts(trainer: Trainer, tokenizer: AutoTokenizer, run_dir: Path) -> None:
    best_dir = run_dir / "best"
    ensure_dir(best_dir)
    trainer.model.save_pretrained(str(best_dir))
    tokenizer.save_pretrained(str(best_dir))
    logger.info("[SAVE] Best model + tokenizer saved to: %s", best_dir)

def resolve_latest_model(model_root: Path) -> Path:
    latest_file = model_root / "LATEST.txt"
    if not latest_file.exists():
        raise RuntimeError(f"LATEST.txt not found in {model_root}")
    run_name = latest_file.read_text(encoding="utf-8").strip()
    best_dir = model_root / run_name / "best"
    if not best_dir.exists():
        raise RuntimeError(f"Best model directory not found: {best_dir}")
    return best_dir


# ===================== Model/Tokenization =====================
MAX_LEN = 192  # helps capture longer names/models
MODEL_DIR = resolve_latest_model(Path(ORC_PARTS_MODEL_DIR))  # .../best
logger.info("Resolved base model dir: %s", MODEL_DIR)


# ===================== DB Setup =====================
db_conf = DatabaseConfig()
engine = db_conf.get_engine()
SessionLocal = sessionmaker(bind=engine, expire_on_commit=False)

# ================== Helper Constants ================
PAGE_SIZE = 1000
ID_CHUNK_SIZE = 1000


# ================== SQL helpers =====================
def _fetch_random_ids(session, n: int) -> List[int]:
    rows = session.execute(
        text("SELECT id FROM training_sample WHERE sample_type='ner' ORDER BY random() LIMIT :n"),
        {"n": n},
    ).fetchall()
    return [int(r[0]) for r in rows]

def _fetch_rows_by_ids(session, id_list: Sequence[int]):
    # IMPORTANT: keep sample_type filter to avoid pulling other sample types
    return session.execute(
        text(
            "SELECT id, text, entities "
            "FROM training_sample "
            "WHERE sample_type='ner' AND id = ANY(:ids)"
        ),
        {"ids": list(id_list)},
    ).fetchall()

def _fetch_rows_page(session, last_id: int, limit: int):
    return session.execute(
        text(
            "SELECT id, text, entities "
            "FROM training_sample "
            "WHERE sample_type='ner' AND id > :last_id "
            "ORDER BY id LIMIT :lim"
        ),
        {"last_id": last_id, "lim": limit},
    ).fetchall()


# ================== NER utilities ===================
def _normalize_entities(entities_val: Any) -> List[Dict[str, Any]]:
    """
    training_sample.entities might be:
      - JSONB (already list[dict])
      - TEXT/VARCHAR containing JSON
      - NULL
    """
    if entities_val is None:
        return []
    if isinstance(entities_val, list):
        return [e for e in entities_val if isinstance(e, dict)]
    if isinstance(entities_val, (str, bytes)):
        try:
            s = entities_val.decode("utf-8") if isinstance(entities_val, bytes) else entities_val
            parsed = json.loads(s)
            if isinstance(parsed, list):
                return [e for e in parsed if isinstance(e, dict)]
        except Exception:
            return []
    return []

def convert_example_to_ner_format(text_val: str, entities_val: Any) -> Dict[str, Any]:
    """
    Convert DB row into token-level BIO.
    Uses simple whitespace tokenization (consistent with your existing pipeline).
    """
    entities = _normalize_entities(entities_val)
    words = text_val.split()

    # token start offsets in original text
    word_starts, pos = [], 0
    for w in words:
        s = text_val.find(w, pos)
        word_starts.append(s)
        pos = s + len(w)

    labels = ["O"] * len(words)

    for ent in entities:
        e_start, e_end = ent.get("start"), ent.get("end")
        raw_type = ent.get("label") or ent.get("entity") or ent.get("type")
        e_type = _canon_entity_type(raw_type)

        if e_start is None or e_end is None or not e_type:
            continue
        if f"B-{e_type}" not in LABEL2ID or f"I-{e_type}" not in LABEL2ID:
            continue

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

def tokenize_example(tokenizer: AutoTokenizer, example: Dict[str, Any], max_length: int) -> Dict[str, torch.Tensor]:
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
                # If a word is split into multiple wordpieces, convert B- to I-
                if LABELS[lab].startswith("B-"):
                    inside = "I-" + LABELS[lab][2:]
                    labels.append(LABEL2ID.get(inside, lab))
                else:
                    labels.append(lab)
            prev = wid

    return {
        "input_ids": tok["input_ids"].squeeze(0),
        "attention_mask": tok["attention_mask"].squeeze(0),
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


# ====================== Datasets ====================
class DBStreamingNERDataset(IterableDataset):
    """Streaming dataset that reads from Postgres with short-lived sessions."""
    def __init__(
        self,
        tokenizer: AutoTokenizer,
        session_factory,
        max_length: int = MAX_LEN,
        max_examples: Optional[int] = None,
        shuffle_buffer_size: int = 1000,
        epoch: int = 0,
        exclude_ids: Optional[Sequence[int]] = None,
        include_only_ids: Optional[Sequence[int]] = None,
    ):
        self.tokenizer = tokenizer
        self.session_factory = session_factory
        self.max_length = max_length
        self.max_examples = max_examples
        self.shuffle_buffer_size = shuffle_buffer_size
        self.epoch = epoch
        self.exclude_ids = set(exclude_ids or [])
        self.include_only_ids = set(include_only_ids or [])

        s = self.session_factory()
        try:
            if self.include_only_ids:
                self._length = s.execute(
                    text(
                        "SELECT COUNT(*) FROM training_sample "
                        "WHERE sample_type='ner' AND id = ANY(:ids)"
                    ),
                    {"ids": list(self.include_only_ids)},
                ).scalar_one()
            elif self.max_examples is not None:
                self._length = int(self.max_examples)
            else:
                total = s.execute(
                    text("SELECT COUNT(*) FROM training_sample WHERE sample_type='ner'")
                ).scalar_one()
                self._length = max(0, int(total) - len(self.exclude_ids))
        finally:
            s.close()

    def __len__(self) -> int:
        return max(1, int(self._length))

    def __iter__(self):
        random.seed(42 + self.epoch)
        buffer: List[Dict[str, torch.Tensor]] = []

        def _yield_buffer():
            nonlocal buffer
            random.shuffle(buffer)
            for item in buffer:
                yield item
            buffer = []

        # Case 1: include-only IDs
        if self.include_only_ids:
            id_list = list(self.include_only_ids)
            for i in range(0, len(id_list), ID_CHUNK_SIZE):
                chunk_ids = id_list[i:i + ID_CHUNK_SIZE]
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
                            try:
                                s.close()
                            except Exception:
                                pass
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

        # Case 2: random sample
        if self.max_examples:
            target = int(self.max_examples)

            s = self.session_factory()
            try:
                sample_ids = _fetch_random_ids(s, target + len(self.exclude_ids))
            finally:
                try:
                    s.close()
                except Exception:
                    pass

            if self.exclude_ids:
                sample_ids = [i for i in sample_ids if i not in self.exclude_ids][:target]

            produced = 0
            for i in range(0, len(sample_ids), ID_CHUNK_SIZE):
                chunk_ids = sample_ids[i:i + ID_CHUNK_SIZE]
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
                            try:
                                s.close()
                            except Exception:
                                pass
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

        # Case 3: full dataset (paged)
        last_id = 0
        while True:
            retry = 0
            while True:
                s = None
                try:
                    s = self.session_factory()
                    rows = _fetch_rows_page(s, last_id, PAGE_SIZE)
                    s.close()
                    break
                except OperationalError:
                    if s is not None:
                        try:
                            s.close()
                        except Exception:
                            pass
                    retry += 1
                    if retry > 3:
                        raise

            if not rows:
                break

            for rid, text_val, entities_val in rows:
                if self.exclude_ids and int(rid) in self.exclude_ids:
                    last_id = int(rid)
                    continue
                ex = convert_example_to_ner_format(text_val, entities_val)
                tok = tokenize_example(self.tokenizer, ex, self.max_length)
                buffer.append(tok)
                if len(buffer) >= self.shuffle_buffer_size:
                    yield from _yield_buffer()
                last_id = int(rid)

        if buffer:
            yield from _yield_buffer()

class ListNERDataset(Dataset):
    def __init__(self, tokenizer: AutoTokenizer, rows, max_length: int = MAX_LEN):
        self.tokenizer = tokenizer
        self.rows = rows
        self.max_length = max_length

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        _, text_val, entities_val = self.rows[idx]
        ex = convert_example_to_ner_format(text_val, entities_val)
        return tokenize_example(self.tokenizer, ex, self.max_length)


# ====================== Utilities ===================
def choose_dataset_size() -> Optional[int]:
    print("Select training dataset size:")
    print("1) Small (≈1,000 samples)")
    print("2) Medium (≈10,000 samples)")
    print("3) Full (all available samples)")
    choice = input("Enter choice (1/2/3): ").strip()
    if choice == "1":
        return 1000
    if choice == "2":
        return 10000
    return None  # full dataset

def count_ner_rows(session_factory) -> int:
    s = session_factory()
    try:
        return int(
            s.execute(text("SELECT COUNT(*) FROM training_sample WHERE sample_type='ner'")).scalar_one()
        )
    finally:
        s.close()

def compute_epoch_and_eval_steps(
    total_examples: int,
    per_device_bs: int,
    grad_accum: int,
    min_interval: int = 150
) -> Tuple[int, int, int]:
    effective_bs = max(1, per_device_bs * max(1, grad_accum))
    steps_per_epoch = max(1, math.ceil(total_examples / effective_bs))
    eval_save_steps = max(min_interval, steps_per_epoch)
    max_steps = steps_per_epoch * 3  # fixed to 3 epochs
    return steps_per_epoch, max_steps, eval_save_steps


# ===================== Weighted Trainer =====================
class WeightedTrainer(Trainer):
    def __init__(self, *args, class_weights=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = None
        if class_weights is not None:
            self.class_weights = torch.tensor(class_weights, dtype=torch.float)

    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.get("labels")
        outputs = model(**{k: v for k, v in inputs.items() if k != "labels"})
        logits = outputs.logits

        if self.class_weights is not None and self.class_weights.device != logits.device:
            self.class_weights = self.class_weights.to(logits.device)

        ls = getattr(self.args, "label_smoothing_factor", 0.0) or 0.0

        loss = F.cross_entropy(
            logits.view(-1, logits.size(-1)),
            labels.view(-1),
            weight=self.class_weights,
            ignore_index=-100,
            label_smoothing=ls,
        )
        return (loss, outputs) if return_outputs else loss


# ===================== Anti-overfitting callbacks =====================
class EarlyStopMinDeltaCallback(TrainerCallback):
    def __init__(self, metric_name="eval_f1", min_delta=1e-3, patience=2):
        self.metric_name = metric_name
        self.min_delta = float(min_delta)
        self.patience = int(patience)
        self.best = None
        self.bad_count = 0

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        if not metrics or self.metric_name not in metrics:
            return
        current = metrics[self.metric_name]
        if self.best is None or current > self.best + self.min_delta:
            self.best = current
            self.bad_count = 0
        else:
            self.bad_count += 1
            if self.bad_count >= self.patience:
                control.should_training_stop = True

class ReduceLROnPlateauCallback(TrainerCallback):
    def __init__(self, metric_name="eval_f1", factor=0.5, patience=2, min_lr=1e-6):
        self.metric_name = metric_name
        self.factor = float(factor)
        self.patience = int(patience)
        self.min_lr = float(min_lr)
        self.best = None
        self.bad = 0

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        if not metrics or self.metric_name not in metrics:
            return
        current = metrics[self.metric_name]
        if self.best is None or current > self.best:
            self.best = current
            self.bad = 0
            return
        self.bad += 1
        if self.bad >= self.patience:
            self.bad = 0
            opt = kwargs["trainer"].optimizer
            for group in opt.param_groups:
                group["lr"] = max(self.min_lr, group["lr"] * self.factor)


# ========================= Main helpers =====================
def maybe_freeze_encoder(model, freeze=True, unfreeze_last_n=2):
    """
    Robust freezing across common HF backbones (bert/distilbert/roberta/etc.).
    Freezes all encoder layers, then unfreezes the last N transformer blocks.
    """
    if not freeze:
        return

    # Freeze everything first
    for _, p in model.named_parameters():
        p.requires_grad = False

    # Always keep classification head trainable
    for name, p in model.named_parameters():
        if any(k in name for k in ["classifier", "crf", "score", "lm_head"]):
            p.requires_grad = True

    # Unfreeze last N encoder blocks for common architectures
    encoder_block_prefixes = [
        "bert.encoder.layer.",
        "roberta.encoder.layer.",
        "distilbert.transformer.layer.",
        "albert.encoder.albert_layer_groups.",
        "deberta.encoder.layer.",
        "deberta_v2.encoder.layer.",
    ]

    # Find the maximum layer index present
    layer_indices = set()
    for name, _ in model.named_parameters():
        for pref in encoder_block_prefixes:
            if pref in name:
                # Extract the int after prefix
                try:
                    tail = name.split(pref, 1)[1]
                    idx = int(tail.split(".", 1)[0])
                    layer_indices.add(idx)
                except Exception:
                    pass

    if not layer_indices:
        # If we can't detect layers, at least unfreeze embeddings + head
        for name, p in model.named_parameters():
            if "embeddings" in name:
                p.requires_grad = True
        return

    max_idx = max(layer_indices)
    for idx in range(max_idx, max_idx - int(unfreeze_last_n), -1):
        for name, p in model.named_parameters():
            for pref in encoder_block_prefixes:
                if f"{pref}{idx}." in name:
                    p.requires_grad = True


# ========================= Main =====================
def main():
    max_examples = choose_dataset_size()

    per_device_bs = 16
    num_epochs = 3
    grad_accum = 1
    eval_frac = 0.15

    tokenizer = AutoTokenizer.from_pretrained(str(MODEL_DIR))

    session = SessionLocal()
    try:
        if max_examples is None:
            total_all = count_ner_rows(SessionLocal)
            eval_count = max(1, int(total_all * eval_frac))

            eval_ids = [
                int(r[0]) for r in session.execute(
                    text(
                        "SELECT id FROM training_sample "
                        "WHERE sample_type='ner' ORDER BY random() LIMIT :k"
                    ),
                    {"k": eval_count},
                ).fetchall()
            ]

            train_dataset = DBStreamingNERDataset(
                tokenizer=tokenizer,
                session_factory=SessionLocal,
                max_length=MAX_LEN,
                shuffle_buffer_size=1000,
                max_examples=None,
                exclude_ids=set(eval_ids),
            )

            eval_rows = session.execute(
                text(
                    "SELECT id, text, entities FROM training_sample "
                    "WHERE sample_type='ner' AND id = ANY(:ids)"
                ),
                {"ids": eval_ids},
            ).fetchall()
            eval_dataset = ListNERDataset(tokenizer, eval_rows, max_length=MAX_LEN)
            total_examples = max(1, total_all - len(eval_rows))

        else:
            pool_rows = session.execute(
                text(
                    "SELECT id, text, entities FROM training_sample "
                    "WHERE sample_type='ner' ORDER BY random() LIMIT :n"
                ),
                {"n": int(max_examples)},
            ).fetchall()

            pool_rows = list(pool_rows)
            random.shuffle(pool_rows)
            cut = max(1, int(len(pool_rows) * (1 - eval_frac)))
            train_rows = pool_rows[:cut]
            eval_rows = pool_rows[cut:]

            train_dataset = ListNERDataset(tokenizer, train_rows, max_length=MAX_LEN)
            eval_dataset = ListNERDataset(tokenizer, eval_rows, max_length=MAX_LEN)
            total_examples = max(1, len(train_rows))
    finally:
        session.close()

    steps_per_epoch, max_steps, eval_save_steps = compute_epoch_and_eval_steps(
        total_examples=total_examples,
        per_device_bs=per_device_bs,
        grad_accum=grad_accum,
        min_interval=150,
    )

    cfg = AutoConfig.from_pretrained(str(MODEL_DIR))
    cfg.hidden_dropout_prob = 0.20
    cfg.attention_probs_dropout_prob = 0.20
    cfg.num_labels = len(LABELS)
    cfg.id2label = ID2LABEL
    cfg.label2id = LABEL2ID

    model = AutoModelForTokenClassification.from_pretrained(str(MODEL_DIR), config=cfg)

    if max_examples is not None:
        maybe_freeze_encoder(model, freeze=True, unfreeze_last_n=2)

    base_dir = Path(ORC_PARTS_MODEL_DIR)
    run_dir = next_run_dir(base_dir)
    logger.info("[TRAIN] Output run directory: %s", run_dir)

    maintain_training_logs(retention_weeks=2)

    training_args = TrainingArguments(
        output_dir=str(run_dir),
        overwrite_output_dir=True,
        per_device_train_batch_size=per_device_bs,
        gradient_accumulation_steps=grad_accum,
        num_train_epochs=num_epochs,
        max_steps=max_steps,
        evaluation_strategy="steps",
        eval_steps=eval_save_steps,
        save_strategy="steps",
        save_steps=eval_save_steps,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        save_total_limit=3,
        weight_decay=0.02,
        warmup_ratio=0.10,
        learning_rate=3e-5,
        label_smoothing_factor=0.10,
        max_grad_norm=1.0,
        logging_steps=max(50, steps_per_epoch // 2),
        remove_unused_columns=False,
    )

    class_weights = [1.0] * len(LABELS)
    for k in ("B-MODEL", "I-MODEL"):
        class_weights[LABEL2ID[k]] = 2.0

    with TrainingLogManager(run_dir=run_dir, to_console=False) as tlogm:
        train_log = tlogm.logger
        cb = tlogm.make_trainer_callback()

        train_log.info("=== Training session starting ===")
        train_log.info("Run dir: %s", run_dir)
        train_log.info("Backbone (latest best): %s | Max seq len: %s", MODEL_DIR, MAX_LEN)
        train_log.info("Labels: %s", LABELS)
        train_log.info("Training examples (est.): %s", total_examples)
        train_log.info("Steps/epoch: %s | Max steps (3 epochs): %s", steps_per_epoch, max_steps)
        train_log.info("Eval/Save every: %s steps", eval_save_steps)

        trainer = WeightedTrainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            compute_metrics=compute_token_metrics,
            callbacks=[
                EarlyStoppingCallback(early_stopping_patience=3),
                EarlyStopMinDeltaCallback(metric_name="eval_f1", min_delta=1e-3, patience=2),
                ReduceLROnPlateauCallback(metric_name="eval_f1", factor=0.5, patience=2, min_lr=1e-6),
                cb,
            ],
            class_weights=class_weights,
        )

        trainer.train()

        save_best_artifacts(trainer, tokenizer, run_dir)
        write_latest_pointer(base_dir, run_dir)
        prune_old_runs(base_dir, keep=5)

        train_log.info("[SAVE] Best model → %s", run_dir / "best")
        train_log.info("=== Training session complete ===")


if __name__ == "__main__":
    main()
