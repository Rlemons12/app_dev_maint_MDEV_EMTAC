# modules/emtac_ai/training_scripts/dataset_intent_train/adpt_strm_db_parts_ner.py
import os
import re
import math
import random
import shutil
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Tuple, Optional

import numpy as np
import torch
from torch.utils.data import IterableDataset, Dataset, TensorDataset
from torch.nn import functional as F

from transformers import (
    AutoTokenizer,
    AutoConfig,
    AutoModelForTokenClassification,
    Trainer,
    TrainingArguments,
    EarlyStoppingCallback,
    TrainerCallback,
    DataCollatorForTokenClassification,
    pipeline,
)

from sqlalchemy import text
from sqlalchemy.orm import sessionmaker
from sqlalchemy.exc import OperationalError
from seqeval.metrics import f1_score, precision_score, recall_score

# --- Project imports (match your codebase) ---
from modules.configuration.config_env import DatabaseConfig
from modules.configuration.config import (
    ORC_PARTS_MODEL_DIR,                       # …/modules/emtac_ai/models/parts
    ORC_TRAINING_DATA_PARTS_LOADSHEET_PATH,    # authoritative loadsheet (Excel)
)
from modules.configuration.log_config import TrainingLogManager, maintain_training_logs

# ===================== Logging =====================
logger = logging.getLogger("ematac_logger")
if not logger.handlers:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# ===================== Model/Tokenization =====================
MODEL_NAME = os.getenv("PARTS_NER_BASE_MODEL", "distilbert-base-uncased")
MAX_LEN = int(os.getenv("PARTS_NER_MAX_LEN", "160"))

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

# Canonicalize various entity label spellings if your DB uses aliases
LABEL_ALIASES = {
    "partnumber": "PART_NUMBER", "part_num": "PART_NUMBER", "pn": "PART_NUMBER",
    "partname": "PART_NAME", "name": "PART_NAME", "desc": "PART_NAME", "description": "PART_NAME",
    "mfg": "MANUFACTURER", "manufacturer": "MANUFACTURER", "oemmfg": "MANUFACTURER",
    "mdl": "MODEL", "model": "MODEL",
    "part": "PART_NAME",
}

class DictTensorDataset(torch.utils.data.Dataset):
    """Wrap three tensors so each item is a dict for HuggingFace Trainer."""
    def __init__(self, input_ids, attention_mask, labels):
        self.input_ids = input_ids
        self.attention_mask = attention_mask
        self.labels = labels

    def __len__(self):
        return self.input_ids.size(0)

    def __getitem__(self, idx):
        return {
            "input_ids": self.input_ids[idx],
            "attention_mask": self.attention_mask[idx],
            "labels": self.labels[idx],
        }


def prompt_active_learning_settings():
    print("\n=== Active Learning Settings ===")
    rounds = input("Max AL rounds (default 2): ").strip()
    min_neg = input("Min new negatives per round (default 150): ").strip()
    max_neg = input("Max negatives per round (default 3000): ").strip()
    total_cap = input("Total negatives cap across all rounds (default 5000): ").strip()
    probe_rows = input("Probe rows per round (default 1000): ").strip()
    tpl_per_row = input("Templates per row (default 4): ").strip()
    al_epochs = input("Epochs per AL round (default 1): ").strip()

    return {
        "rounds": int(rounds) if rounds else 2,
        "min_neg": int(min_neg) if min_neg else 150,
        "max_neg": int(max_neg) if max_neg else 3000,
        "total_cap": int(total_cap) if total_cap else 5000,
        "probe_rows": int(probe_rows) if probe_rows else 1000,
        "tpl_per_row": int(tpl_per_row) if tpl_per_row else 4,
        "al_epochs": int(al_epochs) if al_epochs else 1
    }


def _canon_entity_type(raw):
    if not raw:
        return None
    t = str(raw).strip().upper().replace("-", "_").replace(" ", "_")
    return LABEL_ALIASES.get(t.lower(), t)

# ===================== DB Setup =====================
db_conf = DatabaseConfig()
engine = db_conf.get_engine()
SessionLocal = sessionmaker(bind=engine, expire_on_commit=False)

# ================== Helper Constants ================
PAGE_SIZE = 1000
ID_CHUNK_SIZE = 1000
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

# ================== SQL helpers (DB mode) =====================
def _fetch_rows_by_ids(session, id_list: list):
    return session.execute(
        text("SELECT id, text, entities FROM training_sample WHERE id = ANY(:ids)"),
        {"ids": id_list},
    ).fetchall()

def _fetch_rows_page(session, last_id: int, limit: int):
    return session.execute(
        text(
            "SELECT id, text, entities FROM training_sample "
            "WHERE sample_type='ner' AND id > :last_id ORDER BY id LIMIT :lim"
        ),
        {"last_id": last_id, "lim": limit},
    ).fetchall()

# ================== NER utilities (shared) ===================
def convert_example_to_ner_format(text_val: str, entities):
    """
    Convert a DB row into token-level BIO using the 4-entity schema.
    Unknown/alias labels are canonicalized; anything not in the schema is skipped.
    """
    entities = entities or []
    words = text_val.split()
    word_starts, pos = [], 0
    for w in words:
        s = text_val.find(w, pos)
        word_starts.append(s)
        pos = s + len(w)
    labels = ["O"] * len(words)
    for ent in entities:
        e_start, e_end = ent.get("start"), ent.get("end")
        raw_type = ent.get("label") or ent.get("entity")
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

def tokenize_wordlevel_example(tokenizer, example, max_length: int):
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

# ====================== DATASET MODE A: DB ====================
class DBStreamingNERDataset(IterableDataset):
    """Streaming dataset that reads from Postgres with short-lived sessions."""
    def __init__(self, tokenizer, session_factory, max_length=MAX_LEN,
                 max_examples=None, shuffle_buffer_size=1000,
                 skip_examples=0, epoch=0, request_id=None,
                 exclude_ids=None, include_only_ids=None):
        self.tokenizer = tokenizer
        self.session_factory = session_factory
        self.max_length = max_length
        self.max_examples = max_examples
        self.shuffle_buffer_size = shuffle_buffer_size
        self.skip_examples = skip_examples
        self.epoch = epoch
        self.request_id = request_id
        self.exclude_ids = set(exclude_ids or [])
        self.include_only_ids = set(include_only_ids or [])
        s = self.session_factory()
        try:
            if self.include_only_ids:
                self._length = s.execute(
                    text("SELECT COUNT(*) FROM training_sample WHERE sample_type='ner' AND id = ANY(:ids)"),
                    {"ids": list(self.include_only_ids)},
                ).scalar_one()
            elif self.max_examples is not None:
                self._length = int(self.max_examples)
            else:
                total = s.execute(text("SELECT COUNT(*) FROM training_sample WHERE sample_type='ner'")).scalar_one()
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
        if self.include_only_ids:
            id_list = list(self.include_only_ids)
            for i in range(0, len(id_list), ID_CHUNK_SIZE):
                chunk_ids = id_list[i:i + ID_CHUNK_SIZE]
                rows = self._safe_fetch_by_ids(chunk_ids)
                for _, text_val, entities_val in rows:
                    ex = convert_example_to_ner_format(text_val, entities_val)
                    tok = tokenize_wordlevel_example(self.tokenizer, ex, self.max_length)
                    buffer.append(tok)
                    if len(buffer) >= self.shuffle_buffer_size:
                        yield from _yield_buffer()
            if buffer: yield from _yield_buffer()
            return
        if self.max_examples:
            target = int(self.max_examples)
            sample_ids = self._fetch_random_ids(target + len(self.exclude_ids))
            if self.exclude_ids:
                sample_ids = [i for i in sample_ids if i not in self.exclude_ids][:target]
            for i in range(0, len(sample_ids), ID_CHUNK_SIZE):
                chunk_ids = sample_ids[i:i + ID_CHUNK_SIZE]
                rows = self._safe_fetch_by_ids(chunk_ids)
                for _, text_val, entities_val in rows:
                    ex = convert_example_to_ner_format(text_val, entities_val)
                    tok = tokenize_wordlevel_example(self.tokenizer, ex, self.max_length)
                    buffer.append(tok)
                    produced += 1
                    if len(buffer) >= self.shuffle_buffer_size:
                        yield from _yield_buffer()
                    if produced >= target:
                        if buffer: yield from _yield_buffer()
                        return
            if buffer: yield from _yield_buffer()
            return
        last_id = 0
        while True:
            rows = self._safe_fetch_page(last_id, PAGE_SIZE)
            if not rows: break
            for rid, text_val, entities_val in rows:
                if self.exclude_ids and rid in self.exclude_ids:
                    last_id = rid
                    continue
                ex = convert_example_to_ner_format(text_val, entities_val)
                tok = tokenize_wordlevel_example(self.tokenizer, ex, self.max_length)
                buffer.append(tok)
                if len(buffer) >= self.shuffle_buffer_size:
                    yield from _yield_buffer()
                last_id = rid
        if buffer: yield from _yield_buffer()

    # robust fetchers with retries
    def _fetch_random_ids(self, n: int) -> list:
        s = None
        try:
            s = self.session_factory()
            rows = s.execute(text("SELECT id FROM training_sample WHERE sample_type='ner' ORDER BY random() LIMIT :n"), {"n": n}).fetchall()
            return [r[0] for r in rows]
        finally:
            if s: s.close()

    def _safe_fetch_by_ids(self, chunk_ids):
        retry = 0
        while True:
            s = None
            try:
                s = self.session_factory()
                rows = _fetch_rows_by_ids(s, chunk_ids)
                s.close()
                return rows
            except OperationalError:
                if s is not None:
                    try: s.close()
                    except Exception: pass
                retry += 1
                if retry > 3: raise

    def _safe_fetch_page(self, last_id, page_size):
        retry = 0
        while True:
            s = None
            try:
                s = self.session_factory()
                rows = _fetch_rows_page(s, last_id, page_size)
                s.close()
                return rows
            except OperationalError:
                if s is not None:
                    try: s.close()
                    except Exception: pass
                retry += 1
                if retry > 3: raise

class ListNERDataset(Dataset):
    def __init__(self, tokenizer, rows, max_length=MAX_LEN):
        self.tokenizer = tokenizer
        self.rows = rows
        self.max_length = max_length
    def __len__(self):
        return len(self.rows)
    def __getitem__(self, idx):
        _, text_val, entities_val = self.rows[idx]
        ex = convert_example_to_ner_format(text_val, entities_val)
        return tokenize_wordlevel_example(self.tokenizer, ex, self.max_length)

# ====================== DATASET MODE B: DISTANT ====================
CANONICAL_ALIASES = {
    "balston": ["balston filt", "balston filter", "balston filtration"],
    "parker": ["parker hannifin", "parker hnnfn"],
    "smc": ["smc corporation", "smc corp"],
}
def normalize_manufacturer(text: str) -> str:
    if not text: return ""
    t = text.strip().lower()
    for canon, aliases in CANONICAL_ALIASES.items():
        if t == canon or t in aliases or any(a in t for a in aliases):
            return canon
    return t

def load_templates_from_db(intent_name: str = "parts") -> List[str]:
    templates: List[str] = []
    try:
        db = DatabaseConfig()
        with db.main_session() as s:
            sql = text("""
                SELECT qt.template_text
                FROM query_template qt
                JOIN intent i ON qt.intent_id = i.id
                WHERE i.name = :intent AND qt.is_active = TRUE
                ORDER BY COALESCE(qt.display_order, qt.id)
            """)
            templates = [r[0] for r in s.execute(sql, {"intent": intent_name}).fetchall()]
    except Exception as e:
        logger.warning(f"[Templates] DB load failed for intent='parts': {e}")
    clean, seen = [], set()
    for t in templates or []:
        t2 = (t or "").strip()
        if t2 and t2 not in seen:
            seen.add(t2)
            clean.append(t2)
    return clean

def _read_loadsheet(max_rows: Optional[int] = None) -> List[Dict[str, str]]:
    import pandas as pd
    df = pd.read_excel(ORC_TRAINING_DATA_PARTS_LOADSHEET_PATH)
    df.columns = [str(c).strip().upper() for c in df.columns]
    for col in ["ITEMNUM", "DESCRIPTION", "OEMMFG", "MODEL"]:
        if col not in df.columns:
            raise ValueError(f"Missing column '{col}' in loadsheet")
    if max_rows and len(df) > max_rows:
        df = df.head(max_rows)
    rows = []
    for _, r in df.iterrows():
        rows.append({
            "ITEMNUM": str(r.get("ITEMNUM", "")).strip(),
            "DESCRIPTION": str(r.get("DESCRIPTION", "")).strip(),
            "OEMMFG": str(r.get("OEMMFG", "")).strip(),
            "MODEL": str(r.get("MODEL", "")).strip(),
        })
    return rows

def _render_sentence(tpl: str, desc: str, mfg: str, itemnum: str, model: str) -> str:
    s = (
        tpl.replace("{description}", desc.lower())
           .replace("{manufacturer}", mfg.lower())
           .replace("{itemnum}", itemnum)
           .replace("{model}", model)
    )
    return s


def _align_and_tag(sentence: str, spans: Dict[str, str], tokenizer: AutoTokenizer) -> Tuple[List[int], List[int]]:
    enc = tokenizer(sentence, return_offsets_mapping=True, add_special_tokens=True,
                    truncation=True, max_length=MAX_LEN)
    offsets = enc["offset_mapping"]
    labels = ["O"] * len(offsets)
    candidates = []
    hay = sentence.lower()
    for ent, text in spans.items():
        if not text: continue
        t_norm = text.strip().lower()
        tests = [t_norm]
        if ent == "MANUFACTURER":
            canon = normalize_manufacturer(text)
            if canon != t_norm:
                tests.insert(0, canon)
        found = []
        for needle in tests:
            start = hay.find(needle)
            if start != -1:
                found.append((start, start + len(needle)))
        if not found: continue
        start, end = max(found, key=lambda x: x[1]-x[0])
        candidates.append((ent, start, end))
    candidates.sort(key=lambda z: z[2]-z[1], reverse=True)
    for ent, cs, ce in candidates:
        began = False
        for i, (s, e) in enumerate(offsets):
            if s == e == 0:  # specials
                continue
            if e <= cs or s >= ce:
                continue
            labels[i] = ("B-" if not began else "I-") + ent
            began = True
    label_ids = [LABEL2ID.get(l, LABEL2ID["O"]) for l in labels]
    return enc["input_ids"], label_ids

class DistantSupervisionDataset(Dataset):
    def __init__(self, tokenizer: AutoTokenizer, intent_name: str = "parts"):
        self.tokenizer = tokenizer
        self.intent_name = intent_name
        self.samples = []
        max_rows = int(os.getenv("PARTS_NER_DS_MAX_ROWS", "5000"))
        n_tpl = int(os.getenv("PARTS_NER_TEMPLATES_PER_ROW", "10"))
        rows = _read_loadsheet(max_rows=max_rows)
        templates = load_templates_from_db(intent_name=intent_name)
        if not templates:
            raise RuntimeError("No active templates found for 'parts' in DB")
        def pick_templates(n: int) -> List[str]:
            if len(templates) <= n:
                reps = (n + len(templates) - 1) // len(templates)
                return (templates * reps)[:n]
            return random.sample(templates, n)
        for r in rows:
            itemnum = r["ITEMNUM"]; desc = r["DESCRIPTION"]; mfg = r["OEMMFG"]; model = r["MODEL"]
            canon_mfg = normalize_manufacturer(mfg)
            for tpl in pick_templates(n_tpl):
                sent = _render_sentence(tpl, desc, mfg, itemnum, model)
                spans = {}
                if "{itemnum}" in tpl and itemnum: spans["PART_NUMBER"] = itemnum
                if "{description}" in tpl and desc: spans["PART_NAME"] = desc
                if "{manufacturer}" in tpl and canon_mfg: spans["MANUFACTURER"] = canon_mfg
                if "{model}" in tpl and model: spans["MODEL"] = model
                input_ids, label_ids = _align_and_tag(sent, spans, tokenizer)
                attn = [1] * len(input_ids)
                if len(input_ids) > MAX_LEN:
                    input_ids = input_ids[:MAX_LEN]; label_ids = label_ids[:MAX_LEN]; attn = attn[:MAX_LEN]
                else:
                    pad = MAX_LEN - len(input_ids)
                    input_ids += [tokenizer.pad_token_id] * pad
                    label_ids += [-100] * pad
                    attn += [0] * pad
                self.samples.append({
                    "input_ids": torch.tensor(input_ids, dtype=torch.long),
                    "attention_mask": torch.tensor(attn, dtype=torch.long),
                    "labels": torch.tensor(label_ids, dtype=torch.long),
                })
        random.shuffle(self.samples)
        cut = max(1, int(len(self.samples) * 0.85))
        self.train = self.samples[:cut]
        self.val   = self.samples[cut:]

# ====================== Utilities ===================
def choose_dataset_size():
    print("Select training dataset size for DB mode:")
    print("1) Small (≈1,000 samples)")
    print("2) Medium (≈10,000 samples)")
    print("3) Full (all available samples)")
    choice = input("Enter choice (1/2/3): ").strip()
    if choice == "1": return 1000
    elif choice == "2": return 10000
    else: return None

def count_ner_rows(session_factory) -> int:
    s = session_factory()
    try:
        return s.execute(text("SELECT COUNT(*) FROM training_sample WHERE sample_type='ner'")).scalar_one()
    finally:
        s.close()

def compute_epoch_and_eval_steps(total_examples: int, per_device_bs: int, grad_accum: int, min_interval: int = 150):
    effective_bs = max(1, per_device_bs * max(1, grad_accum))
    steps_per_epoch = max(1, math.ceil(total_examples / effective_bs))
    eval_save_steps = max(min_interval, steps_per_epoch)
    max_steps = steps_per_epoch * 3
    return steps_per_epoch, max_steps, eval_save_steps

# ===================== Weighted Trainer =====================
class WeightedTrainer(Trainer):
    """Trainer with per-class weights + label smoothing."""
    def __init__(self, *args, class_weights=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = None
        if class_weights is not None:
            self.class_weights = torch.tensor(class_weights, dtype=torch.float)
    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.get("labels")
        outputs = model(**{k: v for k, v in inputs.items() if k != "labels"})
        logits = outputs.logits  # (B, T, C)
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
        self.metric_name = metric_name; self.min_delta = float(min_delta)
        self.patience = int(patience); self.best = None; self.bad_count = 0
    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        if not metrics or self.metric_name not in metrics: return
        current = metrics[self.metric_name]
        if self.best is None or current > self.best + self.min_delta:
            self.best = current; self.bad_count = 0
        else:
            self.bad_count += 1
            if self.bad_count >= self.patience:
                control.should_training_stop = True

class ReduceLROnPlateauCallback(TrainerCallback):
    def __init__(self, metric_name="eval_f1", factor=0.5, patience=2, min_lr=1e-6):
        self.metric_name = metric_name; self.factor = float(factor)
        self.patience = int(patience); self.min_lr = float(min_lr)
        self.best = None; self.bad = 0
    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        if not metrics or self.metric_name not in metrics: return
        current = metrics[self.metric_name]
        if self.best is None or current > self.best:
            self.best = current; self.bad = 0; return
        self.bad += 1
        if self.bad >= self.patience:
            self.bad = 0
            opt = kwargs["trainer"].optimizer
            for group in opt.param_groups:
                group["lr"] = max(self.min_lr, group["lr"] * self.factor)

# ========================= Active Learning =========================
def active_learning_mine_hard_negatives(
    tokenizer: AutoTokenizer,
    model_dir: Path,
    probe_rows: int,
    tpl_per_row: int,
    max_negatives: int,
    min_fp_per_sentence: int = 1,
) -> "DictTensorDataset":
    """
    Build a small probe pool from loadsheet + DB templates, run current model,
    collect sentences with false-positive entities, and return them as hard negatives
    (BIO all 'O') in a DictTensorDataset (each item is a dict for HF Trainer).
    """
    logger.info(f"[AL] Mining hard negatives | probe_rows={probe_rows} tpl_per_row={tpl_per_row} max_neg={max_negatives}")

    rows = _read_loadsheet(max_rows=probe_rows)
    templates = load_templates_from_db(intent_name="parts")
    if not templates:
        logger.warning("[AL] No templates available; skipping mining")
        empty = torch.empty((0, MAX_LEN), dtype=torch.long)
        return DictTensorDataset(empty, empty, empty)

    # Build probe sentences with expected spans
    probes: List[Tuple[str, Dict[str, str]]] = []
    for r in rows:
        itemnum = r["ITEMNUM"]; desc = r["DESCRIPTION"]; mfg = r["OEMMFG"]; model = r["MODEL"]
        canon_mfg = normalize_manufacturer(mfg)
        tpls = templates if len(templates) <= tpl_per_row else random.sample(templates, tpl_per_row)
        for tpl in tpls:
            sent = _render_sentence(tpl, desc, mfg, itemnum, model)
            exp = {}
            if "{itemnum}" in tpl and itemnum: exp["PART_NUMBER"] = itemnum
            if "{description}" in tpl and desc: exp["PART_NAME"] = desc
            if "{manufacturer}" in tpl and canon_mfg: exp["MANUFACTURER"] = canon_mfg
            if "{model}" in tpl and model: exp["MODEL"] = model
            probes.append((sent, exp))

    tok = tokenizer
    nlp = pipeline(
        "token-classification",
        model=str(model_dir),
        tokenizer=tok,
        aggregation_strategy="simple",
        device=-1,  # CPU; change if you want GPU
    )

    hard_negs = []
    negatives_added = 0

    def to_tensor_all_O(sentence_ids: List[int]) -> Dict[str, torch.Tensor]:
        attn = [1] * len(sentence_ids)
        if len(sentence_ids) > MAX_LEN:
            ids = sentence_ids[:MAX_LEN]; att = attn[:MAX_LEN]
        else:
            pad = MAX_LEN - len(sentence_ids)
            ids = sentence_ids + [tok.pad_token_id] * pad
            att = attn + [0] * pad
        # label -100 for padding positions, 'O' for real tokens
        labs = [LABEL2ID["O"]] * min(len(sentence_ids), MAX_LEN)
        if len(labs) < MAX_LEN:
            labs += [-100] * (MAX_LEN - len(labs))
        return {
            "input_ids": torch.tensor(ids, dtype=torch.long),
            "attention_mask": torch.tensor(att, dtype=torch.long),
            "labels": torch.tensor(labs, dtype=torch.long),
        }

    def norm_txt(t: str) -> str:
        t = (t or "").strip().lower()
        t = re.sub(r"\s+", " ", t)
        t = re.sub(r"[.,;:!?'\"-]", "", t)
        return t

    for sent, expected in probes:
        if negatives_added >= max_negatives:
            break
        preds = nlp(sent)
        fp = 0
        exp_norm = {k: norm_txt(v) for k, v in expected.items()}
        for p in preds:
            etype = p.get("entity_group", p.get("label", "")).replace("B-", "").replace("I-", "")
            ptxt = p.get("word", p.get("text", "")).replace("##", "")
            if etype not in exp_norm or norm_txt(ptxt) != exp_norm[etype]:
                fp += 1
        if fp >= min_fp_per_sentence:
            enc = tok(sent, add_special_tokens=True, truncation=True, max_length=MAX_LEN)
            hard_negs.append(to_tensor_all_O(enc["input_ids"]))
            negatives_added += 1

    if not hard_negs:
        logger.info("[AL] No hard negatives mined this round.")
        empty = torch.empty((0, MAX_LEN), dtype=torch.long)
        return DictTensorDataset(empty, empty, empty)

    ids  = torch.stack([hn["input_ids"] for hn in hard_negs])
    attn = torch.stack([hn["attention_mask"] for hn in hard_negs])
    labs = torch.stack([hn["labels"] for hn in hard_negs])
    logger.info(f"[AL] Mined hard negatives: {len(hard_negs)}")
    return DictTensorDataset(ids, attn, labs)


def _append_negatives_to_train(train_dataset, mined: TensorDataset) -> TensorDataset:
    """Return a TensorDataset with train + mined (handles streaming conversion)."""
    if isinstance(train_dataset, TensorDataset):
        return TensorDataset(
            torch.cat([train_dataset.tensors[0], mined.tensors[0]], dim=0),
            torch.cat([train_dataset.tensors[1], mined.tensors[1]], dim=0),
            torch.cat([train_dataset.tensors[2], mined.tensors[2]], dim=0),
        )
    # convert streaming dataset to TensorDataset just for AL stage
    logger.info("[AL] Converting streaming dataset into in-memory TensorDataset for AL fine-tune...")
    all_ids, all_attn, all_labels = [], [], []
    for batch in train_dataset:
        all_ids.append(batch["input_ids"]); all_attn.append(batch["attention_mask"]); all_labels.append(batch["labels"])
    td = TensorDataset(torch.stack(all_ids), torch.stack(all_attn), torch.stack(all_labels))
    return TensorDataset(
        torch.cat([td.tensors[0], mined.tensors[0]], dim=0),
        torch.cat([td.tensors[1], mined.tensors[1]], dim=0),
        torch.cat([td.tensors[2], mined.tensors[2]], dim=0),
    )

def run_active_learning_round(
    base_run_dir: Path,
    trainer: Trainer,
    tokenizer: AutoTokenizer,
    train_dataset,
) -> Tuple[TensorDataset, int]:
    """
    Run one AL mining + short fine-tune round.
    Returns (new_train_dataset, mined_count)
    """
    if not (base_run_dir / "best").is_dir():
        logger.warning("[AL] No best checkpoint found; skipping AL.")
        return train_dataset, 0
    epochs      = int(os.getenv("PARTS_NER_AL_EPOCHS", "1"))
    probe_rows  = int(os.getenv("PARTS_NER_AL_PROBE_ROWS", "500"))
    tpl_per_row = int(os.getenv("PARTS_NER_AL_TPL_PER_ROW", "3"))
    max_neg     = int(os.getenv("PARTS_NER_AL_MAX_NEG", "5000"))
    min_fp      = int(os.getenv("PARTS_NER_AL_MIN_FP", "1"))
    mined = active_learning_mine_hard_negatives(
        tokenizer=tokenizer,
        model_dir=base_run_dir / "best",
        probe_rows=probe_rows,
        tpl_per_row=tpl_per_row,
        max_negatives=max_neg,
        min_fp_per_sentence=min_fp,
    )
    mined_count = len(mined)
    if mined_count == 0:
        logger.info("[AL] Nothing to fine-tune on this round.")
        return train_dataset, 0
    new_train = _append_negatives_to_train(train_dataset, mined)
    logger.info(f"[AL] Fine-tuning on {mined_count} hard negatives for {epochs} epoch(s)")
    orig_epochs = trainer.args.num_train_epochs
    trainer.args.num_train_epochs = epochs
    trainer.train_dataset = new_train
    trainer.train()
    trainer.args.num_train_epochs = orig_epochs
    return new_train, mined_count

# === AL (multi-round) with tiny summary report ===
def run_active_learning_multiround(
    run_dir: Path,
    trainer: Trainer,
    tokenizer: AutoTokenizer,
    train_dataset,
    max_rounds: int,
    min_new_negatives: int,
    total_neg_cap: int,
):
    """
    Perform up to max_rounds of AL with early stop if mined < min_new_negatives.
    Also enforces an overall cap on the total negatives added.
    Logs a tiny end-of-run summary (per-round negatives, totals, F1 before/after).
    """
    total_added = 0
    round_stats = []  # (round_idx, mined_count, post_round_f1)

    # Baseline eval before any AL
    baseline_eval = trainer.evaluate()
    baseline_f1 = float(baseline_eval.get("eval_f1", 0.0))
    logger.info(f"[AL] Baseline F1 before AL: {baseline_f1:.4f}")

    for r in range(1, max_rounds + 1):
        logger.info(f"[AL] === Round {r}/{max_rounds} ===")
        train_dataset, mined_count = run_active_learning_round(run_dir, trainer, tokenizer, train_dataset)

        # Evaluate after this round
        round_eval = trainer.evaluate()
        round_f1 = float(round_eval.get("eval_f1", 0.0))

        if mined_count <= 0:
            logger.info(f"[AL] Round {r}: 0 mined — stopping early.")
            break

        total_added += mined_count
        round_stats.append((r, mined_count, round_f1))

        # Persist after each round
        save_best_artifacts(trainer, tokenizer, run_dir)
        write_latest_pointer(Path(ORC_PARTS_MODEL_DIR), run_dir)
        logger.info(f"[AL] Round {r} complete. Mined this round: {mined_count}, total added: {total_added}")

        # Safety checks
        if mined_count < min_new_negatives:
            logger.info(f"[AL] Mined {mined_count} < min_new_negatives ({min_new_negatives}) — stopping.")
            break
        if total_added >= total_neg_cap:
            logger.info(f"[AL] Reached total_neg_cap ({total_neg_cap}) — stopping.")
            break

    # === Final tiny summary report ===
    logger.info("=== Active Learning Summary ===")
    logger.info(f"Baseline F1: {baseline_f1:.4f}")
    for r, negs, f1 in round_stats:
        logger.info(f"Round {r}: Mined {negs} negatives | Post-round F1: {f1:.4f}")
    logger.info(f"Total Negatives Added: {total_added}")
    if round_stats:
        logger.info(f"Final F1 after AL: {round_stats[-1][2]:.4f}")
    else:
        logger.info("No AL rounds performed.")

# ========================= Main =====================
def maybe_freeze_encoder(model, freeze=True, unfreeze_last_n=2, backbone_prefixes=("bert", "distilbert")):
    if not freeze: return
    for name, param in model.named_parameters():
        if name.split(".")[0] in backbone_prefixes:
            param.requires_grad = False
    for k in list(dict(model.named_parameters()).keys()):
        for pfx in backbone_prefixes:
            if f"{pfx}.encoder.layer." in k:
                try:
                    layer_idx = int(k.split(f"{pfx}.encoder.layer.")[1].split(".")[0])
                    if layer_idx >= (11 - (unfreeze_last_n - 1)):
                        model.get_parameter(k).requires_grad = True
                except Exception:
                    pass

def main():
    # ---------- tiny helpers (local to main) ----------
    def ask_yn(msg: str, default: bool = False) -> bool:
        d = "y" if default else "n"
        ans = input(f"{msg} (y/n, default {d}): ").strip().lower()
        if ans == "":
            return default
        return ans.startswith("y")

    def ask_int(msg: str, default: int) -> int:
        ans = input(f"{msg} (default {default}): ").strip()
        if ans == "":
            return default
        try:
            return int(ans)
        except Exception:
            print(f"Invalid integer. Using default {default}.")
            return default

    def ask_choice(msg: str, choices: list, default: str) -> str:
        choices_str = "/".join(choices)
        ans = input(f"{msg} [{choices_str}] (default {default}): ").strip().lower()
        if ans == "":
            return default
        if ans in choices:
            return ans
        print(f"Invalid choice. Using default {default}.")
        return default

    # ---------- choose data mode by prompt (no env needed) ----------
    data_mode = ask_choice("\nSelect data mode", ["db", "distant"], default="db")  # "db" or "distant"

    # ---------- Active Learning: prompt-driven (no PowerShell needed) ----------
    do_active = ask_yn("\nEnable Active Learning?", default=True)

    # Defaults that were previously env-driven
    al_max_rounds    = 2
    al_min_new_neg   = 100
    al_total_neg_cap = 8000
    # per-round mining/fine-tune params (read inside AL helpers via env, so we set env after prompts)
    al_probe_rows    = 1000
    al_tpl_per_row   = 4
    al_epochs        = 1
    al_max_neg       = 3000
    al_min_fp        = 1

    if do_active:
        print("\n=== Active Learning Settings ===")
        al_max_rounds    = ask_int("Max AL rounds", al_max_rounds)
        al_min_new_neg   = ask_int("Min new negatives per round", al_min_new_neg)
        al_total_neg_cap = ask_int("Total negatives cap (across all rounds)", al_total_neg_cap)
        al_probe_rows    = ask_int("Probe rows per round", al_probe_rows)
        al_tpl_per_row   = ask_int("Templates per row", al_tpl_per_row)
        al_epochs        = ask_int("Epochs per AL round", al_epochs)
        al_max_neg       = ask_int("Max negatives to mine per round", al_max_neg)
        al_min_fp        = ask_int("Min FP entities required to keep a sentence", al_min_fp)

        # These helpers read some params from env; set them from prompts:
        os.environ["PARTS_NER_AL_PROBE_ROWS"]   = str(al_probe_rows)
        os.environ["PARTS_NER_AL_TPL_PER_ROW"]  = str(al_tpl_per_row)
        os.environ["PARTS_NER_AL_EPOCHS"]       = str(al_epochs)
        os.environ["PARTS_NER_AL_MAX_NEG"]      = str(al_max_neg)
        os.environ["PARTS_NER_AL_MIN_FP"]       = str(al_min_fp)

    # ---------- keep existing env-based training knobs (optional to prompt later) ----------
    per_device_bs = int(os.getenv("PARTS_NER_BATCH", "8"))
    grad_accum    = int(os.getenv("PARTS_NER_GRAD_ACCUM", "4"))
    num_epochs    = int(os.getenv("PARTS_NER_EPOCHS", "4"))
    eval_frac     = float(os.getenv("PARTS_NER_EVAL_FRAC", "0.15"))

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)

    # ======= Build datasets =======
    if data_mode == "distant":
        logger.info("[DATA] Using distant supervision mode (loadsheet + DB templates)")
        ds = DistantSupervisionDataset(tokenizer, intent_name="parts")
        train_dataset = TensorDataset(
            torch.stack([r["input_ids"] for r in ds.train]),
            torch.stack([r["attention_mask"] for r in ds.train]),
            torch.stack([r["labels"] for r in ds.train]),
        )
        eval_dataset = TensorDataset(
            torch.stack([r["input_ids"] for r in ds.val]),
            torch.stack([r["attention_mask"] for r in ds.val]),
            torch.stack([r["labels"] for r in ds.val]),
        )
        total_examples = len(ds.train)
        data_collator = DataCollatorForTokenClassification(tokenizer)
    else:
        logger.info("[DATA] Using DB mode (training_sample streaming)")
        max_examples = choose_dataset_size()
        session = SessionLocal()
        try:
            if max_examples is None:
                total_examples_all = count_ner_rows(SessionLocal)
                eval_count = max(1, int(total_examples_all * eval_frac))
                eval_ids = [r[0] for r in session.execute(
                    text("SELECT id FROM training_sample WHERE sample_type='ner' ORDER BY random() LIMIT :k"),
                    {"k": eval_count},
                ).fetchall()]
                train_dataset = DBStreamingNERDataset(
                    tokenizer=tokenizer,
                    session_factory=SessionLocal,
                    max_length=MAX_LEN,
                    shuffle_buffer_size=1000,
                    max_examples=None,
                    exclude_ids=set(eval_ids),
                )
                eval_rows = session.execute(
                    text("SELECT id, text, entities FROM training_sample WHERE id = ANY(:ids)"),
                    {"ids": eval_ids},
                ).fetchall()
                eval_dataset = ListNERDataset(tokenizer, eval_rows, max_length=MAX_LEN)
                total_examples = count_ner_rows(SessionLocal) - len(eval_rows)
            else:
                pool_rows = session.execute(
                    text("SELECT id, text, entities FROM training_sample WHERE sample_type='ner' ORDER BY random() LIMIT :n"),
                    {"n": max_examples},
                ).fetchall()
                random.shuffle(pool_rows)
                cut = max(1, int(len(pool_rows) * (1 - eval_frac)))
                train_rows = pool_rows[:cut]; eval_rows = pool_rows[cut:]
                train_dataset = ListNERDataset(tokenizer, train_rows, max_length=MAX_LEN)
                eval_dataset  = ListNERDataset(tokenizer, eval_rows,  max_length=MAX_LEN)
                total_examples = len(train_rows)
        finally:
            session.close()
        data_collator = None

    # -------------- Steps math (auto cadence) --------------
    steps_per_epoch, max_steps, eval_save_steps = compute_epoch_and_eval_steps(
        total_examples=total_examples, per_device_bs=per_device_bs, grad_accum=grad_accum, min_interval=150,
    )

    # -------------- Model with stronger regularization ----------
    cfg = AutoConfig.from_pretrained(MODEL_NAME)
    cfg.hidden_dropout_prob = 0.20
    cfg.attention_probs_dropout_prob = 0.20
    cfg.num_labels = len(LABELS)
    cfg.id2label = ID2LABEL
    cfg.label2id = LABEL2ID
    model = AutoModelForTokenClassification.from_pretrained(MODEL_NAME, config=cfg)

    if data_mode == "db" and os.getenv("PARTS_NER_FREEZE", "true").lower() == "true":
        maybe_freeze_encoder(model, freeze=True, unfreeze_last_n=2)

    base_dir = Path(ORC_PARTS_MODEL_DIR)
    run_dir = next_run_dir(base_dir)
    logger.info(f"[TRAIN] Output run directory: {run_dir}")
    maintain_training_logs(retention_weeks=2)

    training_args = TrainingArguments(
        output_dir=str(run_dir),
        overwrite_output_dir=True,
        per_device_train_batch_size=per_device_bs,
        gradient_accumulation_steps=grad_accum,
        num_train_epochs=num_epochs,          # for logs
        max_steps=max_steps,                  # exact (3 epochs)
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
        learning_rate=float(os.getenv("PARTS_NER_LR", "4e-5")),
        label_smoothing_factor=0.10,
        max_grad_norm=1.0,
        logging_steps=max(50, steps_per_epoch // 2),
        remove_unused_columns=False,
        report_to=["none"],
    )

    # Gently upweight MODEL labels
    class_weights = [1.0] * len(LABELS)
    for k in ("B-MODEL", "I-MODEL"):
        class_weights[LABEL2ID[k]] = float(os.getenv("PARTS_NER_MODEL_CLASS_WEIGHT", "2.0"))

    with TrainingLogManager(run_dir=run_dir, to_console=False) as tlogm:
        train_log = tlogm.logger
        cb = tlogm.make_trainer_callback()
        train_log.info("=== Training session starting ===")
        train_log.info(f"Run dir: {run_dir}")
        train_log.info(f"Backbone: {MODEL_NAME} | Max seq len: {MAX_LEN}")
        train_log.info(f"Labels: {LABELS}")
        train_log.info(f"Training examples (est.): {total_examples}")
        train_log.info(f"Steps/epoch: {steps_per_epoch} | Max steps (3 epochs): {max_steps}")
        train_log.info(f"Eval/Save every: {eval_save_steps} steps")
        train_log.info(f"Data mode: {data_mode}")
        train_log.info(f"Active Learning: {do_active}")

        trainer = WeightedTrainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            data_collator=data_collator,
            compute_metrics=compute_token_metrics,
            callbacks=[
                EarlyStoppingCallback(early_stopping_patience=3),
                EarlyStopMinDeltaCallback(metric_name="eval_f1", min_delta=1e-3, patience=2),
                ReduceLROnPlateauCallback(metric_name="eval_f1", factor=0.5, patience=2, min_lr=1e-6),
                cb,
            ],
            class_weights=class_weights,
        )

        # ======= Stage 1: base training =======
        trainer.train()
        eval_res = trainer.evaluate()
        train_log.info(f"[EVAL] {json.dumps(eval_res, indent=2)}")
        save_best_artifacts(trainer, tokenizer, run_dir)
        write_latest_pointer(base_dir, run_dir)
        prune_old_runs(base_dir, keep=5)

        # ======= Stage 2: multi-round Active Learning (safety capped) =======
        if do_active:
            train_log.info("[AL] Starting multi-round active learning...")
            run_active_learning_multiround(
                run_dir=run_dir,
                trainer=trainer,
                tokenizer=tokenizer,
                train_dataset=train_dataset,
                max_rounds=al_max_rounds,
                min_new_negatives=al_min_new_neg,
                total_neg_cap=al_total_neg_cap,
            )
            save_best_artifacts(trainer, tokenizer, run_dir)
            write_latest_pointer(base_dir, run_dir)
            train_log.info("[AL] Multi-round AL complete.")

        train_log.info(f"[SAVE] Best model → {run_dir / 'best'}")
        train_log.info("=== Training session complete ===")


if __name__ == "__main__":
    main()
