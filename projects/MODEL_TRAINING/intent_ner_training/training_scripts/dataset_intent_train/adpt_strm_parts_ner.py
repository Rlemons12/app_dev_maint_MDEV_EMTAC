"""
Adaptive streaming NER training for 'parts' with detailed logging.

This mirrors your drawings adaptive streaming trainer but swaps in the parts
label schema and PARTS config paths. It keeps your logging approach and
best-checkpoint export behavior.
"""
# Disable TorchDynamo to avoid ONNX DiagnosticOptions import errors
import os
os.environ["TORCHDYNAMO_DISABLE"] = "1"

import os
import json
import sys
import argparse
import logging
import numpy as np
import random
import torch
import psutil
import platform
import shutil
import re
from pathlib import Path

from torch.utils.data import IterableDataset, DataLoader
from datasets import Dataset
from transformers import (
    DistilBertTokenizerFast,
    DistilBertForTokenClassification,
    DataCollatorForTokenClassification,
    Trainer,
    TrainingArguments,
    EarlyStoppingCallback,
)
from seqeval.metrics import f1_score, accuracy_score

# === Config paths for PARTS data/model ===
# (These match your non-streaming parts trainer)
from modules.configuration.config import ORC_PARTS_TRAIN_DATA_DIR, ORC_PARTS_MODEL_DIR  # <- IMPORTANT

# === Your custom logger ===
from modules.configuration.log_config import (
    debug_id, info_id, warning_id, error_id, critical_id,
    with_request_id, set_request_id, get_request_id,
    log_timed_operation
)

# Keep Python's root logger quiet; we route through your custom logger
logging.basicConfig(level=logging.WARNING)

# ---- NER labels for PARTS ----
LABELS = [
    "O",
    "B-PART_NUMBER", "I-PART_NUMBER",
    "B-PART_NAME", "I-PART_NAME",
    "B-MANUFACTURER", "I-MANUFACTURER",
    "B-MODEL", "I-MODEL",
]
ID2LABEL = {i: lab for i, lab in enumerate(LABELS)}
LABEL2ID = {lab: i for i, lab in enumerate(LABELS)}

# Optional: simple regex override for patterns like A1xxxxx (tweak as needed)
PART_NUMBER_RE = re.compile(r"\bA1\d{5}\b", re.IGNORECASE)


# ===== Helper logging utilities =====
def _sorted_id2label_str(id2label):
    try:
        items = sorted(((int(k), v) for k, v in id2label.items()))
    except Exception:
        items = sorted(
            id2label.items(),
            key=lambda t: int(t[0]) if isinstance(t[0], str) and str(t[0]).isdigit() else t[0]
        )
    return ", ".join(f"{k}:{v}" for k, v in items)


def log_label_schema(title: str, id2label: dict, label2id: dict, request_id=None):
    info_id(f"[{title}] id2label={{ {_sorted_id2label_str(id2label)} }}", request_id)
    keys_preview = sorted(list(label2id.keys()))[:10]
    info_id(f"[{title}] label2id keys (first 10)={keys_preview}{'...' if len(label2id) > 10 else ''}", request_id)


def log_model_folder_labels(model_dir: Path, request_id=None):
    cfg_file = model_dir / "config.json"
    if not cfg_file.exists():
        warning_id(f"[Model folder] No config.json at {model_dir}", request_id)
        return None
    try:
        cfg = json.loads(cfg_file.read_text(encoding="utf-8"))
        id2 = cfg.get("id2label", {})
        l2i = cfg.get("label2id", {})
        info_id(f"[Model folder] {model_dir}", request_id)
        info_id(f"[Model folder] model_type={cfg.get('model_type')} architectures={cfg.get('architectures')}", request_id)
        info_id(f"[Model folder] num_labels={cfg.get('num_labels') or cfg.get('_num_labels')}", request_id)
        log_label_schema("Model folder", id2, l2i, request_id)
        return cfg
    except Exception as e:
        error_id(f"Failed to read model folder config.json: {e}", request_id)
        return None


def log_model_object_labels(hf_model, title="Model", request_id=None):
    try:
        c = getattr(hf_model, "config", None)
        if not c:
            warning_id(f"[{title}] no config attribute", request_id)
            return
        id2 = getattr(c, "id2label", {}) or {}
        l2i = getattr(c, "label2id", {}) or {}
        info_id(f"[{title}] num_labels={getattr(c, 'num_labels', 'n/a')}", request_id)
        log_label_schema(title, id2, l2i, request_id)
    except Exception as e:
        error_id(f"Failed logging {title} labels: {e}", request_id)


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
                "memory_gb": memory_gb,
                "cpu_count": cpu_count,
                "cpu_count_physical": cpu_count_physical,
                "platform": platform.system(),
                "processor": platform.processor(),
                "python_version": platform.python_version(),
            }
        except Exception as e:
            warning_id(f"Failed to collect system info, using defaults: {e}", self.request_id)
            return {"memory_gb": 8, "cpu_count": 4, "cpu_count_physical": 2}

    def _get_gpu_info(self):
        gpu_info = {"available": False, "name": None, "memory_gb": 0, "compute_capability": None}
        try:
            if torch.cuda.is_available():
                gpu_info["available"] = True
                gpu_info["name"] = torch.cuda.get_device_name(0)
                props = torch.cuda.get_device_properties(0)
                gpu_info["memory_gb"] = props.total_memory / (1024 ** 3)
                gpu_info["compute_capability"] = f"{props.major}.{props.minor}"
                info_id(f"GPU detected: {gpu_info['name']} ({gpu_info['memory_gb']:.1f}GB, CC {gpu_info['compute_capability']})",
                        self.request_id)
            else:
                info_id("No GPU detected - using CPU training", self.request_id)
        except Exception as e:
            warning_id(f"GPU probe failed: {e}", self.request_id)
        return gpu_info

    def _calculate_optimal_config(self):
        config = {}
        config["use_gpu"] = self.gpu_info["available"]
        config["fp16"] = self.gpu_info["available"]

        if self.gpu_info["available"]:
            m = self.gpu_info["memory_gb"]
            if m >= 16:
                config["batch_size"] = 16; config["max_length"] = 256
            elif m >= 8:
                config["batch_size"] = 8; config["max_length"] = 128
            elif m >= 4:
                config["batch_size"] = 4; config["max_length"] = 128
            else:
                config["batch_size"] = 2; config["max_length"] = 64
        else:
            r = self.system_info["memory_gb"]
            if r >= 32:
                config["batch_size"] = 8; config["max_length"] = 128
            elif r >= 16:
                config["batch_size"] = 4; config["max_length"] = 128
            elif r >= 8:
                config["batch_size"] = 2; config["max_length"] = 64
            else:
                config["batch_size"] = 1; config["max_length"] = 64

        cpu = self.system_info["cpu_count"]
        config["num_workers"] = min(4, max(1, cpu // 4)) if self.gpu_info["available"] else 0

        r = self.system_info["memory_gb"]
        config["shuffle_buffer_size"] = 2000 if r >= 16 else (1000 if r >= 8 else 500)

        target_eff = 16
        config["gradient_accumulation_steps"] = max(1, target_eff // config["batch_size"])
        eff = config["batch_size"] * config["gradient_accumulation_steps"]
        config["learning_rate"] = 5e-5 * (eff / 16)
        return config

    def get_recommended_mode_for_examples(self, max_examples):
        if max_examples is None:
            return {"num_epochs": 2 if self.gpu_info["available"] else 1, "eval_steps": 2000, "save_steps": 2000, "estimated_time_hours": 12 if self.gpu_info["available"] else 48}
        if max_examples <= 10_000:
            return {"num_epochs": 5, "eval_steps": 100, "save_steps": 100, "estimated_time_hours": 0.2 if self.gpu_info["available"] else 0.5}
        if max_examples <= 100_000:
            return {"num_epochs": 3, "eval_steps": 500, "save_steps": 500, "estimated_time_hours": 1 if self.gpu_info["available"] else 3}
        return {"num_epochs": 2, "eval_steps": 1000, "save_steps": 1000, "estimated_time_hours": 4 if self.gpu_info["available"] else 12}

    def print_optimization_summary(self):
        print("\n" + "=" * 60)
        print("SYSTEM OPTIMIZATION SUMMARY")
        print("=" * 60)
        print(f"Platform: {self.system_info['platform']}")
        print(f"RAM: {self.system_info['memory_gb']:.1f}GB")
        print(f"CPUs: {self.system_info['cpu_count']} logical ({self.system_info['cpu_count_physical']} physical)")
        if self.gpu_info["available"]:
            print(f"GPU: {self.gpu_info['name']} ({self.gpu_info['memory_gb']:.1f}GB)")
            print(f"Compute Capability: {self.gpu_info['compute_capability']}")
        else:
            print("GPU: None (CPU training)")
        print("\nOptimal Configuration:")
        print(f"- Batch Size: {self.optimal_config['batch_size']}")
        print(f"- Max Length: {self.optimal_config['max_length']}")
        print(f"- Gradient Accumulation: {self.optimal_config['gradient_accumulation_steps']}")
        print(f"- Learning Rate: {self.optimal_config['learning_rate']:.2e}")
        print(f"- Shuffle Buffer: {self.optimal_config['shuffle_buffer_size']}")
        print(f"- Data Workers: {self.optimal_config['num_workers']}")
        print(f"- Mixed Precision: {self.optimal_config['fp16']}")
        print("=" * 60 + "\n")


class StreamingNERDataset(IterableDataset):
    def __init__(self, jsonl_file, tokenizer, max_length=128, max_examples=None,
                 shuffle_buffer_size=1000, skip_examples=0, epoch=0, request_id=None):
        self.jsonl_file = jsonl_file
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.max_examples = max_examples
        self.shuffle_buffer_size = shuffle_buffer_size
        self.skip_examples = skip_examples
        self.epoch = epoch
        self.request_id = request_id

    def convert_example_to_ner_format(self, example):
        text = example["text"]
        entities = example.get("entities", [])
        words = text.split()
        word_starts, pos = [], 0
        for w in words:
            s = text.find(w, pos)
            word_starts.append(s); pos = s + len(w)
        labels = ["O"] * len(words)
        for ent in entities:
            e_start, e_end, e_type = ent["start"], ent["end"], ent["entity"]
            first = last = None
            for i, w_start in enumerate(word_starts):
                w_end = w_start + len(words[i])
                if w_start < e_end and w_end > e_start:
                    if first is None: first = i
                    last = i
            if first is not None:
                labels[first] = f"B-{e_type}"
                for i in range(first + 1, (last or first) + 1):
                    labels[i] = f"I-{e_type}"
        label_ids = [LABEL2ID.get(l, 0) for l in labels]
        return {"tokens": words, "ner_tags": label_ids}

    def tokenize_example(self, example):
        tok = self.tokenizer(
            example["tokens"],
            is_split_into_words=True,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt"
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
                        labels.append(LABEL2ID[inside])
                    else:
                        labels.append(lab)
                prev = wid
        return {
            "input_ids": tok["input_ids"].squeeze(),
            "attention_mask": tok["attention_mask"].squeeze(),
            "labels": torch.tensor(labels, dtype=torch.long)
        }

    def __iter__(self):
        random.seed(42 + self.epoch)
        buffer, count = [], 0
        try:
            with open(self.jsonl_file, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(f, 1):
                    if line_num <= self.skip_examples: continue
                    if self.max_examples and count >= self.max_examples: break
                    try:
                        ex = json.loads(line)
                        ner = self.convert_example_to_ner_format(ex)
                        tokenized = self.tokenize_example(ner)
                        buffer.append(tokenized)
                        if len(buffer) >= self.shuffle_buffer_size:
                            random.shuffle(buffer)
                            for item in buffer: yield item
                            buffer = []
                        count += 1
                        if count % 1000 == 0:
                            info_id(f"Streamed {count} examples...", self.request_id)
                    except (json.JSONDecodeError, KeyError) as e:
                        warning_id(f"Skipping bad line #{line_num}: {e}", self.request_id)
                        continue
        except FileNotFoundError as e:
            error_id(f"Training data file not found: {e}", self.request_id)
            return
        if buffer:
            random.shuffle(buffer)
            for item in buffer: yield item


class StreamingTrainer(Trainer):
    def __init__(self, streaming_dataset_config=None, request_id=None, **kwargs):
        super().__init__(**kwargs)
        self.streaming_dataset_config = streaming_dataset_config or {}
        self.current_epoch = 0
        self.request_id = request_id

    def get_train_dataloader(self):
        if self.train_dataset is None:
            raise ValueError("Trainer: training requires a train_dataset.")
        ds = StreamingNERDataset(
            epoch=self.current_epoch,
            request_id=self.request_id,
            **self.streaming_dataset_config
        )
        return DataLoader(
            ds,
            batch_size=self.args.per_device_train_batch_size,
            collate_fn=self.data_collator,
            num_workers=self.args.dataloader_num_workers,
            pin_memory=self.args.dataloader_pin_memory and torch.cuda.is_available()
        )

    def _inner_training_loop(self, **kwargs):
        info_id(f"Starting epoch {self.current_epoch}", self.request_id)
        result = super()._inner_training_loop(**kwargs)
        info_id(f"Completed epoch {self.current_epoch}", self.request_id)
        self.current_epoch += 1
        return result


def create_validation_dataset(jsonl_file, tokenizer, max_length=128, val_size=5000, request_id=None):
    """
    Builds a small validation set that accepts EITHER:
      - {"tokens":[...], "ner_tags":[...]}  (already tokenized)
      - {"text":"...", "entities":[{"start":..,"end":..,"entity": "..."}]} (raw)
    Produces a HuggingFace Dataset of dicts compatible with DataCollatorForTokenClassification.
    """
    def align(tokens, word_labels):
        # word->token alignment BEFORE tensors, to preserve word_ids()
        tok = tokenizer(
            tokens,
            is_split_into_words=True,
            truncation=True,
            padding="max_length",
            max_length=max_length,
        )
        word_ids = tok.word_ids()
        aligned = []
        prev = None
        for wid in word_ids:
            if wid is None:
                aligned.append(-100)
            else:
                base = LABEL2ID.get(word_labels[wid], 0)
                if wid != prev:
                    aligned.append(base)
                else:
                    lab = ID2LABEL[base]
                    if lab.startswith("B-"):
                        aligned.append(LABEL2ID.get("I-" + lab[2:], base))
                    else:
                        aligned.append(base)
                prev = wid
        return {
            "input_ids": tok["input_ids"],
            "attention_mask": tok["attention_mask"],
            "labels": aligned,
        }

    def raw_to_words_and_labels(text, entities):
        words = text.split()
        labels = ["O"] * len(words)
        run = 0
        for ent in entities or []:
            e_start, e_end = ent["start"], ent["end"]
            e_type = ent.get("entity") or ent.get("label")
            if not e_type:
                continue
            first = last = None
            run = 0
            for j, w in enumerate(words):
                w_start = text.find(w, run); w_end = w_start + len(w); run = w_end + 1
                if w_start < e_end and w_end > e_start:
                    if first is None: first = j
                    last = j
            if first is not None:
                labels[first] = f"B-{e_type}"
                for k in range(first + 1, (last or first) + 1):
                    labels[k] = f"I-{e_type}"
        return words, labels

    rows = []
    kept, bad = 0, 0
    with open(jsonl_file, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if val_size is not None and i >= val_size:
                break
            try:
                ex = json.loads(line)
                if "tokens" in ex and "ner_tags" in ex:
                    tokens = ex["tokens"]
                    tag_ids = ex["ner_tags"]
                    # convert tag_ids -> BIO tag strings for alignment logic
                    word_labels = [ID2LABEL[t] if isinstance(t, int) else t for t in tag_ids]
                    rows.append(align(tokens, word_labels))
                    kept += 1
                elif "text" in ex:
                    text = ex["text"]
                    entities = ex.get("entities", [])
                    tokens, word_labels = raw_to_words_and_labels(text, entities)
                    rows.append(align(tokens, word_labels))
                    kept += 1
                else:
                    bad += 1
            except Exception:
                bad += 1
                continue

    info_id(f"Validation examples kept={kept}, bad_lines_skipped={bad}", request_id)
    # build Dataset from already-aligned features
    return Dataset.from_dict({
        "input_ids": [r["input_ids"] for r in rows],
        "attention_mask": [r["attention_mask"] for r in rows],
        "labels": [r["labels"] for r in rows],
    })


def simple_compute_metrics_fn(p):
    if hasattr(p, "predictions"):
        predictions, labels = p.predictions, p.label_ids
    else:
        predictions, labels = p
    predictions = np.argmax(predictions, axis=2)

    true_labels, true_predictions = [], []
    for lab, pred in zip(labels, predictions):
        tl = [LABELS[l] for l, _ in zip(lab, pred) if l != -100]
        tp = [LABELS[p] for l, p in zip(lab, pred) if l != -100]
        if tl:
            true_labels.append(tl)
            true_predictions.append(tp)

    try:
        return {
            "f1": f1_score(true_labels, true_predictions),
            "accuracy": accuracy_score(true_labels, true_predictions),
        }
    except Exception:
        return {"f1": 0.0, "accuracy": 0.0}


def _extract_entities_from_text(text, tokenizer, model, id2label):
    model.eval()
    enc = tokenizer(text, return_offsets_mapping=True, return_tensors="pt",
                    truncation=True, max_length=512, return_token_type_ids=False)
    with torch.no_grad():
        out = model(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"])
        logits = out.logits[0]
        probs = torch.softmax(logits, dim=-1)
        pred_ids = logits.argmax(dim=-1)

    tokens = tokenizer.convert_ids_to_tokens(enc["input_ids"][0])
    offsets = enc["offset_mapping"][0].tolist()

    spans, cur = [], None

    def flush():
        nonlocal cur
        if cur is not None and cur["start"] < cur["end"]:
            avg = cur["score_sum"] / max(cur["steps"], 1)
            spans.append({"label": cur["label"], "start": cur["start"], "end": cur["end"],
                          "text": text[cur["start"]:cur["end"]], "score": float(avg)})
        cur = None

    for tok, (s, e), pid, p in zip(tokens, offsets, pred_ids.tolist(), probs.tolist()):
        if tok in ("[CLS]", "[SEP]") or s == e:
            continue
        lab = id2label.get(pid, "O")
        if lab == "O":
            flush(); continue
        if lab.startswith("B-"):
            flush(); cur = {"label": lab[2:], "start": s, "end": e, "score_sum": p[pid], "steps": 1}
        elif lab.startswith("I-"):
            if cur is not None and cur["label"] == lab[2:] and s >= cur["end"]:
                cur["end"] = e; cur["score_sum"] += p[pid]; cur["steps"] += 1
            else:
                flush(); cur = {"label": lab[2:], "start": s, "end": e, "score_sum": p[pid], "steps": 1}
        else:
            flush()
    flush()

    # Regex override for part numbers like A1xxxxx
    overrides = [(m.start(), m.end()) for m in PART_NUMBER_RE.finditer(text)]
    if overrides:
        kept = []
        for sp in spans:
            if any(not (sp["end"] <= s or sp["start"] >= e) for (s, e) in overrides):
                continue  # drop overlapping; we'll insert canonical PART_NUMBER
            kept.append(sp)
        for s, e in overrides:
            kept.append({"label": "PART_NUMBER", "start": s, "end": e,
                         "text": text[s:e], "score": 1.0})
        spans = kept

    spans.sort(key=lambda x: x["start"])
    return spans


def run_demo(tokenizer, model, id2label):
    print("\n=== Quick NER demo (PARTS) ===")
    samples = [
        "Do you have manufacturer part number RB20080S by balston filt?",
        "Need MPN 200-80-BX from Balston FILT, filter tube 10/box.",
        "Looking for item A101576 (model 200-80-BX) made by Balston.",
        "What's the name of A123456?",
    ]
    try:
        for s in samples:
            ents = _extract_entities_from_text(s, tokenizer, model, id2label)
            print(f"\n{s}")
            for ent in ents:
                print(f"→ {ent['label']}: {ent['text']}  (conf: {ent['score']:.3f})")
        while True:
            q = input("\nType a sentence to tag (or press Enter to skip): ").strip()
            if not q:
                break
            ents = _extract_entities_from_text(q, tokenizer, model, id2label)
            for ent in ents:
                print(f"→ {ent['label']}: {ent['text']}  (conf: {ent['score']:.3f})")
    except KeyboardInterrupt:
        pass


def main():
    # Tag this whole run with a request id for coherent logs
    set_request_id()
    rid = get_request_id()
    info_id("Starting adaptive streaming training (PARTS NER)", rid)

    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=['fast', 'small', 'medium', 'full'], default=None)
    parser.add_argument("--max-examples", type=int, default=None)
    parser.add_argument("--force-cpu", action="store_true")
    parser.add_argument("--override-batch-size", type=int, default=None)
    parser.add_argument("--override-max-length", type=int, default=None)
    parser.add_argument("--probe-model-folder-labels", action="store_true",
                        help="Log current model folder config.json label maps before training")
    args = parser.parse_args()

    optimizer = SystemOptimizer(request_id=rid)
    optimizer.print_optimization_summary()

    if not args.mode and not args.max_examples:
        print("Select training mode:\n1. Fast (2K)\n2. Small (10K)\n3. Medium (500K)\n4. Full (ALL)")
        while True:
            choice = input("Enter choice [1/2/3/4]: ").strip()
            if choice in {"1", "2", "3", "4"}:
                args.mode = {"1": "fast", "2": "small", "3": "medium", "4": "full"}[choice]
                break
            print("Please enter 1, 2, 3, or 4")

    if not args.max_examples:
        args.max_examples = {"fast": 2000, "small": 10000, "medium": 500000, "full": None}[args.mode]

    info_id(f"Python executable: {sys.executable}", rid)
    import transformers
    info_id(f"Transformers version: {transformers.__version__}", rid)

    # Data & model locations (parts)
    train_file = os.path.join(ORC_PARTS_TRAIN_DATA_DIR, "ner_train_parts.jsonl")
    model_dir = ORC_PARTS_MODEL_DIR
    os.makedirs(model_dir, exist_ok=True)

    if args.probe_model_folder_labels:
        log_model_folder_labels(Path(model_dir), rid)

    config = optimizer.optimal_config
    mode_cfg = optimizer.get_recommended_mode_for_examples(args.max_examples)

    if args.force_cpu:
        config["use_gpu"] = False
        config["fp16"] = False
        info_id("Forcing CPU training (GPU disabled)", rid)

    if args.override_batch_size:
        config["batch_size"] = args.override_batch_size
        info_id(f"Overriding batch size to {args.override_batch_size}", rid)

    if args.override_max_length:
        config["max_length"] = args.override_max_length
        info_id(f"Overriding max length to {args.override_max_length}", rid)

    # Log the label schema we'll train with
    log_label_schema("Training schema (PARTS)", ID2LABEL, LABEL2ID, rid)

    info_id(f"Training mode: {args.mode or 'custom'}", rid)
    info_id(f"Max examples: {args.max_examples or 'ALL'}", rid)
    info_id(f"Estimated training time: {mode_cfg['estimated_time_hours']:.1f} hours", rid)
    info_id("Using adaptive configuration for your system", rid)

    with log_timed_operation("tokenizer_init", rid):
        tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")

    info_id("Creating validation dataset...", rid)
    with log_timed_operation("build_validation_dataset", rid):
        val_dataset = create_validation_dataset(train_file, tokenizer, config["max_length"], request_id=rid)
    info_id(f"Validation dataset size: {len(val_dataset)}", rid)

    with log_timed_operation("model_init", rid):
        model = DistilBertForTokenClassification.from_pretrained(
            "distilbert-base-uncased",
            num_labels=len(LABELS),
            id2label=ID2LABEL,
            label2id=LABEL2ID,
        )
        log_model_object_labels(model, "Fresh model", rid)

    if config["use_gpu"] and not args.force_cpu:
        model = model.cuda()
        info_id("Moved model to CUDA", rid)

    data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)
    steps_per_epoch = max(1, (args.max_examples or 16000) // config["batch_size"])
    info_id(f"Estimated steps per epoch: {steps_per_epoch}", rid)

    training_args = TrainingArguments(
        output_dir=model_dir,
        evaluation_strategy="steps",  # <-- correct
        save_strategy="steps",
        eval_steps=mode_cfg["eval_steps"],
        save_steps=mode_cfg["save_steps"],
        save_total_limit=3,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,

        learning_rate=config["learning_rate"],
        weight_decay=0.01,
        warmup_ratio=0.1,
        per_device_train_batch_size=config["batch_size"],
        per_device_eval_batch_size=config["batch_size"],
        gradient_accumulation_steps=config["gradient_accumulation_steps"],
        num_train_epochs=mode_cfg["num_epochs"],
        max_steps=steps_per_epoch * mode_cfg["num_epochs"],

        logging_dir=os.path.join(model_dir, "logs"),
        logging_steps=50,
        report_to="none",
        seed=42,
        fp16=config["fp16"] and not args.force_cpu,
        dataloader_pin_memory=config["use_gpu"] and not args.force_cpu,
        dataloader_num_workers=config["num_workers"],
        remove_unused_columns=True,

        prediction_loss_only=False,
        include_inputs_for_metrics=False,
    )

    streaming_config = {
        "jsonl_file": train_file,
        "tokenizer": tokenizer,
        "max_length": config["max_length"],
        "max_examples": args.max_examples,
        "shuffle_buffer_size": config["shuffle_buffer_size"],
    }

    trainer = StreamingTrainer(
        model=model,
        args=training_args,
        train_dataset="placeholder",
        eval_dataset=val_dataset,
        data_collator=data_collator,
        compute_metrics=simple_compute_metrics_fn,
        streaming_dataset_config=streaming_config,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
        request_id=rid,
    )

    info_id("Starting adaptive streaming training...", rid)
    with log_timed_operation("trainer_train", rid):
        trainer.train()

    # === Export BEST checkpoint to ORC_PARTS_MODEL_DIR ===
    best_ckpt = trainer.state.best_model_checkpoint or training_args.output_dir
    info_id(f"Best checkpoint resolved to: {best_ckpt}", rid)

    with log_timed_operation("load_best_model", rid):
        best_model = DistilBertForTokenClassification.from_pretrained(best_ckpt)
        best_model.config.id2label = ID2LABEL
        best_model.config.label2id = LABEL2ID
        log_model_object_labels(best_model, "Best model (pre-save)", rid)

    dst = Path(ORC_PARTS_MODEL_DIR)
    with log_timed_operation("export_best_checkpoint", rid):
        shutil.rmtree(dst, ignore_errors=True)
        dst.mkdir(parents=True, exist_ok=True)
        best_model.save_pretrained(dst)
        tokenizer.save_pretrained(dst)
        (dst / "_BEST_CHECKPOINT.txt").write_text(f"Exported from: {best_ckpt}\n", encoding="utf-8")
        info_id(f"Exported best checkpoint to: {dst}", rid)

    # Log what landed in the folder
    log_model_folder_labels(dst, rid)

    info_id("Evaluating final (best) model...", rid)
    with log_timed_operation("trainer_evaluate", rid):
        val_metrics = trainer.evaluate()
    for k, v in val_metrics.items():
        info_id(f"Final {k}: {v}", rid)

    # Optional interactive sanity check
    run_demo(tokenizer, best_model, ID2LABEL)

    info_id("Adaptive streaming training complete!", rid)


if __name__ == "__main__":
    main()
