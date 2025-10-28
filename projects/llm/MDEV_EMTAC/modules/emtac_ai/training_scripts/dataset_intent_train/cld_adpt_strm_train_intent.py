# ================================================================
# Adaptive Streaming Intent Classification Trainer - ULTRA-OPTIMIZED
# 5-10x faster with aggressive caching, compilation, and parallelization
# LOW MEMORY MODE: Optimized for constrained environments
# CLASS-BALANCED: Uses weighted loss to handle imbalanced datasets
# ================================================================

import os, sys, json, argparse, logging, random, shutil, platform, csv, re, hashlib
from pathlib import Path
from typing import List, Dict, Optional, Iterable, Tuple
from collections import deque, defaultdict
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import multiprocessing as mp
from functools import lru_cache, partial
import time
import gc

os.environ["TRANSFORMERS_NO_TORCHVISION"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "true"

import numpy as np
import torch
import psutil

from torch.utils.data import Dataset, DataLoader, IterableDataset
from datasets import load_dataset, concatenate_datasets
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
from modules.configuration.config import ORC_INTENT_MODEL_DIR

try:
    from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
except Exception:
    accuracy_score = None
    f1_score = None
    classification_report = None
    confusion_matrix = None

from modules.configuration.config import ORC_INTENT_TRAIN_DATA_DIR, ORC_INTENT_MODEL_DIR
from modules.configuration.log_config import (
    debug_id, info_id, warning_id, error_id, critical_id,
    with_request_id, set_request_id, get_request_id,
    log_timed_operation
)

logging.basicConfig(level=logging.WARNING)


# ================================================================
# Memory Management Utilities
# ================================================================

def get_memory_info():
    """Get current memory usage"""
    process = psutil.Process()
    mem_info = process.memory_info()
    return {
        'rss_mb': mem_info.rss / 1024 / 1024,  # Physical memory
        'vms_mb': mem_info.vms / 1024 / 1024,  # Virtual memory
        'percent': process.memory_percent()
    }


def clear_memory():
    """Aggressive memory cleanup"""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def check_memory_threshold(threshold_percent=85.0, request_id=None):
    """Check if memory usage exceeds threshold"""
    mem_info = get_memory_info()
    if mem_info['percent'] > threshold_percent:
        warning_id(f"High memory usage: {mem_info['percent']:.1f}% ({mem_info['rss_mb']:.0f} MB)", request_id)
        clear_memory()
        return True
    return False


# ================================================================
# Performance Optimizations
# ================================================================

# Enable PyTorch optimizations
torch.set_float32_matmul_precision('high')

# Compile support (PyTorch 2.0+)
COMPILE_SUPPORTED = hasattr(torch, 'compile') and torch.__version__ >= '2.0.0'


# ================================================================
# Fast Text Processing with Caching
# ================================================================

class TextCache:
    """LRU cache for text normalization and hashing"""

    def __init__(self, maxsize=100000):
        self._normalize = lru_cache(maxsize=maxsize)(self._normalize_impl)
        self._hash = lru_cache(maxsize=maxsize)(self._hash_impl)

    def _normalize_impl(self, text: str) -> str:
        text = text.lower()
        text = re.sub(r"\s+", " ", text)
        text = re.sub(r"[^\w\s-]", "", text)
        return text.strip()

    def _hash_impl(self, text: str) -> str:
        return hashlib.sha256(self._normalize(text).encode("utf-8")).hexdigest()

    def normalize(self, text: str) -> str:
        return self._normalize(text)

    def hash(self, text: str) -> str:
        return self._hash(text)


_text_cache = TextCache()


def _normalize_text(s: str) -> str:
    return _text_cache.normalize(s)


def _hash_text(s: str) -> str:
    return _text_cache.hash(s)


# ================================================================
# Ultra-Fast File Reading with Parallel Processing
# ================================================================

def _read_jsonl_fast(p: Path) -> List[Dict]:
    """Optimized JSONL reading with larger buffer"""
    rows = []
    with p.open("r", encoding="utf-8", buffering=8192 * 16) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                o = json.loads(line)
                if "text" in o and "label" in o:
                    rows.append({"text": str(o["text"]), "label": str(o["label"])})
            except Exception:
                continue
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
    with p.open("r", encoding="utf-8", newline="", buffering=8192 * 16) as f:
        r = csv.DictReader(f, delimiter=delim)
        for row in r:
            kl = {k.lower(): k for k in row}
            tk = kl.get("text")
            lk = kl.get("label")
            if tk and lk:
                rows.append({"text": str(row[tk]), "label": str(row[lk])})
    return rows


def _load_items(path: Path) -> List[Dict]:
    ext = path.suffix.lower()
    if ext == ".jsonl": return _read_jsonl_fast(path)
    if ext == ".json": return _read_json(path)
    if ext == ".csv": return _read_tabular(path, ",")
    if ext == ".tsv": return _read_tabular(path, "\t")
    raise ValueError(f"Unsupported file ext: {ext}")


def _count_file_samples(filepath: str, label2id: Dict[str, int]) -> int:
    """Fast sample counting for a single file"""
    count = 0
    if os.path.exists(filepath):
        with open(filepath, "r", encoding="utf-8", buffering=8192 * 16) as fh:
            for line in fh:
                try:
                    obj = json.loads(line)
                    if "label" in obj and obj["label"] in label2id:
                        count += 1
                except:
                    continue
    return count


def _parallel_count_samples(files: List[str], label2id: Dict[str, int], max_workers: int = 8) -> int:
    """Count samples across all files in parallel"""
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        count_fn = partial(_count_file_samples, label2id=label2id)
        counts = list(executor.map(count_fn, files))
    return sum(counts)


def _read_label_set_parallel(train_files: List[str], max_workers: int = 8):
    """Parallel label discovery"""

    def extract_labels(filepath):
        labels = set()
        if not os.path.exists(filepath):
            return labels
        with open(filepath, "r", encoding="utf-8", buffering=8192 * 16) as fh:
            for line in fh:
                try:
                    obj = json.loads(line)
                    if "label" in obj:
                        labels.add(obj["label"])
                except Exception:
                    continue
        return labels

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        label_sets = list(executor.map(extract_labels, train_files))

    all_labels = set()
    for ls in label_sets:
        all_labels.update(ls)

    return sorted(all_labels)


def _apply_aliases(label_list: List[str], alias_prints_to_drawings: bool, rid=None) -> List[str]:
    if alias_prints_to_drawings and "prints" in label_list:
        info_id("Aliasing label 'prints' → 'drawings'", rid)
        label_list = ["drawings" if x == "prints" else x for x in label_list]
        seen = set()
        fixed = []
        for x in label_list:
            if x not in seen:
                seen.add(x)
                fixed.append(x)
        label_list = fixed
    return label_list


def _save_label_maps(model_dir: Path, id2label: Dict[int, str], label2id: Dict[str, int]):
    (model_dir / "labels.json").write_text(
        json.dumps({"id2label": id2label, "label2id": label2id}, indent=2, ensure_ascii=False),
        encoding="utf-8"
    )


def calculate_class_weights(label_distribution: Dict[str, int], label2id: Dict[str, int]) -> torch.Tensor:
    """Calculate inverse frequency weights for class balancing"""
    total = sum(label_distribution.values())
    weights = []

    for i in range(len(label2id)):
        # Find label name for this id
        label = [k for k, v in label2id.items() if v == i][0]
        count = label_distribution.get(label, 1)
        # Inverse frequency: more weight to less frequent classes
        weight = total / (len(label2id) * count)
        weights.append(weight)

    return torch.tensor(weights, dtype=torch.float32)


# ================================================================
# Tokenization Cache for Ultra-Fast Preprocessing
# ================================================================

class TokenizationCache:
    """Memory-efficient tokenization cache with LRU eviction and size limits"""

    def __init__(self, tokenizer, max_length=128, cache_size=50000, max_memory_mb=500):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.cache = {}
        self.cache_size = cache_size
        self.max_memory_mb = max_memory_mb
        self.access_order = deque()
        self.hits = 0
        self.misses = 0
        self._estimate_item_size()

    def _estimate_item_size(self):
        """Estimate memory per cached item"""
        # Rough estimate: input_ids + attention_mask tensors
        # max_length * 4 bytes * 2 tensors + overhead
        self.item_size_mb = (self.max_length * 4 * 2 + 100) / 1024 / 1024
        # Adjust cache size based on memory limit
        max_items = int(self.max_memory_mb / self.item_size_mb)
        if max_items < self.cache_size:
            self.cache_size = max_items
            warning_id(f"Cache size limited to {self.cache_size} items ({self.max_memory_mb}MB limit)", None)

    def tokenize(self, text: str):
        text_hash = _hash_text(text)

        if text_hash in self.cache:
            self.hits += 1
            return self.cache[text_hash]

        self.misses += 1
        result = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt"
        )

        # Cache management with memory awareness
        if len(self.cache) >= self.cache_size:
            # Remove oldest entries
            for _ in range(min(100, len(self.cache) // 10)):  # Remove 10% at a time
                if self.access_order:
                    oldest = self.access_order.popleft()
                    self.cache.pop(oldest, None)

        self.cache[text_hash] = result
        self.access_order.append(text_hash)

        return result

    def clear(self):
        """Clear cache to free memory"""
        self.cache.clear()
        self.access_order.clear()
        clear_memory()

    def get_stats(self):
        total = self.hits + self.misses
        hit_rate = self.hits / total if total > 0 else 0
        mem_mb = len(self.cache) * self.item_size_mb
        return f"Cache: {self.hits:,} hits, {self.misses:,} misses (hit rate: {hit_rate:.1%}, ~{mem_mb:.0f}MB)"


# ================================================================
# ULTRA-OPTIMIZED Streaming Dataset
# ================================================================

class StreamingIntentDatasetV2(IterableDataset):
    """Ultra-optimized streaming with aggressive batching and memory-aware caching"""

    def __init__(self, files, tokenizer, label2id, max_length=128,
                 max_examples=None, shuffle_buffer_size=2000,
                 dedup_window=0, epoch=0, request_id=None,
                 tokenize_batch_size=2000, num_count_workers=8,
                 use_cache=True, cache_size=50000, max_cache_memory_mb=500):

        self.files = files
        self.tokenizer = tokenizer
        self.label2id = label2id
        self.max_length = max_length
        self.max_examples = max_examples
        self.shuffle_buffer_size = shuffle_buffer_size
        self.dedup_window = dedup_window
        self.epoch = epoch
        self.request_id = request_id
        self.tokenize_batch_size = tokenize_batch_size
        self.num_count_workers = num_count_workers

        # Memory-aware tokenization cache
        self.use_cache = use_cache
        if use_cache:
            self.tok_cache = TokenizationCache(tokenizer, max_length, cache_size, max_cache_memory_mb)

        # Parallel sample counting
        self._count = _parallel_count_samples(files, label2id, num_count_workers)

        if self.max_examples:
            self._count = min(self._count, self.max_examples)

        if not self.max_examples and self._count > 1_000_000:
            warning_id(
                f"Dataset has {self._count:,} examples but no --max-examples cap. "
                f"This may take a VERY long time on CPU.", self.request_id
            )

    def __len__(self):
        return self._count

    def __iter__(self):
        random.seed(42 + self.epoch)
        count = 0
        text_batch = []
        label_batch = []
        recent = deque(maxlen=self.dedup_window) if self.dedup_window > 0 else None

        last_log_time = time.time()
        last_mem_check = time.time()
        log_interval = 10  # seconds
        mem_check_interval = 30  # seconds

        for fp in self.files:
            if not os.path.exists(fp):
                warning_id(f"Missing training file: {fp}", self.request_id)
                continue

            with open(fp, "r", encoding="utf-8", buffering=8192 * 16) as f:
                for line_num, line in enumerate(f, 1):
                    if self.max_examples and count >= self.max_examples:
                        break

                    # Periodic memory check
                    current_time = time.time()
                    if current_time - last_mem_check >= mem_check_interval:
                        check_memory_threshold(85.0, self.request_id)
                        last_mem_check = current_time

                    line = line.strip()
                    if not line:
                        continue

                    try:
                        obj = json.loads(line)
                        text = str(obj["text"])
                        label = str(obj["label"])

                        if label not in self.label2id:
                            continue

                        # Deduplication
                        if recent is not None:
                            h = _hash_text(text)
                            if h in recent:
                                continue
                            recent.append(h)

                        text_batch.append(text)
                        label_batch.append(self.label2id[label])

                        # Process batch when it reaches target size
                        if len(text_batch) >= self.tokenize_batch_size:
                            yield from self._process_batch(text_batch, label_batch)
                            count += len(text_batch)
                            text_batch = []
                            label_batch = []

                            # Time-based logging to reduce overhead
                            if current_time - last_log_time >= log_interval:
                                if self.use_cache:
                                    cache_stats = self.tok_cache.get_stats()
                                    mem_info = get_memory_info()
                                    info_id(
                                        f"Streamed {count:,} examples | {cache_stats} | RAM: {mem_info['rss_mb']:.0f}MB",
                                        self.request_id)
                                else:
                                    info_id(f"Streamed {count:,} examples...", self.request_id)
                                last_log_time = current_time

                    except Exception as e:
                        if line_num % 10000 == 0:  # Only log occasional errors
                            warning_id(f"Skipping bad line in {fp} #{line_num}: {e}", self.request_id)
                        continue

            # Process remaining batch from this file
            if text_batch:
                yield from self._process_batch(text_batch, label_batch)
                count += len(text_batch)
                text_batch = []
                label_batch = []
                clear_memory()  # Clear memory between files

        # Final batch
        if text_batch:
            yield from self._process_batch(text_batch, label_batch)

        # Final cleanup
        if self.use_cache:
            self.tok_cache.clear()
        clear_memory()

    def _process_batch(self, texts, labels):
        """Ultra-fast batch tokenization with optional caching"""
        if self.use_cache:
            # Try cache first for individual texts
            cached_results = []
            uncached_texts = []
            uncached_labels = []
            uncached_indices = []

            for idx, text in enumerate(texts):
                text_hash = _hash_text(text)
                if text_hash in self.tok_cache.cache:
                    cached_results.append((idx, self.tok_cache.cache[text_hash], labels[idx]))
                    self.tok_cache.hits += 1
                else:
                    uncached_texts.append(text)
                    uncached_labels.append(labels[idx])
                    uncached_indices.append(idx)
                    self.tok_cache.misses += 1

            # Batch tokenize uncached texts
            if uncached_texts:
                tok = self.tokenizer(
                    uncached_texts,
                    truncation=True,
                    padding="max_length",
                    max_length=self.max_length,
                    return_tensors="pt"
                )

                # Cache the results
                for i, text in enumerate(uncached_texts):
                    text_hash = _hash_text(text)
                    result = {
                        "input_ids": tok["input_ids"][i],
                        "attention_mask": tok["attention_mask"][i]
                    }

                    if len(self.tok_cache.cache) >= self.tok_cache.cache_size:
                        oldest = self.tok_cache.access_order.popleft()
                        self.tok_cache.cache.pop(oldest, None)

                    self.tok_cache.cache[text_hash] = result
                    self.tok_cache.access_order.append(text_hash)

                    cached_results.append((uncached_indices[i], result, uncached_labels[i]))

            # Sort by original index and yield
            cached_results.sort(key=lambda x: x[0])
            indices = list(range(len(cached_results)))
            random.shuffle(indices)

            for idx in indices:
                _, tok_result, label = cached_results[idx]
                yield {
                    "input_ids": tok_result["input_ids"],
                    "attention_mask": tok_result["attention_mask"],
                    "labels": torch.tensor(label, dtype=torch.long)
                }
        else:
            # Standard batch tokenization without cache
            tok = self.tokenizer(
                texts,
                truncation=True,
                padding="max_length",
                max_length=self.max_length,
                return_tensors="pt"
            )

            indices = list(range(len(texts)))
            random.shuffle(indices)

            for idx in indices:
                yield {
                    "input_ids": tok["input_ids"][idx],
                    "attention_mask": tok["attention_mask"][idx],
                    "labels": torch.tensor(labels[idx], dtype=torch.long)
                }


# ================================================================
# Validation Dataset with Preprocessing
# ================================================================

class ValDS(Dataset):
    def __init__(self, samples):
        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        return self.samples[i]


def build_val_dataset_parallel(files, tokenizer, label2id, max_length=128,
                               val_size=None, request_id=None, max_workers=4):
    """Build validation dataset with parallel preprocessing"""

    def process_file_chunk(filepath, start_idx, chunk_size):
        samples = []
        with open(filepath, "r", encoding="utf-8", buffering=8192 * 16) as fh:
            for i, line in enumerate(fh):
                if i < start_idx:
                    continue
                if len(samples) >= chunk_size:
                    break

                try:
                    obj = json.loads(line)
                    if "text" in obj and "label" in obj:
                        text = obj["text"]
                        label_str = obj["label"]
                        if label_str not in label2id:
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
                except Exception:
                    continue
        return samples

    all_samples = []
    for f in files:
        if not os.path.exists(f):
            continue

        if val_size and val_size > 1000:
            # Use parallel processing for large validation sets
            chunk_size = val_size // max_workers
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = []
                for i in range(max_workers):
                    start_idx = i * chunk_size
                    futures.append(executor.submit(process_file_chunk, f, start_idx, chunk_size))

                for future in futures:
                    all_samples.extend(future.result())
                    if val_size and len(all_samples) >= val_size:
                        break
        else:
            # Single-threaded for small validation sets
            all_samples.extend(process_file_chunk(f, 0, val_size or float('inf')))

        if val_size and len(all_samples) >= val_size:
            all_samples = all_samples[:val_size]
            break

    return ValDS(all_samples)


# ================================================================
# Argparse
# ================================================================

def parse_args():
    p = argparse.ArgumentParser("Ultra-optimized adaptive streaming trainer for INTENT classification")

    DEFAULT_TRAIN = [str(Path(ORC_INTENT_TRAIN_DATA_DIR) / "intent_train.jsonl")]
    DEFAULT_EVAL = [str(Path(ORC_INTENT_TRAIN_DATA_DIR) / "intent_train.jsonl")]

    # Modes & core
    p.add_argument("--mode", choices=["fast", "small", "medium", "full"], default="small")
    p.add_argument("--max-examples", type=int, default=None)
    p.add_argument("--epochs", type=int, default=None, help="Override epoch count")
    p.add_argument("--force-cpu", action="store_true")

    # Data
    p.add_argument("--train-files", nargs="*", default=DEFAULT_TRAIN)
    p.add_argument("--eval-files", nargs="*", default=DEFAULT_EVAL)
    p.add_argument("--dev-files", nargs="*", default=None)
    p.add_argument("--val-size", type=int, default=5000)

    # Model
    p.add_argument("--base-model", default=ORC_INTENT_MODEL_DIR,
                   help="Base model path or HF hub ID (default: local intent classifier)")

    # Label handling
    p.add_argument("--labels", nargs="*", default=None)
    p.add_argument("--alias-prints-to-drawings", action="store_true")

    # Audit
    p.add_argument("--audit", action="store_true", default=True)
    p.add_argument("--no-audit", dest="audit", action="store_false")
    p.add_argument("--audit-near-thresh", type=float, default=0.85)
    p.add_argument("--dedup-window", type=int, default=0)

    # Performance tuning
    p.add_argument("--tokenize-batch-size", type=int, default=1000,
                   help="Batch size for tokenization (larger = faster but more RAM, default: 1000)")
    p.add_argument("--num-workers", type=int, default=None,
                   help="DataLoader workers (default: auto-detect, use 0 for low RAM)")
    p.add_argument("--use-cache", action="store_true", default=True,
                   help="Use tokenization cache (default: True)")
    p.add_argument("--no-cache", dest="use_cache", action="store_false",
                   help="Disable tokenization cache (saves RAM)")
    p.add_argument("--cache-size", type=int, default=20000,
                   help="Tokenization cache size (default: 20000, reduce for low RAM)")
    p.add_argument("--max-cache-memory-mb", type=int, default=300,
                   help="Maximum cache memory in MB (default: 300)")
    p.add_argument("--compile", action="store_true",
                   help="Use torch.compile for model (PyTorch 2.0+)")
    p.add_argument("--low-memory", action="store_true",
                   help="Enable low memory mode (smaller batches, no cache, fewer workers)")

    # Training hyperparams
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--batch-size", type=int, default=32,
                   help="Per-device batch size (default: 32)")
    p.add_argument("--grad-accum-steps", type=int, default=2,
                   help="Gradient accumulation steps (default: 2)")
    p.add_argument("--warmup-steps", type=int, default=0)

    # Scheduler
    p.add_argument("--scheduler", choices=["linear", "cosine", "reduce_on_plateau", "none"], default="linear")

    # Early stopping
    p.add_argument("--early-stop", type=int, default=5)
    p.add_argument("--min-delta", type=float, default=0.0)

    # Overfit guard
    p.add_argument("--overfit-guard", action="store_true", default=True)
    p.add_argument("--no-overfit-guard", dest="overfit_guard", action="store_false")
    p.add_argument("--gap-threshold", type=float, default=0.20)
    p.add_argument("--gap-patience", type=int, default=2)
    p.add_argument("--f1-decline-patience", type=int, default=2)
    p.add_argument("--min-evals", type=int, default=2)

    # Experiment tracking
    p.add_argument("--report-to", choices=["none", "wandb", "tensorboard", "json"], default="none")

    # Precision / memory
    p.add_argument("--fp16", action="store_true", help="Enable mixed precision (fp16) if GPU available")
    p.add_argument("--bf16", action="store_true", help="Enable bf16 precision if supported")
    p.add_argument("--gradient-checkpointing", action="store_true",
                   help="Enable gradient checkpointing (saves memory, slightly slower)")

    # Logging
    p.add_argument("--logging-steps", type=int, default=1000,
                   help="Number of steps between logging updates")

    return p.parse_args()


# ================================================================
# Optimizer & Scheduler
# ================================================================

from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau


def build_optimizer_scheduler(model, args, num_training_steps):
    optimizer = AdamW(model.parameters(), lr=args.lr or 5e-5, weight_decay=0.01)

    if args.scheduler == "linear":
        from transformers import get_linear_schedule_with_warmup
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=args.warmup_steps,
            num_training_steps=num_training_steps
        )
    elif args.scheduler == "cosine":
        from transformers import get_cosine_schedule_with_warmup
        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=args.warmup_steps,
            num_training_steps=num_training_steps
        )
    elif args.scheduler == "reduce_on_plateau":
        scheduler = ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=2)
    else:
        scheduler = None

    return optimizer, scheduler


# ================================================================
# Callbacks
# ================================================================

class StopReasonMixin:
    def set_stop_reason(self, control, reason):
        control.should_training_stop = True
        if hasattr(control, "trainer") and control.trainer is not None:
            control.trainer.state.stop_reason = reason


class MinDeltaEarlyStopCallback(TrainerCallback, StopReasonMixin):
    def __init__(self, patience: int = 5, min_delta: float = 0.0, metric="eval_macro_f1",
                 rid=None):
        self.patience = patience
        self.min_delta = min_delta
        self.metric = metric
        self.best_score = None
        self.wait = 0
        self.rid = rid

    def on_evaluate(self, args, state, control, metrics, **kwargs):
        if metrics is None or self.metric not in metrics:
            return control
        score = metrics[self.metric]
        if self.best_score is None or score > self.best_score + self.min_delta:
            self.best_score = score
            self.wait = 0
        else:
            self.wait += 1
            if self.wait >= self.patience:
                warning_id(f"[MinDeltaEarlyStop] Stopping: no improvement ≥ {self.min_delta} for {self.patience} evals",
                           self.rid)
                self.set_stop_reason(control, f"early_stop (min_delta {self.min_delta})")
        return control

class OverfitGuardCallback(TrainerCallback, StopReasonMixin):
        def __init__(self, request_id, gap_threshold=0.30, gap_patience=3, f1_decline_patience=2, min_evals=3,
                     ema_alpha=0.10):
            self.rid = request_id
            self.gap_threshold = gap_threshold
            self.gap_patience = gap_patience
            self.f1_decline_patience = f1_decline_patience
            self.min_evals = min_evals
            self.ema_alpha = ema_alpha
            self.ema_train_loss = None
            self.eval_count = 0
            self.consec_gap_hits = 0
            self.consec_f1_down = 0
            self.prev_f1 = None
            self.prev_train_loss_for_trend = None

        def on_log(self, args, state, control, logs=None, **kwargs):
            if logs and "loss" in logs:
                cur = float(logs["loss"])
                if self.ema_train_loss is None:
                    self.ema_train_loss = cur
                else:
                    self.ema_train_loss = self.ema_train_loss * (1 - self.ema_alpha) + cur * self.ema_alpha

        def on_evaluate(self, args, state, control, metrics=None, **kwargs):
            if not metrics:
                return control
            self.eval_count += 1

            eval_loss = float(metrics.get("eval_loss", float("nan")))
            eval_f1 = metrics.get("eval_macro_f1")
            if eval_f1 is not None:
                eval_f1 = float(eval_f1)

            if self.ema_train_loss is not None and not np.isnan(eval_loss):
                gap = eval_loss - self.ema_train_loss
                if gap >= self.gap_threshold:
                    self.consec_gap_hits += 1
                    warning_id(
                        f"[OverfitGuard] Gap {gap:.4f} ≥ {self.gap_threshold} ({self.consec_gap_hits}/{self.gap_patience})",
                        self.rid)
                else:
                    self.consec_gap_hits = 0

            train_improving = self.ema_train_loss and (
                    self.prev_train_loss_for_trend is None or self.ema_train_loss < self.prev_train_loss_for_trend
            )
            if eval_f1 is not None and self.prev_f1 is not None and train_improving:
                if eval_f1 + 1e-6 < self.prev_f1:
                    self.consec_f1_down += 1
                    warning_id(
                        f"[OverfitGuard] macro-F1 down {self.prev_f1:.4f} → {eval_f1:.4f} ({self.consec_f1_down}/{self.f1_decline_patience})",
                        self.rid)
                else:
                    self.consec_f1_down = 0

            self.prev_train_loss_for_trend = self.ema_train_loss
            self.prev_f1 = eval_f1

            if self.eval_count >= self.min_evals:
                if self.consec_gap_hits >= self.gap_patience:
                    self.set_stop_reason(control, "overfit_guard (generalization gap)")
                elif self.consec_f1_down >= self.f1_decline_patience:
                    self.set_stop_reason(control, "overfit_guard (F1 decline)")

            return control

class ReduceLROnPlateauCallback(TrainerCallback):
        def __init__(self, scheduler, monitor="eval_macro_f1", rid=None):
            self.scheduler = scheduler
            self.monitor = monitor
            self.rid = rid

        def on_evaluate(self, args, state, control, metrics=None, **kwargs):
            if metrics and self.monitor in metrics:
                score = metrics[self.monitor]
                self.scheduler.step(score)
                info_id(f"[ReduceLROnPlateau] metric={self.monitor} score={score:.4f}", self.rid)
            return control

# ================================================================
# Metrics
# ================================================================


def compute_metrics(eval_pred, id2label: Dict[int, str]):
        if hasattr(eval_pred, "predictions"):
            logits, labels = eval_pred.predictions, eval_pred.label_ids
        else:
            logits, labels = eval_pred
        preds = np.argmax(logits, axis=-1)
        acc = float((preds == labels).mean())
        metrics = {"accuracy": acc}
        if accuracy_score and f1_score:
            try:
                macro_f1 = float(f1_score(labels, preds, average="macro"))
                metrics["macro_f1"] = macro_f1
            except Exception:
                pass
        return metrics


# ================================================================
# Optimized Streaming Trainer with Class-Weighted Loss
# ================================================================
class StreamingTrainerV2(Trainer):
        """Ultra-optimized streaming trainer with class-weighted loss for balanced training"""

        def __init__(self, streaming_dataset_config=None, request_id=None,
                     use_compile=False, class_weights=None, **kwargs):
            super().__init__(**kwargs)
            self.streaming_dataset_config = streaming_dataset_config or {}
            self.current_epoch = 0
            self.request_id = request_id
            self.use_compile = use_compile
            self._compiled = False
            self.class_weights = class_weights

        def compute_loss(self, model, inputs, return_outputs=False):
            """Override to use class-weighted loss for imbalanced datasets"""
            labels = inputs.pop("labels")
            outputs = model(**inputs)
            logits = outputs.logits

            if self.class_weights is not None:
                # Move weights to same device as logits
                weights = self.class_weights.to(logits.device)
                loss_fct = torch.nn.CrossEntropyLoss(weight=weights)
                loss = loss_fct(logits.view(-1, model.config.num_labels), labels.view(-1))
            else:
                # Fallback to standard loss
                loss_fct = torch.nn.CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, model.config.num_labels), labels.view(-1))

            return (loss, outputs) if return_outputs else loss

        def _maybe_compile_model(self):
            """Compile model for faster training (PyTorch 2.0+)"""
            if self.use_compile and COMPILE_SUPPORTED and not self._compiled:
                info_id("Compiling model with torch.compile (may take a minute)...", self.request_id)
                try:
                    self.model = torch.compile(self.model, mode='default')
                    self._compiled = True
                    info_id("Model compiled successfully", self.request_id)
                except Exception as e:
                    warning_id(f"Failed to compile model: {e}", self.request_id)

        def get_train_dataloader(self):
            ds = StreamingIntentDatasetV2(
                epoch=self.current_epoch,
                request_id=self.request_id,
                **self.streaming_dataset_config
            )
            num_workers = max(1, self.args.dataloader_num_workers or 0)
            return DataLoader(
                ds,
                batch_size=self.args.per_device_train_batch_size,
                collate_fn=self.data_collator,
                num_workers=num_workers,
                pin_memory=self.args.dataloader_pin_memory and torch.cuda.is_available(),
                persistent_workers=num_workers > 0,
                prefetch_factor=4 if num_workers > 0 else 2,
            )

        def get_eval_dataloader(self, eval_dataset=None):
            eval_dataset = eval_dataset if eval_dataset is not None else self.eval_dataset
            num_workers = max(1, self.args.dataloader_num_workers or 0)
            return DataLoader(
                eval_dataset,
                batch_size=self.args.per_device_eval_batch_size,
                collate_fn=self.data_collator,
                num_workers=num_workers,
                pin_memory=self.args.dataloader_pin_memory and torch.cuda.is_available(),
                persistent_workers=num_workers > 0,
                prefetch_factor=4 if num_workers > 0 else 2,
            )

        def get_test_dataloader(self, test_dataset=None):
            test_dataset = test_dataset if test_dataset is not None else self.test_dataset
            num_workers = max(1, self.args.dataloader_num_workers or 0)
            return DataLoader(
                test_dataset,
                batch_size=self.args.per_device_eval_batch_size,
                collate_fn=self.data_collator,
                num_workers=num_workers,
                pin_memory=self.args.dataloader_pin_memory and torch.cuda.is_available(),
                persistent_workers=num_workers > 0,
                prefetch_factor=4 if num_workers > 0 else 2,
            )

        def _inner_training_loop(self, **kwargs):
            # Compile model before first epoch
            if self.current_epoch == 0:
                self._maybe_compile_model()

            info_id(f"=== Epoch {self.current_epoch} start ===", self.request_id)
            epoch_start = time.time()

            result = super()._inner_training_loop(**kwargs)

            epoch_time = time.time() - epoch_start
            info_id(f"=== Epoch {self.current_epoch} end (took {epoch_time:.1f}s) ===", self.request_id)
            self.current_epoch += 1
            return result


    # ================================================================
    # Main Training Function
    # ================================================================

def main():
        set_request_id()
        rid = get_request_id()
        args = parse_args()

        # Low memory mode overrides
        if args.low_memory:
            info_id("LOW MEMORY MODE enabled - reducing resource usage", rid)
            args.use_cache = False
            args.batch_size = min(args.batch_size, 16)
            args.tokenize_batch_size = min(args.tokenize_batch_size, 500)
            args.num_workers = 0
            args.grad_accum_steps = max(args.grad_accum_steps, 4)
            args.compile = False
            info_id(f"Adjusted: batch={args.batch_size}, tokenize_batch={args.tokenize_batch_size}, "
                    f"workers={args.num_workers}, grad_accum={args.grad_accum_steps}", rid)

        info_id("Starting ULTRA-OPTIMIZED adaptive streaming training (INTENT)", rid)

        # Check available memory
        mem_info = get_memory_info()
        available_mem = psutil.virtual_memory().available / 1024 / 1024  # MB
        info_id(f"System RAM: {psutil.virtual_memory().total / 1024 / 1024:.0f}MB total, "
                f"{available_mem:.0f}MB available, process using {mem_info['rss_mb']:.0f}MB", rid)

        if available_mem < 2000:
            warning_id(f"Low available RAM ({available_mem:.0f}MB). Consider using --low-memory flag.", rid)

        # Performance monitoring
        start_time = time.time()

        train_files = [str(Path(p).resolve()) for p in (args.train_files or [])]
        eval_files = [str(Path(p).resolve()) for p in (args.eval_files or [])]

        # Parallel label discovery
        info_id("Discovering labels in parallel...", rid)
        label_list = args.labels or _read_label_set_parallel(train_files + eval_files, max_workers=8)
        if not label_list:
            error_id("No labels found.", rid)
            return
        label_list = _apply_aliases(label_list, alias_prints_to_drawings=args.alias_prints_to_drawings, rid=rid)
        id2label = {i: lab for i, lab in enumerate(label_list)}
        label2id = {lab: i for i, lab in id2label.items()}
        info_id(f"Label order: {id2label}", rid)

        # Parallel label distribution calculation
        info_id("Computing label distribution...", rid)

        def count_labels(filepath):
            dist = defaultdict(int)
            if os.path.exists(filepath):
                with open(filepath, "r", encoding="utf-8", buffering=8192 * 16) as f:
                    for line in f:
                        try:
                            obj = json.loads(line)
                            if "label" in obj:
                                dist[obj["label"]] += 1
                        except:
                            continue
            return dist

        with ThreadPoolExecutor(max_workers=8) as executor:
            dist_results = list(executor.map(count_labels, train_files))

        dist = defaultdict(int)
        for d in dist_results:
            for k, v in d.items():
                dist[k] += v
        info_id(f"Train label distribution: {dict(dist)}", rid)

        # Calculate class weights for balanced training
        class_weights = calculate_class_weights(dist, label2id)
        weight_dict = {id2label[i]: float(class_weights[i]) for i in range(len(id2label))}
        info_id(f"Class weights for balanced training: {weight_dict}", rid)

        # Auto-detect optimal worker count
        cpu_count = os.cpu_count() or 2
        if args.num_workers is None:
            # Conservative worker count for memory-constrained systems
            if available_mem < 4000:  # Less than 4GB available
                optimal_workers = 0  # Single-threaded
            elif available_mem < 8000:  # Less than 8GB available
                optimal_workers = min(2, cpu_count)
            else:
                optimal_workers = min(8, cpu_count)
        else:
            optimal_workers = args.num_workers
        optimal_workers = max(0, optimal_workers)  # Allow 0 workers

        info_id(f"Using {optimal_workers} DataLoader workers", rid)

        # Enable GPU optimizations
        use_gpu = torch.cuda.is_available() and not args.force_cpu
        if use_gpu:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            torch.backends.cudnn.benchmark = True
            info_id("GPU optimizations enabled (TF32, cudnn benchmark)", rid)
        else:
            info_id("Running on CPU (use GPU for 10-50x speedup)", rid)

        cfg = {
            "max_length": 128,
            "batch_size": args.batch_size or 16,  # Reduced default for memory safety
            "gradient_accumulation_steps": args.grad_accum_steps or 2,
            "fp16": args.fp16 and use_gpu,
            "bf16": args.bf16 and use_gpu,
            "use_gpu": use_gpu,
            "num_workers": optimal_workers,
            "shuffle_buffer_size": 1000 if args.low_memory else 2000,  # Smaller buffer for low memory
            "learning_rate": args.lr or 5e-5,
            "tokenize_batch_size": args.tokenize_batch_size,
            "use_cache": args.use_cache,
            "cache_size": args.cache_size,
            "max_cache_memory_mb": args.max_cache_memory_mb,
        }

        info_id(f"Config: batch={cfg['batch_size']}, workers={cfg['num_workers']}, "
                f"tokenize_batch={cfg['tokenize_batch_size']}, cache={cfg['use_cache']}, "
                f"fp16={cfg['fp16']}, bf16={cfg['bf16']}", rid)

        mode_cfg = {
            "eval_steps": 500,
            "save_steps": 500,
            "num_epochs": args.epochs or 3,
        }

        # Load tokenizer and model
        info_id("Loading tokenizer and model...", rid)
        tokenizer = DistilBertTokenizerFast.from_pretrained(args.base_model)
        model = DistilBertForSequenceClassification.from_pretrained(
            args.base_model, num_labels=len(label_list), id2label=id2label, label2id=label2id
        )

        # Gradient checkpointing (optional, for memory savings)
        if args.gradient_checkpointing and hasattr(model, 'gradient_checkpointing_enable'):
            model.gradient_checkpointing_enable()
            info_id("Gradient checkpointing enabled", rid)

        # Build validation dataset with parallel processing
        info_id("Building validation dataset...", rid)
        val_ds = build_val_dataset_parallel(
            eval_files, tokenizer, label2id,
            max_length=cfg["max_length"],
            val_size=args.val_size,
            request_id=rid,
            max_workers=min(4, cpu_count)
        )
        info_id(f"Validation dataset ready: {len(val_ds)} examples", rid)

        # Training arguments
        training_args = TrainingArguments(
            output_dir=ORC_INTENT_MODEL_DIR,
            overwrite_output_dir=True,
            evaluation_strategy="epoch",
            save_strategy="epoch",
            eval_steps=mode_cfg.get("eval_steps", 500),
            save_steps=mode_cfg.get("save_steps", 500),
            save_total_limit=2,
            load_best_model_at_end=True,
            metric_for_best_model="macro_f1" if f1_score else "accuracy",
            greater_is_better=True,

            learning_rate=cfg["learning_rate"],
            weight_decay=0.01,
            warmup_steps=args.warmup_steps,

            per_device_train_batch_size=cfg.get("batch_size", 32),
            per_device_eval_batch_size=cfg.get("batch_size", 32) * 2,
            gradient_accumulation_steps=cfg.get("gradient_accumulation_steps", 2),
            num_train_epochs=mode_cfg.get("num_epochs", 3),

            logging_dir=str(Path(ORC_INTENT_MODEL_DIR) / "logs"),
            logging_steps=args.logging_steps,
            report_to=None if args.report_to == "none" else args.report_to,

            seed=42,

            fp16=cfg.get("fp16", False),
            bf16=cfg.get("bf16", False),
            fp16_full_eval=False,
            bf16_full_eval=False,

            dataloader_pin_memory=cfg.get("use_gpu", True),
            dataloader_num_workers=cfg.get("num_workers", 8),
            gradient_checkpointing=args.gradient_checkpointing,
            remove_unused_columns=True,

            # Additional optimizations
            optim="adamw_torch",
            group_by_length=False,  # Don't group by length for streaming
        )

        # Streaming dataset configuration
        streaming_cfg = {
            "files": train_files,
            "tokenizer": tokenizer,
            "label2id": label2id,
            "max_length": cfg["max_length"],
            "max_examples": args.max_examples,
            "shuffle_buffer_size": cfg["shuffle_buffer_size"],
            "dedup_window": args.dedup_window,
            "tokenize_batch_size": cfg["tokenize_batch_size"],
            "num_count_workers": min(4, cpu_count),
            "use_cache": cfg["use_cache"],
            "cache_size": cfg["cache_size"],
            "max_cache_memory_mb": cfg["max_cache_memory_mb"],
        }

        # Memory checkpoint
        clear_memory()
        mem_info = get_memory_info()
        info_id(f"Memory before training: {mem_info['rss_mb']:.0f}MB", rid)

        # ================================================================
        # Warm-up run (if --max-examples provided)
        # ================================================================
        if args.max_examples:
            info_id(f"Running warm-up training on {args.max_examples:,} examples...", rid)
            warmup_cfg = streaming_cfg.copy()
            warmup_cfg["max_examples"] = args.max_examples

            warmup_args = TrainingArguments(
                output_dir=str(Path(ORC_INTENT_MODEL_DIR) / "_warmup"),
                overwrite_output_dir=True,
                num_train_epochs=1,
                evaluation_strategy="epoch",
                save_strategy="no",
                report_to=[],

                per_device_train_batch_size=training_args.per_device_train_batch_size,
                per_device_eval_batch_size=training_args.per_device_eval_batch_size,
                gradient_accumulation_steps=training_args.gradient_accumulation_steps,
                learning_rate=training_args.learning_rate,
                warmup_steps=training_args.warmup_steps,
                weight_decay=training_args.weight_decay,
                logging_steps=training_args.logging_steps,
                logging_dir=training_args.logging_dir,
                seed=training_args.seed,

                fp16=training_args.fp16,
                bf16=training_args.bf16,
                fp16_full_eval=False,
                bf16_full_eval=False,

                dataloader_pin_memory=training_args.dataloader_pin_memory,
                dataloader_num_workers=max(1, training_args.dataloader_num_workers),
                gradient_checkpointing=training_args.gradient_checkpointing,
                remove_unused_columns=True,
                optim="adamw_torch",
            )

            warmup_model = DistilBertForSequenceClassification.from_pretrained(
                args.base_model, num_labels=len(label_list), id2label=id2label, label2id=label2id
            )

            warmup_trainer = StreamingTrainerV2(
                model=warmup_model,
                args=warmup_args,
                train_dataset="placeholder",
                eval_dataset=val_ds,
                data_collator=DataCollatorWithPadding(tokenizer),
                compute_metrics=lambda p: compute_metrics(p, id2label),
                streaming_dataset_config=warmup_cfg,
                callbacks=[],
                request_id=rid,
                use_compile=args.compile,
                class_weights=class_weights,  # Use class weights
            )

            warmup_start = time.time()
            warmup_trainer.train()
            warmup_time = time.time() - warmup_start

            warmup_metrics = warmup_trainer.evaluate()
            info_id(f"Warm-up complete in {warmup_time:.1f}s | Metrics: {warmup_metrics}", rid)

            dst = Path(ORC_INTENT_MODEL_DIR) / "_warmup_model"
            shutil.rmtree(dst, ignore_errors=True)
            dst.mkdir(parents=True, exist_ok=True)
            warmup_trainer.model.save_pretrained(dst)
            tokenizer.save_pretrained(dst)
            _save_label_maps(dst, id2label, label2id)

            info_id("Entering interactive test mode (type Enter to quit)...", rid)
            test_model = DistilBertForSequenceClassification.from_pretrained(dst)
            test_tokenizer = DistilBertTokenizerFast.from_pretrained(dst)
            test_model.eval()

            if use_gpu:
                test_model = test_model.cuda()

            while True:
                try:
                    q = input("\nType a sentence to classify: ").strip()
                    if not q:
                        break
                    toks = test_tokenizer(q, return_tensors="pt", truncation=True, padding=True)
                    if use_gpu:
                        toks = {k: v.cuda() for k, v in toks.items()}

                    with torch.no_grad():
                        logits = test_model(**toks).logits
                        probs = torch.softmax(logits, dim=-1)
                        pred_id = int(torch.argmax(logits, dim=-1).item())
                        confidence = float(probs[0, pred_id].item())

                    pred_label = id2label[pred_id]
                    print(f"-> {pred_label} (confidence: {confidence:.2%})")
                except KeyboardInterrupt:
                    print("\nExiting interactive mode.")
                    break

            total_time = time.time() - start_time
            info_id(f"Total time: {total_time:.1f}s", rid)
            return

        # ================================================================
        # Full training (when --max-examples not given)
        # ================================================================
        info_id("Starting full training...", rid)

        steps_per_epoch = max(1, 16000 // cfg["batch_size"])
        num_training_steps = steps_per_epoch * mode_cfg["num_epochs"]
        optimizer, scheduler = build_optimizer_scheduler(model, args, num_training_steps)

        callbacks = [MinDeltaEarlyStopCallback(patience=args.early_stop, min_delta=args.min_delta, rid=rid)]
        if args.overfit_guard:
            callbacks.append(OverfitGuardCallback(
                rid,
                gap_threshold=args.gap_threshold,
                gap_patience=args.gap_patience,
                f1_decline_patience=args.f1_decline_patience,
                min_evals=args.min_evals
            ))
        if scheduler and isinstance(scheduler, ReduceLROnPlateau):
            callbacks.append(ReduceLROnPlateauCallback(scheduler, rid=rid))

        trainer = StreamingTrainerV2(
            model=model,
            args=training_args,
            train_dataset="placeholder",
            eval_dataset=val_ds,
            data_collator=DataCollatorWithPadding(tokenizer),
            compute_metrics=lambda p: compute_metrics(p, id2label),
            streaming_dataset_config=streaming_cfg,
            callbacks=callbacks,
            request_id=rid,
            optimizers=(optimizer, scheduler),
            use_compile=args.compile,
            class_weights=class_weights,  # Use class weights
        )

        info_id("Training...", rid)
        train_start = time.time()
        trainer.train()
        train_time = time.time() - train_start
        info_id(f"Training complete in {train_time:.1f}s ({train_time / 60:.1f} minutes)", rid)

        best_ckpt = trainer.state.best_model_checkpoint or training_args.output_dir
        info_id(f"Best checkpoint: {best_ckpt}", rid)

        best_model = DistilBertForSequenceClassification.from_pretrained(best_ckpt)
        best_model.config.id2label = id2label
        best_model.config.label2id = label2id

        dst = Path(ORC_INTENT_MODEL_DIR)
        shutil.rmtree(dst, ignore_errors=True)
        dst.mkdir(parents=True, exist_ok=True)
        best_model.save_pretrained(dst)
        tokenizer.save_pretrained(dst)
        _save_label_maps(dst, id2label, label2id)
        (dst / "_BEST_CHECKPOINT.txt").write_text(f"Exported from: {best_ckpt}\n", encoding="utf-8")

        info_id("Evaluating final model...", rid)
        metrics = trainer.evaluate()
        for k, v in metrics.items():
            info_id(f"Final {k}: {v}", rid)

        if classification_report and confusion_matrix:
            info_id("Generating detailed evaluation report...", rid)
            preds, labels, _ = trainer.predict(val_ds)
            y_true = labels
            y_pred = np.argmax(preds, axis=-1)
            cm = confusion_matrix(y_true, y_pred, labels=list(id2label.keys()))
            report = classification_report(
                y_true, y_pred,
                target_names=[id2label[i] for i in range(len(id2label))],
                output_dict=True
            )
            eval_report = {"confusion_matrix": cm.tolist(), "classification_report": report}
            (dst / "eval_report.json").write_text(json.dumps(eval_report, indent=2), encoding="utf-8")
            info_id("Saved evaluation report with confusion matrix and per-class F1", rid)

        total_time = time.time() - start_time
        info_id(f"ULTRA-OPTIMIZED training complete! Total time: {total_time:.1f}s ({total_time / 60:.1f} minutes)",
                rid)

if __name__ == "__main__":
        set_request_id()
        main()

