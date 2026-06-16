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
from modules.emtac_ai.config import ORC_DRAWINGS_TRAIN_DATA_DIR, ORC_DRAWINGS_MODEL_DIR
from modules.configuration.log_config import debug_id, info_id, error_id, with_request_id

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ---- Single source of truth for NER labels ----
LABELS = [
    "O",
    "B-EQUIPMENT_NUMBER", "I-EQUIPMENT_NUMBER",
    "B-EQUIPMENT_NAME", "I-EQUIPMENT_NAME",
    "B-DRAWING_NUMBER", "I-DRAWING_NUMBER",
    "B-DRAWING_NAME", "I-DRAWING_NAME",
    "B-SPARE_PART_NUMBER", "I-SPARE_PART_NUMBER",
]
ID2LABEL = {i: lab for i, lab in enumerate(LABELS)}
LABEL2ID = {lab: i for i, lab in enumerate(LABELS)}



class SystemOptimizer:
    """Automatically detect system capabilities and optimize training parameters"""

    def __init__(self):
        self.system_info = self._get_system_info()
        self.gpu_info = self._get_gpu_info()
        self.optimal_config = self._calculate_optimal_config()

    def _get_system_info(self):
        """Get system specifications"""
        try:
            memory_gb = psutil.virtual_memory().total / (1024 ** 3)
            cpu_count = psutil.cpu_count(logical=True)
            cpu_count_physical = psutil.cpu_count(logical=False)

            info = {
                "memory_gb": memory_gb,
                "cpu_count": cpu_count,
                "cpu_count_physical": cpu_count_physical,
                "platform": platform.system(),
                "processor": platform.processor(),
                "python_version": platform.python_version(),
            }

            info_id(f"System detected: {memory_gb:.1f}GB RAM, {cpu_count} logical CPUs ({cpu_count_physical} physical)")
            return info

        except Exception as e:
            logger.warning(f"Could not detect system specs: {e}")
            return {"memory_gb": 8, "cpu_count": 4, "cpu_count_physical": 2}

    def _get_gpu_info(self):
        """Detect GPU capabilities"""
        gpu_info = {
            "available": False,
            "name": None,
            "memory_gb": 0,
            "compute_capability": None
        }

        try:
            if torch.cuda.is_available():
                gpu_info["available"] = True
                gpu_info["name"] = torch.cuda.get_device_name(0)
                gpu_info["memory_gb"] = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)

                # Get compute capability
                props = torch.cuda.get_device_properties(0)
                gpu_info["compute_capability"] = f"{props.major}.{props.minor}"

                info_id(f"GPU detected: {gpu_info['name']} ({gpu_info['memory_gb']:.1f}GB)")
            else:
                info_id("No GPU detected - using CPU training")

        except Exception as e:
            logger.warning(f"GPU detection failed: {e}")

        return gpu_info

    def _calculate_optimal_config(self):
        """Calculate optimal training configuration based on system specs"""
        config = {}

        # Determine if we should use GPU
        config["use_gpu"] = self.gpu_info["available"]
        config["fp16"] = self.gpu_info["available"]  # Use mixed precision on GPU

        # Calculate optimal batch size based on available memory
        if self.gpu_info["available"]:
            # GPU memory-based batch size calculation
            gpu_memory = self.gpu_info["memory_gb"]
            if gpu_memory >= 16:
                config["batch_size"] = 16
                config["max_length"] = 256
            elif gpu_memory >= 8:
                config["batch_size"] = 8
                config["max_length"] = 128
            elif gpu_memory >= 4:
                config["batch_size"] = 4
                config["max_length"] = 128
            else:
                config["batch_size"] = 2
                config["max_length"] = 64
        else:
            # CPU/RAM-based batch size calculation
            ram_gb = self.system_info["memory_gb"]
            if ram_gb >= 32:
                config["batch_size"] = 8
                config["max_length"] = 128
            elif ram_gb >= 16:
                config["batch_size"] = 4
                config["max_length"] = 128
            elif ram_gb >= 8:
                config["batch_size"] = 2
                config["max_length"] = 64
            else:
                config["batch_size"] = 1
                config["max_length"] = 64

        # Calculate optimal number of workers
        cpu_count = self.system_info["cpu_count"]
        if self.gpu_info["available"]:
            # With GPU, use some CPU cores for data loading
            config["num_workers"] = min(4, max(1, cpu_count // 4))
        else:
            # CPU only - use fewer workers to avoid overhead
            config["num_workers"] = 0  # Streaming datasets work better with 0 workers

        # Calculate shuffle buffer size based on available memory
        ram_gb = self.system_info["memory_gb"]
        if ram_gb >= 16:
            config["shuffle_buffer_size"] = 2000
        elif ram_gb >= 8:
            config["shuffle_buffer_size"] = 1000
        else:
            config["shuffle_buffer_size"] = 500

        # Gradient accumulation to maintain effective batch size
        target_effective_batch = 16  # Target effective batch size
        config["gradient_accumulation_steps"] = max(1, target_effective_batch // config["batch_size"])

        # Learning rate adjustment based on effective batch size
        effective_batch = config["batch_size"] * config["gradient_accumulation_steps"]
        config["learning_rate"] = 5e-5 * (effective_batch / 16)  # Scale learning rate

        return config

    def get_recommended_mode_for_examples(self, max_examples):
        """Get recommended training parameters for given number of examples"""
        if max_examples is None:
            # Full dataset
            return {
                "num_epochs": 1 if not self.gpu_info["available"] else 2,
                "eval_steps": 2000,
                "save_steps": 2000,
                "estimated_time_hours": 12 if self.gpu_info["available"] else 48
            }
        elif max_examples <= 10000:
            return {
                "num_epochs": 5,
                "eval_steps": 100,
                "save_steps": 100,
                "estimated_time_hours": 0.2 if self.gpu_info["available"] else 0.5
            }
        elif max_examples <= 100000:
            return {
                "num_epochs": 3,
                "eval_steps": 500,
                "save_steps": 500,
                "estimated_time_hours": 1 if self.gpu_info["available"] else 3
            }
        else:
            return {
                "num_epochs": 2,
                "eval_steps": 1000,
                "save_steps": 1000,
                "estimated_time_hours": 4 if self.gpu_info["available"] else 12
            }

    def print_optimization_summary(self):
        """Print summary of detected system and optimizations"""
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
    """Streaming dataset that reads JSONL line by line without loading all into memory"""

    def __init__(self, jsonl_file, tokenizer, max_length=128, max_examples=None,
                 shuffle_buffer_size=1000, skip_examples=0, epoch=0):
        self.jsonl_file = jsonl_file
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.max_examples = max_examples
        self.shuffle_buffer_size = shuffle_buffer_size
        self.skip_examples = skip_examples
        self.epoch = epoch

    def convert_example_to_ner_format(self, example):
        """Convert JSONL example to NER format"""
        text = example["text"]
        entities = example.get("entities", [])

        # Tokenize text to get word boundaries
        words = text.split()
        word_starts = []
        pos = 0
        for word in words:
            start = text.find(word, pos)
            word_starts.append(start)
            pos = start + len(word)

        # Create BIO labels for each word
        labels = ["O"] * len(words)

        for entity in entities:
            entity_start = entity["start"]
            entity_end = entity["end"]
            entity_type = entity["entity"]

            # Find which words overlap with this entity
            first_word = None
            last_word = None

            for i, word_start in enumerate(word_starts):
                word_end = word_start + len(words[i])

                # Check if word overlaps with entity
                if word_start < entity_end and word_end > entity_start:
                    if first_word is None:
                        first_word = i
                    last_word = i

            # Apply BIO tagging
            if first_word is not None:
                labels[first_word] = f"B-{entity_type}"
                for i in range(first_word + 1, (last_word or first_word) + 1):
                    labels[i] = f"I-{entity_type}"

        # Convert labels to IDs
        label_ids = [LABEL2ID.get(label, 0) for label in labels]

        return {"tokens": words, "ner_tags": label_ids}

    def tokenize_example(self, example):
        """Tokenize a single example and align labels"""
        tokenized = self.tokenizer(
            example["tokens"],
            is_split_into_words=True,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt"
        )

        # Align labels with subword tokens
        word_ids = tokenized.word_ids()
        labels = []
        prev_word = None

        for word_id in word_ids:
            if word_id is None:
                labels.append(-100)  # Special tokens
            else:
                label = example["ner_tags"][word_id]
                if word_id != prev_word:
                    labels.append(label)  # First subword keeps original label
                else:
                    # Convert B- to I- for subsequent subwords
                    if LABELS[label].startswith("B-"):
                        inside_label = "I-" + LABELS[label][2:]
                        labels.append(LABEL2ID[inside_label])
                    else:
                        labels.append(label)
                prev_word = word_id

        return {
            "input_ids": tokenized["input_ids"].squeeze(),
            "attention_mask": tokenized["attention_mask"].squeeze(),
            "labels": torch.tensor(labels, dtype=torch.long)
        }

    def __iter__(self):
        """Generator that yields one training example at a time"""
        # Set random seed based on epoch for reproducible shuffling across epochs
        random.seed(42 + self.epoch)

        buffer = []
        count = 0

        with open(self.jsonl_file, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                # Skip examples if needed (for train/val/test splits)
                if line_num <= self.skip_examples:
                    continue

                if self.max_examples and count >= self.max_examples:
                    break

                try:
                    example = json.loads(line)
                    ner_example = self.convert_example_to_ner_format(example)
                    tokenized = self.tokenize_example(ner_example)

                    # Add to shuffle buffer
                    buffer.append(tokenized)

                    # When buffer is full, shuffle and yield
                    if len(buffer) >= self.shuffle_buffer_size:
                        random.shuffle(buffer)
                        for item in buffer:
                            yield item
                        buffer = []

                    count += 1

                    # Progress logging
                    if count % 1000 == 0:
                        info_id(f"Streamed {count} examples...")

                except (json.JSONDecodeError, KeyError) as e:
                    logger.warning(f"Skipping malformed JSON at line {line_num}: {e}")
                    continue

        # Yield remaining items in buffer
        if buffer:
            random.shuffle(buffer)
            for item in buffer:
                yield item


class StreamingTrainer(Trainer):
    """Custom trainer that works with streaming datasets"""

    def __init__(self, streaming_dataset_config=None, **kwargs):
        super().__init__(**kwargs)
        self.streaming_dataset_config = streaming_dataset_config or {}
        self.current_epoch = 0

    def get_train_dataloader(self):
        """Override to create new streaming dataset for each epoch"""
        if self.train_dataset is None:
            raise ValueError("Trainer: training requires a train_dataset.")

        # Create new streaming dataset with current epoch for different shuffling
        streaming_dataset = StreamingNERDataset(
            epoch=self.current_epoch,
            **self.streaming_dataset_config
        )

        return DataLoader(
            streaming_dataset,
            batch_size=self.args.per_device_train_batch_size,
            collate_fn=self.data_collator,
            num_workers=self.args.dataloader_num_workers,
            pin_memory=self.args.dataloader_pin_memory and torch.cuda.is_available()
        )

    def _inner_training_loop(self, **kwargs):
        """Override to track epochs for streaming dataset"""
        # Update epoch counter before each epoch
        original_result = super()._inner_training_loop(**kwargs)
        self.current_epoch += 1
        return original_result


def create_validation_dataset(jsonl_file, tokenizer, max_length=128, val_size=5000):
    """Create a small validation dataset (non-streaming for consistent evaluation)"""
    examples = []

    with open(jsonl_file, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if i >= val_size:
                break
            try:
                example = json.loads(line)
                # Convert to NER format
                text = example["text"]
                entities = example.get("entities", [])

                words = text.split()
                labels = ["O"] * len(words)

                # Apply BIO tagging (same logic as streaming dataset)
                for entity in entities:
                    entity_start = entity["start"]
                    entity_end = entity["end"]
                    entity_type = entity["entity"]

                    first_word = None
                    last_word = None

                    for j, word in enumerate(words):
                        word_start = text.find(word, sum(len(w) + 1 for w in words[:j]))
                        word_end = word_start + len(word)

                        if word_start < entity_end and word_end > entity_start:
                            if first_word is None:
                                first_word = j
                            last_word = j

                    if first_word is not None:
                        labels[first_word] = f"B-{entity_type}"
                        for k in range(first_word + 1, (last_word or first_word) + 1):
                            labels[k] = f"I-{entity_type}"

                label_ids = [LABEL2ID.get(label, 0) for label in labels]
                examples.append({"tokens": words, "ner_tags": label_ids})

            except json.JSONDecodeError:
                continue

    # Convert to dataset and tokenize
    dataset = Dataset.from_list(examples)

    def tokenize_and_align(batch):
        tokenized = tokenizer(
            batch["tokens"],
            is_split_into_words=True,
            truncation=True,
            padding="max_length",
            max_length=max_length,
        )

        aligned = []
        for i, word_labels in enumerate(batch["ner_tags"]):
            word_ids = tokenized.word_ids(batch_index=i)
            prev_word = None
            label_ids = []
            for wid in word_ids:
                if wid is None:
                    label_ids.append(-100)
                else:
                    base_id = word_labels[wid]
                    if wid != prev_word:
                        label_ids.append(base_id)
                    else:
                        # Convert B- to I- for subwords
                        lab = ID2LABEL[base_id]
                        if lab.startswith("B-"):
                            inside = "I-" + lab[2:]
                            label_ids.append(LABEL2ID[inside])
                        else:
                            label_ids.append(base_id)
                    prev_word = wid
            aligned.append(label_ids)
        tokenized["labels"] = aligned
        return tokenized

    return dataset.map(tokenize_and_align, batched=True, remove_columns=["tokens", "ner_tags"])


def simple_compute_metrics_fn(p):
    """Simplified metrics computation"""
    predictions, labels = p
    predictions = np.argmax(predictions, axis=2)

    true_labels = []
    true_predictions = []

    for label, pred in zip(labels, predictions):
        true_label = [LABELS[l] for l, p_ in zip(label, pred) if l != -100]
        true_pred = [LABELS[p_] for l, p_ in zip(label, pred) if l != -100]

        if true_label:
            true_labels.append(true_label)
            true_predictions.append(true_pred)

    try:
        return {
            "f1": f1_score(true_labels, true_predictions),
            "accuracy": accuracy_score(true_labels, true_predictions),
        }
    except Exception as e:
        logger.warning(f"Error computing metrics: {e}")
        return {"f1": 0.0, "accuracy": 0.0}


def _extract_entities_from_text(text, tokenizer, model, id2label):
    """Extract entities from text using the trained model"""
    model.eval()
    enc = tokenizer(
        text,
        return_offsets_mapping=True,
        return_tensors="pt",
        truncation=True,
        max_length=512,
        return_token_type_ids=False,
    )

    with torch.no_grad():
        out = model(
            input_ids=enc["input_ids"],
            attention_mask=enc["attention_mask"],
        )
        logits = out.logits[0]
        probs = torch.softmax(logits, dim=-1)
        pred_ids = logits.argmax(dim=-1)

    tokens = tokenizer.convert_ids_to_tokens(enc["input_ids"][0])
    offsets = enc["offset_mapping"][0].tolist()

    spans = []
    cur = None

    def flush():
        nonlocal cur
        if cur is not None and cur["start"] < cur["end"]:
            avg_score = cur["score_sum"] / max(cur["steps"], 1)
            spans.append({
                "label": cur["label"],
                "start": cur["start"],
                "end": cur["end"],
                "text": text[cur["start"]:cur["end"]],
                "score": float(avg_score),
            })
        cur = None

    for tok, (s, e), pid, p in zip(tokens, offsets, pred_ids.tolist(), probs.tolist()):
        if tok in ("[CLS]", "[SEP]") or s == e:
            continue
        lab = id2label.get(pid, "O")
        if lab == "O":
            flush()
            continue

        if lab.startswith("B-"):
            flush()
            cur = {"label": lab[2:], "start": s, "end": e, "score_sum": p[pid], "steps": 1}
        elif lab.startswith("I-"):
            if cur is not None and cur["label"] == lab[2:] and s >= cur["end"]:
                cur["end"] = e
                cur["score_sum"] += p[pid]
                cur["steps"] += 1
            else:
                flush()
                cur = {"label": lab[2:], "start": s, "end": e, "score_sum": p[pid], "steps": 1}
        else:
            flush()

    flush()
    spans.sort(key=lambda x: x["start"])
    return spans


def run_demo(tokenizer, model, id2label):
    """Interactive demo to test the model"""
    print("\n=== Quick NER demo ===")
    samples = [
        "I need the print for equipment E-1001",
        "Show me the centrifugal pump assembly drawing",
        "Find drawing DWG-12345 for the heat exchanger",
        "Where is part A123456 on the schematic?",
        "I need the P&ID for equipment number PUMP-205"
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
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=['fast', 'small', 'medium', 'full'], default=None,
                        help="Training mode: fast (2K), small (10K), medium (500K), or full (all examples)")
    parser.add_argument("--max-examples", type=int, default=None, help="Maximum number of examples to use")
    parser.add_argument("--force-cpu", action="store_true", help="Force CPU training even if GPU is available")
    parser.add_argument("--override-batch-size", type=int, default=None, help="Override auto-detected batch size")
    parser.add_argument("--override-max-length", type=int, default=None, help="Override auto-detected max length")
    args = parser.parse_args()

    # Initialize system optimizer
    optimizer = SystemOptimizer()
    optimizer.print_optimization_summary()

    # Interactive mode selection if not specified
    if not args.mode and not args.max_examples:
        print("Select training mode:")
        print("1. Fast mode - 2,000 examples")
        print("2. Small mode - 10,000 examples")
        print("3. Medium mode - 500,000 examples")
        print("4. Full mode - All 5M examples")

        while True:
            choice = input("Enter choice [1/2/3/4]: ").strip()
            if choice == "1":
                args.mode = "fast"
                break
            elif choice == "2":
                args.mode = "small"
                break
            elif choice == "3":
                args.mode = "medium"
                break
            elif choice == "4":
                args.mode = "full"
                break
            else:
                print("Please enter 1, 2, 3, or 4")

    # Set max examples based on mode
    if not args.max_examples:
        if args.mode == "fast":
            args.max_examples = 2000
        elif args.mode == "small":
            args.max_examples = 10000
        elif args.mode == "medium":
            args.max_examples = 500000
        elif args.mode == "full":
            args.max_examples = None  # Use all examples

    info_id(f"Python executable: {sys.executable}")
    import transformers
    info_id(f"Transformers version: {transformers.__version__}")

    train_file = os.path.join(ORC_DRAWINGS_TRAIN_DATA_DIR, "intent_train_drawings.jsonl")
    model_dir = ORC_DRAWINGS_MODEL_DIR

    os.makedirs(model_dir, exist_ok=True)

    # Get optimal configuration and mode recommendations
    config = optimizer.optimal_config
    mode_config = optimizer.get_recommended_mode_for_examples(args.max_examples)

    # Apply overrides if specified
    if args.force_cpu:
        config["use_gpu"] = False
        config["fp16"] = False
        info_id("Forcing CPU training (GPU disabled)")

    if args.override_batch_size:
        config["batch_size"] = args.override_batch_size
        info_id(f"Overriding batch size to {args.override_batch_size}")

    if args.override_max_length:
        config["max_length"] = args.override_max_length
        info_id(f"Overriding max length to {args.override_max_length}")

    info_id(f"Training mode: {args.mode or 'custom'}")
    info_id(f"Max examples: {args.max_examples or 'ALL'}")
    info_id(f"Estimated training time: {mode_config['estimated_time_hours']:.1f} hours")
    info_id(f"Using adaptive configuration for your system")

    tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")

    # Create validation dataset (small, non-streaming)
    info_id("Creating validation dataset...")
    val_dataset = create_validation_dataset(train_file, tokenizer, config["max_length"])
    info_id(f"Validation dataset size: {len(val_dataset)}")

    # Initialize model
    model = DistilBertForTokenClassification.from_pretrained(
        "distilbert-base-uncased",
        num_labels=len(LABELS),
        id2label=ID2LABEL,
        label2id=LABEL2ID,
    )

    # Move model to GPU if available and not forced to CPU
    if config["use_gpu"] and not args.force_cpu:
        model = model.cuda()

    data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)

    # Calculate steps per epoch for logging
    if args.max_examples:
        steps_per_epoch = max(1, args.max_examples // config["batch_size"])
    else:
        steps_per_epoch = 1000  # Estimate for full dataset

    info_id(f"Estimated steps per epoch: {steps_per_epoch}")

    training_args = TrainingArguments(
        output_dir=model_dir,
        eval_strategy="steps",
        save_strategy="steps",
        eval_steps=mode_config["eval_steps"],
        save_steps=mode_config["save_steps"],
        save_total_limit=1,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",  # Use loss instead of f1 to avoid key error
        greater_is_better=False,  # Lower loss is better

        learning_rate=config["learning_rate"],
        weight_decay=0.01,
        warmup_ratio=0.1,
        per_device_train_batch_size=config["batch_size"],
        per_device_eval_batch_size=config["batch_size"],
        gradient_accumulation_steps=config["gradient_accumulation_steps"],
        num_train_epochs=mode_config["num_epochs"],
        max_steps=steps_per_epoch * mode_config["num_epochs"],

        logging_dir=os.path.join(model_dir, "logs"),
        logging_steps=50,
        report_to="none",
        seed=42,
        fp16=config["fp16"] and not args.force_cpu,
        dataloader_pin_memory=config["use_gpu"] and not args.force_cpu,
        dataloader_num_workers=config["num_workers"],
        remove_unused_columns=True,

        # Memory optimizations - remove problematic settings
        # prediction_loss_only=True,  # This prevents compute_metrics from being called
        # include_inputs_for_metrics=False,  # This also prevents proper metrics
    )

    # Configuration for streaming dataset
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
        train_dataset="placeholder",  # Will be replaced by streaming dataset
        eval_dataset=val_dataset,
        processing_class=tokenizer,
        data_collator=data_collator,
        compute_metrics=simple_compute_metrics_fn,
        streaming_dataset_config=streaming_config,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3, early_stopping_threshold=0.0)],
    )

    info_id("Starting adaptive streaming training...")
    info_id(f"Optimized for: {optimizer.system_info['platform']} with {optimizer.system_info['memory_gb']:.1f}GB RAM")
    if optimizer.gpu_info["available"] and not args.force_cpu:
        info_id(f"Using GPU acceleration: {optimizer.gpu_info['name']}")
    else:
        info_id("Using CPU training")

    trainer.train()

    info_id("Evaluating final model...")
    val_metrics = trainer.evaluate()
    for k, v in val_metrics.items():
        info_id(f"Final {k}: {v}")

    # Run interactive demo
    run_demo(tokenizer, model, ID2LABEL)

    info_id(f"Saving model to {model_dir}")
    trainer.save_model(model_dir)
    tokenizer.save_pretrained(model_dir)
    info_id("Adaptive streaming training complete!")


if __name__ == "__main__":
    main()