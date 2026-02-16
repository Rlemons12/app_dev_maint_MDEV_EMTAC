from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModelForTokenClassification, Trainer, TrainingArguments
from datasets import load_dataset
import os
from configuration.log_config import debug_id, info_id, error_id, with_request_id
from pathlib import Path
from typing import Dict, Any
from transformers import TrainerCallback
from modules.gpu.gpu_training_adapter import GPUTrainingAdapter
from configuration.mlflow_utils import init_mlflow, end_mlflow

class MLflowLoggingCallback(TrainerCallback):
    """
    Logs Trainer events to MLflow without relying on transformers' integration layer.
    Works across transformers versions.

    - on_train_begin: logs core params
    - on_log: logs loss/grad_norm/lr, etc
    - on_evaluate: logs eval_* metrics (precision/recall/f1)
    - on_train_end: logs artifacts (run_dir, best/)
    """

    def __init__(self, run_dir: Path, base_params: Dict[str, Any]):
        self.run_dir = Path(run_dir)
        self.base_params = dict(base_params)
        self._enabled = False

        # Avoid hard-failing if mlflow isn't installed
        try:
            import mlflow  # noqa: F401
            self._enabled = True
        except Exception:
            self._enabled = False

    def on_train_begin(self, args, state, control, **kwargs):
        if not self._enabled:
            return

        from configuration.mlflow_utils import log_params, log_text

        log_params(self.base_params)

        # Optional: record the TrainingArguments as text for reproducibility
        try:
            log_text("training_args", str(args))
        except Exception:
            pass

    def on_log(self, args, state, control, logs=None, **kwargs):
        if not self._enabled or not logs:
            return
        from configuration.mlflow_utils import log_metrics
        # state.global_step is the cleanest step
        log_metrics(logs, step=int(getattr(state, "global_step", 0) or 0))

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        if not self._enabled or not metrics:
            return
        from configuration.mlflow_utils import log_metrics
        # Metrics often include eval_loss, eval_f1, etc.
        step = int(getattr(state, "global_step", 0) or 0)
        log_metrics(metrics, step=step)

    def on_train_end(self, args, state, control, **kwargs):
        if not self._enabled:
            return
        from configuration.mlflow_utils import log_artifacts

        # Log full run dir + best subdir as artifacts
        log_artifacts(self.run_dir)
        best_dir = self.run_dir / "best"
        if best_dir.exists():
            log_artifacts(self.run_dir, subdir="best")

class IntentTrainer:
    def __init__(self, base_model_dir: str, labels: list[str]):
        """
        base_model_dir: pretrained model name or path
        labels: intent labels (e.g. ["parts", "images", "documents", ...])
        """
        self.base_model_dir = base_model_dir
        self.labels = labels

        self.id2label = {i: label for i, label in enumerate(labels)}
        self.label2id = {label: i for i, label in enumerate(labels)}

        # GPU adapter (local CUDA / remote service / CPU)
        self.gpu_adapter = GPUTrainingAdapter()

    # --------------------------------------------------
    # Training
    # --------------------------------------------------
    @with_request_id
    def train(
        self,
        train_data_path: str,
        output_dir: str,
        epochs: int = 3,
        batch_size: int = 8,
        request_id: str | None = None,
    ):
        """
        train_data_path: JSONL file with fields ["text", "intent"]
        output_dir: directory to save trained model
        """
        run_dir = Path(output_dir)

        try:
            # ---------------------------
            # Load dataset
            # ---------------------------
            info_id(f"Loading dataset from {train_data_path}", request_id)
            dataset = load_dataset("json", data_files=train_data_path)["train"]
            info_id(f"Dataset loaded ({len(dataset)} samples)", request_id)

            # ---------------------------
            # Tokenizer + model
            # ---------------------------
            info_id(f"Loading base model: {self.base_model_dir}", request_id)

            tokenizer = AutoTokenizer.from_pretrained(self.base_model_dir)

            model = AutoModelForSequenceClassification.from_pretrained(
                self.base_model_dir,
                num_labels=len(self.labels),
                id2label=self.id2label,
                label2id=self.label2id,
                ignore_mismatched_sizes=True,
            )

            # GPU / CPU / Remote routing
            model = self.gpu_adapter.prepare_model(model)

            # ---------------------------
            # Dataset prep
            # ---------------------------
            def map_intent(example):
                example["label"] = self.label2id[example["intent"]]
                return example

            dataset = dataset.map(map_intent)

            def tokenize_fn(batch):
                return tokenizer(
                    batch["text"],
                    truncation=True,
                    padding=True,
                )

            tokenized_dataset = dataset.map(
                tokenize_fn,
                batched=True,
                remove_columns=["text", "intent"],
            )

            # ---------------------------
            # Training args
            # ---------------------------
            training_args = TrainingArguments(
                output_dir=str(run_dir),
                num_train_epochs=epochs,
                per_device_train_batch_size=batch_size,
                save_steps=100,
                save_total_limit=2,
                logging_steps=25,
                report_to="none",  # MLflow handled manually
                remove_unused_columns=False,
            )

            # ---------------------------
            # MLflow setup
            # ---------------------------
            base_params: Dict[str, Any] = {
                "task": "intent_classification",
                "base_model": self.base_model_dir,
                "num_labels": len(self.labels),
                "epochs": epochs,
                "batch_size": batch_size,
                "dataset_size": len(dataset),
                "gpu_mode": (
                    "remote"
                    if self.gpu_adapter.remote_available
                    else "cuda"
                    if self.gpu_adapter.local_cuda
                    else "cpu"
                ),
            }

            mlflow_active = None
            try:
                mlflow_active = init_mlflow(
                    experiment_name="MODEL_TRAINING_intent",
                    run_name=run_dir.name,
                    tags={
                        "trainer": "IntentTrainer",
                        "model_type": "sequence_classification",
                    },
                )
            except Exception:
                mlflow_active = None

            # ---------------------------
            # Trainer
            # ---------------------------
            trainer = Trainer(
                model=model,
                args=training_args,
                train_dataset=tokenized_dataset,
                tokenizer=tokenizer,
                callbacks=[
                    MLflowLoggingCallback(
                        run_dir=run_dir,
                        base_params=base_params,
                    )
                ],
            )

            trainer = self.gpu_adapter.wrap_trainer(trainer)

            # ---------------------------
            # Train
            # ---------------------------
            success = False
            info_id(f"Starting training ({epochs} epochs)", request_id)

            try:
                trainer.train()
                success = True
            finally:
                if mlflow_active:
                    end_mlflow(success=success)

            # ---------------------------
            # Save artifacts
            # ---------------------------
            info_id(f"Saving model to {output_dir}", request_id)
            model.save_pretrained(run_dir)
            tokenizer.save_pretrained(run_dir)

            info_id("Intent training complete", request_id)

        except Exception as exc:
            error_id(f"Intent training failed: {exc}", request_id)
            raise

class NERTrainer:
    def __init__(self, base_model_dir: str, labels: list[str]):
        """
        base_model_dir: pretrained model name or path (e.g. "dslim/bert-base-NER")
        labels: entity labels (BIO format)
        """
        self.base_model_dir = base_model_dir
        self.labels = labels

        self.id2label = {i: label for i, label in enumerate(labels)}
        self.label2id = {label: i for i, label in enumerate(labels)}

        # GPU routing
        self.gpu_adapter = GPUTrainingAdapter()

    # --------------------------------------------------
    # Tokenization + alignment
    # --------------------------------------------------
    def tokenize_and_align_labels(self, example, tokenizer):
        tokenized = tokenizer(
            example["tokens"],
            truncation=True,
            padding="max_length",
            max_length=128,
            is_split_into_words=True,
            return_offsets_mapping=False,
        )

        labels = []
        word_ids = tokenized.word_ids()
        prev_word_id = None
        label_ids = example["ner_tags"]

        for word_id in word_ids:
            if word_id is None:
                labels.append(-100)
            elif word_id != prev_word_id:
                labels.append(label_ids[word_id])
            else:
                labels.append(label_ids[word_id])
            prev_word_id = word_id

        tokenized["labels"] = labels
        return tokenized

    # --------------------------------------------------
    # Training
    # --------------------------------------------------
    @with_request_id
    def train(
        self,
        train_data_path: str,
        output_dir: str,
        epochs: int = 3,
        batch_size: int = 4,
        request_id: str | None = None,
    ):
        """
        train_data_path: JSONL with fields ["tokens", "ner_tags"]
        output_dir: directory to save trained model
        """
        run_dir = Path(output_dir)

        try:
            # ---------------------------
            # Load dataset
            # ---------------------------
            info_id(f"Loading NER dataset from {train_data_path}", request_id)
            dataset = load_dataset("json", data_files=train_data_path)["train"]
            info_id(f"Dataset loaded ({len(dataset)} samples)", request_id)

            # ---------------------------
            # Tokenizer + model
            # ---------------------------
            info_id(f"Loading base model: {self.base_model_dir}", request_id)

            tokenizer = AutoTokenizer.from_pretrained(self.base_model_dir)

            model = AutoModelForTokenClassification.from_pretrained(
                self.base_model_dir,
                num_labels=len(self.labels),
                id2label=self.id2label,
                label2id=self.label2id,
                ignore_mismatched_sizes=True,
            )

            # GPU / CPU / Remote routing
            model = self.gpu_adapter.prepare_model(model)

            # ---------------------------
            # Dataset processing
            # ---------------------------
            info_id("Tokenizing and aligning NER labels", request_id)

            tokenized_dataset = dataset.map(
                lambda x: self.tokenize_and_align_labels(x, tokenizer),
                batched=False,
                remove_columns=dataset.column_names,
            )

            # ---------------------------
            # Training args
            # ---------------------------
            training_args = TrainingArguments(
                output_dir=str(run_dir),
                num_train_epochs=epochs,
                per_device_train_batch_size=batch_size,
                save_steps=100,
                save_total_limit=2,
                logging_steps=25,
                report_to="none",  # MLflow handled manually
                remove_unused_columns=False,
            )

            # ---------------------------
            # MLflow setup
            # ---------------------------
            base_params: Dict[str, Any] = {
                "task": "ner",
                "base_model": self.base_model_dir,
                "num_labels": len(self.labels),
                "epochs": epochs,
                "batch_size": batch_size,
                "dataset_size": len(dataset),
                "gpu_mode": (
                    "remote"
                    if self.gpu_adapter.remote_available
                    else "cuda"
                    if self.gpu_adapter.local_cuda
                    else "cpu"
                ),
            }

            mlflow_active = None
            try:
                mlflow_active = init_mlflow(
                    experiment_name="MODEL_TRAINING_ner",
                    run_name=run_dir.name,
                    tags={
                        "trainer": "NERTrainer",
                        "model_type": "token_classification",
                    },
                )
            except Exception:
                mlflow_active = None

            # ---------------------------
            # Trainer
            # ---------------------------
            trainer = Trainer(
                model=model,
                args=training_args,
                train_dataset=tokenized_dataset,
                tokenizer=tokenizer,
                callbacks=[
                    MLflowLoggingCallback(
                        run_dir=run_dir,
                        base_params=base_params,
                    )
                ],
            )

            trainer = self.gpu_adapter.wrap_trainer(trainer)

            # ---------------------------
            # Train
            # ---------------------------
            success = False
            info_id(f"Starting NER training ({epochs} epochs)", request_id)

            try:
                trainer.train()
                success = True
            finally:
                if mlflow_active:
                    end_mlflow(success=success)

            # ---------------------------
            # Save artifacts
            # ---------------------------
            info_id(f"Saving NER model to {output_dir}", request_id)
            model.save_pretrained(run_dir)
            tokenizer.save_pretrained(run_dir)

            info_id("NER training complete", request_id)

        except Exception as exc:
            error_id(f"NER training failed: {exc}", request_id)
            raise


