from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModelForTokenClassification, Trainer, TrainingArguments
from datasets import load_dataset
import os
from modules.configuration.log_config import debug_id, info_id, error_id, with_request_id

class IntentTrainer:
    def __init__(self, base_model_dir, labels):
        """
        base_model_dir: pretrained model name or path (e.g., "distilbert-base-uncased")
        labels: list of intent labels, e.g. ["parts", "images", "documents", "prints", "tools", "troubleshooting"]
        """
        self.base_model_dir = base_model_dir
        self.labels = labels
        self.id2label = {i: label for i, label in enumerate(labels)}
        self.label2id = {label: i for i, label in enumerate(labels)}

    @with_request_id
    def train(self, train_data_path, output_dir, epochs=3, batch_size=8, request_id=None):
        """
        train_data_path: path to JSONL training data with fields "text" and "intent"
        output_dir: directory to save fine-tuned model
        epochs: number of training epochs
        batch_size: batch size per device
        """
        try:
            info_id(f"Loading dataset from {train_data_path}...", request_id)

            dataset = load_dataset("json", data_files=train_data_path)["train"]
            info_id(f"Dataset loaded with {len(dataset)} examples", request_id)

            info_id(f"Loading tokenizer and model from {self.base_model_dir}...", request_id)
            tokenizer = AutoTokenizer.from_pretrained(self.base_model_dir)
            model = AutoModelForSequenceClassification.from_pretrained(
                self.base_model_dir,
                num_labels=len(self.labels),
                id2label=self.id2label,
                label2id=self.label2id,
                ignore_mismatched_sizes=True,
            )

            def map_intent(example):
                example["label"] = self.label2id[example["intent"]]
                return example

            dataset = dataset.map(map_intent)

            def tokenize_fn(examples):
                return tokenizer(examples["text"], truncation=True, padding=True)

            tokenized_dataset = dataset.map(
                tokenize_fn,
                batched=True,
                remove_columns=["text", "intent"]  # keep 'label' column for Trainer
            )

            training_args = TrainingArguments(
                output_dir=output_dir,
                num_train_epochs=epochs,
                per_device_train_batch_size=batch_size,
                save_steps=10,
                save_total_limit=2,
                logging_steps=5,
                report_to="none",
            )

            trainer = Trainer(
                model=model,
                args=training_args,
                train_dataset=tokenized_dataset,
                tokenizer=tokenizer,
            )

            info_id(f"Starting training for {epochs} epochs...", request_id)
            trainer.train()

            info_id(f"Saving fine-tuned model to {output_dir}...", request_id)
            model.save_pretrained(output_dir)
            tokenizer.save_pretrained(output_dir)

            info_id("Training complete!", request_id)

        except Exception as e:
            error_id(f"Training failed: {str(e)}", request_id)
            raise


class NERTrainer:
    def __init__(self, base_model_dir, labels):
        """
        base_model_dir: path to pretrained model directory or model name (e.g., "dslim/bert-base-NER")
        labels: list of entity labels, e.g. ["O", "B-PARTDESC", "B-PARTNUM"]
        """
        self.base_model_dir = base_model_dir
        self.labels = labels
        self.id2label = {i: label for i, label in enumerate(labels)}
        self.label2id = {label: i for i, label in enumerate(labels)}

    def tokenize_and_align_labels(self, example, tokenizer):
        tokenized_inputs = tokenizer(
            example["tokens"],
            truncation=True,
            padding="max_length",
            max_length=128,
            is_split_into_words=True,
            return_offsets_mapping=True,
        )
        labels = []
        word_ids = tokenized_inputs.word_ids()
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
        tokenized_inputs["labels"] = labels
        return tokenized_inputs

    @with_request_id
    def train(self, train_data_path, output_dir, epochs=3, batch_size=4, request_id=None):
        """
        train_data_path: path to JSONL training data
        output_dir: where to save the fine-tuned model
        epochs: training epochs
        batch_size: batch size for training
        """
        try:
            info_id(f"Loading dataset from {train_data_path}...", request_id)
            dataset = load_dataset("json", data_files=train_data_path)["train"]

            info_id(f"Loading tokenizer and model from {self.base_model_dir}...", request_id)
            tokenizer = AutoTokenizer.from_pretrained(self.base_model_dir)
            model = AutoModelForTokenClassification.from_pretrained(
                self.base_model_dir,
                num_labels=len(self.labels),
                id2label=self.id2label,
                label2id=self.label2id,
                ignore_mismatched_sizes=True,
            )

            info_id("Tokenizing and aligning labels...", request_id)
            tokenized_dataset = dataset.map(
                lambda x: self.tokenize_and_align_labels(x, tokenizer),
                batched=False,
                remove_columns=dataset.column_names,
            )

            training_args = TrainingArguments(
                output_dir=output_dir,
                num_train_epochs=epochs,
                per_device_train_batch_size=batch_size,
                save_steps=10,
                save_total_limit=2,
                logging_steps=5,
                report_to="none",
            )

            trainer = Trainer(
                model=model,
                args=training_args,
                train_dataset=tokenized_dataset,
                tokenizer=tokenizer,
            )

            info_id(f"Starting training for {epochs} epochs...", request_id)
            trainer.train()

            info_id(f"Saving fine-tuned model to {output_dir}...", request_id)
            model.save_pretrained(output_dir)
            tokenizer.save_pretrained(output_dir)

            info_id("Training complete!", request_id)

        except Exception as e:
            error_id(f"Training failed: {str(e)}", request_id)
            raise
