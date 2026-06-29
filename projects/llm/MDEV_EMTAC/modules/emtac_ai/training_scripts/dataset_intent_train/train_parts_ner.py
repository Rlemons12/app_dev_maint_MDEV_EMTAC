import os
import argparse
import logging
import numpy as np
from datasets import load_dataset, DatasetDict
from transformers import (
    DistilBertTokenizerFast,
    DistilBertForTokenClassification,
    DataCollatorForTokenClassification,
    Trainer,
    TrainingArguments,
    EarlyStoppingCallback,
)
from seqeval.metrics import f1_score, accuracy_score,classification_report
from modules.emtac_ai.config import ORC_PARTS_TRAIN_DATA_DIR, ORC_PARTS_MODEL_DIR
import re
import torch  # make sure this is imported

PART_NUMBER_RE = re.compile(r"\bA1\d{5}\b", re.IGNORECASE)


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ---- Single source of truth for NER labels ----
LABELS = [
    "O",
    "B-PART_NUMBER", "I-PART_NUMBER",
    "B-PART_NAME", "I-PART_NAME",
    "B-MANUFACTURER", "I-MANUFACTURER",
    "B-MODEL", "I-MODEL",

]
ID2LABEL = {i: lab for i, lab in enumerate(LABELS)}
LABEL2ID = {lab: i for i, lab in enumerate(LABELS)}


def compute_metrics_fn(p):
    import numpy as np
    from seqeval.metrics import (
        f1_score,
        accuracy_score,
        precision_score,
        recall_score,
        classification_report,
    )

    predictions, labels = p
    predictions = np.argmax(predictions, axis=2)

    # Strip out ignored (-100) positions
    true_labels = [
        [LABELS[l] for l, p_ in zip(label, pred) if l != -100]
        for label, pred in zip(labels, predictions)
    ]
    true_predictions = [
        [LABELS[p_] for l, p_ in zip(label, pred) if l != -100]
        for label, pred in zip(labels, predictions)
    ]

    # Log the detailed seqeval classification report
    logger.info("\n" + classification_report(true_labels, true_predictions, digits=3))

    return {
        "precision": precision_score(true_labels, true_predictions),
        "recall": recall_score(true_labels, true_predictions),
        "f1": f1_score(true_labels, true_predictions),
        "accuracy": accuracy_score(true_labels, true_predictions),
    }


def _extract_entities_from_text(text, tokenizer, model, id2label):
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
        )  # logits: [1, seq_len, num_labels]
        logits = out.logits[0]                 # [seq_len, num_labels]
        probs = torch.softmax(logits, dim=-1)  # [seq_len, num_labels]
        pred_ids = logits.argmax(dim=-1)       # [seq_len]

    tokens   = tokenizer.convert_ids_to_tokens(enc["input_ids"][0])
    offsets  = enc["offset_mapping"][0].tolist()

    # Build spans from BIO predictions (using char offsets)
    spans = []
    cur = None  # {label,start,end,score_sum,steps}

    def flush():
        nonlocal cur
        if cur is not None and cur["start"] < cur["end"]:
            # average token prob across the span
            avg_score = cur["score_sum"] / max(cur["steps"], 1)
            spans.append({
                "label": cur["label"],
                "start": cur["start"],
                "end":   cur["end"],
                "text":  text[cur["start"]:cur["end"]],
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

        # remove BIO prefix for the entity label
        if lab.startswith("B-"):
            flush()
            cur = {"label": lab[2:], "start": s, "end": e, "score_sum": p[pid], "steps": 1}
        elif lab.startswith("I-"):
            if cur is not None and cur["label"] == lab[2:] and s >= cur["end"]:
                # extend current span
                cur["end"] = e
                cur["score_sum"] += p[pid]
                cur["steps"] += 1
            else:
                # inconsistent I- (start a fresh span)
                flush()
                cur = {"label": lab[2:], "start": s, "end": e, "score_sum": p[pid], "steps": 1}
        else:
            flush()

    flush()

    # Regex override for A1 + 5 digits ⇒ force PART_NUMBER
    overrides = [(m.start(), m.end()) for m in PART_NUMBER_RE.finditer(text)]
    if overrides:
        # keep spans that DO NOT overlap the override ranges
        kept = []
        for sp in spans:
            if any(not (sp["end"] <= s or sp["start"] >= e) for (s, e) in overrides):
                # drop overlapping span (we’ll replace with PART_NUMBER)
                continue
            kept.append(sp)
        # add hard PART_NUMBER spans (confidence = 1.0 to make it obvious)
        for s, e in overrides:
            kept.append({
                "label": "PART_NUMBER",
                "start": s,
                "end": e,
                "text": text[s:e],
                "score": 1.0,
            })
        spans = kept

    # Sort and return compact objects
    spans.sort(key=lambda x: x["start"])
    return spans


def run_demo(tokenizer, model, id2label):
    print("\n=== Quick NER demo ===")
    samples = [
        "Do you have manufacturer part number RB20080S by balston filt?",
        "Need MPN 200-80-BX from Balston FILT, filter tube 10/box.",
        "Looking for item A101576 (model 200-80-BX) made by Balston.",
        "What’s the name of A123456?",
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
    parser.add_argument("--fast", action="store_true", help="Use a small subset for quick testing")
    args = parser.parse_args()

    if not args.fast:
        choice = input("Do you want to run in FAST mode (small subset for testing)? [y/N]: ").strip().lower()
        if choice == "y":
            args.fast = True

    logger.info(f"Python executable: {os.sys.executable}")
    import transformers
    logger.info(f"Transformers version: {transformers.__version__}")

    train_file = os.path.join(ORC_PARTS_TRAIN_DATA_DIR, "ner_train_parts.jsonl")
    logger.info(f"Loading dataset from {train_file} ...")
    ds = load_dataset("json", data_files={"train": train_file})["train"]
    logger.info(f"Total examples: {len(ds)}")
    logger.info(f"Sample[0]: {ds[0]}")

    if args.fast:
        ds = ds.select(range(min(2000, len(ds))))
        logger.info(f"[FAST MODE] Using {len(ds)} examples")

    dsdict = ds.train_test_split(test_size=0.1, seed=42)
    dsdict = DatasetDict(train=dsdict["train"], validation=dsdict["test"])

    tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")

    def tokenize_and_align(batch):
        tokenized = tokenizer(
            batch["tokens"],
            is_split_into_words=True,
            truncation=True,
            padding="max_length",
            max_length=512,
        )

        def to_inside(lab_id):
            lab = ID2LABEL[lab_id]
            if lab.startswith("B-"):
                inside = "I-" + lab[2:]
                return LABEL2ID[inside]
            return lab_id

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
                        # first subword of this word → keep original (B- or O)
                        label_ids.append(base_id)
                    else:
                        # subsequent subword(s) → convert B-XXX → I-XXX
                        label_ids.append(to_inside(base_id))
                    prev_word = wid
            aligned.append(label_ids)
        tokenized["labels"] = aligned
        return tokenized

    tokenized = dsdict.map(tokenize_and_align, batched=True, remove_columns=dsdict["train"].column_names)

    model = DistilBertForTokenClassification.from_pretrained(
        "distilbert-base-uncased",
        num_labels=len(LABELS),
        id2label=ID2LABEL,
        label2id=LABEL2ID,
    )

    data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)

    try:
        import torch
        use_fp16 = torch.cuda.is_available()
    except Exception:
        use_fp16 = False

    training_args = TrainingArguments(
        output_dir=ORC_PARTS_MODEL_DIR,
        eval_strategy="steps",
        save_strategy="steps",
        eval_steps=200,
        save_steps=200,
        save_total_limit=1,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,

        learning_rate=5e-5,
        weight_decay=0.01,
        warmup_ratio=0.1,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        gradient_accumulation_steps=1,
        num_train_epochs=5,

        logging_dir=os.path.join(ORC_PARTS_MODEL_DIR, "logs"),
        logging_steps=50,
        report_to="none",
        seed=42,
        fp16=use_fp16,
        dataloader_pin_memory=False,
        use_cpu=not use_fp16,
    )


    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized["validation"],
        processing_class=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics_fn,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=1, early_stopping_threshold=0.0)],
    )

    logger.info("Starting training...")
    trainer.train()

    logger.info("Evaluating best checkpoint...")
    metrics = trainer.evaluate()
    for k, v in metrics.items():
        logger.info(f"{k}: {v}")

    run_demo(tokenizer, model, ID2LABEL)

    logger.info(f"Saving model to {ORC_PARTS_MODEL_DIR}")
    trainer.save_model(ORC_PARTS_MODEL_DIR)
    tokenizer.save_pretrained(ORC_PARTS_MODEL_DIR)
    logger.info("Done.")


if __name__ == "__main__":
    main()
