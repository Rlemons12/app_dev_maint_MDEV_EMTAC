import os
import json
import sys
from pathlib import Path
from typing import Dict, List

import torch
from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline

from configuration.log_config import (
    info_id,
    debug_id,
    error_id,
    set_request_id,
)
from configuration.config import MODEL_TRAINING_DRAWINGS_MODEL_DIR
from modules.gpu.gpu_training_adapter import GPUTrainingAdapter


# ============================================================
# Setup
# ============================================================

REQUEST_ID = set_request_id()
info_id("Starting INTERACTIVE DRAWINGS NER PROBE", REQUEST_ID)

gpu_adapter = GPUTrainingAdapter()
info_id(f"[GPU] Adapter: {gpu_adapter.describe()}", REQUEST_ID)


# ============================================================
# Load model (OFFLINE SAFE)
# ============================================================

def load_drawings_ner(model_dir: Path):
    assert model_dir.exists(), f"Model dir does not exist: {model_dir}"

    tokenizer = AutoTokenizer.from_pretrained(
        str(model_dir),
        local_files_only=True,
    )

    model = AutoModelForTokenClassification.from_pretrained(
        str(model_dir),
        local_files_only=True,
    )

    model = gpu_adapter.prepare_model(model)

    nlp = pipeline(
        "ner",
        model=model,
        tokenizer=tokenizer,
        aggregation_strategy="simple",
        device=0 if torch.cuda.is_available() else -1,
    )

    return nlp


# ============================================================
# Entity utilities
# ============================================================

def normalize_word(text: str) -> str:
    return text.replace("##", "").strip()


def extract_entities(results: List[Dict]) -> Dict[str, List[Dict]]:
    entities: Dict[str, List[Dict]] = {}

    for r in results:
        label = r["entity_group"]
        entry = {
            "text": normalize_word(r["word"]),
            "score": round(float(r["score"]), 4),
            "start": r["start"],
            "end": r["end"],
        }
        entities.setdefault(label, []).append(entry)

    return entities


def summarize_entities(entities: Dict[str, List[Dict]]):
    summary = {}
    for label, items in entities.items():
        merged_text = " ".join(i["text"] for i in items)
        avg_score = sum(i["score"] for i in items) / len(items)
        summary[label] = {
            "value": merged_text,
            "avg_confidence": round(avg_score, 4),
            "count": len(items),
        }
    return summary


# ============================================================
# Interactive loop
# ============================================================

def main():
    model_dir = Path(MODEL_TRAINING_DRAWINGS_MODEL_DIR)
    info_id(f"Loading model from {model_dir}", REQUEST_ID)

    try:
        nlp = load_drawings_ner(model_dir)
    except Exception as e:
        error_id(f"Failed to load model: {e}", REQUEST_ID)
        sys.exit(1)

    info_id("Model loaded successfully", REQUEST_ID)

    print("\n" + "=" * 80)
    print("INTERACTIVE DRAWINGS NER PROBE")
    print("Type a query and press Enter.")
    print("Type 'exit' or 'quit' to stop.")
    print("=" * 80 + "\n")

    while True:
        try:
            query = input("🔎 Query> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting.")
            break

        if not query:
            continue

        if query.lower() in {"exit", "quit"}:
            break

        debug_id(f"Query: {query}", REQUEST_ID)

        results = nlp(query)

        print("\n--- RAW MODEL OUTPUT ---")
        for r in results:
            print(
                f"{r['entity_group']:>22} | "
                f"{r['word']:<25} | "
                f"{r['score']:.3f}"
            )

        entities = extract_entities(results)
        summary = summarize_entities(entities)

        print("\n--- EXTRACTED ENTITIES ---")
        if not summary:
            print("⚠️  No entities detected")
        else:
            for label, data in summary.items():
                print(
                    f"{label:<22} → "
                    f"'{data['value']}' "
                    f"(avg_conf={data['avg_confidence']}, tokens={data['count']})"
                )

        # Simple routing hint
        print("\n--- ROUTING HINT ---")
        if summary:
            print("Detected domain signals:")
            for label in summary:
                print(f"  ✔ {label}")
        else:
            print("No domain entities detected")

        print("\n" + "-" * 80 + "\n")


if __name__ == "__main__":
    main()

