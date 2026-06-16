# interactive_intent_test.py

import sys
import time
from pathlib import Path
from typing import Dict

import torch
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
)

from configuration.log_config import (
    set_request_id,
    get_request_id,
    info_id,
    warning_id,
)

from configuration.config import MODEL_TRAINING_INTENT_MODEL_DIR


# =====================================================
# CONFIG
# =====================================================

DEFAULT_THRESHOLD = 0.92
TOP_K = 2


# =====================================================
# LOAD MODEL
# =====================================================

def load_intent_model(model_dir: Path):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    model.to(device)
    model.eval()

    id2label: Dict[int, str] = model.config.id2label

    return tokenizer, model, device, id2label


# =====================================================
# INFERENCE
# =====================================================

@torch.no_grad()
def predict_intent(
    text: str,
    tokenizer,
    model,
    device,
    id2label: Dict[int, str],
):
    inputs = tokenizer(
        text,
        truncation=True,
        padding=True,
        max_length=256,
        return_tensors="pt",
    ).to(device)

    outputs = model(**inputs)
    probs = torch.softmax(outputs.logits, dim=-1)[0]

    scores = {
        id2label[i]: float(probs[i])
        for i in range(len(probs))
    }

    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)

    return ranked


# =====================================================
# INTERACTIVE LOOP
# =====================================================

def interactive_loop():
    set_request_id()
    rid = get_request_id()

    model_dir = Path(MODEL_TRAINING_INTENT_MODEL_DIR)

    info_id(f"Loading intent model from {model_dir}", rid)

    tokenizer, model, device, id2label = load_intent_model(model_dir)

    info_id(
        f"Model ready | device={device} | labels={list(id2label.values())}",
        rid,
    )

    print("\n===================================================")
    print(" INTERACTIVE INTENT TESTER")
    print("===================================================")
    print("Type a query and press ENTER")
    print("Commands:")
    print("  :q      quit")
    print("  :help   show this help")
    print("  :thr X  set confidence threshold (e.g. :thr 0.90)")
    print("===================================================\n")

    threshold = DEFAULT_THRESHOLD

    while True:
        try:
            text = input(">> ").strip()
        except KeyboardInterrupt:
            print("\nExiting.")
            break

        if not text:
            continue

        # -----------------------------
        # Commands
        # -----------------------------
        if text == ":q":
            print("Goodbye.")
            break

        if text == ":help":
            print("\nCommands:")
            print("  :q      quit")
            print("  :help   show help")
            print("  :thr X  set confidence threshold")
            print()
            continue

        if text.startswith(":thr"):
            try:
                threshold = float(text.split()[1])
                print(f"Threshold set to {threshold:.2f}\n")
            except Exception:
                print("Usage: :thr 0.90\n")
            continue

        # -----------------------------
        # Prediction
        # -----------------------------
        t0 = time.time()
        ranked = predict_intent(text, tokenizer, model, device, id2label)
        elapsed = (time.time() - t0) * 1000

        top_label, top_score = ranked[0]

        print("\nPrediction:")
        print(f"  Intent     : {top_label}")
        print(f"  Confidence : {top_score:.4f}")
        print(f"  Latency    : {elapsed:.1f} ms")

        if top_score < threshold:
            warning_id(
                f"[LOW CONFIDENCE] intent={top_label} score={top_score:.4f}",
                rid,
            )
            print(f"  ⚠ Below threshold ({threshold:.2f})")

        print("\nScore breakdown:")
        for label, score in ranked[:TOP_K]:
            print(f"  {label:<10} {score:.4f}")

        print("\n" + "-" * 50 + "\n")


# =====================================================
# ENTRY
# =====================================================

if __name__ == "__main__":
    interactive_loop()
