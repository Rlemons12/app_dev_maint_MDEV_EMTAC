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
from configuration.config import (
    MODEL_TRAINING_PARTS_MODEL_DIR,
    MODEL_TRAINING_DRAWINGS_MODEL_DIR,
)
from modules.gpu.gpu_training_adapter import GPUTrainingAdapter


# ============================================================
# Setup
# ============================================================

REQ_ID = set_request_id()
info_id("Starting COMBINED PARTS + DRAWINGS NER PROBE", REQ_ID)

gpu = GPUTrainingAdapter()
info_id(f"[GPU] Adapter: {gpu.describe()}", REQ_ID)


# ============================================================
# Utilities
# ============================================================

def normalize_word(text: str) -> str:
    return text.replace("##", "").strip()


def summarize(preds: List[Dict]) -> Dict[str, Dict]:
    """
    Collapse token-level predictions into label summaries.
    """
    out: Dict[str, Dict] = {}

    for p in preds:
        label = p.get("entity_group") or p.get("entity")
        word = normalize_word(p.get("word", ""))
        score = float(p.get("score", 0.0))

        if not label:
            continue

        if label not in out:
            out[label] = {
                "tokens": [],
                "scores": [],
            }

        out[label]["tokens"].append(word)
        out[label]["scores"].append(score)

    summary = {}
    for label, v in out.items():
        merged = " ".join(v["tokens"])
        avg_score = sum(v["scores"]) / len(v["scores"])
        summary[label] = {
            "value": merged,
            "avg_conf": round(avg_score, 4),
            "count": len(v["tokens"]),
        }

    return summary


def load_ner_pipeline(model_dir: Path, name: str):
    assert model_dir.exists(), f"{name} model dir missing: {model_dir}"

    tok = AutoTokenizer.from_pretrained(
        model_dir,
        local_files_only=True,
    )

    mdl = AutoModelForTokenClassification.from_pretrained(
        model_dir,
        local_files_only=True,
    )

    mdl = gpu.prepare_model(mdl)

    device = 0 if torch.cuda.is_available() else -1

    info_id(
        f"[{name}] HF pipeline device={device}",
        REQ_ID,
    )

    return pipeline(
        "ner",
        model=mdl,
        tokenizer=tok,
        aggregation_strategy="simple",
        device=device,
    )


# ============================================================
# Load BOTH models
# ============================================================

try:
    parts_nlp = load_ner_pipeline(
        Path(MODEL_TRAINING_PARTS_MODEL_DIR),
        name="PARTS",
    )

    drawings_nlp = load_ner_pipeline(
        Path(MODEL_TRAINING_DRAWINGS_MODEL_DIR),
        name="DRAWINGS",
    )

except Exception as e:
    error_id(f"Failed to load NER models: {e}", REQ_ID)
    sys.exit(1)

info_id("Both NER models loaded successfully", REQ_ID)


# ============================================================
# Interactive loop
# ============================================================

print("\n" + "=" * 90)
print("COMBINED PARTS + DRAWINGS NER INTERACTIVE PROBE")
print("Type a query and see what EACH model extracts")
print("Type 'exit' or 'quit' to stop")
print("=" * 90)

while True:
    try:
        query = input("\n🔎 Query> ").strip()
    except (EOFError, KeyboardInterrupt):
        print("\nExiting.")
        break

    if not query:
        continue

    if query.lower() in {"exit", "quit"}:
        break

    print("\n" + "-" * 90)
    print(f"QUERY: {query}")
    print("-" * 90)

    # -----------------------------
    # PARTS NER
    # -----------------------------
    parts_preds = parts_nlp(query)
    parts_summary = summarize(parts_preds)

    print("\n[ PARTS NER ]")
    if not parts_summary:
        print("  (no entities)")
    else:
        for lbl, data in parts_summary.items():
            print(
                f"  {lbl:<18} → '{data['value']}' "
                f"(avg_conf={data['avg_conf']}, tokens={data['count']})"
            )

    # -----------------------------
    # DRAWINGS NER
    # -----------------------------
    drawings_preds = drawings_nlp(query)
    drawings_summary = summarize(drawings_preds)

    print("\n[ DRAWINGS NER ]")
    if not drawings_summary:
        print("  (no entities)")
    else:
        for lbl, data in drawings_summary.items():
            print(
                f"  {lbl:<18} → '{data['value']}' "
                f"(avg_conf={data['avg_conf']}, tokens={data['count']})"
            )

    # -----------------------------
    # Routing insight
    # -----------------------------
    print("\n[ ROUTING INSIGHT ]")

    if parts_summary and drawings_summary:
        print("  ⚠ BOTH models detected entities")
    elif parts_summary:
        print("  ✔ Parts model detected entities")
    elif drawings_summary:
        print("  ✔ Drawings model detected entities")
    else:
        print("  ❌ No entities detected by either model")

    print("\n" + "-" * 90)
