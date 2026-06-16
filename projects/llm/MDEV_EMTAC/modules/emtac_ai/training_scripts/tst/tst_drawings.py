
# --- path bootstrap so 'modules' is importable when run as a script ---
import sys, pathlib
_PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[4]  # ...\modules\emtac_ai\training_scripts\tst\ -> go up 4 to repo root
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))
# ---------------------------------------------------------------------


import os
import re
import sys
import json
import argparse
from typing import List, Dict, Any, Tuple, Optional

import math
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForTokenClassification,
    pipeline,
)

# --- Project config (model dir) ---
from modules.emtac_ai.config import ORC_DRAWINGS_MODEL_DIR

# --- Your custom logger ---
from modules.configuration.log_config import (
    debug_id, info_id, warning_id, error_id, critical_id,
    with_request_id, log_timed_operation,
    set_request_id, get_request_id
)

# -----------------------------
# Defaults & Console Colors
# -----------------------------
DEFAULT_MODEL_DIR = ORC_DRAWINGS_MODEL_DIR
RESET = "\x1b[0m"
COLORS = {
    "EQUIPMENT_NUMBER": "\x1b[1;35m",
    "EQUIPMENT_NAME": "\x1b[1;36m",
    "DRAWING_NUMBER": "\x1b[1;33m",
    "DRAWING_NAME": "\x1b[1;32m",
    "SPARE_PART_NUMBER": "\x1b[1;34m",
}

# -----------------------------
# Hard rules / heuristics
# -----------------------------
EQUIPMENT_NUMBER_RE = re.compile(r"\b[A-Z]{1,3}-?\d{3,6}\b", re.IGNORECASE)  # E-1001, AFL12100, P-205
DRAWING_NUMBER_RE   = re.compile(r"\b(?:DWG|DRW)[-_]?\d{3,7}\b", re.IGNORECASE)  # DWG-12345
SPARE_PART_RE       = re.compile(r"\b[A-Z]\d{5,7}\b", re.IGNORECASE)  # A123456

KNOWN_EQUIPMENT_TYPES = {
    "PUMP", "HEAT EXCHANGER", "CENTRIFUGAL PUMP", "COMPRESSOR", "VESSEL",
    "TANK", "REACTOR", "COLUMN", "SEPARATOR", "FILTER", "VALVE", "MOTOR",
    "TURBINE", "BOILER", "COOLER", "HEATER", "EXCHANGER"
}

def enforce_hard_rules(text: str, ents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    equipment_matches = [(m.start(), m.end(), m.group(0)) for m in EQUIPMENT_NUMBER_RE.finditer(text)]
    drawing_matches   = [(m.start(), m.end(), m.group(0)) for m in DRAWING_NUMBER_RE.finditer(text)]
    spare_part_matches= [(m.start(), m.end(), m.group(0)) for m in SPARE_PART_RE.finditer(text)]

    all_matches = (
        [(s, e, w, "EQUIPMENT_NUMBER") for s, e, w in equipment_matches] +
        [(s, e, w, "DRAWING_NUMBER")   for s, e, w in drawing_matches] +
        [(s, e, w, "SPARE_PART_NUMBER")for s, e, w in spare_part_matches]
    )
    if not all_matches:
        return ents

    kept = []
    for e in ents:
        if any(not (e["end"] <= s or e["start"] >= t) for (s, t, _, _) in all_matches):
            # Overlaps a hard rule span -> drop model span
            continue
        kept.append(e)

    for s, t, word, label in all_matches:
        kept.append({"start": s, "end": t, "word": word, "entity_group": label, "score": 0.999})

    kept.sort(key=lambda r: r["start"])
    return kept

def equipment_type_fix(text: str, ents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    fixed = []
    for r in ents:
        word_up = r["word"].strip().upper()
        if r["entity_group"] != "EQUIPMENT_NAME" and word_up in KNOWN_EQUIPMENT_TYPES:
            r = {**r, "entity_group": "EQUIPMENT_NAME", "score": max(r.get("score", 0.0), 0.99)}
        fixed.append(r)
    return fixed

# -----------------------------
# Presentation helpers
# -----------------------------
def highlight_text(text: str, entities: List[Dict[str, Any]], use_color: bool = True) -> str:
    if not entities:
        return text
    entities = sorted(entities, key=lambda e: e["start"])
    offset = 0
    out = text
    for ent in entities:
        label = ent["entity_group"]
        color = COLORS.get(label, "") if use_color else ""
        start = max(0, min(len(out), ent["start"] + offset))
        end   = max(0, min(len(out), ent["end"] + offset))
        if end <= start:
            continue
        segment = out[start:end]
        tagged = f"{color}{segment}{RESET}" if (use_color and color) else f"[{label}:{segment}]"
        out = out[:start] + tagged + out[end:]
        offset += len(tagged) - len(segment)
    return out

def tabulate_entities(ents: List[Dict[str, Any]], min_conf: float = 0.0) -> str:
    visible = [e for e in ents if e.get("score", 0.0) >= min_conf]
    if not visible:
        return f"  (no entities at min_conf {min_conf:.2f})\n"
    groups = {}
    for r in visible:
        groups.setdefault(r["entity_group"], []).append(r)
    lines = []
    order = ["EQUIPMENT_NUMBER", "EQUIPMENT_NAME", "DRAWING_NUMBER", "DRAWING_NAME", "SPARE_PART_NUMBER"]
    for label in order:
        items = groups.get(label, [])
        if not items:
            continue
        lines.append(f"\n  {label}:")
        for r in items:
            score = r.get("score", 0.0)
            lines.append(f"    • {r['word']} (conf: {score:.4f}) [{r['start']}:{r['end']}]")
    return "\n".join(lines) + "\n"

def json_entities(ents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return [
        {
            "label": e.get("entity_group"),
            "text": e.get("word"),
            "start": int(e.get("start")),
            "end": int(e.get("end")),
            "score": float(e.get("score", 0.0)),
        }
        for e in ents
    ]

# -----------------------------
# Token-level inspection
# -----------------------------
def softmax_np(x: torch.Tensor) -> torch.Tensor:
    x = x - x.max(dim=-1, keepdim=True).values
    return torch.exp(x) / torch.exp(x).sum(dim=-1, keepdim=True)

def token_debug_table(tokens: List[str], ids: List[int], probs: torch.Tensor, id2label: Dict[int, str], topk: int) -> str:
    lines = []
    for idx, tok in enumerate(tokens):
        p = probs[idx]  # [num_labels]
        top_vals, top_idx = torch.topk(p, k=min(topk, p.shape[-1]))
        tops = [f"{id2label[int(i)]}:{float(v):.3f}" for v, i in zip(top_vals, top_idx)]
        lines.append(f"  {idx:>3}  {tok:<15}  " + " | ".join(tops))
    return "\n".join(lines)

def run_token_level(
    model, tokenizer, text: str, device_idx: int, topk: int = 3
) -> Tuple[List[str], List[int], torch.Tensor, Dict[int, str]]:
    encoded = tokenizer(
        text,
        return_tensors="pt",
        return_offsets_mapping=True,
        truncation=True,
        max_length=512,
    )
    if device_idx >= 0:
        encoded = {k: v.to(device_idx) for k, v in encoded.items()}
        model = model.to(device_idx)

    with torch.no_grad():
        logits = model(**{k: v for k, v in encoded.items() if k in ("input_ids", "attention_mask", "token_type_ids")}).logits[0]
        probs = softmax_np(logits).cpu()

    tokens = tokenizer.convert_ids_to_tokens(encoded["input_ids"][0].cpu().tolist())
    ids = torch.argmax(probs, dim=-1).cpu().tolist()
    id2label = model.config.id2label
    return tokens, ids, probs, id2label

# -----------------------------
# CLI / Runner
# -----------------------------
def parse_args():
    p = argparse.ArgumentParser(description="Drawings NER Model Testing (rich logging + token inspection)")
    p.add_argument("--model-dir", type=str, default=DEFAULT_MODEL_DIR, help="Path to model directory (overrides config)")
    p.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda", "auto"], help="Inference device")
    p.add_argument("--log-level", type=str, default="INFO",
                   choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"], help="Logger verbosity")
    p.add_argument("--no-color", action="store_true", help="Disable ANSI color in highlights")
    p.add_argument("--show-raw", action="store_true", help="Log raw pipeline spans (pre-heuristics)")
    p.add_argument("--json", action="store_true", help="Emit JSON result lines instead of pretty text")
    p.add_argument("--min-conf", type=float, default=0.0, help="Minimum confidence to display in pretty table")
    p.add_argument("--dump-tokens", action="store_true", help="Show per-token top-K label probs")
    p.add_argument("--topk", type=int, default=3, help="Top-K labels to show per token when --dump-tokens")
    p.add_argument("--file", type=str, help="Batch mode: path to a file of queries (one per line)")
    p.add_argument("--keep-case", action="store_true", help="Do not lowercase in logs (visual only; model input unchanged)")
    return p.parse_args()

def resolve_device(arg: str) -> int:
    if arg == "cpu":
        return -1
    if arg == "cuda":
        try:
            return 0 if torch.cuda.is_available() else -1
        except Exception:
            return -1
    if arg == "auto":
        try:
            return 0 if torch.cuda.is_available() else -1
        except Exception:
            return -1
    return -1

@with_request_id
def load_pipeline(model_dir: str, device_flag: str):
    request_id = get_request_id()
    device_index = resolve_device(device_flag)

    with log_timed_operation("load_model_and_tokenizer", request_id):
        info_id(f"Loading model from: {model_dir}", request_id)
        tokenizer = AutoTokenizer.from_pretrained(model_dir)
        model = AutoModelForTokenClassification.from_pretrained(model_dir)
        try:
            id2label = model.config.id2label
            label_set = sorted({v for v in id2label.values()})
            info_id(f"Label map: {id2label}", request_id)
            info_id(f"Label set (sorted): {label_set}", request_id)
        except Exception:
            warning_id("Could not read id2label from model.config", request_id)
        info_id(f"Tokenizer vocab size: {getattr(tokenizer, 'vocab_size', 'unknown')}", request_id)

    with log_timed_operation("build_pipeline", request_id):
        nlp = pipeline(
            "token-classification",
            model=model,
            tokenizer=tokenizer,
            aggregation_strategy="simple",
            device=device_index,
        )
        info_id(f"Pipeline ready (aggregation=simple, device={device_flag}, index={device_index})", request_id)

    return nlp, model, tokenizer, device_index

def run_inference(nlp, text: str, show_raw: bool = False) -> List[Dict[str, Any]]:
    request_id = get_request_id()
    with log_timed_operation("inference", request_id):
        raw = nlp(text)
        if show_raw:
            debug_id(f"RAW spans: {raw}", request_id)

    # Sanity: clip spans to text bounds
    for r in raw:
        if r["start"] < 0 or r["end"] > len(text) or r["start"] >= r["end"]:
            warning_id(f"Span out of bounds; clipped {r}", request_id)
            r["start"] = max(0, min(len(text), r["start"]))
            r["end"] = max(r["start"], min(len(text), r["end"]))
            r["word"] = text[r["start"]:r["end"]]

    with log_timed_operation("postprocess_hard_rules", request_id):
        results = enforce_hard_rules(text, raw)

    with log_timed_operation("postprocess_equipment_type_fix", request_id):
        results = equipment_type_fix(text, results)

    return results

def pretty_emit(text: str, ents: List[Dict[str, Any]], min_conf: float, use_color: bool, emit_json: bool):
    if emit_json:
        print(json.dumps({"text": text, "entities": json_entities(ents)}, ensure_ascii=False))
        return
    print("\nFound entities:\n")
    print(tabulate_entities(ents, min_conf=min_conf))
    print("Highlighted text:")
    print(highlight_text(text, ents, use_color=use_color), "\n")

def do_one(
    nlp, model, tokenizer, device_idx: int, q: str,
    args
):
    request_id = get_request_id()
    raw_for_log = q if args.keep_case else q.lower()
    info_id(f"User query: {raw_for_log}", request_id)

    ents = run_inference(nlp, q, show_raw=args.show_raw)
    debug_id(f"Entities (compact): {json_entities(ents)}", request_id)

    if args.dump_tokens:
        try:
            tokens, ids, probs, id2label = run_token_level(model, tokenizer, q, device_idx, topk=args.topk)
            info_id("Per-token top-K label probabilities:", request_id)
            print(token_debug_table(tokens, ids, probs, id2label, args.topk))
        except Exception as e:
            warning_id(f"Token-level dump failed: {e}", request_id)

    pretty_emit(q, ents, min_conf=args.min_conf, use_color=(not args.no_color), emit_json=args.json)

@with_request_id
def interactive_loop(nlp, model, tokenizer, device_idx: int, args):
    request_id = get_request_id()
    print("=" * 68)
    print("Drawings NER Model Testing Interface (with expanded debugging)")
    print("=" * 68)
    print("Type a query, or 'exit' / 'quit' to stop.\n")
    print("Examples:")
    print("• I need the print for equipment E-1001")
    print("• Show me the centrifugal pump assembly drawing")
    print("• Find drawing DWG-12345 for the heat exchanger")
    print("• Where is part A123456 on the schematic?\n")

    while True:
        try:
            text = input("Enter text: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            info_id("Input interrupted — exiting.", request_id)
            break
        if not text:
            continue
        if text.lower() in {"exit", "quit"}:
            info_id("Exit requested by user.", request_id)
            break

        try:
            do_one(nlp, model, tokenizer, device_idx, text, args)
        except Exception as e:
            error_id(f"Inference error: {e}", request_id)
            critical_id("Continuing interactive session after error.", request_id)

def run_batch(nlp, model, tokenizer, device_idx: int, args):
    request_id = get_request_id()
    path = args.file
    if not os.path.exists(path):
        error_id(f"Batch file not found: {path}", request_id)
        return
    info_id(f"Running batch from file: {path}", request_id)

    with open(path, "r", encoding="utf-8") as f:
        for line_num, q in enumerate(f, start=1):
            q = q.strip()
            if not q:
                continue
            info_id(f"[Batch #{line_num}] >>> {q}", request_id)
            try:
                do_one(nlp, model, tokenizer, device_idx, q, args)
            except Exception as e:
                error_id(f"[Batch #{line_num}] error: {e}", request_id)

@with_request_id
def main():
    args = parse_args()
    os.environ["LOG_LEVEL"] = args.log_level  # let your logger pick this up
    request_id = get_request_id()
    debug_id("Starting main", request_id)
    info_id(f"=== Drawings NER Test Start (REQ {request_id}) ===", request_id)
    info_id(
        "Args: model_dir='%s', device='%s', log_level='%s', no_color=%s, show_raw=%s, json=%s, "
        "min_conf=%.2f, dump_tokens=%s, topk=%d, file=%s, keep_case=%s"
        % (
            args.model_dir, args.device, args.log_level, args.no_color, args.show_raw, args.json,
            args.min_conf, args.dump_tokens, args.topk, args.file, args.keep_case
        ),
        request_id
    )

    # Load pipeline + model
    nlp, model, tokenizer, device_idx = load_pipeline(args.model_dir, args.device)
    device_str = "CPU" if device_idx == -1 else "CUDA"
    print(f"\nDevice set to use {device_str}\n")

    if args.file:
        run_batch(nlp, model, tokenizer, device_idx, args)
    else:
        interactive_loop(nlp, model, tokenizer, device_idx, args)

    info_id(f"=== Drawings NER Test End (REQ {request_id}) ===", request_id)

if __name__ == "__main__":
    set_request_id()
    try:
        main()
    finally:
        pass
