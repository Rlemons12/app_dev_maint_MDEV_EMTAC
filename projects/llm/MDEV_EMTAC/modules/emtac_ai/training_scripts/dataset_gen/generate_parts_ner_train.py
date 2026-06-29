import argparse
import json
import logging
import os
import re
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple, TextIO

import pandas as pd

# =========================
# Label space (final)
# =========================
LABELS = [
    "O",
    "B-PART_NUMBER", "I-PART_NUMBER",
    "B-PART_NAME", "I-PART_NAME",
    "B-MANUFACTURER", "I-MANUFACTURER",
    "B-MODEL", "I-MODEL",
]
ID2LABEL = {i: lab for i, lab in enumerate(LABELS)}
LABEL2ID = {lab: i for i, lab in enumerate(LABELS)}

# =========================
# Tokenization helpers
# =========================
TOKEN_RE = re.compile(r"\w+|\S", re.UNICODE)
def tokenize(text: str) -> List[str]:
    return TOKEN_RE.findall(text)

def find_sublist_indices(tokens: List[str], sub: List[str]) -> Optional[Tuple[int, int]]:
    n, m = len(tokens), len(sub)
    if m == 0 or m > n:
        return None
    for i in range(n - m + 1):
        if tokens[i:i+m] == sub:
            return i, i+m
    return None

def _safe(row: pd.Series, col: str) -> str:
    v = row.get(col, "")
    if pd.isna(v):
        return ""
    return str(v).strip()

# =========================
# Runtime-loaded data
# =========================
# Will be populated at runtime from TXT files:
PARTS_NATURAL_LANGUAGE_VARIATIONS: Dict[str, Dict[str, List[str]]] = {}
PARTS_ENHANCED_QUERY_TEMPLATES: List[str] = []

# Placeholders expected in templates/variations
PLACEHOLDERS = {"itemnum", "description", "manufacturer", "model"}
PH_FMT = re.compile(r"\{([a-zA-Z0-9_]+)\}")

# =========================
# TXT loaders
# =========================
def _load_templates_txt(path: Path) -> List[str]:
    """
    Expects ONE template per line, e.g.:
      PN {itemnum}
      I need {description_casual} from {manufacturer_formal}
      Do you carry {model_contextual}?
    Lines starting with '#' are ignored.
    """
    if not path.exists():
        logging.warning("Templates TXT not found: %s", path)
        return []
    lines = [ln.strip() for ln in path.read_text(encoding="utf-8").splitlines()]
    return [ln for ln in lines if ln and not ln.startswith("#")]

def _load_variations_txt(path: Path) -> Dict[str, Dict[str, List[str]]]:
    """
    Accepts either:
      A) Python/JSON dict file like:
         PARTS_NATURAL_LANGUAGE_VARIATIONS = {
           "ITEMNUM": {"formal": ["part {itemnum}", ...], "casual": [...], "contextual": [...]},
           "DESCRIPTION": {...}, "OEMMFG": {...}, "MODEL": {...}
         }
      B) TXT line format:
         ITEMNUM.formal: part number {itemnum} | SKU {itemnum}
         ITEMNUM.casual: part {itemnum} | item# {itemnum}
         ITEMNUM.contextual: the {itemnum} part | piece numbered {itemnum}
    """
    out: Dict[str, Dict[str, List[str]]] = {}
    if not path.exists():
        logging.warning("Variations file not found: %s", path)
        return out

    raw = path.read_text(encoding="utf-8").strip()

    # --- Try to parse as JSON/Python dict first ---
    try:
        import ast, json as _json

        text = raw
        # Strip assignment prefix like "PARTS_NATURAL_LANGUAGE_VARIATIONS = {...}"
        if "=" in text:
            lhs, rhs = text.split("=", 1)
            # only strip if LHS looks like a var name
            if lhs.strip().replace("_", "").isalpha():
                text = rhs.strip()

        # Try JSON, then Python literal
        try:
            obj = _json.loads(text)
        except Exception:
            obj = ast.literal_eval(text)

        # Expect dict[str -> dict[str -> list[str]]]
        if isinstance(obj, dict):
            for entity_key, buckets in obj.items():
                if not isinstance(buckets, dict):
                    continue
                ekey = str(entity_key).strip().upper()
                out.setdefault(ekey, {})
                for bucket_key, patterns in buckets.items():
                    bkey = str(bucket_key).strip().lower()
                    if isinstance(patterns, list):
                        pats = [str(p).strip() for p in patterns if str(p).strip()]
                        out[ekey][bkey] = pats
            return out
    except Exception:
        # fall through to line parser
        pass

    # --- Fallback: parse TXT "KEY.bucket: pat | pat" ---
    for line in raw.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if ":" not in line or "." not in line.split(":", 1)[0]:
            logging.warning("Bad variations line (ignored): %s", line)
            continue
        head, rhs = line.split(":", 1)
        head = head.strip()
        rhs = rhs.strip()
        try:
            entity_key, bucket = head.split(".", 1)
            entity_key = entity_key.strip().upper()
            bucket = bucket.strip().lower()
        except Exception:
            logging.warning("Bad variations key (ignored): %s", head)
            continue
        pats = [p.strip() for p in rhs.split("|") if p.strip()]
        if pats:
            out.setdefault(entity_key, {}).setdefault(bucket, []).extend(pats)

    return out


def _ensure_minimum_defaults(variations: Dict[str, Dict[str, List[str]]]):
    """Make sure we have at least 1 pattern per (entity, bucket)."""
    def ensure(entity: str, bucket: str, default: List[str]):
        variations.setdefault(entity, {}).setdefault(bucket, [])
        if not variations[entity][bucket]:
            variations[entity][bucket] = default

    ensure("ITEMNUM", "formal",     ["part number {itemnum}"])
    ensure("ITEMNUM", "casual",     ["part {itemnum}"])
    ensure("ITEMNUM", "contextual", ["the {itemnum} part"])

    ensure("DESCRIPTION", "formal",     ["{description}"])
    ensure("DESCRIPTION", "casual",     ["some {description}"])
    ensure("DESCRIPTION", "contextual", ["{description} or similar"])

    ensure("OEMMFG", "formal",     ["manufactured by {manufacturer}"])
    ensure("OEMMFG", "casual",     ["{manufacturer} parts"])
    ensure("OEMMFG", "contextual", ["something from {manufacturer}"])

    ensure("MODEL", "formal",     ["OEM part {model}"])
    ensure("MODEL", "casual",     ["part {model}"])
    ensure("MODEL", "contextual", ["compatible with {model}"])

# =========================
# NL augmentation helpers
# =========================
def sample_slot_variations(itemnum: str, description: str, manufacturer: str, model: str,
                           rng: random.Random,
                           variations: Dict[str, Dict[str, List[str]]]) -> Dict:
    def pick(entity_key: str, bucket: str, **fmt):
        patt = rng.choice(variations[entity_key][bucket])
        return patt.format(**fmt)

    # lowercase for more natural surfaces in desc/mfg
    desc_surf = description.lower() if description else ""
    mfg_surf  = manufacturer.lower() if manufacturer else ""

    return {
        "itemnum": {
            "formal":     pick("ITEMNUM", "formal",     itemnum=itemnum) if itemnum else "",
            "casual":     pick("ITEMNUM", "casual",     itemnum=itemnum) if itemnum else "",
            "contextual": pick("ITEMNUM", "contextual", itemnum=itemnum) if itemnum else "",
        },
        "description": {
            "formal":     pick("DESCRIPTION", "formal",     description=desc_surf) if desc_surf else "",
            "casual":     pick("DESCRIPTION", "casual",     description=desc_surf) if desc_surf else "",
            "contextual": pick("DESCRIPTION", "contextual", description=desc_surf) if desc_surf else "",
        },
        "manufacturer": {
            "formal":     pick("OEMMFG", "formal",     manufacturer=mfg_surf) if mfg_surf else "",
            "casual":     pick("OEMMFG", "casual",     manufacturer=mfg_surf) if mfg_surf else "",
            "contextual": pick("OEMMFG", "contextual", manufacturer=mfg_surf) if mfg_surf else "",
        },
        "model": {
            "formal":     pick("MODEL", "formal",     model=model) if model else "",
            "casual":     pick("MODEL", "casual",     model=model) if model else "",
            "contextual": pick("MODEL", "contextual", model=model) if model else "",
        },
    }

def build_augmented_sentence(row: pd.Series, template: str, rng: random.Random,
                             variations: Dict[str, Dict[str, List[str]]]) -> Tuple[str, Dict[str, str]]:
    # Pull raw values
    itemnum      = _safe(row, "ITEMNUM")
    description  = _safe(row, "DESCRIPTION")
    manufacturer = _safe(row, "OEMMFG")
    model        = _safe(row, "MODEL")

    # sample surfaces
    slots = sample_slot_variations(itemnum, description, manufacturer, model, rng, variations)

    # formatter provides both raw and variant keys
    fmt = {
        "itemnum": itemnum,
        "description": description,
        "manufacturer": manufacturer,
        "model": model,

        "itemnum_formal":       slots["itemnum"]["formal"],
        "itemnum_casual":       slots["itemnum"]["casual"],
        "itemnum_contextual":   slots["itemnum"]["contextual"],

        "description_formal":     slots["description"]["formal"],
        "description_casual":     slots["description"]["casual"],
        "description_contextual": slots["description"]["contextual"],

        "manufacturer_formal":     slots["manufacturer"]["formal"],
        "manufacturer_casual":     slots["manufacturer"]["casual"],
        "manufacturer_contextual": slots["manufacturer"]["contextual"],

        "model_formal":     slots["model"]["formal"],
        "model_casual":     slots["model"]["casual"],
        "model_contextual": slots["model"]["contextual"],
    }

    sentence = template.format(**fmt)

    # Entity map for tagging (canonical/raw values only)
    ent_surface: Dict[str, str] = {}
    if itemnum:
        ent_surface["PART_NUMBER"] = itemnum
    if description:
        ent_surface["PART_NAME"] = description.lower()
    if manufacturer:
        ent_surface["MANUFACTURER"] = manufacturer.lower()
    if model:
        ent_surface["MODEL"] = model

    return sentence, ent_surface

def tag_tokens_surface(sentence: str, entity_surface_map: Dict[str, str]) -> Tuple[List[str], List[int]]:
    tokens = tokenize(sentence)
    tags = ["O"] * len(tokens)
    for ent, surface in entity_surface_map.items():
        if not surface:
            continue
        ent_tokens = tokenize(surface)
        span = find_sublist_indices(tokens, ent_tokens)
        if not span:
            continue
        s, e = span
        tags[s] = f"B-{ent}"
        for i in range(s+1, e):
            tags[i] = f"I-{ent}"
    ner_ids = [LABEL2ID.get(t, 0) for t in tags]
    return tokens, ner_ids

def build_row_sentence(row: pd.Series) -> str:
    pn  = _safe(row, "ITEMNUM")
    nm  = _safe(row, "DESCRIPTION")
    mfg = _safe(row, "OEMMFG")
    mdl = _safe(row, "MODEL")
    parts = []
    if pn:  parts.append(f"part number {pn}")
    if nm:  parts.append(f"name {nm}")
    if mfg: parts.append(f"made by {mfg}")
    if mdl: parts.append(f"model {mdl}")
    return "This is " + ", ".join(parts) + "." if parts else "This is a part."

def build_row_sentence_alt(row: pd.Series) -> str:
    pn  = _safe(row, "ITEMNUM")
    nm  = _safe(row, "DESCRIPTION")
    mfg = _safe(row, "OEMMFG")
    mdl = _safe(row, "MODEL")
    primary = []
    if nm:  primary.append(f"name {nm}")
    if pn:  primary.append(f"part number {pn}")
    if mdl: primary.append(f"model {mdl}")
    if mfg: primary.append(f"made by {mfg}")
    return "Part details: " + ", ".join(primary) + "." if primary else "Part details unavailable."

# =========================
# JSONL writer helpers
# =========================
def write_ner_row(f, sentence: str, ent_surface_map: Dict[str, str]):
    tokens, ner_ids = tag_tokens_surface(sentence, ent_surface_map)
    f.write(json.dumps({"tokens": tokens, "ner_tags": ner_ids}, ensure_ascii=False) + "\n")

def write_intent_row(f: TextIO, text: str):
    f.write(json.dumps({"text": text, "label": "parts"}, ensure_ascii=False) + "\n")

# =========================
# Core generator
# =========================
def generate_ner_training_file(
    excel_path: str,
    out_hf_path: str,
    sheet_name: Optional[str] = None,
    augment_basic: bool = False,
    augment_nl: bool = False,
    seed: int = 42,
    emit_intents: bool = False,
    intent_out_path: Optional[str] = None,
    templates: Optional[List[str]] = None,
    variations: Optional[Dict[str, Dict[str, List[str]]]] = None,
) -> int:
    rng = random.Random(seed)
    logging.info("Reading Excel: %s", excel_path)
    df = pd.read_excel(excel_path, sheet_name=sheet_name) if sheet_name else pd.read_excel(excel_path)
    df.columns = [str(c).strip() for c in df.columns]

    os.makedirs(os.path.dirname(out_hf_path), exist_ok=True)

    # Prepare intent file if requested
    intent_f: Optional[TextIO] = None
    if emit_intents:
        if not intent_out_path:
            raise ValueError("--emit-intents requires intent_out_path")
        os.makedirs(os.path.dirname(intent_out_path), exist_ok=True)
        intent_f = open(intent_out_path, "w", encoding="utf-8")

    # Loaded resources
    tpl_pool = templates or []
    var_map  = variations or {}

    # Safety: ensure minimum defaults for variations
    _ensure_minimum_defaults(var_map)

    rows_written = 0
    with open(out_hf_path, "w", encoding="utf-8") as f_hf:
        for _, row in df.iterrows():
            if augment_nl and tpl_pool:
                tmpl = rng.choice(tpl_pool)
                sentence, ent_map = build_augmented_sentence(row, tmpl, rng, var_map)
                write_ner_row(f_hf, sentence, ent_map)
                if intent_f:
                    write_intent_row(intent_f, sentence)
                rows_written += 1

                if augment_basic:
                    sentence2 = build_row_sentence_alt(row)
                    ent_map2 = {
                        "PART_NUMBER": _safe(row, "ITEMNUM"),
                        "PART_NAME": _safe(row, "DESCRIPTION").lower(),
                        "MANUFACTURER": _safe(row, "OEMMFG").lower(),
                        "MODEL": _safe(row, "MODEL"),
                    }
                    ent_map2 = {k: v for k, v in ent_map2.items() if v}
                    write_ner_row(f_hf, sentence2, ent_map2)
                    if intent_f:
                        write_intent_row(intent_f, sentence2)
                    rows_written += 1
            else:
                sentence = build_row_sentence(row)
                ent_map = {
                    "PART_NUMBER": _safe(row, "ITEMNUM"),
                    "PART_NAME": _safe(row, "DESCRIPTION"),
                    "MANUFACTURER": _safe(row, "OEMMFG"),
                    "MODEL": _safe(row, "MODEL"),
                }
                ent_map = {k: v for k, v in ent_map.items() if v}
                write_ner_row(f_hf, sentence, ent_map)
                if intent_f:
                    write_intent_row(intent_f, sentence)
                rows_written += 1

                if augment_basic:
                    sentence2 = build_row_sentence_alt(row)
                    ent_map2 = {
                        "PART_NUMBER": _safe(row, "ITEMNUM"),
                        "PART_NAME": _safe(row, "DESCRIPTION"),
                        "MANUFACTURER": _safe(row, "OEMMFG"),
                        "MODEL": _safe(row, "MODEL"),
                    }
                    ent_map2 = {k: v for k, v in ent_map2.items() if v}
                    write_ner_row(f_hf, sentence2, ent_map2)
                    if intent_f:
                        write_intent_row(intent_f, sentence2)
                    rows_written += 1

    if intent_f:
        intent_f.close()

    logging.info("Wrote HF token/BIO JSONL: %s (%d rows)", out_hf_path, rows_written)
    return rows_written

# =========================
# CLI + interactive prompts
# =========================
def parse_args() -> argparse.Namespace:
    # Import config here to avoid circular imports during package init
    from modules.configuration import config
    default_excel = os.path.join(config.ORC_TRAINING_DATA_LOADSHEET, "parts_loadsheet.xlsx")

    p = argparse.ArgumentParser(description="Generate Parts NER (and optional intent) training data from Excel + TXT templates.")
    p.add_argument("--excel", required=False, default=default_excel,
                   help=f"Path to parts loadsheet. Default: {default_excel}")
    p.add_argument("--sheet", default=None, help="Worksheet name (optional).")
    p.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])

    # NEW: control NL augmentation & templates/variations source
    p.add_argument("--use-nl", action="store_true",
                   help="Use natural-language templates loaded from TXT (recommended).")
    p.add_argument("--use-basic-aug", action="store_true",
                   help="Also emit a simple alternate phrasing per row.")
    p.add_argument("--templates-file", default=None,
                   help="Override path to PARTS_ENHANCED_QUERY_TEMPLATES.txt (optional).")
    p.add_argument("--variations-file", default=None,
                   help="Override path to PARTS_NATURAL_LANGUAGE_VARIATIONS.txt (optional).")

    # --- emit intents: default ON; disable via --no-emit-intents
    p.add_argument(
        "--emit-intents",
        dest="emit_intents",
        action="store_true",
        default=True,
        help="Write intent JSONL rows for the intent classifier (label='parts'). Default: ON."
    )
    p.add_argument(
        "--no-emit-intents",
        dest="emit_intents",
        action="store_false",
        help="Disable writing the intent JSONL."
    )

    # Optional explicit path for the intent JSONL
    p.add_argument(
        "--intent-out",
        default=None,
        help="Override path for intent JSONL. Default: <ORC_TRAINING_DATA_DIR>/intent_classifier/intent_train_parts.jsonl"
    )

    return p.parse_args()

def _discover_parts_template_paths(templates_override: Optional[str],
                                   variations_override: Optional[str]):
    """Resolve default TXT paths under ORC_QUERY_TEMPLATE_PARTS, unless overrides are given."""
    from modules.configuration import config
    base = Path(config.ORC_QUERY_TEMPLATE_PARTS).resolve()
    tpl = Path(templates_override) if templates_override else (base / "PARTS_ENHANCED_QUERY_TEMPLATES.txt")
    var = Path(variations_override) if variations_override else (base / "PARTS_NATURAL_LANGUAGE_VARIATIONS.txt")
    return tpl, var

def main():
    # Import config inside main as well (keeps module import side-effects minimal)
    from modules.configuration import config
    args = parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level), format="%(levelname)s:%(message)s")

    print("\n=== Parts NER/Intent Dataset Generator (TXT-driven) ===")

    # Resolve TXT sources
    templates_path, variations_path = _discover_parts_template_paths(args.templates_file, args.variations_file)
    logging.info("Templates TXT : %s", templates_path)
    logging.info("Variations TXT: %s", variations_path)

    # Load TXT data
    templates = _load_templates_txt(templates_path)
    variations = _load_variations_txt(variations_path)
    _ensure_minimum_defaults(variations)

    if args.use_nl and not templates:
        logging.error("No templates found but --use-nl was specified. Provide a valid TXT or disable --use-nl.")
        return

    # Intent path (if requested)
    intent_out = None
    if args.emit_intents:
        intent_out = os.path.join(config.ORC_TRAINING_DATA_DIR, "intent_classifier", "intent_train_parts.jsonl")
        if args.intent_out:
            intent_out = args.intent_out
        os.makedirs(os.path.dirname(intent_out), exist_ok=True)
        logging.info("Intent out     : %s", intent_out)

    # NER out
    out_hf = os.path.join(config.ORC_PARTS_TRAIN_DATA_DIR, "ner_train_parts.jsonl")
    os.makedirs(os.path.dirname(out_hf), exist_ok=True)
    logging.info("Excel          : %s", args.excel)
    logging.info("NER out        : %s", out_hf)
    logging.info("Use NL         : %s", args.use_nl)
    logging.info("Use basic aug  : %s", args.use_basic_aug)

    # Generate
    generate_ner_training_file(
        excel_path=args.excel,
        sheet_name=args.sheet,
        out_hf_path=out_hf,
        augment_basic=args.use_basic_aug,
        augment_nl=args.use_nl,
        seed=42,
        emit_intents=args.emit_intents,
        intent_out_path=intent_out,
        templates=templates,
        variations=variations,
    )

if __name__ == "__main__":
    main()
