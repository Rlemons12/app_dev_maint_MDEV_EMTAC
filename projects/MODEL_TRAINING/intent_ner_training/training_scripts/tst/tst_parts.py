import os
import re
import logging
from pathlib import Path

from transformers import (
    AutoTokenizer,
    AutoModelForTokenClassification,
    pipeline,
)
from modules.configuration.config import ORC_PARTS_MODEL_DIR

# -------------------------------------------------------------------
# Model directory resolution (LATEST/DEPLOYED/newest/best/override)
# -------------------------------------------------------------------
RUN_RE = re.compile(r"^\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2}_run-\d{3}$")

def resolve_model_dir(base_dir: Path, prefer_deployed: bool = True) -> Path:
    """
    Resolve the directory to load:
      0) env PARTS_NER_MODEL_DIR (if set)
      1) DEPLOYED.txt -> <run>/best (pinned prod)
      2) LATEST.txt   -> <run>/best (most recent training)
      3) newest timestamped run with best/
      4) base/best
      5) base (direct files)
    """
    # 0) env override
    override = os.getenv("PARTS_NER_MODEL_DIR")
    if override:
        p = Path(override)
        if p.is_dir():
            return (p / "best") if (p / "best").is_dir() else p

    # 1) pinned deployment
    if prefer_deployed and (base_dir / "DEPLOYED.txt").exists():
        name = (base_dir / "DEPLOYED.txt").read_text(encoding="utf-8").strip()
        cand = base_dir / name
        if (cand / "best").is_dir():
            return cand / "best"

    # 2) latest training
    if (base_dir / "LATEST.txt").exists():
        name = (base_dir / "LATEST.txt").read_text(encoding="utf-8").strip()
        cand = base_dir / name
        if (cand / "best").is_dir():
            return cand / "best"

    # 3) newest timestamped run with best/
    runs = [p for p in base_dir.iterdir() if p.is_dir() and RUN_RE.match(p.name)]
    runs.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    for r in runs:
        if (r / "best").is_dir():
            return r / "best"

    # 4) non-versioned best/
    if (base_dir / "best").is_dir():
        return base_dir / "best"

    # 5) last resort: base itself
    return base_dir

# -------------------------------------------------------------------
# Logging setup
# -------------------------------------------------------------------
logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

# -------------------------------------------------------------------
# Basic rules / helpers
# -------------------------------------------------------------------

# Hard rule: PART_NUMBER is always "A1" + 5 digits (case-insensitive)
PART_NUMBER_RE = re.compile(r"\bA1\d{5}\b", re.IGNORECASE)

# Very small gazetteer to nudge obvious brands into MANUFACTURER
KNOWN_BRANDS = {
    "BANNER", "ASHCROFT", "JOY", "DOLLINGER", "PARKER", "BALSTON",
    "SMC", "SKF", "FISHER", "ALLEN-BRADLEY", "ABB", "OMRON", "EMERSON",
    "ROSEMOUNT", "EATON", "BURKERT", "TURCK", "SICK"
}

# Console colors (optional). Added "PART" for single-entity models.
RESET = "\x1b[0m"
COLORS = {
    "PART_NUMBER": "\x1b[1;35m",   # magenta
    "MODEL": "\x1b[1;36m",         # cyan
    "MANUFACTURER": "\x1b[1;33m",  # yellow
    "PART_NAME": "\x1b[1;32m",     # green
    "DOC": "\x1b[1;34m",           # blue
    "PART": "\x1b[1;35m",          # magenta (aggregated B-PART/I-PART)
}

def enforce_part_number_rule(text, ents):
    """Override/insert spans that match A1##### as PART_NUMBER."""
    matches = [(m.start(), m.end(), m.group(0)) for m in PART_NUMBER_RE.finditer(text)]
    if not matches:
        return ents

    # Drop any entity that overlaps a hard-rule span
    kept = []
    for e in ents:
        if any(not (e["end"] <= s or e["start"] >= t) for (s, t, _) in matches):
            continue
        kept.append(e)

    # Add hard-rule entities
    for s, t, word in matches:
        kept.append({
            "start": s,
            "end": t,
            "word": word,
            "entity_group": "PART_NUMBER",
            "score": 0.999
        })

    # Sort by start
    kept.sort(key=lambda r: r["start"])
    return kept

def brand_gazetteer_fix(text, ents):
    """
    If a single-token span matches a known brand and wasn't labeled MANUFACTURER,
    nudge it to MANUFACTURER (unless it's clearly part of a longer part-name).
    """
    fixed = []
    for r in ents:
        word_up = r["word"].strip().upper()
        if r["entity_group"] != "MANUFACTURER" and " " not in word_up and word_up in KNOWN_BRANDS:
            # If it was PART_NAME but very short, prefer MANUFACTURER
            if r["entity_group"] == "PART_NAME":
                span_text = text[r["start"]:r["end"]].strip()
                if len(span_text.split()) <= 2:
                    r = {**r, "entity_group": "MANUFACTURER", "score": max(r["score"], 0.99)}
            else:
                r = {**r, "entity_group": "MANUFACTURER", "score": max(r["score"], 0.99)}
        fixed.append(r)
    return fixed

def highlight_text(text, entities):
    """Insert color tags into the original text without breaking indices."""
    if not entities:
        return text
    entities = sorted(entities, key=lambda e: e["start"])
    offset = 0
    out = text
    for ent in entities:
        label = ent["entity_group"]
        color = COLORS.get(label, "")
        start = ent["start"] + offset
        end = ent["end"] + offset
        segment = out[start:end]
        tagged = f"{color}{segment}{RESET}" if color else segment
        out = out[:start] + tagged + out[end:]
        offset += len(tagged) - len(segment)
    return out

def fmt_entities(ents):
    """Group by label and print nicely. Added 'PART' for single-entity models."""
    if not ents:
        return "  (no entities)\n"

    groups = {}
    for r in ents:
        groups.setdefault(r["entity_group"], []).append(r)

    lines = []
    for label in ["PART_NUMBER", "MODEL", "MANUFACTURER", "PART_NAME", "DOC", "PART"]:
        items = groups.get(label, [])
        if not items:
            continue
        lines.append(f"\n  {label}:")
        for r in items:
            lines.append(f"    • {r['word']} (confidence: {r['score']:.4f})")
    return "\n".join(lines) + "\n"

# -------------------------------------------------------------------
# Main
# -------------------------------------------------------------------
def main():
    base = Path(ORC_PARTS_MODEL_DIR)
    model_dir = resolve_model_dir(base, prefer_deployed=True)
    log.info(f"Loading Parts NER model from: {model_dir}")

    tokenizer = AutoTokenizer.from_pretrained(str(model_dir))
    model = AutoModelForTokenClassification.from_pretrained(str(model_dir))
    nlp = pipeline(
        "token-classification",
        model=model,
        tokenizer=tokenizer,
        aggregation_strategy="simple",  # grouped spans w/ start/end
        device=-1,  # CPU
    )
    log.info("Model loaded successfully!")
    print("Device set to use CPU\n")

    print("=" * 60)
    print("Parts NER Model Testing Interface")
    print("=" * 60)
    print("Enter text containing parts information to extract entities.")
    print("Type 'exit' or 'quit' to stop.")
    print("=" * 60, "\n")

    while True:
        try:
            text = input("Enter text: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not text or text.lower() in {"exit", "quit"}:
            break

        # Run NER
        results = nlp(text)

        # Enforce the A1##### hard rule and apply brand nudge
        results = enforce_part_number_rule(text, results)
        results = brand_gazetteer_fix(text, results)

        # Print results
        print("\nFound entities:\n")
        print(fmt_entities(results))

        print("Highlighted text:")
        print(highlight_text(text, results), "\n")

    print("Bye!")

if __name__ == "__main__":
    main()
