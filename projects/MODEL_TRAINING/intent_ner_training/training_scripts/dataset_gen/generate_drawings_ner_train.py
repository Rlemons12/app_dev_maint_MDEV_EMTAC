import pandas as pd
import json
import os
import random
from pathlib import Path

from modules.configuration.config import (
    ORC_TRAINING_DATA_DIR,
    ORC_QUERY_TEMPLATE_DRAWINGS,
    ORC_DRAWINGS_TRAIN_DATA_DIR
)

# ----------------------------
# Paths & Outputs
# ----------------------------
EXCEL_PATH = os.path.join(ORC_TRAINING_DATA_DIR, "loadsheet", "Active Drawing List.xlsx")

# TXT files (one template per line; variations are optional "KEY: pat1 | pat2" lines)
QUERY_TEMPLATES_PATH = os.path.join(ORC_QUERY_TEMPLATE_DRAWINGS, "DRAWINGS_ENHANCED_QUERY_TEMPLATES.txt")
VARIATIONS_PATH = os.path.join(ORC_QUERY_TEMPLATE_DRAWINGS, "DRAWINGS_NATURAL_LANGUAGE_VARIATIONS.txt")

# Where your rich (NER-friendly) training file goes (same as before)
OUTPUT_DIR = os.path.join(ORC_DRAWINGS_TRAIN_DATA_DIR)
os.makedirs(OUTPUT_DIR, exist_ok=True)
OUTPUT_PATH = os.path.join(OUTPUT_DIR, "intent_train_drawings.jsonl")

# NEW: Simple intent file for your intent classifier
ORC_INTENT_TRAIN_DATA_DIR = os.path.join(ORC_TRAINING_DATA_DIR, "intent_classifier")
os.makedirs(ORC_INTENT_TRAIN_DATA_DIR, exist_ok=True)
INTENT_SIMPLE_PATH = os.path.join(ORC_INTENT_TRAIN_DATA_DIR, "intent_train_drawings.jsonl")

print(f"Looking for templates at: {QUERY_TEMPLATES_PATH}")
print(f"Looking for variations at: {VARIATIONS_PATH}")
print(f"Excel file path: {EXCEL_PATH}")
print(f"Rich (NER-friendly) output path: {OUTPUT_PATH}")
print(f"Intent (simple) output path: {INTENT_SIMPLE_PATH}")

RNG = random.Random(42)

# ----------------------------
# Loaders
# ----------------------------
def _read_txt_lines(path: str):
    p = Path(path)
    if not p.exists():
        return []
    lines = [ln.strip() for ln in p.read_text(encoding="utf-8").splitlines()]
    return [ln for ln in lines if ln and not ln.startswith("#")]

def load_query_templates():
    """
    Load query templates from TXT file.
    Each line is a template, e.g.:
      "Show me drawing [DRAWING_NUMBER]"
      "I need the print for [EQUIPMENT_NAME]"
    """
    lines = _read_txt_lines(QUERY_TEMPLATES_PATH)
    if lines:
        return lines
    # Fallback if file missing/empty
    print("Templates TXT missing or empty; using fallback templates.")
    return [
        "I need the print for equipment [EQUIPMENT_NUMBER]",
        "Show me the drawing for [EQUIPMENT_NAME]",
        "Find drawing [DRAWING_NUMBER]",
        "I need the [DRAWING_NAME]",
        "Show me the print for part # [SPARE_PART_NUMBER]"
    ]

def load_natural_language_variations():
    """
    Load variations from TXT:
      KEY: pat1 | pat2 | pat3
    Example:
      EQUIPMENT_NUMBER: equipment {equipment_number} | eqp {equipment_number}
      DRAWING_NUMBER: drawing {drawing_number} | dwg {drawing_number}
    Placeholders must be lowercase inside braces to be formatted.
    """
    lines = _read_txt_lines(VARIATIONS_PATH)
    variations = {}
    if lines:
        for ln in lines:
            # Split "KEY: pat1 | pat2"
            if ":" not in ln:
                continue
            key, rhs = ln.split(":", 1)
            key = key.strip().upper()
            pats = [p.strip() for p in rhs.split("|") if p.strip()]
            if pats:
                variations[key] = {"_all": pats}  # bucket all patterns
    if variations:
        return variations

    # Fallback to your existing structure if TXT is missing/empty
    print("Variations TXT missing/empty; using fallback variations.")
    return {
        "EQUIPMENT_NUMBER": {"_all": ["equipment {equipment_number}"]},
        "EQUIPMENT_NAME": {"_all": ["{equipment_name}"]},
        "DRAWING_NUMBER": {"_all": ["drawing {drawing_number}"]},
        "DRAWING_NAME": {"_all": ["{drawing_name}"]},
        "SPARE_PART_NUMBER": {"_all": ["part {spare_part_number}"]}
    }

# ----------------------------
# Utils
# ----------------------------
def clean_value(value):
    if pd.isna(value) or value is None:
        return ""
    return str(value).strip()

def parse_spare_parts(spare_part_string):
    if not spare_part_string or pd.isna(spare_part_string):
        return []
    parts = [part.strip() for part in str(spare_part_string).split(",")]
    return [p for p in parts if p]

def get_random_variation(variations_dict, entity_type, value):
    """
    entity_type is UPPER like 'DRAWING_NUMBER'; variations_dict[entity_type]["_all"] is a list of patterns
    where placeholders inside patterns are lowercase {drawing_number}, {equipment_name}, etc.
    """
    if entity_type not in variations_dict:
        return value
    patterns = variations_dict[entity_type].get("_all", [])
    if not patterns:
        return value
    pattern = RNG.choice(patterns)
    placeholder = "{" + entity_type.lower() + "}"
    return pattern.replace(placeholder, value)

# ----------------------------
# Example generation
# ----------------------------
def generate_training_examples(row, query_templates, variations):
    examples = []

    equipment_number = clean_value(row.get("equipment_number", ""))
    equipment_name   = clean_value(row.get("equipment_name", ""))
    drawing_number   = clean_value(row.get("drawing_number", ""))
    drawing_name     = clean_value(row.get("drawing_name", ""))
    spare_raw        = row.get("spare_part_number", "")

    spare_part_numbers = parse_spare_parts(spare_raw)

    if not any([equipment_number, equipment_name, drawing_number, drawing_name, spare_part_numbers]):
        return examples

    for template in query_templates:
        if "[SPARE_PART_NUMBER]" in template and not spare_part_numbers:
            continue

        # If spare parts are referenced, create one per spare
        if "[SPARE_PART_NUMBER]" in template:
            for spn in spare_part_numbers:
                for _ in range(2):
                    query = template
                    entities = []

                    # SPARE_PART_NUMBER
                    v = get_random_variation(variations, "SPARE_PART_NUMBER", spn)
                    query = query.replace("[SPARE_PART_NUMBER]", v)
                    entities.append({"entity": "SPARE_PART_NUMBER", "value": spn,
                                     "start": query.find(v), "end": query.find(v) + len(v)})

                    # Optional others
                    if "[EQUIPMENT_NUMBER]" in query and equipment_number:
                        v2 = get_random_variation(variations, "EQUIPMENT_NUMBER", equipment_number)
                        query = query.replace("[EQUIPMENT_NUMBER]", v2)
                        entities.append({"entity": "EQUIPMENT_NUMBER", "value": equipment_number,
                                         "start": query.find(v2), "end": query.find(v2) + len(v2)})

                    if "[EQUIPMENT_NAME]" in query and equipment_name:
                        v2 = get_random_variation(variations, "EQUIPMENT_NAME", equipment_name)
                        query = query.replace("[EQUIPMENT_NAME]", v2)
                        entities.append({"entity": "EQUIPMENT_NAME", "value": equipment_name,
                                         "start": query.find(v2), "end": query.find(v2) + len(v2)})

                    if "[DRAWING_NUMBER]" in query and drawing_number:
                        v2 = get_random_variation(variations, "DRAWING_NUMBER", drawing_number)
                        query = query.replace("[DRAWING_NUMBER]", v2)
                        entities.append({"entity": "DRAWING_NUMBER", "value": drawing_number,
                                         "start": query.find(v2), "end": query.find(v2) + len(v2)})

                    if "[DRAWING_NAME]" in query and drawing_name:
                        v2 = get_random_variation(variations, "DRAWING_NAME", drawing_name)
                        query = query.replace("[DRAWING_NAME]", v2)
                        entities.append({"entity": "DRAWING_NAME", "value": drawing_name,
                                         "start": query.find(v2), "end": query.find(v2) + len(v2)})

                    if entities and "[" not in query:
                        examples.append({"text": query, "intent": "drawing_search", "entities": entities})
        else:
            # Non-spare templates
            for _ in range(2):
                query = template
                entities = []

                if "[EQUIPMENT_NUMBER]" in query and equipment_number:
                    v = get_random_variation(variations, "EQUIPMENT_NUMBER", equipment_number)
                    query = query.replace("[EQUIPMENT_NUMBER]", v)
                    entities.append({"entity": "EQUIPMENT_NUMBER", "value": equipment_number,
                                     "start": query.find(v), "end": query.find(v) + len(v)})

                if "[EQUIPMENT_NAME]" in query and equipment_name:
                    v = get_random_variation(variations, "EQUIPMENT_NAME", equipment_name)
                    query = query.replace("[EQUIPMENT_NAME]", v)
                    entities.append({"entity": "EQUIPMENT_NAME", "value": equipment_name,
                                     "start": query.find(v), "end": query.find(v) + len(v)})

                if "[DRAWING_NUMBER]" in query and drawing_number:
                    v = get_random_variation(variations, "DRAWING_NUMBER", drawing_number)
                    query = query.replace("[DRAWING_NUMBER]", v)
                    entities.append({"entity": "DRAWING_NUMBER", "value": drawing_number,
                                     "start": query.find(v), "end": query.find(v) + len(v)})

                if "[DRAWING_NAME]" in query and drawing_name:
                    v = get_random_variation(variations, "DRAWING_NAME", drawing_name)
                    query = query.replace("[DRAWING_NAME]", v)
                    entities.append({"entity": "DRAWING_NAME", "value": drawing_name,
                                     "start": query.find(v), "end": query.find(v) + len(v)})

                if entities and "[" not in query:
                    examples.append({"text": query, "intent": "drawing_search", "entities": entities})

    return examples

def generate_combination_examples(df, variations, num_examples=100):
    examples = []
    combo = [
        "I need the {drawing_name} for {equipment_name}",
        "Show me {drawing_name} for equipment {equipment_number}",
        "Find {drawing_name} drawing {drawing_number}",
        "Looking for part {spare_part_number} on {equipment_name}",
        "Where is part {spare_part_number} on equipment {equipment_number}?",
        "I need drawing {drawing_number} for the {equipment_name}",
        "Can you find the {equipment_name} print with part {spare_part_number}?",
        "Show the {drawing_name} for {equipment_name} number {equipment_number}"
    ]

    for _ in range(num_examples):
        row = df.sample(1).iloc[0]
        equipment_number = clean_value(row.get("equipment_number", ""))
        equipment_name   = clean_value(row.get("equipment_name", ""))
        drawing_number   = clean_value(row.get("drawing_number", ""))
        drawing_name     = clean_value(row.get("drawing_name", ""))
        spare_parts_raw  = row.get("spare_part_number", "")
        spare_part_numbers = parse_spare_parts(spare_parts_raw)

        template = random.choice(combo)

        if "{spare_part_number}" in template and not spare_part_numbers:
            continue
        spn = random.choice(spare_part_numbers) if "{spare_part_number}" in template else ""

        query = template
        entities = []

        repl = {
            "{equipment_number}": equipment_number,
            "{equipment_name}": equipment_name,
            "{drawing_number}": drawing_number,
            "{drawing_name}": drawing_name,
            "{spare_part_number}": spn
        }

        for placeholder, val in repl.items():
            if placeholder in query and val:
                etype = placeholder.strip("{}").upper()
                v = get_random_variation(variations, etype, val)
                query = query.replace(placeholder, v)
                entities.append({"entity": etype, "value": val,
                                 "start": query.find(v), "end": query.find(v) + len(v)})

        if entities and "{" not in query:
            examples.append({"text": query, "intent": "drawing_search", "entities": entities})

    return examples

# ----------------------------
# Main
# ----------------------------
def main():
    print("Loading drawing data...")
    df = pd.read_excel(EXCEL_PATH)
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]

    print("Loading query templates and variations...")
    query_templates = load_query_templates()
    variations = load_natural_language_variations()

    print(f"Loaded {len(query_templates)} query templates")
    print(f"Loaded variations for {len(variations)} entity types")

    all_examples = []

    # Row-wise examples
    print("Generating training examples from data rows...")
    for idx, row in df.iterrows():
        examples = generate_training_examples(row, query_templates, variations)
        all_examples.extend(examples)
        if (idx + 1) % 100 == 0:
            print(f"Processed {idx + 1} rows, generated {len(all_examples)} examples so far")

    # Combo examples
    print("Generating combination examples...")
    all_examples.extend(generate_combination_examples(df, variations, num_examples=200))

    # Dedup by text
    unique_examples = []
    seen = set()
    for ex in all_examples:
        if ex["text"] not in seen:
            unique_examples.append(ex)
            seen.add(ex["text"])

    print(f"Generated {len(all_examples)} total examples")
    print(f"After deduplication: {len(unique_examples)} unique examples")

    # -------- Write rich (NER-friendly) JSONL (unchanged output format) --------
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f_out:
        for ex in unique_examples:
            f_out.write(json.dumps(ex, ensure_ascii=False) + "\n")
    print(f"Training data saved to {OUTPUT_PATH}")

    # -------- ALSO write simple INTENT JSONL for classifier --------
    with open(INTENT_SIMPLE_PATH, "w", encoding="utf-8") as f_int:
        for ex in unique_examples:
            f_int.write(json.dumps({"text": ex["text"], "label": "drawings"}, ensure_ascii=False) + "\n")
    print(f"Intent data (simple) saved to {INTENT_SIMPLE_PATH}")

    # Samples
    print("\nSample training examples:")
    for i, ex in enumerate(unique_examples[:5]):
        print(f"{i + 1}. Text: {ex['text']}")
        print(f"   Entities: {ex['entities']}")
        print()

    # Spare part stats
    spare_ex = [ex for ex in unique_examples if any(ent['entity'] == 'SPARE_PART_NUMBER' for ent in ex['entities'])]
    print(f"Examples with spare parts: {len(spare_ex)}")
    uniq_sp = set()
    for ex in spare_ex:
        for ent in ex['entities']:
            if ent['entity'] == 'SPARE_PART_NUMBER':
                uniq_sp.add(ent['value'])
    print(f"Unique spare part numbers: {len(uniq_sp)}")
    if uniq_sp:
        print(f"Sample spare parts: {list(uniq_sp)[:10]}")

if __name__ == "__main__":
    main()
