#!/usr/bin/env python3
import re
from pathlib import Path
from datetime import datetime
import sys

# Import project config so we get ORC_QUERY_TEMPLATE_DRAWINGS
CONFIG_PATH = Path(__file__).resolve().parents[4] / "configuration"
sys.path.append(str(CONFIG_PATH))
import config  # loads ORC_QUERY_TEMPLATE_DRAWINGS from your config.py

# Target file
TARGET = Path(config.ORC_QUERY_TEMPLATE_DRAWINGS) / "DRAWINGS_ENHANCED_QUERY_TEMPLATES.txt"

# Recognize quoted templates (list-style source)
QUOTED_RE = re.compile(r"""["'](.*?)["']\s*,?\s*$""")

# Convert [PLACEHOLDER] → {placeholder}
BRACKET_TO_CURLY = [
    (re.compile(r"\[EQUIPMENT_NUMBER\]", re.I), "{equipment_number}"),
    (re.compile(r"\[EQUIPMENT_NAME\]",   re.I), "{equipment_name}"),
    (re.compile(r"\[DRAWING_NUMBER\]",   re.I), "{drawing_number}"),
    (re.compile(r"\[DRAWING_NAME\]",     re.I), "{drawing_name}"),
    (re.compile(r"\[SPARE_PART_NUMBER\]",re.I), "{spare_part_number}"),
    (re.compile(r"\[LOCATION\]",         re.I), "{location}"),
]

# Valid curly placeholders to preserve (don’t escape these braces)
VALID = [
    "{equipment_number}",
    "{equipment_name}",
    "{drawing_number}",
    "{drawing_name}",
    "{spare_part_number}",
    "{location}",
]
SENT = {v: f"__PH_{i}__" for i, v in enumerate(VALID)}

# Lines to drop entirely if encountered
STRAY_LINE_PATTERNS = [
    re.compile(r'^\s*print\s*\(', re.I),
    re.compile(r'DRAWINGS_ENHANCED_QUERY_TEMPLATES', re.I),
]

def clean_line(s: str) -> str:
    # Convert bracket tokens first
    for pat, repl in BRACKET_TO_CURLY:
        s = pat.sub(repl, s)
    # Protect valid placeholders, escape other braces, restore placeholders
    for k, v in SENT.items():
        s = s.replace(k, v)
    s = s.replace("{", "{{").replace("}", "}}")
    for k, v in SENT.items():
        s = s.replace(v, k)
    return s

def to_plain_templates(text: str) -> list[str]:
    """Support both list-style (quoted) and plain one-per-line files."""
    lines = text.splitlines()
    quoted = []
    plain = []
    for raw in lines:
        if any(p.search(raw) for p in STRAY_LINE_PATTERNS):
            continue
        m = QUOTED_RE.search(raw.strip())
        if m:
            s = m.group(1).strip()
            if s:
                quoted.append(s)
        else:
            # keep plain, non-empty lines that aren’t just brackets/commas
            t = raw.strip()
            if t and t not in ("]", "[", ","):
                plain.append(t)

    # If we found several quoted entries, assume list-style source
    if len(quoted) >= 5:
        return [clean_line(s) for s in quoted if s.strip()]
    # Otherwise treat as already-plain
    return [clean_line(s) for s in plain if s.strip()]

def main():
    if not TARGET.exists():
        raise SystemExit(f"[ERR] File not found: {TARGET}")

    raw = TARGET.read_text(encoding="utf-8")
    backup = TARGET.with_suffix(TARGET.suffix + f".bak_{datetime.now():%Y%m%d_%H%M%S}")
    backup.write_text(raw, encoding="utf-8")

    templates = to_plain_templates(raw)
    out_text = "\n".join(templates) + "\n"
    TARGET.write_text(out_text, encoding="utf-8")

    # Sanity check with a dummy context
    ctx = {
        "equipment_number": "EQ-1001",
        "equipment_name": "Boiler Pump",
        "drawing_number": "DWG-42A",
        "drawing_name": "P&ID-Section-A",
        "spare_part_number": "SP-7788",
        "location": "Area 3",
    }
    bad = []
    for i, line in enumerate(out_text.splitlines(), 1):
        if not line.strip():
            continue
        try:
            line.format(**ctx)
        except Exception as e:
            bad.append((i, line, str(e)))

    print(f"[OK] Cleaned: {TARGET}")
    print(f"[OK] Backup: {backup}")
    print(f"[OK] Total templates: {len(templates)}")
    if bad:
        print("[WARN] Lines not format-safe:")
        for i, l, err in bad[:20]:
            print(f"  L{i}: {l!r} -> {err}")
        if len(bad) > 20:
            print(f"  ...and {len(bad)-20} more")
    else:
        print("[OK] All lines format-safe with curly placeholders.")

if __name__ == "__main__":
    main()
