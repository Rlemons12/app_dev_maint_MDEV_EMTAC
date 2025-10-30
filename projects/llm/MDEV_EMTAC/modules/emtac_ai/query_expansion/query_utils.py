
# modules/emtac_ai/query_expansion/query_utils.py
from __future__ import annotations

import re
from typing import Iterable, List

def dedup_preserve_order(items: Iterable[str]) -> List[str]:
    """
    Remove duplicates while preserving original order.
    Strips whitespace and lowercases for comparison but returns original strings.
    """
    seen = set()
    out: List[str] = []
    for s in items:
        if s is None:
            continue
        raw = str(s).strip()
        key = raw.lower()
        if key and key not in seen:
            seen.add(key)
            out.append(raw)
    return out

def replace_word_boundary(text: str, old: str, new: str) -> str:
    """
    Replace full-word matches of `old` with `new` (case-insensitive).
    """
    if not old or new is None:
        return text
    return re.sub(rf"\b{re.escape(old)}\b", new, text, flags=re.IGNORECASE)

def tokenize_words_lower(text: str) -> List[str]:
    """Crude word tokenizer → lowercase tokens."""
    return re.findall(r"\b\w+\b", (text or "").lower())
