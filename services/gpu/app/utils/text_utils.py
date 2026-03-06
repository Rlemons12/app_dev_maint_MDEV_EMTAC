from __future__ import annotations

import re


def _normalize_text(text: str) -> str:
    """
    Normalize markdown/text for downstream indexing.

    Keeps formatting mostly intact but removes
    excessive whitespace and control characters.
    """

    if not text:
        return ""

    # Normalize newlines
    text = text.replace("\r\n", "\n").replace("\r", "\n")

    # Remove null characters and control chars
    text = re.sub(r"[\x00-\x08\x0B\x0C\x0E-\x1F]", "", text)

    # Collapse excessive whitespace
    text = re.sub(r"[ \t]+", " ", text)

    # Remove excessive blank lines
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip()