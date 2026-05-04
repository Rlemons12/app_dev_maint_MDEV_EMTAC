from __future__ import annotations

import re
from typing import List, Dict, Any, Optional

from modules.configuration.log_config import (
    debug_id,
    info_id,
    warning_id,
    with_request_id,
)

try:
    from transformers import AutoTokenizer
except Exception:
    AutoTokenizer = None


class ContextBuilder:
    """
    Builds a safe, focused, deduplicated context block for RAG.

    Goals:
    - Prevent prompt/question/answer leakage from document chunks
    - Keep context small enough for local models
    - Prefer best ranked chunks first
    - Limit individual chunk size
    - Preserve used_chunks for UI/document linking
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        max_tokens: int = 900,
        include_metadata: bool = False,
        max_chunks: int = 4,
        max_chunk_chars: int = 700,
    ):
        self.max_tokens = max_tokens
        self.include_metadata = include_metadata
        self.max_chunks = max_chunks
        self.max_chunk_chars = max_chunk_chars
        self.model_path = model_path or "E:/emtac/models/llm/TinyLlama"

        self.tokenizer = None

        if AutoTokenizer is not None:
            try:
                self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
            except Exception:
                warning_id(
                    f"[ContextBuilder] Failed loading tokenizer at {self.model_path}; "
                    "falling back to whitespace tokenizer."
                )

    def _num_tokens(self, text: str) -> int:
        if not text:
            return 0

        if not self.tokenizer:
            return len(text.split())

        try:
            return len(self.tokenizer.encode(text, add_special_tokens=False))
        except Exception:
            return len(text.split())

    @staticmethod
    def _sanitize_context(text: str) -> str:
        if not text:
            return ""

        text = str(text).replace("\r\n", "\n").replace("\r", "\n")

        # Remove prompt/dialog artifacts that can cause model continuation.
        text = re.sub(
            r"\b(QUESTION|ANSWER|FINAL ANSWER|USER|ASSISTANT|SYSTEM)\s*:.*",
            "",
            text,
            flags=re.IGNORECASE | re.DOTALL,
        )

        # Remove embedded context markers if they accidentally entered documents.
        text = re.sub(
            r"---\s*CONTEXT\s+START\s*---|---\s*CONTEXT\s+END\s*---",
            "",
            text,
            flags=re.IGNORECASE,
        )

        # Remove retrieval labels if embedded in content.
        text = re.sub(
            r"^\s*\[Chunk\s+\d+\].*$",
            "",
            text,
            flags=re.IGNORECASE | re.MULTILINE,
        )

        # Normalize whitespace.
        text = re.sub(r"[ \t]+", " ", text)
        text = re.sub(r"\n{3,}", "\n\n", text)

        return text.strip()

    def _trim_chunk(self, text: str) -> str:
        if not text:
            return ""

        text = text.strip()

        if len(text) <= self.max_chunk_chars:
            return text

        trimmed = text[: self.max_chunk_chars].rstrip()

        # Prefer not to cut mid-sentence when possible.
        sentence_end = max(
            trimmed.rfind("."),
            trimmed.rfind("?"),
            trimmed.rfind("!"),
            trimmed.rfind("\n"),
        )

        if sentence_end >= int(self.max_chunk_chars * 0.55):
            trimmed = trimmed[: sentence_end + 1].rstrip()

        return trimmed + "..."

    @staticmethod
    def _content_fingerprint(text: str) -> str:
        return re.sub(r"\s+", " ", text.lower()).strip()

    @staticmethod
    def _score_value(chunk: Dict[str, Any]) -> float:
        value = chunk.get("distance", chunk.get("score", 0.0))

        try:
            return float(value)
        except Exception:
            return 0.0

    @with_request_id
    def build_context(
        self,
        retrieved_chunks: List[Dict[str, Any]],
        request_id: Optional[str] = None,
    ) -> Dict[str, Any]:

        retrieved_chunks = retrieved_chunks or []

        debug_id(
            f"[ContextBuilder] Starting with {len(retrieved_chunks)} raw chunks",
            request_id,
        )

        chunks_sorted = sorted(
            retrieved_chunks,
            key=lambda c: (
                self._score_value(c),
                len((c.get("content") or "").strip()),
            ),
        )

        seen_contents = set()
        deduped: List[Dict[str, Any]] = []

        for ch in chunks_sorted:
            raw = (ch.get("content") or "").strip()
            if not raw:
                continue

            sanitized = self._sanitize_context(raw)
            if not sanitized:
                continue

            trimmed = self._trim_chunk(sanitized)
            if not trimmed:
                continue

            fingerprint = self._content_fingerprint(trimmed)
            if fingerprint in seen_contents:
                continue

            seen_contents.add(fingerprint)

            new_ch = dict(ch)
            new_ch["content"] = trimmed

            new_ch.setdefault(
                "chunk_id",
                ch.get("chunk_id")
                or ch.get("id")
                or f"{ch.get('document_id', 'doc')}_{len(deduped) + 1}",
            )

            new_ch.setdefault("document_title", ch.get("document_title"))
            new_ch.setdefault("document_url", ch.get("document_url"))

            deduped.append(new_ch)

        debug_id(
            f"[ContextBuilder] Deduped to {len(deduped)} chunks",
            request_id,
        )

        used_chunks: List[Dict[str, Any]] = []
        context_parts: List[str] = []
        token_count = 0

        for idx, chunk in enumerate(deduped):
            if len(used_chunks) >= self.max_chunks:
                break

            content = (chunk.get("content") or "").strip()
            if not content:
                continue

            if self.include_metadata:
                meta = [f"[Source {idx + 1}]"]

                if chunk.get("document_title"):
                    meta.append(f"Title={chunk['document_title']}")

                header = " | ".join(meta)
                block = f"{header}\n{content}\n\n"
            else:
                block = f"{content}\n\n"

            block_tokens = self._num_tokens(block)

            if block_tokens > self.max_tokens:
                warning_id(
                    f"[ContextBuilder] Skipping oversized chunk ({block_tokens} tokens)",
                    request_id,
                )
                continue

            if token_count + block_tokens > self.max_tokens:
                break

            context_parts.append(block)
            used_chunks.append(chunk)
            token_count += block_tokens

        context_str = "".join(context_parts).strip()

        info_id(
            f"[ContextBuilder] Final context: {token_count} tokens, "
            f"{len(used_chunks)} chunks, "
            f"max_chunks={self.max_chunks}, "
            f"max_chunk_chars={self.max_chunk_chars}",
            request_id,
        )

        return {
            "context": context_str,
            "used_chunks": used_chunks,
        }