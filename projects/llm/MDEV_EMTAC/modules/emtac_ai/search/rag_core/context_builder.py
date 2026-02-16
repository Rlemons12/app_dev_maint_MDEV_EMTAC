from __future__ import annotations

import re
from typing import List, Dict, Any, Optional

from modules.configuration.log_config import (
    debug_id,
    info_id,
    warning_id,
    with_request_id,
)

# Tokenizer source (TinyLlama / Qwen safe)
from transformers import AutoTokenizer


class ContextBuilder:
    """
    Builds a safe, deduplicated, token-aware context for RAG.

    Fixes:
      - Prevents QUESTION/ANSWER dialog leakage
      - Enforces token budgets (default ~1024 tokens)
      - Removes duplicate chunks
      - Sorts by distance → then by chunk length
      - Preserves document + chunk trace metadata for UI linking
      - Metadata headers optional and token-efficient
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        max_tokens: int = 1024,
        include_metadata: bool = True,
    ):
        self.max_tokens = max_tokens
        self.include_metadata = include_metadata

        self.model_path = model_path or "E:/emtac/models/llm/TinyLlama"

        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        except Exception:
            warning_id(
                f"[ContextBuilder] Failed loading tokenizer at {self.model_path}; "
                f"falling back to whitespace tokenizer."
            )
            self.tokenizer = None

    # -------------------------------------------------------------
    # Token counting
    # -------------------------------------------------------------
    def _num_tokens(self, text: str) -> int:
        if not self.tokenizer:
            return len(text.split())
        try:
            return len(self.tokenizer.encode(text, add_special_tokens=False))
        except Exception:
            return len(text.split())

    # -------------------------------------------------------------
    # Context sanitation (CRITICAL)
    # -------------------------------------------------------------
    @staticmethod
    def _sanitize_context(text: str) -> str:
        """
        Remove dialog artifacts that cause LLM continuation.
        """
        text = re.sub(r"\bQUESTION:\b.*", "", text, flags=re.IGNORECASE | re.DOTALL)
        text = re.sub(r"\bANSWER:\b.*", "", text, flags=re.IGNORECASE | re.DOTALL)
        return text.strip()

    # -------------------------------------------------------------
    # Main context builder
    # -------------------------------------------------------------
    @with_request_id
    def build_context(
        self,
        retrieved_chunks: List[Dict[str, Any]],
        request_id: Optional[str] = None,
    ) -> Dict[str, Any]:

        debug_id(
            f"[ContextBuilder] Starting with {len(retrieved_chunks)} raw chunks",
            request_id,
        )

        # ---------------------------------------------------------
        # 1. Sort by similarity, then by content length
        # ---------------------------------------------------------
        chunks_sorted = sorted(
            retrieved_chunks,
            key=lambda c: (c.get("distance", 0.0), len(c.get("content", ""))),
        )

        # ---------------------------------------------------------
        # 2. Deduplicate by sanitized content
        # ---------------------------------------------------------
        seen_contents = set()
        deduped: List[Dict[str, Any]] = []

        for ch in chunks_sorted:
            raw = (ch.get("content") or "").strip()
            if not raw:
                continue

            sanitized = self._sanitize_context(raw)
            if not sanitized:
                continue

            if sanitized in seen_contents:
                continue

            seen_contents.add(sanitized)

            # --- preserve + normalize metadata ---
            new_ch = dict(ch)
            new_ch["content"] = sanitized

            # Stable identifiers for UI linking
            new_ch.setdefault(
                "chunk_id",
                ch.get("chunk_id")
                or f"{ch.get('document_id', 'doc')}_{len(deduped) + 1}",
            )

            # Optional UI metadata (safe passthrough)
            new_ch.setdefault("document_title", ch.get("document_title"))
            new_ch.setdefault("document_url", ch.get("document_url"))

            deduped.append(new_ch)

        debug_id(
            f"[ContextBuilder] Deduped to {len(deduped)} chunks",
            request_id,
        )

        # ---------------------------------------------------------
        # 3. Build final context string
        # ---------------------------------------------------------
        used_chunks: List[Dict[str, Any]] = []
        context_parts: List[str] = []
        token_count = 0

        for idx, chunk in enumerate(deduped):
            content = chunk.get("content", "")
            if not content:
                continue

            if self.include_metadata:
                meta = [f"[Chunk {idx + 1}]"]

                if chunk.get("document_id"):
                    meta.append(f"DocID={chunk['document_id']}")

                if chunk.get("distance") is not None:
                    meta.append(f"Score={chunk['distance']:.4f}")

                header = " | ".join(meta)
                block = f"{header}\n{content}\n\n"
            else:
                block = content + "\n\n"

            block_tokens = self._num_tokens(block)

            if block_tokens > self.max_tokens // 2:
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

        context_str = "".join(context_parts)

        info_id(
            f"[ContextBuilder] Final context: {token_count} tokens, "
            f"{len(used_chunks)} chunks",
            request_id,
        )

        return {
            "context": context_str,
            "used_chunks": used_chunks,
        }
