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
        - Keep normal RAG context small enough for local models
        - Prefer best ranked chunks first
        - Limit individual chunk size
        - Preserve used_chunks for UI/document linking
        - Allow deeper context when document-scoped mode is active

    Normal mode:
        Uses max_chunks.

    Document mode:
        Uses document_scope_max_chunks and document_scope_max_tokens.
        This lets "Ask this document" consider more of the selected manual
        without making every normal RAG answer heavier.
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        max_tokens: int = 900,
        include_metadata: bool = False,
        max_chunks: int = 4,
        max_chunk_chars: int = 700,
        document_scope_max_tokens: int = 1200,
        document_scope_max_chunks: int = 4,
        document_scope_max_chunk_chars: int = 700,
    ):
        self.max_tokens = max_tokens
        self.include_metadata = include_metadata
        self.max_chunks = max_chunks
        self.max_chunk_chars = max_chunk_chars

        self.document_scope_max_tokens = document_scope_max_tokens
        self.document_scope_max_chunks = document_scope_max_chunks
        self.document_scope_max_chunk_chars = document_scope_max_chunk_chars

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

    @staticmethod
    def _safe_int(value: Any) -> Optional[int]:
        if value in (None, "", "None"):
            return None

        if isinstance(value, bool):
            return None

        try:
            return int(value)
        except (TypeError, ValueError):
            return None

    @classmethod
    def _normalize_document_scope(
        cls,
        document_scope: Optional[Dict[str, Any]],
    ) -> Optional[Dict[str, Any]]:
        if not document_scope or not isinstance(document_scope, dict):
            return None

        enabled = document_scope.get("enabled", True)

        if enabled is False:
            return None

        scope_type = (
            document_scope.get("scope_type")
            or document_scope.get("scopeType")
            or "complete_document"
        )

        scope_type = str(scope_type or "").strip() or "complete_document"

        if scope_type != "complete_document":
            return None

        complete_document_id = (
            document_scope.get("complete_document_id")
            or document_scope.get("completed_document_id")
            or document_scope.get("completeDocumentId")
            or document_scope.get("completeDocumentID")
        )

        complete_document_id = cls._safe_int(complete_document_id)

        if complete_document_id is None:
            return None

        document_id = (
            document_scope.get("document_id")
            or document_scope.get("documentId")
        )

        document_id = cls._safe_int(document_id)

        document_name = (
            document_scope.get("document_name")
            or document_scope.get("documentName")
            or document_scope.get("name")
            or document_scope.get("title")
            or f"Document #{complete_document_id}"
        )

        document_name = str(document_name or "").strip() or f"Document #{complete_document_id}"

        return {
            "enabled": True,
            "scope_type": "complete_document",
            "document_id": document_id,
            "complete_document_id": complete_document_id,
            "document_name": document_name,
        }

    @staticmethod
    def _extract_chunk_complete_document_id(chunk: Dict[str, Any]) -> Optional[int]:
        if not isinstance(chunk, dict):
            return None

        direct_value = (
            chunk.get("complete_document_id")
            or chunk.get("completed_document_id")
            or chunk.get("completeDocumentId")
            or chunk.get("completeDocumentID")
        )

        direct_int = ContextBuilder._safe_int(direct_value)

        if direct_int is not None:
            return direct_int

        document = chunk.get("document")

        if isinstance(document, dict):
            nested_value = (
                document.get("complete_document_id")
                or document.get("completed_document_id")
                or document.get("completeDocumentId")
                or document.get("completeDocumentID")
            )

            nested_int = ContextBuilder._safe_int(nested_value)

            if nested_int is not None:
                return nested_int

        complete_document = (
            chunk.get("complete_document")
            or chunk.get("completed_document")
            or chunk.get("completeDocument")
        )

        if isinstance(complete_document, dict):
            complete_document_id = (
                complete_document.get("id")
                or complete_document.get("complete_document_id")
                or complete_document.get("completeDocumentId")
            )

            return ContextBuilder._safe_int(complete_document_id)

        return None

    def _trim_chunk(self, text: str, *, max_chunk_chars: int) -> str:
        if not text:
            return ""

        text = text.strip()

        if len(text) <= max_chunk_chars:
            return text

        trimmed = text[:max_chunk_chars].rstrip()

        # Prefer not to cut mid-sentence when possible.
        sentence_end = max(
            trimmed.rfind("."),
            trimmed.rfind("?"),
            trimmed.rfind("!"),
            trimmed.rfind("\n"),
        )

        if sentence_end >= int(max_chunk_chars * 0.55):
            trimmed = trimmed[: sentence_end + 1].rstrip()

        return trimmed + "..."

    def _resolve_limits(
        self,
        *,
        document_scope: Optional[Dict[str, Any]],
        max_chunks_override: Optional[int],
        max_tokens_override: Optional[int],
        max_chunk_chars_override: Optional[int],
    ) -> Dict[str, int]:
        document_scope_active = bool(self._normalize_document_scope(document_scope))

        if document_scope_active:
            max_chunks = self.document_scope_max_chunks
            max_tokens = self.document_scope_max_tokens
            max_chunk_chars = self.document_scope_max_chunk_chars
        else:
            max_chunks = self.max_chunks
            max_tokens = self.max_tokens
            max_chunk_chars = self.max_chunk_chars

        override_chunks = self._safe_int(max_chunks_override)
        override_tokens = self._safe_int(max_tokens_override)
        override_chars = self._safe_int(max_chunk_chars_override)

        if override_chunks is not None and override_chunks > 0:
            max_chunks = override_chunks

        if override_tokens is not None and override_tokens > 0:
            max_tokens = override_tokens

        if override_chars is not None and override_chars > 0:
            max_chunk_chars = override_chars

        return {
            "max_chunks": max_chunks,
            "max_tokens": max_tokens,
            "max_chunk_chars": max_chunk_chars,
        }

    @with_request_id
    def build_context(
        self,
        retrieved_chunks: List[Dict[str, Any]],
        request_id: Optional[str] = None,
        document_scope: Optional[Dict[str, Any]] = None,
        max_chunks_override: Optional[int] = None,
        max_tokens_override: Optional[int] = None,
        max_chunk_chars_override: Optional[int] = None,
    ) -> Dict[str, Any]:

        retrieved_chunks = retrieved_chunks or []
        normalized_document_scope = self._normalize_document_scope(document_scope)

        limits = self._resolve_limits(
            document_scope=normalized_document_scope,
            max_chunks_override=max_chunks_override,
            max_tokens_override=max_tokens_override,
            max_chunk_chars_override=max_chunk_chars_override,
        )

        max_chunks = limits["max_chunks"]
        max_tokens = limits["max_tokens"]
        max_chunk_chars = limits["max_chunk_chars"]

        debug_id(
            f"[ContextBuilder] Starting with {len(retrieved_chunks)} raw chunks "
            f"document_scope_enabled={bool(normalized_document_scope)} "
            f"complete_document_id="
            f"{normalized_document_scope.get('complete_document_id') if normalized_document_scope else None} "
            f"max_chunks={max_chunks} max_tokens={max_tokens} max_chunk_chars={max_chunk_chars}",
            request_id,
        )

        # Safety filter:
        # If document_scope is active, do not let chunks from another manual
        # enter the answer context.
        if normalized_document_scope:
            selected_complete_document_id = normalized_document_scope.get(
                "complete_document_id"
            )

            before_scope_filter = len(retrieved_chunks)

            retrieved_chunks = [
                chunk
                for chunk in retrieved_chunks
                if self._extract_chunk_complete_document_id(chunk) == selected_complete_document_id
            ]

            after_scope_filter = len(retrieved_chunks)

            if after_scope_filter != before_scope_filter:
                warning_id(
                    "[ContextBuilder] Filtered chunks by document_scope "
                    f"before={before_scope_filter} after={after_scope_filter} "
                    f"complete_document_id={selected_complete_document_id}",
                    request_id,
                )

        chunks_sorted = sorted(
            retrieved_chunks,
            key=lambda c: (
                self._score_value(c),
                len((c.get("content") or c.get("text") or "").strip()),
            ),
        )

        seen_contents = set()
        deduped: List[Dict[str, Any]] = []

        for ch in chunks_sorted:
            raw = (
                ch.get("content")
                or ch.get("text")
                or ch.get("chunk_text")
                or ch.get("page_content")
                or ""
            )

            raw = str(raw).strip()

            if not raw:
                continue

            sanitized = self._sanitize_context(raw)

            if not sanitized:
                continue

            trimmed = self._trim_chunk(
                sanitized,
                max_chunk_chars=max_chunk_chars,
            )

            if not trimmed:
                continue

            fingerprint = self._content_fingerprint(trimmed)

            if fingerprint in seen_contents:
                continue

            seen_contents.add(fingerprint)

            new_ch = dict(ch)
            new_ch["content"] = trimmed
            new_ch["text"] = trimmed

            new_ch.setdefault(
                "chunk_id",
                ch.get("chunk_id")
                or ch.get("id")
                or f"{ch.get('document_id', 'doc')}_{len(deduped) + 1}",
            )

            new_ch.setdefault("document_title", ch.get("document_title"))
            new_ch.setdefault("document_url", ch.get("document_url"))

            if normalized_document_scope:
                new_ch.setdefault(
                    "complete_document_id",
                    normalized_document_scope.get("complete_document_id"),
                )
                new_ch.setdefault(
                    "document_title",
                    normalized_document_scope.get("document_name"),
                )

            deduped.append(new_ch)

        debug_id(
            f"[ContextBuilder] Deduped to {len(deduped)} chunks "
            f"document_scope_enabled={bool(normalized_document_scope)}",
            request_id,
        )

        used_chunks: List[Dict[str, Any]] = []
        context_parts: List[str] = []
        token_count = 0

        for idx, chunk in enumerate(deduped):
            if len(used_chunks) >= max_chunks:
                break

            content = (chunk.get("content") or "").strip()

            if not content:
                continue

            if self.include_metadata:
                meta = [f"[Source {idx + 1}]"]

                if chunk.get("document_title"):
                    meta.append(f"Title={chunk['document_title']}")

                if chunk.get("chunk_id"):
                    meta.append(f"ChunkID={chunk['chunk_id']}")

                if chunk.get("complete_document_id"):
                    meta.append(f"CompleteDocumentID={chunk['complete_document_id']}")

                header = " | ".join(meta)
                block = f"{header}\n{content}\n\n"
            else:
                block = f"{content}\n\n"

            block_tokens = self._num_tokens(block)

            if block_tokens > max_tokens:
                warning_id(
                    f"[ContextBuilder] Skipping oversized chunk ({block_tokens} tokens)",
                    request_id,
                )
                continue

            if token_count + block_tokens > max_tokens:
                debug_id(
                    "[ContextBuilder] Token limit reached "
                    f"token_count={token_count} next_block_tokens={block_tokens} "
                    f"max_tokens={max_tokens}",
                    request_id,
                )
                break

            context_parts.append(block)
            used_chunks.append(chunk)
            token_count += block_tokens

        context_str = "".join(context_parts).strip()

        info_id(
            f"[ContextBuilder] Final context: {token_count} tokens, "
            f"{len(used_chunks)} chunks, "
            f"max_chunks={max_chunks}, "
            f"max_chunk_chars={max_chunk_chars}, "
            f"document_scope_enabled={bool(normalized_document_scope)}, "
            f"complete_document_id="
            f"{normalized_document_scope.get('complete_document_id') if normalized_document_scope else None}",
            request_id,
        )

        return {
            "context": context_str,
            "used_chunks": used_chunks,
            "context_token_count": token_count,
            "context_limits": {
                "max_tokens": max_tokens,
                "max_chunks": max_chunks,
                "max_chunk_chars": max_chunk_chars,
                "document_scope_enabled": bool(normalized_document_scope),
                "complete_document_id": (
                    normalized_document_scope.get("complete_document_id")
                    if normalized_document_scope
                    else None
                ),
            },
        }