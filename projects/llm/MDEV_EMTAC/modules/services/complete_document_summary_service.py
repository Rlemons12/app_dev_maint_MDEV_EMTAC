from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from sqlalchemy import text
from sqlalchemy.orm import Session

from modules.configuration.log_config import (
    with_request_id,
    debug_id,
    info_id,
    warning_id,
    error_id,
)

from modules.emtacdb.emtacdb_fts import CompleteDocument, Document
from modules.services.document_summary_service import DocumentSummaryService
from modules.services.ai_models_embedding_service import AIModelsEmbeddingService


@dataclass
class _ChunkSnapshot:
    id: Optional[int]
    name: Optional[str]
    content: str


@dataclass
class _CompleteDocumentSnapshot:
    id: int
    title: str
    content: str
    existing_summary: str
    existing_rag_metadata: Dict[str, Any]
    chunks: List[_ChunkSnapshot]


class CompleteDocumentSummaryService:
    """
    DB-backed summary workflow for CompleteDocument.

    Important:
      - Prevents PostgreSQL idle-in-transaction timeout.
      - Loads a lightweight snapshot first.
      - Releases DB transaction before local AI/Qwen generation.
      - Reloads the document only when ready to save.
      - Does not store full chunk_summaries in rag_metadata.
      - CompleteDocument.summary remains a clean human-readable summary.
      - rag_metadata["retrieval_text"] is the expanded retrieval text.
    """

    DEFAULT_MAX_CHUNKS_TO_SUMMARIZE = 40
    DEFAULT_MAX_CHUNK_CHARS = 5000
    DEFAULT_FINAL_INPUT_CHARS = 16000

    IMPORTANT_TERMS = {
        "troubleshooting", "fault", "alarm", "warning", "error", "failure",
        "failed", "calibration", "calibrate", "procedure", "maintenance",
        "safety", "automatic", "manual", "sensor", "reset", "replace",
        "inspect", "adjust", "setup", "operation", "startup", "shutdown",
        "interlock", "guard", "pressure", "vacuum", "motor", "servo",
        "valve", "cylinder", "photoeye", "prox", "switch", "port",
        "seal", "filler", "robot", "controller", "power", "installation",
        "configuration", "parameter", "diagnostic", "adjustment",
    }

    BAD_SIGNAL_WORDS = {
        "what", "does", "say", "about", "chunk", "summary", "combined",
        "information", "procedures", "described", "available", "document",
        "manual", "user", "july", "copyright", "page", "section",
    }

    def __init__(
        self,
        *,
        max_chunks_to_summarize: int = DEFAULT_MAX_CHUNKS_TO_SUMMARIZE,
        max_chunk_chars: int = DEFAULT_MAX_CHUNK_CHARS,
        final_input_chars: int = DEFAULT_FINAL_INPUT_CHARS,
        store_summary_embedding: bool = True,
        release_transaction_during_ai: bool = True,
    ):
        self.max_chunks_to_summarize = max_chunks_to_summarize
        self.max_chunk_chars = max_chunk_chars
        self.final_input_chars = final_input_chars
        self.store_summary_embedding = store_summary_embedding
        self.release_transaction_during_ai = release_transaction_during_ai

        self.document_summary_service = DocumentSummaryService(
            max_input_chars=max_chunk_chars,
        )
        self.embedding_service = AIModelsEmbeddingService()

    @with_request_id
    def summarize_complete_document_chunks(
        self,
        *,
        session: Session,
        complete_document_id: int,
        force: bool = False,
        request_id: Optional[str] = None,
    ) -> Dict[str, Any]:

        snapshot = self._load_snapshot(
            session=session,
            complete_document_id=complete_document_id,
            request_id=request_id,
        )

        if snapshot is None:
            return self._result(
                success=False,
                complete_document_id=complete_document_id,
                status="document_not_found",
            )

        if not force and self._snapshot_already_summarized(snapshot):
            summary_text = snapshot.existing_summary
            rag_metadata = snapshot.existing_rag_metadata or {}
            retrieval_text = rag_metadata.get("retrieval_text") or summary_text

            embedding_stored = False
            if self.store_summary_embedding:
                doc = session.get(CompleteDocument, complete_document_id)
                if doc:
                    embedding_stored = self._store_summary_embedding(
                        session=session,
                        doc=doc,
                        summary_text=retrieval_text,
                        request_id=request_id,
                    )

            return self._result(
                success=True,
                complete_document_id=complete_document_id,
                status="already_summarized",
                summary=summary_text,
                rag_metadata=rag_metadata,
                summary_embedding_stored=embedding_stored,
            )

        if self.release_transaction_during_ai:
            try:
                session.rollback()
                debug_id(
                    f"[CompleteDocumentSummaryService] Released DB transaction before AI "
                    f"| complete_document_id={complete_document_id}",
                    request_id,
                )
            except Exception as e:
                warning_id(
                    f"[CompleteDocumentSummaryService] Could not release transaction before AI "
                    f"| complete_document_id={complete_document_id} | error={e}",
                    request_id,
                )

        result = self._summarize_snapshot(
            snapshot=snapshot,
            request_id=request_id,
        )

        if not result.get("success"):
            return result

        doc = session.get(CompleteDocument, complete_document_id)

        if not doc:
            return self._result(
                success=False,
                complete_document_id=complete_document_id,
                status="document_not_found_on_save",
            )

        self._apply_metadata_to_document(
            doc=doc,
            metadata=result.get("rag_metadata") or {},
            request_id=request_id,
        )

        embedding_stored = False
        if self.store_summary_embedding:
            embedding_stored = self._store_summary_embedding(
                session=session,
                doc=doc,
                summary_text=(
                    (result.get("rag_metadata") or {}).get("retrieval_text")
                    or result.get("summary")
                    or ""
                ),
                request_id=request_id,
            )

        result["summary_embedding_stored"] = embedding_stored

        info_id(
            f"[CompleteDocumentSummaryService] CompleteDocument summary updated "
            f"| complete_document_id={complete_document_id} "
            f"| chunks_summarized={result.get('chunks_summarized')} "
            f"| summary_embedding_stored={embedding_stored}",
            request_id,
        )

        return result

    @with_request_id
    def summarize_many_complete_documents(
        self,
        *,
        session: Session,
        complete_document_ids: List[int],
        force: bool = False,
        request_id: Optional[str] = None,
    ) -> Dict[str, Any]:

        results = [
            self.summarize_complete_document_chunks(
                session=session,
                complete_document_id=complete_document_id,
                force=force,
                request_id=request_id,
            )
            for complete_document_id in complete_document_ids
        ]

        return {
            "success": True,
            "documents_requested": len(complete_document_ids),
            "documents_processed": len(results),
            "summarized": sum(
                1
                for r in results
                if r.get("status") in {
                    "summarized_from_chunks",
                    "summarized_from_content",
                }
            ),
            "skipped": sum(1 for r in results if r.get("status") == "already_summarized"),
            "failed": sum(1 for r in results if not r.get("success")),
            "summary_embeddings_stored": sum(
                1 for r in results if r.get("summary_embedding_stored")
            ),
            "results": results,
        }

    def _load_snapshot(
        self,
        *,
        session: Session,
        complete_document_id: int,
        request_id: Optional[str],
    ) -> Optional[_CompleteDocumentSnapshot]:
        try:
            doc = session.get(CompleteDocument, complete_document_id)

            if not doc:
                return None

            chunks = (
                session.query(Document)
                .filter(Document.complete_document_id == complete_document_id)
                .order_by(Document.id.asc())
                .all()
            )

            chunk_snapshots = []

            for chunk in chunks:
                content = self._clean_text(getattr(chunk, "content", None))
                if not content:
                    continue

                chunk_snapshots.append(
                    _ChunkSnapshot(
                        id=getattr(chunk, "id", None),
                        name=getattr(chunk, "name", None),
                        content=content,
                    )
                )

            rag_metadata = getattr(doc, "rag_metadata", None)
            if not isinstance(rag_metadata, dict):
                rag_metadata = {}

            return _CompleteDocumentSnapshot(
                id=int(doc.id),
                title=self._clean_title(getattr(doc, "title", "") or f"Document {doc.id}"),
                content=self._clean_text(getattr(doc, "content", None)),
                existing_summary=self._clean_text(getattr(doc, "summary", None)),
                existing_rag_metadata=dict(rag_metadata),
                chunks=chunk_snapshots,
            )

        except Exception as e:
            error_id(
                f"[CompleteDocumentSummaryService] Failed loading snapshot "
                f"| complete_document_id={complete_document_id} | error={e}",
                request_id,
                exc_info=True,
            )
            return None

    def _summarize_snapshot(
        self,
        *,
        snapshot: _CompleteDocumentSnapshot,
        request_id: Optional[str],
    ) -> Dict[str, Any]:

        if snapshot.chunks:
            selected_chunks = self._select_chunk_snapshots(snapshot.chunks)
            chunk_metadata: List[Dict[str, Any]] = []

            for index, chunk in enumerate(selected_chunks, start=1):
                try:
                    chunk_summary = self.document_summary_service.generate_metadata(
                        title=self._clean_chunk_title(snapshot.title, index),
                        content=self._limit_text(chunk.content, self.max_chunk_chars),
                        source_type="document_chunk",
                        extraction_method="chunk_summary",
                        request_id=request_id,
                    )

                    if not isinstance(chunk_summary, dict):
                        continue

                    chunk_metadata.append(
                        {
                            "summary": self._clean_generated_summary(
                                chunk_summary.get("summary") or ""
                            ),
                            "topics": self._clean_signal_list(
                                chunk_summary.get("topics") or [],
                                limit=8,
                            ),
                            "keywords": self._clean_signal_list(
                                chunk_summary.get("keywords") or [],
                                limit=12,
                            ),
                            "questions_answered": self._clean_questions(
                                chunk_summary.get("questions_answered") or [],
                                limit=5,
                            ),
                            "equipment": self._clean_signal_list(
                                chunk_summary.get("equipment") or [],
                                limit=8,
                            ),
                        }
                    )

                except Exception as e:
                    warning_id(
                        f"[CompleteDocumentSummaryService] Chunk summary failed "
                        f"| complete_document_id={snapshot.id} "
                        f"| chunk_id={chunk.id} | error={e}",
                        request_id,
                    )

            if not chunk_metadata:
                return self._summarize_content_snapshot(
                    snapshot=snapshot,
                    request_id=request_id,
                )

            final_metadata = self._combine_chunk_metadata(
                title=snapshot.title,
                chunk_metadata=chunk_metadata,
                chunks_available=len(snapshot.chunks),
                chunks_summarized=len(chunk_metadata),
                request_id=request_id,
            )

            return self._result(
                success=True,
                complete_document_id=snapshot.id,
                status="summarized_from_chunks",
                summary=final_metadata.get("summary") or "",
                rag_metadata=final_metadata,
                chunks_available=len(snapshot.chunks),
                chunks_summarized=len(chunk_metadata),
                summary_embedding_stored=False,
            )

        return self._summarize_content_snapshot(
            snapshot=snapshot,
            request_id=request_id,
        )

    def _summarize_content_snapshot(
        self,
        *,
        snapshot: _CompleteDocumentSnapshot,
        request_id: Optional[str],
    ) -> Dict[str, Any]:

        if not snapshot.content:
            return self._result(
                success=False,
                complete_document_id=snapshot.id,
                status="no_chunks_or_content",
            )

        metadata = self.document_summary_service.generate_metadata(
            title=snapshot.title,
            content=self._limit_text(snapshot.content, self.final_input_chars),
            source_type="complete_document_content_fallback",
            extraction_method="content_sample_summary",
            request_id=request_id,
        )

        if not isinstance(metadata, dict):
            metadata = {}

        final_metadata = self._normalize_final_metadata(
            title=snapshot.title,
            metadata=metadata,
            status="summarized_from_content",
            chunks_available=0,
            chunks_summarized=0,
        )

        return self._result(
            success=True,
            complete_document_id=snapshot.id,
            status="summarized_from_content",
            summary=final_metadata.get("summary") or "",
            rag_metadata=final_metadata,
            chunks_available=0,
            chunks_summarized=0,
            summary_embedding_stored=False,
        )

    def _combine_chunk_metadata(
        self,
        *,
        title: str,
        chunk_metadata: List[Dict[str, Any]],
        chunks_available: int,
        chunks_summarized: int,
        request_id: Optional[str],
    ) -> Dict[str, Any]:

        reduce_input = self._build_clean_reduce_input(
            title=title,
            chunk_metadata=chunk_metadata,
        )

        try:
            metadata = self.document_summary_service.generate_metadata(
                title=title,
                content=self._limit_text(reduce_input, self.final_input_chars),
                source_type="combined_chunk_summaries",
                extraction_method="clean_map_reduce_chunk_summary",
                request_id=request_id,
            )

            if not isinstance(metadata, dict):
                metadata = {}

        except Exception as e:
            warning_id(
                f"[CompleteDocumentSummaryService] Final reduce summary failed "
                f"| title={title} | error={e}",
                request_id,
            )
            metadata = {}

        fallback = self._fallback_reduce_metadata(
            title=title,
            chunk_metadata=chunk_metadata,
        )

        merged_metadata = {
            "summary": (
                self._clean_generated_summary(metadata.get("summary") or "")
                or fallback.get("summary")
                or ""
            ),
            "topics": self._merge_lists(
                self._clean_signal_list(metadata.get("topics") or [], limit=8),
                fallback.get("topics"),
                limit=8,
            ),
            "keywords": self._merge_lists(
                self._clean_signal_list(metadata.get("keywords") or [], limit=15),
                fallback.get("keywords"),
                limit=15,
            ),
            "questions_answered": self._merge_lists(
                self._clean_questions(metadata.get("questions_answered") or [], limit=10),
                fallback.get("questions_answered"),
                limit=10,
            ),
            "equipment": self._merge_lists(
                self._clean_signal_list(metadata.get("equipment") or [], limit=10),
                fallback.get("equipment"),
                limit=10,
            ),
        }

        return self._normalize_final_metadata(
            title=title,
            metadata=merged_metadata,
            status="summarized_from_chunks",
            chunks_available=chunks_available,
            chunks_summarized=chunks_summarized,
        )

    def _normalize_final_metadata(
        self,
        *,
        title: str,
        metadata: Dict[str, Any],
        status: str,
        chunks_available: int,
        chunks_summarized: int,
    ) -> Dict[str, Any]:

        summary = self._clean_generated_summary(metadata.get("summary") or "")
        if not summary:
            summary = f"{title}: Summary generated from document text."

        topics = self._clean_signal_list(metadata.get("topics") or [], limit=8)
        keywords = self._clean_signal_list(metadata.get("keywords") or [], limit=15)
        questions_answered = self._clean_questions(
            metadata.get("questions_answered") or [],
            limit=10,
        )
        equipment = self._clean_signal_list(metadata.get("equipment") or [], limit=10)

        if not questions_answered:
            questions_answered = self._generate_clean_questions(
                title=title,
                topics=topics,
                keywords=keywords,
                limit=10,
            )

        retrieval_text = self._build_retrieval_text(
            title=title,
            summary=summary,
            topics=topics,
            keywords=keywords,
            equipment=equipment,
            questions_answered=questions_answered,
        )

        return {
            "summary": summary,
            "topics": topics,
            "keywords": keywords,
            "questions_answered": questions_answered,
            "equipment": equipment,
            "retrieval_text": retrieval_text,
            "source_type": "complete_document",
            "extraction_method": (
                "clean_chunk_map_reduce_summary"
                if status == "summarized_from_chunks"
                else "content_sample_summary"
            ),
            "generated_by": "CompleteDocumentSummaryService",
            "status": status,
            "chunks_available": chunks_available,
            "chunk_summary_count": chunks_summarized,
        }

    def _apply_metadata_to_document(
        self,
        *,
        doc: CompleteDocument,
        metadata: Dict[str, Any],
        request_id: Optional[str],
    ) -> None:

        if not doc or not metadata:
            return

        rag_metadata = dict(metadata)
        rag_metadata.pop("chunk_summaries", None)

        assignments = {
            "summary": metadata.get("summary"),
            "rag_metadata": rag_metadata,
            "topics": metadata.get("topics"),
            "keywords": metadata.get("keywords"),
            "questions_answered": metadata.get("questions_answered"),
            "equipment": metadata.get("equipment"),
        }

        for field_name, value in assignments.items():
            if value is None:
                continue

            if hasattr(doc, field_name):
                setattr(doc, field_name, value)
            else:
                warning_id(
                    f"[CompleteDocumentSummaryService] CompleteDocument missing field "
                    f"'{field_name}', skipping",
                    request_id,
                )

        try:
            metadata_size = len(json.dumps(rag_metadata, ensure_ascii=False))
            info_id(
                f"[CompleteDocumentSummaryService] Prepared metadata save "
                f"| complete_document_id={getattr(doc, 'id', None)} "
                f"| rag_metadata_bytes={metadata_size}",
                request_id,
            )
        except Exception:
            pass

    def _store_summary_embedding(
        self,
        *,
        session: Session,
        doc: CompleteDocument,
        summary_text: str,
        request_id: Optional[str],
    ) -> bool:

        summary_text = self._clean_text(summary_text)

        if not summary_text:
            return False

        if not hasattr(doc, "summary_embedding_vector"):
            warning_id(
                "[CompleteDocumentSummaryService] CompleteDocument missing "
                "summary_embedding_vector ORM field; skipping summary embedding",
                request_id,
            )
            return False

        try:
            vector = self.embedding_service.get_embeddings(
                summary_text,
                request_id=request_id,
            )

            if not vector:
                return False

            pgvector_value = "[" + ",".join(str(float(v)) for v in vector) + "]"

            session.execute(
                text(
                    """
                    UPDATE complete_document
                    SET summary_embedding_vector = CAST(:embedding AS vector)
                    WHERE id = :complete_document_id
                    """
                ),
                {
                    "embedding": pgvector_value,
                    "complete_document_id": doc.id,
                },
            )

            debug_id(
                f"[CompleteDocumentSummaryService] Stored summary embedding "
                f"| complete_document_id={doc.id} | dim={len(vector)}",
                request_id,
            )

            return True

        except Exception as e:
            warning_id(
                f"[CompleteDocumentSummaryService] Summary embedding skipped "
                f"| complete_document_id={getattr(doc, 'id', None)} | error={e}",
                request_id,
            )
            return False

    def _snapshot_already_summarized(self, snapshot: _CompleteDocumentSnapshot) -> bool:
        return bool(
            snapshot.existing_summary
            or (
                isinstance(snapshot.existing_rag_metadata, dict)
                and snapshot.existing_rag_metadata.get("summary")
            )
        )

    def _select_chunk_snapshots(
        self,
        chunks: List[_ChunkSnapshot],
    ) -> List[_ChunkSnapshot]:
        if len(chunks) <= self.max_chunks_to_summarize:
            return chunks

        selected: List[_ChunkSnapshot] = []
        selected.extend(chunks[:5])
        selected.extend(chunks[-5:])

        middle_index = len(chunks) // 2
        selected.extend(
            chunks[max(0, middle_index - 3): min(len(chunks), middle_index + 3)]
        )

        scored = []

        for chunk in chunks:
            content = self._clean_text(chunk.content).lower()
            score = sum(1 for term in self.IMPORTANT_TERMS if term in content)

            if score > 0:
                scored.append((score, chunk))

        scored.sort(key=lambda item: item[0], reverse=True)

        for _score, chunk in scored:
            selected.append(chunk)
            if len(selected) >= self.max_chunks_to_summarize:
                break

        deduped: List[_ChunkSnapshot] = []
        seen_ids = set()

        for chunk in selected:
            chunk_id = chunk.id if chunk.id is not None else id(chunk)

            if chunk_id in seen_ids:
                continue

            seen_ids.add(chunk_id)
            deduped.append(chunk)

            if len(deduped) >= self.max_chunks_to_summarize:
                break

        return deduped

    def _build_clean_reduce_input(
        self,
        *,
        title: str,
        chunk_metadata: List[Dict[str, Any]],
    ) -> str:

        parts: List[str] = [
            f"Document title: {title}",
            "",
            "Useful extracted document signals:",
        ]

        for item in chunk_metadata:
            summary = self._clean_generated_summary(item.get("summary") or "")
            topics = self._clean_signal_list(item.get("topics") or [], limit=8)
            keywords = self._clean_signal_list(item.get("keywords") or [], limit=12)
            equipment = self._clean_signal_list(item.get("equipment") or [], limit=8)

            if summary:
                parts.append(f"- Summary: {summary}")

            if topics:
                parts.append(f"- Topics: {', '.join(topics)}")

            if keywords:
                parts.append(f"- Keywords: {', '.join(keywords)}")

            if equipment:
                parts.append(f"- Equipment: {', '.join(equipment)}")

        parts.append("")
        parts.append(
            "Create one clean document-level summary. Do not mention chunk IDs, "
            "chunk names, generated question templates, copyright pages, table of "
            "contents pages, or this prompt. Focus only on what this document helps "
            "a maintenance technician understand, troubleshoot, inspect, configure, "
            "operate, repair, calibrate, or verify."
        )

        return "\n".join(parts).strip()

    def _fallback_reduce_metadata(
        self,
        *,
        title: str,
        chunk_metadata: List[Dict[str, Any]],
    ) -> Dict[str, Any]:

        summaries = self._merge_lists(
            *[
                [self._clean_generated_summary(item.get("summary") or "")]
                for item in chunk_metadata
            ],
            limit=5,
        )

        topics = self._merge_lists(
            *[
                self._clean_signal_list(item.get("topics") or [], limit=8)
                for item in chunk_metadata
            ],
            limit=8,
        )

        keywords = self._merge_lists(
            *[
                self._clean_signal_list(item.get("keywords") or [], limit=12)
                for item in chunk_metadata
            ],
            limit=15,
        )

        equipment = self._merge_lists(
            *[
                self._clean_signal_list(item.get("equipment") or [], limit=8)
                for item in chunk_metadata
            ],
            limit=10,
        )

        questions = self._generate_clean_questions(
            title=title,
            topics=topics,
            keywords=keywords,
            limit=10,
        )

        summary = self._limit_text(
            f"{title}: " + " ".join(summaries[:3]),
            1400,
        )

        return {
            "summary": summary,
            "topics": topics,
            "keywords": keywords,
            "questions_answered": questions,
            "equipment": equipment,
        }

    def _clean_chunk_title(self, title: str, index: int) -> str:
        return f"{self._clean_title(title)} section {index}"

    def _clean_title(self, value: Any) -> str:
        text_value = self._clean_text(value)
        text_value = text_value.replace(" - Combined Chunk Summaries", "")
        text_value = text_value.replace("Combined Chunk Summaries", "")
        return text_value.strip() or "Untitled Document"

    def _clean_generated_summary(self, value: Any) -> str:
        text_value = self._clean_text(value)

        if not text_value:
            return ""

        blocked_prefixes = (
            "CHUNK SUMMARY",
            "Chunk ID:",
            "Name:",
            "Questions Answered:",
            "Topics:",
            "Keywords:",
        )

        lines = []

        for line in text_value.splitlines():
            stripped = line.strip()

            if not stripped:
                continue

            lower = stripped.lower()

            if any(stripped.startswith(prefix) for prefix in blocked_prefixes):
                continue

            if "combined chunk summaries" in lower:
                continue

            if "chunk_" in lower:
                continue

            if lower.startswith("what information is in"):
                continue

            if lower.startswith("what procedures are described in"):
                continue

            if lower.startswith("what troubleshooting information is available in"):
                continue

            lines.append(stripped)

        cleaned = " ".join(lines).strip()

        cleaned = re.sub(
            r"\bchunk[_\-\s]*\d+[_\-\s]*\d*\b:?",
            "",
            cleaned,
            flags=re.I,
        )

        cleaned = cleaned.replace("Document title:", "")
        cleaned = cleaned.replace("Useful extracted document signals:", "")

        return self._limit_text(cleaned.strip(), 1400)

    def _clean_signal_list(
        self,
        values: Any,
        *,
        limit: int = 25,
    ) -> List[str]:
        if values is None:
            return []

        if isinstance(values, str):
            raw_items = re.split(r"[\n,;|]+", values)
        elif isinstance(values, list):
            raw_items = values
        else:
            return []

        cleaned = []
        seen = set()

        for item in raw_items:
            text_value = self._clean_text(item).lower()

            if not text_value:
                continue

            text_value = re.sub(
                r"\bchunk[_\-\s]*\d+[_\-\s]*\d*\b",
                "",
                text_value,
                flags=re.I,
            ).replace("_", " ").strip(" -_:")

            if not text_value:
                continue

            if text_value.startswith("chunk"):
                continue

            if text_value in self.BAD_SIGNAL_WORDS:
                continue

            if len(text_value) < 3:
                continue

            if text_value in seen:
                continue

            seen.add(text_value)
            cleaned.append(text_value)

            if len(cleaned) >= limit:
                break

        return cleaned

    def _clean_questions(
        self,
        values: Any,
        *,
        limit: int = 10,
    ) -> List[str]:
        if not isinstance(values, list):
            return []

        cleaned = []
        seen = set()

        for item in values:
            question = self._clean_text(item)

            if not question:
                continue

            lower = question.lower()

            if "chunk_" in lower:
                continue

            if "combined chunk summaries" in lower:
                continue

            if lower.startswith("what does chunk"):
                continue

            if lower.startswith("what information is in chunk"):
                continue

            if lower.startswith("what procedures are described in chunk"):
                continue

            if lower.startswith("what troubleshooting information is available in chunk"):
                continue

            if lower in seen:
                continue

            seen.add(lower)
            cleaned.append(question)

            if len(cleaned) >= limit:
                break

        return cleaned

    def _generate_clean_questions(
        self,
        *,
        title: str,
        topics: List[str],
        keywords: List[str],
        limit: int = 10,
    ) -> List[str]:
        useful_terms = self._merge_lists(topics, keywords, limit=6)

        questions = [
            f"What does {title} explain?",
            f"What setup, operation, or maintenance procedures are covered in {title}?",
            f"What troubleshooting or safety information is covered in {title}?",
        ]

        for term in useful_terms[:4]:
            questions.append(f"What does {title} say about {term}?")

        return questions[:limit]

    def _build_retrieval_text(
        self,
        *,
        title: str,
        summary: str,
        topics: List[str],
        keywords: List[str],
        equipment: List[str],
        questions_answered: Optional[List[str]] = None,
    ) -> str:
        parts = [
            f"Document title: {title}",
            "",
            f"Document purpose summary: {summary}",
        ]

        if equipment:
            parts.append("")
            parts.append("Equipment, systems, or components mentioned:")
            parts.append(", ".join(equipment[:10]))

        if topics:
            parts.append("")
            parts.append("Main topics covered:")
            parts.append(", ".join(topics[:8]))

        if keywords:
            parts.append("")
            parts.append("Important search keywords:")
            parts.append(", ".join(keywords[:15]))

        if questions_answered:
            parts.append("")
            parts.append("Technician questions this document may help answer:")
            for question in questions_answered[:10]:
                parts.append(f"- {question}")

        parts.append("")
        parts.append(
            "This document may be useful for maintenance, troubleshooting, setup, "
            "calibration, inspection, repair, operation, safety checks, diagnostics, "
            "parts identification, and equipment verification when related terms match."
        )

        return "\n".join(part for part in parts if part is not None).strip()

    def _merge_lists(self, *values: Any, limit: int = 25) -> List[str]:
        merged: List[str] = []
        seen = set()

        for value in values:
            if value is None:
                continue

            if isinstance(value, str):
                items = [value]
            elif isinstance(value, list):
                items = value
            else:
                continue

            for item in items:
                text_value = self._clean_text(item)

                if not text_value:
                    continue

                normalized = text_value.lower()

                if normalized in seen:
                    continue

                seen.add(normalized)
                merged.append(text_value)

                if len(merged) >= limit:
                    return merged

        return merged

    def _result(
        self,
        *,
        success: bool,
        complete_document_id: Optional[int],
        status: str,
        summary: str = "",
        rag_metadata: Optional[Dict[str, Any]] = None,
        chunks_available: int = 0,
        chunks_summarized: int = 0,
        summary_embedding_stored: bool = False,
    ) -> Dict[str, Any]:
        return {
            "success": success,
            "complete_document_id": complete_document_id,
            "status": status,
            "summary": summary,
            "rag_metadata": rag_metadata or {},
            "chunks_available": chunks_available,
            "chunks_summarized": chunks_summarized,
            "summary_embedding_stored": summary_embedding_stored,
        }

    def _clean_text(self, value: Any) -> str:
        if value is None:
            return ""

        if not isinstance(value, str):
            value = str(value)

        value = value.replace("\x00", "").replace("\u0000", "")
        value = re.sub(r"[ \t]+", " ", value)
        value = re.sub(r"\n{3,}", "\n\n", value)

        return value.strip()

    def _limit_text(self, text_value: str, max_chars: int) -> str:
        text_value = self._clean_text(text_value)

        if len(text_value) <= max_chars:
            return text_value

        return text_value[: max_chars - 3].rstrip() + "..."