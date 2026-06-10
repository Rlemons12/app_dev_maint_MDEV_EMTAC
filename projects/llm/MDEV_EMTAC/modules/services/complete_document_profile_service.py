# modules/services/complete_document_profile_service.py

from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Optional

from sqlalchemy.orm import Session

from modules.configuration.log_config import (
    debug_id,
    warning_id,
    error_id,
    with_request_id,
)

from modules.emtacdb.emtacdb_fts import CompleteDocument


class CompleteDocumentProfileService:
    """
    Read-only runtime profile service for CompleteDocument.

    Purpose:
        Load document-level RAG metadata already created by
        CompleteDocumentSummaryService.

    This service does NOT:
        - summarize documents
        - generate embeddings
        - call AI
        - open sessions
        - commit
        - rollback
        - mutate ORM objects

    Intended use:
        Document Mode / "Ask this document"

        Example:
            profile = service.get_profile(
                session=session,
                complete_document_id=252,
                request_id=request_id,
            )

    Returned profile shape:
        {
            "found": True,
            "id": 252,
            "title": "LCFab TSG",
            "rev": "R0",
            "file_path": "...",
            "summary": "...",
            "topics": [...],
            "keywords": [...],
            "questions_answered": [...],
            "equipment": [...],
            "rag_metadata": {...},
            "retrieval_text": "...",
            "profile_text": "..."
        }
    """

    DEFAULT_PROFILE_TEXT_CHARS = 8000

    @with_request_id
    def get_profile(
        self,
        *,
        session: Session,
        complete_document_id: Any,
        request_id: Optional[str] = None,
        max_profile_text_chars: int = DEFAULT_PROFILE_TEXT_CHARS,
    ) -> Dict[str, Any]:
        safe_id = self._safe_int(complete_document_id)

        if safe_id is None:
            warning_id(
                f"[CompleteDocumentProfileService] Invalid complete_document_id={complete_document_id!r}",
                request_id,
            )
            return self._missing_profile(
                complete_document_id=complete_document_id,
                reason="invalid_complete_document_id",
            )

        try:
            doc = session.get(CompleteDocument, safe_id)

            if doc is None:
                warning_id(
                    f"[CompleteDocumentProfileService] CompleteDocument not found id={safe_id}",
                    request_id,
                )
                return self._missing_profile(
                    complete_document_id=safe_id,
                    reason="document_not_found",
                )

            rag_metadata = self._safe_dict(getattr(doc, "rag_metadata", None))

            summary = self._clean_text(
                getattr(doc, "summary", None)
                or rag_metadata.get("summary")
                or ""
            )

            topics = self._safe_list(
                getattr(doc, "topics", None)
                if getattr(doc, "topics", None) is not None
                else rag_metadata.get("topics")
            )

            keywords = self._safe_list(
                getattr(doc, "keywords", None)
                if getattr(doc, "keywords", None) is not None
                else rag_metadata.get("keywords")
            )

            questions_answered = self._safe_list(
                getattr(doc, "questions_answered", None)
                if getattr(doc, "questions_answered", None) is not None
                else rag_metadata.get("questions_answered")
            )

            equipment = self._safe_list(
                getattr(doc, "equipment", None)
                if getattr(doc, "equipment", None) is not None
                else rag_metadata.get("equipment")
            )

            retrieval_text = self._clean_text(
                rag_metadata.get("retrieval_text")
                or self._build_retrieval_text(
                    title=self._title(doc),
                    summary=summary,
                    topics=topics,
                    keywords=keywords,
                    equipment=equipment,
                    questions_answered=questions_answered,
                )
            )

            profile_text = self._build_profile_text(
                title=self._title(doc),
                rev=self._clean_text(getattr(doc, "rev", None)),
                summary=summary,
                topics=topics,
                keywords=keywords,
                equipment=equipment,
                questions_answered=questions_answered,
                retrieval_text=retrieval_text,
                max_chars=max_profile_text_chars,
            )

            profile = {
                "found": True,
                "id": safe_id,
                "complete_document_id": safe_id,
                "title": self._title(doc),
                "document_name": self._title(doc),
                "rev": self._clean_text(getattr(doc, "rev", None)),
                "file_path": self._clean_text(getattr(doc, "file_path", None)),
                "file_basename": self._clean_text(getattr(doc, "file_basename", None)),
                "source_type": self._clean_text(getattr(doc, "source_type", None)),
                "extraction_method": self._clean_text(
                    getattr(doc, "extraction_method", None)
                ),
                "summary": summary,
                "topics": topics,
                "keywords": keywords,
                "questions_answered": questions_answered,
                "equipment": equipment,
                "rag_metadata": rag_metadata,
                "retrieval_text": retrieval_text,
                "profile_text": profile_text,
                "has_summary": bool(summary),
                "has_retrieval_text": bool(retrieval_text),
                "has_profile_signals": bool(
                    summary
                    or topics
                    or keywords
                    or questions_answered
                    or equipment
                    or retrieval_text
                ),
            }

            debug_id(
                "[CompleteDocumentProfileService] Loaded profile "
                f"complete_document_id={safe_id} "
                f"title={profile['title']!r} "
                f"has_summary={profile['has_summary']} "
                f"topics={len(topics)} "
                f"keywords={len(keywords)} "
                f"questions_answered={len(questions_answered)} "
                f"equipment={len(equipment)} "
                f"profile_text_chars={len(profile_text)}",
                request_id,
            )

            return profile

        except Exception as exc:
            error_id(
                "[CompleteDocumentProfileService] Failed to load profile "
                f"complete_document_id={safe_id}: {type(exc).__name__}: {exc}",
                request_id,
                exc_info=True,
            )

            return self._missing_profile(
                complete_document_id=safe_id,
                reason="profile_load_error",
            )

    @with_request_id
    def get_profile_text(
        self,
        *,
        session: Session,
        complete_document_id: Any,
        request_id: Optional[str] = None,
        max_profile_text_chars: int = DEFAULT_PROFILE_TEXT_CHARS,
    ) -> str:
        profile = self.get_profile(
            session=session,
            complete_document_id=complete_document_id,
            request_id=request_id,
            max_profile_text_chars=max_profile_text_chars,
        )

        return str(profile.get("profile_text") or "").strip()

    @staticmethod
    def is_overview_question(question: str) -> bool:
        """
        True when the user is asking for a document-level overview/summary.

        These questions should use CompleteDocument.summary/profile instead of
        semantic chunk retrieval.

        Important:
            This must catch natural Ask This Document phrasing such as:
                - What does this document talk about?
                - What is this document about?
                - What does this cover?
                - Summarize this document.
                - Give me an overview.
        """

        q = CompleteDocumentProfileService._normalize_question(question)

        if not q:
            return False

        patterns = [
            # ----------------------------------------------------------
            # Direct "about" phrasing
            # ----------------------------------------------------------
            r"\bwhat\s+is\s+this\s+(document\s+)?about\b",
            r"\bwhat\s+is\s+this\s+about\b",
            r"\bwhat\s+is\s+it\s+about\b",
            r"\bwhat\s+is\s+the\s+(document|manual|guide|procedure)\s+about\b",
            r"\bwhat\s+is\s+this\s+(manual|guide|procedure)\s+about\b",

            # ----------------------------------------------------------
            # Common contraction normalized from "what's"
            # ----------------------------------------------------------
            r"\bwhat\s+is\s+this\s+(document|manual|guide|procedure)\b",
            r"\bwhat\s+is\s+this\b",
            r"\bwhat\s+is\s+it\b",

            # ----------------------------------------------------------
            # "talk about" / "discuss" phrasing
            # ----------------------------------------------------------
            r"\bwhat\s+does\s+this\s+(document|manual|guide|procedure)\s+talk\s+about\b",
            r"\bwhat\s+does\s+the\s+(document|manual|guide|procedure)\s+talk\s+about\b",
            r"\bwhat\s+does\s+this\s+talk\s+about\b",
            r"\bwhat\s+does\s+it\s+talk\s+about\b",
            r"\bwhat\s+is\s+this\s+(document|manual|guide|procedure)\s+talking\s+about\b",
            r"\bwhat\s+is\s+this\s+talking\s+about\b",
            r"\bwhat\s+does\s+this\s+(document|manual|guide|procedure)\s+discuss\b",
            r"\bwhat\s+does\s+the\s+(document|manual|guide|procedure)\s+discuss\b",
            r"\bwhat\s+does\s+this\s+discuss\b",
            r"\bwhat\s+does\s+it\s+discuss\b",

            # ----------------------------------------------------------
            # Cover/explain/describe phrasing
            # ----------------------------------------------------------
            r"\bwhat\s+does\s+this\s+(document|manual|guide|procedure)\s+(cover|explain|describe|include)\b",
            r"\bwhat\s+does\s+the\s+(document|manual|guide|procedure)\s+(cover|explain|describe|include)\b",
            r"\bwhat\s+does\s+this\s+(cover|explain|describe|include)\b",
            r"\bwhat\s+does\s+it\s+(cover|explain|describe|include)\b",
            r"\bwhat\s+is\s+covered\s+in\s+(this|the)\s+(document|manual|guide|procedure)\b",
            r"\bwhat\s+is\s+included\s+in\s+(this|the)\s+(document|manual|guide|procedure)\b",
            r"\bwhat\s+information\s+is\s+in\s+(this|the)\s+(document|manual|guide|procedure)\b",

            # ----------------------------------------------------------
            # Summary / overview phrasing
            # ----------------------------------------------------------
            r"\bsummarize\s+(this|the)\s*(document|manual|guide|procedure)?\b",
            r"\bsummary\s+of\s+(this|the)\s+(document|manual|guide|procedure)\b",
            r"\bgive\s+me\s+(a\s+)?summary\s+of\s+(this|the)\s*(document|manual|guide|procedure)?\b",
            r"\bgive\s+me\s+(an\s+)?overview\s+of\s+(this|the)\s*(document|manual|guide|procedure)?\b",
            r"\boverview\s+of\s+(this|the)\s+(document|manual|guide|procedure)\b",
            r"\bgive\s+me\s+(a\s+)?document\s+overview\b",
            r"\bdocument\s+overview\b",
            r"\bmanual\s+overview\b",
            r"\bprocedure\s+overview\b",

            # ----------------------------------------------------------
            # Purpose / type / topics
            # ----------------------------------------------------------
            r"\bwhat\s+kind\s+of\s+(document|manual|guide|procedure)\s+is\s+this\b",
            r"\bwhat\s+type\s+of\s+(document|manual|guide|procedure)\s+is\s+this\b",
            r"\bwhat\s+is\s+the\s+purpose\s+of\s+(this|the)\s+(document|manual|guide|procedure)\b",
            r"\bwhat\s+is\s+this\s+(document|manual|guide|procedure)\s+for\b",
            r"\bwhat\s+topics\s+(are\s+)?(covered|included|discussed)\s+in\s+(this|the)\s+(document|manual|guide|procedure)\b",
            r"\bwhat\s+are\s+the\s+main\s+topics\s+in\s+(this|the)\s+(document|manual|guide|procedure)\b",
            r"\bwhat\s+can\s+this\s+(document|manual|guide|procedure)\s+help\s+with\b",
            r"\bwhat\s+can\s+this\s+help\s+with\b",
        ]

        return any(re.search(pattern, q, flags=re.IGNORECASE) for pattern in patterns)

    @staticmethod
    def is_coverage_question(question: str) -> bool:
        """
        True when the user asks whether the selected document discusses a topic.

        These can use profile signals first, then fall back to scoped chunk
        retrieval for proof/details.

        Examples:
            - Does this document talk about sensors?
            - Does it cover pre-start checks?
            - Is there anything about gripper condition?
        """

        q = CompleteDocumentProfileService._normalize_question(question)

        if not q:
            return False

        patterns = [
            r"\bdoes\s+this\s*(document|manual|guide|procedure)?\s*(talk\s+about|cover|mention|include|explain|describe|discuss)\b",
            r"\bdoes\s+the\s+(document|manual|guide|procedure)\s*(talk\s+about|cover|mention|include|explain|describe|discuss)\b",
            r"\bdoes\s+it\s+(talk\s+about|cover|mention|include|explain|describe|discuss)\b",
            r"\bis\s+there\s+(anything\s+)?(about|on|for)\b",
            r"\bis\s+(.*?)\s+(covered|mentioned|included|explained|described|discussed)\b",
            r"\bcan\s+this\s*(document|manual|guide|procedure)?\s*(help|show|explain|describe)\b",
            r"\bwould\s+this\s*(document|manual|guide|procedure)?\s*(help|show|explain|describe)\b",
            r"\bcan\s+this\s+help\s+me\s+understand\b",
            r"\bdoes\s+this\s+help\s+with\b",
            r"\bdoes\s+it\s+help\s+with\b",
        ]

        return any(re.search(pattern, q, flags=re.IGNORECASE) for pattern in patterns)

    @staticmethod
    def build_overview_answer_from_profile(profile: Dict[str, Any]) -> str:
        """
        Deterministic answer for document overview questions.

        This avoids asking the LLM to summarize random semantically-retrieved
        chunks when the user asks broad questions like:
            "What is this document about?"
            "What does this document talk about?"
        """

        if not isinstance(profile, dict) or not profile.get("found"):
            return "I could not find the selected document profile."

        title = CompleteDocumentProfileService._clean_text(
            profile.get("title")
            or profile.get("document_name")
            or "Selected document"
        )
        summary = CompleteDocumentProfileService._clean_text(profile.get("summary"))
        topics = CompleteDocumentProfileService._safe_list(profile.get("topics"))
        equipment = CompleteDocumentProfileService._safe_list(profile.get("equipment"))
        questions = CompleteDocumentProfileService._safe_list(
            profile.get("questions_answered")
        )

        parts: List[str] = []

        if summary:
            parts.append(f"{title}: {summary}")
        else:
            parts.append(
                f"{title}: I found the selected document, but it does not have a saved summary yet."
            )

        if topics:
            parts.append(
                "Main topics: " + ", ".join(str(item) for item in topics[:8]) + "."
            )

        if equipment:
            parts.append(
                "Equipment/components mentioned: "
                + ", ".join(str(item) for item in equipment[:8])
                + "."
            )

        if questions:
            parts.append(
                "It may help answer questions such as: "
                + "; ".join(str(item) for item in questions[:3])
                + "."
            )

        return "\n\n".join(parts).strip()

    @staticmethod
    def build_profile_context(
        *,
        profile: Dict[str, Any],
        max_chars: int = DEFAULT_PROFILE_TEXT_CHARS,
    ) -> str:
        if not isinstance(profile, dict):
            return ""

        text = str(
            profile.get("profile_text")
            or profile.get("retrieval_text")
            or profile.get("summary")
            or ""
        ).strip()

        return CompleteDocumentProfileService._limit_text(text, max_chars)

    @staticmethod
    def _build_profile_text(
        *,
        title: str,
        rev: str,
        summary: str,
        topics: List[str],
        keywords: List[str],
        equipment: List[str],
        questions_answered: List[str],
        retrieval_text: str,
        max_chars: int,
    ) -> str:
        parts: List[str] = [
            f"Selected document title: {title}",
        ]

        if rev:
            parts.append(f"Revision: {rev}")

        if summary:
            parts.extend(["", f"Document summary:\n{summary}"])

        if equipment:
            parts.extend(
                [
                    "",
                    "Equipment/components mentioned:",
                    ", ".join(str(item) for item in equipment[:12]),
                ]
            )

        if topics:
            parts.extend(
                [
                    "",
                    "Main topics:",
                    ", ".join(str(item) for item in topics[:12]),
                ]
            )

        if keywords:
            parts.extend(
                [
                    "",
                    "Important keywords:",
                    ", ".join(str(item) for item in keywords[:20]),
                ]
            )

        if questions_answered:
            parts.append("")
            parts.append("Technician questions this document may help answer:")

            for question in questions_answered[:12]:
                parts.append(f"- {question}")

        if retrieval_text and retrieval_text not in "\n".join(parts):
            parts.extend(["", "Document retrieval profile:", retrieval_text])

        return CompleteDocumentProfileService._limit_text(
            "\n".join(parts).strip(),
            max_chars,
        )

    @staticmethod
    def _build_retrieval_text(
        *,
        title: str,
        summary: str,
        topics: List[str],
        keywords: List[str],
        equipment: List[str],
        questions_answered: List[str],
    ) -> str:
        parts = [
            f"Document title: {title}",
            "",
            f"Document purpose summary: {summary}",
        ]

        if equipment:
            parts.extend(
                [
                    "",
                    "Equipment, systems, or components mentioned:",
                    ", ".join(str(item) for item in equipment[:10]),
                ]
            )

        if topics:
            parts.extend(
                [
                    "",
                    "Main topics covered:",
                    ", ".join(str(item) for item in topics[:8]),
                ]
            )

        if keywords:
            parts.extend(
                [
                    "",
                    "Important search keywords:",
                    ", ".join(str(item) for item in keywords[:15]),
                ]
            )

        if questions_answered:
            parts.append("")
            parts.append("Technician questions this document may help answer:")

            for question in questions_answered[:10]:
                parts.append(f"- {question}")

        parts.extend(
            [
                "",
                (
                    "This document may be useful for maintenance, troubleshooting, setup, "
                    "calibration, inspection, repair, operation, safety checks, diagnostics, "
                    "parts identification, and equipment verification when related terms match."
                ),
            ]
        )

        return "\n".join(str(part) for part in parts if part is not None).strip()

    @staticmethod
    def _missing_profile(
        *,
        complete_document_id: Any,
        reason: str,
    ) -> Dict[str, Any]:
        safe_id = CompleteDocumentProfileService._safe_int(complete_document_id)

        return {
            "found": False,
            "id": safe_id,
            "complete_document_id": safe_id,
            "title": "",
            "document_name": "",
            "rev": "",
            "file_path": "",
            "file_basename": "",
            "source_type": "",
            "extraction_method": "",
            "summary": "",
            "topics": [],
            "keywords": [],
            "questions_answered": [],
            "equipment": [],
            "rag_metadata": {},
            "retrieval_text": "",
            "profile_text": "",
            "has_summary": False,
            "has_retrieval_text": False,
            "has_profile_signals": False,
            "reason": reason,
        }

    @staticmethod
    def _title(doc: CompleteDocument) -> str:
        value = (
            getattr(doc, "title", None)
            or getattr(doc, "file_basename", None)
            or f"Document #{getattr(doc, 'id', '')}"
        )

        return CompleteDocumentProfileService._clean_text(value) or "Selected document"

    @staticmethod
    def _safe_dict(value: Any) -> Dict[str, Any]:
        if isinstance(value, dict):
            return dict(value)

        if isinstance(value, str):
            text_value = value.strip()

            if not text_value:
                return {}

            try:
                parsed = json.loads(text_value)
            except Exception:
                return {}

            if isinstance(parsed, dict):
                return parsed

        return {}

    @staticmethod
    def _safe_list(value: Any) -> List[str]:
        if value is None:
            return []

        if isinstance(value, list):
            raw_items = value
        elif isinstance(value, tuple):
            raw_items = list(value)
        elif isinstance(value, str):
            text_value = value.strip()

            if not text_value:
                return []

            try:
                parsed = json.loads(text_value)
                if isinstance(parsed, list):
                    raw_items = parsed
                else:
                    raw_items = re.split(r"[\n,;|]+", text_value)
            except Exception:
                raw_items = re.split(r"[\n,;|]+", text_value)
        else:
            return []

        cleaned: List[str] = []
        seen = set()

        for item in raw_items:
            text_value = CompleteDocumentProfileService._clean_text(item)

            if not text_value:
                continue

            normalized = text_value.lower()

            if normalized in seen:
                continue

            seen.add(normalized)
            cleaned.append(text_value)

        return cleaned

    @staticmethod
    def _normalize_question(question: str) -> str:
        text_value = (question or "").strip().lower()

        if not text_value:
            return ""

        # Normalize common contractions before punctuation stripping.
        # Without this, "what's this about?" becomes "what s this about",
        # which is harder to match deterministically.
        replacements = {
            "what's": "what is",
            "whats": "what is",
            "what’re": "what are",
            "what're": "what are",
            "it’s": "it is",
            "it's": "it is",
            "that’s": "that is",
            "that's": "that is",
            "there’s": "there is",
            "there's": "there is",
        }

        for source, target in replacements.items():
            text_value = text_value.replace(source, target)

        text_value = re.sub(r"[^\w\s]", " ", text_value)
        text_value = re.sub(r"\s+", " ", text_value).strip()

        return text_value

    @staticmethod
    def _clean_text(value: Any) -> str:
        if value is None:
            return ""

        if not isinstance(value, str):
            value = str(value)

        value = value.replace("\x00", "").replace("\u0000", "")
        value = re.sub(r"[ \t]+", " ", value)
        value = re.sub(r"\n{3,}", "\n\n", value)

        return value.strip()

    @staticmethod
    def _limit_text(value: str, max_chars: int) -> str:
        text_value = CompleteDocumentProfileService._clean_text(value)

        if max_chars <= 0:
            return ""

        if len(text_value) <= max_chars:
            return text_value

        return text_value[: max_chars - 3].rstrip() + "..."

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