from __future__ import annotations

import json
import os
import re
from typing import Any, Dict, List, Optional

import requests

from modules.configuration.log_config import (
    with_request_id,
    debug_id,
    info_id,
    warning_id,
)


class DocumentSummaryService:
    """
    Generates clean RAG metadata for document/chunk text.

    Important:
      - Does NOT use AIStewardManagerService.
      - Does NOT require chat session, user_id, memory, or RAG search.
      - Uses lightweight standalone text-generation paths only.
      - Falls back safely if no AI text generator is available.
    """

    DEFAULT_MAX_INPUT_CHARS = 16000
    DEFAULT_SUMMARY_CHARS = 1400
    DEFAULT_QUESTION_COUNT = 12
    DEFAULT_KEYWORD_COUNT = 25
    DEFAULT_TIMEOUT_SECONDS = 90

    def __init__(
        self,
        max_input_chars: int = DEFAULT_MAX_INPUT_CHARS,
        summary_chars: int = DEFAULT_SUMMARY_CHARS,
        question_count: int = DEFAULT_QUESTION_COUNT,
        keyword_count: int = DEFAULT_KEYWORD_COUNT,
        timeout_seconds: int = DEFAULT_TIMEOUT_SECONDS,
        gpu_generation_model: str = "qwen",
        gpu_max_new_tokens: int = 900,
    ):
        self.max_input_chars = max_input_chars
        self.summary_chars = summary_chars
        self.question_count = question_count
        self.keyword_count = keyword_count
        self.timeout_seconds = timeout_seconds
        self.gpu_generation_model = gpu_generation_model
        self.gpu_max_new_tokens = gpu_max_new_tokens

    @with_request_id
    def generate_metadata(
        self,
        *,
        title: str,
        content: str,
        source_type: Optional[str] = None,
        extraction_method: Optional[str] = None,
        request_id: Optional[str] = None,
    ) -> Dict[str, Any]:

        title = self._clean_title(title)
        content = self._clean_text(content)

        if not content:
            return self._fallback_metadata(
                title=title,
                content="",
                source_type=source_type,
                extraction_method=extraction_method,
                status="empty_content",
                request_id=request_id,
            )

        prompt_text = self._build_prompt_text(content)

        try:
            raw_response = self._call_ai_model(
                title=title,
                content=prompt_text,
                source_type=source_type,
                extraction_method=extraction_method,
                request_id=request_id,
            )

            parsed = self._parse_ai_response(raw_response)

            metadata = {
                "summary": self._limit_text(
                    self._clean_generated_summary(parsed.get("summary")),
                    self.summary_chars,
                ),
                "topics": self._clean_string_list(
                    parsed.get("topics"),
                    max_items=20,
                ),
                "keywords": self._clean_string_list(
                    parsed.get("keywords"),
                    max_items=self.keyword_count,
                ),
                "questions_answered": self._clean_questions(
                    parsed.get("questions_answered"),
                    max_items=self.question_count,
                ),
                "equipment": self._clean_string_list(
                    parsed.get("equipment"),
                    max_items=15,
                ),
                "source_type": source_type,
                "extraction_method": extraction_method,
                "generated_by": "DocumentSummaryService",
                "status": "ai_generated",
            }

            if not metadata["summary"]:
                metadata["summary"] = self._fallback_summary(title, content)

            if not metadata["keywords"]:
                metadata["keywords"] = self._extract_basic_keywords(content)

            if not metadata["topics"]:
                metadata["topics"] = metadata["keywords"][:10]

            if not metadata["questions_answered"]:
                metadata["questions_answered"] = self._fallback_questions(
                    title,
                    metadata["topics"],
                )

            info_id(
                f"[DocumentSummaryService] Generated metadata | "
                f"title='{title}' | topics={len(metadata['topics'])} | "
                f"keywords={len(metadata['keywords'])} | "
                f"questions={len(metadata['questions_answered'])}",
                request_id,
            )

            return metadata

        except Exception as e:
            warning_id(
                f"[DocumentSummaryService] AI metadata generation failed for "
                f"'{title}', using fallback metadata: {e}",
                request_id,
            )

            return self._fallback_metadata(
                title=title,
                content=content,
                source_type=source_type,
                extraction_method=extraction_method,
                status="fallback_after_error",
                request_id=request_id,
            )

    @with_request_id
    def generate_summary(
        self,
        *,
        title: str,
        content: str,
        request_id: Optional[str] = None,
    ) -> str:
        return self.generate_metadata(
            title=title,
            content=content,
            request_id=request_id,
        ).get("summary") or ""

    @with_request_id
    def generate_topics(
        self,
        *,
        title: str,
        content: str,
        request_id: Optional[str] = None,
    ) -> List[str]:
        return self.generate_metadata(
            title=title,
            content=content,
            request_id=request_id,
        ).get("topics") or []

    @with_request_id
    def generate_keywords(
        self,
        *,
        title: str,
        content: str,
        request_id: Optional[str] = None,
    ) -> List[str]:
        return self.generate_metadata(
            title=title,
            content=content,
            request_id=request_id,
        ).get("keywords") or []

    @with_request_id
    def generate_questions_answered(
        self,
        *,
        title: str,
        content: str,
        request_id: Optional[str] = None,
    ) -> List[str]:
        return self.generate_metadata(
            title=title,
            content=content,
            request_id=request_id,
        ).get("questions_answered") or []

    # =========================================================
    # AI CALLS
    # =========================================================

    def _call_ai_model(
        self,
        *,
        title: str,
        content: str,
        source_type: Optional[str],
        extraction_method: Optional[str],
        request_id: Optional[str],
    ) -> str:

        prompt = self._build_prompt(
            title=title,
            content=content,
            source_type=source_type,
            extraction_method=extraction_method,
        )

        result = self._try_ai_models_service(
            prompt=prompt,
            request_id=request_id,
        )
        if result:
            return result

        result = self._try_gpu_adapter_generate(
            prompt=prompt,
            request_id=request_id,
        )
        if result:
            return result

        result = self._try_ai_gateway_http(
            prompt=prompt,
            request_id=request_id,
        )
        if result:
            return result

        result = self._try_gpu_service_http(
            prompt=prompt,
            request_id=request_id,
        )
        if result:
            return result

        raise RuntimeError("No standalone AI text generation service was available")

    def _try_ai_models_service(
        self,
        *,
        prompt: str,
        request_id: Optional[str],
    ) -> str:
        try:
            from modules.services.ai_models_service import AIModelsService

            service = AIModelsService()

            for method_name in (
                "generate_text",
                "generate",
                "complete",
                "completion",
                "chat",
                "ask",
                "run",
            ):
                method = getattr(service, method_name, None)

                if not callable(method):
                    continue

                debug_id(
                    f"[DocumentSummaryService] Trying AIModelsService.{method_name}",
                    request_id,
                )

                try:
                    result = method(
                        prompt=prompt,
                        request_id=request_id,
                    )
                except TypeError:
                    try:
                        result = method(
                            text=prompt,
                            request_id=request_id,
                        )
                    except TypeError:
                        result = method(prompt)

                text = self._response_to_text(result)

                if text:
                    return text

        except ImportError:
            debug_id(
                "[DocumentSummaryService] AIModelsService not available",
                request_id,
            )
        except Exception as e:
            warning_id(
                f"[DocumentSummaryService] AIModelsService text generation failed: {e}",
                request_id,
            )

        return ""

    def _try_gpu_adapter_generate(
        self,
        *,
        prompt: str,
        request_id: Optional[str],
    ) -> str:
        """
        Preferred local GPU text-generation path.

        Uses:
            modules.runtime.gpu_service_adapter_emtac.GPUServerAdapter.generate()

        This avoids AIStewardManagerService and avoids RAG/chat/session dependencies.
        """

        try:
            from modules.runtime.gpu_service_adapter_emtac import GPUServerAdapter

            adapter = GPUServerAdapter(timeout=self.timeout_seconds)

            if not adapter.is_available():
                debug_id(
                    "[DocumentSummaryService] GPU adapter unavailable",
                    request_id,
                )
                return ""

            debug_id(
                "[DocumentSummaryService] Trying GPUServerAdapter.generate",
                request_id,
            )

            result = adapter.generate(
                prompt=prompt,
                model=self.gpu_generation_model,
                max_new_tokens=self.gpu_max_new_tokens,
                temperature=0.1,
                top_p=0.9,
            )

            text = self._response_to_text(result)

            if text:
                return text

        except ImportError:
            debug_id(
                "[DocumentSummaryService] GPUServerAdapter not available",
                request_id,
            )
        except Exception as e:
            warning_id(
                f"[DocumentSummaryService] GPUServerAdapter.generate failed: {e}",
                request_id,
            )

        return ""

    def _try_ai_gateway_http(
        self,
        *,
        prompt: str,
        request_id: Optional[str],
    ) -> str:
        base_url = (
            os.getenv("AI_GATEWAY_URL")
            or os.getenv("EMTAC_AI_GATEWAY_URL")
            or ""
        ).strip().rstrip("/")

        if not base_url:
            return ""

        payloads = [
            {"prompt": prompt},
            {"message": prompt},
            {"question": prompt},
            {"messages": [{"role": "user", "content": prompt}]},
        ]

        endpoints = (
            "/api/ai/generate",
            "/generate",
            "/chat",
            "/completion",
        )

        for endpoint in endpoints:
            url = f"{base_url}{endpoint}"

            for payload in payloads:
                try:
                    debug_id(
                        f"[DocumentSummaryService] Trying AI gateway {url}",
                        request_id,
                    )

                    response = requests.post(
                        url,
                        json=payload,
                        timeout=self.timeout_seconds,
                    )

                    if response.status_code >= 400:
                        continue

                    text = self._response_to_text(response.json())

                    if text:
                        return text

                except Exception:
                    continue

        return ""

    def _try_gpu_service_http(
        self,
        *,
        prompt: str,
        request_id: Optional[str],
    ) -> str:
        base_url = (
            os.getenv("GPU_SERVICE_URL")
            or os.getenv("EMTAC_GPU_SERVICE_URL")
            or os.getenv("SERVICE_GPU_BASE_URL")
            or ""
        ).strip().rstrip("/")

        if not base_url:
            return ""

        endpoints = (
            "/generate",
            "/chat",
            "/complete",
        )

        payloads = [
            {
                "prompt": prompt,
                "model": self.gpu_generation_model,
                "max_new_tokens": self.gpu_max_new_tokens,
                "max_tokens": self.gpu_max_new_tokens,
                "temperature": 0.1,
                "top_p": 0.9,
                "return_full_text": False,
                "echo": False,
                "strip_prompt": True,
            },
            {"message": prompt},
            {"question": prompt},
            {"messages": [{"role": "user", "content": prompt}]},
        ]

        for endpoint in endpoints:
            url = f"{base_url}{endpoint}"

            for payload in payloads:
                try:
                    debug_id(
                        f"[DocumentSummaryService] Trying GPU text endpoint {url}",
                        request_id,
                    )

                    response = requests.post(
                        url,
                        json=payload,
                        timeout=self.timeout_seconds,
                    )

                    if response.status_code >= 400:
                        continue

                    text = self._response_to_text(response.json())

                    if text:
                        return text

                except Exception:
                    continue

        return ""

    # =========================================================
    # PROMPT
    # =========================================================

    def _build_prompt(
        self,
        *,
        title: str,
        content: str,
        source_type: Optional[str],
        extraction_method: Optional[str],
    ) -> str:
        return f"""
You are generating clean RAG metadata for an industrial maintenance document.

Return ONLY valid JSON.
Do not include markdown.
Do not include explanations outside JSON.
Do not mention chunk IDs, chunk names, generated prompts, or this instruction.

Document title:
{title}

Source type:
{source_type or "unknown"}

Extraction method:
{extraction_method or "unknown"}

Document text:
{content}

Return JSON in this exact structure:
{{
  "summary": "Clean document-level summary of what this document helps a technician understand or do.",
  "topics": ["topic 1", "topic 2"],
  "keywords": ["keyword 1", "keyword 2"],
  "questions_answered": [
    "What real technician question could this document help answer?"
  ],
  "equipment": ["equipment or machine name if clearly mentioned"]
}}

Rules:
- Focus on maintenance, troubleshooting, setup, operation, calibration, inspection, repair, diagnostics, safety, parts, alarms, faults, sensors, motors, valves, cylinders, controllers, and equipment behavior.
- Make questions sound like real technician questions.
- Do not create generic questions like "What information is in chunk_1?"
- Do not include words such as chunk, combined chunk summaries, section number, copyright, user manual cover page, or table of contents as topics unless technically meaningful.
- Do not invent alarm numbers, part numbers, procedures, equipment names, or fault causes unless clearly present in the text.
- Keep the summary under {self.summary_chars} characters.
- Return no more than {self.keyword_count} keywords.
- Return no more than {self.question_count} questions.
""".strip()

    def _build_prompt_text(self, content: str) -> str:
        content = self._clean_text(content)

        if len(content) <= self.max_input_chars:
            return content

        head_size = int(self.max_input_chars * 0.45)
        middle_size = int(self.max_input_chars * 0.25)
        tail_size = self.max_input_chars - head_size - middle_size

        middle_start = max((len(content) // 2) - (middle_size // 2), 0)
        middle_end = middle_start + middle_size

        return "\n\n".join(
            [
                content[:head_size],
                "[... middle sample ...]",
                content[middle_start:middle_end],
                "[... ending sample ...]",
                content[-tail_size:],
            ]
        )

    # =========================================================
    # PARSING
    # =========================================================

    def _parse_ai_response(self, raw_response: Any) -> Dict[str, Any]:
        text = self._response_to_text(raw_response)

        if not text:
            return {}

        text = text.strip()

        text = re.sub(r"^```(?:json)?", "", text, flags=re.IGNORECASE).strip()
        text = re.sub(r"```$", "", text).strip()

        try:
            parsed = json.loads(text)
            return parsed if isinstance(parsed, dict) else {}
        except Exception:
            pass

        json_match = re.search(r"\{.*\}", text, flags=re.DOTALL)

        if not json_match:
            return {}

        try:
            parsed = json.loads(json_match.group(0))
            return parsed if isinstance(parsed, dict) else {}
        except Exception:
            return {}

    def _response_to_text(self, response: Any) -> str:
        if response is None:
            return ""

        if isinstance(response, str):
            return response.strip()

        if isinstance(response, dict):
            for key in (
                "answer",
                "response",
                "content",
                "text",
                "message",
                "result",
                "output",
                "generated_text",
                "completion",
            ):
                value = response.get(key)

                if isinstance(value, str) and value.strip():
                    return value.strip()

                if isinstance(value, dict):
                    nested = self._response_to_text(value)
                    if nested:
                        return nested

                if isinstance(value, list):
                    nested = self._response_to_text(value)
                    if nested:
                        return nested

            choices = response.get("choices")
            if isinstance(choices, list) and choices:
                return self._response_to_text(choices[0])

            return ""

        if isinstance(response, list):
            for item in response:
                text = self._response_to_text(item)
                if text:
                    return text
            return ""

        for attr in ("answer", "content", "text", "message", "result", "output"):
            if hasattr(response, attr):
                value = getattr(response, attr)
                text = self._response_to_text(value)
                if text:
                    return text

        return str(response).strip()

    # =========================================================
    # FALLBACKS
    # =========================================================

    def _fallback_metadata(
        self,
        *,
        title: str,
        content: str,
        source_type: Optional[str],
        extraction_method: Optional[str],
        status: str,
        request_id: Optional[str],
    ) -> Dict[str, Any]:
        keywords = self._extract_basic_keywords(content)
        topics = keywords[:10]
        questions = self._fallback_questions(title, topics)

        metadata = {
            "summary": self._fallback_summary(title, content),
            "topics": topics,
            "keywords": keywords,
            "questions_answered": questions,
            "equipment": self._extract_possible_equipment(content),
            "source_type": source_type,
            "extraction_method": extraction_method,
            "generated_by": "DocumentSummaryService",
            "status": status,
        }

        warning_id(
            f"[DocumentSummaryService] Using fallback metadata | "
            f"title='{title}' | status={status}",
            request_id,
        )

        return metadata

    def _fallback_summary(self, title: str, content: str) -> str:
        clean = self._clean_text(content)

        if not clean:
            return (
                f"{title} is a stored document, but no extractable text was "
                f"available for summary generation."
            )

        first_text = self._first_sentences(clean, max_chars=self.summary_chars)

        return self._limit_text(
            f"{title}: {first_text}",
            self.summary_chars,
        )

    def _fallback_questions(self, title: str, topics: List[str]) -> List[str]:
        questions = [
            f"What does {title} explain?",
            f"What setup, operation, or maintenance procedures are covered in {title}?",
            f"What troubleshooting or safety information is covered in {title}?",
        ]

        for topic in topics[:6]:
            questions.append(f"What does {title} say about {topic}?")

        return questions[: self.question_count]

    def _extract_basic_keywords(self, content: str) -> List[str]:
        text = self._clean_text(content).lower()

        if not text:
            return []

        stop_words = {
            "the", "and", "for", "with", "that", "this", "from", "are", "was",
            "were", "you", "your", "have", "has", "had", "not", "but", "all",
            "can", "will", "shall", "may", "into", "then", "than", "when",
            "where", "what", "which", "while", "been", "being", "also", "each",
            "page", "section", "manual", "document", "figure", "table",
            "chunk", "summary", "combined", "information", "described",
            "available", "copyright", "user", "july",
        }

        words = re.findall(r"\b[a-zA-Z][a-zA-Z0-9_\-/]{2,}\b", text)

        counts: Dict[str, int] = {}

        for word in words:
            word = word.strip("-_/").lower()

            if len(word) < 3:
                continue

            if word in stop_words:
                continue

            counts[word] = counts.get(word, 0) + 1

        sorted_words = sorted(
            counts.items(),
            key=lambda item: item[1],
            reverse=True,
        )

        return [word for word, _count in sorted_words[: self.keyword_count]]

    def _extract_possible_equipment(self, content: str) -> List[str]:
        text = self._clean_text(content)

        if not text:
            return []

        candidates = re.findall(
            r"\b(?:[A-Z][A-Za-z0-9\-]{2,})(?:\s+[A-Z][A-Za-z0-9\-]{2,}){0,4}\b",
            text,
        )

        cleaned: List[str] = []
        seen = set()

        bad = {
            "WARNING",
            "CAUTION",
            "NOTE",
            "Table",
            "Figure",
            "Page",
            "Copyright",
            "Manual",
            "User Manual",
        }

        for candidate in candidates:
            candidate = self._clean_text(candidate)

            if not candidate:
                continue

            if candidate in bad:
                continue

            if len(candidate) < 4:
                continue

            normalized = candidate.lower()

            if normalized in seen:
                continue

            seen.add(normalized)
            cleaned.append(candidate)

            if len(cleaned) >= 15:
                break

        return cleaned

    # =========================================================
    # CLEANING
    # =========================================================

    def _clean_title(self, value: Any) -> str:
        text = self._clean_text(value)

        text = text.replace(" - Combined Chunk Summaries", "")
        text = text.replace("Combined Chunk Summaries", "")
        text = re.sub(r"\bchunk[_\-\s]*\d+[_\-\s]*\d*\b", "", text, flags=re.I)

        return text.strip(" -_:") or "Untitled Document"

    def _clean_generated_summary(self, value: Any) -> str:
        text = self._clean_text(value)

        if not text:
            return ""

        lines: List[str] = []

        blocked_prefixes = (
            "CHUNK SUMMARY",
            "Chunk ID:",
            "Name:",
            "Questions Answered:",
            "Topics:",
            "Keywords:",
        )

        for line in text.splitlines():
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

        cleaned = " ".join(lines)

        cleaned = re.sub(
            r"\bchunk[_\-\s]*\d+[_\-\s]*\d*\b:?",
            "",
            cleaned,
            flags=re.I,
        )

        return self._limit_text(cleaned.strip(), self.summary_chars)

    def _clean_questions(
        self,
        value: Any,
        *,
        max_items: int,
    ) -> List[str]:
        items = self._clean_string_list(value, max_items=max_items * 2)

        cleaned: List[str] = []
        seen = set()

        for question in items:
            lower = question.lower()

            if "chunk_" in lower:
                continue

            if "combined chunk summaries" in lower:
                continue

            if lower.startswith("what information is in chunk"):
                continue

            if lower.startswith("what procedures are described in chunk"):
                continue

            if lower.startswith("what troubleshooting information is available in chunk"):
                continue

            normalized = lower.strip()

            if normalized in seen:
                continue

            seen.add(normalized)
            cleaned.append(question)

            if len(cleaned) >= max_items:
                break

        return cleaned

    def _clean_text(self, value: Any) -> str:
        if value is None:
            return ""

        if not isinstance(value, str):
            value = str(value)

        value = value.replace("\x00", "").replace("\u0000", "")
        value = re.sub(r"[ \t]+", " ", value)
        value = re.sub(r"\n{3,}", "\n\n", value)

        return value.strip()

    def _clean_string_list(
        self,
        value: Any,
        *,
        max_items: int,
    ) -> List[str]:
        if value is None:
            return []

        if isinstance(value, str):
            items = re.split(r"[\n,;|]+", value)
        elif isinstance(value, list):
            items = value
        else:
            return []

        cleaned: List[str] = []
        seen = set()

        bad_values = {
            "what", "does", "say", "about", "chunk", "summary", "combined",
            "information", "procedures", "described", "available", "document",
            "manual", "page", "section", "copyright", "user", "july",
        }

        for item in items:
            text = self._clean_text(item)

            if not text:
                continue

            text = re.sub(
                r"\bchunk[_\-\s]*\d+[_\-\s]*\d*\b",
                "",
                text,
                flags=re.I,
            ).strip(" -_:")

            if not text:
                continue

            normalized = text.lower()

            if normalized in bad_values:
                continue

            if normalized in seen:
                continue

            seen.add(normalized)
            cleaned.append(text)

            if len(cleaned) >= max_items:
                break

        return cleaned

    def _limit_text(self, text: str, max_chars: int) -> str:
        text = self._clean_text(text)

        if len(text) <= max_chars:
            return text

        return text[: max_chars - 3].rstrip() + "..."

    def _first_sentences(self, text: str, max_chars: int) -> str:
        text = self._clean_text(text)

        if len(text) <= max_chars:
            return text

        sentences = re.split(r"(?<=[.!?])\s+", text)

        selected: List[str] = []
        total = 0

        for sentence in sentences:
            sentence = self._clean_text(sentence)

            if not sentence:
                continue

            if total + len(sentence) + 1 > max_chars:
                break

            selected.append(sentence)
            total += len(sentence) + 1

        if selected:
            return " ".join(selected)

        return self._limit_text(text, max_chars)