# modules/services/intent/chat_intent_classifier_service.py

from __future__ import annotations

import importlib
import os
import re
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from modules.configuration.log_config import debug_id, error_id, warning_id
from modules.intent.intent_types import ChatIntent, ChatIntentDecision


class ChatIntentClassifierService:
    """
    Local DistilBERT intent classifier.

    IMPORTANT:
    This service does NOT call:
      - AIStewardManagerService
      - AIModelsService
      - RAG/search
      - local LLM generation

    It loads the trained model from:

        MODELS_DISTILBERT_INTENT

    Example .env value:

        MODELS_DISTILBERT_INTENT=E:\\emtac\\models\\modules\\transformers_modules\\chat_intent_distilbert_augmented
    """

    DEFAULT_MODEL_ENV = "MODELS_DISTILBERT_INTENT"
    DEFAULT_MAX_LENGTH = 160

    SUPPORTED_INTENTS = {
        "NEW_TOPIC": ChatIntent.NEW_TOPIC,
        "FOLLOW_UP_CURRENT_SESSION": ChatIntent.FOLLOW_UP_CURRENT_SESSION,
        "RECALL_PRIOR_CONVERSATION": ChatIntent.RECALL_PRIOR_CONVERSATION,
        "DOCUMENT_SCOPED_FOLLOW_UP": ChatIntent.DOCUMENT_SCOPED_FOLLOW_UP,
        "CLARIFICATION": ChatIntent.CLARIFICATION,
    }

    _load_lock = threading.Lock()
    _tokenizer: Optional[Any] = None
    _model: Optional[Any] = None
    _model_path: Optional[str] = None
    _id2label: Dict[int, str] = {}

    def __init__(
        self,
        *,
        model_path: Optional[str] = None,
        device: Optional[str] = None,
        max_length: int = DEFAULT_MAX_LENGTH,
        ai_service: Optional[Any] = None,
    ):
        """
        ai_service is accepted only for backward compatibility.

        The production classifier uses the local DistilBERT model.
        """

        self.model_path = model_path or self._resolve_model_path()
        self.device = device or self._resolve_device()
        self.max_length = int(max_length or self.DEFAULT_MAX_LENGTH)

        if ai_service is not None:
            warning_id(
                "[ChatIntentClassifierService] ai_service was provided but is ignored. "
                "Intent classification now uses local DistilBERT.",
                None,
            )

    def classify(
        self,
        *,
        question: str,
        prompt: Optional[str] = None,
        recent_messages: Optional[List[Dict[str, Any]]] = None,
        conversation_summary: Optional[List[Dict[str, Any]]] = None,
        document_scope: Optional[Dict[str, Any]] = None,
        request_id: Optional[str] = None,
    ) -> ChatIntentDecision:
        """
        Classify the user's question.

        prompt is accepted only for backward compatibility and is ignored.
        """

        normalized_question = (question or "").strip()

        if not normalized_question:
            return ChatIntentDecision.fallback_new_topic("")

        try:
            rule_decision = self._rule_based_decision(
                question=normalized_question,
                recent_messages=recent_messages,
                conversation_summary=conversation_summary,
                document_scope=document_scope,
            )

            if rule_decision is not None:
                debug_id(
                    f"[ChatIntentClassifierService] rule_applied={rule_decision.reason!r} "
                    f"intent={rule_decision.intent.value} "
                    f"confidence={rule_decision.confidence:.4f}",
                    request_id,
                )
                return rule_decision

            tokenizer, model, id2label = self._get_model_bundle(
                model_path=self.model_path,
                device=self.device,
                request_id=request_id,
            )

            inputs = tokenizer(
                normalized_question,
                truncation=True,
                padding=True,
                max_length=self.max_length,
                return_tensors="pt",
            )

            inputs = {
                key: value.to(self.device)
                for key, value in inputs.items()
            }

            with torch.no_grad():
                outputs = model(**inputs)
                probabilities = torch.softmax(outputs.logits, dim=-1)[0]

            best_index = int(torch.argmax(probabilities).item())
            confidence = float(probabilities[best_index].item())

            label = str(id2label.get(best_index, "")).strip().upper()
            intent = self._coerce_intent(label)

            decision = ChatIntentDecision(
                intent=intent,
                confidence=confidence,
                needs_current_session_memory=False,
                needs_semantic_chat_recall=False,
                needs_document_scope=False,
                rewritten_question=normalized_question,
                reason=f"DistilBERT predicted {intent.value}.",
            )

            decision = self._normalize_flags(decision)

            debug_id(
                f"[ChatIntentClassifierService] model intent={decision.intent.value} "
                f"confidence={decision.confidence:.4f} "
                f"device={self.device} "
                f"model_path={self.model_path!r}",
                request_id,
            )

            return decision

        except Exception as exc:
            error_id(
                f"[ChatIntentClassifierService] DistilBERT intent classification failed: "
                f"{type(exc).__name__}: {exc}",
                request_id,
                exc_info=True,
            )

            return ChatIntentDecision.fallback_new_topic(normalized_question)

    @classmethod
    def _get_model_bundle(
        cls,
        *,
        model_path: str,
        device: str,
        request_id: Optional[str],
    ) -> Tuple[Any, Any, Dict[int, str]]:
        safe_model_path = str(Path(model_path).expanduser())

        with cls._load_lock:
            if (
                cls._tokenizer is not None
                and cls._model is not None
                and cls._model_path == safe_model_path
            ):
                return cls._tokenizer, cls._model, cls._id2label

            if not safe_model_path:
                raise RuntimeError("MODELS_DISTILBERT_INTENT is not configured.")

            model_dir = Path(safe_model_path)

            if not model_dir.exists():
                raise FileNotFoundError(
                    f"DistilBERT intent model path does not exist: {safe_model_path}"
                )

            debug_id(
                f"[ChatIntentClassifierService] Loading DistilBERT intent model "
                f"from {safe_model_path!r}",
                request_id,
            )

            tokenizer = AutoTokenizer.from_pretrained(
                safe_model_path,
                local_files_only=True,
            )

            model = AutoModelForSequenceClassification.from_pretrained(
                safe_model_path,
                local_files_only=True,
            )

            model.to(device)
            model.eval()

            id2label = cls._extract_id2label(model)

            cls._tokenizer = tokenizer
            cls._model = model
            cls._model_path = safe_model_path
            cls._id2label = id2label

            debug_id(
                f"[ChatIntentClassifierService] Loaded DistilBERT intent labels: {id2label}",
                request_id,
            )

            return tokenizer, model, id2label

    @classmethod
    def _extract_id2label(cls, model: Any) -> Dict[int, str]:
        raw_id2label = getattr(model.config, "id2label", None) or {}

        id2label: Dict[int, str] = {}

        for index, label in raw_id2label.items():
            try:
                clean_index = int(index)
                clean_label = str(label).strip().upper()
                id2label[clean_index] = clean_label
            except Exception:
                continue

        if id2label:
            return id2label

        raw_label2id = getattr(model.config, "label2id", None) or {}

        for label, index in raw_label2id.items():
            try:
                clean_index = int(index)
                clean_label = str(label).strip().upper()
                id2label[clean_index] = clean_label
            except Exception:
                continue

        if id2label:
            return id2label

        raise RuntimeError(
            "DistilBERT intent model is missing config.id2label/config.label2id."
        )

    @classmethod
    def _resolve_model_path(cls) -> str:
        env_value = os.getenv(cls.DEFAULT_MODEL_ENV)

        if env_value:
            return env_value.strip().strip('"').strip("'")

        config_value = cls._read_config_value(cls.DEFAULT_MODEL_ENV)

        if config_value:
            return str(config_value).strip().strip('"').strip("'")

        return (
            r"E:\emtac\models\modules\transformers_modules"
            r"\chat_intent_distilbert_augmented"
        )

    @staticmethod
    def _read_config_value(name: str) -> Optional[Any]:
        candidate_modules = [
            "modules.configuration.config_env",
            "modules.configuration.model_config",
            "modules.configuration.config",
        ]

        for module_name in candidate_modules:
            try:
                module = importlib.import_module(module_name)
                value = getattr(module, name, None)
                if value:
                    return value
            except Exception:
                continue

        return None

    @staticmethod
    def _resolve_device() -> str:
        """
        Intent classification is intentionally CPU by default.

        DistilBERT intent routing is small and fast on CPU, and this avoids CUDA
        contention/CUBLAS initialization failures when the main LLM/GPU service is
        already using VRAM.
        """
        return os.getenv("CHAT_INTENT_DEVICE", "cpu").strip().lower() or "cpu"

    @classmethod
    def _coerce_intent(cls, value: Any) -> ChatIntent:
        text = str(value or "").strip().upper()

        aliases = {
            "NEW": "NEW_TOPIC",
            "NEW_QUESTION": "NEW_TOPIC",
            "STANDALONE": "NEW_TOPIC",
            "STANDALONE_QUESTION": "NEW_TOPIC",
            "RAG": "NEW_TOPIC",
            "RAG_ONLY": "NEW_TOPIC",

            "FOLLOW_UP": "FOLLOW_UP_CURRENT_SESSION",
            "FOLLOWUP": "FOLLOW_UP_CURRENT_SESSION",
            "CURRENT_SESSION": "FOLLOW_UP_CURRENT_SESSION",
            "CURRENT_SESSION_MEMORY": "FOLLOW_UP_CURRENT_SESSION",

            "RECALL": "RECALL_PRIOR_CONVERSATION",
            "MEMORY_RECALL": "RECALL_PRIOR_CONVERSATION",
            "SEMANTIC_CHAT_RECALL": "RECALL_PRIOR_CONVERSATION",
            "PRIOR_CONVERSATION": "RECALL_PRIOR_CONVERSATION",

            "DOCUMENT_SCOPE": "DOCUMENT_SCOPED_FOLLOW_UP",
            "DOCUMENT_SCOPED": "DOCUMENT_SCOPED_FOLLOW_UP",
            "DOC_FOLLOW_UP": "DOCUMENT_SCOPED_FOLLOW_UP",

            "CLARIFY": "CLARIFICATION",
            "EXPLAIN_THAT": "CLARIFICATION",
        }

        text = aliases.get(text, text)

        if text in cls.SUPPORTED_INTENTS:
            return cls.SUPPORTED_INTENTS[text]

        return ChatIntent.NEW_TOPIC

    @staticmethod
    def _normalize_flags(decision: ChatIntentDecision) -> ChatIntentDecision:
        if decision.intent == ChatIntent.NEW_TOPIC:
            decision.needs_current_session_memory = False
            decision.needs_semantic_chat_recall = False
            decision.needs_document_scope = False

        elif decision.intent == ChatIntent.FOLLOW_UP_CURRENT_SESSION:
            decision.needs_current_session_memory = True
            decision.needs_semantic_chat_recall = False
            decision.needs_document_scope = False

        elif decision.intent == ChatIntent.RECALL_PRIOR_CONVERSATION:
            decision.needs_current_session_memory = True
            decision.needs_semantic_chat_recall = True
            decision.needs_document_scope = False

        elif decision.intent == ChatIntent.DOCUMENT_SCOPED_FOLLOW_UP:
            decision.needs_current_session_memory = True
            decision.needs_semantic_chat_recall = False
            decision.needs_document_scope = True

        elif decision.intent == ChatIntent.CLARIFICATION:
            decision.needs_current_session_memory = True
            decision.needs_semantic_chat_recall = False
            decision.needs_document_scope = False

        if not decision.rewritten_question:
            decision.rewritten_question = ""

        return decision

    def _rule_based_decision(
        self,
        *,
        question: str,
        recent_messages: Optional[List[Dict[str, Any]]],
        conversation_summary: Optional[List[Dict[str, Any]]],
        document_scope: Optional[Dict[str, Any]],
    ) -> Optional[ChatIntentDecision]:
        q = self._normalize_text(question)

        personal_memory_patterns = [
            r"\bwhat\s+is\s+my\s+name\b",
            r"\bdo\s+you\s+remember\s+my\s+name\b",
            r"\bremember\s+my\s+name\b",
            r"\bwho\s+am\s+i\b",
            r"\bwhat\s+do\s+you\s+remember\s+about\s+me\b",
            r"\bwhat\s+have\s+i\s+told\s+you\s+about\s+me\b",
        ]

        if self._matches_any(q, personal_memory_patterns):
            return self._make_decision(
                intent=ChatIntent.RECALL_PRIOR_CONVERSATION,
                confidence=1.0,
                rewritten_question=question,
                reason="personal_memory",
            )

        prior_conversation_patterns = [
            r"\bwhat\s+were\s+we\s+talking\s+about\b",
            r"\bwhat\s+did\s+we\s+talk\s+about\b",
            r"\bwhat\s+did\s+i\s+ask\b",
            r"\bwhat\s+was\s+my\s+last\s+question\b",
            r"\bwhat\s+did\s+you\s+say\s+earlier\b",
            r"\bwhat\s+did\s+you\s+say\s+before\b",
            r"\bpreviously\b",
            r"\bearlier\b",
            r"\blast\s+time\b",
            r"\bdo\s+you\s+remember\b",
            r"\bremember\s+when\b",
        ]

        if self._matches_any(q, prior_conversation_patterns):
            return self._make_decision(
                intent=ChatIntent.RECALL_PRIOR_CONVERSATION,
                confidence=0.99,
                rewritten_question=question,
                reason="prior_conversation_recall",
            )

        clarification_patterns = [
            r"\bwhat\s+do\s+you\s+mean\b",
            r"\bexplain\s+that\b",
            r"\bexplain\s+it\b",
            r"\bsimplify\s+that\b",
            r"\bbreak\s+that\s+down\b",
            r"\bcan\s+you\s+expand\s+on\s+that\b",
            r"\bsay\s+that\s+another\s+way\b",
            r"\breword\s+that\b",
            r"\bcontinue\b",
            r"\bgo\s+on\b",
            r"\btell\s+me\s+more\b",
        ]

        if self._matches_any(q, clarification_patterns):
            return self._make_decision(
                intent=ChatIntent.CLARIFICATION,
                confidence=0.98,
                rewritten_question=question,
                reason="clarification_phrase",
            )

        if self._document_scope_is_active(document_scope):
            document_followup_patterns = [
                r"\bthis\s+document\b",
                r"\bthat\s+document\b",
                r"\bthe\s+document\b",
                r"\bcurrent\s+document\b",
                r"\bthis\s+manual\b",
                r"\bthat\s+manual\b",
                r"\bthe\s+manual\b",
                r"\bthis\s+procedure\b",
                r"\bthat\s+procedure\b",
                r"\bin\s+this\s+document\b",
                r"\bin\s+that\s+document\b",
                r"\bbased\s+on\s+that\b",
                r"\bbased\s+on\s+this\b",
                r"\bfrom\s+that\s+document\b",
                r"\bfrom\s+this\s+document\b",
            ]

            if self._matches_any(q, document_followup_patterns):
                return self._make_decision(
                    intent=ChatIntent.DOCUMENT_SCOPED_FOLLOW_UP,
                    confidence=0.99,
                    rewritten_question=question,
                    reason="active_document_scope",
                )

        return None

    def _make_decision(
        self,
        *,
        intent: ChatIntent,
        confidence: float,
        rewritten_question: str,
        reason: str,
    ) -> ChatIntentDecision:
        decision = ChatIntentDecision(
            intent=intent,
            confidence=confidence,
            needs_current_session_memory=False,
            needs_semantic_chat_recall=False,
            needs_document_scope=False,
            rewritten_question=rewritten_question,
            reason=reason,
        )

        return self._normalize_flags(decision)

    @staticmethod
    def _normalize_text(value: str) -> str:
        text = (value or "").strip().lower()
        text = re.sub(r"[^\w\s]", " ", text)
        text = re.sub(r"\s+", " ", text).strip()
        return text

    @staticmethod
    def _matches_any(text: str, patterns: List[str]) -> bool:
        return any(
            re.search(pattern, text, flags=re.IGNORECASE)
            for pattern in patterns
        )

    @staticmethod
    def _document_scope_is_active(document_scope: Optional[Dict[str, Any]]) -> bool:
        if not isinstance(document_scope, dict):
            return False

        if document_scope.get("enabled") is False:
            return False

        return bool(
            document_scope.get("complete_document_id")
            or document_scope.get("completeDocumentId")
            or document_scope.get("completeDocumentID")
            or document_scope.get("document_id")
            or document_scope.get("documentId")
            or document_scope.get("document_name")
            or document_scope.get("documentName")
            or document_scope.get("name")
            or document_scope.get("title")
        )