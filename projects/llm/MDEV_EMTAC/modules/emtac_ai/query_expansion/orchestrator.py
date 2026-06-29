
# modules/emtac_ai/query_expansion/orchestrator.py
from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional
from modules.configuration.log_config import info_id, debug_id, warning_id, get_request_id, with_request_id
from modules.emtac_ai.emtac_intent_entity import IntentEntityPlugin
from .query_expansion_core import QueryExpansionRAG
from modules.configuration.config import ORC_INTENT_MODEL_DIR, ORC_PARTS_MODEL_DIR
from modules.configuration.log_config import get_request_id

class EMTACQueryExpansionOrchestrator:
    """
    Slim orchestrator that delegates expansion details to QueryExpansionRAG.
    """
    def __init__(
        self,
        ai_model: Optional[Any] = None,
        intent_model_dir: Optional[str] = None,
        ner_model_dir: Optional[str] = None,
        domain: str = "maintenance"
    ):
        # Normalize to config defaults if caller passes None or legacy strings
        legacy_intent = {"models/intent", "intent", "./models/intent", ".\\models\\intent"}
        legacy_ner    = {"models/ner", "ner", "./models/ner", ".\\models\\ner"}

        intent_model_dir = ORC_INTENT_MODEL_DIR if (not intent_model_dir or intent_model_dir in legacy_intent) else intent_model_dir
        ner_model_dir    = ORC_PARTS_MODEL_DIR if (not ner_model_dir or ner_model_dir in legacy_ner) else ner_model_dir  # swap if your NER base differs

        debug_id(f"[ORCH] using intent_model_dir={intent_model_dir}", get_request_id())
        debug_id(f"[ORCH] using ner_model_dir={ner_model_dir}", get_request_id())

        self.intent_ner = IntentEntityPlugin(intent_model_dir=intent_model_dir, ner_model_dir=ner_model_dir)
        self.engine = QueryExpansionRAG(ai_model=ai_model, domain=domain)

    @with_request_id
    def process_query_complete_pipeline(
        self,
        query: str,
        enable_ai: bool = True,
        confidence_scale: Optional[Callable[[float], int]] = None
    ) -> Dict[str, Any]:
        """
        Full pipeline: intent -> NER -> expansions (rules/entities/AI) -> final set.
        confidence_scale: optional function mapping [0..1] -> max variants.
        """
        req = get_request_id()
        intent, conf = self.intent_ner.classify_intent(query)
        ents = self.intent_ner.extract_entities(query)

        info_id(f"[ORCH] intent='{intent}' conf={conf:.3f} ents={len(ents)}", req)

        result = self.engine.comprehensive_expand(
            query=query,
            intent=intent,
            entities=ents,
            enable_ai=enable_ai
        )

        # Confidence-aware truncation (optional)
        expanded = result["final_expanded_queries"]
        if confidence_scale is not None and isinstance(conf, (int, float)):
            try:
                cap = max(3, int(confidence_scale(conf)))
                expanded = expanded[:cap]
                result["final_expanded_queries"] = expanded
                debug_id(f"[ORCH] confidence-capped to {cap} variants", req)
            except Exception as e:
                warning_id(f"[ORCH] confidence_scale failed: {e}", req)

        result["intent"] = intent
        result["confidence"] = conf
        result["entities"] = ents
        return result

    @with_request_id
    def run_search_with_expansions(
        self,
        user_query: str,
        search_fn: Callable[[str, int], List[Any]],
        top_k: int = 5,
        enable_ai: bool = True
    ) -> Dict[str, Any]:
        """
        Convenience helper to integrate expansions with your search backend.
        `search_fn` must accept (query, top_k) and return a list of docs/snippets.
        """
        pipeline = self.process_query_complete_pipeline(user_query, enable_ai=enable_ai)
        expanded = pipeline["final_expanded_queries"]
        all_docs = []
        for q in expanded:
            try:
                docs = search_fn(q, top_k=top_k) or []
                all_docs.extend(docs)
            except Exception as e:
                warning_id(f"[ORCH] search_fn failed on '{q}': {e}", get_request_id())
        pipeline["search_results"] = all_docs
        return pipeline
