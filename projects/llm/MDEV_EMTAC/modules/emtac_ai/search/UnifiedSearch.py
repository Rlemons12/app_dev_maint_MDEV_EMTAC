# UnifiedSearch.py (cleaned: regex backend removed)
from __future__ import annotations

import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, Any, Optional, List, Callable, Tuple

from modules.configuration.log_config import logger, with_request_id, info_id, warning_id, error_id
from modules.configuration.log_config import debug_id, get_request_id
from modules.emtac_ai.adpators.base_search_adapter import PartsSearchAdapter, DrawingsSearchAdapter
# ────────────────────────────────────────────────────────────────
# Orchestrator
# ────────────────────────────────────────────────────────────────
try:
    from modules.emtac_ai.query_expansion.orchestrator import EMTACQueryExpansionOrchestrator as Orchestrator
except Exception:
    Orchestrator = None

# Vector (AggregateSearch)
try:
    from modules.emtac_ai import AggregateSearch
except Exception:
    AggregateSearch = None

# FTS
try:
    from modules.emtacdb.emtacdb_fts import CompleteDocument
except Exception:
    CompleteDocument = None


# ────────────────────────────────────────────────────────────────
# Tracking primitives
# ────────────────────────────────────────────────────────────────
@dataclass
class SearchEvent:
    query: str
    user_id: Optional[str]
    method: str
    started_at: float
    request_id: Optional[str] = None
    intent: Optional[str] = None
    backend: Optional[str] = None
    entities: Dict[str, Any] = field(default_factory=dict)
    result_count: int = 0
    success: bool = False
    error: Optional[str] = None


class SearchTracker:
    def __init__(self, db_session=None):
        self.db_session = db_session

    def start(self, query: str, user_id: Optional[str], method: str, request_id: Optional[str]) -> SearchEvent:
        return SearchEvent(query=query, user_id=user_id, method=method, started_at=time.time(), request_id=request_id)

    def finish(
        self,
        ev: SearchEvent,
        result_count: int,
        success: bool,
        intent: Optional[str],
        backend: Optional[str],
        entities: Dict[str, Any],
        error: Optional[str],
    ) -> Dict[str, Any]:
        ev.result_count = int(result_count or 0)
        ev.success = bool(success)
        ev.intent = intent
        ev.backend = backend
        ev.entities = entities or {}
        ev.error = error
        return {
            "query": ev.query,
            "user_id": ev.user_id or "anonymous",
            "request_id": ev.request_id,
            "method": ev.method,
            "intent": ev.intent,
            "backend": ev.backend,
            "entities": ev.entities,
            "result_count": ev.result_count,
            "success": ev.success,
            "error": ev.error,
            "duration_ms": int((time.time() - ev.started_at) * 1000),
            "timestamp": datetime.utcnow().isoformat(),
        }


def _fmt_kvs(**kvs):
    return " ".join(f"{k}={v}" for k, v in kvs.items() if v not in (None, {}, [], ""))


# ────────────────────────────────────────────────────────────────
# UnifiedSearch Hub
# ────────────────────────────────────────────────────────────────
class UnifiedSearch:
    """
    Hub for:
      - Tracking searches
      - Routing to orchestrator / vector / FTS
      - Organizing results for the UI
    """

    def __init__(
        self,
        db_session=None,
        enable_vector: bool = True,
        enable_fts: bool = True,
        enable_orchestrator: bool = True,
        intent_model_dir: Optional[str] = None,
        ner_model_dirs: Optional[Dict[str, str]] = None,
        ai_model=None,
        domain: str = "maintenance",
    ):
        self.db_session = getattr(self, "db_session", None) or db_session
        self.tracker = SearchTracker(self.db_session)

        self.backends: Dict[str, Callable[[str], Dict[str, Any]]] = {}
        self.orchestrator = None
        self.vector_engine = None
        self.fts_enabled = False

        # Init backends
        self._init_orchestrator(enable_orchestrator, intent_model_dir, ner_model_dirs, ai_model, domain)
        self._init_vector(enable_vector)
        self._init_fts(enable_fts)

        logger.info("UnifiedSearch hub initialized.")

    # ---------- Initialization helpers ----------
    def _init_orchestrator(self, enable: bool, intent_dir, ner_dirs, ai_model=None, domain="maintenance"):
        if not enable or Orchestrator is None:
            return
        try:
            self.orchestrator = Orchestrator(
                ai_model=ai_model,
                intent_model_dir=intent_dir,
                ner_model_dir=ner_dirs.get("default") if isinstance(ner_dirs, dict) else None,
                domain=domain,
            )
            self.register_backend("orchestrator",
                                  lambda q, request_id=None: self._call_orchestrator(q, request_id=request_id))
            logger.info("Orchestrator backend registered.")
        except Exception as e:
            logger.error(f"Failed to init orchestrator: {e}", exc_info=True)

    def _init_vector(self, enable: bool):
        if not enable or not AggregateSearch:
            return
        try:
            try:
                self.vector_engine = AggregateSearch()
            except TypeError:
                self.vector_engine = AggregateSearch(self.db_session)
            self.register_backend("vector", self._call_vector_search)
            logger.info("Vector backend registered.")
        except Exception as e:
            logger.warning(f"Vector backend unavailable: {e}", exc_info=True)

    def _init_fts(self, enable: bool):
        if not enable or not CompleteDocument:
            return
        self.fts_enabled = True
        self.register_backend("fts", self._call_fts_search)
        logger.info("FTS backend registered.")

    # ---------- Public Entry ----------
    @with_request_id
    def execute_unified_search(self, question: str, user_id: Optional[str] = None, request_id: Optional[str] = None) -> Dict[str, Any]:
        q = (question or "").strip()
        if len(q) < 2:
            return self._bad_request_response("Please provide a more detailed question.")

        order = [b for b in ["orchestrator", "vector", "fts"] if b in self.backends]
        for method_name in order:
            ev = self.tracker.start(query=q, user_id=user_id, method=method_name, request_id=request_id)
            try:
                raw = self.backends[method_name](q)
                results = self._extract_results(raw)
                intent, entities = self._extract_intent_entities(raw)
                success = len(results) > 0
                self.tracker.finish(ev, len(results), success, intent, method_name, entities, None)
                if success:
                    return self._enhance_unified_response(q, method_name, intent, entities, self._organize_results_by_type(results), raw)
            except Exception as e:
                self.tracker.finish(ev, 0, False, None, method_name, {}, str(e))
        return self._no_unified_results_response(q)

    # ---------- Backend registration ----------
    def register_backend(self, name: str, fn: Callable[[str], Dict[str, Any]]):
        self.backends[name] = fn

    # ---------- Backend callers ----------
    @with_request_id
    def _call_orchestrator(self, question: str, request_id: Optional[str] = None) -> Dict[str, Any]:
        if not self.orchestrator:
            return {"method": "orchestrator", "results": [], "intent": "unknown", "total_results": 0}

        rid = request_id or get_request_id()
        debug_id(f"[UnifiedSearch] _call_orchestrator called with question='{question}'", rid)

        try:
            info_id(f"[UnifiedSearch] Running orchestrator.process_query_complete_pipeline...", rid)
            payload = self.orchestrator.process_query_complete_pipeline(
                query=question, enable_ai=True
            ) or {}

            intent = payload.get("intent")
            entities = payload.get("entities") or []
            results = []

            # 🔄 normalize entities into a dict for adapters
            ent_map = {}
            for ent in entities:
                lbl = ent.get("label")
                if lbl:
                    ent_map.setdefault(lbl, []).append(ent.get("entity"))

            if intent == "parts":
                adapter = PartsSearchAdapter(session=self.db_session, request_id=rid)
                results = adapter.search(query=question, entities=ent_map)  # ✅ dict not list

            elif intent == "drawings":
                adapter = DrawingsSearchAdapter(session=self.db_session, request_id=rid)
                results = adapter.search(query=question, entities=ent_map)  # ✅ use dict here too

            # extend with documents/images if needed

            payload.setdefault("method", "orchestrator")
            payload.setdefault("intent", intent or "unknown")
            payload["results"] = results
            payload["total_results"] = len(results)

            debug_id(f"[UnifiedSearch] Final orchestrator payload keys={list(payload.keys())}", rid)
            return payload

        except Exception as e:
            error_id(f"[UnifiedSearch] Error in _call_orchestrator: {e}", rid, exc_info=True)
            return {
                "method": "orchestrator",
                "intent": "error",
                "results": [],
                "total_results": 0,
                "error": str(e),
            }

    def _call_vector_search(self, question: str) -> Dict[str, Any]:
        if not self.vector_engine:
            return {"method": "vector", "results": []}
        for cand in ["execute_aggregated_search", "search", "execute_search", "__call__"]:
            fn = getattr(self.vector_engine, cand, None)
            if fn:
                try:
                    out = fn(question)
                    break
                except Exception:
                    continue
        else:
            return {"method": "vector", "results": []}
        if isinstance(out, dict):
            out.setdefault("method", "vector")
            out.setdefault("results", out.get("results") or [])
            return out
        return {"method": "vector", "results": out or []}

    def _call_fts_search(self, question: str) -> Dict[str, Any]:
        docs = CompleteDocument.search_by_text(question, limit=25, session=self.db_session)
        results = [{"id": getattr(d, "id", None), "title": getattr(d, "title", ""), "snippet": (getattr(d, "content", "") or "")[:400]} for d in (docs or [])]
        return {"status": "success", "search_method": "fts", "total_results": len(results), "results": results}

    # ---------- Helpers ----------
    def _extract_results(self, raw: Dict[str, Any]) -> List[Dict[str, Any]]:
        return raw.get("results", []) if isinstance(raw, dict) else []

    def _extract_intent_entities(self, raw: Dict[str, Any]) -> Tuple[Optional[str], Dict[str, Any]]:
        if not isinstance(raw, dict):
            return None, {}
        return raw.get("intent") or raw.get("detected_intent"), raw.get("entities") or raw.get("nlp_analysis", {}).get("entities", {})

    def _organize_results_by_type(self, results: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
        buckets = {"parts": [], "drawings": [], "images": [], "positions": [], "other": []}
        for r in results or []:
            t = (r.get("type") or r.get("entity_type") or "other").lower()
            if t.startswith("part"):
                buckets["parts"].append(r)
            elif "drawing" in t:
                buckets["drawings"].append(r)
            elif "image" in t:
                buckets["images"].append(r)
            elif "position" in t or "location" in t:
                buckets["positions"].append(r)
            else:
                buckets["other"].append(r)
        return buckets

    def _enhance_unified_response(self, question, method, intent, entities, organized, raw) -> Dict[str, Any]:
        return {
            "search_type": "unified",
            "status": "success",
            "query": question,
            "timestamp": datetime.utcnow().isoformat(),
            "detected_intent": intent or "UNKNOWN",
            "entities": entities or {},
            "results_by_type": organized,
            "total_results": sum(len(v) for v in organized.values()),
            "search_method": method,
        }

    def _no_unified_results_response(self, question: str) -> Dict[str, Any]:
        return {"search_type": "unified", "status": "no_results", "query": question, "results_by_type": {"parts": [], "drawings": [], "images": [], "positions": [], "other": []}, "timestamp": datetime.utcnow().isoformat(), "search_method": "none"}

    def _bad_request_response(self, msg: str) -> Dict[str, Any]:
        return {"search_type": "unified", "status": "error", "message": msg, "results_by_type": {}, "timestamp": datetime.utcnow().isoformat()}
