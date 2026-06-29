# unified_search_service.py  (updated to use the new query_expansion orchestrator)
from __future__ import annotations

import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, Any, Optional, List, Callable, Tuple

from modules.configuration.log_config import logger, with_request_id
from modules.configuration.log_config import debug_id, get_request_id  # optional, used for extra logs

# ──────────────────────────────────────────────────────────────────────────────
# Orchestrator (NEW LOCATION)
# ──────────────────────────────────────────────────────────────────────────────
try:
    # New orchestrator lives here
    from modules.emtac_ai.query_expansion.orchestrator import EMTACQueryExpansionOrchestrator as Orchestrator
except Exception:
    Orchestrator = None

# Optional vector layer (AggregateSearch)
try:
    from modules.emtac_ai import AggregateSearch
except Exception:
    AggregateSearch = None

# Optional FTS model
try:
    from modules.emtacdb.emtacdb_fts import CompleteDocument
except Exception:
    CompleteDocument = None


# ──────────────────────────────────────────────────────────────────────────────
# Tracking primitives
# ──────────────────────────────────────────────────────────────────────────────
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
    """
    Pluggable tracker. In production, persist to search_analytics (DB).
    """
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
        payload = {
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
        return payload


def _fmt_kvs(**kvs):
    return " ".join(f"{k}={v}" for k, v in kvs.items() if v not in (None, {}, [], ""))


# ──────────────────────────────────────────────────────────────────────────────
# UnifiedSearch Hub
# ──────────────────────────────────────────────────────────────────────────────
class UnifiedSearch:
    """
    Hub for:
      - Tracking searches
      - Routing to orchestrator / vector / FTS / regex
      - Organizing results for the UI
    """

    def __init__(
        self,
        db_session=None,
        enable_vector: bool = True,
        enable_fts: bool = True,
        enable_regex: bool = False,
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
        self.regex_enabled = False

        # Init backends
        self._init_orchestrator(enable_orchestrator, intent_model_dir, ner_model_dirs, ai_model, domain)
        self._init_vector(enable_vector)
        self._init_fts(enable_fts)
        self._init_regex(enable_regex)

        logger.info("UnifiedSearch hub initialized.")

    # ---------- Initialization helpers ----------
    def _init_orchestrator(
        self,
        enable_orchestrator: bool,
        intent_dir: Optional[str],
        ner_dirs: Optional[Dict[str, str]],
        ai_model=None,
        domain: str = "maintenance",
    ):
        if not enable_orchestrator:
            self._log("warning", "Orchestrator disabled by flag")
            return

        if Orchestrator is None:
            self._log("error", "New orchestrator not importable at modules.emtac_ai.query_expansion.orchestrator")
            return

        try:
            # Use the new orchestrator directly
            self.orchestrator = Orchestrator(
                ai_model=ai_model,
                intent_model_dir=intent_dir,
                ner_model_dir=ner_dirs.get("default") if isinstance(ner_dirs, dict) else None,  # optional
                domain=domain,
            )
            self.register_backend("orchestrator",
                                  lambda q, request_id=None: self._call_orchestrator(q, request_id=request_id))
            self._log("info", "Orchestrator backend registered")
        except Exception as e:
            import traceback
            self._log("error", "Failed to init new orchestrator", error=str(e))
            logger.error("Orchestrator init traceback:\n" + traceback.format_exc())

    def _init_vector(self, enable_vector: bool):
        if not enable_vector or not AggregateSearch:
            return
        try:
            try:
                self.vector_engine = AggregateSearch()
            except TypeError:
                # Older signatures may want a session
                self.vector_engine = AggregateSearch(self.db_session)
            self.register_backend("vector", self._call_vector_search)
            logger.info("Vector backend registered (AggregateSearch).")
        except Exception as e:
            logger.warning(f"Vector backend unavailable: {e}", exc_info=True)

    def _init_fts(self, enable_fts: bool):
        if not enable_fts or not CompleteDocument:
            return
        self.fts_enabled = True
        self.register_backend("fts", self._call_fts_search)
        logger.info("FTS backend registered.")

    def _init_regex(self, enable_regex: bool):
        self.regex_enabled = bool(enable_regex)
        if self.regex_enabled:
            self.register_backend("regex", self._call_regex_search)
            logger.info("Regex backend registered.")

    # ---------- Logging ----------
    def _log(self, level: str, msg: str, request_id: Optional[str] = None, **kvs):
        tag = f"[REQ-{request_id}]" if request_id else ""
        tail = _fmt_kvs(**kvs)
        line = f"{tag} {msg}" + (f" | {tail}" if tail else "")
        getattr(logger, level if level in ("debug", "info", "warning", "error") else "info")(line)

    # ---------- Public Entry ----------
    @with_request_id
    def execute_unified_search(
        self,
        question: str,
        user_id: Optional[str] = None,
        request_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        q = (question or "").strip()
        if len(q) < 2:
            self._log("warning", "Question too short", request_id, user_id=user_id, length=len(q))
            return self._bad_request_response("Please provide a more detailed question.")

        self._log("info", "execute_unified_search: received question", request_id, user_id=user_id, length=len(q))

        order = [b for b in ["orchestrator", "vector", "fts", "regex"] if b in self.backends]
        self._log("debug", "Backend order", request_id, order=",".join(order))
        if "orchestrator" not in self.backends:
            self._log("warning", "Orchestrator not registered; falling back to other backends")

        for method_name in order:
            t0 = time.perf_counter()
            ev = self.tracker.start(query=q, user_id=user_id, method=method_name, request_id=request_id)
            self._log("debug", "Dispatching to backend", request_id, backend=method_name)

            try:
                raw = self.backends[method_name](q)  # call backend
                dt_ms = int((time.perf_counter() - t0) * 1000)

                results = self._extract_results(raw)
                intent, entities = self._extract_intent_entities(raw)
                success = len(results) > 0

                analytics = self.tracker.finish(
                    ev,
                    result_count=len(results),
                    success=success,
                    intent=intent,
                    backend=method_name,
                    entities=entities,
                    error=None,
                )

                keys = ",".join(sorted(raw.keys())) if isinstance(raw, dict) else type(raw).__name__
                self._log("debug", "Backend returned", request_id, backend=method_name, duration_ms=dt_ms,
                          results=len(results), success=success, intent=intent or "", keys=keys)
                logger.debug(f"Search analytics: {analytics}")

                if success:
                    organized = self._organize_results_by_type(results)
                    self._log("info", "UnifiedSearch success", request_id, backend=method_name,
                              organized_types=",".join(organized.keys()),
                              total_results=sum(len(v) for v in organized.values()))
                    return self._enhance_unified_response(
                        question=q,
                        method=method_name,
                        intent=intent,
                        entities=entities,
                        organized=organized,
                        raw=raw,
                    )

            except Exception as e:
                dt_ms = int((time.perf_counter() - t0) * 1000)
                self._log("error", f"{method_name} backend failed", request_id, duration_ms=dt_ms, error=str(e))
                self.tracker.finish(ev, result_count=0, success=False, intent=None, backend=method_name, entities={}, error=str(e))
                # Try next backend

        self._log("warning", "No backend produced results", request_id, question_preview=q[:80])
        return self._no_unified_results_response(q)

    # ---------- Backend registration ----------
    def register_backend(self, name: str, fn: Callable[[str], Dict[str, Any]]):
        self.backends[name] = fn

    # ---------- Backend callers ----------
    def _call_orchestrator(self, question: str, request_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Bridges the new orchestrator API to UnifiedSearch expectations.
        - Calls process_query_complete_pipeline(...)
        - Ensures a 'results' list is present for downstream
        """
        if not self.orchestrator:
            self._log("warning", "Orchestrator not initialized", request_id)
            return {"method": "orchestrator", "results": []}

        self._log("debug", "Orchestrator.process_query_complete_pipeline: start", request_id, q_len=len(question))
        t0 = time.perf_counter()
        try:
            payload = self.orchestrator.process_query_complete_pipeline(query=question, enable_ai=True) or {}
            dt_ms = int((time.perf_counter() - t0) * 1000)

            # Normalize keys for UnifiedSearch
            # If your orchestrator already returns 'results', this is a no-op.
            if "results" not in payload:
                payload["results"] = payload.get("results", [])

            intent = payload.get("intent")
            keys = ",".join(sorted(payload.keys()))
            self._log("info", "Orchestrator.process_query_complete_pipeline: done", request_id,
                      duration_ms=dt_ms, intent=intent or "", results=len(payload["results"]), keys=keys)
            payload.setdefault("method", "orchestrator")
            return payload
        except Exception as e:
            dt_ms = int((time.perf_counter() - t0) * 1000)
            self._log("error", "Orchestrator call failed", request_id, duration_ms=dt_ms, error=str(e))
            return {"method": "orchestrator", "results": [], "error": str(e)}

    def _call_vector_search(self, question: str) -> Dict[str, Any]:
        if not self.vector_engine:
            return {"method": "vector", "results": []}

        # Be resilient to different AggregateSearch APIs
        candidates = [
            "execute_aggregated_search",
            "search",
            "execute_search",
            "__call__",
        ]
        out = None
        last_err = None
        for name in candidates:
            fn = getattr(self.vector_engine, name, None)
            if fn:
                try:
                    out = fn(question)
                    break
                except Exception as e:
                    last_err = e

        if out is None:
            raise last_err or AttributeError(
                "AggregateSearch has no compatible search method (tried: execute_aggregated_search, search, execute_search, __call__)"
            )

        if isinstance(out, dict):
            out.setdefault("method", "vector")
            out.setdefault("results", out.get("results") or [])
            return out

        # If the engine returns a plain list of rows
        return {"method": "vector", "results": out or []}

    def _call_fts_search(self, question: str) -> Dict[str, Any]:
        logger.debug("Starting search_by_text")
        try:
            docs = CompleteDocument.search_by_text(question, limit=25, session=self.db_session)
        except TypeError:
            try:
                docs = CompleteDocument.search_by_text(question, limit=25)
            except TypeError:
                docs = CompleteDocument.search_by_text(question)

        results = []
        for d in (docs or []):
            results.append({
                "id": getattr(d, "id", None),
                "title": getattr(d, "title", getattr(d, "name", "")),
                "snippet": (getattr(d, "content", "") or "")[:400],
                "source": "fts_completed_document",
            })

        return {
            "status": "success",
            "search_method": "fts",
            "total_results": len(results),
            "results": results,
        }

    def _call_regex_search(self, question: str) -> Dict[str, Any]:
        return {"method": "regex", "results": []}

    # ---------- Result shaping ----------
    def _extract_results(self, raw: Dict[str, Any]) -> List[Dict[str, Any]]:
        if not isinstance(raw, dict):
            return []
        res = raw.get("results")
        return res if isinstance(res, list) else []

    def _extract_intent_entities(self, raw: Dict[str, Any]) -> Tuple[Optional[str], Dict[str, Any]]:
        intent = None
        entities: Dict[str, Any] = {}
        if isinstance(raw, dict):
            intent = raw.get("intent") or raw.get("detected_intent")
            entities = raw.get("entities") or raw.get("nlp_analysis", {}).get("entities", {})
        return intent, entities

    def _organize_results_by_type(self, results: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
        parts, drawings, images, positions, other = [], [], [], [], []
        for r in results or []:
            t = (r.get("type") or r.get("entity_type") or "other").lower()
            if t.startswith("part"):
                parts.append(r)
            elif "drawing" in t or t == "print":
                drawings.append(r)
            elif "image" in t or "photo" in t:
                images.append(r)
            elif "position" in t or "location" in t:
                positions.append(r)
            else:
                other.append(r)
        return {"parts": parts, "drawings": drawings, "images": images, "positions": positions, "other": other}

    def _create_unified_search_summary(self, question: str, buckets: Dict[str, List[Dict[str, Any]]], intent: Optional[str]) -> str:
        total = sum(len(v) for v in buckets.values())
        intent_txt = f" (intent: {intent})" if intent else ""
        return f"Found {total} results for '{question}'{intent_txt}."

    def _generate_quick_actions(self, buckets: Dict[str, List[Dict[str, Any]]], question: str) -> List[Dict[str, Any]]:
        actions = []
        if buckets.get("parts"):
            actions.append({"label": "View Parts", "action": "OPEN_TAB", "target": "parts"})
        if buckets.get("drawings"):
            actions.append({"label": "View Drawings", "action": "OPEN_TAB", "target": "drawings"})
        if buckets.get("images"):
            actions.append({"label": "View Images", "action": "OPEN_TAB", "target": "images"})
        if not actions:
            actions.append({"label": "Refine Search", "action": "SUGGEST_REWRITE", "target": None})
        return actions

    def _generate_related_searches(self, question: str, intent: Optional[str], nlp_analysis: Dict[str, Any]) -> List[str]:
        q = question.strip().rstrip("?")
        if intent == "parts":
            return [f"{q} manufacturer", f"{q} part number", f"{q} compatible parts"]
        if intent == "drawings":
            return [f"{q} electrical", f"{q} mechanical", f"{q} revision"]
        if intent == "positions":
            return [f"{q} location A", f"{q} station layout", f"{q} vicinity parts"]
        return [f"{q} documents", f"{q} images", f"{q} parts"]

    # ---------- Response helpers ----------
    def _enhance_unified_response(
        self,
        question: str,
        method: str,
        intent: Optional[str],
        entities: Dict[str, Any],
        organized: Dict[str, List[Dict[str, Any]]],
        raw: Dict[str, Any],
    ) -> Dict[str, Any]:
        return {
            "search_type": "unified",
            "status": "success",
            "query": question,
            "timestamp": datetime.utcnow().isoformat(),
            "detected_intent": intent or "UNKNOWN",
            "entities": entities or {},
            "results_by_type": organized,
            "summary": self._create_unified_search_summary(question, organized, intent),
            "quick_actions": self._generate_quick_actions(organized, question),
            "related_searches": self._generate_related_searches(question, intent, raw.get("nlp_analysis") or {}),
            "total_results": sum(len(v) for v in organized.values()),
            "search_method": method,
        }

    def _no_unified_results_response(self, question: str) -> Dict[str, Any]:
        return {
            "search_type": "unified",
            "status": "no_results",
            "query": question,
            "message": f"No results found for: {question}",
            "results_by_type": {"parts": [], "drawings": [], "images": [], "positions": [], "other": []},
            "timestamp": datetime.utcnow().isoformat(),
            "search_method": "none",
        }

    def _bad_request_response(self, msg: str) -> Dict[str, Any]:
        return {
            "search_type": "unified",
            "status": "error",
            "message": msg,
            "results_by_type": {},
            "timestamp": datetime.utcnow().isoformat(),
        }
