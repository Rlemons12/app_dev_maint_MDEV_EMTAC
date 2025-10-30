#!/usr/bin/env python3
"""
Search Adapters
---------------
- BaseSearchAdapter: abstract interface for all orchestrator search backends.
- PartsSearchAdapter: bridges NER entities → Part ORM search (FTS + fielded).
- DrawingsSearchAdapter: bridges NER entities → Drawing ORM search (fielded + text).

Usage example:
    parts = PartsSearchAdapter()
    p_results = parts.search(
        query="BALSTON FILT 200-80",
        entities={"PART_NUMBER": ["A101576"], "MANUFACTURER": ["BALSTON"]},
    )

    drawings = DrawingsSearchAdapter()
    d_results = drawings.search(
        query="11023-A rev B electrical",
        entities={"DRAWING_NUMBER": ["11023-A"], "DRAWING_TYPE": ["Electrical"], "DRAWING_REVISION": ["B"]},
    )
"""

from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional, Tuple, Callable
from dataclasses import dataclass
from enum import Enum

# ---- Logging (your custom logger) ---------------------------------------------
try:
    from modules.configuration.log_config import (
        debug_id, info_id, warning_id, error_id,
        with_request_id, get_request_id, log_timed_operation
    )
except Exception:  # graceful fallback for early dev
    def debug_id(msg, rid=None): print(f"DEBUG: {msg}")
    def info_id(msg, rid=None): print(f"INFO:  {msg}")
    def warning_id(msg, rid=None): print(f"WARN:  {msg}")
    def error_id(msg, rid=None): print(f"ERROR: {msg}")
    def with_request_id(fn): return fn
    def get_request_id(): return "no-request-id"
    class log_timed_operation:
        def __init__(self, *_a, **_k): pass
        def __enter__(self): return self
        def __exit__(self, *args): return False

# ---- Database session provider ------------------------------------------------
SessionFactory = Callable[[], "Session"]  # forward-declared type alias

try:
    # Adjust if your DatabaseConfig lives elsewhere
    from modules.configuration.config_env import DatabaseConfig
except Exception:
    DatabaseConfig = None  # will require external session or custom factory

# ---- SQLAlchemy / ORM imports -------------------------------------------------
try:
    from sqlalchemy.orm import Session
except Exception:
    Session = None  # type: ignore

# Adjust these imports to your actual model locations:
try:
    # e.g., from modules.emtacdb.models.part import Part
    from modules.emtacdb.emtacdb_fts import Part  # noqa
except Exception:
    Part = None  # type: ignore

try:
    # e.g., from modules.emtacdb.models.drawing import Drawing, DrawingType
    from modules.emtacdb.emtacdb_fts import Drawing, DrawingType  # noqa
except Exception:
    Drawing = None  # type: ignore
    class DrawingType(Enum):  # minimal fallback to avoid NameError in dev
        OTHER = "Other"


# ==============================================================================
# Base adapter
# ==============================================================================

@dataclass
class SearchItem:
    """Normalized item returned to the orchestrator."""
    type: str
    score: float
    source: str
    payload: Dict[str, Any]


class BaseSearchAdapter(ABC):
    """
    Abstract base for all search adapters.
    Concrete implementations must provide `search()`.
    """

    def __init__(self,
                 request_id: Optional[str] = None,
                 session: Optional["Session"] = None,
                 session_factory: Optional[SessionFactory] = None):
        """
        Args:
            request_id: For log correlation
            session: Optional SQLAlchemy session (adapter won't close it)
            session_factory: Optional callable returning a new session when needed
        """
        self.request_id = request_id or get_request_id()
        self._external_session = session
        self._session_factory = session_factory

    @abstractmethod
    def search(self, query: str, entities: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Perform the search and return a list of dicts consumable by the orchestrator."""
        raise NotImplementedError

    # ---- Helpers for session lifecycle ---------------------------------------

    def _resolve_session_factory(self) -> SessionFactory:
        """Pick an available session factory or raise a clear error."""
        if self._session_factory:
            return self._session_factory

        if self._external_session is not None:
            # If an external session is provided, we still return a no-op factory
            # that just yields that session (and signals not to close it).
            return lambda: self._external_session

        if DatabaseConfig is not None:
            cfg = DatabaseConfig()
            return cfg.get_main_session  # type: ignore[attr-defined]

        raise RuntimeError(
            "No session, no session_factory, and DatabaseConfig unavailable. "
            "Provide `session=` or `session_factory=` to the adapter."
        )

    def _open_session(self) -> Tuple["Session", bool]:
        """
        Returns:
            (session, close_when_done) where close_when_done indicates whether
            the adapter owns the session lifetime (True) or not (False).
        """
        if self._external_session is not None:
            return self._external_session, False
        session = self._resolve_session_factory()()
        return session, True

    @staticmethod
    def _as_plain_dict(item: SearchItem) -> Dict[str, Any]:
        """Convert a SearchItem dataclass to a plain dict."""
        return {"type": item.type, "score": item.score, "source": item.source, **item.payload}


# ==============================================================================
# PartsSearchAdapter: uses your Part.search and Part.fts_search
# ==============================================================================

class PartsSearchAdapter(BaseSearchAdapter):
    """
    Maps NER entities → Part ORM filters and executes:
      1) FTS (if appropriate), with ranking
      2) Fielded search (ILIKE/exact), as fallback or primary
    Normalizes results for the orchestrator.
    """

    DEFAULT_LIMIT = 50

    # Map NER labels to Part.search kwargs
    ENTITY_TO_PART_KW = {
        "PART_NUMBER": "part_number",
        "PART_NAME": "name",
        "MANUFACTURER": "oem_mfg",
        "MODEL": "model",
        "CLASS_FLAG": "class_flag",
        # extend as you add NER labels
    }

    def __init__(self,
                 request_id: Optional[str] = None,
                 session: Optional["Session"] = None,
                 session_factory: Optional[SessionFactory] = None,
                 limit: int = DEFAULT_LIMIT):
        super().__init__(request_id=request_id, session=session, session_factory=session_factory)
        self.limit = int(limit)

        if Part is None:
            raise RuntimeError("Part model import failed. Adjust the import path to your Part ORM class.")

    # ---- Public API -----------------------------------------------------------

    @with_request_id
    def search(self, query: str, entities: Dict[str, Any], request_id: Optional[str] = None) -> List[Dict[str, Any]]:
        rid = request_id or self.request_id
        sess, close_when_done = self._open_session()
        try:
            with log_timed_operation("PartsSearchAdapter.search", rid):
                strategy = self._pick_strategy(query, entities)
                debug_id(f"[PartsAdapter] strategy={strategy}; entities={entities}", rid)

                if strategy == "fts_first":
                    items = self._try_fts_then_fielded(query, entities, sess, rid)
                else:
                    items = self._fielded_then_optional_fts(query, entities, sess, rid)

                info_id(f"[PartsAdapter] returned {len(items)} items", rid)
                return [self._as_plain_dict(i) for i in items]

        except Exception as e:
            error_id(f"[PartsAdapter] fatal error: {e}", rid)
            return []
        finally:
            if close_when_done:
                sess.close()
                debug_id("[PartsAdapter] session closed", rid)

    # ---- Strategy & execution -------------------------------------------------

    def _pick_strategy(self, query: str, entities: Dict[str, Any]) -> str:
        """
        Heuristic: if we have strong field entities, do fielded first.
        If mostly free text, try FTS first.
        """
        strong = any(entities.get(k) for k in ("PART_NUMBER", "MANUFACTURER", "MODEL"))
        has_text = bool(query and query.strip())
        if has_text and not strong:
            return "fts_first"
        return "field_first"

    def _try_fts_then_fielded(self, query: str, entities: Dict[str, Any], session: "Session", rid: str) -> List[SearchItem]:
        # 1) FTS phase
        try:
            with log_timed_operation("Part.fts_search", rid):
                ranked = Part.fts_search(
                    search_text=query,
                    limit=self.limit,
                    request_id=rid,
                    session=session
                )
            if ranked:
                debug_id(f"[PartsAdapter] FTS returned {len(ranked)} rows", rid)
                return self._normalize_ranked(ranked, rid)
        except Exception as e:
            warning_id(f"[PartsAdapter] FTS failed; falling back to fielded: {e}", rid)

        # 2) Fielded fallback
        return self._run_fielded(query, entities, session, rid)

    def _fielded_then_optional_fts(self, query: str, entities: Dict[str, Any], session: "Session", rid: str) -> List[SearchItem]:
        items = self._run_fielded(query, entities, session, rid)
        if items:
            return items

        # Optionally try FTS if fielded produced nothing and we have text
        if query and query.strip():
            try:
                with log_timed_operation("Part.fts_search (optional)", rid):
                    ranked = Part.fts_search(
                        search_text=query,
                        limit=self.limit,
                        request_id=rid,
                        session=session
                    )
                if ranked:
                    debug_id(f"[PartsAdapter] Optional FTS returned {len(ranked)} rows", rid)
                    return self._normalize_ranked(ranked, rid)
            except Exception as e:
                warning_id(f"[PartsAdapter] Optional FTS also failed: {e}", rid)

        return []

    def _run_fielded(self, _query: str, entities: Dict[str, Any], session: "Session", rid: str) -> List[SearchItem]:
        kwargs = self._build_field_filters(entities)
        exact = kwargs.pop("exact_match", False)
        with log_timed_operation("Part.search(fielded)", rid):
            rows = Part.search(
                search_text=None,        # keep fielded phase purely structured
                fields=None,
                exact_match=exact,
                use_fts=False,
                limit=self.limit,
                request_id=rid,
                session=session,
                **kwargs
            )
        debug_id(f"[PartsAdapter] Fielded returned {len(rows)} rows", rid)
        return self._normalize_unranked(rows, rid)

    # ---- Mapping & normalization ---------------------------------------------

    def _build_field_filters(self, entities: Dict[str, Any]) -> Dict[str, Any]:
        """
        Map NER entities to Part.search kwargs. If we have a part_number, default to exact match.
        """
        kwargs: Dict[str, Any] = {}
        for ner_label, part_kw in self.ENTITY_TO_PART_KW.items():
            vals = entities.get(ner_label) or []
            if not vals:
                continue
            kwargs[part_kw] = vals[0]  # take the top candidate (expand to OR if needed)

        # exact if a specific part number is present
        kwargs["exact_match"] = "part_number" in kwargs
        return kwargs

    def _normalize_ranked(self, ranked: List[Tuple["Part", float]], _rid: str) -> List[SearchItem]:
        items: List[SearchItem] = []
        for part, rank in ranked:
            items.append(SearchItem(
                type="part",
                score=float(rank),
                source="parts_fts",
                payload={
                    "id": part.id,
                    "part_number": part.part_number,
                    "name": part.name,
                    "manufacturer": part.oem_mfg,
                    "model": part.model,
                    "class_flag": part.class_flag,
                    "notes": part.notes,
                    "documentation": part.documentation,
                }
            ))
        return items

    def _normalize_unranked(self, rows: List["Part"], _rid: str) -> List[SearchItem]:
        items: List[SearchItem] = []
        for part in rows:
            items.append(SearchItem(
                type="part",
                score=0.5,  # neutral score for unranked results; tune as needed
                source="parts_field",
                payload={
                    "id": part.id,
                    "part_number": part.part_number,
                    "name": part.name,
                    "manufacturer": part.oem_mfg,
                    "model": part.model,
                    "class_flag": part.class_flag,
                    "notes": part.notes,
                    "documentation": part.documentation,
                }
            ))
        return items


# ==============================================================================
# DrawingsSearchAdapter: uses your Drawing.search
# ==============================================================================

class DrawingsSearchAdapter(BaseSearchAdapter):
    """
    Maps NER → Drawing.search filters and returns normalized results.

    Strategy:
      - Build fielded filters from entities
      - If no strong entities, use free-text (search_text) across default fields
      - (If you add FTS later, mirror the pattern used in PartsSearchAdapter)
    """

    DEFAULT_LIMIT = 50

    # NER → Drawing.search kwarg map
    ENTITY_TO_KW = {
        "DRAWING_NUMBER": "drw_number",
        "DRAWING_NAME": "drw_name",
        "EQUIPMENT_NAME": "drw_equipment_name",
        "SPARE_PART_NUMBER": "drw_spare_part_number",
        "DRAWING_REVISION": "drw_revision",
        "DRAWING_TYPE": "drw_type",
        # Optional if your NER emits it:
        "FILE_PATH": "file_path",
    }

    def __init__(self,
                 request_id: Optional[str] = None,
                 session: Optional["Session"] = None,
                 session_factory: Optional[SessionFactory] = None,
                 limit: int = DEFAULT_LIMIT):
        super().__init__(request_id=request_id, session=session, session_factory=session_factory)
        self.limit = int(limit)
        if Drawing is None:
            raise RuntimeError("Drawing model import failed. Adjust the import path to your Drawing ORM class.")

    # ---- Public API -----------------------------------------------------------

    @with_request_id
    def search(self, query: str, entities: Dict[str, Any], request_id: Optional[str] = None) -> List[Dict[str, Any]]:
        rid = request_id or self.request_id
        session, close_when_done = self._open_session()

        try:
            with log_timed_operation("DrawingsSearchAdapter.search", rid):
                strong = self._has_strong_entities(entities)
                kwargs = self._build_field_filters(entities)

                search_text = None
                exact = False

                # If we have no strong entities but do have text,
                # let Drawing.search use text across its default fields.
                if not strong and query and query.strip():
                    search_text = query.strip()

                debug_id(f"[DrawingsAdapter] strong={strong}, kwargs={kwargs}, search_text={search_text!r}", rid)

                rows = Drawing.search(
                    search_text=search_text,
                    fields=None,             # let model pick defaults when using search_text
                    exact_match=exact,
                    drawing_id=None,         # NER rarely sets this; add if needed
                    limit=self.limit,
                    request_id=rid,
                    session=session,
                    **kwargs
                )

                info_id(f"[DrawingsAdapter] returned {len(rows)} rows", rid)
                return [self._normalize_row(r) for r in rows]

        except Exception as e:
            error_id(f"[DrawingsAdapter] fatal error: {e}", rid)
            return []
        finally:
            if close_when_done:
                session.close()
                debug_id("[DrawingsAdapter] session closed", rid)

    # ---- Helpers --------------------------------------------------------------

    def _has_strong_entities(self, entities: Dict[str, Any]) -> bool:
        """Consider these 'strong': number, type, revision, equipment."""
        for k in ("DRAWING_NUMBER", "DRAWING_TYPE", "DRAWING_REVISION", "EQUIPMENT_NAME"):
            if entities.get(k):
                return True
        return False

    def _build_field_filters(self, entities: Dict[str, Any]) -> Dict[str, Any]:
        """
        Map NER entities → Drawing.search kwargs. Coerce type to a known value when possible.
        """
        kwargs: Dict[str, Any] = {}

        for ner_label, kw in self.ENTITY_TO_KW.items():
            vals = entities.get(ner_label) or []
            if not vals:
                continue
            val = vals[0]

            if kw == "drw_type":
                # Best-effort coercion to one of your enum values (case-insensitive)
                kwargs[kw] = self._coerce_type(val)
            else:
                kwargs[kw] = val
        return kwargs

    def _coerce_type(self, text: str) -> str:
        """
        Try to match a free-text type to a known DrawingType value (case-insensitive).
        Falls back to the given text if no match.
        """
        try:
            valid_types = [t.value for t in DrawingType]  # type: ignore
            low = text.strip().lower()
            for vt in valid_types:
                if vt.lower() == low:
                    return vt
        except Exception:
            pass
        return text

    @staticmethod
    def _normalize_row(d: "Drawing") -> Dict[str, Any]:
        """Normalize a Drawing ORM row into orchestrator-friendly dict."""
        return {
            "type": "drawing",
            "score": 0.5,  # neutral; raise later if you add ranking
            "source": "drawings_field",
            "id": d.id,
            "number": d.drw_number,
            "name": d.drw_name,
            "equipment_name": d.drw_equipment_name,
            "revision": d.drw_revision,
            "spare_part_number": d.drw_spare_part_number,
            "drawing_type": d.drw_type,
            "file_path": d.file_path,
            "url": f"/drawings/view/{d.id}",
        }
