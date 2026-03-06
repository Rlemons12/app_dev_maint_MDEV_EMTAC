from __future__ import annotations

import os
import sys
import json
import uuid
import threading
import traceback
import contextvars
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, Optional, Callable

from modules.observability.models import TraceSpan, TraceEvent


# --------------------------------------------------
# Context (thread + async safe)
# --------------------------------------------------

ctx_trace_id = contextvars.ContextVar("trace_id", default=None)
ctx_request_id = contextvars.ContextVar("request_id", default=None)
ctx_stack = contextvars.ContextVar("span_stack", default=())
ctx_span_map = contextvars.ContextVar("span_map", default=None)


# --------------------------------------------------
# Config Defaults
# --------------------------------------------------

DEFAULT_ALLOWED_PREFIXES = ("modules.", "blueprints.", "plugins.")

DEFAULT_IGNORED_FUNCTIONS = {
    "wrapper",
    "debug_id",
    "info_id",
    "warning_id",
    "error_id",
    "log_with_id",
    "get_request_id",
    "<genexpr>",
    "<listcomp>",
    "<lambda>",
}

MAX_SPANS_PER_TRACE_DEFAULT = 5000
MAX_META_BYTES_DEFAULT = 8192


# --------------------------------------------------
# Utilities
# --------------------------------------------------

def utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _safe_json_size(obj: Any, *, max_bytes: int) -> Dict[str, Any]:
    try:
        raw = json.dumps(obj, default=str)
        if len(raw.encode("utf-8")) <= max_bytes:
            return obj
        return {
            "truncated": True,
            "note": f"metadata exceeded {max_bytes} bytes",
        }
    except Exception:
        return {"nonserializable": True}


def _safe_relpath(path: str, root: str) -> str:
    try:
        return os.path.relpath(path, root).replace("\\", "/")
    except Exception:
        return (path or "").replace("\\", "/")


# --------------------------------------------------
# Trace Config
# --------------------------------------------------

@dataclass
class TraceConfig:
    enabled: bool = True
    deep_profile: bool = False
    capture_exceptions: bool = True
    allowed_prefixes: tuple[str, ...] = DEFAULT_ALLOWED_PREFIXES
    ignored_functions: set[str] = None
    max_spans_per_trace: int = MAX_SPANS_PER_TRACE_DEFAULT
    max_meta_bytes: int = MAX_META_BYTES_DEFAULT

    def __post_init__(self):
        if self.ignored_functions is None:
            self.ignored_functions = set(DEFAULT_IGNORED_FUNCTIONS)


# --------------------------------------------------
# High-End Tracer
# --------------------------------------------------

class HighEndTracer:
    """
    Pure tracing engine.

    - Span lifecycle
    - Event handling
    - Deep profiling
    - Context management
    - No DB transactions
    """

    def __init__(
        self,
        *,
        project_root: Optional[str] = None,
        config: Optional[TraceConfig] = None,
    ):
        self.config = config or TraceConfig()
        self.project_root = project_root or os.path.abspath(
            os.path.join(os.path.dirname(__file__), "../../")
        )

    # --------------------------------------------------
    # Context Management
    # --------------------------------------------------

    def new_trace_context(self, trace_id: uuid.UUID):
        ctx_trace_id.set(trace_id)
        ctx_stack.set(tuple())
        ctx_span_map.set({"_span_count": 0})

    def set_request_id(self, request_id: Optional[str]):
        ctx_request_id.set(request_id)

    def get_request_id(self) -> Optional[str]:
        return ctx_request_id.get()

    def get_trace_id(self) -> Optional[uuid.UUID]:
        return ctx_trace_id.get()

    def attach_session(self, session):
        span_map = ctx_span_map.get()
        if span_map is not None:
            span_map["_db_session"] = session

    # --------------------------------------------------
    # Span Context Manager
    # --------------------------------------------------

    class _SpanCtx:
        def __init__(self, tracer: "HighEndTracer", name: str, meta: Optional[Dict[str, Any]]):
            self.tracer = tracer
            self.name = name
            self.meta = meta or {}
            self.span_id = None

        def __enter__(self):
            self.span_id = self.tracer._start_span(self.name, self.meta)
            return self

        def __exit__(self, exc_type, exc, tb):
            self.tracer._end_span(self.span_id, exc_type, exc, tb)
            return False

    def span(self, name: str, *, meta: Optional[Dict[str, Any]] = None):
        return HighEndTracer._SpanCtx(self, name, meta)

    # --------------------------------------------------
    # Span Lifecycle
    # --------------------------------------------------

    def _start_span(self, name: str, meta: Dict[str, Any]) -> uuid.UUID:
        span_map = ctx_span_map.get()
        stack = list(ctx_stack.get())

        if not span_map:
            return uuid.uuid4()

        if span_map["_span_count"] >= self.config.max_spans_per_trace:
            return uuid.uuid4()

        trace_id = ctx_trace_id.get()
        request_id = ctx_request_id.get()

        span_id = uuid.uuid4()
        parent_id = stack[-1] if stack else None

        stack.append(span_id)
        ctx_stack.set(tuple(stack))

        meta_bounded = _safe_json_size(meta, max_bytes=self.config.max_meta_bytes)

        span = TraceSpan(
            id=span_id,
            trace_id=trace_id,
            parent_span_id=parent_id,
            name=name,

            qualified_name=meta.get("qualified_name"),
            module_name=meta.get("module_name"),
            file_path=meta.get("file_path"),
            line_number=meta.get("line_number"),

            metadata_json=meta_bounded,
            depth=len(stack) - 1,
            started_at=utcnow(),
            status="running",
            request_id=request_id,
            thread_id=threading.get_ident(),
            process_id=os.getpid(),
        )

        span_map[span_id] = span
        span_map["_span_count"] += 1

        session = span_map.get("_db_session")
        if session:
            session.add(span)

        return span_id

    def _end_span(self, span_id, exc_type, exc, tb):
        span_map = ctx_span_map.get()
        if not span_map:
            return

        span = span_map.get(span_id)
        if not span:
            return

        try:
            # --------------------------------------------------
            # End timing
            # --------------------------------------------------
            span.ended_at = utcnow()

            if span.started_at:
                span.duration_ms = (
                                           span.ended_at - span.started_at
                                   ).total_seconds() * 1000.0
            else:
                span.duration_ms = 0.0

            # --------------------------------------------------
            # Status handling
            # --------------------------------------------------
            if exc:
                span.status = "failed"

                if self.config.capture_exceptions:
                    try:
                        span.exception = "".join(
                            traceback.format_exception(exc_type, exc, tb)
                        )
                    except Exception:
                        span.exception = "Exception formatting failed"
            else:
                span.status = "completed"

            # --------------------------------------------------
            # Automatic Performance Alert (Slow Span Detection)
            # --------------------------------------------------
            try:
                threshold_ms = 500  # Adjust threshold as needed

                if (
                        span.duration_ms is not None
                        and span.duration_ms > threshold_ms
                ):
                    session = span_map.get("_db_session")

                    if session:
                        from modules.observability.models import TraceAlert

                        alert = TraceAlert(
                            span_id=span.id,
                            metric_type="duration_ms",
                            threshold_value=threshold_ms,
                            actual_value=span.duration_ms,
                            severity="warning",
                        )

                        session.add(alert)

            except Exception:
                # Alerts must never break tracing
                pass

        except Exception:
            # Span closing must never break production code
            pass

        # --------------------------------------------------
        # Safe stack cleanup
        # --------------------------------------------------
        try:
            stack = list(ctx_stack.get())

            if stack and stack[-1] == span_id:
                stack.pop()
                ctx_stack.set(tuple(stack))
            else:
                # Defensive: remove if it exists somewhere deeper
                if span_id in stack:
                    stack.remove(span_id)
                    ctx_stack.set(tuple(stack))

        except Exception:
            pass

    def force_close_open_spans(self, status="completed"):
        span_map = ctx_span_map.get()
        if not span_map:
            return

        stack = list(ctx_stack.get())
        while stack:
            sid = stack.pop()
            span = span_map.get(sid)
            if span and not span.ended_at:
                span.ended_at = utcnow()
                span.duration_ms = (
                    span.ended_at - span.started_at
                ).total_seconds() * 1000.0
                span.status = status

        ctx_stack.set(tuple(stack))

    # --------------------------------------------------
    # Event Handling (Safe)
    # --------------------------------------------------

    def event(self, event_type: str, payload: Dict[str, Any]):
        try:
            span_map = ctx_span_map.get()
            stack = list(ctx_stack.get())

            if not span_map or not stack:
                return

            current_span_id = stack[-1]
            session = span_map.get("_db_session")

            if not session:
                return

            event = TraceEvent(
                span_id=current_span_id,
                event_type=event_type,
                payload=_safe_json_size(
                    payload,
                    max_bytes=self.config.max_meta_bytes,
                ),
            )

            session.add(event)

        except Exception:
            # Observability must NEVER break business logic
            pass

    # --------------------------------------------------
    # Deep Profiling
    # --------------------------------------------------

    def build_profiler(self) -> Callable:
        allowed = self.config.allowed_prefixes
        ignored = self.config.ignored_functions
        project_root = self.project_root

        def profiler(frame, event, arg):
            module_name = frame.f_globals.get("__name__", "")
            func_name = frame.f_code.co_name

            if not module_name.startswith(allowed):
                return profiler
            if func_name in ignored:
                return profiler

            if event == "call":
                filename = frame.f_code.co_filename
                line_no = frame.f_lineno
                rel_path = _safe_relpath(filename, project_root)

                meta = {
                    "module_name": module_name,
                    "file_path": rel_path,
                    "line_number": line_no,
                    "qualified_name": f"{module_name}.{func_name}",
                }

                self._start_span(func_name, meta)

            elif event == "return":
                stack = list(ctx_stack.get())
                if stack:
                    self._end_span(stack[-1], None, None, None)

            return profiler

        return profiler


# --------------------------------------------------
# Singleton
# --------------------------------------------------

tracer = HighEndTracer()