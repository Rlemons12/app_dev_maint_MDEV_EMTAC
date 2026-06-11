# modules/decorators/trace_decorator.py

from __future__ import annotations

import sys
import time
import uuid
import traceback
import logging
from functools import wraps
from typing import Callable, Any, Optional, Dict

from modules.configuration.config_env import get_db_config
from modules.observability.models import TraceSession
from modules.observability.high_end_tracer import tracer, utcnow
from modules.observability.config_trace import (
    TRACE_DEEP_PROFILE_ENV,
    TRACE_CAPTURE_ARGS_ENV,
    TRACE_CAPTURE_RETURN_ENV,
    is_trace_entrypoint_enabled,
    resolve_behavior_flags,
    get_service_name,
    get_environment_name,
)

db = get_db_config()
logger = logging.getLogger(__name__)


# ==========================================================
# Safe Preview Helpers
# ==========================================================

def _safe_preview(
    value: Any,
    *,
    limit: int = 500,
) -> str:
    try:
        text = repr(value)
    except Exception:
        return "<unserializable>"

    if len(text) > limit:
        return text[:limit] + "...[truncated]"

    return text


def _safe_args_preview(
    args: tuple,
    *,
    max_items: int = 10,
    item_limit: int = 200,
) -> list[str]:
    preview: list[str] = []

    for item in list(args)[:max_items]:
        preview.append(_safe_preview(item, limit=item_limit))

    return preview


def _safe_kwargs_preview(
    kwargs: Dict[str, Any],
    *,
    max_items: int = 30,
    item_limit: int = 200,
) -> Dict[str, str]:
    preview: Dict[str, str] = {}

    for key, value in list(kwargs.items())[:max_items]:
        preview[str(key)] = _safe_preview(value, limit=item_limit)

    return preview


# ==========================================================
# Trace Engine / Config Helpers
# ==========================================================

def _is_tracer_engine_enabled() -> bool:
    """
    Check the low-level tracer singleton switch.

    This is separate from config_trace.py because tracer.config.enabled belongs
    to the tracing engine itself.
    """

    try:
        return bool(tracer.config.enabled)
    except Exception:
        return False


def _should_create_trace(
    *,
    enabled: Optional[bool],
    enabled_env: Optional[str],
) -> bool:
    """
    Decide if this entrypoint should create a root TraceSession.

    Rules:
        1. The tracer engine must be enabled.
        2. config_trace master setting must be enabled.
        3. decorator enabled=False disables this entrypoint.
        4. enabled_env controls this specific trace group.
    """

    if not _is_tracer_engine_enabled():
        return False

    return is_trace_entrypoint_enabled(
        enabled=enabled,
        enabled_env=enabled_env,
    )


def _resolve_request_id() -> str:
    """
    Resolve request_id from the active trace context if available.

    The actual Flask route wrapper may pass request_id in kwargs, but depending
    on decorator order, it may already be bound only inside log_config context.
    """

    request_id = tracer.get_request_id()

    if request_id:
        return str(request_id)

    return f"trace-{uuid.uuid4()}"


def _resolve_request_id_from_call(
    *,
    kwargs: Dict[str, Any],
) -> str:
    request_id = kwargs.get("request_id") or tracer.get_request_id()

    if request_id:
        return str(request_id)

    return f"trace-{uuid.uuid4()}"


def _build_root_meta(
    *,
    func: Callable[..., Any],
    entry_name: str,
    trace_id: uuid.UUID,
    request_id: str,
    enabled_env: Optional[str],
    deep_profile: bool,
    capture_args: bool,
    capture_return: bool,
    args: tuple,
    kwargs: Dict[str, Any],
) -> Dict[str, Any]:
    root_meta: Dict[str, Any] = {
        "entrypoint": entry_name,
        "qualified_name": f"{func.__module__}.{func.__name__}",
        "module_name": func.__module__,
        "trace_id": str(trace_id),
        "request_id": request_id,
        "enabled_env": enabled_env,
        "deep_profile": deep_profile,
        "capture_args": capture_args,
        "capture_return": capture_return,
    }

    if capture_args:
        try:
            root_meta["args_preview"] = _safe_args_preview(args)
        except Exception:
            root_meta["args_preview"] = ["<unavailable>"]

        try:
            root_meta["kwargs_preview"] = _safe_kwargs_preview(kwargs)
        except Exception:
            root_meta["kwargs_preview"] = {"error": "<unavailable>"}

    return root_meta


# ==========================================================
# Safe Event Helpers
# ==========================================================

def _emit_exception_event(
    *,
    request_id: str,
    trace_id: uuid.UUID,
    exc: BaseException,
) -> None:
    try:
        tracer.event(
            "exception",
            {
                "request_id": request_id,
                "trace_id": str(trace_id),
                "type": type(exc).__name__,
                "message": str(exc),
                "stack": traceback.format_exc(limit=50),
            },
        )
    except Exception as event_exc:
        logger.exception(
            "[TRACE] Failed logging exception event | request_id=%s | trace_id=%s | err=%s",
            request_id,
            trace_id,
            event_exc,
        )


def _emit_return_preview_event(
    *,
    request_id: str,
    trace_id: uuid.UUID,
    result: Any,
) -> None:
    try:
        tracer.event(
            "return_preview",
            {
                "request_id": request_id,
                "trace_id": str(trace_id),
                "value": _safe_preview(result, limit=500),
            },
        )
    except Exception as return_event_exc:
        logger.exception(
            "[TRACE] Failed logging return preview | request_id=%s | trace_id=%s | err=%s",
            request_id,
            trace_id,
            return_event_exc,
        )


# ==========================================================
# Nested Entrypoint Safety
# ==========================================================

def _run_as_child_span_only(
    *,
    func: Callable[..., Any],
    args: tuple,
    kwargs: Dict[str, Any],
    entry_name: str,
    request_id: str,
    capture_return: bool,
) -> Any:
    """
    Safety path for accidental nested @trace_entrypoint usage.

    If a trace is already active, this decorator must NOT create another
    TraceSession. It becomes a child span instead.

    This prevents:
        - Multiple root traces for one user question
        - Broken parent/child trees
        - Trace context cleanup happening too early
        - Confusing dashboard results
    """

    active_trace_id = tracer.get_trace_id()

    with tracer.span(
        entry_name,
        meta={
            "entrypoint": entry_name,
            "qualified_name": f"{func.__module__}.{func.__name__}",
            "module_name": func.__module__,
            "request_id": request_id,
            "trace_id": str(active_trace_id) if active_trace_id else None,
            "nested_entrypoint_suppressed": True,
        },
    ):
        try:
            result = func(*args, **kwargs)

            if capture_return:
                _emit_return_preview_event(
                    request_id=request_id,
                    trace_id=active_trace_id or uuid.uuid4(),
                    result=result,
                )

            return result

        except Exception as exc:
            _emit_exception_event(
                request_id=request_id,
                trace_id=active_trace_id or uuid.uuid4(),
                exc=exc,
            )
            raise


# ==========================================================
# Persistence Helpers
# ==========================================================

def _create_trace_session(
    *,
    trace_id: uuid.UUID,
    request_id: str,
    service_name: str,
    environment: str,
) -> bool:
    """
    Create TraceSession in a short-lived DB session.

    Returns:
        True if created successfully, False otherwise.
    """

    try:
        with db.main_session() as session:
            trace_session = TraceSession(
                id=trace_id,
                request_id=request_id,
                started_at=utcnow(),
                status="running",
                service_name=service_name,
                environment=environment,
                sampled=True,
            )
            session.add(trace_session)
            session.commit()

        return True

    except Exception as create_exc:
        logger.exception(
            "[TRACE] Failed creating TraceSession | request_id=%s | trace_id=%s | err=%s",
            request_id,
            trace_id,
            create_exc,
        )
        return False


def _persist_trace_buffers_and_finalize(
    *,
    trace_id: uuid.UUID,
    request_id: str,
    root_span_id: Optional[uuid.UUID],
    start_perf_counter: float,
    business_exc: Optional[BaseException],
    service_name: str,
    environment: str,
    trace_session_created: bool,
) -> None:
    """
    Drain buffered spans/events and finalize TraceSession.

    Observability failures are logged and suppressed. This function must never
    invalidate successful business logic.
    """

    try:
        buffered = tracer.drain_trace_buffers()
        buffered_spans = buffered.get("spans", [])
        buffered_events = buffered.get("events", [])
    except Exception as drain_exc:
        logger.exception(
            "[TRACE] Failed draining trace buffers | request_id=%s | trace_id=%s | err=%s",
            request_id,
            trace_id,
            drain_exc,
        )
        buffered_spans = []
        buffered_events = []

    if not trace_session_created:
        return

    try:
        with db.main_session() as session:
            # Re-attach session only for this short flush window.
            try:
                tracer.attach_session(session)
            except Exception:
                pass

            # Persist spans.
            for span in buffered_spans:
                try:
                    session.merge(span)
                except Exception as span_persist_exc:
                    logger.exception(
                        "[TRACE] Failed persisting span | request_id=%s | trace_id=%s | span_id=%s | err=%s",
                        request_id,
                        trace_id,
                        getattr(span, "id", None),
                        span_persist_exc,
                    )

            # Persist events.
            for event in buffered_events:
                try:
                    session.add(event)
                except Exception as event_persist_exc:
                    logger.exception(
                        "[TRACE] Failed persisting event | request_id=%s | trace_id=%s | span_id=%s | err=%s",
                        request_id,
                        trace_id,
                        getattr(event, "span_id", None),
                        event_persist_exc,
                    )

            # Finalize TraceSession.
            trace_session = session.get(TraceSession, trace_id)

            if trace_session is not None:
                trace_session.root_span_id = root_span_id
                trace_session.ended_at = utcnow()
                trace_session.duration_ms = (
                    time.perf_counter() - start_perf_counter
                ) * 1000.0
                trace_session.status = (
                    "failed" if business_exc is not None else "completed"
                )
                trace_session.service_name = service_name
                trace_session.environment = environment

                session.commit()

            else:
                logger.warning(
                    "[TRACE] TraceSession missing during finalization | request_id=%s | trace_id=%s",
                    request_id,
                    trace_id,
                )

    except Exception as trace_commit_exc:
        logger.exception(
            "[TRACE] Finalization commit failed | request_id=%s | trace_id=%s | err=%s",
            request_id,
            trace_id,
            trace_commit_exc,
        )

    finally:
        try:
            tracer.attach_session(None)
        except Exception:
            pass


# ==========================================================
# Trace Entrypoint Decorator
# ==========================================================

def trace_entrypoint(
    _func: Callable = None,
    *,
    name: Optional[str] = None,

    # Master/per-route control
    enabled: Optional[bool] = None,
    enabled_env: Optional[str] = None,

    # Behavior controls
    deep_profile: bool = False,
    capture_args: bool = False,
    capture_return: bool = False,

    # Optional runtime/env overrides for behavior controls
    deep_profile_env: Optional[str] = TRACE_DEEP_PROFILE_ENV,
    capture_args_env: Optional[str] = TRACE_CAPTURE_ARGS_ENV,
    capture_return_env: Optional[str] = TRACE_CAPTURE_RETURN_ENV,

    # Optional TraceSession labels
    service_name: Optional[str] = None,
    environment: Optional[str] = None,
):
    """
    Top-level trace boundary decorator.

    Behavior:
        - Creates one TraceSession for a request-level entrypoint
        - Creates one root span
        - Executes business logic outside the observability DB session
        - Buffers spans/events in memory via the global tracer singleton
        - Finalizes TraceSession and flushes spans/events afterward
        - Suppresses observability failures so tracing never breaks business logic
        - Prevents accidental nested root traces

    Runtime/env switches come from modules.observability.config_trace.

    Common settings:
        EMTAC_TRACE_ENABLED=1              master tracing on/off
        EMTAC_TRACE_CHAT_ENABLED=1         chat answer tracing
        EMTAC_TRACE_PAYLOAD_ENABLED=1      chat payload tracing
        EMTAC_TRACE_FEEDBACK_ENABLED=0     feedback route tracing
        EMTAC_TRACE_HEALTH_ENABLED=0       health/metrics tracing
        EMTAC_TRACE_DEEP_PROFILE=0         deep sys.setprofile tracing
        EMTAC_TRACE_CAPTURE_ARGS=0         capture root args preview
        EMTAC_TRACE_CAPTURE_RETURN=0       capture root return preview

    Example route usage:

        @trace_entrypoint(
            name="chat.ask",
            enabled_env="EMTAC_TRACE_CHAT_ENABLED",
            deep_profile=False,
            capture_args=False,
            capture_return=False,
        )
        def ask(...):
            ...

    Important:
        Use @trace_entrypoint only on true request-level entrypoints.
        Use tracer.span(...) inside coordinators/orchestrators/services.
    """

    def decorator(func: Callable[..., Any]):

        @wraps(func)
        def wrapper(*args, **kwargs):
            if not _should_create_trace(
                enabled=enabled,
                enabled_env=enabled_env,
            ):
                return func(*args, **kwargs)

            behavior_flags = resolve_behavior_flags(
                deep_profile=deep_profile,
                capture_args=capture_args,
                capture_return=capture_return,
                deep_profile_env=deep_profile_env,
                capture_args_env=capture_args_env,
                capture_return_env=capture_return_env,
            )

            entry_name = name or func.__name__

            request_id = _resolve_request_id_from_call(kwargs=kwargs)

            # ---------------------------------------------------------
            # Prevent multiple root TraceSessions for one active trace.
            # ---------------------------------------------------------
            existing_trace_id = tracer.get_trace_id()

            if existing_trace_id is not None:
                return _run_as_child_span_only(
                    func=func,
                    args=args,
                    kwargs=kwargs,
                    entry_name=entry_name,
                    request_id=request_id,
                    capture_return=behavior_flags.capture_return,
                )

            trace_id = uuid.uuid4()
            tracer.new_trace_context(trace_id)
            tracer.set_request_id(request_id)

            business_exc: Optional[BaseException] = None
            result: Any = None
            profiler = None
            root_span_id = None
            start_perf_counter = time.perf_counter()

            effective_service_name = service_name or get_service_name()
            effective_environment = environment or get_environment_name()

            trace_session_created = _create_trace_session(
                trace_id=trace_id,
                request_id=request_id,
                service_name=effective_service_name,
                environment=effective_environment,
            )

            try:
                # Ensure no long-lived DB session is attached during business execution.
                try:
                    tracer.attach_session(None)
                except Exception:
                    pass

                # ---------------------------------------------------------
                # Root span
                # ---------------------------------------------------------
                root_meta = _build_root_meta(
                    func=func,
                    entry_name=entry_name,
                    trace_id=trace_id,
                    request_id=request_id,
                    enabled_env=enabled_env,
                    deep_profile=behavior_flags.deep_profile,
                    capture_args=behavior_flags.capture_args,
                    capture_return=behavior_flags.capture_return,
                    args=args,
                    kwargs=kwargs,
                )

                try:
                    root_span_id = tracer._start_span(entry_name, root_meta)
                except Exception as span_start_exc:
                    logger.exception(
                        "[TRACE] Failed starting root span | request_id=%s | trace_id=%s | err=%s",
                        request_id,
                        trace_id,
                        span_start_exc,
                    )
                    root_span_id = None

                # ---------------------------------------------------------
                # Optional deep profiling
                # ---------------------------------------------------------
                profiler = (
                    tracer.build_profiler()
                    if behavior_flags.deep_profile
                    else None
                )

                if profiler:
                    try:
                        sys.setprofile(profiler)
                    except Exception as profile_exc:
                        logger.exception(
                            "[TRACE] Failed enabling profiler | request_id=%s | trace_id=%s | err=%s",
                            request_id,
                            trace_id,
                            profile_exc,
                        )
                        profiler = None

                # ---------------------------------------------------------
                # Execute business logic
                # ---------------------------------------------------------
                try:
                    result = func(*args, **kwargs)

                    if behavior_flags.capture_return:
                        _emit_return_preview_event(
                            request_id=request_id,
                            trace_id=trace_id,
                            result=result,
                        )

                except Exception as exc:
                    business_exc = exc

                    _emit_exception_event(
                        request_id=request_id,
                        trace_id=trace_id,
                        exc=exc,
                    )

                finally:
                    # -----------------------------------------------------
                    # Disable profiler
                    # -----------------------------------------------------
                    if profiler:
                        try:
                            sys.setprofile(None)
                        except Exception as disable_profile_exc:
                            logger.exception(
                                "[TRACE] Failed disabling profiler | request_id=%s | trace_id=%s | err=%s",
                                request_id,
                                trace_id,
                                disable_profile_exc,
                            )

                    # -----------------------------------------------------
                    # Close root span safely
                    # -----------------------------------------------------
                    try:
                        if root_span_id is not None:
                            tracer._end_span(
                                root_span_id,
                                type(business_exc) if business_exc is not None else None,
                                business_exc,
                                business_exc.__traceback__ if business_exc is not None else None,
                            )
                    except Exception as span_end_exc:
                        logger.exception(
                            "[TRACE] Failed ending root span | request_id=%s | trace_id=%s | err=%s",
                            request_id,
                            trace_id,
                            span_end_exc,
                        )

                    # -----------------------------------------------------
                    # Force close any remaining spans safely
                    # -----------------------------------------------------
                    try:
                        tracer.force_close_open_spans(
                            status=(
                                "failed"
                                if business_exc is not None
                                else "completed"
                            )
                        )
                    except Exception as force_close_exc:
                        logger.exception(
                            "[TRACE] Failed force-closing spans | request_id=%s | trace_id=%s | err=%s",
                            request_id,
                            trace_id,
                            force_close_exc,
                        )

                    # -----------------------------------------------------
                    # Persist spans/events and finalize TraceSession
                    # -----------------------------------------------------
                    _persist_trace_buffers_and_finalize(
                        trace_id=trace_id,
                        request_id=request_id,
                        root_span_id=root_span_id,
                        start_perf_counter=start_perf_counter,
                        business_exc=business_exc,
                        service_name=effective_service_name,
                        environment=effective_environment,
                        trace_session_created=trace_session_created,
                    )

                # ---------------------------------------------------------
                # Re-raise original business exception after trace cleanup
                # ---------------------------------------------------------
                if business_exc is not None:
                    raise business_exc

                return result

            finally:
                # Best-effort tracer cleanup so stale context does not leak.
                try:
                    tracer.attach_session(None)
                except Exception:
                    pass

                try:
                    tracer.clear_trace_context()
                except Exception:
                    pass

        return wrapper

    # Allow both @trace_entrypoint and @trace_entrypoint(...)
    if _func and callable(_func):
        return decorator(_func)

    return decorator