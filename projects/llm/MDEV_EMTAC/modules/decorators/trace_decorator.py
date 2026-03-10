# modules/decorators/trace_decorator.py

from __future__ import annotations

import sys
import time
import uuid
import traceback
import logging
from functools import wraps
from typing import Callable, Any, Optional

from modules.configuration.config_env import get_db_config
from modules.observability.models import TraceSession
from modules.observability.high_end_tracer import tracer, utcnow

db = get_db_config()
logger = logging.getLogger(__name__)


def trace_entrypoint(
    _func: Callable = None,
    *,
    name: Optional[str] = None,
    deep_profile: bool = False,
    capture_args: bool = False,
    capture_return: bool = False,
):
    """
    Top-level trace boundary decorator.

    Behavior:
    - Creates TraceSession in a short-lived DB session
    - Creates root span
    - Executes business logic outside the observability DB session
    - Buffers spans/events in memory via tracer
    - Finalizes TraceSession + flushes spans/events in a fresh short-lived DB session
    - Optional deep profiling

    Important:
    - Business exceptions are re-raised normally.
    - Observability finalization failures are logged and suppressed so they do
      not turn successful business work into failed business results.
    """

    def decorator(func: Callable[..., Any]):

        @wraps(func)
        def wrapper(*args, **kwargs):
            if not tracer.config.enabled:
                return func(*args, **kwargs)

            trace_id = uuid.uuid4()
            tracer.new_trace_context(trace_id)

            request_id = kwargs.get("request_id") or tracer.get_request_id()
            if not request_id:
                request_id = f"trace-{trace_id}"

            tracer.set_request_id(request_id)

            entry_name = name or func.__name__

            business_exc: Optional[BaseException] = None
            result: Any = None
            profiler = None
            root_span_id = None
            start = time.perf_counter()
            trace_session_created = False

            try:
                # ---------------------------------------------------------
                # Create TraceSession in a short-lived DB session
                # ---------------------------------------------------------
                try:
                    with db.main_session() as session:
                        trace_session = TraceSession(
                            id=trace_id,
                            request_id=request_id,
                            started_at=utcnow(),
                            status="running",
                        )
                        session.add(trace_session)
                        session.commit()
                        trace_session_created = True
                except Exception as create_exc:
                    logger.exception(
                        "[TRACE] Failed creating TraceSession | request_id=%s | trace_id=%s | err=%s",
                        request_id,
                        trace_id,
                        create_exc,
                    )

                # Ensure no long-lived DB session is attached during business execution
                try:
                    tracer.attach_session(None)
                except Exception:
                    pass

                # ---------------------------------------------------------
                # Root span
                # ---------------------------------------------------------
                root_meta = {
                    "entrypoint": entry_name,
                    "qualified_name": f"{func.__module__}.{func.__name__}",
                    "module_name": func.__module__,
                }

                if capture_args:
                    try:
                        root_meta["args_preview"] = [str(a)[:200] for a in args[:10]]
                    except Exception:
                        root_meta["args_preview"] = ["<unavailable>"]

                    try:
                        root_meta["kwargs_preview"] = {
                            k: str(v)[:200] for k, v in list(kwargs.items())[:30]
                        }
                    except Exception:
                        root_meta["kwargs_preview"] = {"error": "<unavailable>"}

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

                profiler = tracer.build_profiler() if deep_profile else None
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

                    if capture_return:
                        try:
                            tracer.event(
                                "return_preview",
                                {"value": str(result)[:500]},
                            )
                        except Exception as return_event_exc:
                            logger.exception(
                                "[TRACE] Failed logging return preview | request_id=%s | trace_id=%s | err=%s",
                                request_id,
                                trace_id,
                                return_event_exc,
                            )

                except Exception as exc:
                    business_exc = exc

                    try:
                        tracer.event(
                            "exception",
                            {
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
                            status=("failed" if business_exc is not None else "completed")
                        )
                    except Exception as force_close_exc:
                        logger.exception(
                            "[TRACE] Failed force-closing spans | request_id=%s | trace_id=%s | err=%s",
                            request_id,
                            trace_id,
                            force_close_exc,
                        )

                    # -----------------------------------------------------
                    # Drain buffered spans/events from tracer
                    # -----------------------------------------------------
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

                    # -----------------------------------------------------
                    # Finalize TraceSession and persist spans/events
                    # in one fresh short-lived DB session
                    # -----------------------------------------------------
                    if trace_session_created:
                        try:
                            with db.main_session() as session:
                                # Re-attach session only for this short flush window
                                try:
                                    tracer.attach_session(session)
                                except Exception:
                                    pass

                                # Persist spans
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

                                # Persist events
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

                                # Finalize TraceSession
                                trace_session = session.get(TraceSession, trace_id)

                                if trace_session is not None:
                                    trace_session.ended_at = utcnow()
                                    trace_session.duration_ms = (
                                        time.perf_counter() - start
                                    ) * 1000.0
                                    trace_session.status = (
                                        "failed" if business_exc is not None else "success"
                                    )

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
                            # Never invalidate successful business work due to observability failure

                        finally:
                            try:
                                tracer.attach_session(None)
                            except Exception:
                                pass

                # ---------------------------------------------------------
                # Re-raise original business exception after trace cleanup
                # ---------------------------------------------------------
                if business_exc is not None:
                    raise business_exc

                return result

            finally:
                # Best-effort tracer cleanup so stale context doesn't leak
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