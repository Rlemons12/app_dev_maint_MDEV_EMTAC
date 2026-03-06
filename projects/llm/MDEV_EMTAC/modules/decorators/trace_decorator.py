# modules/decorators/trace_decorator.py

from __future__ import annotations

import sys
import time
import uuid
import traceback
from functools import wraps
from typing import Callable, Any, Optional

from modules.configuration.config_env import get_db_config
from modules.observability.models import TraceSession
from modules.observability.high_end_tracer import tracer, utcnow

db = get_db_config()


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

    - Creates TraceSession
    - Creates root span
    - Commits all spans at end
    - Optional deep profiling
    """

    def decorator(func: Callable[..., Any]):

        @wraps(func)
        def wrapper(*args, **kwargs):

            if not tracer.config.enabled:
                return func(*args, **kwargs)

            trace_id = uuid.uuid4()
            tracer.new_trace_context(trace_id)

            request_id = kwargs.get("request_id") or tracer.get_request_id()
            tracer.set_request_id(request_id)

            entry_name = name or func.__name__

            with db.main_session() as session:

                # Create TraceSession
                trace_session = TraceSession(
                    id=trace_id,
                    request_id=request_id,
                    started_at=utcnow(),
                    status="running",
                )

                session.add(trace_session)
                session.flush()

                tracer.attach_session(session)

                # Root span
                root_meta = {
                    "entrypoint": entry_name,
                    "qualified": f"{func.__module__}.{func.__name__}",
                }

                if capture_args:
                    root_meta["args_preview"] = [str(a)[:200] for a in args[:10]]
                    root_meta["kwargs_preview"] = {
                        k: str(v)[:200] for k, v in list(kwargs.items())[:30]
                    }

                root_span_id = tracer._start_span(entry_name, root_meta)

                profiler = tracer.build_profiler() if deep_profile else None

                start = time.perf_counter()

                if profiler:
                    sys.setprofile(profiler)

                try:
                    result = func(*args, **kwargs)
                    trace_session.status = "success"

                    if capture_return:
                        tracer.event(
                            "return_preview",
                            {"value": str(result)[:500]},
                        )

                    return result

                except Exception as e:
                    trace_session.status = "failed"

                    tracer.event(
                        "exception",
                        {
                            "type": type(e).__name__,
                            "message": str(e),
                            "stack": traceback.format_exc(limit=50),
                        },
                    )

                    raise

                finally:
                    if profiler:
                        sys.setprofile(None)

                    tracer._end_span(root_span_id, None, None, None)

                    tracer.force_close_open_spans(
                        status=(
                            "failed"
                            if trace_session.status == "failed"
                            else "completed"
                        )
                    )

                    trace_session.ended_at = utcnow()
                    trace_session.duration_ms = (
                        time.perf_counter() - start
                    ) * 1000.0

                    session.commit()

        return wrapper

    # Allow both @trace_entrypoint and @trace_entrypoint(...)
    if _func and callable(_func):
        return decorator(_func)

    return decorator