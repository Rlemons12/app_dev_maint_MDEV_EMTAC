# modules/decorators/integration_trace.py

from functools import wraps
from typing import Callable, Any, Optional, Dict, Tuple

from modules.configuration.log_config import debug_id
from modules.observability.high_end_tracer import tracer


def integration_trace(
    func: Callable = None,
    *,
    injected_kwargs: Optional[Dict[str, Any]] = None,
    injected_args: Optional[Tuple[Any, ...]] = None,
    forced_return: Optional[Any] = None,
    show_injection: bool = True,
):
    """
    Testing utility decorator.

    Features:
        - Inject positional arguments
        - Inject keyword arguments
        - Force return value
        - Optionally show what was injected
        - Optionally emit tracer events if trace is active
        - Does NOT create TraceSession
    """

    def decorator(inner_func: Callable):

        @wraps(inner_func)
        def wrapper(*args, **kwargs):

            request_id = kwargs.get("request_id")

            original_args = args
            original_kwargs = kwargs

            modified_args = list(args)
            modified_kwargs = dict(kwargs)

            # -------------------------------------------------
            # Apply Injection
            # -------------------------------------------------

            if injected_args is not None:
                modified_args = list(injected_args)

            if injected_kwargs:
                modified_kwargs.update(injected_kwargs)

            # -------------------------------------------------
            # Show Injection (Logging + Tracer Event)
            # -------------------------------------------------

            if show_injection and (injected_args or injected_kwargs):

                injection_info = {
                    "function": inner_func.__qualname__,
                    "original_args": _safe_preview(original_args),
                    "original_kwargs": _safe_preview(original_kwargs),
                    "injected_args": _safe_preview(injected_args),
                    "injected_kwargs": _safe_preview(injected_kwargs),
                }

                # Structured debug log
                debug_id(
                    f"[integration_trace] Injection → {injection_info}",
                    request_id,
                )

                # If tracing active, emit event
                if tracer.get_trace_id():
                    tracer.event("integration_injection", injection_info)

            # -------------------------------------------------
            # Forced Return
            # -------------------------------------------------

            if forced_return is not None:

                if show_injection:
                    debug_id(
                        f"[integration_trace] Forced Return → {forced_return}",
                        request_id,
                    )

                if tracer.get_trace_id():
                    tracer.event(
                        "integration_forced_return",
                        {"value": _safe_preview(forced_return)},
                    )

                return forced_return

            # -------------------------------------------------
            # Execute Function
            # -------------------------------------------------

            return inner_func(*modified_args, **modified_kwargs)

        return wrapper

    if func and callable(func):
        return decorator(func)

    return decorator


# ------------------------------------------------------------
# Safe Preview Helpers
# ------------------------------------------------------------

def _safe_preview(value, limit=300):
    try:
        text = repr(value)
    except Exception:
        return "<unserializable>"

    if len(text) > limit:
        return text[:limit] + "...[truncated]"
    return text