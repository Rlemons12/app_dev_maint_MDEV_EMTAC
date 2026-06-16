from __future__ import annotations

import functools
from typing import Any, Callable

from modules.ai.search_pathway.audit.search_audit_context import SearchAuditContext
from modules.ai.search_pathway.audit.search_audit_logger import (
    get_search_audit_log_manager,
)


def _find_audit_context(args: tuple[Any, ...], kwargs: dict[str, Any]) -> SearchAuditContext | None:
    """
    Locate a SearchAuditContext from:
        - keyword argument: audit_context
        - first positional object having .audit_context
        - any positional argument that is itself a SearchAuditContext
    """

    audit_context = kwargs.get("audit_context")

    if isinstance(audit_context, SearchAuditContext):
        return audit_context

    for arg in args:
        if isinstance(arg, SearchAuditContext):
            return arg

        possible_context = getattr(arg, "audit_context", None)
        if isinstance(possible_context, SearchAuditContext):
            return possible_context

    return None


def _safe_snapshot(value: Any, max_text_length: int = 500) -> Any:
    """
    Build a safe, small snapshot for logging/auditing.

    This prevents accidentally storing huge payloads, document text,
    embeddings, or large binary-like objects in stage snapshots.
    """

    if value is None:
        return None

    if isinstance(value, (str, int, float, bool)):
        if isinstance(value, str) and len(value) > max_text_length:
            return value[:max_text_length] + "...[truncated]"
        return value

    if isinstance(value, dict):
        snapshot = {}

        for key, item in value.items():
            key_text = str(key).lower()

            if "embedding" in key_text:
                snapshot[key] = "[embedding omitted]"
                continue

            if "content" in key_text or "text" in key_text:
                if isinstance(item, str):
                    snapshot[key] = _safe_snapshot(item, max_text_length=max_text_length)
                else:
                    snapshot[key] = f"[{type(item).__name__} omitted]"
                continue

            snapshot[key] = _safe_snapshot(item, max_text_length=max_text_length)

        return snapshot

    if isinstance(value, list):
        return {
            "type": "list",
            "count": len(value),
            "sample": [_safe_snapshot(item, max_text_length=max_text_length) for item in value[:3]],
        }

    return {
        "type": type(value).__name__,
        "repr": _safe_snapshot(repr(value), max_text_length=max_text_length),
    }


def audit_stage(
    stage_name: str,
    *,
    capture_input: bool = False,
    capture_output: bool = True,
    output_summarizer: Callable[[Any], dict[str, Any]] | None = None,
):
    """
    Decorator for capturing one search pathway stage.

    This decorator:
        - does not create sessions
        - does not commit
        - does not rollback
        - does not write directly to PostgreSQL

    It only records timing and small snapshots into SearchAuditContext.
    The orchestrator should later pass the context to SearchAuditService.
    """

    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            audit_context = _find_audit_context(args, kwargs)

            if not audit_context:
                return func(*args, **kwargs)

            audit_log_manager = get_search_audit_log_manager()

            input_snapshot = None
            if capture_input:
                input_snapshot = {
                    "args_count": len(args),
                    "kwargs_keys": sorted(kwargs.keys()),
                }

                if "question" in kwargs:
                    input_snapshot["question"] = _safe_snapshot(kwargs.get("question"))

            stage_record = audit_context.stage_tracker.start_stage(
                stage_name=stage_name,
                input_snapshot=input_snapshot,
            )

            audit_log_manager.log_stage_start(
                request_id=audit_context.request_id,
                pathway_name=audit_context.pathway_name,
                stage_name=stage_name,
            )

            try:
                result = func(*args, **kwargs)

                if output_summarizer:
                    output_snapshot = output_summarizer(result)
                elif capture_output:
                    output_snapshot = _safe_snapshot(result)
                else:
                    output_snapshot = None

                audit_context.stage_tracker.complete_stage(
                    stage_record,
                    output_snapshot=output_snapshot,
                )

                audit_log_manager.log_stage_success(
                    request_id=audit_context.request_id,
                    pathway_name=audit_context.pathway_name,
                    stage_name=stage_name,
                    duration_ms=stage_record.duration_ms,
                    output_count=(
                        output_snapshot.get("count")
                        if isinstance(output_snapshot, dict)
                        else None
                    ),
                )

                return result

            except Exception as exc:
                audit_context.stage_tracker.fail_stage(stage_record, exc)

                audit_log_manager.log_stage_failure(
                    request_id=audit_context.request_id,
                    pathway_name=audit_context.pathway_name,
                    stage_name=stage_name,
                    error=exc,
                    duration_ms=stage_record.duration_ms,
                )

                raise

        return wrapper

    return decorator