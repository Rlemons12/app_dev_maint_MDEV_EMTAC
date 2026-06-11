from __future__ import annotations

from typing import Any, Dict, List, Optional, Union
from uuid import UUID

from flask import Blueprint, jsonify, render_template, request, url_for
from sqlalchemy import func

from modules.configuration.config_env import get_db_config
from modules.observability.models import TraceSession, TraceSpan

db = get_db_config()

# IMPORTANT:
# This blueprint name matches the endpoint name seen in your logs:
# "trace_dashboard.trace_dashboard"
trace_dashboard_bp = Blueprint("trace_dashboard", __name__)


# ---------------------------------------------------------
# Helpers
# ---------------------------------------------------------

def _safe_limit(
    raw_value: Optional[str],
    default: int = 50,
    maximum: int = 200,
) -> int:
    try:
        value = int(raw_value or default)
    except (TypeError, ValueError):
        return default

    if value < 1:
        return 1
    if value > maximum:
        return maximum
    return value


def _parse_trace_id(trace_id: str) -> Union[UUID, str]:
    """
    Normalize trace_id if it is a UUID string.
    Falls back to raw string for compatibility if your DB column is not UUID.
    """
    try:
        return UUID(trace_id)
    except (TypeError, ValueError):
        return trace_id


def _iso(dt: Any) -> Optional[str]:
    return dt.isoformat() if dt else None


def _get_template_urls() -> Dict[str, str]:
    """
    Build template API URLs using url_for.
    If url_for fails for any reason, provide safe fallbacks so the template
    still renders instead of crashing.
    """
    try:
        trace_recent_api_url = url_for("trace_dashboard.trace_recent")
    except Exception:
        trace_recent_api_url = "/dashboards/trace/api/recent"

    try:
        trace_graph_api_base = url_for(
            "trace_dashboard.trace_graph",
            trace_id="__TRACE_ID__",
        )
    except Exception:
        trace_graph_api_base = "/dashboards/trace/api/graph/__TRACE_ID__"

    return {
        "trace_recent_api_url": trace_recent_api_url,
        "trace_graph_api_base": trace_graph_api_base,
    }


# ---------------------------------------------------------
# HTML UI
# ---------------------------------------------------------

@trace_dashboard_bp.get("/trace")
def trace_dashboard():
    template_urls = _get_template_urls()

    return render_template(
        "dashboards/trace_dashboard.html",
        trace_recent_api_url=template_urls["trace_recent_api_url"],
        trace_graph_api_base=template_urls["trace_graph_api_base"],
    )


# ---------------------------------------------------------
# Recent Traces (from Postgres)
# ---------------------------------------------------------

@trace_dashboard_bp.get("/trace/api/recent")
def trace_recent():
    limit = _safe_limit(request.args.get("limit"), default=50, maximum=200)

    with db.main_session() as session:
        traces: List[TraceSession] = (
            session.query(TraceSession)
            .order_by(TraceSession.started_at.desc())
            .limit(limit)
            .all()
        )

        trace_ids = [trace.id for trace in traces]

        span_counts_map: Dict[Any, int] = {}
        ok_counts_map: Dict[Any, int] = {}
        error_counts_map: Dict[Any, int] = {}
        duration_map: Dict[Any, Optional[float]] = {}

        if trace_ids:
            span_count_rows = (
                session.query(
                    TraceSpan.trace_id,
                    func.count(TraceSpan.id).label("span_count"),
                )
                .filter(TraceSpan.trace_id.in_(trace_ids))
                .group_by(TraceSpan.trace_id)
                .all()
            )
            span_counts_map = {
                row.trace_id: int(row.span_count)
                for row in span_count_rows
            }

            status_rows = (
                session.query(
                    TraceSpan.trace_id,
                    TraceSpan.status,
                    func.count(TraceSpan.id).label("status_count"),
                )
                .filter(TraceSpan.trace_id.in_(trace_ids))
                .group_by(TraceSpan.trace_id, TraceSpan.status)
                .all()
            )

            for row in status_rows:
                status_value = (row.status or "").lower()
                if status_value == "error":
                    error_counts_map[row.trace_id] = int(row.status_count)
                else:
                    ok_counts_map[row.trace_id] = (
                        ok_counts_map.get(row.trace_id, 0) + int(row.status_count)
                    )

            duration_rows = (
                session.query(
                    TraceSpan.trace_id,
                    func.max(TraceSpan.duration_ms).label("max_duration_ms"),
                )
                .filter(TraceSpan.trace_id.in_(trace_ids))
                .filter(TraceSpan.parent_span_id.is_(None))
                .group_by(TraceSpan.trace_id)
                .all()
            )
            duration_map = {
                row.trace_id: (
                    float(row.max_duration_ms)
                    if row.max_duration_ms is not None
                    else None
                )
                for row in duration_rows
            }

        results: List[Dict[str, Any]] = []

        for trace in traces:
            trace_id_value = trace.id
            span_count = span_counts_map.get(trace_id_value, 0)
            error_count = error_counts_map.get(trace_id_value, 0)
            ok_count = ok_counts_map.get(
                trace_id_value,
                max(span_count - error_count, 0),
            )
            total_duration_ms = duration_map.get(trace_id_value)

            results.append(
                {
                    "trace_id": str(trace.id),
                    "last_seen": _iso(trace.started_at),
                    "span_count": span_count,
                    "ok_count": ok_count,
                    "error_count": error_count,
                    "total_duration_ms": total_duration_ms,
                }
            )

    return jsonify({"recent": results})


# ---------------------------------------------------------
# Graph (tree nodes from Postgres)
# ---------------------------------------------------------

@trace_dashboard_bp.get("/trace/api/graph/<trace_id>")
def trace_graph(trace_id: str):
    normalized_trace_id = _parse_trace_id(trace_id)

    with db.main_session() as session:
        spans: List[TraceSpan] = (
            session.query(TraceSpan)
            .filter(TraceSpan.trace_id == normalized_trace_id)
            .order_by(
                TraceSpan.started_at.asc(),
                TraceSpan.depth.asc(),
                TraceSpan.id.asc(),
            )
            .all()
        )

        if not spans:
            return jsonify(
                {
                    "trace_id": trace_id,
                    "nodes": [],
                    "summary": {
                        "span_count": 0,
                        "ok_count": 0,
                        "error_count": 0,
                        "total_duration_ms": 0,
                        "started_at": None,
                        "ended_at": None,
                    },
                }
            )

        root_start = None
        latest_end = None

        for span in spans:
            if span.started_at is not None:
                if root_start is None or span.started_at < root_start:
                    root_start = span.started_at

            if span.ended_at is not None:
                if latest_end is None or span.ended_at > latest_end:
                    latest_end = span.ended_at

        nodes: List[Dict[str, Any]] = []
        ok_count = 0
        error_count = 0

        for span in spans:
            status_value = (span.status or "ok").lower()

            if status_value == "error":
                error_count += 1
            else:
                ok_count += 1

            relative_start_ms: Optional[float] = None
            relative_end_ms: Optional[float] = None

            if root_start is not None and span.started_at is not None:
                relative_start_ms = (
                    (span.started_at - root_start).total_seconds() * 1000.0
                )

            if root_start is not None and span.ended_at is not None:
                relative_end_ms = (
                    (span.ended_at - root_start).total_seconds() * 1000.0
                )

            duration_ms: Optional[float] = (
                float(span.duration_ms)
                if span.duration_ms is not None
                else None
            )

            if (
                duration_ms is None
                and relative_start_ms is not None
                and relative_end_ms is not None
            ):
                duration_ms = max(relative_end_ms - relative_start_ms, 0.0)

            nodes.append(
                {
                    "id": str(span.id),
                    "trace_id": str(span.trace_id),
                    "parent": str(span.parent_span_id) if span.parent_span_id else None,
                    "function": span.name,
                    "depth": span.depth or 0,
                    "duration_ms": duration_ms,
                    "status": status_value,
                    "request_id": span.request_id,
                    "exception": None,  # Add later if persisted in DB
                    "started_at": _iso(span.started_at),
                    "ended_at": _iso(span.ended_at),
                    "relative_start_ms": relative_start_ms,
                    "relative_end_ms": relative_end_ms,
                }
            )

        total_duration_ms = 0.0

        if root_start is not None and latest_end is not None:
            total_duration_ms = max(
                (latest_end - root_start).total_seconds() * 1000.0,
                0.0,
            )
        else:
            root_spans = [
                node
                for node in nodes
                if node["parent"] is None and node["duration_ms"] is not None
            ]
            if root_spans:
                total_duration_ms = max(
                    float(node["duration_ms"])
                    for node in root_spans
                )

    return jsonify(
        {
            "trace_id": trace_id,
            "nodes": nodes,
            "summary": {
                "span_count": len(nodes),
                "ok_count": ok_count,
                "error_count": error_count,
                "total_duration_ms": total_duration_ms,
                "started_at": _iso(root_start),
                "ended_at": _iso(latest_end),
            },
        }
    )

# ==========================================================
# API: Span Events
# ==========================================================

@trace_dashboard_bp.get("/trace/api/span/<span_id>/events")
def trace_span_events(span_id: str):
    """
    Return TraceEvent rows for a selected span.

    Full URL when blueprint is registered with url_prefix="/dashboards":

        GET /dashboards/trace/api/span/<span_id>/events

    Used by the dashboard Selected Span panel so clicking a span can show:
        - chat_intent_decision
        - rag_run_input
        - rag_chunks_retrieved_raw
        - rag_chunks_selected_final
        - rag_run_result
    """

    try:
        span_uuid = uuid.UUID(str(span_id))
    except Exception:
        return jsonify({
            "status": "invalid_span_id",
            "message": "Invalid span_id. Expected UUID.",
            "span_id": span_id,
            "events": [],
        }), 400

    with db.main_session() as session:
        span = session.get(TraceSpan, span_uuid)

        if span is None:
            return jsonify({
                "status": "not_found",
                "message": "Span not found.",
                "span_id": str(span_uuid),
                "events": [],
            }), 404

        events = (
            session.query(TraceEvent)
            .filter(TraceEvent.span_id == span_uuid)
            .order_by(TraceEvent.created_at.asc(), TraceEvent.id.asc())
            .all()
        )

        results = []

        for event in events:
            results.append({
                "id": str(event.id),
                "span_id": str(event.span_id),
                "event_type": event.event_type,
                "payload": event.payload if isinstance(event.payload, dict) else event.payload,
                "created_at": _iso(event.created_at),
            })

    debug_id(
        f"[trace_dashboard] Span events loaded span_id={span_id} events={len(results)}",
        None,
    )

    return jsonify({
        "status": "success",
        "span_id": str(span_uuid),
        "trace_id": str(span.trace_id),
        "span_name": span.name,
        "event_count": len(results),
        "events": results,
    })