from __future__ import annotations

from typing import List
from collections import defaultdict
import uuid

from flask import Blueprint, jsonify, request, render_template
from sqlalchemy import func,select

from modules.configuration.config_env import get_db_config
from modules.configuration.log_config import debug_id
from modules.observability.models import TraceSession, TraceSpan


trace_dashboard_bp = Blueprint(
    "trace_dashboard",
    __name__,
)

db = get_db_config()


# ==========================================================
# Dashboard View (HTML Page)
# ==========================================================

@trace_dashboard_bp.get("/trace")
def trace_dashboard():
    return render_template("dashboards/trace_dashboard.html")


# ==========================================================
# API: Trace Graph (Advanced)
# ==========================================================

@trace_dashboard_bp.get("/trace/api/graph/<trace_id>")
def trace_graph(trace_id: str):

    try:
        trace_uuid = uuid.UUID(trace_id)
    except Exception:
        return jsonify({"error": "Invalid trace_id"}), 400

    with db.main_session() as session:

        stmt = select(
            TraceSpan.id,
            TraceSpan.trace_id,
            TraceSpan.parent_span_id,
            TraceSpan.name,
            TraceSpan.qualified_name,
            TraceSpan.module_name,
            TraceSpan.file_path,
            TraceSpan.line_number,
            TraceSpan.depth,
            TraceSpan.duration_ms,
            TraceSpan.status,
            TraceSpan.request_id,
            TraceSpan.thread_id,
            TraceSpan.metadata_json,
            TraceSpan.started_at,
            TraceSpan.ended_at,
        ).where(
            TraceSpan.trace_id == trace_uuid
        ).order_by(
            TraceSpan.started_at.asc()
        )

        rows = session.execute(stmt).all()

    if not rows:
        return jsonify({"nodes": []})

    # ------------------------------------------------
    # Convert rows to simple dict structure
    # ------------------------------------------------

    spans = []
    for r in rows:
        spans.append({
            "id": str(r.id),
            "trace_id": str(r.trace_id),
            "parent": str(r.parent_span_id) if r.parent_span_id else None,
            "function": r.name,
            "qualified_name": r.qualified_name,
            "module": r.module_name,
            "file": r.file_path,
            "line": r.line_number,
            "depth": r.depth,
            "duration_ms": float(r.duration_ms or 0),
            "status": r.status,
            "request_id": r.request_id,
            "thread_id": r.thread_id,
            "metadata": r.metadata_json,
            "started_at": r.started_at.isoformat() if r.started_at else None,
            "ended_at": r.ended_at.isoformat() if r.ended_at else None,
        })

    # ------------------------------------------------
    # Build children map
    # ------------------------------------------------

    span_by_id = {s["id"]: s for s in spans}
    children_map = defaultdict(list)

    for s in spans:
        if s["parent"]:
            children_map[s["parent"]].append(s["id"])

    # ------------------------------------------------
    # Compute self-time
    # ------------------------------------------------

    for s in spans:
        total = s["duration_ms"]
        children = children_map.get(s["id"], [])
        child_time = sum(
            span_by_id[c]["duration_ms"]
            for c in children
        )
        s["self_time_ms"] = max(total - child_time, 0)
        s["child_count"] = len(children)

    debug_id(
        f"[trace_dashboard] Graph loaded trace_id={trace_id} nodes={len(spans)}",
        None,
    )

    return jsonify({"nodes": spans})


# ==========================================================
# API: Recent Traces
# ==========================================================

@trace_dashboard_bp.get("/trace/api/recent")
def trace_recent():

    limit = int(request.args.get("limit", "50"))

    with db.main_session() as session:

        traces: List[TraceSession] = (
            session.query(TraceSession)
            .order_by(TraceSession.started_at.desc())
            .limit(limit)
            .all()
        )

        trace_ids = [t.id for t in traces]

        span_counts = (
            session.query(
                TraceSpan.trace_id,
                func.count(TraceSpan.id)
            )
            .filter(TraceSpan.trace_id.in_(trace_ids))
            .group_by(TraceSpan.trace_id)
            .all()
        )

        span_count_map = {str(tid): count for tid, count in span_counts}

        root_spans = (
            session.query(TraceSpan)
            .filter(
                TraceSpan.trace_id.in_(trace_ids),
                TraceSpan.parent_span_id == None
            )
            .all()
        )

        root_map = {str(s.trace_id): s.name for s in root_spans}

        results = []

        for t in traces:
            results.append({
                "trace_id": str(t.id),
                "request_id": t.request_id,
                "root_function": root_map.get(str(t.id)),
                "started_at": t.started_at.isoformat() if t.started_at else None,
                "duration_ms": float(t.duration_ms or 0),
                "status": t.status,
                "span_count": span_count_map.get(str(t.id), 0),
            })

    return jsonify({"recent": results})