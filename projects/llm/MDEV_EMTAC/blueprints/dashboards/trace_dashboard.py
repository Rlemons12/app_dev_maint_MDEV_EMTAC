from __future__ import annotations

from typing import Any, Dict, List, Optional
from collections import defaultdict
import uuid

from flask import Blueprint, jsonify, request, render_template, url_for
from sqlalchemy import func, select

from modules.configuration.config_env import get_db_config
from modules.configuration.log_config import debug_id
from modules.observability.models import TraceSession, TraceSpan
from modules.observability.config_trace import (
    get_dashboard_trace_settings_payload,
    set_runtime_trace_settings,
    clear_runtime_trace_setting,
    clear_all_runtime_trace_settings,
    export_env_template,
)


trace_dashboard_bp = Blueprint(
    "trace_dashboard",
    __name__,
)

db = get_db_config()


# ==========================================================
# Helpers
# ==========================================================

def _safe_limit(
    raw_value: Optional[str],
    *,
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


def _iso(value: Any) -> Optional[str]:
    return value.isoformat() if value else None


def _json_data() -> Dict[str, Any]:
    data = request.get_json(silent=True) or {}

    if not isinstance(data, dict):
        return {}

    return data


def _is_failed_status(status_value: Optional[str]) -> bool:
    return str(status_value or "").strip().lower() in {
        "error",
        "failed",
        "exception",
    }


def _is_ok_status(status_value: Optional[str]) -> bool:
    return str(status_value or "").strip().lower() in {
        "ok",
        "success",
        "completed",
        "complete",
    }


def _ui_status(status_value: Optional[str]) -> str:
    """
    Normalize backend statuses into dashboard buckets.

    UI buckets:
        ok
        error
        unknown
    """

    if _is_failed_status(status_value):
        return "error"

    if _is_ok_status(status_value):
        return "ok"

    return "unknown"


def _get_template_urls() -> Dict[str, str]:
    """
    Build dashboard API URLs.

    The fallback values keep the template from crashing if url_for has an issue.
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

    try:
        trace_settings_api_url = url_for("trace_dashboard.trace_settings")
    except Exception:
        trace_settings_api_url = "/dashboards/trace/api/settings"

    try:
        trace_settings_update_api_url = url_for(
            "trace_dashboard.trace_settings_update"
        )
    except Exception:
        trace_settings_update_api_url = "/dashboards/trace/api/settings/update"

    try:
        trace_settings_clear_api_url = url_for(
            "trace_dashboard.trace_settings_clear"
        )
    except Exception:
        trace_settings_clear_api_url = "/dashboards/trace/api/settings/clear"

    try:
        trace_env_template_api_url = url_for(
            "trace_dashboard.trace_env_template"
        )
    except Exception:
        trace_env_template_api_url = "/dashboards/trace/api/settings/env-template"

    return {
        "trace_recent_api_url": trace_recent_api_url,
        "trace_graph_api_base": trace_graph_api_base,
        "trace_settings_api_url": trace_settings_api_url,
        "trace_settings_update_api_url": trace_settings_update_api_url,
        "trace_settings_clear_api_url": trace_settings_clear_api_url,
        "trace_env_template_api_url": trace_env_template_api_url,
    }


# ==========================================================
# Dashboard View HTML Page
# ==========================================================

@trace_dashboard_bp.get("/trace")
def trace_dashboard():
    template_urls = _get_template_urls()

    return render_template(
        "dashboards/trace_dashboard.html",
        trace_recent_api_url=template_urls["trace_recent_api_url"],
        trace_graph_api_base=template_urls["trace_graph_api_base"],
        trace_settings_api_url=template_urls["trace_settings_api_url"],
        trace_settings_update_api_url=template_urls["trace_settings_update_api_url"],
        trace_settings_clear_api_url=template_urls["trace_settings_clear_api_url"],
        trace_env_template_api_url=template_urls["trace_env_template_api_url"],
    )


# ==========================================================
# API: Runtime Trace Settings
# ==========================================================

@trace_dashboard_bp.get("/trace/api/settings")
def trace_settings():
    """
    Return current trace settings.

    Resolution order:
        1. Runtime dashboard override
        2. config_trace.py defaults

    .env is intentionally not used for trace control.
    """

    return jsonify(get_dashboard_trace_settings_payload())


@trace_dashboard_bp.post("/trace/api/settings/update")
def trace_settings_update():
    """
    Update runtime trace settings.

    Accepted payload:

        {
            "settings": {
                "EMTAC_TRACE_ENABLED": true,
                "EMTAC_TRACE_CHAT_ENABLED": true,
                "EMTAC_TRACE_PAYLOAD_ENABLED": false
            },
            "updated_by": "trace_dashboard"
        }

    Also accepts direct key/value payload:

        {
            "EMTAC_TRACE_ENABLED": true,
            "EMTAC_TRACE_CHAT_ENABLED": false
        }
    """

    data = _json_data()

    settings = data.get("settings")

    if settings is None:
        reserved_keys = {
            "updated_by",
            "updatedBy",
            "request_id",
            "requestId",
        }

        settings = {
            key: value
            for key, value in data.items()
            if key not in reserved_keys
        }

    if not isinstance(settings, dict):
        return jsonify({
            "status": "invalid_input",
            "message": "Expected JSON object with a settings dictionary.",
        }), 400

    updated_by = (
        data.get("updated_by")
        or data.get("updatedBy")
        or request.headers.get("X-User-Id")
        or request.headers.get("X-Employee-Id")
        or "trace_dashboard"
    )

    request_id = (
        data.get("request_id")
        or data.get("requestId")
        or request.headers.get("X-Request-Id")
    )

    result = set_runtime_trace_settings(
        settings,
        updated_by=str(updated_by),
        request_id=str(request_id) if request_id else None,
    )

    status_code = (
        200
        if result.get("status") in {"success", "partial_success"}
        else 400
    )

    return jsonify(result), status_code


@trace_dashboard_bp.post("/trace/api/settings/clear")
def trace_settings_clear():
    """
    Clear runtime trace overrides.

    Clear one key:

        {"key": "EMTAC_TRACE_CHAT_ENABLED"}

    Clear all runtime overrides:

        {"all": true}

    Clearing makes the setting fall back to config_trace.py defaults.
    """

    data = _json_data()

    clear_all = bool(
        data.get("all")
        or data.get("clear_all")
        or data.get("clearAll")
    )

    if clear_all:
        return jsonify(clear_all_runtime_trace_settings())

    key = data.get("key")

    if not key:
        return jsonify({
            "status": "invalid_input",
            "message": "Expected key or all=true.",
        }), 400

    result = clear_runtime_trace_setting(str(key))
    status_code = 200 if result.get("status") == "success" else 400

    return jsonify(result), status_code


@trace_dashboard_bp.get("/trace/api/settings/env-template")
def trace_env_template():
    """
    Compatibility endpoint for the dashboard button.

    This no longer returns real .env settings. It returns the config_trace.py
    defaults block because trace control now lives in config_trace.py.
    """

    return jsonify({
        "status": "success",
        "env_template": export_env_template(),
        "config_defaults": export_env_template(),
    })


# ==========================================================
# API: Trace Graph Advanced
# ==========================================================

@trace_dashboard_bp.get("/trace/api/graph/<trace_id>")
def trace_graph(trace_id: str):
    try:
        trace_uuid = uuid.UUID(trace_id)
    except Exception:
        return jsonify({"error": "Invalid trace_id"}), 400

    with db.main_session() as session:
        stmt = (
            select(
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
                TraceSpan.exception,
                TraceSpan.request_id,
                TraceSpan.thread_id,
                TraceSpan.process_id,
                TraceSpan.metadata_json,
                TraceSpan.started_at,
                TraceSpan.ended_at,
            )
            .where(TraceSpan.trace_id == trace_uuid)
            .order_by(
                TraceSpan.started_at.asc(),
                TraceSpan.depth.asc(),
                TraceSpan.id.asc(),
            )
        )

        rows = session.execute(stmt).all()

    if not rows:
        return jsonify({
            "trace_id": trace_id,
            "nodes": [],
            "summary": {
                "span_count": 0,
                "ok_count": 0,
                "error_count": 0,
                "unknown_count": 0,
                "root_count": 0,
                "total_duration_ms": 0,
                "started_at": None,
                "ended_at": None,
            },
        })

    # ------------------------------------------------
    # Determine trace timing
    # ------------------------------------------------

    root_start = None
    latest_end = None

    for row in rows:
        if row.started_at is not None:
            if root_start is None or row.started_at < root_start:
                root_start = row.started_at

        if row.ended_at is not None:
            if latest_end is None or row.ended_at > latest_end:
                latest_end = row.ended_at

    # ------------------------------------------------
    # Convert rows to dashboard dict structure
    # ------------------------------------------------

    spans: List[Dict[str, Any]] = []

    ok_count = 0
    error_count = 0
    unknown_count = 0

    for row in rows:
        raw_status = row.status
        ui_status = _ui_status(raw_status)

        if ui_status == "error":
            error_count += 1
        elif ui_status == "ok":
            ok_count += 1
        else:
            unknown_count += 1

        duration_ms = float(row.duration_ms or 0)

        relative_start_ms = None
        relative_end_ms = None

        if root_start is not None and row.started_at is not None:
            relative_start_ms = (
                (row.started_at - root_start).total_seconds() * 1000.0
            )

        if root_start is not None and row.ended_at is not None:
            relative_end_ms = (
                (row.ended_at - root_start).total_seconds() * 1000.0
            )

        if (
            duration_ms <= 0
            and relative_start_ms is not None
            and relative_end_ms is not None
        ):
            duration_ms = max(relative_end_ms - relative_start_ms, 0.0)

        spans.append({
            "id": str(row.id),
            "trace_id": str(row.trace_id),
            "parent": str(row.parent_span_id) if row.parent_span_id else None,

            # Existing dashboard fields
            "function": row.name,
            "qualified_name": row.qualified_name,
            "module": row.module_name,
            "file": row.file_path,
            "line": row.line_number,
            "depth": int(row.depth or 0),
            "duration_ms": duration_ms,
            "status": ui_status,
            "raw_status": raw_status,
            "exception": row.exception,
            "request_id": row.request_id,
            "thread_id": row.thread_id,
            "process_id": row.process_id,
            "metadata": row.metadata_json,
            "started_at": _iso(row.started_at),
            "ended_at": _iso(row.ended_at),

            # New dashboard-compatible aliases
            "module_name": row.module_name,
            "file_path": row.file_path,
            "line_number": row.line_number,
            "metadata_json": row.metadata_json,
            "relative_start_ms": relative_start_ms,
            "relative_end_ms": relative_end_ms,
        })

    # ------------------------------------------------
    # Build children map
    # ------------------------------------------------

    span_by_id = {span["id"]: span for span in spans}
    children_map = defaultdict(list)

    for span in spans:
        if span["parent"]:
            children_map[span["parent"]].append(span["id"])

    # ------------------------------------------------
    # Compute self-time and child count
    # ------------------------------------------------

    for span in spans:
        total = float(span["duration_ms"] or 0)
        children = children_map.get(span["id"], [])

        child_time = sum(
            float(span_by_id[child_id].get("duration_ms") or 0)
            for child_id in children
            if child_id in span_by_id
        )

        span["self_time_ms"] = max(total - child_time, 0)
        span["child_count"] = len(children)

    root_count = len([span for span in spans if not span["parent"]])

    if root_start is not None and latest_end is not None:
        total_duration_ms = max(
            (latest_end - root_start).total_seconds() * 1000.0,
            0.0,
        )
    else:
        root_durations = [
            float(span["duration_ms"] or 0)
            for span in spans
            if not span["parent"]
        ]
        total_duration_ms = max(root_durations) if root_durations else 0.0

    debug_id(
        f"[trace_dashboard] Graph loaded trace_id={trace_id} nodes={len(spans)}",
        None,
    )

    return jsonify({
        "trace_id": trace_id,
        "nodes": spans,
        "summary": {
            "span_count": len(spans),
            "ok_count": ok_count,
            "error_count": error_count,
            "unknown_count": unknown_count,
            "root_count": root_count,
            "total_duration_ms": total_duration_ms,
            "started_at": _iso(root_start),
            "ended_at": _iso(latest_end),
        },
    })


# ==========================================================
# API: Recent Traces
# ==========================================================

@trace_dashboard_bp.get("/trace/api/recent")
def trace_recent():
    limit = _safe_limit(
        request.args.get("limit"),
        default=50,
        maximum=200,
    )

    with db.main_session() as session:
        traces: List[TraceSession] = (
            session.query(TraceSession)
            .order_by(TraceSession.started_at.desc())
            .limit(limit)
            .all()
        )

        trace_ids = [trace.id for trace in traces]

        span_count_map: Dict[str, int] = {}
        ok_count_map: Dict[str, int] = {}
        error_count_map: Dict[str, int] = {}
        unknown_count_map: Dict[str, int] = {}
        root_map: Dict[str, str] = {}
        root_duration_map: Dict[str, float] = {}

        if trace_ids:
            span_counts = (
                session.query(
                    TraceSpan.trace_id,
                    func.count(TraceSpan.id).label("span_count"),
                )
                .filter(TraceSpan.trace_id.in_(trace_ids))
                .group_by(TraceSpan.trace_id)
                .all()
            )

            span_count_map = {
                str(trace_id_value): int(count or 0)
                for trace_id_value, count in span_counts
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

            for trace_id_value, status_value, status_count in status_rows:
                key = str(trace_id_value)
                count = int(status_count or 0)
                ui_status = _ui_status(status_value)

                if ui_status == "error":
                    error_count_map[key] = error_count_map.get(key, 0) + count
                elif ui_status == "ok":
                    ok_count_map[key] = ok_count_map.get(key, 0) + count
                else:
                    unknown_count_map[key] = unknown_count_map.get(key, 0) + count

            root_spans = (
                session.query(TraceSpan)
                .filter(
                    TraceSpan.trace_id.in_(trace_ids),
                    TraceSpan.parent_span_id.is_(None),
                )
                .order_by(TraceSpan.started_at.asc())
                .all()
            )

            for span in root_spans:
                key = str(span.trace_id)

                root_map.setdefault(key, span.name)

                if span.duration_ms is not None:
                    root_duration_map[key] = max(
                        root_duration_map.get(key, 0.0),
                        float(span.duration_ms or 0),
                    )

        results: List[Dict[str, Any]] = []

        for trace in traces:
            trace_key = str(trace.id)

            span_count = int(span_count_map.get(trace_key, 0))
            error_count = int(error_count_map.get(trace_key, 0))
            ok_count = int(ok_count_map.get(trace_key, 0))
            unknown_count = int(unknown_count_map.get(trace_key, 0))

            if span_count and (ok_count + error_count + unknown_count) < span_count:
                ok_count = max(span_count - error_count - unknown_count, 0)

            total_duration_ms = (
                float(trace.duration_ms)
                if trace.duration_ms is not None
                else float(root_duration_map.get(trace_key, 0.0))
            )

            results.append({
                "trace_id": trace_key,
                "request_id": trace.request_id,
                "root_function": root_map.get(trace_key),
                "started_at": _iso(trace.started_at),
                "last_seen": _iso(trace.started_at),
                "ended_at": _iso(trace.ended_at),
                "duration_ms": total_duration_ms,
                "total_duration_ms": total_duration_ms,
                "status": trace.status,
                "span_count": span_count,
                "ok_count": ok_count,
                "error_count": error_count,
                "unknown_count": unknown_count,
                "service_name": trace.service_name,
                "environment": trace.environment,
                "sampled": trace.sampled,
            })

    return jsonify({"recent": results})