from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict

from flask import Blueprint, jsonify, render_template, request


document_search_display_demo_bp = Blueprint(
    "document_search_display_demo_bp",
    __name__,
)


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _safe_text(value: Any, max_len: int = 2000) -> str:
    if value is None:
        return ""

    text = str(value).strip()

    if len(text) > max_len:
        return text[:max_len] + "...[truncated]"

    return text


def _feedback_file() -> Path:
    path = Path("logs") / "demo_feedback" / "document_search_display_demo_feedback.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


@document_search_display_demo_bp.get("/document-search-display")
def document_search_display_demo():
    return render_template("test_demos/document_search_display_demo.html")


@document_search_display_demo_bp.post("/document-search-display/api/feedback")
def document_search_display_demo_feedback():
    payload: Dict[str, Any] = request.get_json(silent=True) or {}

    record = {
        "feedback_id": str(uuid.uuid4()),
        "created_at": _utc_now_iso(),
        "demo": "document_search_display",
        "search_easy": _safe_text(payload.get("search_easy"), 100),
        "results_clear": _safe_text(payload.get("results_clear"), 100),
        "display_easy": _safe_text(payload.get("display_easy"), 100),
        "user_role": _safe_text(payload.get("user_role"), 200),
        "what_worked": _safe_text(payload.get("what_worked"), 2000),
        "what_was_confusing": _safe_text(payload.get("what_was_confusing"), 2000),
        "suggested_changes": _safe_text(payload.get("suggested_changes"), 2000),
        "client": {
            "remote_addr": request.remote_addr,
            "user_agent": _safe_text(request.headers.get("User-Agent"), 500),
        },
    }

    feedback_file = _feedback_file()

    with feedback_file.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")

    return jsonify(
        {
            "ok": True,
            "message": "Feedback saved.",
            "feedback_id": record["feedback_id"],
        }
    )