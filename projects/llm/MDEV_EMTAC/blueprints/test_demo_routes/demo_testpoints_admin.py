from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

from flask import Blueprint, jsonify, request, session


demo_testpoints_admin_bp = Blueprint(
    "demo_testpoints_admin",
    __name__,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_FILE = PROJECT_ROOT / "data" / "demo_testpoints.json"


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _is_admin() -> bool:
    return session.get("user_level") == "ADMIN"


def _admin_required_response():
    return jsonify(
        {
            "ok": False,
            "error": "Admin access required.",
        }
    ), 403


def _ensure_data_file() -> None:
    DATA_FILE.parent.mkdir(parents=True, exist_ok=True)

    if DATA_FILE.exists():
        return

    seed_data = [
        {
            "id": str(uuid.uuid4()),
            "name": "Document Search Display Demo",
            "description": "Test document search display, dropdown tiles, and result clarity.",
            "route_path": "/test-demos/document-search-display",
            "category": "Document Search",
            "enabled": True,
            "tablet_visible": True,
            "requires_admin": False,
            "sort_order": 10,
            "created_at": _utc_now_iso(),
            "updated_at": _utc_now_iso(),
        }
    ]

    DATA_FILE.write_text(
        json.dumps(seed_data, indent=2),
        encoding="utf-8",
    )


def _load_testpoints() -> List[Dict[str, Any]]:
    _ensure_data_file()

    try:
        return json.loads(DATA_FILE.read_text(encoding="utf-8"))
    except Exception:
        return []


def _save_testpoints(testpoints: List[Dict[str, Any]]) -> None:
    DATA_FILE.parent.mkdir(parents=True, exist_ok=True)
    DATA_FILE.write_text(
        json.dumps(testpoints, indent=2),
        encoding="utf-8",
    )


def _safe_text(value: Any, max_len: int = 500) -> str:
    text = str(value or "").strip()

    if len(text) > max_len:
        return text[:max_len]

    return text


def _safe_bool(value: Any, default: bool = False) -> bool:
    if value is None:
        return default

    return bool(value)


def _safe_int(value: Any, default: int = 100) -> int:
    try:
        return int(value)
    except Exception:
        return default


@demo_testpoints_admin_bp.get("/admin/demo-testpoints/api")
def list_demo_testpoints():
    if not _is_admin():
        return _admin_required_response()

    testpoints = _load_testpoints()
    testpoints.sort(key=lambda item: int(item.get("sort_order", 100)))

    return jsonify(
        {
            "ok": True,
            "testpoints": testpoints,
        }
    )


@demo_testpoints_admin_bp.post("/admin/demo-testpoints/api")
def create_demo_testpoint():
    if not _is_admin():
        return _admin_required_response()

    payload = request.get_json(silent=True) or {}

    name = _safe_text(payload.get("name"), 160)
    route_path = _safe_text(payload.get("route_path"), 500)

    if not name:
        return jsonify({"ok": False, "error": "Name is required."}), 400

    if not route_path:
        return jsonify({"ok": False, "error": "Route path is required."}), 400

    if not route_path.startswith("/"):
        return jsonify({"ok": False, "error": "Route path must start with /."}), 400

    testpoints = _load_testpoints()

    record = {
        "id": str(uuid.uuid4()),
        "name": name,
        "description": _safe_text(payload.get("description"), 500),
        "route_path": route_path,
        "category": _safe_text(payload.get("category"), 160) or "General",
        "enabled": _safe_bool(payload.get("enabled"), True),
        "tablet_visible": _safe_bool(payload.get("tablet_visible"), True),
        "requires_admin": _safe_bool(payload.get("requires_admin"), False),
        "sort_order": _safe_int(payload.get("sort_order"), 100),
        "created_at": _utc_now_iso(),
        "updated_at": _utc_now_iso(),
    }

    testpoints.append(record)
    testpoints.sort(key=lambda item: int(item.get("sort_order", 100)))

    _save_testpoints(testpoints)

    return jsonify(
        {
            "ok": True,
            "testpoint": record,
        }
    )


@demo_testpoints_admin_bp.post("/admin/demo-testpoints/api/<testpoint_id>/toggle")
def toggle_demo_testpoint(testpoint_id: str):
    if not _is_admin():
        return _admin_required_response()

    testpoints = _load_testpoints()

    for item in testpoints:
        if str(item.get("id")) == str(testpoint_id):
            item["enabled"] = not bool(item.get("enabled", False))
            item["updated_at"] = _utc_now_iso()

            _save_testpoints(testpoints)

            return jsonify(
                {
                    "ok": True,
                    "testpoint": item,
                }
            )

    return jsonify({"ok": False, "error": "Testpoint not found."}), 404


@demo_testpoints_admin_bp.delete("/admin/demo-testpoints/api/<testpoint_id>")
def delete_demo_testpoint(testpoint_id: str):
    if not _is_admin():
        return _admin_required_response()

    testpoints = _load_testpoints()
    remaining = [
        item for item in testpoints
        if str(item.get("id")) != str(testpoint_id)
    ]

    if len(remaining) == len(testpoints):
        return jsonify({"ok": False, "error": "Testpoint not found."}), 404

    _save_testpoints(remaining)

    return jsonify(
        {
            "ok": True,
            "deleted_id": testpoint_id,
        }
    )


@demo_testpoints_admin_bp.get("/test-demos/api/testpoints")
def list_tablet_visible_demo_testpoints():
    testpoints = _load_testpoints()

    visible = [
        item for item in testpoints
        if bool(item.get("enabled")) and bool(item.get("tablet_visible"))
    ]

    visible.sort(key=lambda item: int(item.get("sort_order", 100)))

    return jsonify(
        {
            "ok": True,
            "testpoints": visible,
        }
    )