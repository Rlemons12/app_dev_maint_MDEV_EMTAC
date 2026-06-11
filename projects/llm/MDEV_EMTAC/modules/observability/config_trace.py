# modules/observability/config_trace.py

from __future__ import annotations

import threading
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from typing import Any, Dict, Optional


# ==========================================================
# Trace Setting Keys
# ==========================================================

MASTER_TRACE_ENV = "EMTAC_TRACE_ENABLED"

TRACE_CHAT_ENV = "EMTAC_TRACE_CHAT_ENABLED"
TRACE_PAYLOAD_ENV = "EMTAC_TRACE_PAYLOAD_ENABLED"
TRACE_FEEDBACK_ENV = "EMTAC_TRACE_FEEDBACK_ENABLED"
TRACE_HEALTH_ENV = "EMTAC_TRACE_HEALTH_ENABLED"

TRACE_DEEP_PROFILE_ENV = "EMTAC_TRACE_DEEP_PROFILE"
TRACE_CAPTURE_ARGS_ENV = "EMTAC_TRACE_CAPTURE_ARGS"
TRACE_CAPTURE_RETURN_ENV = "EMTAC_TRACE_CAPTURE_RETURN"

TRACE_SERVICE_NAME_ENV = "EMTAC_TRACE_SERVICE_NAME"
TRACE_ENVIRONMENT_ENV = "EMTAC_ENV"

ALLOW_TRACE_DASHBOARD_WITHOUT_LOGIN_ENV = "ALLOW_TRACE_DASHBOARD_WITHOUT_LOGIN"


# ==========================================================
# File-Based Defaults
# ==========================================================
# This file is now the source of truth for trace defaults.
#
# Runtime dashboard changes override these values until Flask restarts.
# .env is intentionally NOT used by this module.

DEFAULT_TRACE_VALUES: Dict[str, bool] = {
    MASTER_TRACE_ENV: True,

    TRACE_CHAT_ENV: True,
    TRACE_PAYLOAD_ENV: True,
    TRACE_FEEDBACK_ENV: False,
    TRACE_HEALTH_ENV: False,

    TRACE_DEEP_PROFILE_ENV: False,
    TRACE_CAPTURE_ARGS_ENV: False,
    TRACE_CAPTURE_RETURN_ENV: False,

    ALLOW_TRACE_DASHBOARD_WITHOUT_LOGIN_ENV: True,
}

DEFAULT_TRACE_SERVICE_NAME = "emtac_chat"
DEFAULT_TRACE_ENVIRONMENT = "development"


# ==========================================================
# Runtime Overrides
# ==========================================================
# Dashboard changes update these values.
#
# Important:
#   - Runtime only.
#   - No app restart required.
#   - Resets when Flask restarts.
#   - Falls back to DEFAULT_TRACE_VALUES above.

_runtime_lock = threading.RLock()
_runtime_overrides: Dict[str, bool] = {}
_runtime_metadata: Dict[str, Dict[str, Any]] = {}


# ==========================================================
# Dataclasses
# ==========================================================

@dataclass(frozen=True)
class TraceConfigSnapshot:
    """
    Current resolved trace settings.

    This is useful for:
        - dashboard display
        - debugging
        - decorator decisions
    """

    master_enabled: bool
    chat_enabled: bool
    payload_enabled: bool
    feedback_enabled: bool
    health_enabled: bool

    deep_profile: bool
    capture_args: bool
    capture_return: bool

    allow_dashboard_without_login: bool

    service_name: str
    environment: str

    source: Dict[str, str]
    runtime_overrides: Dict[str, bool]

    generated_at: str


@dataclass(frozen=True)
class TraceBehaviorFlags:
    """
    Resolved behavior flags for one trace entrypoint.
    """

    deep_profile: bool
    capture_args: bool
    capture_return: bool


# ==========================================================
# Basic Helpers
# ==========================================================

def utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def normalize_key(key: Any) -> str:
    return str(key or "").strip().upper()


def is_known_trace_key(key: str) -> bool:
    return normalize_key(key) in DEFAULT_TRACE_VALUES


def get_known_trace_keys() -> list[str]:
    return list(DEFAULT_TRACE_VALUES.keys())


def parse_bool(value: Any, default: bool = False) -> bool:
    """
    Parse common bool-like values safely.

    Truthy:
        true, 1, yes, y, on

    Falsy:
        false, 0, no, n, off
    """

    if value is None:
        return default

    if isinstance(value, bool):
        return value

    if isinstance(value, int):
        return value != 0

    text = str(value).strip().lower()

    if text in {"1", "true", "yes", "y", "on"}:
        return True

    if text in {"0", "false", "no", "n", "off"}:
        return False

    return default


# ==========================================================
# Runtime Override API
# ==========================================================

def set_runtime_trace_setting(
    key: str,
    value: Any,
    *,
    updated_by: Optional[str] = None,
    request_id: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Set one runtime trace setting.

    This is intended for dashboard/API use.

    Returns a small result dict safe for JSON.
    """

    normalized_key = normalize_key(key)

    if not is_known_trace_key(normalized_key):
        return {
            "status": "invalid_key",
            "key": normalized_key,
            "message": f"Unknown trace setting: {normalized_key}",
            "known_keys": get_known_trace_keys(),
        }

    bool_value = parse_bool(value, default=DEFAULT_TRACE_VALUES[normalized_key])

    with _runtime_lock:
        _runtime_overrides[normalized_key] = bool_value
        _runtime_metadata[normalized_key] = {
            "updated_at": utc_iso(),
            "updated_by": updated_by,
            "request_id": request_id,
            "source": "runtime",
        }

    return {
        "status": "success",
        "key": normalized_key,
        "value": bool_value,
        "metadata": get_runtime_trace_metadata(normalized_key),
    }


def set_runtime_trace_settings(
    values: Dict[str, Any],
    *,
    updated_by: Optional[str] = None,
    request_id: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Set multiple runtime trace settings.

    Unknown keys are reported but do not stop valid keys from being applied.
    """

    if not isinstance(values, dict):
        return {
            "status": "invalid_input",
            "message": "values must be a dictionary",
            "updated": {},
            "errors": [],
        }

    updated: Dict[str, bool] = {}
    errors: list[Dict[str, Any]] = []

    for key, value in values.items():
        result = set_runtime_trace_setting(
            key,
            value,
            updated_by=updated_by,
            request_id=request_id,
        )

        if result.get("status") == "success":
            updated[result["key"]] = result["value"]
        else:
            errors.append(result)

    return {
        "status": "success" if not errors else "partial_success",
        "updated": updated,
        "errors": errors,
        "snapshot": get_trace_config_snapshot_dict(),
    }


def clear_runtime_trace_setting(key: str) -> Dict[str, Any]:
    """
    Clear one runtime override.

    After clearing, the setting falls back to config_trace.py defaults.
    """

    normalized_key = normalize_key(key)

    if not is_known_trace_key(normalized_key):
        return {
            "status": "invalid_key",
            "key": normalized_key,
            "message": f"Unknown trace setting: {normalized_key}",
            "known_keys": get_known_trace_keys(),
        }

    with _runtime_lock:
        existed = normalized_key in _runtime_overrides
        _runtime_overrides.pop(normalized_key, None)
        _runtime_metadata.pop(normalized_key, None)

    return {
        "status": "success",
        "key": normalized_key,
        "cleared": existed,
        "resolved_value": get_trace_bool(normalized_key),
        "resolved_source": get_trace_bool_source(normalized_key),
    }


def clear_all_runtime_trace_settings() -> Dict[str, Any]:
    """
    Clear all runtime overrides.
    """

    with _runtime_lock:
        cleared_count = len(_runtime_overrides)
        _runtime_overrides.clear()
        _runtime_metadata.clear()

    return {
        "status": "success",
        "cleared_count": cleared_count,
        "snapshot": get_trace_config_snapshot_dict(),
    }


def get_runtime_trace_overrides() -> Dict[str, bool]:
    with _runtime_lock:
        return dict(_runtime_overrides)


def get_runtime_trace_metadata(key: Optional[str] = None) -> Dict[str, Any]:
    with _runtime_lock:
        if key is None:
            return dict(_runtime_metadata)

        return dict(_runtime_metadata.get(normalize_key(key), {}))


# ==========================================================
# Resolved Setting API
# ==========================================================

def get_trace_bool(key: str, default: Optional[bool] = None) -> bool:
    """
    Resolve one trace bool setting.

    Priority:
        1. Runtime override from dashboard/API
        2. DEFAULT_TRACE_VALUES in this file
        3. Provided default
        4. False
    """

    normalized_key = normalize_key(key)

    fallback = (
        DEFAULT_TRACE_VALUES.get(normalized_key)
        if normalized_key in DEFAULT_TRACE_VALUES
        else default
    )

    if fallback is None:
        fallback = False

    with _runtime_lock:
        if normalized_key in _runtime_overrides:
            return bool(_runtime_overrides[normalized_key])

    return bool(fallback)


def get_trace_bool_source(key: str) -> str:
    """
    Return where a setting came from:
        runtime
        config_trace
        unknown_default
    """

    normalized_key = normalize_key(key)

    with _runtime_lock:
        if normalized_key in _runtime_overrides:
            return "runtime"

    if normalized_key in DEFAULT_TRACE_VALUES:
        return "config_trace"

    return "unknown_default"


def get_trace_text(key: str, default: str) -> str:
    """
    Resolve text settings from config_trace.py only.

    .env is intentionally ignored.
    """

    normalized_key = normalize_key(key)

    if normalized_key == TRACE_SERVICE_NAME_ENV:
        return DEFAULT_TRACE_SERVICE_NAME

    if normalized_key == TRACE_ENVIRONMENT_ENV:
        return DEFAULT_TRACE_ENVIRONMENT

    return default


def is_master_trace_enabled() -> bool:
    return get_trace_bool(MASTER_TRACE_ENV, DEFAULT_TRACE_VALUES[MASTER_TRACE_ENV])


def is_trace_group_enabled(
    key: Optional[str],
    *,
    default: bool = True,
) -> bool:
    """
    Resolve a specific trace group.

    Example:
        is_trace_group_enabled("EMTAC_TRACE_CHAT_ENABLED")
    """

    if not key:
        return default

    return get_trace_bool(key, default=default)


def is_trace_entrypoint_enabled(
    *,
    enabled: Optional[bool] = None,
    enabled_env: Optional[str] = None,
) -> bool:
    """
    Used by @trace_entrypoint.

    Rules:
        - EMTAC_TRACE_ENABLED must be enabled
        - decorator enabled=False disables this entrypoint
        - enabled_env controls the specific trace group
    """

    if not is_master_trace_enabled():
        return False

    if enabled is False:
        return False

    if enabled_env:
        return is_trace_group_enabled(enabled_env, default=True)

    if enabled is True:
        return True

    return True


def resolve_bool_option(
    *,
    explicit_value: bool,
    env_name: Optional[str],
) -> bool:
    """
    Resolve a decorator bool option.

    The name env_name is kept for compatibility with trace_decorator.py,
    but it now resolves against config_trace.py/runtime settings only.
    """

    if env_name:
        return get_trace_bool(env_name, default=explicit_value)

    return bool(explicit_value)


def resolve_behavior_flags(
    *,
    deep_profile: bool = False,
    capture_args: bool = False,
    capture_return: bool = False,
    deep_profile_env: Optional[str] = TRACE_DEEP_PROFILE_ENV,
    capture_args_env: Optional[str] = TRACE_CAPTURE_ARGS_ENV,
    capture_return_env: Optional[str] = TRACE_CAPTURE_RETURN_ENV,
) -> TraceBehaviorFlags:
    """
    Resolve behavior flags for a trace entrypoint.

    This lets the dashboard turn expensive features on/off live.
    """

    return TraceBehaviorFlags(
        deep_profile=resolve_bool_option(
            explicit_value=deep_profile,
            env_name=deep_profile_env,
        ),
        capture_args=resolve_bool_option(
            explicit_value=capture_args,
            env_name=capture_args_env,
        ),
        capture_return=resolve_bool_option(
            explicit_value=capture_return,
            env_name=capture_return_env,
        ),
    )


# ==========================================================
# Snapshot / Dashboard API Helpers
# ==========================================================

def get_trace_config_snapshot() -> TraceConfigSnapshot:
    source = {
        key: get_trace_bool_source(key)
        for key in DEFAULT_TRACE_VALUES
    }

    return TraceConfigSnapshot(
        master_enabled=get_trace_bool(MASTER_TRACE_ENV),
        chat_enabled=get_trace_bool(TRACE_CHAT_ENV),
        payload_enabled=get_trace_bool(TRACE_PAYLOAD_ENV),
        feedback_enabled=get_trace_bool(TRACE_FEEDBACK_ENV),
        health_enabled=get_trace_bool(TRACE_HEALTH_ENV),

        deep_profile=get_trace_bool(TRACE_DEEP_PROFILE_ENV),
        capture_args=get_trace_bool(TRACE_CAPTURE_ARGS_ENV),
        capture_return=get_trace_bool(TRACE_CAPTURE_RETURN_ENV),

        allow_dashboard_without_login=get_trace_bool(
            ALLOW_TRACE_DASHBOARD_WITHOUT_LOGIN_ENV
        ),

        service_name=get_trace_text(
            TRACE_SERVICE_NAME_ENV,
            DEFAULT_TRACE_SERVICE_NAME,
        ),
        environment=get_trace_text(
            TRACE_ENVIRONMENT_ENV,
            DEFAULT_TRACE_ENVIRONMENT,
        ),

        source=source,
        runtime_overrides=get_runtime_trace_overrides(),

        generated_at=utc_iso(),
    )


def get_trace_config_snapshot_dict() -> Dict[str, Any]:
    return asdict(get_trace_config_snapshot())


def get_dashboard_trace_settings_payload() -> Dict[str, Any]:
    """
    JSON-friendly payload for dashboard settings UI.
    """

    snapshot = get_trace_config_snapshot_dict()

    settings: Dict[str, Dict[str, Any]] = {}

    labels = {
        MASTER_TRACE_ENV: "Master tracing",
        TRACE_CHAT_ENV: "Chat answer tracing",
        TRACE_PAYLOAD_ENV: "Chat payload tracing",
        TRACE_FEEDBACK_ENV: "Feedback tracing",
        TRACE_HEALTH_ENV: "Health/metrics tracing",
        TRACE_DEEP_PROFILE_ENV: "Deep profile",
        TRACE_CAPTURE_ARGS_ENV: "Capture arguments",
        TRACE_CAPTURE_RETURN_ENV: "Capture return value",
        ALLOW_TRACE_DASHBOARD_WITHOUT_LOGIN_ENV: "Allow dashboard without login",
    }

    descriptions = {
        MASTER_TRACE_ENV: "Master on/off switch for all trace entrypoints.",
        TRACE_CHAT_ENV: "Trace /chatbot/ask and the answer-first chat pathway.",
        TRACE_PAYLOAD_ENV: "Trace /chatbot/ask/payload and supporting payload projection.",
        TRACE_FEEDBACK_ENV: "Trace feedback/rating routes.",
        TRACE_HEALTH_ENV: "Trace health, metrics, and dashboard routes.",
        TRACE_DEEP_PROFILE_ENV: "Enable deep sys.setprofile function tracing. Can be noisy.",
        TRACE_CAPTURE_ARGS_ENV: "Capture short argument previews on root spans.",
        TRACE_CAPTURE_RETURN_ENV: "Capture short return-value previews on root spans.",
        ALLOW_TRACE_DASHBOARD_WITHOUT_LOGIN_ENV: "Development bypass for trace dashboard access.",
    }

    current_values = {
        MASTER_TRACE_ENV: snapshot["master_enabled"],
        TRACE_CHAT_ENV: snapshot["chat_enabled"],
        TRACE_PAYLOAD_ENV: snapshot["payload_enabled"],
        TRACE_FEEDBACK_ENV: snapshot["feedback_enabled"],
        TRACE_HEALTH_ENV: snapshot["health_enabled"],
        TRACE_DEEP_PROFILE_ENV: snapshot["deep_profile"],
        TRACE_CAPTURE_ARGS_ENV: snapshot["capture_args"],
        TRACE_CAPTURE_RETURN_ENV: snapshot["capture_return"],
        ALLOW_TRACE_DASHBOARD_WITHOUT_LOGIN_ENV: snapshot["allow_dashboard_without_login"],
    }

    for key in DEFAULT_TRACE_VALUES:
        settings[key] = {
            "key": key,
            "label": labels.get(key, key),
            "description": descriptions.get(key, ""),
            "value": bool(current_values.get(key, False)),
            "default": bool(DEFAULT_TRACE_VALUES.get(key, False)),
            "source": get_trace_bool_source(key),
            "metadata": get_runtime_trace_metadata(key),
        }

    return {
        "status": "success",
        "settings": settings,
        "snapshot": snapshot,
        "known_keys": get_known_trace_keys(),
    }


# ==========================================================
# Convenience Helpers for Decorators
# ==========================================================

def get_service_name() -> str:
    return get_trace_text(
        TRACE_SERVICE_NAME_ENV,
        DEFAULT_TRACE_SERVICE_NAME,
    )


def get_environment_name() -> str:
    return get_trace_text(
        TRACE_ENVIRONMENT_ENV,
        DEFAULT_TRACE_ENVIRONMENT,
    )


def get_standard_trace_envs() -> Dict[str, str]:
    """
    Helper so routes can import known setting names instead of hardcoding strings.

    Names are kept as *_ENV for compatibility, but these values are no longer
    read from .env.
    """

    return {
        "master": MASTER_TRACE_ENV,
        "chat": TRACE_CHAT_ENV,
        "payload": TRACE_PAYLOAD_ENV,
        "feedback": TRACE_FEEDBACK_ENV,
        "health": TRACE_HEALTH_ENV,
        "deep_profile": TRACE_DEEP_PROFILE_ENV,
        "capture_args": TRACE_CAPTURE_ARGS_ENV,
        "capture_return": TRACE_CAPTURE_RETURN_ENV,
        "service_name": TRACE_SERVICE_NAME_ENV,
        "environment": TRACE_ENVIRONMENT_ENV,
        "allow_dashboard_without_login": ALLOW_TRACE_DASHBOARD_WITHOUT_LOGIN_ENV,
    }


def export_env_template() -> str:
    """
    Kept for compatibility with the existing dashboard endpoint.

    Since .env is no longer used for trace control, this returns a config_trace.py
    reference block instead of an .env block.
    """

    lines = [
        "# Trace control is now managed by modules/observability/config_trace.py",
        "# Runtime dashboard overrides apply immediately and reset when Flask restarts.",
        "",
        "DEFAULT_TRACE_VALUES = {",
    ]

    for key, value in DEFAULT_TRACE_VALUES.items():
        lines.append(f'    "{key}": {value},')

    lines.extend([
        "}",
        "",
        f'DEFAULT_TRACE_SERVICE_NAME = "{DEFAULT_TRACE_SERVICE_NAME}"',
        f'DEFAULT_TRACE_ENVIRONMENT = "{DEFAULT_TRACE_ENVIRONMENT}"',
    ])

    return "\n".join(lines)


# ==========================================================
# Optional Self-Test
# ==========================================================

def self_test() -> Dict[str, Any]:
    """
    Lightweight sanity check.
    """

    before = get_trace_config_snapshot_dict()

    test_key = TRACE_HEALTH_ENV
    original_override_exists = test_key in get_runtime_trace_overrides()
    original_value = get_runtime_trace_overrides().get(test_key)

    set_result = set_runtime_trace_setting(
        test_key,
        True,
        updated_by="self_test",
        request_id="trace-config-self-test",
    )

    after_set = get_trace_bool(test_key)

    if original_override_exists:
        set_runtime_trace_setting(
            test_key,
            original_value,
            updated_by="self_test_restore",
            request_id="trace-config-self-test",
        )
    else:
        clear_runtime_trace_setting(test_key)

    after_restore = get_trace_config_snapshot_dict()

    return {
        "status": "success",
        "before": before,
        "set_result": set_result,
        "after_set": after_set,
        "after_restore": after_restore,
    }


if __name__ == "__main__":
    import json

    print(json.dumps(self_test(), indent=2, default=str))
    print()
    print(export_env_template())