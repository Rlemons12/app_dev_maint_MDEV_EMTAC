"""
Simple route smoke tests for EMTAC Tablet Edge Agent endpoints.

File:
    modules/database_manager/tablet_edge/test_tablet_edge_routes.py

Run from project root while Flask server is running:

    python -m modules.database_manager.tablet_edge.test_tablet_edge_routes

Recommended PowerShell usage:

    $env:EMTAC_TABLET_EDGE_BASE_URL = "http://172.19.194.129:5000"
    .\.venv\Scripts\python.exe -m modules.database_manager.tablet_edge.test_tablet_edge_routes

Optional environment variables:

    EMTAC_TABLET_EDGE_BASE_URL
        Example:
            http://172.19.194.129:5000
            http://127.0.0.1:5000

    EMTAC_TABLET_EDGE_TEST_UID
        Example:
            00000000-0000-0000-0000-000000000101

Important:
    This script now fails loudly if the server returns an HTML login page
    instead of JSON. That prevents false-success results when the global
    Flask login guard redirects API requests to /login.
"""

from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any
from uuid import uuid4

import requests
from requests import Response, Session


logger = logging.getLogger("tablet_edge_route_tests")


DEFAULT_BASE_URL = "http://127.0.0.1:5000"

BASE_URL = os.getenv(
    "EMTAC_TABLET_EDGE_BASE_URL",
    DEFAULT_BASE_URL,
).rstrip("/")

TABLET_UID = os.getenv(
    "EMTAC_TABLET_EDGE_TEST_UID",
    str(uuid4()),
)

TABLET_NAME = os.getenv(
    "EMTAC_TABLET_EDGE_TEST_NAME",
    "EMTAC-ROUTE-TEST-01",
)

REQUEST_TIMEOUT_SECONDS = int(
    os.getenv("EMTAC_TABLET_EDGE_TIMEOUT_SECONDS", "15")
)


@dataclass
class RouteTestResult:
    """
    Summary record for one route smoke test.
    """

    method: str
    path: str
    status_code: int
    duration_ms: int
    success: bool


def configure_logging() -> None:
    """
    Configure route test logging.
    """
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )


def compact_for_log(body: Any, *, max_collection_items: int = 5) -> Any:
    """
    Reduce large JSON payloads before logging.

    This keeps /dropdown-cache/full from dumping a huge payload to the console.
    """
    if isinstance(body, dict):
        compact: dict[str, Any] = {}

        for key, value in body.items():
            if isinstance(value, list):
                compact[key] = {
                    "_type": "list",
                    "count": len(value),
                    "sample": value[:max_collection_items],
                }
            elif isinstance(value, dict):
                compact[key] = {
                    "_type": "dict",
                    "keys": list(value.keys())[:max_collection_items],
                    "count": len(value),
                }
            else:
                compact[key] = value

        return compact

    if isinstance(body, list):
        return {
            "_type": "list",
            "count": len(body),
            "sample": body[:max_collection_items],
        }

    return body


def get_response_snippet(response: Response, *, max_chars: int = 1200) -> str:
    """
    Return a safe response text snippet for error messages.
    """
    text = response.text or ""
    text = text.strip()

    if len(text) > max_chars:
        return text[:max_chars] + "...[truncated]"

    return text


def response_looks_like_html(response: Response) -> bool:
    """
    Detect whether the server returned HTML instead of API JSON.
    """
    content_type = response.headers.get("Content-Type", "").lower()
    snippet = get_response_snippet(response, max_chars=500).lower()

    if "text/html" in content_type:
        return True

    if snippet.startswith("<!doctype html"):
        return True

    if snippet.startswith("<html"):
        return True

    if "<title>login" in snippet:
        return True

    if '<form action="/login"' in snippet:
        return True

    return False


def parse_json_response(response: Response, method: str, path: str) -> dict[str, Any]:
    """
    Parse and validate a JSON response.

    Raises:
        RuntimeError if the response is HTML, non-JSON, or not a JSON object.
    """
    if response_looks_like_html(response):
        snippet = get_response_snippet(response)

        raise RuntimeError(
            "\n"
            f"Expected JSON but received HTML for {method.upper()} {path}.\n"
            "This usually means Flask redirected the API request to the login page.\n"
            "Check global_login_check() and make sure /tablet-edge/ routes are bypassed.\n"
            f"Status: {response.status_code}\n"
            f"Content-Type: {response.headers.get('Content-Type')}\n"
            f"Response snippet:\n{snippet}"
        )

    try:
        body = response.json()
    except Exception as exc:
        snippet = get_response_snippet(response)

        raise RuntimeError(
            "\n"
            f"Expected JSON but could not parse response for {method.upper()} {path}.\n"
            f"Status: {response.status_code}\n"
            f"Content-Type: {response.headers.get('Content-Type')}\n"
            f"Response snippet:\n{snippet}"
        ) from exc

    if not isinstance(body, dict):
        raise RuntimeError(
            f"Expected JSON object for {method.upper()} {path}, "
            f"but received: {type(body).__name__}"
        )

    return body


def assert_success_body(
    body: dict[str, Any],
    *,
    method: str,
    path: str,
    require_success_true: bool = True,
) -> None:
    """
    Validate the standard Tablet Edge response body.
    """
    if require_success_true and body.get("success") is not True:
        raise RuntimeError(
            "\n"
            f"Route returned success != True for {method.upper()} {path}.\n"
            f"Response body:\n{json.dumps(body, indent=2, default=str)}"
        )


def request_json(
    session: Session,
    method: str,
    path: str,
    payload: dict[str, Any] | None = None,
    *,
    require_success_true: bool = True,
    expected_status_codes: tuple[int, ...] = (200,),
) -> dict[str, Any]:
    """
    Send an HTTP request and require a valid JSON response.

    This intentionally disables automatic redirects so login redirects are
    detected as failures instead of silently following to /login.
    """
    url = f"{BASE_URL}{path}"
    method_upper = method.upper()

    logger.info("%s %s", method_upper, url)

    started = time.perf_counter()

    try:
        response = session.request(
            method=method_upper,
            url=url,
            json=payload,
            timeout=REQUEST_TIMEOUT_SECONDS,
            allow_redirects=False,
        )
    except requests.RequestException as exc:
        raise RuntimeError(
            "\n"
            f"Request failed before receiving a response: {method_upper} {url}\n"
            f"Error: {exc}\n"
            "Check that Flask is running and EMTAC_TABLET_EDGE_BASE_URL is correct."
        ) from exc

    duration_ms = int((time.perf_counter() - started) * 1000)

    logger.info(
        "Status=%s Duration=%sms Content-Type=%s",
        response.status_code,
        duration_ms,
        response.headers.get("Content-Type"),
    )

    if response.is_redirect or response.status_code in {301, 302, 303, 307, 308}:
        raise RuntimeError(
            "\n"
            f"Unexpected redirect for {method_upper} {path}.\n"
            f"Status: {response.status_code}\n"
            f"Location: {response.headers.get('Location')}\n"
            "This usually means the route is blocked by the login guard."
        )

    body = parse_json_response(response, method_upper, path)

    logger.info(
        "Response body:\n%s",
        json.dumps(compact_for_log(body), indent=2, default=str),
    )

    if response.status_code not in expected_status_codes:
        raise RuntimeError(
            "\n"
            f"Unexpected status code for {method_upper} {path}.\n"
            f"Expected: {expected_status_codes}\n"
            f"Actual: {response.status_code}\n"
            f"Response body:\n{json.dumps(body, indent=2, default=str)}"
        )

    assert_success_body(
        body,
        method=method_upper,
        path=path,
        require_success_true=require_success_true,
    )

    return body


def require_key(body: dict[str, Any], key: str, *, route_name: str) -> Any:
    """
    Require a key in a response body.
    """
    if key not in body:
        raise RuntimeError(
            f"Missing expected key '{key}' in response for {route_name}.\n"
            f"Response body:\n{json.dumps(body, indent=2, default=str)}"
        )

    return body[key]


def assert_int_at_least(
    value: Any,
    minimum: int,
    *,
    field_name: str,
    route_name: str,
) -> None:
    """
    Assert a numeric field is at least a minimum value.
    """
    try:
        numeric_value = int(value)
    except Exception as exc:
        raise RuntimeError(
            f"Expected integer field '{field_name}' for {route_name}, got {value!r}"
        ) from exc

    if numeric_value < minimum:
        raise RuntimeError(
            f"Expected '{field_name}' for {route_name} to be >= {minimum}, "
            f"got {numeric_value}"
        )


def test_health(session: Session, results: list[RouteTestResult]) -> None:
    """
    Test GET /tablet-edge/health.
    """
    path = "/tablet-edge/health"
    started = time.perf_counter()

    body = request_json(session, "GET", path)

    duration_ms = int((time.perf_counter() - started) * 1000)

    if body.get("status") != "ok":
        raise RuntimeError(
            f"Unexpected health status: {json.dumps(body, indent=2, default=str)}"
        )

    results.append(RouteTestResult("GET", path, 200, duration_ms, True))


def test_routes(session: Session, results: list[RouteTestResult]) -> None:
    """
    Test GET /tablet-edge/routes.
    """
    path = "/tablet-edge/routes"
    started = time.perf_counter()

    body = request_json(session, "GET", path)

    duration_ms = int((time.perf_counter() - started) * 1000)

    routes = require_key(body, "routes", route_name=path)

    if not isinstance(routes, list) or not routes:
        raise RuntimeError("/tablet-edge/routes did not return a non-empty routes list.")

    results.append(RouteTestResult("GET", path, 200, duration_ms, True))


def test_register(session: Session, results: list[RouteTestResult]) -> None:
    """
    Test POST /tablet-edge/register.
    """
    path = "/tablet-edge/register"

    register_payload = {
        "tablet_uid": TABLET_UID,
        "tablet_name": TABLET_NAME,
        "device_make": "RouteTest",
        "device_model": "PowerShell/Python",
        "android_version": "test",
        "app_version": "0.1.0-test",
        "assigned_area": "Development",
        "assigned_station": "Route Test",
        "assigned_role": "maintenance_tablet",
    }

    started = time.perf_counter()

    body = request_json(session, "POST", path, register_payload)

    duration_ms = int((time.perf_counter() - started) * 1000)

    tablet_device_id = require_key(body, "tablet_device_id", route_name=path)
    assert_int_at_least(
        tablet_device_id,
        1,
        field_name="tablet_device_id",
        route_name=path,
    )

    if str(body.get("tablet_uid")) != str(TABLET_UID):
        raise RuntimeError(
            f"Register response tablet_uid mismatch. "
            f"Expected={TABLET_UID} Actual={body.get('tablet_uid')}"
        )

    results.append(RouteTestResult("POST", path, 200, duration_ms, True))


def test_heartbeat(session: Session, results: list[RouteTestResult]) -> None:
    """
    Test POST /tablet-edge/heartbeat.
    """
    path = "/tablet-edge/heartbeat"

    heartbeat_payload = {
        "tablet_uid": TABLET_UID,
        "app_version": "0.1.0-test",
        "current_page_url": "/assistant",
        "quality_level": "good",
    }

    started = time.perf_counter()

    body = request_json(session, "POST", path, heartbeat_payload)

    duration_ms = int((time.perf_counter() - started) * 1000)

    require_key(body, "tablet_device_id", route_name=path)
    require_key(body, "server_time_utc", route_name=path)

    results.append(RouteTestResult("POST", path, 200, duration_ms, True))


def test_network_events(session: Session, results: list[RouteTestResult]) -> None:
    """
    Test POST /tablet-edge/network-events.
    """
    path = "/tablet-edge/network-events"

    network_event_payload = {
        "tablet_uid": TABLET_UID,
        "events": [
            {
                "event_type": "server_health_good",
                "quality_level": "good",
                "server_url": BASE_URL,
                "page_url": "/assistant",
                "latency_ms": 120,
                "avg_latency_ms": 140,
                "consecutive_failures": 0,
                "is_online": True,
                "ssid": "TestWiFi",
                "wifi_rssi": -55,
                "signal_level": 4,
                "ip_address": "127.0.0.1",
                "gateway_address": "127.0.0.1",
                "message": "Route smoke test network event.",
            }
        ],
    }

    started = time.perf_counter()

    body = request_json(session, "POST", path, network_event_payload)

    duration_ms = int((time.perf_counter() - started) * 1000)

    accepted = require_key(body, "accepted", route_name=path)
    assert_int_at_least(accepted, 1, field_name="accepted", route_name=path)

    results.append(RouteTestResult("POST", path, 200, duration_ms, True))


def test_health_samples(session: Session, results: list[RouteTestResult]) -> None:
    """
    Test POST /tablet-edge/health-samples.
    """
    path = "/tablet-edge/health-samples"

    now_utc = datetime.now(timezone.utc).isoformat()

    health_sample_payload = {
        "tablet_uid": TABLET_UID,
        "samples": [
            {
                "sampled_at": now_utc,
                "server_reachable": True,
                "server_latency_ms": 120,
                "quality_level": "good",
                "battery_percent": 88,
                "is_charging": False,
                "ssid": "TestWiFi",
                "wifi_rssi": -55,
                "signal_level": 4,
                "app_foreground": True,
                "current_page_url": "/assistant",
            }
        ],
    }

    started = time.perf_counter()

    body = request_json(session, "POST", path, health_sample_payload)

    duration_ms = int((time.perf_counter() - started) * 1000)

    accepted = require_key(body, "accepted", route_name=path)
    assert_int_at_least(accepted, 1, field_name="accepted", route_name=path)

    results.append(RouteTestResult("POST", path, 200, duration_ms, True))


def test_dropdown_cache_status(session: Session, results: list[RouteTestResult]) -> None:
    """
    Test GET /tablet-edge/dropdown-cache/status.
    """
    path = f"/tablet-edge/dropdown-cache/status?tablet_uid={TABLET_UID}"

    started = time.perf_counter()

    body = request_json(session, "GET", path)

    duration_ms = int((time.perf_counter() - started) * 1000)

    require_key(body, "current_version", route_name=path)
    caches = require_key(body, "caches", route_name=path)

    if not isinstance(caches, dict):
        raise RuntimeError("dropdown-cache/status expected 'caches' to be a JSON object.")

    results.append(RouteTestResult("GET", path, 200, duration_ms, True))


def test_dropdown_cache_full(session: Session, results: list[RouteTestResult]) -> None:
    """
    Test GET /tablet-edge/dropdown-cache/full.
    """
    path = f"/tablet-edge/dropdown-cache/full?tablet_uid={TABLET_UID}"

    started = time.perf_counter()

    body = request_json(session, "GET", path)

    duration_ms = int((time.perf_counter() - started) * 1000)

    require_key(body, "cache_version", route_name=path)
    require_key(body, "cache_summary", route_name=path)

    results.append(RouteTestResult("GET", path, 200, duration_ms, True))


def test_dropdown_cache_delta(session: Session, results: list[RouteTestResult]) -> None:
    """
    Test GET /tablet-edge/dropdown-cache/delta.
    """
    since = datetime.now(timezone.utc).isoformat()
    path = f"/tablet-edge/dropdown-cache/delta?since={since}"

    started = time.perf_counter()

    body = request_json(session, "GET", path)

    duration_ms = int((time.perf_counter() - started) * 1000)

    require_key(body, "full_refresh_required", route_name=path)

    results.append(RouteTestResult("GET", path, 200, duration_ms, True))


def test_offline_events(session: Session, results: list[RouteTestResult]) -> None:
    """
    Test POST /tablet-edge/offline-events/sync.
    """
    path = "/tablet-edge/offline-events/sync"

    now_utc = datetime.now(timezone.utc).isoformat()

    offline_event_payload = {
        "tablet_uid": TABLET_UID,
        "events": [
            {
                "local_event_id": str(uuid4()),
                "event_type": "route_test_event",
                "client_created_at": now_utc,
                "event_payload": {
                    "message": "Offline event route smoke test.",
                    "source": "test_tablet_edge_routes.py",
                },
            }
        ],
    }

    started = time.perf_counter()

    body = request_json(session, "POST", path, offline_event_payload)

    duration_ms = int((time.perf_counter() - started) * 1000)

    accepted = require_key(body, "accepted", route_name=path)
    assert_int_at_least(accepted, 1, field_name="accepted", route_name=path)

    results.append(RouteTestResult("POST", path, 200, duration_ms, True))


def test_app_logs(session: Session, results: list[RouteTestResult]) -> None:
    """
    Test POST /tablet-edge/app-logs.
    """
    path = "/tablet-edge/app-logs"

    now_utc = datetime.now(timezone.utc).isoformat()

    app_log_payload = {
        "tablet_uid": TABLET_UID,
        "logs": [
            {
                "log_level": "INFO",
                "log_source": "test_tablet_edge_routes.py",
                "message": "Route smoke test app log.",
                "client_created_at": now_utc,
                "context": {
                    "base_url": BASE_URL,
                    "tablet_uid": TABLET_UID,
                    "test": "app_logs",
                },
            }
        ],
    }

    started = time.perf_counter()

    body = request_json(session, "POST", path, app_log_payload)

    duration_ms = int((time.perf_counter() - started) * 1000)

    accepted = require_key(body, "accepted", route_name=path)
    assert_int_at_least(accepted, 1, field_name="accepted", route_name=path)

    results.append(RouteTestResult("POST", path, 200, duration_ms, True))


def log_summary(results: list[RouteTestResult]) -> None:
    """
    Log final route smoke test summary.
    """
    logger.info("")
    logger.info("=" * 80)
    logger.info("Tablet Edge route smoke test summary")
    logger.info("=" * 80)

    for result in results:
        logger.info(
            "%-6s %-60s status=%s duration=%sms success=%s",
            result.method,
            result.path,
            result.status_code,
            result.duration_ms,
            result.success,
        )

    logger.info("=" * 80)
    logger.info("Total routes tested: %s", len(results))
    logger.info("All route smoke tests completed successfully.")
    logger.info("=" * 80)


def run_tests() -> None:
    """
    Run all Tablet Edge route smoke tests.
    """
    logger.info("Using BASE_URL=%s", BASE_URL)
    logger.info("Using TABLET_UID=%s", TABLET_UID)
    logger.info("Using TABLET_NAME=%s", TABLET_NAME)
    logger.info("Using REQUEST_TIMEOUT_SECONDS=%s", REQUEST_TIMEOUT_SECONDS)

    results: list[RouteTestResult] = []

    with requests.Session() as session:
        test_health(session, results)
        test_routes(session, results)
        test_register(session, results)
        test_heartbeat(session, results)
        test_network_events(session, results)
        test_health_samples(session, results)
        test_dropdown_cache_status(session, results)
        test_dropdown_cache_full(session, results)
        test_dropdown_cache_delta(session, results)
        test_offline_events(session, results)
        test_app_logs(session, results)

    log_summary(results)


def main() -> None:
    """
    Script entrypoint.
    """
    configure_logging()
    run_tests()


if __name__ == "__main__":
    main()