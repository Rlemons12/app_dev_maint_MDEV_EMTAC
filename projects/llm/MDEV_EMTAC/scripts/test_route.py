from __future__ import annotations

import argparse
import json
import sys
import traceback
from pathlib import Path
from typing import Any, Dict, Optional

# ------------------------------------------------------------
# Make project root importable
# ------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# ------------------------------------------------------------
# IMPORT YOUR APP FACTORY
# ------------------------------------------------------------
from ai_emtac import create_app


# ------------------------------------------------------------
# DEFAULT TEST SESSION
# Adjust these if your auth system changes
# ------------------------------------------------------------
DEFAULT_SESSION = {
    "user_id": 1,
    "username": "test_user",
    "logged_in": True,
    "user_level": "ADMIN",
    "login_record_id": 1,
}


# ------------------------------------------------------------
# HELPERS
# ------------------------------------------------------------
def safe_json_loads(raw: Optional[str], label: str) -> Optional[Dict[str, Any]]:
    if not raw:
        return None

    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON for {label}: {exc}") from exc

    if not isinstance(parsed, dict):
        raise ValueError(f"{label} must be a JSON object")
    return parsed


def print_banner(title: str) -> None:
    print("\n" + "=" * 100)
    print(title)
    print("=" * 100)


def print_request_summary(
    method: str,
    url: str,
    query: Optional[Dict[str, Any]],
    json_body: Optional[Dict[str, Any]],
    form_data: Optional[Dict[str, Any]],
    headers: Optional[Dict[str, Any]],
    session_data: Optional[Dict[str, Any]],
) -> None:
    print_banner("REQUEST SUMMARY")
    print(f"Method: {method}")
    print(f"URL: {url}")
    print(f"Query Params: {json.dumps(query, indent=4, default=str) if query else '{}'}")
    print(f"JSON Body: {json.dumps(json_body, indent=4, default=str) if json_body else '{}'}")
    print(f"Form Data: {json.dumps(form_data, indent=4, default=str) if form_data else '{}'}")
    print(f"Headers: {json.dumps(headers, indent=4, default=str) if headers else '{}'}")
    print(f"Session: {json.dumps(session_data, indent=4, default=str) if session_data else '{}'}")


def print_response(response) -> None:
    print_banner("RESPONSE")
    print(f"Status Code: {response.status_code}")
    print(f"Content-Type: {response.content_type}")
    print()

    try:
        payload = response.get_json()
        if payload is not None:
            print(json.dumps(payload, indent=4, default=str))
            return
    except Exception:
        pass

    print(response.data.decode("utf-8", errors="replace"))


def build_client_request_kwargs(
    query: Optional[Dict[str, Any]],
    json_body: Optional[Dict[str, Any]],
    form_data: Optional[Dict[str, Any]],
    headers: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    kwargs: Dict[str, Any] = {}

    if query:
        kwargs["query_string"] = query

    if headers:
        kwargs["headers"] = headers

    if json_body is not None:
        kwargs["json"] = json_body
    elif form_data is not None:
        kwargs["data"] = form_data

    return kwargs


def get_request_callable(client, method: str):
    method = method.upper()

    mapping = {
        "GET": client.get,
        "POST": client.post,
        "PUT": client.put,
        "PATCH": client.patch,
        "DELETE": client.delete,
        "OPTIONS": client.options,
    }

    if method not in mapping:
        raise ValueError(
            f"Unsupported method '{method}'. "
            f"Supported methods: {', '.join(mapping.keys())}"
        )

    return mapping[method]


def run_route_test(
    *,
    method: str,
    url: str,
    query: Optional[Dict[str, Any]] = None,
    json_body: Optional[Dict[str, Any]] = None,
    form_data: Optional[Dict[str, Any]] = None,
    headers: Optional[Dict[str, Any]] = None,
    session_data: Optional[Dict[str, Any]] = None,
    expected_status: Optional[int] = None,
    require_json: bool = False,
    show_url_map: bool = False,
) -> int:
    app = create_app()

    if show_url_map:
        print_banner("REGISTERED ROUTES")
        for rule in sorted(app.url_map.iter_rules(), key=lambda r: str(r)):
            print(f"{rule.methods} {rule}")

    print_request_summary(
        method=method,
        url=url,
        query=query,
        json_body=json_body,
        form_data=form_data,
        headers=headers,
        session_data=session_data,
    )

    with app.test_client() as client:
        if session_data:
            with client.session_transaction() as sess:
                for key, value in session_data.items():
                    sess[key] = value

        request_callable = get_request_callable(client, method)
        request_kwargs = build_client_request_kwargs(
            query=query,
            json_body=json_body,
            form_data=form_data,
            headers=headers,
        )

        response = request_callable(url, **request_kwargs)

        print_response(response)

        if expected_status is not None and response.status_code != expected_status:
            raise AssertionError(
                f"Expected status {expected_status}, got {response.status_code}"
            )

        if require_json:
            payload = response.get_json()
            if payload is None:
                raise AssertionError("Expected JSON response but route did not return JSON")

        print_banner("RESULT")
        print("[PASS] Route test completed successfully.")
        return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Reusable EMTAC Flask route test harness"
    )

    parser.add_argument("method", help="HTTP method, e.g. GET POST PUT DELETE")
    parser.add_argument("url", help="Route URL, e.g. /get_bom_list_data_v2")

    parser.add_argument(
        "--query",
        help='Query params as JSON object, e.g. \'{"page": 1, "q": "test"}\'',
    )
    parser.add_argument(
        "--json-body",
        help='JSON body as JSON object, e.g. \'{"name": "abc"}\'',
    )
    parser.add_argument(
        "--form-data",
        help='Form data as JSON object, e.g. \'{"username": "bob"}\'',
    )
    parser.add_argument(
        "--headers",
        help='Headers as JSON object, e.g. \'{"X-Test": "1"}\'',
    )
    parser.add_argument(
        "--session",
        help=(
            "Session data as JSON object. "
            "If omitted, authenticated default session is used."
        ),
    )
    parser.add_argument(
        "--no-auth",
        action="store_true",
        help="Do not inject default authenticated session",
    )
    parser.add_argument(
        "--expected-status",
        type=int,
        help="Expected HTTP status code",
    )
    parser.add_argument(
        "--require-json",
        action="store_true",
        help="Fail if response is not JSON",
    )
    parser.add_argument(
        "--show-routes",
        action="store_true",
        help="Print all registered Flask routes before testing",
    )

    return parser.parse_args()


def main() -> int:
    args = parse_args()

    query = safe_json_loads(args.query, "query")
    json_body = safe_json_loads(args.json_body, "json-body")
    form_data = safe_json_loads(args.form_data, "form-data")
    headers = safe_json_loads(args.headers, "headers")

    if json_body is not None and form_data is not None:
        raise ValueError("Use either --json-body or --form-data, not both")

    if args.no_auth:
        session_data = None
    elif args.session:
        session_data = safe_json_loads(args.session, "session")
    else:
        session_data = DEFAULT_SESSION.copy()

    return run_route_test(
        method=args.method,
        url=args.url,
        query=query,
        json_body=json_body,
        form_data=form_data,
        headers=headers,
        session_data=session_data,
        expected_status=args.expected_status,
        require_json=args.require_json,
        show_url_map=args.show_routes,
    )


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print_banner("FAILURE")
        print(f"{type(exc).__name__}: {exc}")
        print()
        traceback.print_exc()
        raise SystemExit(1)