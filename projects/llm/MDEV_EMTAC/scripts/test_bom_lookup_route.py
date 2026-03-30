from __future__ import annotations

import json
import sys
import traceback
from pathlib import Path

# ------------------------------------------------------------
# Make project root importable
# ------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# ------------------------------------------------------------
# IMPORT YOUR APP FACTORY
# (based on the file you showed)
# ------------------------------------------------------------
from ai_emtac import create_app


# ------------------------------------------------------------
# CREATE APP INSTANCE
# ------------------------------------------------------------
app = create_app()


# ------------------------------------------------------------
# HELPER
# ------------------------------------------------------------
def print_response(response) -> None:
    print("=" * 80)
    print("ROUTE TEST RESPONSE")
    print("=" * 80)
    print(f"Status Code: {response.status_code}")
    print(f"Content-Type: {response.content_type}")
    print()

    try:
        payload = response.get_json()
        print(json.dumps(payload, indent=4, default=str))
    except Exception:
        print(response.data.decode("utf-8", errors="replace"))


# ------------------------------------------------------------
# TEST
# ------------------------------------------------------------
def main() -> None:

    with app.test_client() as client:

        # Simulate logged-in session (required by @login_required)
        with client.session_transaction() as sess:
            sess["user_id"] = 1
            sess["username"] = "test_user"
            sess["logged_in"] = True
            sess["user_level"] = "ADMIN"
            sess["login_record_id"] = 1  # your middleware checks this

        # ----------------------------------------------------
        # IMPORTANT: adjust URL if blueprint has prefix
        # ----------------------------------------------------
        response = client.get("/get_bom_list_data_v2")

        print_response(response)

        if response.status_code != 200:
            raise AssertionError(f"Expected 200, got {response.status_code}")

        payload = response.get_json()

        required_keys = {"success", "message", "data"}
        missing = required_keys - set(payload.keys())
        if missing:
            raise AssertionError(f"Missing keys: {sorted(missing)}")

        print()
        print("[PASS] Route returned valid authenticated JSON response.")


# ------------------------------------------------------------
if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print()
        print(f"[FAIL] {type(exc).__name__}: {exc}")
        traceback.print_exc()
        raise SystemExit(1)