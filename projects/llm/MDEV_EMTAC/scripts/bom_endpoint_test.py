from __future__ import annotations

import requests

BASE_URL = "http://172.19.194.129:5000"

ENDPOINTS_TO_TEST = [

    # ───────── BOM ─────────
    "/bill_of_materials",
    "/api/bom/get_bom_list_data",
    "/api/bom/get_parts_position_data",
    "/search_bill_of_material",

    # ───── Enter New Part ─────
    "/enter_new_part/get_part_form_data",
    "/enter_new_part/enter_part",
    "/enter_new_part/part_image/1",

    # ───── Update Part ─────
    "/update_part/edit_part/1",
    "/update_part/edit_part_ajax/1",
    "/update_part/search_part",
    "/update_part/search_part_ajax",
    "/update_part/part_image/1",
]


def test_endpoint(base_url: str, endpoint: str) -> None:
    url = f"{base_url.rstrip('/')}{endpoint}"

    print("=" * 90)
    print(f"TESTING: {endpoint}")
    print(f"URL: {url}")

    try:
        response = requests.get(url, allow_redirects=False, timeout=10)
    except requests.RequestException as exc:
        print(f"REQUEST ERROR: {exc}")
        return

    status = response.status_code
    print(f"STATUS: {status}")

    if status == 404:
        print("❌ ROUTE NOT FOUND")

    elif status == 302:
        location = response.headers.get("Location", "")
        print(f"➡️ REDIRECT → {location}")

        if "/login" in location:
            print("✅ ROUTE EXISTS (login required)")
        else:
            print("➡️ OTHER REDIRECT")

    elif status == 200:
        print("✅ ROUTE EXISTS AND ACCESSIBLE")

    else:
        print("⚠️ OTHER RESPONSE")

    print("CONTENT-TYPE:", response.headers.get("Content-Type"))


def test_all():
    for endpoint in ENDPOINTS_TO_TEST:
        test_endpoint(BASE_URL, endpoint)


if __name__ == "__main__":
    test_all()