#!/usr/bin/env python3
"""
Quick test to check Flask endpoints and blueprint registration
"""

import requests
import json

BASE_URL = "http://127.0.0.1:5000"  # Use exact URL from your server


def test_basic_connectivity():
    """Test basic server connectivity"""
    print("🔌 Testing basic connectivity...")

    try:
        response = requests.get(f"{BASE_URL}/", timeout=5)
        print(f"✅ Server responds - Status: {response.status_code}")
        print(f"   Content-Type: {response.headers.get('Content-Type', 'Unknown')}")
        print(f"   Server: {response.headers.get('Server', 'Unknown')}")
        return True
    except Exception as e:
        print(f"❌ Cannot connect: {e}")
        return False


def test_drawing_endpoints():
    """Test drawing-specific endpoints"""
    print("\n🎯 Testing drawing endpoints...")

    endpoints_to_test = [
        "/drawings/search",
        "/drawings/types",
        "/drawings/search?limit=1",
        "/api/drawings/search",  # Alternative path
    ]

    for endpoint in endpoints_to_test:
        try:
            print(f"\n   Testing: {endpoint}")
            response = requests.get(f"{BASE_URL}{endpoint}", timeout=10)

            print(f"      Status: {response.status_code}")
            print(f"      Content-Type: {response.headers.get('Content-Type', 'Unknown')}")

            if response.status_code == 200:
                try:
                    data = response.json()
                    print(f"      ✅ Valid JSON - Keys: {list(data.keys()) if isinstance(data, dict) else type(data)}")
                    if isinstance(data, dict) and 'count' in data:
                        print(f"      📊 Results: {data.get('count')} items")
                except json.JSONDecodeError:
                    print(f"      ❌ Invalid JSON - Response: {response.text[:100]}...")

            elif response.status_code == 404:
                print(f"      ❌ Not Found - Check blueprint registration")

            elif response.status_code == 500:
                print(f"      ❌ Server Error - Check Flask logs")
                print(f"      Response: {response.text[:200]}...")

            else:
                print(f"      ⚠️  Unexpected status")
                print(f"      Response: {response.text[:200]}...")

        except Exception as e:
            print(f"      ❌ Request failed: {e}")


def check_blueprint_registration():
    """Check if blueprints are registered"""
    print("\n📋 Checking common Flask patterns...")

    # Test common Flask endpoints
    test_paths = [
        "/",
        "/api",
        "/health",
        "/status",
    ]

    for path in test_paths:
        try:
            response = requests.get(f"{BASE_URL}{path}", timeout=5)
            print(f"   {path}: {response.status_code}")
        except:
            print(f"   {path}: Not accessible")


def main():
    """Run all tests"""
    print("🧪 Quick Flask Endpoint Test")
    print("=" * 40)

    if not test_basic_connectivity():
        return

    test_drawing_endpoints()
    check_blueprint_registration()

    print("\n" + "=" * 40)
    print("💡 Next Steps:")
    print("1. If drawing endpoints return 404:")
    print("   - Check blueprint registration in your main app file")
    print("   - Ensure: app.register_blueprint(drawing_routes)")
    print("\n2. If you get 500 errors:")
    print("   - Check Flask console for error details")
    print("   - Look for import or database connection errors")
    print("\n3. If endpoints work:")
    print("   - Update your test script to use http://127.0.0.1:5000")


if __name__ == "__main__":
    main()