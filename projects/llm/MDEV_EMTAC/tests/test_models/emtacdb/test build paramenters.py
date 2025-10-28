# Test function to verify the fix works
def test_build_search_parameters():
    """Test the updated parameter building"""

    # Mock analysis and doc objects for testing
    class MockDoc:
        def __init__(self, text):
            self.text = text

    # Test cases that should extract part numbers correctly
    test_cases = [
        {
            "query": "find part number A115957",
            "expected_part": "A115957",
            "description": "Should extract A115957, not 'NUMBER'"
        },
        {
            "query": "search for part A115957",
            "expected_part": "A115957",
            "description": "Should extract A115957 directly"
        },
        {
            "query": "part A115957",
            "expected_part": "A115957",
            "description": "Simple part reference"
        },
        {
            "query": "show me part number 115957",
            "expected_part": "115957",
            "description": "Numeric part number"
        },
        {
            "query": "part number for bearing assembly",
            "expected_search": "bearing assembly",
            "description": "Should use search_text for descriptions"
        },
        {
            "query": "need part number for BEARING ASSEMBLY",
            "expected_search": "BEARING ASSEMBLY",
            "description": "Description request"
        }
    ]

    # Mock search parameters builder (would be a method of your class)
    def mock_build_search_parameters(query):
        mock_analysis = {"entities": {"part_numbers": [], "areas": [], "equipment": [], "numbers": []}}
        mock_doc = MockDoc(query)

        # Use the updated method (you would call self._build_search_parameters in real code)
        return _build_search_parameters(None, mock_analysis, mock_doc)

    print("🧪 Testing updated _build_search_parameters method:")
    print("=" * 60)

    for i, test in enumerate(test_cases, 1):
        query = test["query"]
        result = mock_build_search_parameters(query)

        print(f"\nTest {i}: {test['description']}")
        print(f"Query: '{query}'")
        print(f"Result: {result}")

        # Check if we got the expected result
        if "expected_part" in test:
            expected = test["expected_part"]
            actual = result.get("part_number")
            status = "✅ PASS" if actual == expected else "❌ FAIL"
            print(f"Expected part: '{expected}', Got: '{actual}' - {status}")

        elif "expected_search" in test:
            expected = test["expected_search"].lower()
            actual = result.get("search_text", "").lower()
            status = "✅ PASS" if expected in actual else "❌ FAIL"
            print(f"Expected search text containing: '{expected}', Got: '{actual}' - {status}")

        print("-" * 40)


if __name__ == "__main__":
    test_build_search_parameters()