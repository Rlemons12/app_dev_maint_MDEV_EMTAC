# Test function to verify the fixes
def test_part_extraction():
    """Test the part number extraction"""
    test_queries = [
        "find part number A115957",
        "search for part A115957",
        "part A115957",
        "show me part number 115957",
        "A115957",
        "need part number for bearing assembly"
    ]

    for query in test_queries:
        result = fix_part_search_extraction(query)
        print(f"Query: '{query}' -> Part: '{result.get('part_number', 'NONE')}'")


if __name__ == "__main__":
    test_part_extraction()