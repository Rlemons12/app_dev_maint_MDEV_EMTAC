# Quick debug script to test part search directly
# You can run this in a Python console or add it temporarily to test

from modules.configuration.config_env import DatabaseConfig
from modules.emtacdb.emtacdb_fts import Part


# Test 1: Check if the part exists at all
def test_part_exists():
    db_config = DatabaseConfig()
    session = db_config.get_main_session()

    try:
        # Check exact match
        exact_part = session.query(Part).filter(Part.part_number == 'A115957').first()
        print(f"Exact match for 'A115957': {exact_part}")
        if exact_part:
            print(f"  Found: {exact_part.part_number} - {exact_part.name}")

        # Check case-insensitive match
        ilike_part = session.query(Part).filter(Part.part_number.ilike('A115957')).first()
        print(f"Case-insensitive match for 'A115957': {ilike_part}")

        # Check partial match
        partial_parts = session.query(Part).filter(Part.part_number.ilike('%115957%')).all()
        print(f"Partial matches for '115957': {len(partial_parts)} found")
        for p in partial_parts[:5]:  # Show first 5
            print(f"  {p.part_number} - {p.name}")

        # Show some existing parts for reference
        sample_parts = session.query(Part).limit(5).all()
        print(f"\nSample parts in database:")
        for p in sample_parts:
            print(f"  {p.part_number} - {p.name}")

    finally:
        session.close()


# Test 2: Test Part.search method directly
def test_part_search_method():
    db_config = DatabaseConfig()
    session = db_config.get_main_session()

    try:
        print("\n=== Testing Part.search method ===")

        # Test 1: Search by part_number
        results1 = Part.search(session=session, part_number='A115957', limit=5)
        print(f"Part.search(part_number='A115957'): {len(results1)} results")
        for p in results1:
            print(f"  {p.part_number} - {p.name}")

        # Test 2: Search by search_text
        results2 = Part.search(session=session, search_text='A115957', limit=5)
        print(f"Part.search(search_text='A115957'): {len(results2)} results")
        for p in results2:
            print(f"  {p.part_number} - {p.name}")

        # Test 3: Search with broader text
        results3 = Part.search(session=session, search_text='115957', limit=5)
        print(f"Part.search(search_text='115957'): {len(results3)} results")
        for p in results3:
            print(f"  {p.part_number} - {p.name}")

    finally:
        session.close()


# Run the tests
if __name__ == "__main__":
    test_part_exists()
    test_part_search_method()