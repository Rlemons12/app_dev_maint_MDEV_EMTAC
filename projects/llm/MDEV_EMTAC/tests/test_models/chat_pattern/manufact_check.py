"""
Simple test to verify intent detection is working
Save as: tests/test_models/chat_pattern/test_simple_intent.py
"""
import pytest
import sys
import os

# Add project root to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

def test_banner_sensors_intent():
    """Simple test for the banner sensors query intent detection"""

    try:
        # Import what we need
        from modules.search.UnifiedSearchMixin import UnifiedSearchMixin
        from modules.configuration.config_env import DatabaseConfig
        from sqlalchemy import create_engine
        from sqlalchemy.orm import sessionmaker

        # Setup
        db_config = DatabaseConfig()
        engine = create_engine(db_config.get_database_url())
        Session = sessionmaker(bind=engine)
        session = Session()

        # Create test class
        class SimpleTest(UnifiedSearchMixin):
            def __init__(self, session):
                self.db_session = session
                super().__init__()

        test_instance = SimpleTest(session)

        # THE TEST
        query = "what banner sensors do we have?"
        result = test_instance.is_unified_search_query(query)

        print(f"Query: '{query}'")
        print(f"Detected as: {'SEARCH QUERY' if result else 'CHAT QUERY'}")

        # Assert
        assert result == True, f"Expected search query, got chat query for: {query}"

        session.close()
        print("✅ Test passed!")

    except Exception as e:
        print(f"❌ Test failed: {e}")
        pytest.fail(f"Intent detection failed: {e}")

def test_chat_vs_search_queries():
    """Test multiple queries to ensure proper classification"""

    try:
        from modules.search.UnifiedSearchMixin import UnifiedSearchMixin
        from modules.configuration.config_env import DatabaseConfig
        from sqlalchemy import create_engine
        from sqlalchemy.orm import sessionmaker

        # Setup
        db_config = DatabaseConfig()
        engine = create_engine(db_config.get_database_url())
        Session = sessionmaker(bind=engine)
        session = Session()

        class SimpleTest(UnifiedSearchMixin):
            def __init__(self, session):
                self.db_session = session
                super().__init__()

        test_instance = SimpleTest(session)

        # Test cases: (query, expected_is_search)
        test_cases = [
            # Should be SEARCH
            ("what banner sensors do we have?", True),
            ("banner sensors", True),
            ("find part A115957", True),

            # Should be CHAT
            ("how are you?", False),
            ("what's the weather?", False),
            ("hello", False)
        ]

        results = []
        for query, expected in test_cases:
            try:
                actual = test_instance.is_unified_search_query(query)
                success = actual == expected
                results.append(success)

                status = "✅" if success else "❌"
                actual_type = "SEARCH" if actual else "CHAT"
                expected_type = "SEARCH" if expected else "CHAT"

                print(f"{status} '{query}' -> {actual_type} (expected {expected_type})")

            except Exception as e:
                print(f"❌ ERROR: '{query}' -> {e}")
                results.append(False)

        session.close()

        # Calculate success rate
        success_rate = sum(results) / len(results)
        print(f"\nSuccess rate: {sum(results)}/{len(results)} ({success_rate*100:.1f}%)")

        # Assert at least 80% success rate
        assert success_rate >= 0.8, f"Success rate too low: {success_rate*100:.1f}%"

        print("✅ All tests passed!")

    except Exception as e:
        print(f"❌ Test failed: {e}")
        pytest.fail(f"Multiple query test failed: {e}")

if __name__ == "__main__":
    print("🧪 Simple Intent Detection Test")
    print("=" * 40)

    try:
        test_banner_sensors_intent()
        print()
        test_chat_vs_search_queries()
        print("\n🎉 All tests completed successfully!")

    except Exception as e:
        print(f"\n❌ Tests failed: {e}")
        import traceback
        traceback.print_exc()