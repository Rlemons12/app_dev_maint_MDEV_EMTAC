# =====================================================
# PYTEST-BASED SEARCH TESTING SUITE - FIXED VERSION
# Professional testing framework with proper fixtures and reporting
# =====================================================

import pytest
import time
import logging
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
import pandas as pd
from dataclasses import dataclass, asdict
import re
import json
from pathlib import Path
from sqlalchemy import text

logger = logging.getLogger(__name__)


# =====================================================
# TEST CONFIGURATION AND FIXTURES
# =====================================================

@pytest.fixture(scope="session")
def search_system():
    """Initialize search system once per test session."""
    try:
        # FIXED: Import the correct search system
        from modules.search.nlp_search import SpaCyEnhancedAggregateSearch
        from modules.configuration.config_env import DatabaseConfig

        # Initialize with database session
        db_config = DatabaseConfig()
        session = db_config.get_main_session()

        system = SpaCyEnhancedAggregateSearch(session=session)
        logger.info("SpaCy Enhanced search system initialized successfully")
        return system
    except Exception as e:
        logger.error(f"Could not initialize search system: {e}")
        pytest.skip(f"Could not initialize search system: {e}")


@pytest.fixture(scope="session")
def aggregate_search():
    """Initialize AggregateSearch system for fallback testing."""
    try:
        from modules.search.aggregate_search import AggregateSearch
        from modules.configuration.config_env import DatabaseConfig

        db_config = DatabaseConfig()
        session = db_config.get_main_session()

        system = AggregateSearch(session=session)
        logger.info("AggregateSearch system initialized successfully")
        return system
    except Exception as e:
        logger.warning(f"Could not initialize AggregateSearch: {e}")
        return None


@pytest.fixture(scope="session")
def db_session():
    """Initialize database session once per test session."""
    try:
        from modules.configuration.config_env import DatabaseConfig
        db_config = DatabaseConfig()
        session = db_config.get_main_session()
        logger.info("Database session initialized successfully")
        yield session
        session.close()
    except Exception as e:
        pytest.skip(f"Could not initialize database session: {e}")


@pytest.fixture(scope="session")
def all_parts(db_session):
    """Load all parts from database once per test session."""
    try:
        from modules.emtacdb.emtacdb_fts import Part
        parts = db_session.query(Part).limit(100).all()  # Limit for testing
        logger.info(f"Loaded {len(parts)} parts for testing")
        return parts
    except Exception as e:
        pytest.skip(f"Could not load parts from database: {e}")


@pytest.fixture(scope="session")
def test_config():
    """Test configuration settings."""
    return {
        'max_execution_time_ms': 5000,  # Fail if search takes >5 seconds
        'minimum_success_rate': 70,  # Minimum acceptable success rate %
        'top_result_weight': 0.8,  # Weight for finding target in top result
        'export_results': True,  # Export detailed results to CSV
        'test_subset_size': 50,  # Test subset size for faster testing
    }


# =====================================================
# TEST DATA CLASSES
# =====================================================

@dataclass
class SearchTestCase:
    """Individual search test case."""
    part_id: int
    part_number: str
    part_name: str
    query: str
    query_type: str
    expected_intent: str = "FIND_PART"


@dataclass
class SearchTestResult:
    """Results from executing a search test case."""
    test_case: SearchTestCase
    detected_intent: str
    intent_confidence: float
    search_method: str
    result_count: int
    found_target_part: bool
    target_part_rank: Optional[int]
    execution_time_ms: int
    success: bool
    error_message: Optional[str] = None


# =====================================================
# TEST GENERATORS
# =====================================================

class SearchTestGenerator:
    """Generate test cases for different query patterns."""

    @staticmethod
    def generate_direct_part_number_tests(part) -> List[SearchTestCase]:
        """Generate direct part number test cases."""
        part_number = getattr(part, 'part_number', '')
        if not part_number:
            return []

        queries = [
            f"find part {part_number}",
            f"search for part {part_number}",
            f"show part {part_number}",
            part_number  # Just the part number
        ]

        return [
            SearchTestCase(
                part_id=getattr(part, 'id'),
                part_number=part_number,
                part_name=getattr(part, 'name', ''),
                query=query,
                query_type='direct_part_number'
            ) for query in queries
        ]

    @staticmethod
    def generate_description_lookup_tests(part) -> List[SearchTestCase]:
        """Generate description lookup test cases."""
        name = getattr(part, 'name', '')
        if not name or len(name) < 3:
            return []

        # Clean description for testing
        clean_name = SearchTestGenerator._clean_description(name)
        if not clean_name:
            return []

        queries = [
            f"what is the part number for {clean_name}",
            f"part number for {clean_name}",
            f"I need the part number for {clean_name}"
        ]

        return [
            SearchTestCase(
                part_id=getattr(part, 'id'),
                part_number=getattr(part, 'part_number', ''),
                part_name=name,
                query=query,
                query_type='description_lookup'
            ) for query in queries
        ]

    @staticmethod
    def generate_equipment_type_tests(part) -> List[SearchTestCase]:
        """Generate equipment type search test cases."""
        name = getattr(part, 'name', '')
        notes = getattr(part, 'notes', '')

        equipment_types = SearchTestGenerator._extract_equipment_types(name, notes)
        if not equipment_types:
            return []

        test_cases = []
        for eq_type in equipment_types[:1]:  # Limit to top 1 type
            queries = [
                f"I'm looking for a {eq_type}",
                f"show me {eq_type}s"
            ]

            for query in queries:
                test_cases.append(SearchTestCase(
                    part_id=getattr(part, 'id'),
                    part_number=getattr(part, 'part_number', ''),
                    part_name=name,
                    query=query,
                    query_type='equipment_type_search'
                ))

        return test_cases

    @staticmethod
    def _clean_description(description: str) -> str:
        """Clean description for query testing."""
        if not description:
            return ""

        cleaned = description.lower()

        # Remove specifications that might be too specific
        cleaned = re.sub(r'\d+[\"\'\-\s]*x[\s\-]*\d+[\"\']?', '', cleaned)
        cleaned = re.sub(r'\d+\.\d+[\"\']*', '', cleaned)
        cleaned = re.sub(r'\d+[\"\']+', '', cleaned)
        cleaned = re.sub(r'\d+[\-\s]*\d*\s*v(olt)?s?', '', cleaned)
        cleaned = re.sub(r'part\s*#?\s*\w+', '', cleaned)

        # Clean up spaces
        cleaned = ' '.join(cleaned.split()).strip()

        return cleaned if len(cleaned) > 2 else ""

    @staticmethod
    def _extract_equipment_types(name: str, notes: str) -> List[str]:
        """Extract equipment types from part info."""
        text = f"{name} {notes}".lower()

        equipment_types = [
            'motor', 'pump', 'valve', 'bearing', 'gear', 'sensor', 'switch',
            'filter', 'belt', 'relay', 'coupling', 'seal', 'gasket',
            'transformer', 'compressor', 'fan', 'actuator', 'controller'
        ]

        return [eq_type for eq_type in equipment_types if eq_type in text]


# =====================================================
# SEARCH TEST EXECUTOR
# =====================================================

class SearchTestExecutor:
    """Execute and analyze search tests."""

    def __init__(self, search_system, config: Dict):
        self.search_system = search_system
        self.config = config

    def execute_search_test(self, test_case: SearchTestCase) -> SearchTestResult:
        """Execute a single search test."""
        start_time = time.time()

        try:
            # Execute search using the NLP enhanced system
            result = self.search_system.execute_nlp_aggregated_search(test_case.query)

            execution_time = int((time.time() - start_time) * 1000)

            # Analyze results
            found_target, rank = self._check_if_target_found(result, test_case)

            # Extract search metadata
            nlp_analysis = result.get('nlp_analysis', {})
            detected_intent = nlp_analysis.get('detected_intent', 'UNKNOWN')
            intent_confidence = nlp_analysis.get('overall_confidence', 0.0)
            search_method = result.get('search_type', 'unknown')

            # Determine success criteria
            success = (
                    found_target and
                    execution_time <= self.config['max_execution_time_ms'] and
                    result.get('status') == 'success'
            )

            return SearchTestResult(
                test_case=test_case,
                detected_intent=detected_intent,
                intent_confidence=intent_confidence,
                search_method=search_method,
                result_count=result.get('count', 0),
                found_target_part=found_target,
                target_part_rank=rank,
                execution_time_ms=execution_time,
                success=success
            )

        except Exception as e:
            execution_time = int((time.time() - start_time) * 1000)

            return SearchTestResult(
                test_case=test_case,
                detected_intent="ERROR",
                intent_confidence=0.0,
                search_method="error",
                result_count=0,
                found_target_part=False,
                target_part_rank=None,
                execution_time_ms=execution_time,
                success=False,
                error_message=str(e)
            )

    def _check_if_target_found(self, search_result: Dict, test_case: SearchTestCase) -> Tuple[bool, Optional[int]]:
        """Check if target part was found in search results."""
        # Check different result formats
        parts = []

        # Check for unified search format
        if 'results_by_type' in search_result:
            parts = search_result.get('results_by_type', {}).get('parts', [])

        # Check for direct results format
        elif 'results' in search_result:
            results = search_result.get('results', [])
            # Filter for parts
            parts = [r for r in results if r.get('type') == 'part' or 'part_number' in r]

        for rank, result_part in enumerate(parts, 1):
            result_id = result_part.get('id')
            result_part_number = result_part.get('part_number', '')

            if (result_id == test_case.part_id or
                    (result_part_number and result_part_number == test_case.part_number)):
                return True, rank

        return False, None

    def log_test_results(self, test_name: str, results: List[SearchTestResult]):
        """Log test results for analysis."""
        if not results:
            logger.warning(f"{test_name}: No results to analyze")
            return

        success_count = sum(1 for r in results if r.success)
        total_count = len(results)
        avg_time = sum(r.execution_time_ms for r in results) / total_count

        logger.info(
            f"{test_name}: {success_count}/{total_count} success "
            f"({success_count / total_count * 100:.1f}%), avg {avg_time:.0f}ms"
        )

        # Log failures for debugging
        failures = [r for r in results if not r.success]
        if failures:
            logger.debug(f"Failures in {test_name}:")
            for failure in failures[:3]:  # Show first 3 failures
                logger.debug(f"  - '{failure.test_case.query}' -> {failure.error_message or 'No target found'}")


# =====================================================
# PYTEST TEST CLASSES
# =====================================================

class TestSearchSystemHealth:
    """Test basic search system health and connectivity."""

    def test_search_system_available(self, search_system):
        """Test that search system is available and functional."""
        assert search_system is not None
        assert hasattr(search_system, 'execute_nlp_aggregated_search')

    def test_database_connectivity(self, db_session):
        """Test database connectivity."""
        assert db_session is not None
        # FIXED: Use proper SQLAlchemy text() for raw SQL
        result = db_session.execute(text("SELECT 1")).scalar()
        assert result == 1

    def test_parts_table_accessible(self, all_parts):
        """Test that parts table is accessible and has data."""
        assert len(all_parts) > 0

        # Test first part has expected attributes
        first_part = all_parts[0]
        assert hasattr(first_part, 'id')
        assert hasattr(first_part, 'part_number') or hasattr(first_part, 'name')

    def test_search_system_methods(self, search_system):
        """Test that search system has required methods."""
        assert hasattr(search_system, 'execute_nlp_aggregated_search')
        assert hasattr(search_system, 'analyze_user_input')

        # Test basic functionality
        result = search_system.analyze_user_input("test query")
        assert isinstance(result, dict)


class TestDirectPartNumberSearch:
    """Test direct part number search functionality."""

    def test_direct_part_number_search_sample(self, search_system, all_parts, test_config):
        """Test direct part number searches on a sample of parts."""
        # Test first 10 parts to keep test time reasonable
        test_parts = all_parts[:10]
        executor = SearchTestExecutor(search_system, test_config)

        all_results = []

        for part in test_parts:
            test_cases = SearchTestGenerator.generate_direct_part_number_tests(part)

            if not test_cases:
                continue

            part_results = []
            for test_case in test_cases:
                result = executor.execute_search_test(test_case)
                part_results.append(result)
                all_results.append(result)

            # Log results for this part
            part_number = getattr(part, 'part_number', 'unknown')
            executor.log_test_results(f"Direct search for {part_number}", part_results)

        # Overall analysis
        if all_results:
            success_rate = sum(1 for r in all_results if r.success) / len(all_results) * 100
            executor.log_test_results("Overall direct part number search", all_results)

            # Should have at least 60% success rate for direct part numbers
            assert success_rate >= 60, f"Direct part number search success rate too low: {success_rate:.1f}%"
        else:
            pytest.skip("No valid test cases generated")

    def test_specific_part_patterns(self, search_system, test_config):
        """Test specific part number patterns that should work."""
        executor = SearchTestExecutor(search_system, test_config)

        # Test with known patterns
        test_queries = [
            "find part A115957",
            "search for part 123456",
            "show part XYZ-789",
            "part ABC123"
        ]

        results = []
        for query in test_queries:
            # Create a mock test case
            test_case = SearchTestCase(
                part_id=0,  # We don't care about finding specific part
                part_number="TEST",
                part_name="Test Part",
                query=query,
                query_type="direct_part_number"
            )

            result = executor.execute_search_test(test_case)
            results.append(result)

            # At minimum, search should not error and should detect FIND_PART intent
            assert result.error_message is None, f"Search failed for '{query}': {result.error_message}"

            # Intent should be detected as FIND_PART or similar
            valid_intents = ["FIND_PART", "FIND_PART_BY_NUMBER", "PART_SEARCH"]
            assert any(intent in result.detected_intent for intent in valid_intents), \
                f"Wrong intent detected for '{query}': {result.detected_intent}"

        executor.log_test_results("Specific part pattern tests", results)


class TestDescriptionLookupSearch:
    """Test part number lookup by description."""

    def test_description_lookup_sample(self, search_system, all_parts, test_config):
        """Test description-based part number lookup on a sample."""
        # Test first 5 parts for description lookup
        test_parts = all_parts[:5]
        executor = SearchTestExecutor(search_system, test_config)

        all_results = []

        for part in test_parts:
            test_cases = SearchTestGenerator.generate_description_lookup_tests(part)

            if not test_cases:
                continue

            part_results = []
            for test_case in test_cases:
                result = executor.execute_search_test(test_case)
                part_results.append(result)
                all_results.append(result)

            # Log results for this part
            part_name = getattr(part, 'name', 'unknown')
            executor.log_test_results(f"Description lookup for {part_name}", part_results)

        if all_results:
            success_rate = sum(1 for r in all_results if r.success) / len(all_results) * 100
            executor.log_test_results("Overall description lookup search", all_results)

            # Description lookup is harder, so lower threshold
            assert success_rate >= 40, f"Description lookup success rate too low: {success_rate:.1f}%"
        else:
            pytest.skip("No valid description lookup tests generated")

    def test_description_lookup_patterns(self, search_system, test_config):
        """Test specific description lookup patterns."""
        executor = SearchTestExecutor(search_system, test_config)

        test_queries = [
            "what is the part number for bearing assembly",
            "part number for motor valve",
            "I need the part number for hydraulic pump"
        ]

        results = []
        for query in test_queries:
            test_case = SearchTestCase(
                part_id=0,
                part_number="TEST",
                part_name="Test Part",
                query=query,
                query_type="description_lookup"
            )

            result = executor.execute_search_test(test_case)
            results.append(result)

            # Should not error and should detect as part search
            assert result.error_message is None, f"Search failed for '{query}': {result.error_message}"

        executor.log_test_results("Description lookup pattern tests", results)


class TestPerformanceRequirements:
    """Test performance requirements."""

    def test_search_response_time(self, search_system):
        """Test that searches complete within reasonable time."""
        test_queries = [
            "find part A115957",
            "what's the part number for motor",
            "I'm looking for a bearing"
        ]

        for query in test_queries:
            start_time = time.time()

            result = search_system.execute_nlp_aggregated_search(query)

            execution_time = int((time.time() - start_time) * 1000)

            # Should complete within 5 seconds
            assert execution_time <= 5000, f"Query '{query}' took {execution_time}ms (too slow)"

            logger.info(f"Query '{query}' completed in {execution_time}ms")

    def test_concurrent_searches(self, search_system):
        """Test that multiple searches can be handled."""
        import threading
        import queue

        queries = [
            "find part ABC123",
            "search for motor",
            "what is the part number for valve"
        ]

        results_queue = queue.Queue()

        def execute_search(query):
            try:
                start_time = time.time()
                result = search_system.execute_nlp_aggregated_search(query)
                execution_time = time.time() - start_time
                results_queue.put((query, execution_time, None))
            except Exception as e:
                results_queue.put((query, 0, str(e)))

        # Start threads
        threads = []
        for query in queries:
            thread = threading.Thread(target=execute_search, args=(query,))
            threads.append(thread)
            thread.start()

        # Wait for completion
        for thread in threads:
            thread.join(timeout=10)  # 10 second timeout

        # Check results
        completed_searches = 0
        while not results_queue.empty():
            query, exec_time, error = results_queue.get()
            if error is None:
                completed_searches += 1
                logger.info(f"Concurrent search '{query}' completed in {exec_time:.3f}s")
            else:
                logger.error(f"Concurrent search '{query}' failed: {error}")

        # At least 2 out of 3 should complete successfully
        assert completed_searches >= 2, f"Only {completed_searches} out of {len(queries)} concurrent searches completed"


class TestIntentDetection:
    """Test intent detection accuracy."""

    def test_intent_detection_accuracy(self, search_system):
        """Test that intent detection works correctly."""
        test_cases = [
            ("find part A115957", ["FIND_PART", "FIND_PART_BY_NUMBER"]),
            ("what's the part number for bearing", ["FIND_PART", "PART_SEARCH"]),
            ("I'm looking for a gear", ["FIND_PART", "EQUIPMENT_SEARCH"]),
            ("search for motor", ["FIND_PART", "EQUIPMENT_SEARCH"]),
        ]

        correct_detections = 0

        for query, expected_intents in test_cases:
            try:
                result = search_system.execute_nlp_aggregated_search(query)

                nlp_analysis = result.get('nlp_analysis', {})
                detected_intent = nlp_analysis.get('detected_intent', 'UNKNOWN')

                if any(expected in detected_intent for expected in expected_intents):
                    correct_detections += 1
                    logger.info(f"✓ Intent correct: '{query}' -> '{detected_intent}'")
                else:
                    logger.warning(
                        f"✗ Intent mismatch: '{query}' -> detected '{detected_intent}', expected one of {expected_intents}")

            except Exception as e:
                logger.error(f"Error testing intent for '{query}': {e}")

        accuracy = correct_detections / len(test_cases) * 100
        logger.info(f"Intent detection accuracy: {accuracy:.1f}% ({correct_detections}/{len(test_cases)})")

        # Should have at least 70% accuracy
        assert accuracy >= 70, f"Intent detection accuracy too low: {accuracy:.1f}%"


class TestSearchMethodFallbacks:
    """Test search method fallbacks and robustness."""

    def test_invalid_queries(self, search_system):
        """Test handling of invalid or edge case queries."""
        invalid_queries = [
            "",  # Empty query
            "   ",  # Whitespace only
            "find part",  # Incomplete query
            "what is the",  # Incomplete query
            "asdfghjkl",  # Random string
            "find part !@#$%",  # Special characters
        ]

        for query in invalid_queries:
            try:
                result = search_system.execute_nlp_aggregated_search(query)

                # Should not crash, and should return a valid response structure
                assert isinstance(result, dict), f"Invalid result type for query '{query}'"
                assert 'status' in result, f"Missing status in result for query '{query}'"

                logger.info(f"Invalid query handled: '{query}' -> {result.get('status')}")

            except Exception as e:
                # Should not raise exceptions for invalid queries
                pytest.fail(f"Search system crashed on invalid query '{query}': {e}")

    def test_aggregate_search_fallback(self, search_system, aggregate_search):
        """Test fallback to aggregate search if available."""
        if aggregate_search is None:
            pytest.skip("AggregateSearch not available for fallback testing")

        # Test both systems with same query
        test_query = "find part ABC123"

        try:
            nlp_result = search_system.execute_nlp_aggregated_search(test_query)
            agg_result = aggregate_search.execute_aggregated_search(test_query)

            # Both should return valid responses
            assert isinstance(nlp_result, dict)
            assert isinstance(agg_result, dict)

            logger.info(f"NLP search result: {nlp_result.get('status')}")
            logger.info(f"Aggregate search result: {agg_result.get('status')}")

        except Exception as e:
            logger.error(f"Fallback test failed: {e}")


# =====================================================
# CONFTEST.PY CONTENT
# =====================================================

CONFTEST_CONTENT = '''# conftest.py - Pytest configuration for search tests

import pytest
import logging
from datetime import datetime

def pytest_configure(config):
    """Configure pytest with custom logging."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

def pytest_sessionstart(session):
    """Called after the Session object has been created."""
    print(f"\\n🔍 Starting search system test suite at {datetime.now()}")

def pytest_sessionfinish(session, exitstatus):
    """Called after whole test run finished."""
    print(f"\\n✅ Search system test suite completed with exit status: {exitstatus}")

@pytest.fixture(autouse=True)
def log_test_name(request):
    """Automatically log test names."""
    test_name = request.node.name
    logging.info(f"Starting test: {test_name}")
    yield
    logging.info(f"Completed test: {test_name}")

def pytest_collection_modifyitems(config, items):
    """Add custom markers to tests."""
    for item in items:
        # Mark slow tests
        if "sample" in item.nodeid or "concurrent" in item.nodeid:
            item.add_marker(pytest.mark.slow)

        # Mark performance tests
        if "performance" in item.nodeid or "response_time" in item.nodeid:
            item.add_marker(pytest.mark.performance)
'''

# =====================================================
# USAGE AND SETUP INSTRUCTIONS
# =====================================================

SETUP_INSTRUCTIONS = '''
# =====================================================
# SEARCH SYSTEM TESTING SETUP INSTRUCTIONS
# =====================================================

## 1. Install Dependencies
pip install pytest pytest-html pytest-xdist pytest-cov

## 2. Create Test Directory Structure
mkdir -p tests/search
touch tests/__init__.py
touch tests/search/__init__.py

## 3. Save Files
# Save the main test file as: tests/search/test_search_comprehensive.py
# Save conftest.py as: tests/conftest.py

## 4. Create pytest.ini
cat > pytest.ini << 'EOF'
[tool:pytest]
testpaths = tests
python_files = test_*.py
python_classes = Test*
python_functions = test_*

markers =
    slow: marks tests as slow (deselect with '-m "not slow"')
    performance: marks performance tests
    unit: marks unit tests
    integration: marks integration tests

log_cli = true
log_cli_level = INFO
log_cli_format = %(asctime)s [%(levelname)8s] %(name)s: %(message)s

addopts = 
    --tb=short
    -v
    --strict-markers
    --durations=10
EOF

## 5. Run Tests

# Run all tests:
pytest tests/search/

# Run only fast tests:
pytest tests/search/ -m "not slow"

# Run with HTML report:
pytest tests/search/ --html=report.html --self-contained-html

# Run specific test class:
pytest tests/search/test_search_comprehensive.py::TestSearchSystemHealth

# Run with coverage:
pytest tests/search/ --cov=modules.search --cov-report=html

## 6. Analyze Results
open htmlcov/index.html  # Coverage report
open report.html         # Test report

## 7. Continuous Integration Example (.github/workflows/tests.yml)
name: Search Tests
on: [push, pull_request]
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v3
    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: '3.9'
    - name: Install dependencies
      run: |
        pip install -r requirements.txt
        pip install pytest pytest-html pytest-cov
    - name: Run tests
      run: pytest tests/search/ --junitxml=test-results.xml --cov=modules.search
    - name: Upload results
      uses: actions/upload-artifact@v3
      with:
        name: test-results
        path: |
          test-results.xml
          htmlcov/
'''

if __name__ == "__main__":
    print("Search System Testing Suite")
    print("=" * 50)
    print(SETUP_INSTRUCTIONS)
    print("\nConftest.py content:")
    print("=" * 30)
    print(CONFTEST_CONTENT)