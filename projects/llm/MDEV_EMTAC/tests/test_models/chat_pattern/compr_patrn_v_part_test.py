#!/usr/bin/env python3
"""
Comprehensive Pattern vs Part Tester

Tests EVERY pattern against EVERY part to generate complete coverage reports.
Shows which patterns can find which parts and overall success rates.
"""

import os
import re
import csv
import json
import logging
import sys
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from collections import defaultdict
import pandas as pd

# Simple path setup
current_dir = os.path.dirname(os.path.abspath(__file__))
base_dir = os.path.abspath(os.path.join(current_dir, '..', '..', '..'))
sys.path.insert(0, base_dir)

print(f"[SETUP] Added to path: {base_dir}")

# Direct imports
from modules.configuration.config_env import DatabaseConfig
from modules.search import SearchIntent, IntentPattern
from modules.emtacdb.emtacdb_fts import Part
from modules.configuration.config import DATABASE_URL
from sqlalchemy.orm import Session

print("[OK] All modules imported successfully!")
print(f"[OK] DATABASE_URL: {DATABASE_URL[:50]}...")


@dataclass
class PatternPartTest:
    """Result of testing one pattern against one part."""
    pattern_id: int
    pattern_text: str
    intent_name: str
    part_id: int
    part_number: str
    part_name: str
    test_queries: List[str]
    successful_queries: List[str]
    extracted_values: List[str]
    success_count: int
    total_queries: int
    success_rate: float
    best_extraction: Optional[str]
    exact_match: bool


class ComprehensivePatternTester:
    """Tests every pattern against every part."""

    def __init__(self):
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)

        # Database setup
        self.db_config = DatabaseConfig()
        self.session = self.db_config.get_main_session()

        self.patterns = []
        self.parts = []
        self.test_results = []

        print("[OK] Comprehensive Pattern Tester initialized")

    def load_patterns(self, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """Load all active patterns from database."""
        try:
            query = self.session.query(
                IntentPattern.id,
                IntentPattern.pattern_text,
                IntentPattern.pattern_type,
                IntentPattern.success_rate,
                IntentPattern.usage_count,
                SearchIntent.name.label('intent_name'),
                SearchIntent.display_name.label('intent_display_name')
            ).join(
                SearchIntent, IntentPattern.intent_id == SearchIntent.id
            ).filter(
                IntentPattern.is_active == True,
                SearchIntent.is_active == True
            ).order_by(SearchIntent.name, IntentPattern.success_rate.desc())

            if limit:
                query = query.limit(limit)

            patterns = []
            for row in query.all():
                try:
                    compiled_regex = re.compile(row.pattern_text, re.IGNORECASE)
                    pattern = {
                        'id': row.id,
                        'pattern_text': row.pattern_text,
                        'pattern_type': row.pattern_type,
                        'success_rate': row.success_rate or 0.0,
                        'usage_count': row.usage_count or 0,
                        'intent_name': row.intent_name,
                        'intent_display_name': row.intent_display_name or row.intent_name,
                        'compiled_regex': compiled_regex
                    }
                    patterns.append(pattern)
                except re.error as e:
                    self.logger.warning(f"Invalid regex pattern {row.id}: {e}")
                    continue

            self.patterns = patterns
            self.logger.info(f"Loaded {len(patterns)} active patterns")
            return patterns

        except Exception as e:
            self.logger.error(f"Error loading patterns: {e}")
            return []

    def load_parts(self, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """Load all parts from database."""
        try:
            query = self.session.query(Part).filter(Part.part_number.isnot(None))

            if limit:
                query = query.limit(limit)

            parts = []
            for part in query.all():
                part_data = {
                    'id': part.id,
                    'part_number': part.part_number or '',
                    'name': part.name or '',
                    'oem_mfg': part.oem_mfg or '',
                    'model': part.model or '',
                    'class_flag': part.class_flag or '',
                    'type': part.type or '',
                    'notes': part.notes or '',
                    'documentation': part.documentation or ''
                }
                parts.append(part_data)

            self.parts = parts
            self.logger.info(f"Loaded {len(parts)} parts for testing")
            return parts

        except Exception as e:
            self.logger.error(f"Error loading parts: {e}")
            return []

    def generate_test_queries_for_part(self, part_number: str) -> List[str]:
        """Generate comprehensive test queries for a part number."""
        queries = [
            # Basic find patterns
            f"find {part_number}",
            f"find part {part_number}",
            f"find this part {part_number}",
            f"find the part {part_number}",
            f"find part number {part_number}",
            f"find this part number {part_number}",
            f"find the part number {part_number}",

            # Search patterns
            f"search {part_number}",
            f"search for {part_number}",
            f"search part {part_number}",
            f"search for part {part_number}",

            # Show patterns
            f"show {part_number}",
            f"show me {part_number}",
            f"show part {part_number}",
            f"show me part {part_number}",

            # Get patterns
            f"get {part_number}",
            f"get part {part_number}",
            f"get me {part_number}",
            f"get me part {part_number}",

            # Lookup patterns
            f"lookup {part_number}",
            f"look up {part_number}",
            f"lookup part {part_number}",
            f"look up part {part_number}",

            # Conversational patterns
            f"I need {part_number}",
            f"I need part {part_number}",
            f"where is {part_number}",
            f"where is part {part_number}",
            f"help me find {part_number}",
            f"help me find part {part_number}",

            # Urgent patterns
            f"{part_number} urgent",
            f"{part_number} ASAP",
            f"need {part_number} asap",
            f"urgent {part_number}",

            # Simple patterns
            f"{part_number}",
            f"part {part_number}",
            f"part number {part_number}",

            # Case variations
            f"FIND PART {part_number.upper()}",
            f"find part {part_number.lower()}",

            # With spacing variations
            f"find   part   {part_number}",
            f"find part  {part_number}",
        ]

        return queries

    def test_pattern_against_part(self, pattern: Dict[str, Any], part: Dict[str, Any]) -> PatternPartTest:
        """Test a single pattern against a single part with multiple query variations."""

        part_number = part['part_number']
        test_queries = self.generate_test_queries_for_part(part_number)

        successful_queries = []
        extracted_values = []
        regex = pattern['compiled_regex']

        for query in test_queries:
            match = regex.search(query)
            if match:
                extracted = match.group(1) if match.groups() else match.group(0)
                successful_queries.append(query)
                extracted_values.append(extracted)

        # Determine best extraction and exact match
        exact_match = False
        best_extraction = None

        if extracted_values:
            # Find the best extraction (exact match preferred)
            for extracted in extracted_values:
                if extracted.upper() == part_number.upper():
                    best_extraction = extracted
                    exact_match = True
                    break

            # If no exact match, take the first extraction
            if not best_extraction:
                best_extraction = extracted_values[0]

        success_count = len(successful_queries)
        total_queries = len(test_queries)
        success_rate = (success_count / total_queries) * 100 if total_queries > 0 else 0

        return PatternPartTest(
            pattern_id=pattern['id'],
            pattern_text=pattern['pattern_text'],
            intent_name=pattern['intent_name'],
            part_id=part['id'],
            part_number=part_number,
            part_name=part['name'],
            test_queries=test_queries,
            successful_queries=successful_queries,
            extracted_values=extracted_values,
            success_count=success_count,
            total_queries=total_queries,
            success_rate=success_rate,
            best_extraction=best_extraction,
            exact_match=exact_match
        )

    def run_comprehensive_test(self, max_patterns: Optional[int] = None, max_parts: Optional[int] = None) -> List[
        PatternPartTest]:
        """Run comprehensive test of all patterns against all parts."""

        self.logger.info("Starting comprehensive pattern vs part testing...")

        # Load data
        patterns = self.load_patterns(limit=max_patterns)
        parts = self.load_parts(limit=max_parts)

        if not patterns:
            self.logger.error("No patterns loaded")
            return []

        if not parts:
            self.logger.error("No parts loaded")
            return []

        results = []
        total_tests = len(patterns) * len(parts)

        self.logger.info(f"Running {total_tests:,} pattern vs part tests...")
        print(f"Testing {len(patterns)} patterns against {len(parts)} parts...")

        # Test each pattern against each part
        for i, pattern in enumerate(patterns):
            pattern_results = []

            for j, part in enumerate(parts):
                result = self.test_pattern_against_part(pattern, part)
                pattern_results.append(result)
                results.append(result)

                # Progress logging
                current_test = i * len(parts) + j + 1
                if current_test % 100 == 0:
                    progress = (current_test / total_tests) * 100
                    self.logger.info(f"Progress: {progress:.1f}% ({current_test:,}/{total_tests:,})")

            # Log pattern summary
            exact_matches = sum(1 for r in pattern_results if r.exact_match)
            any_matches = sum(1 for r in pattern_results if r.success_count > 0)

            print(f"Pattern {pattern['id']:3d} [{pattern['intent_name']:12}]: "
                  f"{exact_matches:4d}/{len(parts)} exact matches, "
                  f"{any_matches:4d}/{len(parts)} any matches "
                  f"- {pattern['pattern_text'][:50]}...")

        self.test_results = results
        self.logger.info(f"Completed comprehensive testing. Total results: {len(results):,}")
        return results

    def generate_pattern_summary_report(self) -> pd.DataFrame:
        """Generate summary report by pattern."""

        pattern_summaries = []

        # Group results by pattern
        pattern_groups = defaultdict(list)
        for result in self.test_results:
            pattern_groups[result.pattern_id].append(result)

        for pattern_id, results in pattern_groups.items():
            if not results:
                continue

            first_result = results[0]

            # Calculate statistics
            total_parts = len(results)
            exact_matches = sum(1 for r in results if r.exact_match)
            any_matches = sum(1 for r in results if r.success_count > 0)
            avg_success_rate = sum(r.success_rate for r in results) / len(results)

            # Find best and worst performing parts
            best_parts = [r.part_number for r in results if r.exact_match][:5]
            worst_parts = [r.part_number for r in results if r.success_count == 0][:5]

            pattern_summaries.append({
                'pattern_id': pattern_id,
                'pattern_text': first_result.pattern_text,
                'intent_name': first_result.intent_name,
                'total_parts_tested': total_parts,
                'exact_matches': exact_matches,
                'exact_match_rate': (exact_matches / total_parts) * 100,
                'any_matches': any_matches,
                'any_match_rate': (any_matches / total_parts) * 100,
                'avg_query_success_rate': avg_success_rate,
                'sample_successful_parts': '; '.join(best_parts),
                'sample_failed_parts': '; '.join(worst_parts)
            })

        # Sort by exact match rate descending
        pattern_summaries.sort(key=lambda x: x['exact_match_rate'], reverse=True)

        return pd.DataFrame(pattern_summaries)

    def generate_part_summary_report(self) -> pd.DataFrame:
        """Generate summary report by part."""

        part_summaries = []

        # Group results by part
        part_groups = defaultdict(list)
        for result in self.test_results:
            part_groups[result.part_id].append(result)

        for part_id, results in part_groups.items():
            if not results:
                continue

            first_result = results[0]

            # Calculate statistics
            total_patterns = len(results)
            exact_matches = sum(1 for r in results if r.exact_match)
            any_matches = sum(1 for r in results if r.success_count > 0)
            avg_success_rate = sum(r.success_rate for r in results) / len(results)

            # Find best and worst performing patterns
            best_patterns = [f"{r.pattern_id}({r.intent_name})" for r in results if r.exact_match][:5]
            worst_patterns = [f"{r.pattern_id}({r.intent_name})" for r in results if r.success_count == 0][:5]

            part_summaries.append({
                'part_id': part_id,
                'part_number': first_result.part_number,
                'part_name': first_result.part_name,
                'total_patterns_tested': total_patterns,
                'exact_matches': exact_matches,
                'exact_match_rate': (exact_matches / total_patterns) * 100,
                'any_matches': any_matches,
                'any_match_rate': (any_matches / total_patterns) * 100,
                'avg_query_success_rate': avg_success_rate,
                'sample_successful_patterns': '; '.join(best_patterns),
                'sample_failed_patterns': '; '.join(worst_patterns)
            })

        # Sort by exact match rate descending
        part_summaries.sort(key=lambda x: x['exact_match_rate'], reverse=True)

        return pd.DataFrame(part_summaries)

    def generate_overall_summary(self) -> Dict[str, Any]:
        """Generate overall system summary statistics."""

        total_tests = len(self.test_results)
        if total_tests == 0:
            return {}

        exact_matches = sum(1 for r in self.test_results if r.exact_match)
        any_matches = sum(1 for r in self.test_results if r.success_count > 0)
        avg_success_rate = sum(r.success_rate for r in self.test_results) / total_tests

        # Pattern performance
        pattern_stats = defaultdict(list)
        for result in self.test_results:
            pattern_stats[result.pattern_id].append(result)

        best_patterns = []
        worst_patterns = []

        for pattern_id, results in pattern_stats.items():
            exact_rate = sum(1 for r in results if r.exact_match) / len(results) * 100
            pattern_text = results[0].pattern_text
            intent_name = results[0].intent_name

            if exact_rate > 80:
                best_patterns.append(f"Pattern {pattern_id} ({intent_name}): {exact_rate:.1f}%")
            elif exact_rate < 10:
                worst_patterns.append(f"Pattern {pattern_id} ({intent_name}): {exact_rate:.1f}%")

        return {
            'timestamp': datetime.now().isoformat(),
            'total_tests': total_tests,
            'total_patterns': len(self.patterns),
            'total_parts': len(self.parts),
            'exact_matches': exact_matches,
            'exact_match_rate': (exact_matches / total_tests) * 100,
            'any_matches': any_matches,
            'any_match_rate': (any_matches / total_tests) * 100,
            'avg_query_success_rate': avg_success_rate,
            'best_performing_patterns': best_patterns[:10],
            'worst_performing_patterns': worst_patterns[:10]
        }

    def save_detailed_results(self, filename: str = None) -> str:
        """Save detailed test results to CSV."""
        if not filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"comprehensive_pattern_test_detailed_{timestamp}.csv"

        with open(filename, 'w', newline='', encoding='utf-8') as csvfile:
            fieldnames = [
                'pattern_id', 'pattern_text', 'intent_name',
                'part_id', 'part_number', 'part_name',
                'success_count', 'total_queries', 'success_rate',
                'exact_match', 'best_extraction',
                'sample_successful_queries', 'sample_extracted_values'
            ]
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

            writer.writeheader()
            for result in self.test_results:
                writer.writerow({
                    'pattern_id': result.pattern_id,
                    'pattern_text': result.pattern_text,
                    'intent_name': result.intent_name,
                    'part_id': result.part_id,
                    'part_number': result.part_number,
                    'part_name': result.part_name,
                    'success_count': result.success_count,
                    'total_queries': result.total_queries,
                    'success_rate': round(result.success_rate, 2),
                    'exact_match': result.exact_match,
                    'best_extraction': result.best_extraction,
                    'sample_successful_queries': '; '.join(result.successful_queries[:3]),
                    'sample_extracted_values': '; '.join(result.extracted_values[:3])
                })

        self.logger.info(f"Detailed results saved to: {filename}")
        return filename

    def save_summary_reports(self, base_filename: str = None) -> Dict[str, str]:
        """Save all summary reports."""
        if not base_filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            base_filename = f"pattern_test_summary_{timestamp}"

        files = {}

        # Pattern summary
        pattern_df = self.generate_pattern_summary_report()
        pattern_file = f"{base_filename}_by_pattern.csv"
        pattern_df.to_csv(pattern_file, index=False)
        files['pattern_summary'] = pattern_file

        # Part summary
        part_df = self.generate_part_summary_report()
        part_file = f"{base_filename}_by_part.csv"
        part_df.to_csv(part_file, index=False)
        files['part_summary'] = part_file

        # Overall summary
        overall = self.generate_overall_summary()
        overall_file = f"{base_filename}_overall.json"
        with open(overall_file, 'w') as f:
            json.dump(overall, f, indent=2)
        files['overall_summary'] = overall_file

        self.logger.info(f"Summary reports saved: {list(files.values())}")
        return files

    def print_summary_report(self):
        """Print a comprehensive summary to console."""
        overall = self.generate_overall_summary()

        print("\n" + "=" * 80)
        print("COMPREHENSIVE PATTERN vs PART TEST RESULTS")
        print("=" * 80)
        print(f"Test completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Patterns tested: {overall['total_patterns']}")
        print(f"Parts tested: {overall['total_parts']}")
        print(f"Total test combinations: {overall['total_tests']:,}")

        print(f"\nOVERALL PERFORMANCE:")
        print(f"• Exact matches: {overall['exact_matches']:,} ({overall['exact_match_rate']:.1f}%)")
        print(f"• Any matches: {overall['any_matches']:,} ({overall['any_match_rate']:.1f}%)")
        print(f"• Average query success rate: {overall['avg_query_success_rate']:.1f}%")

        print(f"\nBEST PERFORMING PATTERNS:")
        for pattern in overall['best_performing_patterns'][:5]:
            print(f"  {pattern}")

        print(f"\nWORST PERFORMING PATTERNS:")
        for pattern in overall['worst_performing_patterns'][:5]:
            print(f"  {pattern}")

        # Find your specific pattern
        your_pattern = r"find\s+(?:this\s+|the\s+)?part\s+(?:number\s+)?([A-Za-z0-9\-\.]{3,})"
        your_results = [r for r in self.test_results if your_pattern in r.pattern_text]

        if your_results:
            exact_matches = sum(1 for r in your_results if r.exact_match)
            any_matches = sum(1 for r in your_results if r.success_count > 0)

            print(f"\nYOUR SPECIFIC PATTERN ANALYSIS:")
            print(f"• Pattern: {your_pattern}")
            print(
                f"• Exact matches: {exact_matches}/{len(your_results)} ({exact_matches / len(your_results) * 100:.1f}%)")
            print(f"• Any matches: {any_matches}/{len(your_results)} ({any_matches / len(your_results) * 100:.1f}%)")

            # Show some successful parts
            successful_parts = [r.part_number for r in your_results if r.exact_match][:10]
            if successful_parts:
                print(f"• Successfully finds: {', '.join(successful_parts)}")

        print("\n" + "=" * 80)

    def close(self):
        """Clean up database connection."""
        if self.session:
            self.session.close()


def main():
    """Main execution function."""
    print("Comprehensive Pattern vs Part Testing Framework")
    print("=" * 60)

    # Initialize tester
    tester = ComprehensivePatternTester()

    try:
        # Ask user for limits to avoid overwhelming system
        print("\nFor initial testing, we recommend limiting the scope:")
        print("• Full test: All patterns (~112) vs All parts (~thousands) = Very long")
        print("• Limited test: 10 patterns vs 50 parts = Quick demo")

        choice = input("\nRun [F]ull test or [L]imited test? (F/L): ").upper()

        if choice == 'L':
            max_patterns = 10
            max_parts = 50
            print(f"Running limited test: {max_patterns} patterns vs {max_parts} parts")
        else:
            max_patterns = None
            max_parts = None
            print("Running FULL comprehensive test - this may take a while...")

        # Run comprehensive test
        results = tester.run_comprehensive_test(
            max_patterns=max_patterns,
            max_parts=max_parts
        )

        if results:
            # Generate and save reports
            detailed_file = tester.save_detailed_results()
            summary_files = tester.save_summary_reports()

            # Print summary to console
            tester.print_summary_report()

            print(f"\nFiles created:")
            print(f"• Detailed results: {detailed_file}")
            for report_type, filename in summary_files.items():
                print(f"• {report_type}: {filename}")

        else:
            print("No test results generated. Check database connection and data.")

    except KeyboardInterrupt:
        print("\nTest interrupted by user")
    except Exception as e:
        print(f"Error during testing: {e}")
        import traceback
        traceback.print_exc()
    finally:
        tester.close()


if __name__ == "__main__":
    main()