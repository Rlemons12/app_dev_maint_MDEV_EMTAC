import json
import pandas as pd
from datetime import datetime
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt
import seaborn as sns
from dataclasses import dataclass, asdict
import numpy as np


@dataclass
class EntityMatch:
    """Track individual entity matching results."""
    expected: str
    predicted: List[str]
    exact_match: bool
    partial_match: bool
    confidence_score: float
    match_type: str  # 'exact', 'partial', 'none'


@dataclass
class QueryResult:
    """Track results for a single query test."""
    query_id: str
    row_index: int
    template_index: int
    query_text: str
    query_category: str  # 'single', 'double', 'triple' entity
    language_style: str  # 'formal', 'casual', 'contextual'

    # Entity results
    part_number_result: EntityMatch = None
    part_name_result: EntityMatch = None
    manufacturer_result: EntityMatch = None
    model_result: EntityMatch = None

    # Overall metrics
    total_entities_expected: int = 0
    total_entities_found: int = 0
    overall_success: bool = False
    execution_time_ms: float = 0.0


class PerformanceTracker:
    """Comprehensive performance tracking for NER testing with visualizations."""

    def __init__(self):
        self.results: List[QueryResult] = []
        self.start_time = datetime.now()
        self.pattern_performance = {}
        self.entity_confusion_matrix = {}

        # Set up plotting style
        plt.style.use('default')
        sns.set_palette("husl")

    def add_result(self, result: QueryResult):
        """Add a single query result."""
        self.results.append(result)

    def calculate_entity_metrics(self, entity_type: str) -> Dict:
        """Calculate precision, recall, F1 for specific entity type."""
        relevant_results = []

        for result in self.results:
            entity_result = getattr(result, f"{entity_type.lower()}_result")
            if entity_result:
                relevant_results.append(entity_result)

        if not relevant_results:
            return {'precision': 0, 'recall': 0, 'f1': 0, 'count': 0}

        exact_matches = sum(1 for r in relevant_results if r.exact_match)
        partial_matches = sum(1 for r in relevant_results if r.partial_match)
        total = len(relevant_results)

        # Use partial matches for recall (more lenient)
        recall = partial_matches / total if total > 0 else 0

        # Use exact matches for precision (more strict)
        predicted_entities = sum(1 for r in relevant_results if r.predicted)
        precision = exact_matches / predicted_entities if predicted_entities > 0 else 0

        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

        return {
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'exact_matches': exact_matches,
            'partial_matches': partial_matches,
            'total_tests': total,
            'exact_accuracy': exact_matches / total if total > 0 else 0,
            'partial_accuracy': partial_matches / total if total > 0 else 0
        }

    def calculate_pattern_performance(self) -> Dict:
        """Analyze performance by regex pattern categories."""
        pattern_stats = {}

        # Group by language style
        for style in ['formal', 'casual', 'contextual']:
            style_results = [r for r in self.results if r.language_style == style]
            if style_results:
                successful = sum(1 for r in style_results if r.overall_success)
                pattern_stats[f'language_{style}'] = {
                    'success_rate': successful / len(style_results),
                    'total_tests': len(style_results),
                    'successful': successful
                }

        # Group by query complexity
        for category in ['single', 'double', 'triple']:
            cat_results = [r for r in self.results if r.query_category == category]
            if cat_results:
                successful = sum(1 for r in cat_results if r.overall_success)
                pattern_stats[f'complexity_{category}'] = {
                    'success_rate': successful / len(cat_results),
                    'total_tests': len(cat_results),
                    'successful': successful
                }

        return pattern_stats

    def generate_performance_report(self) -> Dict:
        """Generate comprehensive performance report."""
        total_tests = len(self.results)
        successful_queries = sum(1 for r in self.results if r.overall_success)

        report = {
            'summary': {
                'total_tests': total_tests,
                'successful_queries': successful_queries,
                'overall_success_rate': successful_queries / total_tests if total_tests > 0 else 0,
                'average_execution_time_ms': sum(
                    r.execution_time_ms for r in self.results) / total_tests if total_tests > 0 else 0,
                'test_duration_minutes': (datetime.now() - self.start_time).total_seconds() / 60
            },
            'entity_performance': {},
            'pattern_performance': self.calculate_pattern_performance(),
            'top_failures': self.get_top_failures(),
            'confidence_analysis': self.analyze_confidence_scores()
        }

        # Calculate metrics for each entity type
        for entity_type in ['part_number', 'part_name', 'manufacturer', 'model']:
            report['entity_performance'][entity_type] = self.calculate_entity_metrics(entity_type)

        return report

    def get_top_failures(self, limit: int = 20) -> List[Dict]:
        """Get the most common failure patterns."""
        failures = []

        for result in self.results:
            if not result.overall_success:
                failure_info = {
                    'query': result.query_text,
                    'category': result.query_category,
                    'style': result.language_style,
                    'expected_entities': result.total_entities_expected,
                    'found_entities': result.total_entities_found,
                    'missing_entities': []
                }

                # Check which entities failed
                for entity_type in ['part_number', 'part_name', 'manufacturer', 'model']:
                    entity_result = getattr(result, f"{entity_type}_result")
                    if entity_result and not entity_result.partial_match:
                        failure_info['missing_entities'].append({
                            'type': entity_type,
                            'expected': entity_result.expected,
                            'predicted': entity_result.predicted
                        })

                failures.append(failure_info)

        return failures[:limit]

    def analyze_confidence_scores(self) -> Dict:
        """Analyze confidence scores for predictions."""
        all_scores = []
        correct_scores = []
        incorrect_scores = []

        for result in self.results:
            for entity_type in ['part_number', 'part_name', 'manufacturer', 'model']:
                entity_result = getattr(result, f"{entity_type}_result")
                if entity_result and entity_result.confidence_score > 0:
                    all_scores.append(entity_result.confidence_score)
                    if entity_result.exact_match:
                        correct_scores.append(entity_result.confidence_score)
                    else:
                        incorrect_scores.append(entity_result.confidence_score)

        return {
            'average_confidence': sum(all_scores) / len(all_scores) if all_scores else 0,
            'correct_predictions_confidence': sum(correct_scores) / len(correct_scores) if correct_scores else 0,
            'incorrect_predictions_confidence': sum(incorrect_scores) / len(
                incorrect_scores) if incorrect_scores else 0,
            'confidence_threshold_analysis': self.find_optimal_confidence_threshold(correct_scores, incorrect_scores)
        }

    def find_optimal_confidence_threshold(self, correct_scores: List[float], incorrect_scores: List[float]) -> Dict:
        """Find optimal confidence threshold for filtering predictions."""
        if not correct_scores or not incorrect_scores:
            return {}

        thresholds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
        best_threshold = 0.5
        best_f1 = 0

        threshold_analysis = {}

        for threshold in thresholds:
            tp = sum(1 for score in correct_scores if score >= threshold)
            fp = sum(1 for score in incorrect_scores if score >= threshold)
            fn = sum(1 for score in correct_scores if score < threshold)

            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

            threshold_analysis[threshold] = {
                'precision': precision,
                'recall': recall,
                'f1': f1
            }

            if f1 > best_f1:
                best_f1 = f1
                best_threshold = threshold

        return {
            'optimal_threshold': best_threshold,
            'optimal_f1': best_f1,
            'threshold_analysis': threshold_analysis
        }

    def create_visualizations(self, save_plots: bool = True):
        """Generate comprehensive visualizations of test results."""
        if not self.results:
            print("No results to visualize")
            return

        report = self.generate_performance_report()

        # Create figure with subplots
        fig = plt.figure(figsize=(20, 16))

        # 1. Entity Performance Bar Chart
        plt.subplot(2, 3, 1)
        entity_data = report['entity_performance']
        entities = list(entity_data.keys())
        exact_scores = [entity_data[e]['exact_accuracy'] for e in entities if entity_data[e]['total_tests'] > 0]
        partial_scores = [entity_data[e]['partial_accuracy'] for e in entities if entity_data[e]['total_tests'] > 0]

        x = np.arange(len(entities))
        width = 0.35

        plt.bar(x - width / 2, exact_scores, width, label='Exact Match', alpha=0.8)
        plt.bar(x + width / 2, partial_scores, width, label='Partial Match', alpha=0.8)
        plt.xlabel('Entity Type')
        plt.ylabel('Accuracy')
        plt.title('Entity Recognition Accuracy')
        plt.xticks(x, [e.replace('_', '\n').upper() for e in entities], rotation=45)
        plt.legend()
        plt.grid(axis='y', alpha=0.3)

        # 2. Language Style Performance
        plt.subplot(2, 3, 2)
        pattern_data = report['pattern_performance']
        language_styles = [k for k in pattern_data.keys() if k.startswith('language_')]
        language_rates = [pattern_data[k]['success_rate'] for k in language_styles]
        language_labels = [k.replace('language_', '').title() for k in language_styles]

        colors = sns.color_palette("husl", len(language_styles))
        plt.pie(language_rates, labels=language_labels, autopct='%1.1f%%', colors=colors)
        plt.title('Performance by Language Style')

        # 3. Query Complexity Performance
        plt.subplot(2, 3, 3)
        complexity_keys = [k for k in pattern_data.keys() if k.startswith('complexity_')]
        complexity_rates = [pattern_data[k]['success_rate'] for k in complexity_keys]
        complexity_labels = [k.replace('complexity_', '').title() + ' Entity' for k in complexity_keys]

        plt.bar(complexity_labels, complexity_rates, color=sns.color_palette("viridis", len(complexity_labels)))
        plt.xlabel('Query Complexity')
        plt.ylabel('Success Rate')
        plt.title('Performance by Query Complexity')
        plt.xticks(rotation=45)
        plt.grid(axis='y', alpha=0.3)

        # 4. Execution Time Distribution
        plt.subplot(2, 3, 4)
        execution_times = [r.execution_time_ms for r in self.results]
        plt.hist(execution_times, bins=30, alpha=0.7, color='skyblue', edgecolor='black')
        plt.xlabel('Execution Time (ms)')
        plt.ylabel('Frequency')
        plt.title('Query Execution Time Distribution')
        plt.axvline(np.mean(execution_times), color='red', linestyle='--',
                    label=f'Mean: {np.mean(execution_times):.1f}ms')
        plt.legend()
        plt.grid(alpha=0.3)

        # 5. Confidence Score Analysis
        plt.subplot(2, 3, 5)
        confidence_data = report['confidence_analysis']
        if 'confidence_threshold_analysis' in confidence_data and confidence_data['confidence_threshold_analysis']:
            threshold_analysis = confidence_data['confidence_threshold_analysis']['threshold_analysis']
            thresholds = list(threshold_analysis.keys())
            f1_scores = [threshold_analysis[t]['f1'] for t in thresholds]

            plt.plot(thresholds, f1_scores, marker='o', linewidth=2, markersize=8)
            plt.xlabel('Confidence Threshold')
            plt.ylabel('F1 Score')
            plt.title('F1 Score vs Confidence Threshold')
            plt.grid(alpha=0.3)

            # Mark optimal threshold
            optimal_threshold = confidence_data['confidence_threshold_analysis']['optimal_threshold']
            optimal_f1 = confidence_data['confidence_threshold_analysis']['optimal_f1']
            plt.axvline(optimal_threshold, color='red', linestyle='--',
                        label=f'Optimal: {optimal_threshold} (F1: {optimal_f1:.3f})')
            plt.legend()

        # 6. Success Rate Heatmap by Style and Complexity
        plt.subplot(2, 3, 6)

        # Create matrix for heatmap
        styles = ['formal', 'casual', 'contextual']
        complexities = ['single', 'double', 'triple']
        heatmap_data = np.zeros((len(styles), len(complexities)))

        for i, style in enumerate(styles):
            for j, complexity in enumerate(complexities):
                style_results = [r for r in self.results if
                                 r.language_style == style and r.query_category == complexity]
                if style_results:
                    success_rate = sum(1 for r in style_results if r.overall_success) / len(style_results)
                    heatmap_data[i, j] = success_rate

        sns.heatmap(heatmap_data,
                    xticklabels=[c.title() for c in complexities],
                    yticklabels=[s.title() for s in styles],
                    annot=True,
                    fmt='.2%',
                    cmap='RdYlGn',
                    center=0.8)
        plt.title('Success Rate: Style vs Complexity')

        plt.tight_layout()

        if save_plots:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            plt.savefig(f'ner_performance_analysis_{timestamp}.png', dpi=300, bbox_inches='tight')
            print(f"Visualizations saved to: ner_performance_analysis_{timestamp}.png")

        plt.show()

    def save_results(self, filename: str = None):
        """Save detailed results to JSON file."""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"ner_test_results_{timestamp}.json"

        # Convert results to serializable format
        serializable_results = []
        for result in self.results:
            result_dict = asdict(result)
            serializable_results.append(result_dict)

        data = {
            'test_metadata': {
                'start_time': self.start_time.isoformat(),
                'end_time': datetime.now().isoformat(),
                'total_results': len(self.results)
            },
            'performance_report': self.generate_performance_report(),
            'detailed_results': serializable_results
        }

        with open(filename, 'w') as f:
            json.dump(data, f, indent=2, default=str)

        print(f"Results saved to: {filename}")
        return filename

    def print_summary_report(self):
        """Print a concise summary report."""
        report = self.generate_performance_report()

        print("\n" + "=" * 80)
        print("NER MODEL PERFORMANCE SUMMARY")
        print("=" * 80)

        # Overall metrics
        print(f"Total Tests: {report['summary']['total_tests']}")
        print(f"Overall Success Rate: {report['summary']['overall_success_rate']:.2%}")
        print(f"Average Execution Time: {report['summary']['average_execution_time_ms']:.1f}ms")
        print(f"Test Duration: {report['summary']['test_duration_minutes']:.1f} minutes")

        # Entity performance
        print(f"\nEntity Performance:")
        print("-" * 50)
        for entity, metrics in report['entity_performance'].items():
            if metrics['total_tests'] > 0:
                print(f"{entity.upper():15} | "
                      f"Exact: {metrics['exact_accuracy']:.2%} "
                      f"Partial: {metrics['partial_accuracy']:.2%} "
                      f"F1: {metrics['f1']:.3f} "
                      f"({metrics['total_tests']} tests)")

        # Pattern performance
        print(f"\nPattern Performance:")
        print("-" * 50)
        for pattern, metrics in report['pattern_performance'].items():
            print(
                f"{pattern:20} | Success: {metrics['success_rate']:.2%} ({metrics['successful']}/{metrics['total_tests']})")

        # Confidence analysis
        conf = report['confidence_analysis']
        print(f"\nConfidence Analysis:")
        print("-" * 50)
        print(f"Average Confidence: {conf['average_confidence']:.3f}")
        print(f"Correct Predictions: {conf['correct_predictions_confidence']:.3f}")
        print(f"Incorrect Predictions: {conf['incorrect_predictions_confidence']:.3f}")

        if 'optimal_threshold' in conf['confidence_threshold_analysis']:
            opt = conf['confidence_threshold_analysis']
            print(f"Optimal Threshold: {opt['optimal_threshold']:.1f} (F1: {opt['optimal_f1']:.3f})")


# Test the enhanced tracker
if __name__ == "__main__":
    print("Enhanced Performance Tracker with Visualizations loaded successfully!")

    # Quick test
    tracker = PerformanceTracker()
    print(f"Tracker initialized at: {tracker.start_time}")
    print("Visualization capabilities enabled!")