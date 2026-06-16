import os
import re
import time
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass

import pandas as pd
from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline
from sqlalchemy import text

# Project imports
from modules.emtac_ai.training_scripts.performance_tst_model.performance_tracker import (
    PerformanceTracker, QueryResult, EntityMatch
)
from modules.configuration.config import (
    ORC_PARTS_MODEL_DIR,                       # …/modules/emtac_ai/models/parts
    ORC_TRAINING_DATA_PARTS_LOADSHEET_PATH,    # …/training_data/loadsheet/parts_loadsheet.xlsx
)
from modules.configuration.config_env import DatabaseConfig
from modules.configuration.log_config import (
    TrainingLogManager, set_request_id, info_id, warning_id, error_id
)

# Try to use ORM helper if present; otherwise fallback to raw SQL
try:
    # If your models expose QueryTemplate.get_active_texts(session, intent_name)
    from modules.emtac_ai.emtac_ai_db_models import QueryTemplate  # noqa: F401
    _HAS_QT_MODEL = True
except Exception:
    _HAS_QT_MODEL = False


# ------------------------------ DB templates -------------------------------
def load_templates_from_db(intent_name: str = "parts") -> List[str]:
    """
    Fetch active query templates for the given intent from DB.
    Returns a de-duplicated, stripped list. Empty list if none / error.
    """
    templates: List[str] = []
    try:
        db = DatabaseConfig()
        with db.main_session() as s:
            if _HAS_QT_MODEL:
                # Preferred: use your model helper
                templates = QueryTemplate.get_active_texts(s, intent_name)  # type: ignore[attr-defined]
            else:
                # Fallback: raw SQL
                sql = text("""
                    SELECT qt.template_text
                    FROM query_template qt
                    JOIN intent i ON qt.intent_id = i.id
                    WHERE i.name = :intent
                      AND qt.is_active = TRUE
                    ORDER BY COALESCE(qt.display_order, qt.id)
                """)
                templates = [r[0] for r in s.execute(sql, {"intent": intent_name}).fetchall()]
    except Exception as e:
        warning_id(f"[Templates] DB load failed for intent='{intent_name}': {e}")

    # normalize/dedupe
    clean: List[str] = []
    seen = set()
    for t in templates or []:
        t2 = (t or "").strip()
        if t2 and t2 not in seen:
            seen.add(t2)
            clean.append(t2)
    return clean


# ------------------------------ Model resolving -----------------------------
RUN_RE = re.compile(r"^\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2}_run-\d{3}$")

def resolve_model_dir(base_dir: Path, prefer_deployed: bool = True) -> Path:
    """
    Resolve model directory in this order:
      0) env PARTS_NER_MODEL_DIR (if set)
      1) base/DEPLOYED.txt -> <run>/best (pinned)
      2) base/LATEST.txt   -> <run>/best (most recent training)
      3) newest timestamped <run>/best
      4) base/best
      5) base
    """
    # 0) explicit override
    override = os.getenv("PARTS_NER_MODEL_DIR")
    if override:
        p = Path(override)
        return (p / "best") if (p / "best").is_dir() else p

    # 1) pinned deployment
    if prefer_deployed and (base_dir / "DEPLOYED.txt").exists():
        name = (base_dir / "DEPLOYED.txt").read_text(encoding="utf-8").strip()
        cand = base_dir / name
        if (cand / "best").is_dir():
            return cand / "best"

    # 2) latest
    if (base_dir / "LATEST.txt").exists():
        name = (base_dir / "LATEST.txt").read_text(encoding="utf-8").strip()
        cand = base_dir / name
        if (cand / "best").is_dir():
            return cand / "best"

    # 3) newest timestamped run
    runs = [p for p in base_dir.iterdir() if p.is_dir() and RUN_RE.match(p.name)]
    runs.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    for r in runs:
        if (r / "best").is_dir():
            return r / "best"

    # 4) non-versioned best/
    if (base_dir / "best").is_dir():
        return base_dir / "best"

    # 5) last resort
    return base_dir


# ----------------------------- Data structures ------------------------------
@dataclass
class EntitySpan:
    start: int
    end: int
    label: str
    text: str
    confidence: float = 0.0

    def __post_init__(self):
        if self.label.startswith(('B-', 'I-')):
            self.label = self.label[2:]
        self.normalized_text = self.normalize_text(self.text)

    @staticmethod
    def normalize_text(text: str) -> str:
        if not text:
            return ""
        normalized = text.lower().strip()
        normalized = re.sub(r'\s+', ' ', normalized)
        normalized = re.sub(r'[.,;:!?"\'-]', '', normalized)
        return normalized

    def matches(self, other: 'EntitySpan', match_type: str = "exact") -> bool:
        if self.label != other.label:
            return False
        if match_type == "exact":
            return (self.start == other.start and
                    self.end == other.end and
                    self.normalized_text == other.normalized_text)
        elif match_type == "partial":
            span_overlap = not (self.end <= other.start or self.start >= other.end)
            text_match = self.normalized_text == other.normalized_text
            return span_overlap and text_match
        elif match_type == "text_only":
            return self.normalized_text == other.normalized_text
        return False

    def overlap_ratio(self, other: 'EntitySpan') -> float:
        if self.label != other.label:
            return 0.0
        overlap_start = max(self.start, other.start)
        overlap_end = min(self.end, other.end)
        if overlap_start >= overlap_end:
            return 0.0
        overlap_length = overlap_end - overlap_start
        total_length = max(self.end - self.start, other.end - other.start)
        return overlap_length / total_length if total_length > 0 else 0.0


# ------------------------------ Evaluator -----------------------------------
class ImprovedNEREvaluator:
    def __init__(self):
        self.manufacturer_aliases = {
            'balston': ['balston filt', 'balston filter', 'balston filtration'],
            'parker': ['parker hannifin', 'parker hnnfn'],
            'smc': ['smc corporation', 'smc corp'],
        }
        self.part_number_patterns = [
            re.compile(r'\bA1\d{5}\b', re.IGNORECASE),
            re.compile(r'\b\d{3}-\d{2}-[A-Z]{2}\b', re.IGNORECASE),
        ]

    def normalize_manufacturer(self, manufacturer: str) -> str:
        if not manufacturer:
            return ""
        normalized = manufacturer.lower().strip()
        for canonical, aliases in self.manufacturer_aliases.items():
            if normalized == canonical or normalized in aliases:
                return canonical
        return normalized

    def extract_entities_from_prediction(self, predicted_entities: List[Dict]) -> List[EntitySpan]:
        entities = []
        for ent in predicted_entities:
            if 'entity_group' in ent:
                label = ent['entity_group']
                text = ent.get('word', ent.get('text', '')).replace('##', '')
                confidence = ent.get('score', 0.0)
                start = ent.get('start', 0)
                end = ent.get('end', 0)
            else:
                label = ent.get('label', ent.get('entity_group', ''))
                text = ent.get('text', ent.get('word', ''))
                confidence = ent.get('score', ent.get('confidence', 0.0))
                start = ent.get('start', 0)
                end = ent.get('end', 0)

            span = EntitySpan(start=start, end=end, label=label, text=text, confidence=confidence)
            if span.label == 'MANUFACTURER':
                span.normalized_text = self.normalize_manufacturer(span.text)
            entities.append(span)
        return entities

    def extract_entities_from_expected(self, expected_entities: Dict[str, str]) -> List[EntitySpan]:
        entities = []
        for label, text in expected_entities.items():
            if not text or pd.isna(text):
                continue
            span = EntitySpan(start=0, end=len(str(text)), label=label, text=str(text), confidence=1.0)
            if span.label == 'MANUFACTURER':
                span.normalized_text = self.normalize_manufacturer(span.text)
            entities.append(span)
        return entities

    def find_best_matches(
        self, predicted: List[EntitySpan], expected: List[EntitySpan]
    ) -> Tuple[List[Tuple[EntitySpan, EntitySpan]], List[EntitySpan], List[EntitySpan]]:
        matches = []
        unmatched_predicted = list(predicted)
        unmatched_expected = list(expected)

        # text-only first
        for pred in predicted[:]:
            for exp in expected[:]:
                if pred.matches(exp, "text_only"):
                    matches.append((pred, exp))
                    if pred in unmatched_predicted:
                        unmatched_predicted.remove(pred)
                    if exp in unmatched_expected:
                        unmatched_expected.remove(exp)
                    break

        # partial text similarity next
        for pred in unmatched_predicted[:]:
            best_match = None
            best_score = 0.0
            for exp in unmatched_expected:
                if pred.label == exp.label:
                    pred_words = set(pred.normalized_text.split())
                    exp_words = set(exp.normalized_text.split())
                    if pred_words and exp_words:
                        inter = pred_words & exp_words
                        union = pred_words | exp_words
                        jaccard = len(inter) / len(union)
                        substring = 0.8 if (pred.normalized_text in exp.normalized_text
                                            or exp.normalized_text in pred.normalized_text) else 0.0
                        combined = max(jaccard, substring)
                        if combined > 0.5 and combined > best_score:
                            best_match = exp
                            best_score = combined
            if best_match:
                matches.append((pred, best_match))
                unmatched_predicted.remove(pred)
                unmatched_expected.remove(best_match)

        return matches, unmatched_predicted, unmatched_expected

    def evaluate_prediction(self, predicted_entities: List[Dict], expected_entities: Dict[str, str]) -> Dict:
        pred_spans = self.extract_entities_from_prediction(predicted_entities)
        exp_spans = self.extract_entities_from_expected(expected_entities)

        matches, false_positives, false_negatives = self.find_best_matches(pred_spans, exp_spans)
        tp = len(matches)
        fp = len(false_positives)
        fn = len(false_negatives)

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

        entity_metrics = {}
        for entity_type in ['PART_NUMBER', 'PART_NAME', 'MANUFACTURER', 'MODEL']:
            pred_count = sum(1 for s in pred_spans if s.label == entity_type)
            exp_count = sum(1 for s in exp_spans if s.label == entity_type)
            match_count = sum(1 for p, e in matches if p.label == entity_type)
            ent_precision = match_count / pred_count if pred_count > 0 else 0.0
            ent_recall = match_count / exp_count if exp_count > 0 else 0.0
            ent_f1 = 2 * ent_precision * ent_recall / (ent_precision + ent_recall) if (ent_precision + ent_recall) > 0 else 0.0
            entity_metrics[entity_type] = {
                'precision': ent_precision,
                'recall': ent_recall,
                'f1': ent_f1,
                'predicted': pred_count,
                'expected': exp_count,
                'matched': match_count
            }

        return {
            'overall': {
                'precision': precision,
                'recall': recall,
                'f1': f1,
                'true_positives': tp,
                'false_positives': fp,
                'false_negatives': fn,
                'exact_match': (fp == 0 and fn == 0)
            },
            'entities': entity_metrics,
            'matches': matches,
            'false_positives': false_positives,
            'false_negatives': false_negatives
        }


# ------------------------------ Tester --------------------------------------
class ComprehensiveNERTester:
    """Main class for running comprehensive NER tests."""

    def __init__(self, excel_path: str, model_path: str, intent_name: str = "parts"):
        self.excel_path = excel_path
        self.model_path = model_path
        self.intent_name = intent_name

        # logging (training logger)
        self.req_id = set_request_id()
        self.tlogm = TrainingLogManager(run_dir=None, run_name=f"comprehensive_test_{intent_name}")
        self.logger = self.tlogm.logger
        info_id(f"[ComprehensiveTest] Initialized for intent='{intent_name}'", self.req_id)

        self.nlp = None
        self.tracker = PerformanceTracker()
        self.evaluator = ImprovedNEREvaluator()
        self.bad_templates = []   # <--- ADD THIS: collects (row_idx, template_idx, template, error)

        # DB-backed templates
        self.templates = load_templates_from_db(intent_name=self.intent_name)

    def load_model(self):
        """Load the trained NER model."""
        info_id(f"Loading NER model from checkpoint: {self.model_path}", self.req_id)
        try:
            tokenizer = AutoTokenizer.from_pretrained(self.model_path)
            model = AutoModelForTokenClassification.from_pretrained(self.model_path)
            self.nlp = pipeline(
                "token-classification",
                model=model,
                tokenizer=tokenizer,
                aggregation_strategy="simple",
                device=-1  # CPU
            )
            info_id("Model loaded successfully!", self.req_id)
            return True
        except Exception as e:
            error_id(f"Failed to load model: {e}", self.req_id)
            return False

    def load_inventory_data(self, max_rows: int = None) -> Optional[pd.DataFrame]:
        """Load inventory data from Excel."""
        info_id(f"Loading inventory data from {self.excel_path}", self.req_id)
        try:
            df = pd.read_excel(self.excel_path)
            df.columns = [str(c).strip() for c in df.columns]
            if max_rows and len(df) > max_rows:
                df = df.head(max_rows)
                info_id(f"Limited to first {max_rows} rows for testing", self.req_id)
            info_id(f"Loaded {len(df)} inventory rows", self.req_id)
            return df
        except Exception as e:
            error_id(f"Failed to load inventory data: {e}", self.req_id)
            return None

    @staticmethod
    def normalize_text(text: str) -> str:
        if not text or pd.isna(text):
            return ""
        return str(text).strip()

    def extract_entities_from_prediction(self, results: List[Dict]) -> Dict[str, List[str]]:
        """(Kept for compatibility if needed elsewhere.)"""
        entities = {'PART_NAME': [], 'MANUFACTURER': [], 'PART_NUMBER': [], 'MODEL': []}
        for result in results:
            entity_type = result['entity_group'].replace('B-', '').replace('I-', '')
            if entity_type in entities:
                word = result.get('word', result.get('text', '')).replace('##', '')
                entities[entity_type].append(word)
        for entity_type in entities:
            if entities[entity_type]:
                entities[entity_type] = [' '.join(entities[entity_type])]
        return entities

    def check_entity_match(self, predicted: List[str], expected: str) -> EntityMatch:
        expected_norm = self.normalize_text(expected)
        if not expected_norm:
            return EntityMatch(expected=expected, predicted=predicted, exact_match=True,
                               partial_match=True, confidence_score=1.0, match_type='exact')
        predicted_entities = [{
            'entity_group': 'TEMP',
            'word': p,
            'score': 1.0,
            'start': 0,
            'end': len(p)
        } for p in predicted]
        expected_entities = {'TEMP': expected}
        eval_result = self.evaluator.evaluate_prediction(predicted_entities, expected_entities)
        exact_match = eval_result['overall']['exact_match']
        partial_match = eval_result['overall']['f1'] > 0.5
        match_type = 'exact' if exact_match else ('partial' if partial_match else 'none')
        confidence = eval_result['overall']['f1']
        return EntityMatch(expected=expected, predicted=predicted, exact_match=exact_match,
                           partial_match=partial_match, confidence_score=confidence, match_type=match_type)

    def generate_test_query(self, row: pd.Series, template: str) -> Tuple[str, Dict, str, str]:
        description = str(row.get('DESCRIPTION', '')).strip()
        manufacturer = str(row.get('OEMMFG', '')).strip()
        itemnum = str(row.get('ITEMNUM', '')).strip()
        model = str(row.get('MODEL', '')).strip()
        query = template.format(
            description=description.lower(),
            manufacturer=manufacturer.lower(),
            itemnum=itemnum,
            model=model
        )
        expected = {}
        if '{itemnum}' in template:
            expected['PART_NUMBER'] = itemnum
        if '{description}' in template:
            expected['PART_NAME'] = description
        if '{manufacturer}' in template:
            expected['MANUFACTURER'] = manufacturer
        if '{model}' in template:
            expected['MODEL'] = model
        entity_count = len(expected)
        category = 'single' if entity_count == 1 else ('double' if entity_count == 2 else 'triple')
        if any(word in template.lower() for word in ['need', 'require', 'part number']):
            style = 'formal'
        elif any(word in template.lower() for word in ['any', 'some', 'stuff']):
            style = 'casual'
        else:
            style = 'contextual'
        return query, expected, category, style

    def test_single_query(self, row_idx: int, template_idx: int, row: pd.Series, template: str) -> QueryResult:
        # Try to build the query from the template. If it fails (bad braces/keys), record and skip gracefully.
        try:
            query, expected, category, style = self.generate_test_query(row, template)
        except Exception as e:
            error_id(f"Template format failed for row={row_idx}, template_idx={template_idx}: {e}", self.req_id)
            # keep a copy for end-of-run summary
            self.bad_templates.append((row_idx, template_idx, template, str(e)))
            # Return a minimal "skipped" result so the tracker can continue
            return QueryResult(
                query_id=f"row_{row_idx}_template_{template_idx}",
                row_index=row_idx,
                template_index=template_idx,
                query_text=f"[SKIPPED due to template error] {template}",
                query_category="n/a",
                language_style="skipped",
                total_entities_expected=0,
                total_entities_found=0,
                overall_success=False,
                execution_time_ms=0.0
            )

        # Normal path
        query_id = f"row_{row_idx}_template_{template_idx}"
        start_time = time.time()
        try:
            predictions = self.nlp(query)
            execution_time = (time.time() - start_time) * 1000
            eval_result = self.evaluator.evaluate_prediction(predictions, expected)
            result = QueryResult(
                query_id=query_id,
                row_index=row_idx,
                template_index=template_idx,
                query_text=query,
                query_category=category,
                language_style=style,
                total_entities_expected=len(expected),
                total_entities_found=len([m for m in eval_result['matches']]),
                execution_time_ms=execution_time
            )
            matched_entities = {match[1].label: match[0] for match in eval_result['matches']}
            for entity_type, expected_value in expected.items():
                if entity_type in matched_entities:
                    mr = EntityMatch(
                        expected=expected_value,
                        predicted=[matched_entities[entity_type].text],
                        exact_match=True, partial_match=True,
                        confidence_score=matched_entities[entity_type].confidence,
                        match_type='exact'
                    )
                else:
                    mr = EntityMatch(
                        expected=expected_value, predicted=[],
                        exact_match=False, partial_match=False,
                        confidence_score=0.0, match_type='none'
                    )
                if entity_type == 'PART_NUMBER':
                    result.part_number_result = mr
                elif entity_type == 'PART_NAME':
                    result.part_name_result = mr
                elif entity_type == 'MANUFACTURER':
                    result.manufacturer_result = mr
                elif entity_type == 'MODEL':
                    result.model_result = mr
            result.overall_success = eval_result['overall']['f1'] >= 0.8
            return result
        except Exception as e:
            execution_time = (time.time() - start_time) * 1000
            error_id(f"Query failed for {query_id}: {e}", self.req_id)
            return QueryResult(
                query_id=query_id, row_index=row_idx, template_index=template_idx,
                query_text=query, query_category=category, language_style=style,
                total_entities_expected=len(expected), total_entities_found=0,
                overall_success=False, execution_time_ms=execution_time
            )

    def run_comprehensive_test(self, max_rows: int = 50):
        info_id("=" * 80, self.req_id)
        info_id("STARTING COMPREHENSIVE NER MODEL TEST", self.req_id)
        info_id("=" * 80, self.req_id)

        if not self.load_model():
            return None
        df = self.load_inventory_data(max_rows)
        if df is None:
            return None

        if not self.templates:
            warning_id("No templates available; aborting test.", self.req_id)
            return None

        total_tests = len(df) * len(self.templates)
        info_id(f"Will run {total_tests} total tests ({len(df)} rows × {len(self.templates)} templates)", self.req_id)
        completed_tests = 0

        for row_idx, row in df.iterrows():
            itemnum = row.get('ITEMNUM', 'Unknown')
            info_id(f"Testing row {row_idx + 1}/{len(df)}: {itemnum}", self.req_id)
            for template_idx, template in enumerate(self.templates):
                result = self.test_single_query(row_idx, template_idx, row, template)
                self.tracker.add_result(result)
                completed_tests += 1
                if completed_tests % 100 == 0:
                    info_id(f"  Progress: {completed_tests}/{total_tests} tests completed", self.req_id)

        info_id(f"Completed all {completed_tests} tests!", self.req_id)
        return self.tracker

    def save_and_report(self, output_file: str = None):
        info_id("=" * 80, self.req_id)
        info_id("GENERATING PERFORMANCE REPORT", self.req_id)
        info_id("=" * 80, self.req_id)

        self.tracker.print_summary_report()
        saved_file = self.tracker.save_results(output_file)

        # ---- NEW: End-of-run bad-template summary ----
        if self.bad_templates:
            info_id(f"Bad templates encountered: {len(self.bad_templates)}", self.req_id)
            # De-duplicate identical (template, error) pairs to avoid noise
            seen = set()
            condensed = []
            for row_idx, tpl_idx, tpl, err in self.bad_templates:
                key = (tpl, err)
                if key not in seen:
                    seen.add(key)
                    condensed.append((row_idx, tpl_idx, tpl, err))

            # Log a concise list
            info_id("Listing unique bad templates (first 50 shown):", self.req_id)
            for i, (row_idx, tpl_idx, tpl, err) in enumerate(condensed[:50], 1):
                warning_id(f"[#{i}] row={row_idx} template_idx={tpl_idx} | error={err} | template={tpl}", self.req_id)

            # Also print to console so you see it immediately
            print("\n=== Bad Templates Summary ===")
            print(f"Total bad instances: {len(self.bad_templates)} | Unique: {len(condensed)}")
            for i, (row_idx, tpl_idx, tpl, err) in enumerate(condensed[:50], 1):
                print(f"[#{i}] row={row_idx} template_idx={tpl_idx}")
                print(f"      error: {err}")
                print(f"      template: {tpl}\n")
            if len(condensed) > 50:
                print(f"... and {len(condensed) - 50} more")

        return saved_file


# --------------------------------- Main -------------------------------------
def main():
    # Canonical inputs from config
    excel_path = ORC_TRAINING_DATA_PARTS_LOADSHEET_PATH  # authoritative parts loadsheet
    base_parts_dir = Path(ORC_PARTS_MODEL_DIR)

    # Resolve current model dir (DEPLOYED → LATEST → newest → best → base)
    model_dir = resolve_model_dir(base_parts_dir, prefer_deployed=True)

    # Run with training logger context for clean open/close
    req_id = set_request_id()
    with TrainingLogManager(run_dir=None, run_name="comprehensive_parts_test") as _tlog:
        info_id("Comprehensive NER Model Testing", req_id)
        info_id(f"Excel file: {excel_path}", req_id)
        info_id(f"Model path: {model_dir}", req_id)
        info_id(f"Resolved checkpoint directory: {model_dir}", req_id)

        # Get user input
        try:
            max_rows = int(input("How many inventory rows to test? (default 50): ") or "50")
        except ValueError:
            max_rows = 50
            warning_id("Invalid input. Using default of 50 rows.", req_id)

        tester = ComprehensiveNERTester(
            excel_path=str(excel_path),
            model_path=str(model_dir),
            intent_name="parts"
        )
        tracker = tester.run_comprehensive_test(max_rows)

        if tracker:
            output_file = f"ner_comprehensive_test_{max_rows}rows.json"
            saved_path = tester.save_and_report(output_file)
            info_id(f"Testing complete! Results saved to {saved_path}", req_id)
            info_id(f"Total tests run: {len(tracker.results)}", req_id)
            successful = sum(1 for r in tracker.results if r.overall_success)
            success_rate = successful / len(tracker.results) if tracker.results else 0
            info_id(f"Overall success rate: {success_rate:.2%}", req_id)
            print(f"\nTesting complete! Results saved to {saved_path}")
            print(f"Total tests run: {len(tracker.results)}")
            print(f"Overall success rate: {success_rate:.2%}")
        else:
            error_id("Testing failed", req_id)
            print("Testing failed")


if __name__ == "__main__":
    main()
