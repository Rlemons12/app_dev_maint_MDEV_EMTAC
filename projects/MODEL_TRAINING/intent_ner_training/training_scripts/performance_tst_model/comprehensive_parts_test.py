# ================================================================
# COMPREHENSIVE PARTS NER TEST (DROP-IN READY)
# ================================================================
# - Offline-safe
# - Uses trained Parts NER models
# - DB-backed query templates
# - Inventory Excel validation
# - Span-level precision/recall/F1
# ================================================================

import os
import re
import time
from pathlib import Path
from typing import Dict, List
from dataclasses import dataclass

import pandas as pd
from transformers import (
    AutoTokenizer,
    AutoModelForTokenClassification,
    pipeline,
)
from sqlalchemy import text

# ------------------------------------------------
# EMTAC imports (ALIGNED)
# ------------------------------------------------
from intent_ner_training.training_scripts.performance_tst_model.performance_tracker import (
    PerformanceTracker,
    QueryResult,
)

from configuration.config import (
    MODEL_TRAINING_PARTS_MODEL_DIR,
    MODEL_TRAINING_TRAINING_DATA_PARTS_LOADSHEET_PATH,
)

from configuration.config_env import TrainingDatabaseConfig
from configuration.log_config import (
    TrainingLogManager,
    set_request_id,
    info_id,
    warning_id,
)

try:
    from modules.emtac_ai.emtac_ai_db_models import QueryTemplate
    _HAS_QT_MODEL = True
except Exception:
    _HAS_QT_MODEL = False

from modules.gpu.gpu_training_adapter import GPUTrainingAdapter

# ================================================================
# MODEL RESOLUTION (OFFLINE-SAFE)
# ================================================================
RUN_RE = re.compile(r"^\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2}_run-\d{3}$")


def safe_format_template(tpl: str, values: Dict[str, str]) -> str:
    """
    Safely format templates containing arbitrary braces.
    Only known placeholders are substituted.
    """
    # Escape all braces
    s = tpl.replace("{", "{{").replace("}", "}}")

    # Re-enable known placeholders
    for k, v in values.items():
        s = s.replace(f"{{{{{k}}}}}", v)

    return s

def resolve_model_dir(base_dir: Path) -> Path:
    override = os.getenv("PARTS_NER_MODEL_DIR")
    if override:
        p = Path(override)
        return p / "best" if (p / "best").is_dir() else p

    for marker in ("DEPLOYED.txt", "LATEST.txt"):
        if (base_dir / marker).exists():
            name = (base_dir / marker).read_text().strip()
            p = base_dir / name / "best"
            if p.is_dir():
                return p

    runs = sorted(
        [d for d in base_dir.iterdir() if d.is_dir() and RUN_RE.match(d.name)],
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    for r in runs:
        if (r / "best").is_dir():
            return r / "best"

    if (base_dir / "best").is_dir():
        return base_dir / "best"

    raise RuntimeError(f"No valid Parts NER model found in {base_dir}")


# ================================================================
# DB QUERY TEMPLATES
# ================================================================
def load_templates_from_db(intent_name: str = "parts") -> List[str]:
    templates: List[str] = []

    try:
        db = TrainingDatabaseConfig()
        with db.get_session() as s:
            if _HAS_QT_MODEL:
                templates = QueryTemplate.get_active_texts(s, intent_name)
            else:
                rows = s.execute(text("""
                    SELECT qt.template_text
                    FROM query_template qt
                    JOIN intent i ON qt.intent_id = i.id
                    WHERE i.name = :intent
                      AND qt.is_active = TRUE
                    ORDER BY qt.id
                """), {"intent": intent_name}).fetchall()
                templates = [r[0] for r in rows]

    except Exception as e:
        warning_id(f"[Templates] Failed to load templates: {e}")

    if not templates:
        warning_id("[Templates] No templates loaded — evaluation skipped")

    return list(dict.fromkeys(t.strip() for t in templates if t))


# ================================================================
# ENTITY STRUCTURE
# ================================================================
@dataclass
class EntitySpan:
    start: int
    end: int
    label: str
    text: str
    confidence: float

    def __post_init__(self):
        if self.label.startswith(("B-", "I-")):
            self.label = self.label[2:]
        self.norm = re.sub(r"[^\w]+", " ", self.text.lower()).strip()


# ================================================================
# EVALUATOR
# ================================================================
class PartsNEREvaluator:
    def evaluate(self, preds: List[Dict], expected: Dict[str, str]) -> Dict:
        pred_spans = [
            EntitySpan(
                start=p.get("start", 0),
                end=p.get("end", 0),
                label=(p.get("entity_group") or p.get("entity") or ""),
                text=p.get("word", p.get("text", "")),
                confidence=p.get("score", 0.0),
            )
            for p in preds
        ]

        matches = {}
        for label, value in expected.items():
            norm = re.sub(r"[^\w]+", " ", value.lower()).strip()
            for p in pred_spans:
                if p.label == label and p.norm == norm:
                    matches[label] = p

        tp = len(matches)
        fp = len(pred_spans) - tp
        fn = len(expected) - tp

        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0

        return {"tp": tp, "fp": fp, "fn": fn, "precision": precision, "recall": recall, "f1": f1}


# ================================================================
# TESTER
# ================================================================
class ComprehensivePartsNERTester:
    def __init__(self, excel_path: str, model_dir: str):
        self.excel_path = excel_path
        self.model_dir = model_dir
        self.req_id = set_request_id()
        self.tracker = PerformanceTracker()
        self.evaluator = PartsNEREvaluator()
        self.templates = load_templates_from_db("parts")
        self.nlp = None

    def load_model(self):
        info_id(f"Loading Parts NER model: {self.model_dir}", self.req_id)

        # Initialize GPU adapter
        gpu = GPUTrainingAdapter()
        device = gpu.get_device(prefer_local=True)

        info_id(
            f"[GPU] Adapter status: {gpu.describe()}",
            self.req_id,
        )

        tok = AutoTokenizer.from_pretrained(
            self.model_dir,
            local_files_only=True,
        )

        mdl = AutoModelForTokenClassification.from_pretrained(
            self.model_dir,
            local_files_only=True,
        )

        # Move model (local CUDA or CPU; remote handled separately)
        mdl = gpu.prepare_model(mdl)

        # HuggingFace pipeline device:
        #   -1 = CPU
        #    0 = cuda:0
        # NOTE: pipeline does NOT support remote GPU directly
        if device.type == "cuda":
            hf_device = 0
        else:
            hf_device = -1

        info_id(
            f"[GPU] Using HuggingFace pipeline device={hf_device}",
            self.req_id,
        )

        self.nlp = pipeline(
            "token-classification",
            model=mdl,
            tokenizer=tok,
            aggregation_strategy="first",
            device=hf_device,
        )

    def run(self, max_rows: int = 50):
        if not self.templates:
            warning_id("No templates loaded — cannot run evaluation", self.req_id)
            return self.tracker

        self.load_model()
        df = pd.read_excel(self.excel_path).head(max_rows)

        for ridx, row in df.iterrows():
            for tidx, tpl in enumerate(self.templates):
                values = {
                    "itemnum": str(row.get("ITEMNUM", "")),
                    "description": str(row.get("DESCRIPTION", "")),
                    "manufacturer": str(row.get("OEMMFG", "")),
                    "oemmfg": str(row.get("OEMMFG", "")),
                    "model": str(row.get("MODEL", "")),
                }

                query = safe_format_template(tpl, values)

                expected = {}
                tpl_l = tpl.lower()

                if "{itemnum}" in tpl_l:
                    expected["PART_NUMBER"] = str(row.get("ITEMNUM", ""))
                if "{description}" in tpl_l:
                    expected["PART_NAME"] = str(row.get("DESCRIPTION", ""))
                if "{manufacturer}" in tpl_l or "{oemmfg}" in tpl_l:
                    expected["MANUFACTURER"] = str(row.get("OEMMFG", ""))
                if "{model}" in tpl_l:
                    expected["MODEL"] = str(row.get("MODEL", ""))

                start = time.time()
                preds = self.nlp(query)
                elapsed = (time.time() - start) * 1000

                evalr = self.evaluator.evaluate(preds, expected)

                self.tracker.add_result(QueryResult(
                    query_id=f"{ridx}_{tidx}",
                    row_index=ridx,
                    template_index=tidx,
                    query_text=query,
                    query_category="parts",
                    language_style="mixed",
                    total_entities_expected=len(expected),
                    total_entities_found=evalr["tp"],
                    execution_time_ms=elapsed,
                    overall_success=evalr["f1"] >= 0.8,
                ))

        return self.tracker

    def interactive_loop(self):
        """
        Interactive NER query loop using the loaded model.
        """
        if self.nlp is None:
            info_id("Model not loaded yet — loading now", self.req_id)
            self.load_model()

        print("\n" + "=" * 80)
        print("INTERACTIVE PARTS NER MODE")
        print("Type a query and press Enter")
        print("Type 'exit' or 'quit' to stop")
        print("=" * 80)

        while True:
            try:
                query = input("\n> ").strip()
                if query.lower() in {"exit", "quit"}:
                    print("Exiting interactive mode.")
                    break

                preds = self.nlp(query)

                if not preds:
                    print("No entities detected.")
                    continue

                print("\nDetected Entities:")
                for p in preds:
                    label = p.get("entity_group") or p.get("entity")
                    text = p.get("word")
                    score = p.get("score", 0.0)
                    print(f"  {label:15} | {text:20} | {score:.3f}")

            except KeyboardInterrupt:
                print("\nInterrupted. Exiting interactive mode.")
                break
            except Exception as e:
                print(f"Error: {e}")


# ================================================================
# MAIN
# ================================================================
def main():
    base_dir = Path(MODEL_TRAINING_PARTS_MODEL_DIR)
    model_dir = resolve_model_dir(base_dir)
    excel = MODEL_TRAINING_TRAINING_DATA_PARTS_LOADSHEET_PATH

    with TrainingLogManager(run_dir=None, run_name="comprehensive_parts_ner_test"):
        tester = ComprehensivePartsNERTester(str(excel), str(model_dir))
        rows = int(input("Rows to test (default 50): ") or "50")
        tracker = tester.run(rows)
        tracker.print_summary_report()
        tracker.save_results("parts_ner_test_results.json")

        # --------------------------------------------------
        # Optional interactive mode
        # --------------------------------------------------
        resp = input("\nEnter interactive mode? (y/n): ").strip().lower()
        if resp == "y":
            tester.interactive_loop()


if __name__ == "__main__":
    main()
