"""
Direct UnifiedSearch → Orchestrator Test
----------------------------------------

This script bypasses:
  - Flask
  - the UI
  - chatbox
  - AistManager

It directly calls UnifiedSearch.execute_unified_search()
so we can see EXACTLY what the orchestrator returns inside UnifiedSearch.
"""

import os
import sys
from pprint import pprint


# ---------------------------------------------------------
# 1. Bootstrap project root (important for imports)
# ---------------------------------------------------------
def bootstrap_paths():
    this_file = os.path.abspath(__file__)
    demos_dir = os.path.dirname(this_file)
    tests_dir = os.path.dirname(demos_dir)
    project_root = os.path.dirname(tests_dir)

    if project_root not in sys.path:
        sys.path.insert(0, project_root)

    print(f"[BOOTSTRAP] Added project root: {project_root}")
    return project_root


ROOT = bootstrap_paths()


# ---------------------------------------------------------
# 2. Import UnifiedSearch + DB Session
# ---------------------------------------------------------
from modules.emtacdb.emtacdb_fts import Session
from modules.emtac_ai.search.UnifiedSearch import UnifiedSearch


# ---------------------------------------------------------
# 3. Build a UnifiedSearch instance
# ---------------------------------------------------------
def build_unified_search():
    """
    This builds UnifiedSearch EXACTLY like the app does,
    but without AistManager or Flask.

    - orchestrator enabled
    - vector optional
    - RAG optional
    """
    session = Session()

    us = UnifiedSearch(
        db_session=session,
        enable_vector=False,        # leave off unless needed
        enable_fts=False,           # only if you want FTS fallback
        enable_orchestrator=True,   # <<--- IMPORTANT
        intent_model_dir=None,      # auto-load from DB
        ner_model_dirs=None,        # auto-load from DB
        ai_model=None,              # default model (optional)
        domain="maintenance",
    )

    return us


# ---------------------------------------------------------
# 4. Run a test query
# ---------------------------------------------------------
def run_test(query: str):
    us = build_unified_search()

    print(f"\n[TEST] Query: {query}\n")

    response = us.execute_unified_search(query)

    print("\n================= UNIFIED SEARCH RESULT =================")
    pprint(response)
    print("=========================================================\n")


# ---------------------------------------------------------
# 5. Main entry
# ---------------------------------------------------------
if __name__ == "__main__":
    test_query = "Do we have any parts for the pinch valve at the fill station?"
    run_test(test_query)
