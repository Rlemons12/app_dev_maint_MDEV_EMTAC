"""
FULL REAL-WORLD ORCHESTRATOR DEMO
---------------------------------

This demo simulates a complete EMTAC orchestrator workflow, including:

✓ Fake DBServices for all domains:
    - Troubleshooting
    - Documents
    - Images
    - Parts
    - Tools
    - Drawings
    - Position hierarchy
    - pgvector search (simulated embeddings)

✓ Full routing engine:
    - Troubleshooting router
    - Documents router
    - Images router
    - Parts router
    - Tools router
    - Drawings router

✓ A trace-mode that prints detailed internal flow
✓ A stress test (100 random queries)
"""

# ======================================================================
# BOOTSTRAP PROJECT PATHS
# ======================================================================
from bootstrap import bootstrap_paths
bootstrap_paths()   # Ensures modules/ is importable everywhere


# ======================================================================
# IMPORTS
# ======================================================================
import random
import time
from unittest.mock import MagicMock, patch
import argparse

from modules.emtac_ai.intent_ner.intent_orchestrator import IntentOrchestrator

# Routers
from modules.emtac_ai.intent_ner.routers.troubleshooting_router import troubleshooting_router
from modules.emtac_ai.intent_ner.routers.documents_router import documents_router
from modules.emtac_ai.intent_ner.routers.images_router import images_router
from modules.emtac_ai.intent_ner.routers.parts_router import parts_router
from modules.emtac_ai.intent_ner.routers.tools_router import tools_router
from modules.emtac_ai.intent_ner.routers.drawings_router import drawings_router


# ======================================================================
# FAKE DATABASE MODELS
# ======================================================================
class FakeProblem:
    def __init__(self, id=1, name="Overheating", description="Simulated overheating issue"):
        self.id = id
        self.name = name
        self.description = description


class FakeDocument:
    def __init__(self, id=5):
        self.id = id
        self.title = "Demo Document"
        self.file_path = "/demo/path/mock_document.pdf"


class FakeImage:
    def __init__(self, id=10):
        self.id = id
        self.title = "Mock Image"
        self.file_path = "/demo/path/mock_image.png"


class FakePart:
    def __init__(self, id=99):
        self.id = id
        self.part_number = "PN-12345"
        self.name = "Mock Part"


class FakeTool:
    def __init__(self, id=300):
        self.id = id
        self.tool_name = "Mock Wrench"
        self.description = "Used for testing demo."


class FakeDrawing:
    def __init__(self, id=77):
        self.id = id
        self.drw_number = "DRW-200"
        self.drw_name = "Station 2 Assembly"


# ======================================================================
# FAKE DB SERVICES (mocking the real DBServices)
# ======================================================================
mock_db = MagicMock()
mock_db.troubleshooting = MagicMock()
mock_db.documents = MagicMock()
mock_db.images = MagicMock()
mock_db.parts = MagicMock()
mock_db.tools = MagicMock()
mock_db.drawings = MagicMock()
mock_db.search = MagicMock()  # Simulated pgvector search


# ----------------------------------------------------------------------
# Troubleshooting
# ----------------------------------------------------------------------
mock_db.troubleshooting.search_problems.return_value = [FakeProblem()]
mock_db.troubleshooting.get_problem.return_value = FakeProblem()
mock_db.troubleshooting.resolve_query.return_value = {
    "status": "resolved",
    "problem": FakeProblem(),
    "tree": {"root": "mock-root"},
}
mock_db.troubleshooting.find_related.return_value = {"related_issues": ["Mock Issue A"]}

# ----------------------------------------------------------------------
# Documents
# ----------------------------------------------------------------------
mock_db.documents.get.return_value = FakeDocument()
mock_db.documents.find_related.return_value = ["Mock Doc A", "Mock Doc B"]

# ----------------------------------------------------------------------
# Images
# ----------------------------------------------------------------------
mock_db.images.search_by_title.return_value = [FakeImage()]
mock_db.images.search_by_position.return_value = [FakeImage()]
mock_db.images.get.return_value = FakeImage()

# ----------------------------------------------------------------------
# Parts
# ----------------------------------------------------------------------
mock_db.parts.search_by_part_number.return_value = [FakePart()]
mock_db.parts.search_by_position.return_value = [FakePart()]
mock_db.parts.get.return_value = FakePart()

# ----------------------------------------------------------------------
# Tools
# ----------------------------------------------------------------------
mock_db.tools.search_by_name.return_value = [FakeTool()]
mock_db.tools.get.return_value = FakeTool()

# ----------------------------------------------------------------------
# Drawings
# ----------------------------------------------------------------------
mock_db.drawings.search_by_keywords.return_value = [FakeDrawing()]
mock_db.drawings.get.return_value = FakeDrawing()

# ----------------------------------------------------------------------
# Fake pgvector search
# ----------------------------------------------------------------------
mock_db.search.semantic_query.return_value = [
    {"type": "document", "id": 5, "score": 0.89},
    {"type": "image", "id": 10, "score": 0.77},
]


# ======================================================================
# PATCH DBServices BEFORE imports
# ======================================================================
patcher = patch("modules.services.DBServices", return_value=mock_db)
patcher.start()


# ======================================================================
# FAKE INTENT ENGINE
# ======================================================================
class DemoIntentEngine:
    def detect_intent(self, text):
        t = text.lower()

        if "motor" in t:
            return {"intent": "Troubleshooting", "confidence": 0.91}
        if "document" in t:
            return {"intent": "Documents", "confidence": 0.82}
        if "image" in t:
            return {"intent": "Images", "confidence": 0.88}
        if "part" in t:
            return {"intent": "Parts", "confidence": 0.87}
        if "tool" in t:
            return {"intent": "Tools", "confidence": 0.86}
        if "drawing" in t or "print" in t:
            return {"intent": "Drawings", "confidence": 0.89}

        return {"intent": "General_Chat", "confidence": 0.50}


# ======================================================================
# FAKE NER ENGINE
# ======================================================================
class DemoNerEngine:
    def extract_entities(self, text):
        t = text.lower()
        e = {}

        if "motor" in t:
            e["problem_name"] = "Overheating"

        if "document" in t:
            e["doc_id"] = 5

        if "image" in t:
            e["image_id"] = 10

        if "part" in t:
            e["part_number"] = "PN-12345"

        if "tool" in t:
            e["tool_name"] = "Mock Wrench"

        if "drawing" in t or "print" in t:
            e["drawing_number"] = "DRW-200"

        return e


# ======================================================================
# ROUTER MAP
# ======================================================================
routers = {
    "Troubleshooting": troubleshooting_router,
    "Documents": documents_router,
    "Images": images_router,
    "Parts": parts_router,
    "Tools": tools_router,
    "Drawings": drawings_router,
    "General_Chat": lambda **kw: {"fallback": True, "input": kw},
    "default": lambda **kw: {"fallback": True, "input": kw},
}


# ======================================================================
# BUILD ORCHESTRATOR
# ======================================================================
orchestrator = IntentOrchestrator(
    intent_engine=DemoIntentEngine(),
    ner_engine=DemoNerEngine(),
    routers=routers,
)


# ======================================================================
# TRACE MODE
# ======================================================================
def trace(text):
    print("\n==================== TRACE MODE ====================")
    print("USER:", text)
    out = orchestrator.process(text, trace=True)
    print("TRACE OUTPUT:")
    print(out)
    print("====================================================\n")


# ======================================================================
# STRESS TEST (100 RANDOM QUERIES)
# ======================================================================
stress_queries = [
    "why is the motor overheating",
    "show me document 5",
    "find image of station 2",
    "lookup part pn-12345",
    "what tool do i need for station 3",
    "show me the drawing for assembly",
    "hello how are you",
]

def stress_test():
    print("\n==================== STRESS TEST ====================")
    latencies = []
    for _ in range(100):
        q = random.choice(stress_queries)
        start = time.time()
        orchestrator.process(q)
        latencies.append(time.time() - start)

    avg = sum(latencies) / len(latencies)
    print(f"AVG LATENCY over 100 queries: {avg:.4f} sec")
    print("====================================================\n")


# ======================================================================
# NORMAL DEMO
# ======================================================================
def run_demo(text):
    print("\nUSER:", text)
    print("OUTPUT:", orchestrator.process(text))


# ======================================================================
# MAIN
# ======================================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--stress", action="store_true")
    parser.add_argument("--trace", type=str)
    args = parser.parse_args()

    if args.trace:
        trace(args.trace)
        exit()

    if args.stress:
        stress_test()
        exit()

    # Normal demo sequence
    run_demo("Why is the motor overheating?")
    run_demo("Show me document 5")
    run_demo("Show me the image for station 2")
    run_demo("Find part PN-12345")
    run_demo("What tool do I need?")
    run_demo("Show me the drawing for assembly")
    run_demo("Hello, how are you?")
