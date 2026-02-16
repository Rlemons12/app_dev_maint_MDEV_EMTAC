import os
from pathlib import Path

# =========================================================
# Resolve MODEL_TRAINING base
# =========================================================

MODEL_TRAINING_BASE_DIR = Path(
    os.getenv("MODEL_TRAINING_BASE_DIR", Path(__file__).resolve().parents[1])
).resolve()

# intent_ner_training root (THIS is the real working root)
INTENT_NER_ROOT = MODEL_TRAINING_BASE_DIR / "intent_ner_training"

# Backward-compatible alias (keep name, fix value)
INTENT_NER_MODEL_DIR = INTENT_NER_ROOT


# =========================================================
# Models directory (inside intent_ner_training)
# =========================================================

MODEL_TRAINING_MODELS_DIR = INTENT_NER_ROOT / "models"


# =========================================================
# Training data
# =========================================================

MODEL_TRAINING_TRAINING_DATA_ROOT = INTENT_NER_ROOT / "training_data"
MODEL_TRAINING_TRAINING_DATA_DIR = MODEL_TRAINING_TRAINING_DATA_ROOT / "datasets"
MODEL_TRAINING_TRAINING_DATA_LOADSHEET = MODEL_TRAINING_TRAINING_DATA_DIR / "loadsheet"


# =========================================================
# Specific model directories
# =========================================================

MODEL_TRAINING_INTENT_MODEL_DIR = MODEL_TRAINING_MODELS_DIR / "intent_classifier"
MODEL_TRAINING_PARTS_MODEL_DIR = MODEL_TRAINING_MODELS_DIR / "parts"
MODEL_TRAINING_IMAGES_MODEL_DIR = MODEL_TRAINING_MODELS_DIR / "images"
MODEL_TRAINING_DOCUMENTS_MODEL_DIR = MODEL_TRAINING_MODELS_DIR / "documents"
MODEL_TRAINING_DRAWINGS_MODEL_DIR = MODEL_TRAINING_MODELS_DIR / "drawings"
MODEL_TRAINING_TOOLS_MODEL_DIR = MODEL_TRAINING_MODELS_DIR / "tools"
MODEL_TRAINING_TROUBLESHOOTING_MODEL_DIR = MODEL_TRAINING_MODELS_DIR / "troubleshooting"
MODELS_DISTILBERT_INTENT="E:\emtac\models\modules\transformers_modules\distilbert_intent"

# =========================================================
# Specific training data directories
# =========================================================

MODEL_TRAINING_INTENT_TRAIN_DATA_DIR = MODEL_TRAINING_TRAINING_DATA_DIR / "intent_classifier"
MODEL_TRAINING_PARTS_TRAIN_DATA_DIR = MODEL_TRAINING_TRAINING_DATA_DIR / "parts"
MODEL_TRAINING_IMAGES_TRAIN_DATA_DIR = MODEL_TRAINING_TRAINING_DATA_DIR / "images"
MODEL_TRAINING_DOCUMENTS_TRAIN_DATA_DIR = MODEL_TRAINING_TRAINING_DATA_DIR / "documents"
MODEL_TRAINING_DRAWINGS_TRAIN_DATA_DIR = MODEL_TRAINING_TRAINING_DATA_DIR / "drawings"
MODEL_TRAINING_TOOLS_TRAIN_DATA_DIR = MODEL_TRAINING_TRAINING_DATA_DIR / "tools"
MODEL_TRAINING_TROUBLESHOOTING_TRAIN_DATA_DIR = MODEL_TRAINING_TRAINING_DATA_DIR / "troubleshooting"


# =========================================================
# Query templates
# =========================================================

MODEL_TRAINING_QUERY_TEMPLATES_TRAIN_DATA_DIR = (
    MODEL_TRAINING_TRAINING_DATA_ROOT / "query_templates"
)
MODEL_TRAINING_QUERY_TEMPLATE_PARTS = (
    MODEL_TRAINING_QUERY_TEMPLATES_TRAIN_DATA_DIR / "parts"
)
MODEL_TRAINING_QUERY_TEMPLATE_DRAWINGS = (
    MODEL_TRAINING_QUERY_TEMPLATES_TRAIN_DATA_DIR / "drawings"
)


# =========================================================
# Training scripts & modules
# =========================================================

MODEL_TRAINING_TRAINING_SCRIPTS_DIR = INTENT_NER_ROOT / "training_scripts"
MODEL_TRAINING_TRAINING_SCRIPTS_DATASET_GEN_DIR = (
    MODEL_TRAINING_TRAINING_SCRIPTS_DIR / "dataset_gen"
)
MODEL_TRAINING_TRAINING_SCRIPTS_INTENT_TRAIN_DIR = (
    MODEL_TRAINING_TRAINING_SCRIPTS_DIR / "dataset_intent_train"
)
MODEL_TRAINING_TRAINING_SCRIPTS_PERFORMANCE_DIR = (
    MODEL_TRAINING_TRAINING_SCRIPTS_DIR / "performance_tst_model"
)
MODEL_TRAINING_TRAINING_SCRIPTS_TST_DIR = (
    MODEL_TRAINING_TRAINING_SCRIPTS_DIR / "tst"
)

MODEL_TRAINING_TRAINING_MODULE_DIR = INTENT_NER_ROOT / "training_module"


# =========================================================
# Load sheets
# =========================================================

MODEL_TRAINING_TRAINING_DATA_DRAWINGS_LOADSHEET_PATH = (
    MODEL_TRAINING_TRAINING_DATA_LOADSHEET / "drawing_loadsheet.xlsx"
)
MODEL_TRAINING_TRAINING_DATA_PARTS_LOADSHEET_PATH = (
    MODEL_TRAINING_TRAINING_DATA_LOADSHEET / "parts_loadsheet.xlsx"
)


# =========================================================
# Dictionaries (unchanged API)
# =========================================================

MODEL_DIRS = {
    "intent_classifier": MODEL_TRAINING_INTENT_MODEL_DIR,
    "parts": MODEL_TRAINING_PARTS_MODEL_DIR,
    "images": MODEL_TRAINING_IMAGES_MODEL_DIR,
    "documents": MODEL_TRAINING_DOCUMENTS_MODEL_DIR,
    "drawings": MODEL_TRAINING_DRAWINGS_MODEL_DIR,
    "tools": MODEL_TRAINING_TOOLS_MODEL_DIR,
    "troubleshooting": MODEL_TRAINING_TROUBLESHOOTING_MODEL_DIR,
}

TRAIN_DATA_DIRS = {
    "intent_classifier": MODEL_TRAINING_INTENT_TRAIN_DATA_DIR,
    "parts": MODEL_TRAINING_PARTS_TRAIN_DATA_DIR,
    "images": MODEL_TRAINING_IMAGES_TRAIN_DATA_DIR,
    "documents": MODEL_TRAINING_DOCUMENTS_TRAIN_DATA_DIR,
    "drawings": MODEL_TRAINING_DRAWINGS_TRAIN_DATA_DIR,
    "tools": MODEL_TRAINING_TOOLS_TRAIN_DATA_DIR,
    "troubleshooting": MODEL_TRAINING_TROUBLESHOOTING_TRAIN_DATA_DIR,
}

ALL_DIRS = [
    MODEL_TRAINING_BASE_DIR,
    INTENT_NER_ROOT,
    MODEL_TRAINING_MODELS_DIR,
    MODEL_TRAINING_TRAINING_DATA_ROOT,
    MODEL_TRAINING_TRAINING_DATA_DIR,
    MODEL_TRAINING_TRAINING_DATA_LOADSHEET,
    MODEL_TRAINING_QUERY_TEMPLATES_TRAIN_DATA_DIR,
    MODEL_TRAINING_QUERY_TEMPLATE_PARTS,
    MODEL_TRAINING_QUERY_TEMPLATE_DRAWINGS,
    MODEL_TRAINING_TRAINING_SCRIPTS_DIR,
    MODEL_TRAINING_TRAINING_SCRIPTS_DATASET_GEN_DIR,
    MODEL_TRAINING_TRAINING_SCRIPTS_INTENT_TRAIN_DIR,
    MODEL_TRAINING_TRAINING_SCRIPTS_PERFORMANCE_DIR,
    MODEL_TRAINING_TRAINING_SCRIPTS_TST_DIR,
    MODEL_TRAINING_TRAINING_MODULE_DIR,
] + list(MODEL_DIRS.values()) + list(TRAIN_DATA_DIRS.values())

