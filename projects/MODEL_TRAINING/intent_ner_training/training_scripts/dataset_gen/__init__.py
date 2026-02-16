"""
Dataset Generation Package

This package contains scripts for building training datasets for
NER and Intent classification across different domains (drawings,
images, parts). It exposes high-level entry points for generating
examples programmatically instead of running the scripts only via CLI.
"""

from . import build_intent_from_templates
from . import generate_drawings_ner_train
from . import generate_images_ner_train
from . import generate_parts_ner_train
from . import updt_generate_parts_ner_train

# ----------------------------
# Public API
# ----------------------------

__all__ = [
    # Drawings
    "generate_drawings_dataset",
    "generate_drawings_intents",

    # Images
    "generate_image_training_data",
    "save_training_data",
    "create_multi_class_training_script",

    # Parts (legacy + updated)
    "generate_ner_training_file",
    "generate_parts_dataset_db",
]

# ----------------------------
# Re-exports for convenience
# ----------------------------

# --- Drawings ---
generate_drawings_dataset = generate_drawings_ner_train.main
generate_drawings_intents = generate_drawings_ner_train.load_query_templates

# --- Images ---
generate_image_training_data = generate_images_ner_train.generate_image_training_data
save_training_data = generate_images_ner_train.save_training_data
create_multi_class_training_script = generate_images_ner_train.create_multi_class_training_script

# --- Parts (file-based) ---
generate_ner_training_file = generate_parts_ner_train.generate_ner_training_file

# --- Parts (DB-backed, updated) ---
def generate_parts_dataset_db(session, out_dir: str = "output"):
    """
    Convenience wrapper around updt_generate_parts_ner_train
    for generating both NER and Intent datasets from DB + loadsheet.
    """
    intent, templates, ds, sheet_name, cand_map = \
        updt_generate_parts_ner_train.load_from_db(session)

    df = updt_generate_parts_ner_train.read_parts_frame(ds, updt_generate_parts_ner_train.detect_repo_root())
    colmap = updt_generate_parts_ner_train.resolve_columns_from_pcm(ds, session, df.columns.tolist())

    ner_samples = updt_generate_parts_ner_train.build_ner_samples(df, colmap)
    intent_samples = updt_generate_parts_ner_train.build_intent_samples(templates, df, colmap)

    updt_generate_parts_ner_train.save_jsonl(ner_samples, intent_samples, out_dir)

    return ner_samples, intent_samples
