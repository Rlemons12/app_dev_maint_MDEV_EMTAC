from transformers import AutoTokenizer, AutoModelForTokenClassification

p = r"E:\emtac\projects\MODEL_TRAINING\intent_ner_training\models\drawings\2025-12-19_00-00-01_run-001\best"

AutoTokenizer.from_pretrained(p, local_files_only=True)
AutoModelForTokenClassification.from_pretrained(p, local_files_only=True)

print("DRAWINGS MODEL LOADS CLEANLY")
