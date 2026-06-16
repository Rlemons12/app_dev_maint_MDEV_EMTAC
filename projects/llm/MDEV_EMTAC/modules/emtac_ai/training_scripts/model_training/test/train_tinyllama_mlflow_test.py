"""
TinyLlama MLflow Smoke Test (Offline, DB-Free)
Verifies that MLflow logging, local TinyLlama model, and dataset tokenization all work
without requiring internet or PostgreSQL access.
"""

import os
import sys
import torch
import mlflow
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments
from datasets import Dataset

# ---------------------------------------------------------------------
# 0. Force offline mode and dummy database
# ---------------------------------------------------------------------
os.environ["DATABASE_URL"] = "sqlite:///dummy.db"     # Prevent Postgres connection
os.environ["TRANSFORMERS_OFFLINE"] = "1"              # Disable internet downloads

# ---------------------------------------------------------------------
# 1. Import base paths from centralized config
# ---------------------------------------------------------------------
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../../.."))
sys.path.append(project_root)

try:
    from modules.configuration import config
except ImportError:
    raise ImportError(
        "Could not import 'modules.configuration.config'. "
        "Ensure this script is run from the project root or that PYTHONPATH includes the 'modules' directory."
    )

BASE_DIR = config.BASE_DIR
model_path = getattr(config, "MODELS_TINY_LLAMA_DIR", None)
logs_dir = getattr(config, "SMOKE_TESTS_DIR", None)

if not logs_dir or not os.path.isdir(logs_dir):
    raise FileNotFoundError(
        f"Smoke test log directory not found. Expected path from config.SMOKE_TESTS_DIR:\n{logs_dir}\n"
        "Please verify SMOKE_TESTS in your .env or config.py."
    )

if not model_path or not os.path.isdir(model_path):
    raise FileNotFoundError(
        f"TinyLlama model path not found. Expected from config:\n{model_path}\n"
        "Please verify MODELS_TINY_LLAMA_DIR in your .env or config.py."
    )

# ---------------------------------------------------------------------
# 2. Prepare smoke test workspace
# ---------------------------------------------------------------------
mlruns_path = os.path.join(logs_dir, "mlruns")
os.makedirs(mlruns_path, exist_ok=True)

# Create temp and result directories
tmp_dir = os.path.join(logs_dir, "tmp")
os.makedirs(tmp_dir, exist_ok=True)
os.environ["TMPDIR"] = tmp_dir
os.environ["TEMP"] = tmp_dir

results_dir = os.path.join(logs_dir, "results_tinyllama")
os.makedirs(results_dir, exist_ok=True)

# ---------------------------------------------------------------------
# 3. Configure MLflow (local, offline)
# ---------------------------------------------------------------------
# Use your centralized smoke test directory from config
mlruns_path = os.path.join(config.SMOKE_TESTS_DIR, "mlruns")
os.makedirs(mlruns_path, exist_ok=True)

# Set both tracking and artifact storage paths explicitly
mlflow.set_tracking_uri(f"file:///{mlruns_path}")
mlflow.set_experiment("TinyLlama_SmokeTest")
os.environ["MLFLOW_ARTIFACT_URI"] = f"file:///{mlruns_path}"


# ---------------------------------------------------------------------
# 4. Load model and tokenizer
# ---------------------------------------------------------------------
print(f"Loading model from: {model_path}")
tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
model = AutoModelForCausalLM.from_pretrained(model_path, local_files_only=True)

device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)
print(f"Using device: {device}")

# ---------------------------------------------------------------------
# 5. Offline dummy dataset
# ---------------------------------------------------------------------
data = {
    "text": [
        "The quick brown fox jumps over the lazy dog.",
        "TinyLlama is a small but mighty model for language tasks.",
        "MLflow helps track experiments efficiently.",
    ]
}

dataset = Dataset.from_dict(data)
split = dataset.train_test_split(test_size=0.33)
train_dataset = split["train"]
eval_dataset = split["test"]

def tokenize_fn(batch):
    return tokenizer(batch["text"], truncation=True, padding="max_length", max_length=64)

train_dataset = train_dataset.map(tokenize_fn, batched=True)
eval_dataset = eval_dataset.map(tokenize_fn, batched=True)

train_dataset.set_format(type="torch", columns=["input_ids", "attention_mask"])
eval_dataset.set_format(type="torch", columns=["input_ids", "attention_mask"])

train_dataset = train_dataset.map(lambda x: {"labels": x["input_ids"]})
eval_dataset = eval_dataset.map(lambda x: {"labels": x["input_ids"]})

# ---------------------------------------------------------------------
# 6. Training arguments (minimal smoke test)
# ---------------------------------------------------------------------
output_dir = os.path.join(results_dir, "run_output")
os.makedirs(output_dir, exist_ok=True)

training_args = TrainingArguments(
    output_dir=output_dir,
    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,
    num_train_epochs=0.01,
    max_steps=2,
    learning_rate=5e-5,
    logging_steps=1,
    evaluation_strategy="steps",
    eval_steps=1,
    save_steps=2,
    report_to=[],  # Disable WandB and TensorBoard
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
)

# ---------------------------------------------------------------------
# 7. Run smoke test with MLflow logging
# ---------------------------------------------------------------------
passed = False
try:
    with mlflow.start_run(run_name="TinyLlama_SmokeTest_Run"):
        mlflow.log_param("model_path", model_path)
        mlflow.log_param("epochs", training_args.num_train_epochs)
        mlflow.log_param("max_steps", training_args.max_steps)
        mlflow.log_param("device", device)
        mlflow.log_param("artifact_base", logs_dir)

        print("Starting TinyLlama smoke test (2 steps)...")
        trainer.train()

        print("Running quick evaluation...")
        metrics = trainer.evaluate()
        for key, value in metrics.items():
            mlflow.log_metric(key, float(value))

        # Save and log results
        test_save_dir = os.path.join(results_dir, "tinyllama_smoke_artifact")
        os.makedirs(test_save_dir, exist_ok=True)
        model.save_pretrained(test_save_dir)
        tokenizer.save_pretrained(test_save_dir)
        mlflow.log_artifacts(test_save_dir)

        print("Smoke test complete. Check MLflow UI or logs/smoke_tests/mlruns directory.")
        passed = True
except Exception as e:
    print("Smoke test failed:", str(e))
    passed = False

# ---------------------------------------------------------------------
# 8. Final status summary
# ---------------------------------------------------------------------
if passed:
    print("\n==============================")
    print("TINYLLAMA SMOKE TEST PASSED")
    print("==============================")
    sys.exit(0)
else:
    print("\n==============================")
    print("TINYLLAMA SMOKE TEST FAILED")
    print("==============================")
    sys.exit(1)
