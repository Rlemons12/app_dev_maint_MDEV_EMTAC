"""
Qwen2.5-3B-Instruct MLflow Smoke Test (Offline, DB-Free)
Verifies MLflow logging, offline model loading, and tokenization
using the Qwen2.5-3B-Instruct model defined in your .env (MODELS_QWEN_DIR).
"""

import os
import sys
import torch
import mlflow
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments
from datasets import Dataset
from dotenv import load_dotenv

# ---------------------------------------------------------------------
# 0. Load environment and force offline mode
# ---------------------------------------------------------------------
load_dotenv()  # Load paths and configs from your .env
os.environ["DATABASE_URL"] = "sqlite:///dummy.db"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

# ---------------------------------------------------------------------
# 1. Resolve directories from environment
# ---------------------------------------------------------------------
MODEL_PATH = os.getenv("MODELS_QWEN_DIR")
LOGS_DIR = os.getenv("SMOKE_TESTS")

if not MODEL_PATH or not os.path.isdir(MODEL_PATH):
    raise FileNotFoundError(
        f"Qwen model directory not found:\n{MODEL_PATH}\n"
        "Please verify MODELS_QWEN_DIR in your .env file."
    )

if not LOGS_DIR or not os.path.isdir(LOGS_DIR):
    raise FileNotFoundError(
        f"Smoke test logs directory not found:\n{LOGS_DIR}\n"
        "Please verify SMOKE_TESTS_DIR in your .env file."
    )

# ---------------------------------------------------------------------
# 2. Prepare MLflow workspace
# ---------------------------------------------------------------------
mlruns_path = os.path.join(LOGS_DIR, "mlruns")
os.makedirs(mlruns_path, exist_ok=True)

tmp_dir = os.path.join(LOGS_DIR, "tmp_qwen")
os.makedirs(tmp_dir, exist_ok=True)
os.environ["TMPDIR"] = tmp_dir
os.environ["TEMP"] = tmp_dir

results_dir = os.path.join(LOGS_DIR, "results_qwen")
os.makedirs(results_dir, exist_ok=True)

mlflow.set_tracking_uri(f"file:///{mlruns_path}")
mlflow.set_experiment("Qwen2.5-3B-SmokeTest")
os.environ["MLFLOW_ARTIFACT_URI"] = f"file:///{mlruns_path}"

# ---------------------------------------------------------------------
# 3. Load model and tokenizer (offline)
# ---------------------------------------------------------------------
print(f"Loading Qwen model from: {MODEL_PATH}")
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, local_files_only=True)

# Handle missing pad/eos tokens gracefully
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, local_files_only=True)
model.config.pad_token_id = tokenizer.pad_token_id
model.config.eos_token_id = tokenizer.eos_token_id

device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)
print(f"Using device: {device}")

# ---------------------------------------------------------------------
# 4. Dummy dataset for smoke testing
# ---------------------------------------------------------------------
data = {
    "text": [
        "Qwen2.5-3B-Instruct is an instruction-tuned model designed for complex reasoning.",
        "This smoke test validates the MLflow integration and offline tokenization.",
        "Verifying that Trainer and evaluation run without errors.",
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
# 5. Training arguments (short run)
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
    report_to=[],  # Disable WandB/TensorBoard
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
)

# ---------------------------------------------------------------------
# 6. Run the smoke test
# ---------------------------------------------------------------------
passed = False
try:
    with mlflow.start_run(run_name="Qwen2.5-3B_SmokeTest_Run"):
        mlflow.log_param("model_path", MODEL_PATH)
        mlflow.log_param("epochs", training_args.num_train_epochs)
        mlflow.log_param("max_steps", training_args.max_steps)
        mlflow.log_param("device", device)
        mlflow.log_param("artifact_base", LOGS_DIR)

        print("Starting Qwen2.5-3B-Instruct smoke test (2 steps)...")
        trainer.train()

        print("Running quick evaluation...")
        metrics = trainer.evaluate()
        for k, v in metrics.items():
            mlflow.log_metric(k, float(v))

        # Save artifacts
        test_save_dir = os.path.join(results_dir, "qwen_smoke_artifact")
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
# 7. Final status summary
# ---------------------------------------------------------------------
if passed:
    print("\n==============================")
    print("QWEN2.5-3B SMOKE TEST PASSED")
    print("==============================")
    sys.exit(0)
else:
    print("\n==============================")
    print("QWEN2.5-3B SMOKE TEST FAILED")
    print("==============================")
    sys.exit(1)
