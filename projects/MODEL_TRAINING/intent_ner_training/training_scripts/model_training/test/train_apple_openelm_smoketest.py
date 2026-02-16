"""
Apple OpenELM-1_1B-Instruct MLflow Smoke Test (Offline)
Verifies MLflow logging, offline model loading, and tokenization.
"""

import os
import sys
import torch
import mlflow
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments
from datasets import Dataset
from dotenv import load_dotenv

# ---------------------------------------------------------------------
# 0. Load environment and enforce offline mode
# ---------------------------------------------------------------------
load_dotenv()
os.environ["DATABASE_URL"] = "sqlite:///dummy.db"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

# ---------------------------------------------------------------------
# 1. Resolve directories from environment
# ---------------------------------------------------------------------
MODEL_PATH = os.getenv("MODELS_APPLE_ELM_DIR")
LOGS_DIR = os.getenv("SMOKE_TESTS")

if not MODEL_PATH or not os.path.isdir(MODEL_PATH):
    raise FileNotFoundError(
        f"Apple OpenELM model directory not found:\n{MODEL_PATH}\n"
        "Please verify MODELS_APPLE_ELM_DIR in your .env file."
    )

if not LOGS_DIR or not os.path.isdir(LOGS_DIR):
    raise FileNotFoundError(
        f"Smoke test logs directory not found:\n{LOGS_DIR}\n"
        "Please verify SMOKE_TESTS in your .env file."
    )

# ---------------------------------------------------------------------
# 2. Prepare MLflow workspace
# ---------------------------------------------------------------------
mlruns_path = os.path.join(LOGS_DIR, "mlruns")
os.makedirs(mlruns_path, exist_ok=True)

tmp_dir = os.path.join(LOGS_DIR, "tmp_openelm")
os.makedirs(tmp_dir, exist_ok=True)
os.environ["TMPDIR"] = tmp_dir
os.environ["TEMP"] = tmp_dir

results_dir = os.path.join(LOGS_DIR, "results_openelm")
os.makedirs(results_dir, exist_ok=True)

mlflow.set_tracking_uri(f"file:///{mlruns_path}")
mlflow.set_experiment("Apple_OpenELM_SmokeTest")
os.environ["MLFLOW_ARTIFACT_URI"] = f"file:///{mlruns_path}"

# ---------------------------------------------------------------------
# 3. Load model and tokenizer (safe offline fallback handling)
# ---------------------------------------------------------------------
import os
from transformers import AutoTokenizer, AutoModelForCausalLM

print(f"Loading Apple OpenELM model from: {MODEL_PATH}")

# Get local GPT-2 path from environment (.env)
GPT2_FALLBACK_PATH = os.getenv("MODELS_GPT2_DIR", r"E:\emtac\models\llm\GPT-2")

try:
    # Try to load tokenizer from the OpenELM model directory
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_PATH,
        local_files_only=True,
        trust_remote_code=True
    )
    print("✅ Tokenizer loaded successfully from OpenELM directory.")
except Exception as e:
    print(f"⚠️ Tokenizer not found or unrecognized ({type(e).__name__}): {e}")
    print(f"➡️  Falling back to local GPT-2 tokenizer at: {GPT2_FALLBACK_PATH}")
    tokenizer = AutoTokenizer.from_pretrained(
        GPT2_FALLBACK_PATH,
        local_files_only=True
    )
    print("✅ GPT-2 tokenizer fallback loaded successfully.")

# Handle missing pad/eos tokens gracefully
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# Load the model (trust custom code for Apple OpenELM)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    local_files_only=True,
    trust_remote_code=True
)
model.config.pad_token_id = tokenizer.pad_token_id
model.config.eos_token_id = tokenizer.eos_token_id

# Choose device and move model
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)
print(f"🧠 Using device: {device}")

# ---------------------------------------------------------------------
# 4. Dummy dataset
# ---------------------------------------------------------------------
data = {
    "text": [
        "Apple's OpenELM-1_1B-Instruct is a lightweight instruct-tuned model.",
        "This smoke test validates offline model loading and MLflow logging.",
        "We are verifying tokenization, Trainer setup, and short-step training.",
    ]
}
dataset = Dataset.from_dict(data)
split = dataset.train_test_split(test_size=0.33)
train_dataset, eval_dataset = split["train"], split["test"]

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
    report_to=[],  # Disable wandb/tensorboard
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
)

# ---------------------------------------------------------------------
# 6. Run smoke test
# ---------------------------------------------------------------------
passed = False
try:
    with mlflow.start_run(run_name="Apple_OpenELM_SmokeTest_Run"):
        mlflow.log_param("model_path", MODEL_PATH)
        mlflow.log_param("epochs", training_args.num_train_epochs)
        mlflow.log_param("max_steps", training_args.max_steps)
        mlflow.log_param("device", device)
        mlflow.log_param("artifact_base", LOGS_DIR)

        print("Starting Apple OpenELM smoke test (2 steps)...")
        trainer.train()

        print("Running quick evaluation...")
        metrics = trainer.evaluate()
        for k, v in metrics.items():
            mlflow.log_metric(k, float(v))

        test_save_dir = os.path.join(results_dir, "openelm_smoke_artifact")
        os.makedirs(test_save_dir, exist_ok=True)
        model.save_pretrained(test_save_dir)
        tokenizer.save_pretrained(test_save_dir)
        mlflow.log_artifacts(test_save_dir)

        print("Smoke test complete. Check MLflow logs at logs/smoke_tests/mlruns.")
        passed = True
except Exception as e:
    print("Smoke test failed:", str(e))
    passed = False

# ---------------------------------------------------------------------
# 7. Final summary
# ---------------------------------------------------------------------
if passed:
    print("\n==============================")
    print("APPLE OPENELM SMOKE TEST PASSED")
    print("==============================")
    sys.exit(0)
else:
    print("\n==============================")
    print("APPLE OPENELM SMOKE TEST FAILED")
    print("==============================")
    sys.exit(1)

