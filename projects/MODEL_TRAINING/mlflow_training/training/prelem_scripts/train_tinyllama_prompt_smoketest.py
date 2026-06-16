"""
TinyLlama Prompt Smoke Test (Offline + Local MLflow)
----------------------------------------------------
Loads local TinyLlama model (no internet required)
Runs one short inference
Logs results and prompt/response to MLflow (offline)
Saves all artifacts under E:\emtac\logs\smoke_tests
"""

import os
import sys
import torch
import mlflow
from transformers import AutoTokenizer, AutoModelForCausalLM

# ============================================================== #
# 0. Environment Setup (Offline / Local Paths)
# ============================================================== #
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["DATABASE_URL"] = "sqlite:///dummy.db"

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../../.."))
sys.path.append(project_root)

try:
    from modules.configuration import config
except ImportError:
    raise ImportError(
        "Could not import 'modules.configuration.config'. "
        "Make sure PYTHONPATH includes the 'modules' directory."
    )

MODEL_PATH = getattr(config, "MODELS_TINY_LLAMA_DIR", r"E:\emtac\models\llm\TinyLlama_1_1B")
LOGS_DIR = getattr(config, "SMOKE_TESTS_DIR", r"E:\emtac\logs\smoke_tests")

if not os.path.isdir(MODEL_PATH):
    raise FileNotFoundError(f"Model path not found:\n{MODEL_PATH}")
if not os.path.isdir(LOGS_DIR):
    raise FileNotFoundError(f"Logs directory not found:\n{LOGS_DIR}")

# ============================================================== #
# 1. MLflow Configuration
# ============================================================== #
mlruns_path = os.path.join(LOGS_DIR, "mlruns")
os.makedirs(mlruns_path, exist_ok=True)

# --- Ensure .trash folder exists and is valid ---
trash_path = os.path.join(mlruns_path, ".trash")
if os.path.exists(trash_path):
    import shutil
    try:
        shutil.rmtree(trash_path)
        print(f"[INFO] Removed stale MLflow trash folder: {trash_path}")
    except Exception as e:
        print(f"[WARN] Could not remove .trash folder: {e}")
os.makedirs(trash_path, exist_ok=True)

# --- Define MLflow experiment ---
experiment_name = "TinyLlama_PromptSmokeTest"
artifact_uri = f"file:///{mlruns_path.replace(os.sep, '/')}"

client = mlflow.tracking.MlflowClient()

# Remove any corrupted experiment entries
try:
    exp = client.get_experiment_by_name(experiment_name)
except Exception as e:
    print(f"[WARN] Could not query existing experiments: {e}")
    exp = None

if exp is not None:
    try:
        exp_dir = os.path.join(mlruns_path, exp.experiment_id)
        if not os.path.exists(exp_dir):
            print(f"[WARN] Experiment ID {exp.experiment_id} missing on disk. Removing stale registry entry...")
            client.delete_experiment(exp.experiment_id)
            exp = None
    except Exception as e:
        print(f"[WARN] Cleanup check failed: {e}")

if exp is None:
    exp_id = client.create_experiment(name=experiment_name, artifact_location=artifact_uri)
    print(f"[INFO] Created new MLflow experiment: {experiment_name} (ID: {exp_id})")
else:
    exp_id = exp.experiment_id
    print(f"[INFO] Reusing existing MLflow experiment: {experiment_name} (ID: {exp_id})")

# Apply URI configuration
os.environ["MLFLOW_TRACKING_URI"] = artifact_uri
os.environ["MLFLOW_ARTIFACT_URI"] = artifact_uri
mlflow.set_tracking_uri(artifact_uri)
mlflow.set_experiment(experiment_name)

print(f"[INFO] MLflow tracking initialized at: {artifact_uri}")
print(f"[INFO] MLflow experiment ID: {exp_id}")
print(f"[INFO] MLflow artifacts will be stored in: {artifact_uri}")

# ============================================================== #
# 2. Model and Prompt Configuration
# ============================================================== #
prompt_text = "Explain in one sentence what TinyLlama is."
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"[INFO] Using device: {device}")
print(f"[INFO] Loading model from: {MODEL_PATH}")

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, local_files_only=True)
model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, local_files_only=True).to(device)
model.eval()

# ============================================================== #
# 3. Run Inference
# ============================================================== #
inputs = tokenizer(prompt_text, return_tensors="pt").to(device)
outputs = model.generate(
    **inputs,
    max_new_tokens=50,
    do_sample=True,
    temperature=0.7,
    top_p=0.9
)
response_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

print("\n=== PROMPT ===")
print(prompt_text)
print("=== RESPONSE ===")
print(response_text)

# ============================================================== #
# 4. Log to MLflow
# ============================================================== #
passed = False
try:
    with mlflow.start_run(run_name="TinyLlama_PromptSmokeTest_Run", experiment_id=exp_id) as run:
        run_id = run.info.run_id
        print(f"[INFO] MLflow run started: {run_id}")

        mlflow.log_param("model_path", MODEL_PATH)
        mlflow.log_param("device", str(device))
        mlflow.log_param("prompt_text", prompt_text)
        mlflow.log_param("response_preview", response_text[:120])

        try:
            if hasattr(mlflow, "prompts"):
                mlflow.prompts.log_prompt(
                    prompt=prompt_text,
                    response=response_text,
                    context="TinyLlama smoke test",
                    metadata={"response_length": len(response_text.split())},
                )
                print("[INFO] Logged structured prompt to MLflow Prompts tab.")
            else:
                print("[WARN] MLflow version has no 'prompts' module. Skipping structured log.")
        except Exception as e:
            print(f"[WARN] Structured prompt log failed: {e}")

        # Save local results
        artifacts_dir = os.path.join(LOGS_DIR, "tinyllama_prompt_results")
        os.makedirs(artifacts_dir, exist_ok=True)
        output_file = os.path.join(artifacts_dir, "tinyllama_response.txt")

        with open(output_file, "w", encoding="utf-8") as f:
            f.write(f"Prompt:\n{prompt_text}\n\nResponse:\n{response_text}\n")

        mlflow.log_artifact(output_file)
        print(f"[INFO] Logged artifact: {output_file}")

        print("\nSmoke test complete. View results in MLflow UI.")
        print(f"   Command: mlflow ui --backend-store-uri {artifact_uri} --port 5010")
        print("   URL: http://127.0.0.1:5010/#/experiments")
        print(f"   Experiment: {experiment_name}")
        passed = True

except Exception as e:
    print(f"[ERROR] Smoke test failed: {e}")
    passed = False

# ============================================================== #
# 5. Final Status Summary
# ============================================================== #
if passed:
    print("\n==============================")
    print("TINYLLAMA PROMPT SMOKE TEST PASSED")
    print("==============================")
    sys.exit(0)
else:
    print("\n==============================")
    print("TINYLLAMA PROMPT SMOKE TEST FAILED")
    print("==============================")
    sys.exit(1)
