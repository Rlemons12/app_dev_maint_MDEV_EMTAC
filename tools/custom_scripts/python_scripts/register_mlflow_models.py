import os
import hashlib
import time
import csv
import shutil
from datetime import datetime
from dotenv import load_dotenv
import mlflow
from mlflow import MlflowClient
from mlflow.exceptions import MlflowException

# ------------------------------------------------------------
# 1. Load environment variables (.env file)
# ------------------------------------------------------------
dotenv_path = os.getenv("DOTENV_PATH", os.path.join("E:", "emtac", ".env"))
load_dotenv(dotenv_path)

MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "postgresql://postgres:emtac123@localhost:5432/mlflowdb")
MODELS_LLM_DIR = os.getenv("MODELS_LLM_DIR", r"E:\emtac\models\llm")
ARTIFACT_DIR = os.getenv("MLFLOW_ARTIFACT_DIR", r"E:\emtac\logs\smoke_tests\mlartifacts")
SUMMARY_CSV_PATH = os.path.join(os.path.dirname(ARTIFACT_DIR), "mlflow_registry_summary.csv")

# Ensure artifact directory exists
os.makedirs(ARTIFACT_DIR, exist_ok=True)
ARTIFACT_URI = f"file:///{ARTIFACT_DIR.replace(os.sep, '/')}"

# Apply MLflow configuration
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
mlflow.set_registry_uri(MLFLOW_TRACKING_URI)

# Create or reuse global experiment
GLOBAL_EXPERIMENT = "model_registry_sync"
if mlflow.get_experiment_by_name(GLOBAL_EXPERIMENT) is None:
    mlflow.create_experiment(GLOBAL_EXPERIMENT, artifact_location=ARTIFACT_URI)
mlflow.set_experiment(GLOBAL_EXPERIMENT)

client = MlflowClient()

print(f"Connected to MLflow tracking server: {MLFLOW_TRACKING_URI}")
print(f"Artifact directory set to: {ARTIFACT_URI}")
print(f"Scanning model root: {MODELS_LLM_DIR}\n")


# ------------------------------------------------------------
# 2. Utility: Calculate folder hash
# ------------------------------------------------------------
def calculate_folder_hash(folder_path):
    hash_md5 = hashlib.md5()
    for root, _, files in os.walk(folder_path):
        for f in sorted(files):
            file_path = os.path.join(root, f)
            if os.path.isfile(file_path):
                try:
                    with open(file_path, "rb") as file:
                        while chunk := file.read(8192):
                            hash_md5.update(chunk)
                except Exception:
                    continue
    return hash_md5.hexdigest()


# ------------------------------------------------------------
# 3. Utility: Prune old artifact runs (keep last 3)
# ------------------------------------------------------------
def prune_old_artifacts(model_dir, keep_last=3):
    if not os.path.exists(model_dir):
        return []

    subdirs = [os.path.join(model_dir, d) for d in os.listdir(model_dir)
               if os.path.isdir(os.path.join(model_dir, d))]
    if len(subdirs) <= keep_last:
        return []

    subdirs.sort(key=lambda d: os.path.getmtime(d), reverse=True)
    to_delete = subdirs[keep_last:]

    deleted = []
    for d in to_delete:
        try:
            shutil.rmtree(d, ignore_errors=True)
            deleted.append(os.path.basename(d))
        except Exception as e:
            print(f"Could not delete old artifact folder {d}: {e}")
    return deleted


# ------------------------------------------------------------
# 4. CSV Logging Utilities
# ------------------------------------------------------------
if not os.path.exists(SUMMARY_CSV_PATH):
    with open(SUMMARY_CSV_PATH, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["timestamp", "model_name", "version", "status", "hash", "notes"])

def log_csv_entry(model_name, version, status, folder_hash, notes=""):
    with open(SUMMARY_CSV_PATH, "a", newline="", encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow([
            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            model_name,
            version,
            status,
            folder_hash,
            notes
        ])


# ------------------------------------------------------------
# 5. Gather already registered models
# ------------------------------------------------------------
registered_models = [m.name for m in client.search_registered_models()]
print(f"Found {len(registered_models)} models already in the registry.\n")


# ------------------------------------------------------------
# 6. Walk through model directories
# ------------------------------------------------------------
for root, dirs, files in os.walk(MODELS_LLM_DIR):
    if any(f.endswith((".safetensors", "config.json", "model.safetensors")) for f in files):
        model_name = os.path.basename(root)

        if model_name.lower() in ("snapshots", "refs"):
            continue

        print(f"Processing model folder: {model_name}")

        if model_name not in registered_models:
            try:
                client.create_registered_model(model_name)
                print(f"Created new registered model entry: {model_name}")
                registered_models.append(model_name)
            except MlflowException as e:
                print(f"Could not create model '{model_name}': {e}")
                log_csv_entry(model_name, "N/A", "failed", "", f"register_failed: {e}")
                continue

        folder_hash = calculate_folder_hash(root)
        hash_log_path = os.path.join(ARTIFACT_DIR, f"{model_name}_last_hash.txt")

        if os.path.exists(hash_log_path):
            with open(hash_log_path, "r") as f:
                last_hash = f.read().strip()
            if last_hash == folder_hash:
                print(f"No changes detected for '{model_name}', skipping.\n")
                log_csv_entry(model_name, "-", "skipped", folder_hash, "unchanged")
                continue

        with open(hash_log_path, "w") as f:
            f.write(folder_hash)

        experiment_name = f"{model_name}_registration"
        experiment_dir = os.path.join(ARTIFACT_DIR, model_name)
        os.makedirs(experiment_dir, exist_ok=True)
        experiment_uri = f"file:///{experiment_dir.replace(os.sep, '/')}"

        exp = mlflow.get_experiment_by_name(experiment_name)
        if exp is None:
            exp_id = mlflow.create_experiment(experiment_name, artifact_location=experiment_uri)
            print(f"Created experiment '{experiment_name}' with artifact root {experiment_uri}")
        else:
            exp_id = exp.experiment_id
            print(f"Using existing experiment '{experiment_name}' (ID={exp_id})")

        try:
            with mlflow.start_run(run_name=f"{model_name}_upload", experiment_id=exp_id) as run:
                mlflow.log_artifacts(root, artifact_path="model")

                source = os.path.join(experiment_dir, run.info.run_id, "artifacts", "model")
                mv = client.create_model_version(
                    name=model_name,
                    source=f"file:///{source.replace(os.sep, '/')}",
                    run_id=run.info.run_id
                )
                print(f"Registered version {mv.version} for '{model_name}'")

                log_csv_entry(model_name, mv.version, "success", folder_hash)

                deleted = prune_old_artifacts(experiment_dir, keep_last=3)
                if deleted:
                    print(f"Pruned old artifacts for '{model_name}': {deleted}")
                    log_csv_entry(model_name, mv.version, "pruned", folder_hash, f"removed: {deleted}")

        except MlflowException as e:
            print(f"Failed to create version for '{model_name}': {e}\n")
            log_csv_entry(model_name, "N/A", "failed", folder_hash, str(e))

        time.sleep(1)


# ------------------------------------------------------------
# 7. Completion message
# ------------------------------------------------------------
print("\nModel registration sync complete.")
print(f"Summary log written to: {SUMMARY_CSV_PATH}")
