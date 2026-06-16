import os

def create_project_dirs(base_dir="modules/emtac_ai"):
    models_dir = os.path.join(base_dir, "models")
    training_data_dir = os.path.join(base_dir, "training_data", "datasets")

    # List of model names
    model_names = [
        "intent_classifier",
        "parts",
        "images",
        "documents",
        "prints",
        "tools",
        "troubleshooting"
    ]

    # Create models and training_data/datasets directories
    for name in model_names:
        model_path = os.path.join(models_dir, name)
        data_path = os.path.join(training_data_dir, name)
        os.makedirs(model_path, exist_ok=True)
        os.makedirs(data_path, exist_ok=True)
        print(f"Created directories:\n  Model dir: {model_path}\n  Training data dir: {data_path}\n")

if __name__ == "__main__":
    create_project_dirs()
