import os
from pathlib import Path
from modules.emtac_ai.orchestrator.orchestrator import Orchestrator


def check_model_directories():
    """Check what files are available in model directories"""

    base_path = Path("modules/emtac_ai/models")
    print("=== Model Directory Analysis ===\n")

    # Check intent classifier
    intent_path = base_path / "intent_classifier"
    print(f"Intent Classifier Path: {intent_path}")
    print(f"Exists: {intent_path.exists()}")
    if intent_path.exists():
        files = list(intent_path.iterdir())
        print(f"Files found: {[f.name for f in files]}")

        # Check for essential files
        config_exists = (intent_path / "config.json").exists()
        model_files = [f for f in files if f.suffix in ['.bin', '.safetensors']]
        tokenizer_files = [f for f in files if 'tokenizer' in f.name.lower()]

        print(f"Has config.json: {config_exists}")
        print(f"Model files: {[f.name for f in model_files]}")
        print(f"Tokenizer files: {[f.name for f in tokenizer_files]}")
    print()

    # Check NER model directories
    ner_dirs = ["parts", "images", "documents", "prints", "tools", "troubleshooting"]
    for ner_dir in ner_dirs:
        ner_path = base_path / ner_dir
        print(f"{ner_dir.title()} NER Path: {ner_path}")
        print(f"Exists: {ner_path.exists()}")
        if ner_path.exists():
            files = list(ner_path.iterdir())
            print(f"Files: {[f.name for f in files]}")
        print()


def test_with_absolute_paths():
    """Test orchestrator with absolute paths"""

    print("=== Testing with Absolute Paths ===\n")

    # Get current working directory
    cwd = Path.cwd()
    print(f"Current working directory: {cwd}")

    # Build absolute paths
    intent_model_path = cwd / "modules/emtac_ai/models/intent_classifier"

    ner_dirs = {}
    for intent in ["parts", "images", "documents", "prints", "tools", "troubleshooting"]:
        ner_path = cwd / f"modules/emtac_ai/models/{intent}"
        ner_dirs[intent] = str(ner_path)

    print(f"Intent model absolute path: {intent_model_path}")
    print(f"Intent model exists: {intent_model_path.exists()}")

    try:
        orchestrator = Orchestrator(
            intent_model_dir=str(intent_model_path),
            ner_model_dirs=ner_dirs,
            intent_labels=["parts", "images", "documents", "prints", "tools", "troubleshooting"],
            ner_labels=["O", "B-PARTDESC", "B-PARTNUM"]
        )

        # Test one case
        test_text = "Show me all Banner sensors with part number PR-48371"
        print(f"\nTesting: '{test_text}'")
        result = orchestrator.process_prompt(test_text)
        print(f"Result: {result}")

    except Exception as e:
        print(f"Error creating orchestrator: {e}")
        import traceback
        traceback.print_exc()


def create_minimal_test_model():
    """Create a minimal test model structure if none exists"""

    print("=== Creating Minimal Test Model ===\n")

    intent_path = Path("modules/emtac_ai/models/intent_classifier")

    if not intent_path.exists():
        print(f"Creating directory: {intent_path}")
        intent_path.mkdir(parents=True, exist_ok=True)

    # Check if we can use a pretrained model from huggingface
    try:
        from transformers import AutoTokenizer, AutoModelForSequenceClassification

        # Use a small pretrained model for testing
        model_name = "distilbert-base-uncased"
        print(f"Downloading {model_name} for testing...")

        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            num_labels=6,  # 6 intents
            ignore_mismatched_sizes=True
        )

        # Save to local directory
        tokenizer.save_pretrained(intent_path)
        model.save_pretrained(intent_path)

        print(f"Test model saved to: {intent_path}")
        return True

    except Exception as e:
        print(f"Could not create test model: {e}")
        return False


def run_full_test():
    """Run the complete test suite"""

    # First check what we have
    check_model_directories()

    # Try with absolute paths
    test_with_absolute_paths()

    # If no model exists, try to create one
    intent_path = Path("modules/emtac_ai/models/intent_classifier")
    if not intent_path.exists() or not list(intent_path.glob("*.json")):
        print("\nNo valid model found, attempting to create test model...")
        if create_minimal_test_model():
            print("Test model created, trying again...")
            test_with_absolute_paths()


if __name__ == "__main__":
    run_full_test()