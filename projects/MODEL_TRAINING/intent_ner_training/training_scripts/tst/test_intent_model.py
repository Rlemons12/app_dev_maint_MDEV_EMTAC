# test_intent_model.py
import sys
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

# Add project root
sys.path.insert(0, str(Path(__file__).parent))
from modules.configuration.config import ORC_INTENT_MODEL_DIR


def load_model():
    """Load the intent model"""
    model_dir = Path(ORC_INTENT_MODEL_DIR)
    print(f"Loading model from: {model_dir}")

    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    model.eval()

    return tokenizer, model


def predict(text, tokenizer, model):
    """Predict intent for a single text"""
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)

    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        probs = torch.softmax(logits, dim=-1)
        pred_id = torch.argmax(logits, dim=-1).item()
        confidence = probs[0, pred_id].item()

    # Get label from model config
    label = model.config.id2label[pred_id]

    return label, confidence


def main():
    print("=" * 60)
    print("INTENT MODEL - INTERACTIVE TESTER")
    print("=" * 60)
    print()

    tokenizer, model = load_model()
    print("Model loaded successfully!")
    print(f"Labels: {list(model.config.label2id.keys())}")
    print()
    print("Type your queries (or press Enter to quit):")
    print("-" * 60)

    while True:
        try:
            query = input("\nQuery: ").strip()
            if not query:
                print("Exiting...")
                break

            label, confidence = predict(query, tokenizer, model)
            print(f"Intent: {label} (confidence: {confidence:.2%})")

        except KeyboardInterrupt:
            print("\n\nExiting...")
            break
        except Exception as e:
            print(f"Error: {e}")


if __name__ == "__main__":
    main()