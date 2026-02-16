import os
import sys

# Add parent 'emtac_ai' folder to sys.path for importing config
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from config import MODEL_DIRS  # Import config from modules/emtac_ai/config.py

from transformers import AutoModelForSequenceClassification, AutoModelForTokenClassification, AutoTokenizer

def save_base_intent_model(model_name="distilbert-base-uncased", output_dir=None):
    print(f"Downloading and saving base intent model to {output_dir}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=6)  # 6 intents
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print("Base intent model saved.\n")

def save_base_ner_model(model_name="dslim/bert-base-NER", output_dir=None):
    print(f"Downloading and saving base NER model to {output_dir}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForTokenClassification.from_pretrained(model_name)
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print("Base NER model saved.\n")

def main():
    # Save intent classifier base model
    save_base_intent_model(output_dir=MODEL_DIRS["intent_classifier"])

    # Save base NER models for all specialized intents
    for intent in ["parts", "images", "documents", "prints", "tools", "troubleshooting"]:
        print(f"Setting up base NER model for intent: {intent}")
        save_base_ner_model(output_dir=MODEL_DIRS[intent])

if __name__ == "__main__":
    main()
