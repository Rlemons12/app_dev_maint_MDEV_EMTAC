#!/usr/bin/env python3
"""
Simple script to test if your NER model is working at all.
Run this directly with: python simple_model_test.py
"""

import os
import sys
import torch
from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline

# Add your config import path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from modules.emtac_ai.config import ORC_PARTS_MODEL_DIR


def check_model_basic():
    """Basic check to see if model works at all"""

    model_path = ORC_PARTS_MODEL_DIR
    print(f"Testing model at: {model_path}")

    # Check if model directory exists
    if not os.path.exists(model_path):
        print(f"❌ Model directory does not exist: {model_path}")
        return False

    # List files in model directory
    print(f"Files in model directory:")
    for file in os.listdir(model_path):
        print(f"  - {file}")

    try:
        # Load model
        print("\n1. Loading tokenizer and model...")
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForTokenClassification.from_pretrained(model_path)

        print(f"   ✅ Model loaded successfully")
        print(f"   - Vocab size: {tokenizer.vocab_size}")
        print(f"   - Num labels: {model.config.num_labels}")
        print(f"   - Labels: {model.config.id2label}")

        # Check simple prediction
        print("\n2. Checking simple prediction...")
        sample_text = "I need part number A101576"

        # Method 1: Raw model prediction
        inputs = tokenizer(sample_text, return_tensors="pt", padding=True, truncation=True)

        model.eval()
        with torch.no_grad():
            outputs = model(**inputs)
            predictions = torch.argmax(outputs.logits, dim=-1)[0]
            probs = torch.softmax(outputs.logits, dim=-1)[0]

        tokens = tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])
        predicted_labels = [model.config.id2label[p.item()] for p in predictions]
        max_probs = torch.max(probs, dim=-1)[0]

        print(f"   Text: '{sample_text}'")
        print(f"   Tokens: {tokens}")
        print(f"   Predictions: {predicted_labels}")
        print(f"   Max probabilities: {[f'{p:.3f}' for p in max_probs.tolist()]}")

        # Check if any non-O predictions
        non_o_preds = [i for i, label in enumerate(predicted_labels) if label != 'O']
        print(f"   Non-O predictions: {len(non_o_preds)} out of {len(predicted_labels)}")

        if non_o_preds:
            print("   ✅ Model found some entities!")
            for i in non_o_preds:
                print(f"      Token '{tokens[i]}' -> {predicted_labels[i]} (prob: {max_probs[i]:.3f})")
        else:
            print("   ❌ Model only predicts 'O' labels")

        # Method 2: Pipeline prediction
        print("\n3. Checking with pipeline...")
        try:
            nlp = pipeline(
                "ner",
                model=model,
                tokenizer=tokenizer,
                aggregation_strategy="simple",
                device=-1
            )
            pipeline_results = nlp(sample_text)

            print(f"   Pipeline results: {len(pipeline_results)} entities")
            if pipeline_results:
                for result in pipeline_results:
                    print(f"      {result['entity_group']}: '{result['word']}' (conf: {result['score']:.3f})")
            else:
                print("   ❌ Pipeline found no entities")

        except Exception as e:
            print(f"   ❌ Pipeline failed: {e}")

        # Method 3: Check without aggregation
        print("\n4. Checking without aggregation...")
        try:
            nlp_raw = pipeline(
                "ner",
                model=model,
                tokenizer=tokenizer,
                aggregation_strategy=None,
                device=-1
            )
            raw_results = nlp_raw(sample_text)

            non_o_raw = [r for r in raw_results if r['entity'] != 'O']
            print(f"   Raw results: {len(raw_results)} predictions, {len(non_o_raw)} non-O")

            if non_o_raw:
                print("   ✅ Found non-O predictions:")
                for r in non_o_raw[:5]:  # Show first 5
                    print(f"      '{r['word']}' -> {r['entity']} (conf: {r['score']:.3f})")
            else:
                print("   ❌ All predictions are 'O'")

        except Exception as e:
            print(f"   ❌ Raw pipeline failed: {e}")

        return True

    except Exception as e:
        print(f"❌ Failed to load or test model: {e}")
        import traceback
        traceback.print_exc()
        return False


def check_multiple_queries():
    """Check the specific queries that failed in comprehensive evaluation"""

    print("\n" + "=" * 60)
    print("CHECKING FAILED QUERIES")
    print("=" * 60)

    failed_queries = [
        "I need part number A101576",
        "Do you have filter tube 10/box?",
        "I'm looking for something from balston filt",
        "Can I get model 200-80-BX?",
        "Do you stock part A101576?"
    ]

    try:
        model_path = ORC_PARTS_MODEL_DIR
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForTokenClassification.from_pretrained(model_path)
        nlp = pipeline("ner", model=model, tokenizer=tokenizer, aggregation_strategy="simple", device=-1)

        for i, query in enumerate(failed_queries, 1):
            print(f"\nQuery {i}: '{query}'")
            results = nlp(query)

            if results:
                print(f"   ✅ Found {len(results)} entities:")
                for r in results:
                    print(f"      {r['entity_group']}: '{r['word']}' (confidence: {r['score']:.3f})")
            else:
                print("   ❌ No entities found")

    except Exception as e:
        print(f"❌ Checking failed: {e}")


def main():
    print("=" * 60)
    print("NER MODEL BASIC CHECK")
    print("=" * 60)

    # Basic functionality check
    success = check_model_basic()

    if success:
        # Check specific failed queries
        check_multiple_queries()

    print("\n" + "=" * 60)
    print("TEST COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()