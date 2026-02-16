import os
from transformers import pipeline, AutoModelForSequenceClassification, AutoTokenizer

def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    model_dir = os.path.abspath(os.path.join(base_dir, "..", "..", "..", "models", "intent_classifier"))

    print(f"Loading intent classification model from {model_dir} ...")
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)

    classifier = pipeline("text-classification", model=model, tokenizer=tokenizer, return_all_scores=True)

    print("=== Intent Classification Test ===")
    print("Type a sentence to detect intent, or 'exit' to quit.\n")

    while True:
        text = input("> ").strip()
        if text.lower() in {"exit", "quit"}:
            print("Exiting.")
            break

        results = classifier(text)
        scores = results[0]
        sorted_scores = sorted(scores, key=lambda x: x["score"], reverse=True)

        print("Predicted intents (top 3):")
        for r in sorted_scores[:3]:
            print(f"  {r['label']}: {r['score']:.4f}")
        print()

if __name__ == "__main__":
    main()
