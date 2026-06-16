import os
import datetime
from modules.emtac_transformer.hf_intent_entity_plugin import HFIntentEntityPlugin

def test_inference():
    print("=== INTERACTIVE NER Inference ===")
    base_dir = os.path.dirname(os.path.dirname(__file__))
    model_dir = os.path.join(base_dir, "models", "ner-custom-test")
    plugin = HFIntentEntityPlugin(model_dir=model_dir)
    log_path = os.path.join(base_dir, "ner_user_log.txt")

    while True:
        text = input("\nEnter a sentence to analyze (or type 'exit' to quit):\n> ").strip()
        if text.lower() in {"exit", "quit"}:
            print("Exiting NER test.")
            break

        entities = plugin.extract_entities(text)

        if entities:
            print("Entities found:")
            for ent in entities:
                print(
                    f" - {ent.get('word', '')} [{ent.get('entity_group', '')}] (score={float(ent.get('score', 0)):.2f})")
        else:
            print("No entities found.")

        # Log input and results with timestamp
        with open(log_path, "a", encoding="utf-8") as logf:
            logf.write(f"{datetime.datetime.now().isoformat()} | Input: {text}\nEntities: {entities}\n\n")


def test_training():
    print("\n=== TEST: Training Run (DRY RUN) ===")
    base_dir = os.path.dirname(os.path.dirname(__file__))
    model_dir = os.path.join(base_dir, "models", "bert-base-ner-extracted")
    # UPDATE this path to where your generated file is:
    train_data_path = os.path.join(
        base_dir, "training_data", "datasets", "parts_requests", "parts_ner_train.jsonl"
    )
    try:
        plugin = HFIntentEntityPlugin(model_dir=model_dir)
        plugin.train(
            train_data=train_data_path,
            output_dir=os.path.join(base_dir, "ner-custom-test"),
            epochs=1
        )
        print("Training test completed! (Check 'ner-custom-test')")
    except Exception as e:
        print("Training test failed:", e)

if __name__ == "__main__":
    test_inference()
    # Uncomment this line to test training:
    #test_training()
