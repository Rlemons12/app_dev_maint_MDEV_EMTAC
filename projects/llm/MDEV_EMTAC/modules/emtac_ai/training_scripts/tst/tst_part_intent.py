import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch.nn.functional as F

# Path to your saved trained model
MODEL_DIR = r"C:\Users\cetax\emtact_transformer\modules\emtac_ai\models\parts"

def load_model_and_tokenizer(model_dir):
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    model.eval()  # Set to evaluation mode
    return tokenizer, model

def predict_intent(text, tokenizer, model):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
    logits = outputs.logits
    probs = F.softmax(logits, dim=1).squeeze()
    confidence, predicted_class_idx = torch.max(probs, dim=0)
    return predicted_class_idx.item(), confidence.item(), probs.tolist()


def main():
    tokenizer, model = load_model_and_tokenizer(MODEL_DIR)

    # Replace with your intent labels list in correct order
    intent_labels = ['parts', 'other_intent_1', 'other_intent_2']  # example, update with your labels!

    print("Part Intent Inference (type 'exit' to quit)")
    while True:
        user_input = input("Ask a part-related question: ").strip()
        if user_input.lower() in ('exit', 'quit'):
            break

        class_idx, confidence, all_probs = predict_intent(user_input, tokenizer, model)
        intent_name = intent_labels[class_idx] if class_idx < len(intent_labels) else "Unknown"

        print(f"Predicted Intent: {intent_name} (Confidence: {confidence:.3f})")
        print(f"All class probabilities: {all_probs}")

if __name__ == "__main__":
    main()
