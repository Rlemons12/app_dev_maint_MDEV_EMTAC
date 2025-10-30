import json
import random
from pathlib import Path

# Import your custom logging functions
from modules.configuration.log_config import info_id, debug_id, error_id

def generate_image_training_data():
    """Generate balanced training data for image intent classification"""

    training_examples = []

    # === POSITIVE EXAMPLES (images intent) ===

    # Template-based generation with better grammar
    image_templates = [
        "show me {article} {descriptor} {subject} {context}",
        "can I see {article} {descriptor} {subject} {context}",
        "display {article} {descriptor} {subject} {context}",
        "I want to see {article} {descriptor} {subject} {context}",
        "I need to view {article} {descriptor} {subject} {context}",
        "pull up {article} {descriptor} {subject} {context}",
        "bring up {article} {descriptor} {subject} {context}",
        "I'd like to see {article} {descriptor} {subject} {context}",
        "can you show {article} {descriptor} {subject} {context}",
        "I want to look at {article} {descriptor} {subject} {context}",
    ]

    # Better organized subjects with appropriate articles
    image_subjects = [
        {"subject": "image", "article": "the"},
        {"subject": "picture", "article": "the"},
        {"subject": "photo", "article": "the"},
        {"subject": "diagram", "article": "the"},
        {"subject": "schematic", "article": "the"},
        {"subject": "blueprint", "article": "the"},
        {"subject": "drawing", "article": "the"},
        {"subject": "illustration", "article": "the"},
        {"subject": "chart", "article": "the"},
        {"subject": "visual", "article": "the"},
        {"subject": "documentation", "article": "the"},
        {"subject": "specs", "article": "the"},
        {"subject": "images", "article": ""},
        {"subject": "pictures", "article": ""},
        {"subject": "diagrams", "article": ""},
        {"subject": "visuals", "article": ""},
    ]

    descriptors = ["", "detailed", "technical", "exploded", "3D", "annotated", "labeled"]
    contexts = ["", "of this part", "for this component", "of the assembly", "for maintenance", "for installation"]

    # Generate template-based examples
    for _ in range(200):  # Reduced from 500
        template = random.choice(image_templates)
        subject_info = random.choice(image_subjects)
        descriptor = random.choice(descriptors)
        context = random.choice(contexts)

        text = template.format(
            article=subject_info["article"],
            descriptor=descriptor,
            subject=subject_info["subject"],
            context=context
        )

        # Clean up extra spaces and articles
        text = " ".join(text.split())
        text = text.replace(" the ", " ").replace("the ", "") if not subject_info["article"] else text

        training_examples.append({
            "text": text.capitalize(),
            "intent": "images"
        })

    # High-quality hand-crafted positive examples
    positive_examples = [
        "Can I get a visual of this?",
        "Show me what this looks like",
        "I need the technical drawings",
        "Pull up the part diagram",
        "Display the schematic",
        "I want to see the blueprint",
        "Get me the visual documentation",
        "Show the exploded view",
        "I need to see the assembly drawing",
        "Can you display the part image?",
        "Bring up the picture",
        "Show me the visual specs",
        "I want the part photo",
        "Can I see the component diagram?",
        "Show the cross-section view",
        "I need the maintenance diagram",
        "Show me the installation drawing",
        "I want to see how it looks",
        "Can you show me what it is?",
        "I need a picture of this",
        "Show me the visual reference",
        "I want to see the technical drawing",
        "Can you pull up the image?",
        "Show the visual information",
        "I need to see what this is",
        "Display the part picture",
        "Show me the component view",
        "I want the technical visualization",
        "Show the visual breakdown",
        "Display what this looks like",
        "Show me the part layout",
        "I want to see the structure",
        "Can you show the visual?",
        "Show me the part breakdown",
        "I need the visual reference",
        "Can I see how it's assembled?",
        "Show the part configuration",
        # Short/casual variations
        "pic please",
        "show pic",
        "image?",
        "visual please",
        "diagram please",
        "show image",
        "need visual",
        "pic of this",
        "show drawing",
        "see diagram"
    ]

    for text in positive_examples:
        training_examples.append({
            "text": text,
            "intent": "images"
        })

    # === NEGATIVE EXAMPLES (non-image intents) ===

    # Parts/specification requests (not visual)
    parts_examples = [
        "What is the part number?",
        "I need the part number for this",
        "What's the SKU?",
        "Give me the model number",
        "What part is this?",
        "I need the part specifications",
        "What are the dimensions?",
        "Tell me about this component",
        "What is this part called?",
        "I need the part details",
        "What's the description?",
        "Give me the part info",
        "I need technical specifications",
        "What are the specs?",
        "Tell me the measurements",
        "What material is this?",
        "I need the datasheet",
        "What's the weight?",
        "Give me the properties",
        "What's the compatibility?"
    ]

    # General help/support requests
    help_examples = [
        "I need help",
        "Can you assist me?",
        "I have a question",
        "I'm having trouble",
        "How do I use this?",
        "What should I do?",
        "I need support",
        "Can you help me?",
        "I don't understand",
        "I'm confused",
        "How does this work?",
        "I need assistance",
        "Can you explain?",
        "I have an issue",
        "Something's wrong"
    ]

    # Pricing/availability requests
    business_examples = [
        "How much does this cost?",
        "What's the price?",
        "Is this in stock?",
        "When will it ship?",
        "How long for delivery?",
        "Where can I buy this?",
        "I want to order this",
        "Add to cart",
        "I need pricing",
        "What's available?",
        "Can I purchase this?",
        "I want to buy",
        "Quote please",
        "Check availability",
        "Order status"
    ]

    # Add negative examples with appropriate labels
    for text in parts_examples:
        training_examples.append({"text": text, "intent": "parts"})

    for text in help_examples:
        training_examples.append({"text": text, "intent": "help"})

    for text in business_examples:
        training_examples.append({"text": text, "intent": "business"})

    return training_examples


def save_training_data(examples, output_path):
    """Save training examples to JSONL file"""
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w', encoding='utf-8') as f:
        for example in examples:
            f.write(json.dumps(example) + '\n')

    info_id(f"Saved {len(examples)} training examples to {output_path}")

    # Print distribution
    intent_counts = {}
    for example in examples:
        intent = example['intent']
        intent_counts[intent] = intent_counts.get(intent, 0) + 1

    info_id("Intent distribution:")
    for intent, count in intent_counts.items():
        info_id(f"  {intent}: {count}")


def create_multi_class_training_script():
    """Generate training script for multi-class classification"""
    script_content = '''import os
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import torch
import numpy as np

# Paths
DATA_PATH = r"modules/emtac_ai/training_data/datasets/images/intent_train_images.jsonl"
MODEL_SAVE_PATH = r"modules/emtac_ai/models/intent_classifier_multi"

# Load dataset
dataset = load_dataset('json', data_files=DATA_PATH)
train_dataset = dataset['train']

# Load tokenizer and model
MODEL_NAME = "distilbert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

# Get unique labels and create mapping
unique_labels = list(set(train_dataset['intent']))
label_to_id = {label: idx for idx, label in enumerate(unique_labels)}
id_to_label = {idx: label for label, idx in label_to_id.items()}
num_labels = len(unique_labels)

print(f"Found {num_labels} labels: {unique_labels}")
print(f"Label mapping: {label_to_id}")

model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME, 
    num_labels=num_labels,
    label2id=label_to_id,
    id2label=id_to_label
)

# Preprocessing
def preprocess_function(examples):
    return tokenizer(examples['text'], truncation=True, padding='max_length', max_length=128)

def label_to_int(examples):
    examples["labels"] = [label_to_id[label] for label in examples["intent"]]
    return examples

tokenized_train = train_dataset.map(preprocess_function, batched=True)
tokenized_train = tokenized_train.map(label_to_int, batched=True)
tokenized_train.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])

# Metrics
def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)

    precision, recall, f1, _ = precision_recall_fscore_support(labels, predictions, average='weighted')
    accuracy = accuracy_score(labels, predictions)

    return {
        'accuracy': accuracy,
        'f1': f1,
        'precision': precision,
        'recall': recall
    }

# Training arguments
training_args = TrainingArguments(
    output_dir=MODEL_SAVE_PATH,
    num_train_epochs=5,
    per_device_train_batch_size=16,
    evaluation_strategy="no",
    save_strategy="epoch",
    logging_dir='./logs',
    logging_steps=10,
    save_total_limit=2,
    load_best_model_at_end=False,
    seed=42,
    warmup_steps=100,
    weight_decay=0.01,
)

# Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
)

# Train
trainer.train()

# Save
trainer.save_model(MODEL_SAVE_PATH)
tokenizer.save_pretrained(MODEL_SAVE_PATH)

print(f"Model and tokenizer saved to {MODEL_SAVE_PATH}")
'''
    return script_content



if __name__ == "__main__":
    # Generate training data
    training_data = generate_image_training_data()

    # Shuffle the data
    random.shuffle(training_data)

    # Save to file
    output_path = "modules/emtac_ai/training_data/datasets/images/intent_train_images.jsonl"
    save_training_data(training_data, output_path)

    # Print some examples
    info_id("Sample training examples:")
    for intent in ["images", "parts", "help", "business"]:
        examples = [ex for ex in training_data if ex["intent"] == intent][:3]
        info_id(f"{intent.upper()} examples:")
        for ex in examples:
            info_id(f"  '{ex['text']}'")

    # Create training script
    script_content = create_multi_class_training_script()
    with open("train_multi_intent_classifier.py", "w") as f:
        f.write(script_content)

    info_id("Created training script: train_multi_intent_classifier.py")
