import transformers
print("Transformers version:", transformers.__version__)

from transformers import TrainingArguments
print("TrainingArguments loaded from:", TrainingArguments.__module__)

args = TrainingArguments(
    output_dir='./test_model',
    evaluation_strategy='no',
    per_device_train_batch_size=8,
    num_train_epochs=1,
)

print("TrainingArguments loaded successfully")
