from transformers import AutoModelForTokenClassification, AutoTokenizer

model_name = "dslim/bert-base-NER"
save_dir = "../models/bert-base-ner-extracted"

tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir="./bert-base-ner-files")
model = AutoModelForTokenClassification.from_pretrained(model_name, cache_dir="./bert-base-ner-files")

# Save all files to the extracted directory in flat format
tokenizer.save_pretrained(save_dir)
model.save_pretrained(save_dir)

print("Saved flat model files to:", save_dir)
