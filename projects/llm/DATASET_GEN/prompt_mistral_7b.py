# -*- coding: utf-8 -*-
import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
# ------------------------------------------------------------
# CONFIG
# ------------------------------------------------------------
MODEL_DIR = r"E:\emtac\models\llm\mistral_7b_v03"
# Force offline
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_HUB_OFFLINE"] = "1"
# ------------------------------------------------------------
# Load tokenizer
# ------------------------------------------------------------
print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(
    MODEL_DIR,
    local_files_only=True,
    use_fast=True
)
# ------------------------------------------------------------
# Load model on GPU
# ------------------------------------------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Loading model on {device}...")

model = AutoModelForCausalLM.from_pretrained(
    MODEL_DIR,
    local_files_only=True,
    torch_dtype=torch.float16 if device == "cuda" else torch.float32
)
model.to(device)
model.eval()
# ------------------------------------------------------------
# Simple prompt loop
# ------------------------------------------------------------
while True:
    user_input = input("\nYour prompt: ").strip()
    if user_input.lower() in ["exit", "quit"]:
        break
    inputs = tokenizer(user_input, return_tensors="pt").to(device)
    print("\nGenerating…")
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=200,
            temperature=0.7,
            do_sample=True
        )
    result = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print("\n=== MODEL RESPONSE ===")
    print(result)
