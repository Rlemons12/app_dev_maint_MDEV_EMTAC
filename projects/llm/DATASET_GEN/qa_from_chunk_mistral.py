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
# Load model (CPU-only, no device_map)
# ------------------------------------------------------------
print("Loading model... (this may take a while on CPU)")
model = AutoModelForCausalLM.from_pretrained(
    MODEL_DIR,
    local_files_only=True,
    dtype=torch.float32
)

model.to("cpu")
model.eval()

# ------------------------------------------------------------
# Q&A GENERATION FUNCTION
# ------------------------------------------------------------
def generate_qa(chunk: str, num_questions: int = 1) -> str:
    """
    Creates 1 or more Q&A pairs based solely on the provided chunk.
    """

    prompt = f"""
You are an assistant trained to generate Q&A pairs from text.

CHUNK:
\"\"\"{chunk}\"\"\"

TASK:
Generate {num_questions} question and answer pairs based ONLY on the content above.

FORMAT STRICTLY AS:
Question: <question>
Answer: <answer>
"""

    inputs = tokenizer(prompt, return_tensors="pt").to("cpu")

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=250,
            temperature=0.4,
            do_sample=True
        )

    full_output = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    return full_output


# ------------------------------------------------------------
# MAIN LOOP
# ------------------------------------------------------------
print("\nReady. Paste a text chunk to generate a Q&A pair.")
print("Type 'exit' to quit.\n")

while True:
    print("\nPaste your chunk below (press Enter when done):")
    chunk = input("> ").strip()

    if chunk.lower() in ["exit", "quit"]:
        break

    print("\nGenerating Q&A…")
    result = generate_qa(chunk, num_questions=1)

    print("\n=== Q&A RESULT ===")
    print(result)
    print("==================\n")
