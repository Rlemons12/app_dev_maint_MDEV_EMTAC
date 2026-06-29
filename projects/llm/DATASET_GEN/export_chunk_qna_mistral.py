#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Generate Q&A pairs from all chunks in the QNA_CHUNKS table
using the offline Mistral-7B model.

Output: chunks_qna_export.txt
"""

import os
import sys
import torch
from datetime import datetime
from pathlib import Path

from transformers import AutoTokenizer, AutoModelForCausalLM

# ------------------------------------------------------------
# EMTAC DB IMPORTS (your existing database stack)
# ------------------------------------------------------------
sys.path.append(str(Path(__file__).resolve().parents[2]))  # add project root

from option_c_qna.qanda_db.qna_service import QADatabaseService
from option_c_qna.configuration.config import cfg


# ------------------------------------------------------------
# MODEL PATH
# ------------------------------------------------------------
MODEL_DIR = r"E:\emtac\models\llm\mistral_7b_v03"

# Offline mode
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_HUB_OFFLINE"] = "1"


# ------------------------------------------------------------
# LOAD TOKENIZER + MODEL
# ------------------------------------------------------------
print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(
    MODEL_DIR,
    local_files_only=True,
    use_fast=True
)

print("Loading model... (CPU mode)")
model = AutoModelForCausalLM.from_pretrained(
    MODEL_DIR,
    local_files_only=True,
    dtype=torch.float32
).to("cpu")
model.eval()


# ------------------------------------------------------------
# Q&A GENERATION FUNCTION
# ------------------------------------------------------------
def generate_qa(chunk_text: str, num_questions: int = 1) -> str:
    """
    Creates Q&A pairs using Mistral-7B.
    """
    prompt = f"""
You are an AI assistant trained to create high-quality factual questions
strictly from a given text chunk.

CHUNK:
\"\"\"{chunk_text}\"\"\"


TASK:
Generate {num_questions} Question and Answer pairs that rely ONLY on the content of the CHUNK.

FORMAT EXACTLY AS:
Question: <question>
Answer: <answer>
"""

    inputs = tokenizer(prompt, return_tensors="pt").to("cpu")

    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=300,
            temperature=0.35,
            top_p=0.95,
            do_sample=True
        )

    decoded = tokenizer.decode(output[0], skip_special_tokens=True)
    return decoded


# ------------------------------------------------------------
# MAIN EXECUTION
# ------------------------------------------------------------
def main():
    print("\n=== CHUNK → Q&A EXPORT (Mistral 7B) ===\n")

    # DB service
    db = QADatabaseService()

    print("Loading chunks from database...")
    chunks = db.session.execute("""
        SELECT id, chunk_text
        FROM qna_chunks
        ORDER BY id;
    """).fetchall()

    print(f"Found {len(chunks)} chunks.")

    # Output file
    outfile = Path("chunks_qna_export.txt")
    fout = outfile.open("w", encoding="utf-8")

    fout.write(f"Q&A Export Generated: {datetime.now()}\n")
    fout.write("Model: Mistral-7B\n")
    fout.write("==========================================\n\n")

    # Process each chunk
    for chunk_id, chunk_text in chunks:
        print(f"Processing chunk {chunk_id}...")

        qa_text = generate_qa(chunk_text)

        fout.write(f"--- Chunk ID: {chunk_id} ---\n")
        fout.write(qa_text)
        fout.write("\n\n")

    fout.close()

    print("\nExport complete!")
    print(f"Saved to: {outfile.resolve()}\n")


if __name__ == "__main__":
    main()
