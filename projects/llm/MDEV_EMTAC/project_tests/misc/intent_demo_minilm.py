"""
Intent Demo using PGVector + MiniLM
-----------------------------------

THIS VERSION LOADS ANCHORS FROM:
E:\emtac\projects\llm\MDEV_EMTAC\Database\DB_LOADSHEETS\ancho_texts.xlsx

The spreadsheet must contain columns:
    Category      → intent name
    Anchor Text   → example text phrase

At startup:
- Loads Excel
- Groups rows by Category
- Encodes each Anchor Text using MiniLM
- Stores tensor embeddings
- Performs semantic similarity intent classification
"""

import os
import torch
import pandas as pd
from sentence_transformers import SentenceTransformer, util

# ----------------------------------------------------------------------
# 1. MODEL PATH
# ----------------------------------------------------------------------
MODEL_PATH = r"E:\emtac\models\cache\all-MiniLM-L6-v2"

print("Loading MiniLM model...")
model = SentenceTransformer(MODEL_PATH)


# ----------------------------------------------------------------------
# 2. Load anchors from Excel instead of DB
# ----------------------------------------------------------------------
EXCEL_FILE = r"E:\emtac\projects\llm\MDEV_EMTAC\Database\DB_LOADSHEETS\ancho_texts.xlsx"

def load_intent_anchors_from_excel():
    if not os.path.exists(EXCEL_FILE):
        raise FileNotFoundError(f"Excel file not found: {EXCEL_FILE}")

    print(f"Loading anchors from Excel: {EXCEL_FILE}")

    df = pd.read_excel(EXCEL_FILE)

    # Validate expected columns
    required_cols = {"Category", "Anchor Text"}
    if not required_cols.issubset(set(df.columns)):
        raise ValueError(
            f"Excel must contain columns: {required_cols}, "
            f"found: {list(df.columns)}"
        )

    # Convert NaN to empty strings
    df["Category"] = df["Category"].astype(str).str.strip()
    df["Anchor Text"] = df["Anchor Text"].astype(str).str.strip()

    intents = {}

    for _, row in df.iterrows():
        intent = row["Category"]
        text = row["Anchor Text"]

        if intent not in intents:
            intents[intent] = []

        # Append anchor text (we compute embeddings later)
        intents[intent].append(text)

    print(f"Loaded {sum(len(v) for v in intents.values())} anchors "
          f"across {len(intents)} categories (intents).")

    return intents


ANCHOR_TEXTS = load_intent_anchors_from_excel()


# ----------------------------------------------------------------------
# 3. Convert anchor texts → embeddings (torch tensors)
# ----------------------------------------------------------------------
def encode_anchors(intent_dict):
    anchor_tensors = {}

    print("Encoding anchor texts into embeddings...")

    for intent, phrases in intent_dict.items():
        embeddings = model.encode(phrases, convert_to_tensor=True)
        anchor_tensors[intent] = embeddings  # shape: [N, 384]

    print("Embedding preparation complete.")
    return anchor_tensors


ANCHOR_TENSORS = encode_anchors(ANCHOR_TEXTS)


# ----------------------------------------------------------------------
# 4. Intent scoring
# ----------------------------------------------------------------------
def classify_intent(text):
    """
    Encodes user query → compares to anchor embeddings → returns best match.
    """
    query_vec = model.encode(text, convert_to_tensor=True)

    scores = {}

    for intent, tensor in ANCHOR_TENSORS.items():
        sims = util.cos_sim(query_vec, tensor)  # shape [1, N]
        mean_sim = float(sims.mean())
        scores[intent] = mean_sim

    best_intent = max(scores, key=scores.get)
    return best_intent, scores[best_intent], scores


# ----------------------------------------------------------------------
# 5. Interactive CLI Demo
# ----------------------------------------------------------------------
if __name__ == "__main__":
    print("\nIntent Classification Demo (Excel-Powered)")
    print("Loaded Intents:", ", ".join(ANCHOR_TEXTS.keys()))
    print("Type 'exit' to quit.\n")

    while True:
        q = input("Ask something: ").strip()
        if q.lower() in ("exit", "quit"):
            break

        intent, score, breakdown = classify_intent(q)

        print("\n--- Similarity Scores ---")
        for k, v in breakdown.items():
            print(f"{k:20s}: {v:.4f}")

        print("\nPredicted Intent:", intent)
        print(f"Confidence: {score:.4f}\n")
