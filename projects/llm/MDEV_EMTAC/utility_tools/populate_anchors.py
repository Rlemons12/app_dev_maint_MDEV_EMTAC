"""
populate_anchors.py
-------------------

Loads high-volume anchor sentences from Excel file and populates the
intent_anchor PostgreSQL table with MiniLM embeddings.

Excel file structure (required):
    Category        -> intent name
    Anchor Text     -> example anchor phrase

Steps:
  1. Read Excel anchor list
  2. Normalize + dedupe + merge Troubleshooting categories
  3. Generate MiniLM embeddings from offline local model
  4. Insert into intent_anchor table

Run:
    python populate_anchors.py
"""

import sys
import os
from pprint import pprint

import pandas as pd
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer

from modules.configuration.config_env import DatabaseConfig
from modules.emtac_ai.intent_ner.models.intent_anchor import IntentAnchor


# ============================================================
# Load .env first
# ============================================================
load_dotenv()

MINILM_PATH = os.getenv("MODEL_MINILM_DIR")   # e.g., E:\emtac\models\cache\all-MiniLM-L6-v2

if not MINILM_PATH or not os.path.exists(MINILM_PATH):
    print(f"ERROR: MiniLM model path is missing or invalid: {MINILM_PATH}")
    print("Set MODEL_MINILM_DIR in your .env file.")
    sys.exit(1)

EXCEL_FILE = r"E:\emtac\projects\llm\MDEV_EMTAC\Database\DB_LOADSHEETS\ancho_texts.xlsx"

# Force offline HF
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_DATASETS_OFFLINE"] = "1"


# ============================================================
# 1. Load anchors from Excel (REPLACES HARD-CODED DICT)
# ============================================================
def load_excel_anchors():
    if not os.path.exists(EXCEL_FILE):
        raise FileNotFoundError(f"Excel anchor file not found: {EXCEL_FILE}")

    print(f"\nLoading anchors from Excel: {EXCEL_FILE}")

    df = pd.read_excel(EXCEL_FILE)

    required_cols = {"Category", "Anchor Text"}
    if not required_cols.issubset(df.columns):
        raise ValueError(
            f"Excel must contain columns {required_cols}, but found {df.columns.tolist()}"
        )

    # Normalize text
    df["Category"] = df["Category"].astype(str).str.strip()
    df["Anchor Text"] = df["Anchor Text"].astype(str).str.strip()

    # Merge enhanced category → troubleshooting
    df.loc[df["Category"].str.lower() == "troubleshooting_enhanced",
           "Category"] = "Troubleshooting"

    # Remove exact duplicates
    df = df.drop_duplicates()

    # Group by category into dictionary of lists
    anchors = {}
    for _, row in df.iterrows():
        intent = row["Category"]
        text = row["Anchor Text"]

        if intent not in anchors:
            anchors[intent] = []

        anchors[intent].append(text)

    print(f"Loaded {sum(len(v) for v in anchors.values())} total anchors.")
    print(f"Detected {len(anchors)} unique intents.\n")

    return anchors


# ============================================================
# 2. Load MiniLM encoder
# ============================================================
def load_embedder():
    print(f"Loading MiniLM model from: {MINILM_PATH}\n")

    try:
        model = SentenceTransformer(MINILM_PATH)
        print("MiniLM loaded successfully.\n")
        return model
    except Exception as e:
        print(f"ERROR loading model: {e}")
        sys.exit(1)


# ============================================================
# 3. Populate DB
# ============================================================
def populate():
    anchors = load_excel_anchors()
    embedder = load_embedder()

    rows_to_insert = []

    print("Encoding anchor texts into embeddings...\n")

    for intent, sentences in anchors.items():
        for sentence in sentences:
            embedding = embedder.encode(sentence).tolist()

            rows_to_insert.append({
                "intent": intent,
                "text": sentence,
                "embedding": embedding
            })

    print(f"Prepared {len(rows_to_insert)} rows for insertion.\n")

    # Insert into DB using IntentAnchor.add_many
    try:
        IntentAnchor.add_many(rows_to_insert)
        print("SUCCESS: Anchor embeddings inserted.\n")

    except Exception as e:
        print(f"ERROR inserting into DB: {e}")
        sys.exit(1)

    print("Anchor counts per intent:")
    pprint({k: len(v) for k, v in anchors.items()})


# ============================================================
# Entry Point
# ============================================================
if __name__ == "__main__":
    populate()
