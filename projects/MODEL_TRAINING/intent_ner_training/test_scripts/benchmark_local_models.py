import os
import json
import re
from transformers import T5Tokenizer, T5ForConditionalGeneration
import torch
from tqdm import tqdm
from modules.configuration.log_config import logger
from modules.configuration.config_env import DatabaseConfig  # Ensures .env is loaded

# Optional imports for file parsing
try:
    import fitz  # PyMuPDF
    from docx import Document
except ImportError:
    fitz = None
    Document = None

# =========================================
# ENVIRONMENT INTEGRATION
# =========================================
ORC_BASE_DIR = os.getenv("ORC_BASE_DIR", r"E:\emtac\models")
ORC_TRAINING_DATA_DIR = os.path.join(ORC_BASE_DIR, "training_data", "datasets")
MODEL_DIR = os.getenv("MODEL_mrm8488_DIR", r"E:\emtac\models\cache\t5-base-finetuned-question-generation-ap")

os.makedirs(ORC_TRAINING_DATA_DIR, exist_ok=True)
logger.info(f"[QG] Training datasets directory: {ORC_TRAINING_DATA_DIR}")
logger.info(f"[QG] Using model directory: {MODEL_DIR}")

# =========================================
# MODEL SETUP
# =========================================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

try:
    logger.info("[QG] Loading model and tokenizer...")
    tokenizer = T5Tokenizer.from_pretrained(MODEL_DIR)
    model = T5ForConditionalGeneration.from_pretrained(MODEL_DIR)
    model.to(DEVICE)
    model.eval()
    logger.info("[QG] Model loaded successfully.")
except Exception as e:
    logger.error(f"[QG] Failed to load model from {MODEL_DIR}: {e}")
    raise

# =========================================
# UTILITIES
# =========================================

def clean_text(text):
    """Normalize whitespace and remove junk characters."""
    return re.sub(r"\s+", " ", text.strip())

def chunk_text(text, max_tokens=256):
    """Split text into smaller, model-friendly chunks."""
    sentences = re.split(r'(?<=[.?!])\s+', text)
    chunks, current_chunk = [], ""
    for sentence in sentences:
        token_len = len(tokenizer.encode(current_chunk + " " + sentence))
        if token_len < max_tokens:
            current_chunk += " " + sentence
        else:
            chunks.append(current_chunk.strip())
            current_chunk = sentence
    if current_chunk:
        chunks.append(current_chunk.strip())
    return chunks

def generate_question(context):
    """Generate multiple questions for a given passage."""
    input_text = f"generate question: {context}"
    inputs = tokenizer.encode(input_text, return_tensors="pt", truncation=True).to(DEVICE)
    outputs = model.generate(
        inputs,
        max_length=64,
        num_beams=5,
        num_return_sequences=3,
        early_stopping=True
    )
    questions = [tokenizer.decode(o, skip_special_tokens=True) for o in outputs]
    return list(set(questions))

# =========================================
# FILE EXTRACTION HELPERS
# =========================================

def extract_text_from_file(file_path):
    """Detect file type and extract text accordingly."""
    ext = os.path.splitext(file_path)[1].lower()

    if ext == ".txt":
        with open(file_path, "r", encoding="utf-8") as f:
            return f.read()

    elif ext == ".docx":
        if not Document:
            raise ImportError("python-docx is not installed. Run 'pip install python-docx'")
        doc = Document(file_path)
        return "\n".join([para.text for para in doc.paragraphs])

    elif ext == ".pdf":
        if not fitz:
            raise ImportError("PyMuPDF (fitz) is not installed. Run 'pip install PyMuPDF'")
        text = ""
        with fitz.open(file_path) as pdf:
            for page in pdf:
                text += page.get_text()
        return text

    else:
        raise ValueError(f"Unsupported file type: {ext}")

# =========================================
# MAIN PIPELINE
# =========================================

def main():
    print("Enter the full path to your source file (.txt, .docx, or .pdf):")
    input_path = input("Source file path: ").strip('"')

    if not os.path.isfile(input_path):
        logger.error(f"[QG] File not found: {input_path}")
        print("File not found. Exiting.")
        return

    file_name = os.path.splitext(os.path.basename(input_path))[0]
    output_path = os.path.join(ORC_TRAINING_DATA_DIR, f"{file_name}_qna.jsonl")

    logger.info(f"[QG] Reading and extracting text from: {input_path}")
    try:
        raw_text = extract_text_from_file(input_path)
    except Exception as e:
        logger.error(f"[QG] Failed to extract text: {e}")
        print(f"Error reading file: {e}")
        return

    text = clean_text(raw_text)
    chunks = chunk_text(text)
    logger.info(f"[QG] Split input into {len(chunks)} chunks.")
    qna_data = []

    for chunk in tqdm(chunks, desc="Generating Q&A pairs"):
        try:
            questions = generate_question(chunk)
            for q in questions:
                answer = chunk.split(".")[0].strip()
                qna_data.append({
                    "context": chunk,
                    "question": q,
                    "answer": answer
                })
        except Exception as e:
            logger.warning(f"[QG] Skipped a chunk due to error: {e}")

    logger.info(f"[QG] Generated {len(qna_data)} Q&A pairs.")

    with open(output_path, "w", encoding="utf-8") as f:
        for item in qna_data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    logger.info(f"[QG] Saved dataset to {output_path}")
    print(f"Dataset created successfully: {output_path}")

# =========================================
# ENTRY POINT
# =========================================
if __name__ == "__main__":
    main()
