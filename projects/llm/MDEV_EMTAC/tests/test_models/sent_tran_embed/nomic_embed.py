import time
import numpy as np
from sentence_transformers import SentenceTransformer

def cosine_similarity(vec1, vec2):
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

def main():
    print("[INFO] Loading model...")
    start = time.time()
    model = SentenceTransformer("nomic-ai/nomic-embed-text-v1", trust_remote_code=True)
    print(f"[INFO] Model loaded in {time.time() - start:.2f} sec\n")

    # Example inputs
    doc = "The motor's output shaft drives a gearbox that reduces speed."
    query = "What does a gearbox do in a motor system?"

    print("[INFO] Embedding inputs...")
    start = time.time()
    doc_embed = model.encode([doc])[0]
    query_embed = model.encode([query])[0]
    print(f"[INFO] Embeddings generated in {time.time() - start:.2f} sec\n")

    # Similarity
    similarity = cosine_similarity(doc_embed, query_embed)

    print("📌 Embedding dimension:", len(doc_embed))
    print(f"📈 Cosine similarity: {similarity:.4f}")
    print("✅ Test complete.")

if __name__ == "__main__":
    main()
