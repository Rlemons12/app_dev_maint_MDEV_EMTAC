"""
EMTAC RAG STRESS TEST — 5 questions end-to-end

Measures:
  • Embedding latency
  • Retrieval latency
  • Context-building latency
  • Answer generation latency
  • Total pipeline latency
  • Chunk + token usage

Run:
    python rag_stress_test.py
"""

import time
from statistics import mean

from modules.emtac_ai.search.rag_core.embedder import DBConfiguredEmbedder
from modules.emtac_ai.search.rag_core.retriever import PgVectorRetriever
from modules.emtac_ai.search.rag_core.context_builder import ContextBuilder
from modules.emtac_ai.search.rag_core.answer_generator import DBConfiguredAnswerGenerator
from modules.emtac_ai.search.rag_core.rag_pipeline import RAGPipeline


# ------------------------------------------------------------
# 5-QUESTION SET FOR STRESS TESTING
# ------------------------------------------------------------
QUESTIONS = [
    "What steps are needed to replace a fill nozzle?",
    "How do I troubleshoot a port not feeding correctly?",
    "What should I check if the bag clamp cylinder is not actuating?",
    "Explain how the vial port mandrel assembly works.",
    "How do I perform a basic calibration on the fill station sensors?"
]


# ------------------------------------------------------------
# UTILITY — TIMED EXECUTION WRAPPER
# ------------------------------------------------------------
def timed(fn, *args, **kwargs):
    t0 = time.time()
    result = fn(*args, **kwargs)
    t1 = time.time()
    return result, (t1 - t0)


# ------------------------------------------------------------
# MAIN STRESS TEST
# ------------------------------------------------------------
def run_stress_test():
    print("\n===================================================")
    print("         EMTAC RAG STRESS TEST — 5 QUESTIONS       ")
    print("===================================================\n")

    embedder = DBConfiguredEmbedder()
    retriever = PgVectorRetriever()
    ctx_builder = ContextBuilder(max_tokens=4000)
    answer_gen = DBConfiguredAnswerGenerator()
    pipeline = RAGPipeline()

    stats = {
        "embed_times": [],
        "retrieval_times": [],
        "context_times": [],
        "answer_times": [],
        "total_times": [],
        "chunks_used": []
    }

    # --------------------------------------------------------
    # PROCESS EACH QUESTION
    # --------------------------------------------------------
    for i, q in enumerate(QUESTIONS, start=1):
        print(f"\n---------------------------------------------------")
        print(f"QUESTION {i}: {q}")
        print("---------------------------------------------------")

        # --- 1. EMBEDDING ---
        vec, t_embed = timed(embedder.embed_query, q)
        print(f"Embedding time: {t_embed:.3f}s")

        # --- 2. RETRIEVAL ---
        chunks, t_retrieve = timed(retriever.retrieve, vec, 5)
        print(f"Retrieval time: {t_retrieve:.3f}s (chunks={len(chunks)})")

        # --- 3. CONTEXT BUILD ---
        ctx_obj, t_ctx = timed(ctx_builder.build_context, chunks)
        print(f"Context building time: {t_ctx:.3f}s | "
              f"Chunks used: {len(ctx_obj['used_chunks'])}")

        # --- 4. ANSWER GEN ---
        answer_obj, t_ans = timed(answer_gen.generate_answer, q, ctx_obj["context"])
        short_answer = answer_obj.get("answer", "")[:200].replace("\n", " ")
        print(f"Answer generation time: {t_ans:.3f}s")
        print(f"Answer preview: {short_answer}...")

        # --- 5. FULL PIPELINE (single call) ---
        _, t_total = timed(pipeline.run, q, 5)
        print(f"Full pipeline time: {t_total:.3f}s")

        # Save stats
        stats["embed_times"].append(t_embed)
        stats["retrieval_times"].append(t_retrieve)
        stats["context_times"].append(t_ctx)
        stats["answer_times"].append(t_ans)
        stats["total_times"].append(t_total)
        stats["chunks_used"].append(len(ctx_obj["used_chunks"]))

    # --------------------------------------------------------
    # SUMMARY REPORT
    # --------------------------------------------------------
    print("\n===================================================")
    print("                 STRESS TEST SUMMARY               ")
    print("===================================================\n")

    print(f"Average embedding time:       {mean(stats['embed_times']):.3f}s")
    print(f"Average retrieval time:       {mean(stats['retrieval_times']):.3f}s")
    print(f"Average context build time:   {mean(stats['context_times']):.3f}s")
    print(f"Average answer time:          {mean(stats['answer_times']):.3f}s")
    print(f"Average full pipeline time:   {mean(stats['total_times']):.3f}s")
    print(f"Average chunks used:          {mean(stats['chunks_used']):.2f}")

    print("\nDONE.\n")


# ------------------------------------------------------------
# RUN IF INVOKED DIRECTLY
# ------------------------------------------------------------
if __name__ == "__main__":
    run_stress_test()
