"""
EMTAC RAG STRESS TEST — 5 questions end-to-end (TEST CONFIG MODE)
"""

import os
import time
from statistics import mean


# ============================================================
# FORCE TEST ENVIRONMENT BEFORE ANY PROJECT IMPORTS
# ============================================================

from test_config import (
    bootstrap_test_env,
    bootstrap_schema,
    truncate_all_tables,
)

bootstrap_test_env()

db_url = os.getenv("DATABASE_URL")

if "test" not in db_url.lower():
    raise RuntimeError(
        f"Refusing to run stress test on non-test DB:\n{db_url}"
    )

# ============================================================
# SAFE TO IMPORT PROJECT MODULES
# ============================================================

from modules.configuration.base import Base

# 🔥 CRITICAL — Import full ORM registry
from modules.orm_registry import register_all_models
register_all_models()


from modules.emtac_ai.search.rag_core.embedder import DBConfiguredEmbedder
from modules.emtac_ai.search.rag_core.retriever import PgVectorRetriever
from modules.emtac_ai.search.rag_core.context_builder import ContextBuilder
from modules.emtac_ai.search.rag_core.answer_generator import DBConfiguredAnswerGenerator
from modules.emtac_ai.search.rag_core.rag_pipeline import RAGPipeline


# ============================================================
# INITIALIZE DB
# ============================================================

engine = bootstrap_schema(Base)
truncate_all_tables(engine)


# ============================================================
# QUESTIONS
# ============================================================

QUESTIONS = [
    "What steps are needed to replace a fill nozzle?",
    "How do I troubleshoot a port not feeding correctly?",
    "What should I check if the bag clamp cylinder is not actuating?",
    "Explain how the vial port mandrel assembly works.",
    "How do I perform a basic calibration on the fill station sensors?"
]


# ============================================================
# TIMING WRAPPER
# ============================================================

def timed(fn, *args, **kwargs):
    t0 = time.time()
    result = fn(*args, **kwargs)
    t1 = time.time()
    return result, (t1 - t0)


# ============================================================
# MAIN
# ============================================================

def run_stress_test():

    print("\n===================================================")
    print("   EMTAC RAG STRESS TEST — TEST DATABASE MODE     ")
    print("===================================================\n")
    print(f"Using DB: {db_url}\n")

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

    for i, q in enumerate(QUESTIONS, start=1):

        print(f"\n---------------------------------------------------")
        print(f"QUESTION {i}: {q}")
        print("---------------------------------------------------")

        vec, t_embed = timed(embedder.embed_query, q)
        print(f"Embedding time: {t_embed:.3f}s")

        chunks, t_retrieve = timed(retriever.retrieve, vec, 5)
        print(f"Retrieval time: {t_retrieve:.3f}s (chunks={len(chunks)})")

        ctx_obj, t_ctx = timed(ctx_builder.build_context, chunks)
        print(
            f"Context build time: {t_ctx:.3f}s | "
            f"Chunks used: {len(ctx_obj['used_chunks'])}"
        )

        answer_obj, t_ans = timed(
            answer_gen.generate_answer,
            q,
            ctx_obj["context"]
        )

        short_answer = answer_obj.get("answer", "")[:200].replace("\n", " ")
        print(f"Answer generation time: {t_ans:.3f}s")
        print(f"Answer preview: {short_answer}...")

        _, t_total = timed(pipeline.run, q, 5)
        print(f"Full pipeline time: {t_total:.3f}s")

        stats["embed_times"].append(t_embed)
        stats["retrieval_times"].append(t_retrieve)
        stats["context_times"].append(t_ctx)
        stats["answer_times"].append(t_ans)
        stats["total_times"].append(t_total)
        stats["chunks_used"].append(len(ctx_obj["used_chunks"]))

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


if __name__ == "__main__":
    run_stress_test()
