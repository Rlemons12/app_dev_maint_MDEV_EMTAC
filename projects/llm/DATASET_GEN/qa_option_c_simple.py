#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Option C – Multi-model Q&A generator (GPU-backed, OFFLINE)

Pipeline:
    Document → structure.json → cleaned chunks
             → FLAN-T5 (GPU service) for question generation
             → FLAN + TinyLlama + Qwen + Gemma + OpenELM (GPU service)
             → JSONL output (one record per question)

NO torch
NO transformers
NO local model loading
Pure orchestration + HTTP inference
"""

import os
import sys
import json
import argparse
import logging
import time
from pathlib import Path
from typing import List, Dict, Optional

# -------------------------------------------------------------------
# Stage 1 + Stage 2
# -------------------------------------------------------------------
from structure_extractor import DocumentStructureExtractor
from structure_chunk_loader import StructureChunkLoader

# -------------------------------------------------------------------
# GPU Adapter
# -------------------------------------------------------------------
from gpu_adapter import GPUAdapter, GPUServiceError

# -------------------------------------------------------------------
# LOGGING
# -------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger("optionC_gpu")

# -------------------------------------------------------------------
# GPU MODEL MAP (EXPLICIT — NO MAGIC STRINGS)
# -------------------------------------------------------------------
GPU_MODELS = {
    "flan": "flan",
    "tiny": "tinylama",     # NOTE: single 'l' per service log
    "qwen": "mistral",      # service currently exposes mistral instead
    "gemma": "mistral",     # optional: map to same backend if desired
    "openelm": "mistral",   # optional: same here
}

# ===================================================================
# FLAN (Question + Answer)
# ===================================================================

class FLAN_QA_Model:
    """
    GPU-backed FLAN-T5 wrapper.
    """

    def __init__(self, adapter: GPUAdapter):
        self.adapter = adapter
        self.model = GPU_MODELS["flan"]

    # ---------------------------------------------------------------
    # QUESTION GENERATION
    # ---------------------------------------------------------------
    def generate_questions(
        self,
        context: str,
        n: int = 3,
        max_len: int = 48,
    ) -> List[str]:

        prompt = (
            "You are generating multiple factual questions based ONLY on the context.\n"
            f"Write {n} different short questions.\n"
            "Each question must be on its own line.\n"
            "Ask about facts explicitly stated in the context.\n\n"
            f"CONTEXT:\n{context}\n\n"
            "QUESTIONS:\n"
        )

        text = self.adapter.generate(
            prompt=prompt,
            model=self.model,
            max_tokens=max_len * n,
            temperature=0.4,
            top_p=0.9,
            num_beams=5,
            do_sample=False,
            timeout=300,
        )

        lines = [l.strip() for l in text.splitlines() if l.strip()]
        questions: List[str] = []

        for q in lines:
            q = q.replace("Q:", "").replace("Question:", "").strip()
            if not q.endswith("?"):
                q += "?"
            if q not in questions:
                questions.append(q)
            if len(questions) == n:
                break

        return questions

    # ---------------------------------------------------------------
    # ANSWER GENERATION
    # ---------------------------------------------------------------
    def generate_answer(
        self,
        context: str,
        question: str,
        max_len: int = 64,
    ) -> str:

        prompt = (
            "You are answering a factual knowledge-check question.\n"
            "Use ONLY information explicitly stated in the context.\n"
            "Keep the answer short and precise.\n\n"
            f"CONTEXT:\n{context}\n\n"
            f"QUESTION:\n{question}\n\n"
            "ANSWER:"
        )

        return self.adapter.generate(
            prompt=prompt,
            model=self.model,
            max_tokens=max_len,
            temperature=0.3,
            top_p=0.9,
            num_beams=4,
            do_sample=False,
            timeout=300,
        ).strip()

# ===================================================================
# GENERIC GPU ANSWER GENERATOR (Tiny / Qwen / Gemma / OpenELM)
# ===================================================================

class GPUAnswerGenerator:
    """
    Unified GPU-backed causal LLM generator.
    """

    def __init__(self, adapter: GPUAdapter, model_key: str):
        self.adapter = adapter
        self.model = GPU_MODELS[model_key]
        self.name = model_key.upper()

    def generate_answer(
        self,
        context: str,
        question: str,
        *,
        max_tokens: int = 80,
        temperature: float = 0.7,
        top_p: float = 0.9,
        do_sample: bool = True,
        repetition_penalty: Optional[float] = None,
    ) -> str:

        prompt = (
            "CONTEXT:\n"
            f"{context}\n\n"
            f"QUESTION: {question}\n"
            "ANSWER:"
        )

        return self.adapter.generate(
            prompt=prompt,
            model=self.model,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            do_sample=do_sample,
            repetition_penalty=repetition_penalty,
            timeout=300,
        ).strip()

# ===================================================================
# STRUCTURE JSON HANDLING
# ===================================================================

def get_structure_json(doc_path: Path, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    structure_path = out_dir / f"{doc_path.stem}_structure.json"

    if structure_path.exists():
        log.info(f"[STRUCTURE] Using existing file: {structure_path}")
        return structure_path

    log.info(f"[STRUCTURE] Extracting → {structure_path}")
    extractor = DocumentStructureExtractor(str(doc_path))
    structure = extractor.extract()
    extractor.save(structure, structure_path)
    return structure_path

# ===================================================================
# MAIN PIPELINE
# ===================================================================

def run_option_c(
    structure_json: Path,
    out_dir: Path,
    max_chunks: Optional[int] = None,
    min_context_len: int = 40,
) -> Path:

    log.info("[STAGE 2] Loading cleaned chunks")

    loader = StructureChunkLoader(
        structure_path=str(structure_json),
        min_length=min_context_len,
        dedupe=True,
        merge_headings=True,
    )

    clean_chunks = loader.load_clean_chunks()
    log.info(f"[CHUNKS] Loaded {len(clean_chunks)} chunks")

    if max_chunks:
        clean_chunks = clean_chunks[:max_chunks]
        log.info(f"[CHUNKS] Truncated to {max_chunks}")

    # ----------------------------------------------------
    # GPU ADAPTER + MODELS
    # ----------------------------------------------------
    adapter = GPUAdapter(timeout=300)
    adapter.health()

    flan    = FLAN_QA_Model(adapter)
    tiny    = GPUAnswerGenerator(adapter, "tiny")
    qwen    = GPUAnswerGenerator(adapter, "qwen")
    gemma   = GPUAnswerGenerator(adapter, "gemma")
    openelm = GPUAnswerGenerator(adapter, "openelm")

    # ----------------------------------------------------
    # OUTPUT
    # ----------------------------------------------------
    out_dir.mkdir(parents=True, exist_ok=True)
    base = structure_json.stem.replace("_structure", "")
    out_path = out_dir / f"{base}_multi_model_optionC.jsonl"

    log.info(f"[OUTPUT] → {out_path}")

    written = 0

    with open(out_path, "w", encoding="utf-8") as f:
        for idx, chunk in enumerate(clean_chunks, start=1):

            context = (chunk.get("pipeline_context") or chunk.get("text") or "").strip()
            if len(context) < min_context_len:
                continue

            log.info("────────────────────────────────────")
            log.info(f"[CHUNK {idx}]")

            try:
                questions = flan.generate_questions(context, n=3)
            except GPUServiceError as e:
                log.error(f"[FLAN-Q] {e}")
                continue

            for q_idx, q in enumerate(questions, start=1):
                log.info(f"[Q{q_idx}] {q}")

                def run(label, fn):
                    try:
                        t0 = time.time()
                        ans = fn()
                        log.info(f"[A-{label}] {ans}")
                        log.info(f"[TIMING] {label} {time.time() - t0:.2f}s")
                        return ans
                    except Exception as e:
                        log.error(f"[{label}] {e}")
                        return None

                record = {
                    "chunk_id": chunk.get("chunk_id"),
                    "page": chunk.get("page"),
                    "section": chunk.get("section"),
                    "subsection": chunk.get("subsection"),
                    "question_index": q_idx,
                    "question": q,
                    "answer_flan": run("FLAN", lambda: flan.generate_answer(context, q)),
                    "answer_tiny": run("TINY", lambda: tiny.generate_answer(context, q)),
                    "answer_qwen": run("QWEN", lambda: qwen.generate_answer(context, q)),
                    "answer_gemma": run(
                        "GEMMA",
                        lambda: gemma.generate_answer(
                            context, q,
                            temperature=0.1,
                            do_sample=False,
                        ),
                    ),
                    "answer_openelm": run(
                        "OPENELM",
                        lambda: openelm.generate_answer(
                            context, q,
                            do_sample=False,
                            repetition_penalty=1.12,
                        ),
                    ),
                    "context": context,
                }

                f.write(json.dumps(record, ensure_ascii=False) + "\n")
                written += 1

    log.info("────────────────────────────────────")
    log.info(f"[DONE] Wrote {written} Q&A pairs → {out_path}")
    return out_path

# ===================================================================
# CLI
# ===================================================================

def parse_args():
    p = argparse.ArgumentParser(description="Option C (GPU-backed)")
    p.add_argument("input", help="Document path or *_structure.json")
    p.add_argument("--out-dir", default="optionC_outputs")
    p.add_argument("--structure-dir", default="structure_maps")
    p.add_argument("--max-chunks", type=int)
    p.add_argument("--min-context-len", type=int, default=40)
    return p.parse_args()

def main():
    args = parse_args()
    inp = Path(args.input)

    if not inp.exists():
        log.error(f"Input not found: {inp}")
        sys.exit(1)

    if inp.suffix.lower() == ".json" and inp.name.endswith("_structure.json"):
        structure_json = inp
    else:
        structure_json = get_structure_json(inp, Path(args.structure_dir))

    run_option_c(
        structure_json,
        Path(args.out_dir),
        max_chunks=args.max_chunks,
        min_context_len=args.min_context_len,
    )

if __name__ == "__main__":
    main()
