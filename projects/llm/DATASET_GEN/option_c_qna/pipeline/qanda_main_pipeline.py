#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Option C v3.0 – Q&A Dataset Pipeline (Service-Layer + Run Tracking)
-------------------------------------------------------------------

Primary usage (production path):

    python -m option_c_qna.pipeline.qanda_main_pipeline \
        "path/to/document.docx" \
        --stage full \
        --models qwen \
        --max-chunks 3 \
        --embed

Stages:
    1. structure   -> extract structure.json (filesystem only)
    2. clean       -> structure.json -> cleaned chunks  + DB insert (legacy)
    3. questions   -> generate questions only           + DB insert (legacy)
    4. answers     -> generate answers only             + DB insert + ranking (legacy)
    5. full        -> run 1–4 end-to-end (service-layer + run tracking)
    6. export      -> export best/worst Q&A to fine-tuning formats
    7. rank        -> recompute rankings from DB (maintenance utility)

Design notes:
    - Full pipeline (--stage full) is the production path.
      It:
        * Creates a PipelineRun (UTC timestamps)
        * Uses QADatabaseService for Document, Chunk, Question, Answer,
          AnswerRanking, Embedding
        * Updates PipelineRun.document_id
        * Marks run finished / failed
    - Single-stage modes are for debugging / development and use simpler
      DB logic (legacy, no run tracking).
"""

from __future__ import annotations

import os
import sys
import json
import argparse
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Any
from datetime import datetime, timezone

from option_c_qna.configuration import cfg
from option_c_qna.configuration.logging_config import get_qna_logger
from option_c_qna.configuration.pg_db_config import get_qna_session
from option_c_qna.qanda_db import get_qa_service
from option_c_qna.qanda_db.qa_db import (
    LLMModel,
    Document,
    Chunk,
    Question,
    Answer,
    AnswerRanking,
    Embedding,  # retained to keep ORM imports complete
    PipelineRun,
)
from option_c_qna.models.flan_qa import FLAN_QA_Model

from option_c_qna.document_structure_extractor.structure_extractor import (
    DocumentStructureExtractor,
)
from option_c_qna.document_structure_extractor.structure_chunk_loader import (
    StructureChunkLoader,
)

from gpu_adapter import GPUAdapter

_GPU = GPUAdapter()

# -------------------------------------------------------------------
# LOGGER
# -------------------------------------------------------------------
log = get_qna_logger("qanda_pipeline")

# -------------------------------------------------------------------
# OPTIONAL – Semantic Similarity / Embedding (MiniLM)
# -------------------------------------------------------------------
try:
    from sentence_transformers import SentenceTransformer, util  # type: ignore

    _similarity_model = SentenceTransformer("all-MiniLM-L6-v2")
    HAS_SIM_MODEL = True
    log.info("[SIM] Loaded all-MiniLM-L6-v2 for question similarity + embeddings.")
except Exception as e:  # noqa: BLE001
    _similarity_model = None
    HAS_SIM_MODEL = False
    log.warning(
        "[SIM] Could not load SentenceTransformer (all-MiniLM-L6-v2). "
        "Similarity-based dedupe + embeddings disabled. Error: %s",
        e,
    )

# -------------------------------------------------------------------
# OUTPUT DIRECTORIES
# -------------------------------------------------------------------
STRUCT_DIR = cfg.STRUCTURE_DIR
CLEAN_DIR = cfg.CLEAN_DIR
QUESTION_DIR = cfg.QUESTIONS_DIR
ANSWER_DIR = cfg.ANSWERS_DIR

# -------------------------------------------------------------------
# HYPERPARAMETERS / DEFAULTS
# -------------------------------------------------------------------
DEFAULT_NUM_QUESTIONS = 3
DEFAULT_MAX_Q_RETRIES = 3
DEFAULT_SIMILARITY_THRESHOLD = 0.70
DEFAULT_MIN_QUESTION_LEN = 12

# Hybrid sampling: per-model answer samples
NUM_DETERMINISTIC_SAMPLES = 3
NUM_STOCHASTIC_SAMPLES = 5
TOTAL_SAMPLES_PER_MODEL = NUM_DETERMINISTIC_SAMPLES + NUM_STOCHASTIC_SAMPLES

EMBED_MODEL_NAME = "all-MiniLM-L6-v2"

# -------------------------------------------------------------------
# GLOBAL MODEL CACHES (load-once-and-reuse)
# -------------------------------------------------------------------
_FLAN_QA_MODEL: Optional[FLAN_QA_Model] = None
_ANSWER_MODEL_CACHE: Optional[Dict[str, Any]] = None


# ==================================================================
# RUN HELPERS
# ==================================================================
def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _create_pipeline_run(
    session,
    document_id: Optional[int],
    run_type: str,
    options_json: Dict[str, Any],
    models_json: Dict[str, Any],
    env_json: Dict[str, Any],
) -> int:
    run = PipelineRun(
        document_id=document_id,
        run_type=run_type,
        options_json=options_json,
        models_json=models_json,
        env_json=env_json,
        started_at=_utc_now(),
    )
    session.add(run)
    session.flush()  # assigns id
    session.commit()
    return int(run.id)


def _attach_document_to_run(session, run_id: int, document_id: int) -> None:
    run = session.get(PipelineRun, run_id)
    if not run:
        raise RuntimeError(f"PipelineRun id={run_id} not found (attach_document_to_run).")
    run.document_id = document_id
    session.commit()
    log.info("[RUN] Attached document_id=%s to run_id=%s", document_id, run_id)


def _finish_pipeline_run(run_id: int, success: bool, error_message: Optional[str]) -> None:
    session = get_qna_session()
    try:
        run = session.get(PipelineRun, run_id)
        if not run:
            raise RuntimeError(f"PipelineRun id={run_id} not found (finish).")

        run.finished_at = _utc_now()
        run.success = bool(success)

        if error_message:
            run.error_message = str(error_message)[:4000]

        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


# ==================================================================
# MODELS
# ==================================================================
def get_flan_model() -> FLAN_QA_Model:
    """
    Lazily load FLAN-T5-Large once per process and reuse.
    Used for question generation.
    """
    global _FLAN_QA_MODEL
    if _FLAN_QA_MODEL is None:
        log.info("[MODEL] Loading FLAN-T5-Large (FLAN_QA_Model) once...")
        _FLAN_QA_MODEL = FLAN_QA_Model()
    return _FLAN_QA_MODEL


def _init_answer_models(
    selected_models: Optional[List[str]] = None,
    model_cache: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Load answer LLMs with caching support.

    Priority:
        1) model_cache arg (externally supplied)
        2) global _ANSWER_MODEL_CACHE
        3) DB (LLMModel.load_all_enabled) -> cache globally
    """
    global _ANSWER_MODEL_CACHE

    if model_cache is not None:
        log.info("[ANSWERS] Using externally provided preloaded model cache.")
        all_models = {name.lower(): obj for name, obj in model_cache.items()}

    elif _ANSWER_MODEL_CACHE is not None:
        log.info("[ANSWERS] Using global preloaded model cache.")
        all_models = {name.lower(): obj for name, obj in _ANSWER_MODEL_CACHE.items()}

    else:
        try:
            raw_models = LLMModel.load_all_enabled()
        except Exception as exc:
            log.error("[ANSWERS] Failed to load enabled models from qna_models.")
            log.exception(exc)
            raise

        all_models = {name.lower(): obj for name, obj in (raw_models or {}).items()}
        if not all_models:
            raise RuntimeError("No enabled LLM models available (qna_models empty).")

        _ANSWER_MODEL_CACHE = dict(all_models)
        log.info("[ANSWERS] Cached %d answer models.", len(all_models))

    if not selected_models:
        log.info("[ANSWERS] Using ALL enabled models: %s", list(all_models.keys()))
        return all_models

    selected_lower = [m.strip().lower() for m in selected_models if m.strip()]
    filtered = {name: obj for name, obj in all_models.items() if name in selected_lower}

    for req in selected_lower:
        if req not in all_models:
            log.warning("[ANSWERS] Requested model '%s' not found/enabled.", req)

    if not filtered:
        raise RuntimeError(
            "No valid answer models remain after filtering. "
            f"Requested={selected_lower} Enabled={list(all_models.keys())}"
        )

    log.info("[ANSWERS] Using models: %s", list(filtered.keys()))
    return filtered


def preload_all_answer_models() -> Dict[str, Any]:
    """
    Loads all enabled answer models from qna_models table ONE TIME.
    Populates global cache.
    """
    global _ANSWER_MODEL_CACHE
    if _ANSWER_MODEL_CACHE is not None:
        log.info("[PRELOAD] Using existing global model cache.")
        return _ANSWER_MODEL_CACHE

    enabled_models = LLMModel.load_all_enabled()
    if not enabled_models:
        raise RuntimeError("[PRELOAD] No enabled models found in qna_models table.")

    loaded: Dict[str, Any] = {}
    log.info("[PRELOAD] Found %d enabled models: %s", len(enabled_models), list(enabled_models.keys()))

    for name, model_obj in enabled_models.items():
        try:
            log.info("[PRELOAD] Initializing model '%s'...", name)
            _ = model_obj  # ensure constructor side-effects happened
            loaded[name.lower()] = model_obj
            log.info("[PRELOAD] Model '%s' loaded successfully.", name)
        except Exception as exc:
            log.error("[PRELOAD] FAILED to load model '%s'. Skipping.", name)
            log.exception(exc)

    if not loaded:
        raise RuntimeError("[PRELOAD] All models failed to load.")

    _ANSWER_MODEL_CACHE = loaded
    log.info("[PRELOAD] Successfully preloaded %d models.", len(loaded))
    return loaded


# ==================================================================
# EMBEDDING HELPER
# ==================================================================
def compute_embedding(text: str) -> Optional[List[float]]:
    if not HAS_SIM_MODEL or not text:
        return None
    try:
        vec = _similarity_model.encode(text)
        return vec.tolist()
    except Exception as e:  # noqa: BLE001
        log.warning("[EMBED] Failed to compute embedding: %s", e)
        return None


# ==================================================================
# QUESTION TEMPLATE + QUALITY + SIMILARITY
# ==================================================================
QUESTION_TEMPLATES = [
    "What is {focus}?",
    "What does {focus} refer to?",
    "Where is {focus} located?",
    "Which {focus} is mentioned?",
    "When does {focus} occur?",
]


def apply_question_templates(raw_questions: List[str]) -> List[str]:
    import re

    normalized: List[str] = []
    for idx, q in enumerate(raw_questions or []):
        q = (q or "").strip()
        if not q:
            continue

        if q[0].isupper() and q.endswith("?"):
            normalized.append(q)
            continue

        match = re.search(r"(what|where|which|when|who)\s+(.*)", q, flags=re.IGNORECASE)
        focus = match.group(2).strip().rstrip("?") if match else q.rstrip("?")

        template = QUESTION_TEMPLATES[idx % len(QUESTION_TEMPLATES)]
        normalized.append(template.format(focus=focus))

    return normalized


def question_quality_pass(question: str, context: str) -> bool:
    if not question:
        return False

    q = question.strip()
    if len(q) < DEFAULT_MIN_QUESTION_LEN:
        return False

    lower_q = q.lower()
    if not lower_q.startswith(("what", "where", "which", "when", "who")):
        return False

    if not q.endswith("?"):
        return False

    q_words = set(lower_q.rstrip("?").split())
    c_words = set((context or "").lower().split())
    overlap = len(q_words & c_words)

    return overlap >= 2


def dedupe_questions(
    questions: List[str],
    similarity_threshold: float = DEFAULT_SIMILARITY_THRESHOLD,
) -> List[str]:
    cleaned: List[str] = []
    if not questions:
        return cleaned

    if not HAS_SIM_MODEL:
        seen = set()
        for q in questions:
            key = q.strip().lower()
            if key and key not in seen:
                cleaned.append(q)
                seen.add(key)
        return cleaned

    embeddings = []
    for q in questions:
        q = q.strip()
        if not q:
            continue
        emb = _similarity_model.encode(q)
        if embeddings:
            sims = util.cos_sim(emb, embeddings)[0]
            if float(max(sims)) > similarity_threshold:
                continue
        embeddings.append(emb)
        cleaned.append(q)

    return cleaned


def generate_questions_multi_pass(
    flan_model: FLAN_QA_Model,
    context: str,
    n: int = DEFAULT_NUM_QUESTIONS,
    max_retries: int = DEFAULT_MAX_Q_RETRIES,
    similarity_threshold: float = DEFAULT_SIMILARITY_THRESHOLD,
) -> List[str]:
    last_raw: List[str] = []

    for attempt in range(1, max_retries + 1):
        raw = flan_model.generate_questions(context, n=n) or []
        last_raw = raw

        templated = apply_question_templates(raw)
        passed = [q for q in templated if question_quality_pass(q, context)]
        deduped = dedupe_questions(passed, similarity_threshold=similarity_threshold)

        if deduped:
            log.debug(
                "[QUESTION] PASS success attempt=%d kept=%d/%d ctx_len=%d",
                attempt,
                len(deduped),
                len(raw),
                len(context or ""),
            )
            return deduped

        log.debug(
            "[QUESTION] PASS fail attempt=%d raw=%d passed=%d deduped=%d",
            attempt,
            len(raw),
            len(passed),
            len(deduped),
        )

    if last_raw:
        log.warning(
            "[QUESTION] Retries exhausted; returning last raw FLAN questions (templated) without filtering."
        )
        return apply_question_templates(last_raw)

    log.warning("[QUESTION] Multi-pass generation produced no questions.")
    return []


# ==================================================================
# ANSWER RANKING
# ==================================================================
def rank_answers(answers: Dict[str, str], context: str) -> List[Tuple[str, str, float]]:
    ranked: List[Tuple[str, str, float]] = []
    c_words = set((context or "").lower().split())

    for model_name, ans in (answers or {}).items():
        if not ans:
            ranked.append((model_name, ans, float("-inf")))
            continue

        a_words = set(ans.lower().split())
        overlap = len(a_words & c_words)
        score = float(overlap)

        if overlap == 0:
            score -= 5.0

        length = len(a_words)
        if 5 <= length <= 40:
            score += 1.0

        ranked.append((model_name, ans, score))

    ranked.sort(key=lambda x: x[2], reverse=True)
    return ranked


# ======================================================================
# STAGE 1 — STRUCTURE EXTRACTION
# ======================================================================
def stage_structure_only(doc_path: Path) -> Path:
    STRUCT_DIR.mkdir(parents=True, exist_ok=True)
    out = STRUCT_DIR / f"{doc_path.stem}_structure.json"

    if out.exists():
        log.info("[STRUCTURE] Using existing: %s", out)
        return out

    log.info("[STRUCTURE] Extracting -> %s", out)
    extractor = DocumentStructureExtractor(str(doc_path))
    structure = extractor.extract()
    extractor.save(structure, out)
    log.info("[STRUCTURE] Wrote structure JSON: %s", out)
    return out


# ======================================================================
# STAGE 2 — CLEAN CHUNKS + DB INSERT (legacy)
# ======================================================================
def stage_clean_chunks(structure_json: Path, min_len: int = 40) -> Path:
    CLEAN_DIR.mkdir(parents=True, exist_ok=True)
    out = CLEAN_DIR / f"{structure_json.stem.replace('_structure', '')}_clean.jsonl"

    log.info("[CLEAN] Loading structure -> %s", structure_json)

    loader = StructureChunkLoader(
        structure_path=str(structure_json),
        min_length=min_len,
        dedupe=True,
        merge_headings=True,
    )
    chunks = loader.load_clean_chunks()
    log.info("[CLEAN] Loaded %d cleaned chunks", len(chunks))

    session = get_qna_session()
    try:
        document = Document(
            file_name=structure_json.stem.replace("_structure", ""),
            file_path=str(structure_json),
        )
        session.add(document)
        session.flush()

        with open(out, "w", encoding="utf-8") as f:
            for c in chunks:
                f.write(json.dumps(c, ensure_ascii=False) + "\n")
                session.add(
                    Chunk(
                        document_id=document.id,
                        chunk_id=c["chunk_id"],
                        context=c["pipeline_context"],
                        page=c.get("page"),
                        section=c.get("section"),
                        subsection=c.get("subsection"),
                    )
                )

        session.commit()
        log.info("[CLEAN] Inserted document_id=%s with %d chunks", document.id, len(chunks))
        log.info("[CLEAN] Wrote cleaned JSONL -> %s", out)
        return out

    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


# ======================================================================
# STAGE 2 — CLEAN + RUN TRACKING (service-layer)
# ======================================================================
def stage_clean_chunks_tracked(
    doc_path: Path,
    structure_json: Path,
    min_len: int,
    run_id: int,
    embed: bool,
) -> Tuple[Path, int, List[Dict[str, Any]]]:
    svc = get_qa_service()

    CLEAN_DIR.mkdir(parents=True, exist_ok=True)
    out = CLEAN_DIR / f"{structure_json.stem.replace('_structure', '')}_clean.jsonl"

    log.info("[CLEAN/TRACK] Loading structure -> %s", structure_json)

    loader = StructureChunkLoader(
        structure_path=str(structure_json),
        min_length=min_len,
        dedupe=True,
        merge_headings=True,
    )
    chunks = loader.load_clean_chunks()
    log.info("[CLEAN/TRACK] Loaded %d cleaned chunks", len(chunks))

    document = svc.add_document(
        run_id=run_id,
        file_name=doc_path.stem,
        file_path=str(doc_path),
    )
    document_id = int(document.id)

    # Write JSONL + insert chunks
    with open(out, "w", encoding="utf-8") as f:
        for c in chunks:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")

            chunk_obj = svc.add_chunk(
                run_id=run_id,
                document_id=document_id,
                chunk_id=c["chunk_id"],
                context=c["pipeline_context"],
                page=c.get("page"),
                section=c.get("section"),
                subsection=c.get("subsection"),
            )

            if embed:
                vec = compute_embedding(chunk_obj.context)
                if vec is not None:
                    svc.add_embedding(
                        run_id=run_id,
                        parent_type="chunk",
                        parent_id=chunk_obj.id,
                        model_name=EMBED_MODEL_NAME,
                        embedding_vector=vec,
                        metadata={
                            "source": "pipeline_clean",
                            "doc_id": document_id,
                            "chunk_id": chunk_obj.chunk_id,
                        },
                    )

    log.info("[CLEAN/TRACK] Document id=%s, chunks=%d, run_id=%s", document_id, len(chunks), run_id)
    log.info("[CLEAN/TRACK] Wrote cleaned JSONL -> %s", out)

    return out, document_id, chunks


# ======================================================================
# STAGE 3 — QUESTIONS (legacy)
# ======================================================================
def stage_generate_questions(
    clean_jsonl: Path,
    num_questions: int = DEFAULT_NUM_QUESTIONS,
    max_q_retries: int = DEFAULT_MAX_Q_RETRIES,
    similarity_threshold: float = DEFAULT_SIMILARITY_THRESHOLD,
    max_chunks: Optional[int] = None,
) -> Path:
    QUESTION_DIR.mkdir(parents=True, exist_ok=True)
    out = QUESTION_DIR / f"{clean_jsonl.stem}_questions.jsonl"

    log.info("[QUESTION] Loading FLAN model for question generation...")
    flan = get_flan_model()

    with open(clean_jsonl, "r", encoding="utf-8") as fin:
        clean_chunks = [json.loads(line) for line in fin]

    if max_chunks is not None:
        clean_chunks = clean_chunks[:max_chunks]

    if not clean_chunks:
        log.warning("[QUESTION] No chunks found in clean JSONL; nothing to do.")
        return out

    chunk_ids = {c["chunk_id"] for c in clean_chunks}

    session = get_qna_session()
    try:
        db_chunks = session.query(Chunk).filter(Chunk.chunk_id.in_(list(chunk_ids))).all()
        chunk_map = {c.chunk_id: c.id for c in db_chunks}
    finally:
        session.close()

    written_groups = 0
    total_questions = 0

    with open(out, "w", encoding="utf-8") as fout:
        for chunk in clean_chunks:
            context = chunk["pipeline_context"]
            pipeline_chunk_id = chunk["chunk_id"]

            flan_questions = generate_questions_multi_pass(
                flan_model=flan,
                context=context,
                n=num_questions,
                max_retries=max_q_retries,
                similarity_threshold=similarity_threshold,
            ) or []

            record = {
                "chunk_id": pipeline_chunk_id,
                "page": chunk.get("page"),
                "section": chunk.get("section"),
                "subsection": chunk.get("subsection"),
                "context": context,
                "questions": flan_questions,
            }
            fout.write(json.dumps(record, ensure_ascii=False) + "\n")

            db_chunk_id = chunk_map.get(pipeline_chunk_id)
            if db_chunk_id is None:
                log.warning("[QUESTION] No DB chunk found for chunk_id=%s; skipping DB insert.", pipeline_chunk_id)
                continue

            session = get_qna_session()
            try:
                for idx, question_text in enumerate(flan_questions, start=1):
                    session.add(Question(chunk_id=db_chunk_id, question=question_text, question_index=idx))
                    total_questions += 1
                session.commit()
            except Exception:
                session.rollback()
                raise
            finally:
                session.close()

            written_groups += 1

    log.info("[QUESTION] Wrote %d chunk question groups to %s", written_groups, out)
    log.info("[QUESTION] Inserted %d questions into DB", total_questions)
    return out


# ======================================================================
# STAGE 3 — QUESTIONS (tracked, service-layer)
# ======================================================================
def stage_generate_questions_tracked(
    clean_chunks: List[Dict[str, Any]],
    document_id: int,
    run_id: int,
    num_questions: int,
    max_q_retries: int,
    similarity_threshold: float,
    max_chunks: Optional[int],
    embed: bool,
) -> Tuple[Path, List[Dict[str, Any]]]:
    QUESTION_DIR.mkdir(parents=True, exist_ok=True)
    out = QUESTION_DIR / "questions.jsonl"

    log.info("[QUESTION/TRACK] Loading FLAN model for question generation...")
    flan = get_flan_model()
    svc = get_qa_service()

    if max_chunks is not None:
        clean_chunks = clean_chunks[:max_chunks]

    if not clean_chunks:
        log.warning("[QUESTION/TRACK] No cleaned chunks received; nothing to do.")
        return out, []

    session = get_qna_session()
    try:
        doc = session.get(Document, document_id)
        if not doc:
            log.error("[QUESTION/TRACK] Document id=%s not found", document_id)
            return out, []
        chunk_by_chunk_id = {c.chunk_id: c for c in doc.chunks}
    finally:
        session.close()

    records: List[Dict[str, Any]] = []
    for chunk in clean_chunks:
        context = chunk["pipeline_context"]
        pipeline_chunk_id = chunk["chunk_id"]

        flan_questions = generate_questions_multi_pass(
            flan_model=flan,
            context=context,
            n=num_questions,
            max_retries=max_q_retries,
            similarity_threshold=similarity_threshold,
        ) or []

        records.append(
            {
                "chunk_id": pipeline_chunk_id,
                "page": chunk.get("page"),
                "section": chunk.get("section"),
                "subsection": chunk.get("subsection"),
                "context": context,
                "questions": flan_questions,
            }
        )

    with open(out, "w", encoding="utf-8") as fout:
        for rec in records:
            fout.write(json.dumps(rec, ensure_ascii=False) + "\n")

    total_questions = 0
    for rec in records:
        pipeline_chunk_id = rec["chunk_id"]
        q_texts = rec["questions"] or []

        chunk_obj = chunk_by_chunk_id.get(pipeline_chunk_id)
        if not chunk_obj:
            log.warning("[QUESTION/TRACK] No DB chunk found for chunk_id=%s; skipping.", pipeline_chunk_id)
            continue

        for idx, question_text in enumerate(q_texts, start=1):
            q_obj = svc.add_question(
                run_id=run_id,
                chunk_id=chunk_obj.id,
                question_text=question_text,
                question_index=idx,
            )
            total_questions += 1

            if embed:
                vec = compute_embedding(q_obj.question)
                if vec is not None:
                    svc.add_embedding(
                        run_id=run_id,
                        parent_type="question",
                        parent_id=q_obj.id,
                        model_name=EMBED_MODEL_NAME,
                        embedding_vector=vec,
                        metadata={
                            "source": "pipeline_questions",
                            "doc_id": document_id,
                            "chunk_id": chunk_obj.chunk_id,
                            "question_index": idx,
                        },
                    )

    log.info("[QUESTION/TRACK] Inserted %d questions into DB (doc_id=%s run_id=%s)", total_questions, document_id, run_id)
    log.info("[QUESTION/TRACK] Wrote questions JSONL -> %s", out)

    return out, records


# ======================================================================
# STAGE 4 — ANSWERS (legacy)
# ======================================================================
def stage_generate_answers(
    question_jsonl: Path,
    models: Optional[List[str]] = None,
    max_chunks: Optional[int] = None,
) -> Path:
    ANSWER_DIR.mkdir(parents=True, exist_ok=True)
    out = ANSWER_DIR / f"{question_jsonl.stem}_answers.jsonl"

    log.info("[ANSWERS] Loading answer models...")
    model_objects = _init_answer_models(models)

    with open(question_jsonl, "r", encoding="utf-8") as fin:
        question_items = [json.loads(line) for line in fin]

    if max_chunks is not None:
        question_items = question_items[:max_chunks]

    if not question_items:
        log.warning("[ANSWERS] No question records found; nothing to do.")
        return out

    pipeline_chunk_ids = {item["chunk_id"] for item in question_items}

    session = get_qna_session()
    try:
        db_chunks = session.query(Chunk).filter(Chunk.chunk_id.in_(list(pipeline_chunk_ids))).all()
        chunk_map = {c.chunk_id: c.id for c in db_chunks}

        db_questions = session.query(Question).filter(Question.chunk_id.in_(list(chunk_map.values()))).all()
        question_map = {(q.chunk_id, q.question_index): q.id for q in db_questions}
    finally:
        session.close()

    written_records = 0
    total_answers = 0
    total_rankings = 0

    with open(out, "w", encoding="utf-8") as fout:
        for item in question_items:
            context = item["context"]
            pipeline_chunk_id = item["chunk_id"]
            questions = item.get("questions") or []

            db_chunk_id = chunk_map.get(pipeline_chunk_id)
            if db_chunk_id is None:
                log.warning("[ANSWERS] No DB chunk for chunk_id=%s; skipping.", pipeline_chunk_id)
                continue

            for idx, q_text in enumerate(questions, start=1):
                per_model_best_answer: Dict[str, str] = {}
                per_model_samples: Dict[str, List[Dict[str, Any]]] = {}

                for model_name, model_obj in model_objects.items():
                    sample_answers: Dict[str, str] = {}

                    for i in range(NUM_DETERMINISTIC_SAMPLES):
                        sample_answers[f"{model_name}#det{i+1}"] = model_obj.generate_answer(context, q_text)

                    for i in range(NUM_STOCHASTIC_SAMPLES):
                        sample_answers[f"{model_name}#stoch{i+1}"] = model_obj.generate_answer(context, q_text)

                    ranked_samples = rank_answers(sample_answers, context)
                    if not ranked_samples:
                        continue

                    best_key, best_ans, best_score = ranked_samples[0]
                    per_model_best_answer[model_name] = best_ans
                    per_model_samples[model_name] = [
                        {"sample_id": key, "answer": ans_text, "score": float(score)}
                        for (key, ans_text, score) in ranked_samples
                    ]

                if not per_model_best_answer:
                    continue

                cross_model_ranked = rank_answers(per_model_best_answer, context)

                best_model, best_answer, best_score = cross_model_ranked[0]
                worst_model, worst_answer, worst_score = (
                    cross_model_ranked[-1] if len(cross_model_ranked) > 1 else cross_model_ranked[0]
                )
                answer_scores = {m: float(s) for (m, _, s) in cross_model_ranked}

                rec = {
                    "chunk_id": pipeline_chunk_id,
                    "question_index": idx,
                    "question": q_text,
                    "context": context,
                    "best_model": best_model,
                    "best_answer": best_answer,
                    "worst_model": worst_model,
                    "worst_answer": worst_answer,
                    "answer_scores": answer_scores,
                    **{f"answer_{m}": a for m, a in per_model_best_answer.items()},
                    "per_model_samples": per_model_samples,
                }
                fout.write(json.dumps(rec, ensure_ascii=False) + "\n")

                db_question_id = question_map.get((db_chunk_id, idx))
                if db_question_id is None:
                    continue

                session = get_qna_session()
                try:
                    for rank_idx, (model_name, best_ans_for_model, score) in enumerate(cross_model_ranked):
                        session.add(
                            Answer(
                                question_id=db_question_id,
                                model_name=model_name,
                                model_type="causal_lm",
                                model_path=None,
                                answer_text=best_ans_for_model,
                                score=float(score),
                                is_best=(rank_idx == 0),
                                is_worst=(rank_idx == len(cross_model_ranked) - 1 and len(cross_model_ranked) > 1),
                            )
                        )
                        total_answers += 1

                    session.add(
                        AnswerRanking(
                            question_id=db_question_id,
                            best_model=best_model,
                            best_answer=best_answer,
                            worst_model=worst_model,
                            worst_answer=worst_answer,
                            answer_scores=answer_scores,
                        )
                    )
                    total_rankings += 1
                    session.commit()
                except Exception:
                    session.rollback()
                    raise
                finally:
                    session.close()

                written_records += 1

    log.info("[ANSWERS] Wrote %d Q/A records -> %s", written_records, out)
    log.info("[ANSWERS] Inserted %d answers into DB", total_answers)
    log.info("[ANSWERS] Inserted %d ranking rows into DB", total_rankings)
    return out


# ======================================================================
# STAGE 4 — ANSWERS (tracked, service-layer)
# ======================================================================
def stage_generate_answers_tracked(
    question_records: List[Dict[str, Any]],
    document_id: int,
    run_id: int,
    models: List[str],
    max_chunks: Optional[int],
    embed: bool,
    model_cache: Optional[Dict[str, Any]] = None,
) -> Path:
    ANSWER_DIR.mkdir(parents=True, exist_ok=True)
    out = ANSWER_DIR / "answers.jsonl"

    log.info("[ANSWERS/TRACK] Initializing model-batched answer generation...")
    svc = get_qa_service()

    # ------------------------------------------------------------------
    # Load models ONCE (this is the critical change)
    # ------------------------------------------------------------------
    model_objects = {m: None for m in models}


    if max_chunks is not None:
        question_records = question_records[:max_chunks]

    if not question_records:
        log.warning("[ANSWERS/TRACK] No question_records provided; nothing to do.")
        return out

    # ------------------------------------------------------------------
    # Resolve DB mappings once
    # ------------------------------------------------------------------
    session = get_qna_session()
    try:
        doc = session.get(Document, document_id)
        if not doc:
            raise RuntimeError(f"Document id={document_id} not found")

        chunk_by_chunk_id = {c.chunk_id: c for c in doc.chunks}

        db_questions = (
            session.query(Question)
            .join(Chunk, Question.chunk_id == Chunk.id)
            .filter(Chunk.document_id == document_id)
            .all()
        )
        question_map = {(q.chunk_id, q.question_index): q for q in db_questions}
    finally:
        session.close()

    # ------------------------------------------------------------------
    # Pre-expand all (chunk, question) work items
    # ------------------------------------------------------------------
    work_items: List[Dict[str, Any]] = []

    for item in question_records:
        chunk_id = item["chunk_id"]
        context = item["context"]
        questions = item.get("questions") or []

        chunk_obj = chunk_by_chunk_id.get(chunk_id)
        if not chunk_obj:
            continue

        for idx, q_text in enumerate(questions, start=1):
            q_row = question_map.get((chunk_obj.id, idx))
            if not q_row:
                continue

            work_items.append(
                {
                    "chunk": chunk_obj,
                    "question_row": q_row,
                    "question_text": q_text,
                    "context": context,
                }
            )

    if not work_items:
        log.warning("[ANSWERS/TRACK] No valid work items resolved.")
        return out

    # ------------------------------------------------------------------
    # MODEL-BATCHED GENERATION
    # ------------------------------------------------------------------
    all_results: Dict[int, Dict[str, str]] = {}  # question_id → model → answer

    for model_name, model_obj in model_objects.items():
        log.info("[ANSWERS/TRACK] Generating answers with model '%s' (%d questions)",
                 model_name, len(work_items))

        for wi in work_items:
            qid = wi["question_row"].id
            context = wi["context"]
            question = wi["question_text"]

            try:
                answer = _GPU.generate(
                    model=model_name,
                    prompt=f"Context:\n{context}\n\nQuestion:\n{question}\n\nAnswer:",
                    max_tokens=128,
                    temperature=0.7,
                    top_p=0.95,
                )
            except Exception as e:
                log.warning(
                    "[ANSWERS/TRACK] Model '%s' failed on question_id=%s: %s",
                    model_name,
                    qid,
                    e,
                )
                continue

            all_results.setdefault(qid, {})[model_name] = answer

    # ------------------------------------------------------------------
    # RANKING + DB INSERT
    # ------------------------------------------------------------------
    written = 0
    total_answers = 0
    total_rankings = 0

    with open(out, "w", encoding="utf-8") as fout:
        for wi in work_items:
            q_row = wi["question_row"]
            context = wi["context"]
            answers_by_model = all_results.get(q_row.id)

            if not answers_by_model:
                continue

            ranked = rank_answers(answers_by_model, context)
            if not ranked:
                continue

            best_model, best_answer, _ = ranked[0]
            worst_model, worst_answer, _ = ranked[-1]
            answer_scores = {m: float(s) for (m, _, s) in ranked}

            record = {
                "chunk_id": wi["chunk"].chunk_id,
                "question_index": q_row.question_index,
                "question": q_row.question,
                "context": context,
                "best_model": best_model,
                "best_answer": best_answer,
                "worst_model": worst_model,
                "worst_answer": worst_answer,
                "answer_scores": answer_scores,
                **{f"answer_{m}": a for m, a in answers_by_model.items()},
            }
            fout.write(json.dumps(record, ensure_ascii=False) + "\n")

            for rank_idx, (model_name, ans_text, score) in enumerate(ranked):
                a_obj = svc.add_answer(
                    run_id=run_id,
                    question_id=q_row.id,
                    model_name=model_name,
                    answer_text=ans_text,
                    model_type="causal_lm",
                    model_path=None,
                    score=float(score),
                    is_best=(rank_idx == 0),
                    is_worst=(rank_idx == len(ranked) - 1),
                )
                total_answers += 1

                if embed:
                    vec = compute_embedding(ans_text)
                    if vec is not None:
                        svc.add_embedding(
                            run_id=run_id,
                            parent_type="answer",
                            parent_id=a_obj.id,
                            model_name=EMBED_MODEL_NAME,
                            embedding_vector=vec,
                            metadata={
                                "source": "pipeline_answers",
                                "doc_id": document_id,
                                "model_name": model_name,
                            },
                        )

            svc.add_answer_ranking(
                run_id=run_id,
                question_id=q_row.id,
                best_model=best_model,
                best_answer=best_answer,
                worst_model=worst_model,
                worst_answer=worst_answer,
                answer_scores=answer_scores,
            )
            total_rankings += 1
            written += 1

    log.info("[ANSWERS/TRACK] Wrote %d answer records -> %s", written, out)
    log.info("[ANSWERS/TRACK] Inserted %d answers into DB", total_answers)
    log.info("[ANSWERS/TRACK] Inserted %d ranking rows into DB", total_rankings)

    return out



# ======================================================================
# STAGE 6 — EXPORT DATASET
# ======================================================================
def stage_export_dataset(answers_jsonl: Path, export_format: str = "alpaca") -> Path:
    if not answers_jsonl.exists():
        raise FileNotFoundError(f"Answers JSONL not found: {answers_jsonl}")

    export_path = answers_jsonl.with_suffix(f".{export_format}.jsonl")
    log.info("[EXPORT] Exporting %s -> %s (format=%s)", answers_jsonl, export_path, export_format)

    qna_items: List[Dict[str, str]] = []
    with open(answers_jsonl, "r", encoding="utf-8") as fin:
        for line in fin:
            rec = json.loads(line)
            question = rec.get("question") or ""
            context = rec.get("context") or ""
            best_answer = rec.get("best_answer") or ""
            worst_answer = rec.get("worst_answer") or ""
            if question and best_answer:
                qna_items.append(
                    {"question": question, "context": context, "best_answer": best_answer, "worst_answer": worst_answer}
                )

    with open(export_path, "w", encoding="utf-8") as fout:
        if export_format == "alpaca":
            for item in qna_items:
                fout.write(
                    json.dumps(
                        {"instruction": item["question"], "input": item["context"], "output": item["best_answer"]},
                        ensure_ascii=False,
                    )
                    + "\n"
                )

        elif export_format == "chatml":
            for item in qna_items:
                fout.write(
                    json.dumps(
                        {
                            "messages": [
                                {
                                    "role": "system",
                                    "content": "You are a manufacturing training assistant. Answer strictly using the provided context.",
                                },
                                {"role": "user", "content": item["context"]},
                                {"role": "assistant", "content": item["best_answer"]},
                            ]
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )

        elif export_format == "orpo":
            for item in qna_items:
                fout.write(
                    json.dumps(
                        {"prompt": item["question"], "chosen": item["best_answer"], "rejected": item["worst_answer"]},
                        ensure_ascii=False,
                    )
                    + "\n"
                )
        else:
            raise ValueError(f"Unknown export_format: {export_format}")

    log.info("[EXPORT] Wrote %d Q&A items to %s", len(qna_items), export_path)
    return export_path


# ======================================================================
# STAGE 7 — RANK EXISTING ANSWERS (maintenance utility)
# ======================================================================
def stage_rank_answers() -> None:
    log.info("[RANK] Running answer ranking stage...")

    session = get_qna_session()
    try:
        all_questions = session.query(Question).all()
        log.info("[RANK] Found %d questions.", len(all_questions))

        for q in all_questions:
            answers = session.query(Answer).filter(Answer.question_id == q.id).all()
            if not answers:
                continue

            answer_scores: Dict[str, float] = {}
            for a in answers:
                s = float(a.score or 0.0)
                answer_scores[a.model_name] = max(answer_scores.get(a.model_name, float("-inf")), s)

            best_model = max(answer_scores, key=answer_scores.get)
            worst_model = min(answer_scores, key=answer_scores.get)

            best_answer_obj = max(
                (a for a in answers if a.model_name == best_model),
                key=lambda x: float(x.score or 0.0),
            )
            worst_answer_obj = min(
                (a for a in answers if a.model_name == worst_model),
                key=lambda x: float(x.score or 0.0),
            )

            session.add(
                AnswerRanking(
                    question_id=q.id,
                    best_model=best_model,
                    best_answer=best_answer_obj.answer_text,
                    worst_model=worst_model,
                    worst_answer=worst_answer_obj.answer_text,
                    answer_scores=answer_scores,
                )
            )

        session.commit()
        log.info("[RANK] Ranking stage complete.")
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


# ======================================================================
# FULL PIPELINE (service-layer + run tracking)
# ======================================================================
def run_full_pipeline(
    doc_path: Path,
    max_chunks: Optional[int] = None,
    min_context_len: int = 40,
    num_questions: int = DEFAULT_NUM_QUESTIONS,
    max_q_retries: int = DEFAULT_MAX_Q_RETRIES,
    similarity_threshold: float = DEFAULT_SIMILARITY_THRESHOLD,
    models: Optional[List[str]] = None,
    embed: bool = False,
    test_mode: bool = False,
) -> Path:
    session = get_qna_session()

    log.info("=== run_full_pipeline() STARTED ===")
    log.info("Document path: %s", doc_path)

    model_list = [m.strip().lower() for m in (models or []) if m.strip()]
    log.debug("Requested answer models: %s", model_list)

    if test_mode:
        log.warning("TEST MODE ENABLED: forcing 1 chunk, 1 question, no retries, no embeddings.")
        max_chunks = 1
        num_questions = 1
        max_q_retries = 0
        similarity_threshold = 0.0
        min_context_len = 1
        embed = False
        os.environ["OPTION_C_TEST_MODE"] = "1"

    log.info("Preloading all enabled answer models...")
    preloaded_answer_models = preload_all_answer_models()
    log.debug("Preloaded models: %s", list(preloaded_answer_models.keys()))

    options = {
        "max_chunks": max_chunks,
        "min_context_len": min_context_len,
        "num_questions": num_questions,
        "max_q_retries": max_q_retries,
        "similarity_threshold": similarity_threshold,
        "embed": embed,
        "test_mode": test_mode,
    }

    env = {
        "source_path": str(doc_path),
        "cwd": str(os.getcwd()),
        "timestamp_utc": _utc_now().isoformat(),
    }

    run_id = _create_pipeline_run(
        session=session,
        document_id=None,
        run_type="full",
        options_json=options,
        models_json={"answers": model_list},
        env_json=env,
    )
    log.info("Created pipeline run_id=%s", run_id)

    try:
        log.info("Stage 1 – structure")
        struct_path = stage_structure_only(doc_path)

        log.info("Stage 2 – clean + insert (tracked)")
        clean_path, document_id, chunks = stage_clean_chunks_tracked(
            doc_path=doc_path,
            structure_json=struct_path,
            min_len=min_context_len,
            run_id=run_id,
            embed=embed,
        )
        log.info("Clean complete: document_id=%s", document_id)

        _attach_document_to_run(session, run_id, document_id)

        log.info("Stage 3 – questions (tracked)")
        questions_path, question_records = stage_generate_questions_tracked(
            clean_chunks=chunks,
            document_id=document_id,
            run_id=run_id,
            num_questions=num_questions,
            max_q_retries=max_q_retries,
            similarity_threshold=similarity_threshold,
            max_chunks=max_chunks,
            embed=embed,
        )

        log.info("Stage 4 – answers (tracked, cached models)")
        answers_path = stage_generate_answers_tracked(
            question_records=question_records,
            document_id=document_id,
            run_id=run_id,
            models=model_list,
            max_chunks=max_chunks,
            embed=embed,
            model_cache=preloaded_answer_models,
        )

        log.info("Stage 5 – finish run")
        _finish_pipeline_run(run_id, success=True, error_message=None)

        log.info("=== run_full_pipeline() COMPLETE ===")
        return answers_path

    except Exception as e:
        log.error("EXCEPTION TRIGGERED IN PIPELINE")
        log.exception(e)
        _finish_pipeline_run(run_id, success=False, error_message=str(e))
        raise

    finally:
        session.close()


# ======================================================================
# CLI
# ======================================================================
def _get_default_models_from_db() -> List[str]:
    """
    Returns enabled model names from qna_models as a list.
    Tries a couple of common helper APIs; falls back to load_all_enabled().
    """
    try:
        if hasattr(LLMModel, "get_default_model_names"):
            names = LLMModel.get_default_model_names()
            return [str(n).strip().lower() for n in (names or []) if str(n).strip()]
    except Exception as e:
        log.warning("[CLI] LLMModel.get_default_model_names failed: %s", e)

    try:
        enabled = LLMModel.load_all_enabled() or {}
        return [k.strip().lower() for k in enabled.keys() if k.strip()]
    except Exception as e:
        log.warning("[CLI] Failed to load enabled models from DB: %s", e)
        return []


def parse_args():
    p = argparse.ArgumentParser(description="Option C v3.0 Q&A Pipeline (Service-layer + Run Tracking)")
    p.add_argument("input", help="Path to document OR structure/clean/questions/answers JSONL")
    p.add_argument(
        "--stage",
        type=str,
        default="full",
        choices=["structure", "clean", "questions", "answers", "rank", "export", "full"],
        help="Which part of the pipeline to run",
    )
    p.add_argument("--min-context-len", type=int, default=40)
    p.add_argument("--max-chunks", type=int, default=None)
    p.add_argument("--num-questions", type=int, default=DEFAULT_NUM_QUESTIONS)
    p.add_argument("--max-q-retries", type=int, default=DEFAULT_MAX_Q_RETRIES)
    p.add_argument("--similarity-threshold", type=float, default=DEFAULT_SIMILARITY_THRESHOLD)
    p.add_argument("--models", type=str, default=None, help="Comma-separated list of models. Default: all enabled models.")
    p.add_argument("--embed", action="store_true", help="If set, store embeddings.")
    p.add_argument("--export-format", type=str, choices=["alpaca", "chatml", "orpo"], default="alpaca")
    p.add_argument("--test", action="store_true", help="Run minimal 1-chunk/1-question wiring test.")
    return p.parse_args()


def main():
    args = parse_args()
    inp = Path(args.input)

    if not inp.exists():
        log.error("Input not found: %s", inp)
        sys.exit(1)

    if args.models:
        selected_models = [m.strip().lower() for m in args.models.split(",") if m.strip()]
    else:
        log.info("[CLI] --models not supplied; loading enabled LLMs from qna_models...")
        selected_models = _get_default_models_from_db()
        log.info("[CLI] Enabled models from DB: %s", selected_models)

    if args.stage == "structure":
        stage_structure_only(inp)
        return

    if args.stage == "clean":
        struct = inp if inp.name.endswith("_structure.json") else stage_structure_only(inp)
        stage_clean_chunks(struct, min_len=args.min_context_len)
        return

    if args.stage == "questions":
        if inp.name.endswith("_clean.jsonl"):
            clean = inp
        else:
            struct = stage_structure_only(inp)
            clean = stage_clean_chunks(struct, min_len=args.min_context_len)

        stage_generate_questions(
            clean,
            num_questions=args.num_questions,
            max_q_retries=args.max_q_retries,
            similarity_threshold=args.similarity_threshold,
            max_chunks=args.max_chunks,
        )
        return

    if args.stage == "answers":
        if inp.name.endswith("_questions.jsonl"):
            qs = inp
        else:
            struct = stage_structure_only(inp)
            clean = stage_clean_chunks(struct, min_len=args.min_context_len)
            qs = stage_generate_questions(
                clean,
                num_questions=args.num_questions,
                max_q_retries=args.max_q_retries,
                similarity_threshold=args.similarity_threshold,
                max_chunks=args.max_chunks,
            )

        stage_generate_answers(qs, models=selected_models, max_chunks=args.max_chunks)
        return

    if args.stage == "rank":
        stage_rank_answers()
        return

    if args.stage == "export":
        if inp.name.endswith("_answers.jsonl"):
            answers_jsonl = inp

        elif inp.name.endswith("_questions.jsonl"):
            answers_jsonl = stage_generate_answers(inp, models=selected_models, max_chunks=args.max_chunks)

        elif inp.name.endswith("_clean.jsonl"):
            qs = stage_generate_questions(
                inp,
                num_questions=args.num_questions,
                max_q_retries=args.max_q_retries,
                similarity_threshold=args.similarity_threshold,
                max_chunks=args.max_chunks,
            )
            answers_jsonl = stage_generate_answers(qs, models=selected_models, max_chunks=args.max_chunks)

        else:
            struct = stage_structure_only(inp)
            clean = stage_clean_chunks(struct, min_len=args.min_context_len)
            qs = stage_generate_questions(
                clean,
                num_questions=args.num_questions,
                max_q_retries=args.max_q_retries,
                similarity_threshold=args.similarity_threshold,
                max_chunks=args.max_chunks,
            )
            answers_jsonl = stage_generate_answers(qs, models=selected_models, max_chunks=args.max_chunks)

        stage_export_dataset(answers_jsonl=answers_jsonl, export_format=args.export_format)
        return

    # FULL PIPELINE
    if not selected_models:
        log.error("[CLI] No enabled LLM models found and none specified via --models.")
        sys.exit(1)

    if args.test:
        return run_full_pipeline(
            inp,
            max_chunks=1,
            min_context_len=20,
            num_questions=1,
            max_q_retries=0,
            similarity_threshold=0.0,
            models=selected_models,
            embed=False,
            test_mode=True,
        )

    run_full_pipeline(
        inp,
        max_chunks=args.max_chunks,
        min_context_len=args.min_context_len,
        num_questions=args.num_questions,
        max_q_retries=args.max_q_retries,
        similarity_threshold=args.similarity_threshold,
        models=selected_models,
        embed=args.embed,
        test_mode=False,
    )


if __name__ == "__main__":
    main()
