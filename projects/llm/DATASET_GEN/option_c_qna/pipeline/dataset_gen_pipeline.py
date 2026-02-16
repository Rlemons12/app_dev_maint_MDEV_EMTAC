from pathlib import Path
from typing import List, Dict, Optional, Tuple, Any

# -------------------------------------------------------------------
# DATASET GENERATION PIPELINE CLASS
# -------------------------------------------------------------------


class DatasetGenPipeline:
    """
    Full inline class version of the Option C v3.0 Q&A dataset pipeline.

    This class encapsulates:
        - Model caching
        - Structure extraction
        - Cleaning + DB insert
        - Question generation
        - Answer generation + ranking
        - Export (alpaca/chatml/orpo)
        - DB-only re-ranking
        - Full run orchestration with PipelineRun tracking

    Logic is preserved from the original module; only minimal
    changes were made to:
        - Convert free functions into methods.
        - Route internal calls through `self.*`.
    """

    # ----------------------------------------------------------------
    # CLASS-LEVEL MODEL CACHES
    # ----------------------------------------------------------------
    _FLAN_QA_MODEL: Optional[FLAN_QA_Model] = None
    _ANSWER_MODEL_CACHE: Optional[Dict[str, Any]] = None

    def __init__(self) -> None:
        # Currently stateless; all heavy objects are cached at class level.
        pass

    # =================================================================
    # PIPELINE RUN HELPERS (PipelineRun only – everything else via svc)
    # =================================================================

    def _bind_run_to_document(self, session, run_id: int, document_id: int) -> None:
        """
        Links a PipelineRun to the Document it processed.
        MUST run early so PipelineRun exists before adding PipelineRunItems.
        """

        if run_id is None:
            logger.error("[RUN] WARNING: run_id=None — cannot bind document to run")
            return

        from option_c_qna.qanda_db import PipelineRun  # local import to avoid circulars

        run = (
            session.query(PipelineRun)
            .filter(PipelineRun.id == run_id)
            .first()
        )

        if run is None:
            logger.error(f"[RUN] ERROR: No PipelineRun found with id={run_id}")
            return

        logger.info(f"[RUN] Binding document_id={document_id} → run_id={run_id}")

        run.document_id = document_id

        try:
            session.commit()   # REQUIRED so that run_id becomes durable
        except Exception as exc:
            session.rollback()
            logger.error(f"[RUN] ERROR: Failed to bind run_id={run_id} to document_id={document_id}")
            logger.info(exc)
            raise

    def _create_pipeline_run(
        self,
        session,
        document_id: Optional[int],
        run_type: str,
        options_json: Dict[str, Any],
        models_json: Dict[str, Any],
        env_json: Dict[str, Any],
    ) -> int:
        """
        Create a PipelineRun row and commit it.
        Returns the run_id.
        """
        run = PipelineRun(
            document_id=document_id,
            run_type=run_type,
            options_json=options_json,
            models_json=models_json,
            env_json=env_json,
            started_at=datetime.now(timezone.utc),   # explicit UTC
        )

        session.add(run)
        session.flush()

        try:
            session.commit()
        except Exception:
            session.rollback()
            raise

        return run.id

    def _attach_document_to_run(self, run_id: int, document_id: int) -> None:
        """
        Set PipelineRun.document_id once the Document exists.
        """
        session = get_qna_session()
        try:
            run = session.get(PipelineRun, run_id)
            if not run:
                logger.warning("[RUN] attach_document: run id=%s not found", run_id)
                return
            run.document_id = document_id
            session.commit()
            logger.info(
                "[RUN] Attached document_id=%s to run_id=%s",
                document_id,
                run_id,
            )
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()

    def _finish_pipeline_run(
        self,
        run_id: int,
        success: bool,
        error_message: Optional[str] = None,
    ) -> None:
        """
        Marks a pipeline run as finished.
        Updates finished_at, success state, and error_message.
        """

        from option_c_qna.qanda_db import get_qa_service

        qa_service = get_qa_service()
        session = qa_service._auto_session()

        run = session.get(PipelineRun, run_id)
        if not run:
            raise RuntimeError(f"PipelineRun id={run_id} not found.")

        # Always store timestamps in UTC
        now = datetime.now(timezone.utc)

        # Update timestamp + status
        run.finished_at = now
        run.success = success

        # Optional error message
        if error_message:
            run.error_message = str(error_message)[:4000]  # safe trim

        try:
            session.commit()
        except Exception as e:
            session.rollback()
            raise RuntimeError(f"Failed to finish pipeline run id={run_id}: {e}")

    # =================================================================
    # MODEL LOADING / CACHING
    # =================================================================

    def get_flan_model(self) -> FLAN_QA_Model:
        """
        Lazily load FLAN-T5-Large once per process and reuse.
        Used for question generation and (optionally) as an answer model.
        """
        if self._FLAN_QA_MODEL is None:
            logger.info("[MODEL] Loading FLAN-T5-Large (FLAN_QA_Model) once...")
            type(self)._FLAN_QA_MODEL = FLAN_QA_Model()
        return self._FLAN_QA_MODEL  # type: ignore[return-value]

    def _init_answer_models(
        self,
        selected_models: Optional[List[str]] = None,
        model_cache: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Load answer LLMs with caching support.

        Priority of sources:
            1. If model_cache is provided → use that (direct injection)
            2. Else if global cache exists → reuse it
            3. Else → load from qna_models table and cache globally

        Args:
            selected_models: Optional list of model names supplied via CLI
            model_cache: Optional dict {name: model_instance} (preloaded externally)

        Returns:
            Dict[str, Any]: {model_name: model_instance}
        """

        # ------------------------------------------------------------
        # STEP 0 — Directly supplied preloaded cache
        # ------------------------------------------------------------
        if model_cache is not None:
            logger.info("[ANSWERS] Using externally provided preloaded model cache.")
            all_models = {name.lower(): obj for name, obj in model_cache.items()}

        # ------------------------------------------------------------
        # STEP 0.5 — Use class cache if already populated
        # ------------------------------------------------------------
        elif self._ANSWER_MODEL_CACHE is not None:
            logger.info("[ANSWERS] Using global preloaded model cache.")
            all_models = {name.lower(): obj for name, obj in self._ANSWER_MODEL_CACHE.items()}

        else:
            # ------------------------------------------------------------
            # STEP 1 — Load from DB and populate the class cache
            # ------------------------------------------------------------
            try:
                raw_models = LLMModel.load_all_enabled()
            except Exception as exc:
                logger.error("[ANSWERS] Failed to load enabled models from qna_models.")
                logger.exception(exc)
                raise

            all_models = {name.lower(): obj for name, obj in raw_models.items()}

            if not all_models:
                logger.error("[ANSWERS] No enabled LLM models found in qna_models!")
                raise RuntimeError("No enabled LLM models available.")

            type(self)._ANSWER_MODEL_CACHE = all_models
            logger.info("[ANSWERS] Cached %d answer models.", len(all_models))

        # ------------------------------------------------------------
        # STEP 2 — No CLI filter → return all models
        # ------------------------------------------------------------
        if not selected_models:
            logger.info("[ANSWERS] Using ALL enabled models: %s", list(all_models.keys()))
            return all_models

        # ------------------------------------------------------------
        # STEP 3 — Normalize the requested model list
        # ------------------------------------------------------------
        selected_lower = [m.strip().lower() for m in selected_models if m.strip()]

        # ------------------------------------------------------------
        # STEP 4 — Apply filter
        # ------------------------------------------------------------
        filtered = {
            name: obj for name, obj in all_models.items()
            if name in selected_lower
        }

        # ------------------------------------------------------------
        # STEP 5 — Warn about invalid model names
        # ------------------------------------------------------------
        for req in selected_lower:
            if req not in all_models:
                logger.warning(
                    f"[ANSWERS] Requested model '{req}' is NOT enabled or not found in qna_models."
                )

        # ------------------------------------------------------------
        # STEP 6 — Enforce valid result set
        # ------------------------------------------------------------
        if not filtered:
            logger.error(
                "[ANSWERS] No valid answer models remain after filtering.\n"
                "Requested=%s\nEnabled=%s",
                selected_lower,
                list(all_models.keys())
            )
            raise RuntimeError("No valid answer models remain after filtering.")

        logger.info("[ANSWERS] Using models: %s", list(filtered.keys()))
        return filtered

    def preload_all_answer_models(self) -> Dict[str, Any]:
        """
        Loads all enabled answer models from qna_models table ONE TIME.
        Returns a dict: {model_name: model_instance}

        - Safe offline loading
        - Logs failures but continues with remaining models
        - Populates the class-level _ANSWER_MODEL_CACHE
        """
        # Already loaded → return cache
        if self._ANSWER_MODEL_CACHE is not None:
            logger.info("[PRELOAD] Using existing global model cache.")
            return self._ANSWER_MODEL_CACHE

        # Load enabled model records from DB
        try:
            enabled_models = LLMModel.load_all_enabled()  # {"flan": instance, ...}
        except Exception as exc:
            logger.error("[PRELOAD] Failed to load enabled models from qna_models.")
            logger.exception(exc)
            raise

        if not enabled_models:
            raise RuntimeError("[PRELOAD] No enabled models found in qna_models table.")

        logger.info("[PRELOAD] Found %d enabled models: %s",
                 len(enabled_models), list(enabled_models.keys()))

        loaded: Dict[str, Any] = {}

        # Loop over each enabled model
        for name, model_obj in enabled_models.items():
            try:
                logger.info(f"[PRELOAD] Initializing model '{name}'...")
                _ = model_obj  # ensures constructor ran

                loaded[name.lower()] = model_obj
                logger.info(f"[PRELOAD] Model '{name}' loaded successfully.")

            except Exception as exc:
                logger.error(f"[PRELOAD] FAILED to load model '{name}'. Skipping.")
                logger.exception(exc)
                continue

        if not loaded:
            raise RuntimeError("[PRELOAD] All models failed to load!")

        type(self)._ANSWER_MODEL_CACHE = loaded

        logger.info("[PRELOAD] Successfully preloaded %d models.", len(loaded))
        return loaded

    # =================================================================
    # EMBEDDING HELPER
    # =================================================================

    def compute_embedding(self, text: str) -> Optional[List[float]]:
        """
        Compute a vector embedding for text using MiniLM, if available.
        Returns list[float] or None.
        """
        if not HAS_SIM_MODEL or not text:
            return None
        try:
            vec = _similarity_model.encode(text)
            return vec.tolist()
        except Exception as e:  # noqa: BLE001
            logger.warning("[EMBED] Failed to compute embedding: %s", e)
            return None

    # =================================================================
    # QUESTION TEMPLATE + QUALITY + SIMILARITY LOGIC
    # =================================================================

    def apply_question_templates(self, raw_questions: List[str]) -> List[str]:
        """
        Normalizes questions into a small set of controlled stems using a
        very light heuristic. This keeps them more consistent for training.
        """
        import re

        normalized: List[str] = []

        for idx, q in enumerate(raw_questions):
            q = (q or "").strip()
            if not q:
                continue

            # If it already looks fine, keep as-is
            if q[0].isupper() and q.endswith("?"):
                normalized.append(q)
                continue

            # Extract a rough "focus" phrase after the question word
            match = re.search(
                r"(what|where|which|when|who)\s+(.*)",
                q,
                flags=re.IGNORECASE,
            )
            if match:
                focus = match.group(2).strip().rstrip("?")
            else:
                focus = q.rstrip("?")

            template = QUESTION_TEMPLATES[idx % len(QUESTION_TEMPLATES)]
            normalized.append(template.format(focus=focus))

        return normalized

    def question_quality_pass(self, question: str, context: str) -> bool:
        """
        PASS 1 – Basic quality gate for generated questions.
        Returns True if the question is worth keeping.
        """
        if not question:
            return False

        q = question.strip()
        if len(q) < DEFAULT_MIN_QUESTION_LEN:
            return False

        lower_q = q.lower()
        if not lower_q.startswith(
            ("what", "where", "which", "when", "who")
        ):
            return False

        if not q.endswith("?"):
            return False

        # Must share some words with context (simple overlap check)
        q_words = set(lower_q.rstrip("?").split())
        c_words = set(context.lower().split())
        overlap = len(q_words & c_words)
        if overlap < 2:
            return False

        return True

    def dedupe_questions(
        self,
        questions: List[str],
        similarity_threshold: float = DEFAULT_SIMILARITY_THRESHOLD,
    ) -> List[str]:
        """
        PASS 2 – Remove near-duplicate questions using MiniLM embeddings,
        if available. Falls back to naive set-based dedupe if model is missing.
        """
        cleaned: List[str] = []

        if not questions:
            return cleaned

        if not HAS_SIM_MODEL:
            # Fallback: simple unique filter with lowercasing
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
                    # Too similar to something we already kept
                    continue

            embeddings.append(emb)
            cleaned.append(q)

        return cleaned

    def generate_questions_multi_pass(
        self,
        context: str,
        flan_model: Optional[FLAN_QA_Model] = None,
        n: int = DEFAULT_NUM_QUESTIONS,
        max_retries: int = DEFAULT_MAX_Q_RETRIES,
        similarity_threshold: float = DEFAULT_SIMILARITY_THRESHOLD,
    ) -> List[str]:
        """
        Core question generation routine:

            1) Generate with FLAN
            2) Apply templates
            3) PASS 1 (quality)
            4) PASS 2 (similarity dedupe)
            5) Retry if we end up with nothing
        """
        last_raw: List[str] = []

        if flan_model is None:
            flan_model = self.get_flan_model()

        for attempt in range(1, max_retries + 1):
            raw = flan_model.generate_questions(context, n=n) or []
            last_raw = raw

            templated = self.apply_question_templates(raw)
            passed = [q for q in templated if self.question_quality_pass(q, context)]

            deduped = self.dedupe_questions(passed, similarity_threshold=similarity_threshold)

            if deduped:
                logger.debug(
                    "[QUESTION] PASS pipeline success on attempt %d – kept %d/%d "
                    "questions (context length=%d chars).",
                    attempt,
                    len(deduped),
                    len(raw),
                    len(context),
                )
                return deduped

            logger.debug(
                "[QUESTION] PASS pipeline failed on attempt %d – retrying. "
                "raw=%d, passed=%d, deduped=%d",
                attempt,
                len(raw),
                len(passed),
                len(deduped),
            )

        # Fallback: return whatever we last got, even if junk
        if last_raw:
            logger.warning(
                "[QUESTION] Multi-pass generation exhausted retries; returning "
                "last raw FLAN questions without filtering."
            )
            templated = self.apply_question_templates(last_raw)
            return templated

        logger.warning("[QUESTION] Multi-pass generation produced no questions at all.")
        return []
    # =================================================================
    # ANSWER RELEVANCE + RANKING
    # =================================================================

    def answer_relevance_check(self, question: str, answer: str, context: str) -> float:
        """
        Light relevance check between (Q+A) and context using MiniLM if present.
        Returns a float similarity score (0.0–1.0).
        """
        if not HAS_SIM_MODEL:
            return 0.0

        try:
            combo = f"{question} || {answer}"
            v1 = _similarity_model.encode(combo)
            v2 = _similarity_model.encode(context)
            sim = util.cos_sim(v1, v2).item()
            return float(sim)
        except Exception as e:
            logger.debug("[SIM] Relevance check failed: %s", e)
            return 0.0

    def rank_answers(self) -> None:
        """
        DB-only ranking of all answers in `pipeline_run_answers` table.
        Operates without regenerating anything.

        Uses MiniLM if available; otherwise falls back to simple heuristics.
        """
        from option_c_qna.qanda_db import get_qa_service, PipelineRunAnswer

        qa_service = get_qa_service()
        session = qa_service._auto_session()

        try:
            all_rows = (
                session.query(PipelineRunAnswer)
                .filter(PipelineRunAnswer.score.is_(None))
                .all()
            )

            if not all_rows:
                logger.info("[RANK] No unrated answers found.")
                return

            logger.info("[RANK] Ranking %d answers...", len(all_rows))

            for row in all_rows:
                q = row.question_text or ""
                a = row.answer_text or ""
                ctx = row.context_text or ""

                score = 0.0
                if HAS_SIM_MODEL:
                    try:
                        v1 = _similarity_model.encode(f"{q} || {a}")
                        v2 = _similarity_model.encode(ctx)
                        score = float(util.cos_sim(v1, v2).item())
                    except Exception:
                        score = 0.0
                else:
                    # Heuristic fallback
                    overlap = len(set(q.lower().split()) & set(ctx.lower().split()))
                    score = min(1.0, overlap / 15.0)

                row.score = score

            session.commit()
            logger.info("[RANK] Completed ranking.")

        except Exception:
            session.rollback()
            raise

        finally:
            session.close()

    # =================================================================
    # STRUCTURE EXTRACTION
    # =================================================================

    def stage_structure_only(self, doc_path: Path) -> Path:
        """
        Stage 1: Structure extraction (Document → structure.json)
        """
        extractor = DocumentStructureExtractor(doc_path)
        result = extractor.extract_structure()

        if not result or "structure_json" not in result:
            raise RuntimeError(
                f"[STRUCTURE] DocumentStructureExtractor failed for {doc_path}"
            )

        struct_path = Path(result["structure_json"])
        logger.info("[STRUCTURE] Saved: %s", struct_path)
        return struct_path

    # =================================================================
    # CLEAN CHUNKS (Stage 2)
    # =================================================================

    def stage_clean_chunks(
        self,
        structure_json: Path,
        min_len: int = 40,
    ) -> Path:
        """
        Stage 2: Load structure.json, clean + normalize chunks, save _clean.jsonl.
        """
        loader = StructureChunkLoader(structure_json)
        chunks = loader.load_cleaned_chunks(min_len=min_len)

        if not chunks:
            raise RuntimeError(
                f"[CLEAN] No cleaned chunks produced from {structure_json}"
            )

        out_path = structure_json.with_name(
            structure_json.stem.replace("_structure", "") + "_clean.jsonl"
        )

        with out_path.open("w", encoding="utf-8") as f:
            for ch in chunks:
                f.write(json.dumps(ch, ensure_ascii=False) + "\n")

        logger.info("[CLEAN] Saved %d chunks → %s", len(chunks), out_path)
        return out_path

    def stage_clean_chunks_tracked(
        self,
        doc_path: Path,
        structure_json: Path,
        min_len: int,
        run_id: int,
        embed: bool,
    ) -> Tuple[Path, int, List[Dict[str, Any]]]:
        """
        Stage 2 (tracked): inserts cleaned chunks into DB and produces
        (output_path, document_id, chunk_records).
        """
        from option_c_qna.qanda_db import get_qa_service

        qa_service = get_qa_service()
        session = qa_service._auto_session()

        # ------------------------------------------------------------
        # PART 1 — Load cleaned chunks
        # ------------------------------------------------------------
        loader = StructureChunkLoader(structure_json)
        cleaned = loader.load_cleaned_chunks(min_len=min_len)

        if not cleaned:
            raise RuntimeError("[CLEAN/TRACKED] No cleaned chunks loaded.")

        # ------------------------------------------------------------
        # PART 2 — Determine or create Document row
        # ------------------------------------------------------------
        doc_title = doc_path.name
        document = qa_service.get_or_create_document(session, doc_title, str(doc_path))
        document_id = document.id

        # Attach document_id to run
        self._attach_document_to_run(run_id, document_id)

        # ------------------------------------------------------------
        # PART 3 — Insert chunks into DB
        # ------------------------------------------------------------
        out: List[Dict[str, Any]] = []

        for idx, chunk in enumerate(cleaned, start=1):
            text = chunk["text"]
            section = chunk.get("section", "")

            emb = self.compute_embedding(text) if embed else None

            rec = qa_service.insert_clean_chunk(
                session=session,
                run_id=run_id,
                document_id=document_id,
                section=section,
                text=text,
                order_index=idx,
                embedding=emb,
            )

            out.append(rec)

        session.commit()

        # ------------------------------------------------------------
        # PART 4 — Write _clean.jsonl (as in untracked version)
        # ------------------------------------------------------------
        clean_path = structure_json.with_name(
            structure_json.stem.replace("_structure", "") + "_clean.jsonl"
        )

        with clean_path.open("w", encoding="utf-8") as f:
            for chunk in cleaned:
                f.write(json.dumps(chunk, ensure_ascii=False) + "\n")

        logger.info(
            "[CLEAN/TRACKED] Inserted %d chunks (doc_id=%s) → %s",
            len(out),
            document_id,
            clean_path,
        )

        return clean_path, document_id, out
    # =====================================================================
    # ANSWER RELEVANCE + RANKING  (original logic)
    # =====================================================================

    def answer_relevance_check(self, answer: str, context: str) -> bool:
        """
        Simple check that an answer is at least somewhat grounded in the context.
        (Original boolean version)
        """
        if not answer:
            return False

        ans = answer.strip()
        if len(ans.split()) < 3:
            return False

        a_words = set(ans.lower().split())
        c_words = set(context.lower().split())
        overlap = len(a_words & c_words)

        return overlap >= 2

    def rank_answers(
        self,
        answers: Dict[str, str],
        context: str,
    ) -> List[Tuple[str, str, float]]:
        """
        Compute a simple score for each model's answer and return sorted list:
            [(model_name, answer_text, score), ...] (descending by score)

        Scoring:
            - +overlap word count with context
            - +1 if length in [5, 40] words
            - -5 if overlap == 0 (likely hallucination)
        """
        ranked: List[Tuple[str, str, float]] = []
        c_words = set(context.lower().split())

        for model_name, ans in answers.items():
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
    # STAGE 3 — QUESTIONS (legacy, no run tracking)
    # ======================================================================

    def stage_generate_questions(
        self,
        clean_jsonl: Path,
        num_questions: int = DEFAULT_NUM_QUESTIONS,
        max_q_retries: int = DEFAULT_MAX_Q_RETRIES,
        similarity_threshold: float = DEFAULT_SIMILARITY_THRESHOLD,
        max_chunks: Optional[int] = None,
    ) -> Path:
        """
        Legacy per-stage version used by CLI --stage questions (no run tracking).
        """
        QUESTION_DIR.mkdir(parents=True, exist_ok=True)
        out = QUESTION_DIR / f"{clean_jsonl.stem}_questions.jsonl"

        logger.info("[QUESTION] Loading FLAN model for question generation...")
        flan = self.get_flan_model()

        # Load all cleaned chunks
        with open(clean_jsonl, "r", encoding="utf-8") as fin:
            clean_chunks = [json.loads(line) for line in fin]

        if max_chunks is not None:
            clean_chunks = clean_chunks[:max_chunks]

        if not clean_chunks:
            logger.warning("[QUESTION] No chunks found in clean JSONL; nothing to do.")
            return out

        # Resolve chunk_id -> DB chunk.id
        chunk_ids = {c["chunk_id"] for c in clean_chunks}
        logger.info(f"[QUESTION] Resolving {len(chunk_ids)} chunk_ids to DB rows...")

        session = get_qna_session()
        try:
            db_chunks = (
                session.query(Chunk)
                .filter(Chunk.chunk_id.in_(list(chunk_ids)))
                .all()
            )
            chunk_map = {c.chunk_id: c.id for c in db_chunks}
        finally:
            session.close()

        missing_chunk_ids = chunk_ids - set(chunk_map.keys())
        if missing_chunk_ids:
            logger.warning(
                "[QUESTION] %d chunk_ids not found in DB (first few: %s)",
                len(missing_chunk_ids),
                list(missing_chunk_ids)[:5],
            )

        written_groups = 0
        total_questions = 0

        svc = get_qa_service()

        with open(out, "w", encoding="utf-8") as fout:
            for chunk in clean_chunks:
                context = chunk["pipeline_context"]
                pipeline_chunk_id = chunk["chunk_id"]

                flan_questions = self.generate_questions_multi_pass(
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
                    logger.warning(
                        "[QUESTION] No DB chunk found for chunk_id=%s; "
                        "skipping DB insert for this chunk.",
                        pipeline_chunk_id,
                    )
                    continue

                # Legacy mode: no run_id, so we go straight to ORM session
                session = get_qna_session()
                try:
                    for idx, question_text in enumerate(flan_questions, start=1):
                        q_obj = Question(
                            chunk_id=db_chunk_id,
                            question=question_text,
                            question_index=idx,
                        )
                        session.add(q_obj)
                        total_questions += 1
                    session.commit()
                except Exception:
                    session.rollback()
                    raise
                finally:
                    session.close()

                written_groups += 1

        logger.info(
            "[QUESTION] Wrote %d chunk question groups to %s",
            written_groups,
            out,
        )
        logger.info("[QUESTION] Inserted %d questions into DB", total_questions)

        return out

    # ======================================================================
    # STAGE 3 (FULL PIPELINE) — QUESTIONS + RUN TRACKING (SERVICE)
    # Optimized: no JSONL reload, returns question_records in-memory
    # ======================================================================

    def stage_generate_questions_tracked(
        self,
        clean_chunks: List[Dict[str, Any]],   # in-memory
        document_id: int,
        run_id: int,
        num_questions: int,
        max_q_retries: int,
        similarity_threshold: float,
        max_chunks: Optional[int],
        embed: bool,
    ) -> Tuple[Path, List[Dict[str, Any]]]:
        """
        Optimized full pipeline version:
            - uses clean_chunks *already in memory*
            - generates questions
            - inserts qna_questions via service
            - attaches PipelineRunItem (handled inside service)
            - optionally stores question embeddings
            - returns (artifact_path, question_records_in_memory)
        """

        QUESTION_DIR.mkdir(parents=True, exist_ok=True)
        out = QUESTION_DIR / "questions.jsonl"

        logger.info("[QUESTION/TRACK] Loading FLAN model for question generation...")
        flan = self.get_flan_model()
        svc = get_qa_service()

        # -------------------------------------------------------------
        # If max_chunks is set, trim in-memory chunks list
        # -------------------------------------------------------------
        if max_chunks is not None:
            clean_chunks = clean_chunks[:max_chunks]

        if not clean_chunks:
            logger.warning("[QUESTION/TRACK] No cleaned chunks received; nothing to do.")
            return out, []

        # -------------------------------------------------------------
        # Build quick map of pipeline_chunk_id → DB Chunk object
        # -------------------------------------------------------------
        session = get_qna_session()
        try:
            doc = session.get(Document, document_id)
            if not doc:
                logger.error("[QUESTION/TRACK] Document id=%s not found", document_id)
                return out, []

            chunk_by_chunk_id = {c.chunk_id: c for c in doc.chunks}
        finally:
            session.close()

        # -------------------------------------------------------------
        # Generate questions for each chunk (in memory)
        # -------------------------------------------------------------
        records: List[Dict[str, Any]] = []
        for chunk in clean_chunks:
            context = chunk["pipeline_context"]
            pipeline_chunk_id = chunk["chunk_id"]

            flan_questions = self.generate_questions_multi_pass(
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

        # -------------------------------------------------------------
        # Write JSONL artifact (no re-reading later)
        # -------------------------------------------------------------
        with open(out, "w", encoding="utf-8") as fout:
            for rec in records:
                fout.write(json.dumps(rec, ensure_ascii=False) + "\n")

        # -------------------------------------------------------------
        # Insert questions into DB
        # -------------------------------------------------------------
        total_questions = 0
        for rec in records:
            pipeline_chunk_id = rec["chunk_id"]
            q_texts = rec["questions"] or []

            chunk_obj = chunk_by_chunk_id.get(pipeline_chunk_id)
            if not chunk_obj:
                logger.warning(
                    "[QUESTION/TRACK] No DB chunk found for chunk_id=%s; skipping.",
                    pipeline_chunk_id,
                )
                continue

            for idx, question_text in enumerate(q_texts, start=1):
                q_obj = svc.add_question(
                    run_id=run_id,
                    chunk_id=chunk_obj.id,
                    question_text=question_text,
                    question_index=idx,
                )
                total_questions += 1

                # Optional embedding
                if embed:
                    vec = self.compute_embedding(q_obj.question)
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

        # -------------------------------------------------------------
        # Logging
        # -------------------------------------------------------------
        logger.info(
            "[QUESTION/TRACK] Inserted %d questions into DB for document_id=%s (run_id=%s)",
            total_questions,
            document_id,
            run_id,
        )
        logger.info("[QUESTION/TRACK] Wrote questions JSONL -> %s", out)

        # -------------------------------------------------------------
        # Return JSONL path + in-memory question_records (NEW)
        # -------------------------------------------------------------
        return out, records

    # ======================================================================
    # STAGE 4 — ANSWERS (legacy, no run tracking)
    # ======================================================================

    def stage_generate_answers(
        self,
        question_jsonl: Path,
        models: Optional[List[str]] = None,
        max_chunks: Optional[int] = None,
    ) -> Path:
        """
        Legacy per-stage version, used by CLI --stage answers.
        Keeps behavior unchanged (no run tracking).
        """
        ANSWER_DIR.mkdir(parents=True, exist_ok=True)
        out = ANSWER_DIR / f"{question_jsonl.stem}_answers.jsonl"

        logger.info("[ANSWERS] Loading answer models...")
        model_objects = self._init_answer_models(models)

        with open(question_jsonl, "r", encoding="utf-8") as fin:
            question_items = [json.loads(line) for line in fin]

        if max_chunks is not None:
            question_items = question_items[:max_chunks]

        if not question_items:
            logger.warning("[ANSWERS] No question records found; nothing to do.")
            return out

        pipeline_chunk_ids = {item["chunk_id"] for item in question_items}

        session = get_qna_session()
        try:
            db_chunks = (
                session.query(Chunk)
                .filter(Chunk.chunk_id.in_(list(pipeline_chunk_ids)))
                .all()
            )
            chunk_map = {c.chunk_id: c.id for c in db_chunks}

            db_questions = (
                session.query(Question)
                .filter(Question.chunk_id.in_(list(chunk_map.values())))
                .all()
            )
            question_map = {(q.chunk_id, q.question_index): q.id for q in db_questions}
        finally:
            session.close()

        missing_chunk_ids = pipeline_chunk_ids - set(chunk_map.keys())
        if missing_chunk_ids:
            logger.warning(
                "[ANSWERS] %d chunk_ids have questions but no DB chunk; first few: %s",
                len(missing_chunk_ids),
                list(missing_chunk_ids)[:5],
            )

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
                    logger.warning(
                        "[ANSWERS] No DB chunk found for chunk_id=%s; "
                        "skipping DB answer inserts for this chunk's questions.",
                        pipeline_chunk_id,
                    )
                    continue

                for idx, q_text in enumerate(questions, start=1):
                    per_model_best_answer: Dict[str, str] = {}
                    per_model_samples: Dict[str, List[Dict[str, Any]]] = {}

                    for model_name, model_obj in model_objects.items():
                        sample_answers: Dict[str, str] = {}

                        # deterministic samples
                        for i in range(NUM_DETERMINISTIC_SAMPLES):
                            sample_key = f"{model_name}#det{i+1}"
                            ans = model_obj.generate_answer(context, q_text)
                            sample_answers[sample_key] = ans

                        # stochastic samples
                        for i in range(NUM_STOCHASTIC_SAMPLES):
                            sample_key = f"{model_name}#stoch{i+1}"`
                            ans = model_obj.generate_answer(context, q_text)
                            sample_answers[sample_key] = ans

                        ranked_samples = self.rank_answers(sample_answers, context)

                        if not ranked_samples:
                            logger.warning(
                                "[ANSWERS] Model %s produced no usable samples "
                                "for chunk_id=%s question_index=%d",
                                model_name,
                                pipeline_chunk_id,
                                idx,
                            )
                            continue

                        best_key, best_ans, best_score = ranked_samples[0]
                        if len(ranked_samples) > 1:
                            worst_key, worst_ans, worst_score = ranked_samples[-1]
                        else:
                            worst_key, worst_ans, worst_score = (
                                best_key,
                                best_ans,
                                best_score,
                            )

                        per_model_best_answer[model_name] = best_ans

                        per_model_samples[model_name] = [
                            {
                                "sample_id": key,
                                "answer": ans_text,
                                "score": float(score),
                            }
                            for (key, ans_text, score) in ranked_samples
                        ]

                    if not per_model_best_answer:
                        logger.warning(
                            "[ANSWERS] No models produced valid answers for "
                            "chunk_id=%s question_index=%d",
                            pipeline_chunk_id,
                            idx,
                        )
                        continue

                    cross_model_ranked = self.rank_answers(per_model_best_answer, context)

                    best_model, best_answer, best_score = cross_model_ranked[0]
                    if len(cross_model_ranked) > 1:
                        worst_model, worst_answer, worst_score = cross_model_ranked[-1]
                    else:
                        worst_model, worst_answer, worst_score = (
                            best_model,
                            best_answer,
                            best_score,
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
                        logger.warning(
                            "[ANSWERS] No DB question row for chunk_id=%s, db_chunk_id=%s, "
                            "question_index=%d; skipping DB inserts for this question.",
                            pipeline_chunk_id,
                            db_chunk_id,
                            idx,
                        )
                        continue

                    # Insert answers + ranking directly via ORM
                    session = get_qna_session()
                    try:
                        for rank_idx, (model_name, best_ans_for_model, score) in enumerate(
                            cross_model_ranked
                        ):
                            is_best = rank_idx == 0
                            is_worst = rank_idx == len(cross_model_ranked) - 1 and len(
                                cross_model_ranked
                            ) > 1

                            a_obj = Answer(
                                question_id=db_question_id,
                                model_name=model_name,
                                model_type="causal_lm",
                                model_path=None,
                                answer_text=best_ans_for_model,
                                score=float(score),
                                is_best=is_best,
                                is_worst=is_worst,
                            )
                            session.add(a_obj)
                            total_answers += 1

                        r_obj = AnswerRanking(
                            question_id=db_question_id,
                            best_model=best_model,
                            best_answer=best_answer,
                            worst_model=worst_model,
                            worst_answer=worst_answer,
                            answer_scores=answer_scores,
                        )
                        session.add(r_obj)
                        total_rankings += 1

                        session.commit()
                    except Exception:
                        session.rollback()
                        raise
                    finally:
                        session.close()

                    written_records += 1

        logger.info(
            "[ANSWERS] Wrote %d Q/A records → %s",
            written_records,
            out,
        )
        logger.info("[ANSWERS] Inserted %d answers into DB", total_answers)
        logger.info("[ANSWERS] Inserted %d ranking rows into DB", total_rankings)

        return out

    # ======================================================================
    # STAGE 4 (FULL PIPELINE) — ANSWERS + RUN TRACKING (SERVICE)
    # ======================================================================

    def stage_generate_answers_tracked(
        self,
        question_records: List[Dict[str, Any]],
        document_id: int,
        run_id: int,
        models: List[str],
        max_chunks: Optional[int],
        embed: bool,
        model_cache: Optional[Dict[str, Any]] = None,
    ) -> Path:
        """
        Full tracked answer generation stage for Option C pipeline.
        """

        ANSWER_DIR.mkdir(parents=True, exist_ok=True)
        out = ANSWER_DIR / "answers.jsonl"

        logger.info("[ANSWERS/TRACK] Loading answer models...")
        model_objects = self._init_answer_models(models, model_cache=model_cache)

        # -------------------------------------------------------------
        # Apply max_chunks limit
        # -------------------------------------------------------------
        if max_chunks is not None:
            question_records = question_records[:max_chunks]

        if not question_records:
            logger.warning("[ANSWERS/TRACK] No question records received; skipping answers.")
            return out

        # -------------------------------------------------------------
        # Build quick lookup: Document.chunks (DB-bound)
        # -------------------------------------------------------------
        session = get_qna_session()
        try:
            doc = session.get(Document, document_id)
            if not doc:
                raise RuntimeError(f"[ANSWERS/TRACK] Document id={document_id} not found.")
            chunk_by_chunk_id = {c.chunk_id: c for c in doc.chunks}
        finally:
            session.close()

        # -------------------------------------------------------------
        # Service for DB inserts
        # -------------------------------------------------------------
        svc = get_qa_service()

        # -------------------------------------------------------------
        # Answer artifact + accumulator
        # -------------------------------------------------------------
        written_records = 0
        total_answers = 0
        total_rankings = 0

        with open(out, "w", encoding="utf-8") as fout:

            # ---------------------------------------------------------
            # Iterate through each chunk's questions
            # ---------------------------------------------------------
            for rec in question_records:
                context = rec["context"]
                pipeline_chunk_id = rec["chunk_id"]
                q_texts = rec.get("questions") or []

                chunk_obj = chunk_by_chunk_id.get(pipeline_chunk_id)
                if not chunk_obj:
                    logger.warning(
                        "[ANSWERS/TRACK] Missing DB chunk for chunk_id=%s; skipping.",
                        pipeline_chunk_id,
                    )
                    continue

                # -----------------------------------------------------
                # Loop through each question in the chunk
                # -----------------------------------------------------
                for q_index, q_text in enumerate(q_texts, start=1):

                    per_model_best_answer: Dict[str, str] = {}
                    per_model_samples: Dict[str, List[Dict[str, Any]]] = {}

                    for model_name, model_obj in model_objects.items():
                        generated_samples: Dict[str, str] = {}

                        # Deterministic samples
                        for i in range(NUM_DETERMINISTIC_SAMPLES):
                            key = f"{model_name}#det{i+1}"
                            ans = model_obj.generate_answer(context, q_text)
                            generated_samples[key] = ans

                        # Stochastic samples
                        for i in range(NUM_STOCHASTIC_SAMPLES):
                            key = f"{model_name}#stoch{i+1}"
                            ans = model_obj.generate_answer(context, q_text)
                            generated_samples[key] = ans

                        # Rank samples for this model
                        ranked_samples = self.rank_answers(generated_samples, context)

                        if not ranked_samples:
                            logger.warning(
                                "[ANSWERS/TRACK] Model %s produced no ranked samples "
                                "(chunk_id=%s q_index=%d).",
                                model_name,
                                pipeline_chunk_id,
                                q_index,
                            )
                            continue

                        best_sample_key, best_ans, best_score = ranked_samples[0]

                        per_model_best_answer[model_name] = best_ans
                        per_model_samples[model_name] = [
                            {
                                "sample_id": key,
                                "answer": ans_text,
                                "score": float(score),
                            }
                            for (key, ans_text, score) in ranked_samples
                        ]

                    if not per_model_best_answer:
                        logger.warning(
                            "[ANSWERS/TRACK] No answers found for chunk_id=%s q_index=%d.",
                            pipeline_chunk_id,
                            q_index,
                        )
                        continue

                    cross_ranked = self.rank_answers(per_model_best_answer, context)

                    if not cross_ranked:
                        logger.warning(
                            "[ANSWERS/TRACK] Cross-model ranking failed for "
                            "chunk_id=%s q_index=%d.",
                            pipeline_chunk_id,
                            q_index,
                        )
                        continue

                    best_model, best_answer, best_score = cross_ranked[0]

                    if len(cross_ranked) > 1:
                        worst_model, worst_answer, worst_score = cross_ranked[-1]
                    else:
                        worst_model, worst_answer, worst_score = (
                            best_model,
                            best_answer,
                            best_score,
                        )

                    answer_scores = {m: float(s) for (m, _, s) in cross_ranked}

                    # -------------------------------------------------
                    # Write to artifact JSONL
                    # -------------------------------------------------
                    artifact_record = {
                        "chunk_id": pipeline_chunk_id,
                        "question_index": q_index,
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

                    fout.write(json.dumps(artifact_record, ensure_ascii=False) + "\n")
                    written_records += 1

                    # -------------------------------------------------
                    # Insert best/worst answers + ranking into DB
                    # -------------------------------------------------
                    db_q = svc.get_or_create_question(
                        run_id=run_id,
                        chunk_id=chunk_obj.id,
                        question_text=q_text,
                        question_index=q_index,
                    )

                    for rank_idx, (model_name, ans_text, score) in enumerate(cross_ranked):
                        a_obj = svc.add_answer(
                            run_id=run_id,
                            question_id=db_q.id,
                            model_name=model_name,
                            answer_text=ans_text,
                            score=float(score),
                            is_best=(rank_idx == 0),
                            is_worst=(rank_idx == len(cross_ranked) - 1),
                        )
                        total_answers += 1

                        if embed:
                            vec = self.compute_embedding(ans_text)
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
                                        "chunk_id": chunk_obj.chunk_id,
                                        "question_index": q_index,
                                        "model_name": model_name,
                                    },
                                )

                    svc.add_answer_ranking(
                        run_id=run_id,
                        question_id=db_q.id,
                        best_model=best_model,
                        best_answer=best_answer,
                        worst_model=worst_model,
                        worst_answer=worst_answer,
                        answer_scores=answer_scores,
                    )
                    total_rankings += 1

        logger.info("[ANSWERS/TRACK] Wrote %d Q/A records → %s", written_records, out)
        logger.info("[ANSWERS/TRACK] Inserted %d answers into DB", total_answers)
        logger.info("[ANSWERS/TRACK] Inserted %d ranking rows into DB", total_rankings)

        return out
