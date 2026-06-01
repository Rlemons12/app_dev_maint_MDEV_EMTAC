# modules/services/qanda_embedding_service.py

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Sequence, Tuple

from sqlalchemy.orm import Session

from modules.configuration.log_config import (
    with_request_id,
    debug_id,
    info_id,
    warning_id,
    error_id,
)

from modules.services.ai_models_embedding_service import AIModelsEmbeddingService

from modules.emtacdb.emtacdb_fts import QandA


class QandAEmbeddingService:
    """
    Handles embedding generation and storage for existing Q&A history rows.

    Responsibilities:
        - Load an existing QandA row
        - Generate question and/or answer embeddings
        - Update that same QandA row
        - Optionally update embedding status metadata if those columns exist

    This service should NOT:
        - Create new QandA rows
        - Create feedback/comment rows
        - Own the original answer-generation workflow
        - Generate duplicate Q&A history entries

    Main public method:
        - embed_existing_qanda(...)
    """

    STATUS_PENDING = "pending"
    STATUS_PARTIAL = "partial"
    STATUS_COMPLETE = "complete"
    STATUS_FAILED = "failed"
    STATUS_SKIPPED = "skipped"

    def __init__(self):
        self.embedding_service = AIModelsEmbeddingService()

    # ---------------------------------------------------------
    # Main API
    # ---------------------------------------------------------

    @with_request_id
    def embed_existing_qanda(
        self,
        *,
        session: Session,
        qa_id,
        question: Optional[str],
        answer: Optional[str],
        embed_question: bool = True,
        embed_answer: bool = True,
        request_id: Optional[str] = None,
        skip_existing: bool = True,
        commit: bool = True,
    ) -> bool:
        """
        Generate embeddings for an existing QandA row and update that same row.

        Args:
            session:
                Active SQLAlchemy session.

            qa_id:
                Existing QandA.id value. Can be UUID or UUID string.

            question:
                Question text. If None, this service will use qa_record.question.

            answer:
                Answer text. If None, this service will use qa_record.answer.

            embed_question:
                Whether to generate/update the question embedding.

            embed_answer:
                Whether to generate/update the answer embedding.

            request_id:
                Request-scoped logging id.

            skip_existing:
                If True, do not regenerate an embedding if the row already has one.

            commit:
                If True, commit after updating the row.
                If False, caller is responsible for committing.

        Returns:
            True if embeddings were created/updated successfully.
            False if no row was found, nothing was embedded, or an error occurred.
        """

        qa_record = None

        try:
            normalized_qa_id = self._normalize_qa_id(qa_id)

            qa_record = self._get_qanda_row(
                session=session,
                qa_id=normalized_qa_id,
                request_id=request_id,
            )

            if qa_record is None:
                warning_id(
                    f"[QandAEmbeddingService] No QandA row found for qa_id={qa_id}",
                    request_id,
                )
                return False

            source_question = self._clean_text(
                question if question is not None else getattr(qa_record, "question", None)
            )
            source_answer = self._clean_text(
                answer if answer is not None else getattr(qa_record, "answer", None)
            )

            tasks = self._build_embedding_tasks(
                qa_record=qa_record,
                question=source_question,
                answer=source_answer,
                embed_question=embed_question,
                embed_answer=embed_answer,
                skip_existing=skip_existing,
                request_id=request_id,
            )

            if not tasks:
                self._set_embedding_status(
                    qa_record=qa_record,
                    status=self._resolve_status_from_existing_embeddings(qa_record),
                    error_message=None,
                )

                if commit:
                    session.commit()

                warning_id(
                    f"[QandAEmbeddingService] Nothing to embed for qa_id={qa_id}. "
                    f"embed_question={embed_question} embed_answer={embed_answer} "
                    f"skip_existing={skip_existing}",
                    request_id,
                )
                return False

            labels = [task_label for task_label, _ in tasks]
            texts = [task_text for _, task_text in tasks]

            debug_id(
                f"[QandAEmbeddingService] Generating QandA embeddings "
                f"qa_id={qa_record.id} labels={labels} text_count={len(texts)}",
                request_id,
            )

            vectors = self.embedding_service.get_embeddings_batch(
                texts,
                request_id=request_id,
            )

            vectors = self._validate_vectors(
                vectors=vectors,
                expected_count=len(texts),
                labels=labels,
            )

            model_details = self._get_embedding_model_details(
                request_id=request_id,
            )

            question_embedding = None
            answer_embedding = None

            for label, vector in zip(labels, vectors):
                if label == "question":
                    question_embedding = vector
                elif label == "answer":
                    answer_embedding = vector

            if question_embedding is not None:
                qa_record.question_embedding = question_embedding

            if answer_embedding is not None:
                qa_record.answer_embedding = answer_embedding

            self._set_optional_metadata(
                qa_record=qa_record,
                model_details=model_details,
                vector_dimension=self._resolve_vector_dimension(vectors),
            )

            self._set_embedding_status(
                qa_record=qa_record,
                status=self._resolve_status_from_existing_embeddings(qa_record),
                error_message=None,
            )

            if commit:
                session.commit()

            debug_id(
                f"[QandAEmbeddingService] Staged embeddings for qa_id={qa_record.id} "
                f"question_staged={question_embedding is not None} "
                f"answer_staged={answer_embedding is not None}",
                request_id,
            )

            return True

        except Exception as exc:
            try:
                session.rollback()
            except Exception:
                pass

            error_id(
                f"[QandAEmbeddingService] Failed to embed QandA row qa_id={qa_id}: {exc}",
                request_id,
                exc_info=True,
            )

            self._try_mark_failed(
                session=session,
                qa_record=qa_record,
                qa_id=qa_id,
                error_message=str(exc),
                request_id=request_id,
            )

            return False

    # ---------------------------------------------------------
    # Convenience APIs
    # ---------------------------------------------------------

    @with_request_id
    def embed_answer_only(
        self,
        *,
        session: Session,
        qa_id,
        answer: Optional[str] = None,
        request_id: Optional[str] = None,
        skip_existing: bool = True,
        commit: bool = True,
    ) -> bool:
        """
        Convenience method for updating only the answer embedding.
        Useful after the answer has already been returned to the user.
        """

        return self.embed_existing_qanda(
            session=session,
            qa_id=qa_id,
            question=None,
            answer=answer,
            embed_question=False,
            embed_answer=True,
            request_id=request_id,
            skip_existing=skip_existing,
            commit=commit,
        )

    @with_request_id
    def embed_question_only(
        self,
        *,
        session: Session,
        qa_id,
        question: Optional[str] = None,
        request_id: Optional[str] = None,
        skip_existing: bool = True,
        commit: bool = True,
    ) -> bool:
        """
        Convenience method for updating only the question embedding.
        """

        return self.embed_existing_qanda(
            session=session,
            qa_id=qa_id,
            question=question,
            answer=None,
            embed_question=True,
            embed_answer=False,
            request_id=request_id,
            skip_existing=skip_existing,
            commit=commit,
        )

    @with_request_id
    def embed_missing_qanda(
        self,
        *,
        session: Session,
        qa_id,
        request_id: Optional[str] = None,
        commit: bool = True,
    ) -> bool:
        """
        Convenience method for filling only missing embeddings on an existing row.

        This loads question/answer text from the database row.
        """

        return self.embed_existing_qanda(
            session=session,
            qa_id=qa_id,
            question=None,
            answer=None,
            embed_question=True,
            embed_answer=True,
            request_id=request_id,
            skip_existing=True,
            commit=commit,
        )

    # ---------------------------------------------------------
    # Row Loading
    # ---------------------------------------------------------

    @staticmethod
    def _normalize_qa_id(qa_id):
        """
        Normalizes qa_id for UUID(as_uuid=True) primary keys.

        If conversion fails, the original value is returned so SQLAlchemy can
        still attempt to use it.
        """

        if qa_id is None:
            return None

        if isinstance(qa_id, uuid.UUID):
            return qa_id

        try:
            return uuid.UUID(str(qa_id))
        except Exception:
            return qa_id

    @staticmethod
    def _get_qanda_row(
        *,
        session: Session,
        qa_id,
        request_id: Optional[str] = None,
    ) -> Optional[QandA]:
        """
        Loads one QandA row by primary key.
        """

        if qa_id is None:
            warning_id(
                "[QandAEmbeddingService] Cannot load QandA row because qa_id is None",
                request_id,
            )
            return None

        try:
            return session.get(QandA, qa_id)
        except Exception as exc:
            warning_id(
                f"[QandAEmbeddingService] session.get(QandA, qa_id) failed for "
                f"qa_id={qa_id}: {exc}. Falling back to query().filter_by().first().",
                request_id,
            )

            try:
                return session.query(QandA).filter_by(id=qa_id).first()
            except Exception as fallback_exc:
                error_id(
                    f"[QandAEmbeddingService] Failed to load QandA row qa_id={qa_id}: "
                    f"{fallback_exc}",
                    request_id,
                    exc_info=True,
                )
                return None

    # ---------------------------------------------------------
    # Task Building
    # ---------------------------------------------------------

    def _build_embedding_tasks(
        self,
        *,
        qa_record: QandA,
        question: Optional[str],
        answer: Optional[str],
        embed_question: bool,
        embed_answer: bool,
        skip_existing: bool,
        request_id: Optional[str] = None,
    ) -> List[Tuple[str, str]]:
        """
        Builds the list of embedding tasks.

        Returns:
            [
                ("question", "question text"),
                ("answer", "answer text"),
            ]
        """

        tasks: List[Tuple[str, str]] = []

        if embed_question:
            if skip_existing and self._has_vector(getattr(qa_record, "question_embedding", None)):
                debug_id(
                    f"[QandAEmbeddingService] Skipping question embedding for "
                    f"qa_id={qa_record.id}; embedding already exists.",
                    request_id,
                )
            elif question:
                tasks.append(("question", question))
            else:
                warning_id(
                    f"[QandAEmbeddingService] Question embedding requested but no "
                    f"question text exists for qa_id={qa_record.id}",
                    request_id,
                )

        if embed_answer:
            if skip_existing and self._has_vector(getattr(qa_record, "answer_embedding", None)):
                debug_id(
                    f"[QandAEmbeddingService] Skipping answer embedding for "
                    f"qa_id={qa_record.id}; embedding already exists.",
                    request_id,
                )
            elif answer:
                tasks.append(("answer", answer))
            else:
                warning_id(
                    f"[QandAEmbeddingService] Answer embedding requested but no "
                    f"answer text exists for qa_id={qa_record.id}",
                    request_id,
                )

        return tasks

    # ---------------------------------------------------------
    # Text / Vector Helpers
    # ---------------------------------------------------------

    @staticmethod
    def _clean_text(value: Optional[str]) -> Optional[str]:
        """
        Normalizes text input.
        """

        if not isinstance(value, str):
            return None

        value = value.strip()

        return value or None

    @staticmethod
    def _has_vector(value: Any) -> bool:
        """
        Returns True if an embedding column appears to contain a usable vector.
        """

        if value is None:
            return False

        try:
            return len(value) > 0
        except Exception:
            return True

    @staticmethod
    def _validate_vectors(
        *,
        vectors: Any,
        expected_count: int,
        labels: Sequence[str],
    ) -> List[List[float]]:
        """
        Validates and normalizes returned vectors.

        AIModelsEmbeddingService should already return List[List[float]], but this
        gives the QandA update layer one final safety check.
        """

        if vectors is None:
            raise RuntimeError("Embedding service returned None")

        try:
            vectors_list = list(vectors)
        except Exception as exc:
            raise RuntimeError(
                f"Embedding service returned a non-iterable result: {exc}"
            ) from exc

        if len(vectors_list) != expected_count:
            raise RuntimeError(
                f"Embedding count mismatch. expected={expected_count} "
                f"got={len(vectors_list)} labels={list(labels)}"
            )

        normalized_vectors: List[List[float]] = []

        for vector_index, vector in enumerate(vectors_list):
            if vector is None:
                raise RuntimeError(
                    f"Embedding vector for label={labels[vector_index]} is None"
                )

            if hasattr(vector, "detach"):
                vector = vector.detach()

            if hasattr(vector, "cpu"):
                vector = vector.cpu()

            if hasattr(vector, "tolist"):
                vector = vector.tolist()

            if isinstance(vector, tuple):
                vector = list(vector)

            if not isinstance(vector, list):
                raise RuntimeError(
                    f"Embedding vector for label={labels[vector_index]} must be list-like. "
                    f"Got {type(vector).__name__}"
                )

            if not vector:
                raise RuntimeError(
                    f"Embedding vector for label={labels[vector_index]} is empty"
                )

            clean_vector: List[float] = []

            for value_index, value in enumerate(vector):
                try:
                    clean_vector.append(float(value))
                except (TypeError, ValueError) as exc:
                    raise RuntimeError(
                        f"Embedding vector value could not be converted to float. "
                        f"label={labels[vector_index]} index={value_index} value={value!r}"
                    ) from exc

            normalized_vectors.append(clean_vector)

        return normalized_vectors

    @staticmethod
    def _resolve_vector_dimension(vectors: Sequence[Sequence[float]]) -> Optional[int]:
        """
        Returns the dimension of the first vector, if available.
        """

        if not vectors:
            return None

        first = vectors[0]

        try:
            return len(first)
        except Exception:
            return None

    # ---------------------------------------------------------
    # Optional Metadata
    # ---------------------------------------------------------

    def _get_embedding_model_details(
        self,
        *,
        request_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Safely resolves embedding model metadata.

        Works with both the older AIModelsEmbeddingService and the updated one
        that includes get_current_model_details().
        """

        details: Dict[str, Any] = {
            "model_name": None,
            "backend": None,
            "expected_dimension": None,
        }

        try:
            if hasattr(self.embedding_service, "get_current_model_details"):
                model_details = self.embedding_service.get_current_model_details(
                    request_id=request_id,
                )

                if isinstance(model_details, dict):
                    details["model_name"] = model_details.get("model_name")
                    details["backend"] = model_details.get("backend")
                    details["expected_dimension"] = model_details.get("expected_dimension")

                    return details

            if hasattr(self.embedding_service, "get_current_model_name"):
                details["model_name"] = self.embedding_service.get_current_model_name(
                    request_id=request_id,
                )

            return details

        except Exception as exc:
            warning_id(
                f"[QandAEmbeddingService] Could not resolve embedding model details: {exc}",
                request_id,
            )
            return details

    def _set_optional_metadata(
        self,
        *,
        qa_record: QandA,
        model_details: Dict[str, Any],
        vector_dimension: Optional[int],
    ) -> None:
        """
        Sets optional metadata columns if they exist on the QandA model.

        This lets you add these columns later without breaking this service now:

            embedding_model
            embedding_backend
            embedding_dimension
            embedded_at
            embedding_error
        """

        model_name = model_details.get("model_name")
        backend = model_details.get("backend")
        expected_dimension = model_details.get("expected_dimension")

        dimension = expected_dimension or vector_dimension

        self._safe_set_attr(qa_record, "embedding_model", model_name)
        self._safe_set_attr(qa_record, "embedding_backend", backend)
        self._safe_set_attr(qa_record, "embedding_dimension", dimension)
        self._safe_set_attr(qa_record, "embedded_at", datetime.now(timezone.utc))
        self._safe_set_attr(qa_record, "embedding_error", None)

    def _set_embedding_status(
        self,
        *,
        qa_record: QandA,
        status: str,
        error_message: Optional[str],
    ) -> None:
        """
        Sets embedding status fields if they exist.
        """

        self._safe_set_attr(qa_record, "embedding_status", status)
        self._safe_set_attr(qa_record, "embedding_error", error_message)

        if status in {self.STATUS_COMPLETE, self.STATUS_PARTIAL}:
            self._safe_set_attr(qa_record, "embedded_at", datetime.now(timezone.utc))

    @staticmethod
    def _safe_set_attr(obj: Any, attr_name: str, value: Any) -> None:
        """
        Sets an attribute only if it exists on the mapped object.

        This keeps the service compatible with the current QandA model even if
        optional metadata columns have not been added yet.
        """

        if hasattr(obj, attr_name):
            setattr(obj, attr_name, value)

    def _resolve_status_from_existing_embeddings(self, qa_record: QandA) -> str:
        """
        Determines status from current row embedding columns.
        """

        has_question = self._has_vector(getattr(qa_record, "question_embedding", None))
        has_answer = self._has_vector(getattr(qa_record, "answer_embedding", None))

        if has_question and has_answer:
            return self.STATUS_COMPLETE

        if has_question or has_answer:
            return self.STATUS_PARTIAL

        return self.STATUS_SKIPPED

    # ---------------------------------------------------------
    # Failure Handling
    # ---------------------------------------------------------

    def _try_mark_failed(
        self,
        *,
        session: Session,
        qa_record: Optional[QandA],
        qa_id,
        error_message: str,
        request_id: Optional[str] = None,
    ) -> None:
        """
        Best-effort update to mark a QandA row as failed if optional status
        columns exist.

        This should never raise back to the caller.
        """

        try:
            if qa_record is None:
                normalized_qa_id = self._normalize_qa_id(qa_id)
                qa_record = self._get_qanda_row(
                    session=session,
                    qa_id=normalized_qa_id,
                    request_id=request_id,
                )

            if qa_record is None:
                return

            self._set_embedding_status(
                qa_record=qa_record,
                status=self.STATUS_FAILED,
                error_message=error_message[:2000] if error_message else None,
            )

            session.commit()

        except Exception as exc:
            try:
                session.rollback()
            except Exception:
                pass

            warning_id(
                f"[QandAEmbeddingService] Could not mark embedding failure for "
                f"qa_id={qa_id}: {exc}",
                request_id,
            )