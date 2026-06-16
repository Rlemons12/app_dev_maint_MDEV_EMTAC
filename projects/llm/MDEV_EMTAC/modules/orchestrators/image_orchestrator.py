# modules/orchestrators/image_orchestrator.py

from __future__ import annotations

import os
import uuid
import inspect
from pathlib import Path
from typing import Optional, Dict, Any, List

from werkzeug.utils import secure_filename

from modules.configuration.config import DATABASE_DIR, DATABASE_PATH_IMAGES_FOLDER
from modules.configuration.log_config import (
    with_request_id,
    get_request_id,
    debug_id,
    info_id,
    warning_id,
    error_id,
)

from modules.emtacdb.emtacdb_fts import Image as DBImage

from modules.orchestrators.base_orchestrator import BaseOrchestrator

from modules.services.ai_model_image_service import AIModelImageService
from modules.services.image_service import ImageService
from modules.services.image_embedding_service import ImageEmbeddingService
from modules.services.image_position_service import ImagePositionService
from modules.services.image_completed_document_association_service import (
    ImageCompletedDocumentAssociationService,
)
from modules.services.image_task_association_service import (
    ImageTaskAssociationService,
)
from modules.services.image_problem_association_service import (
    ImageProblemAssociationService,
)


class ImageOrchestrator(BaseOrchestrator):
    """
    Image domain orchestrator.

    Responsibilities:
    - Own transactions
    - Coordinate image + embedding + associations
    - Return structured domain results
    - Never leak ORM dependency across boundary responses when avoidable

    Important workflow distinction:

    process_upload():
        - Saves images permanently
        - Creates Image rows
        - Generates and stores embeddings

    compare_uploaded_image():
        - Saves a temporary query image
        - Generates a query embedding
        - Searches existing stored image embeddings
        - Does NOT create a new Image row
        - Does NOT store the temporary query embedding
    """

    COMPARE_IMAGE_EXTENSIONS = {
        ".jpg",
        ".jpeg",
        ".png",
        ".gif",
        ".bmp",
        ".tif",
        ".tiff",
        ".webp",
    }

    def __init__(self):
        super().__init__()

        self.image_service = ImageService()
        self.embedding_service = ImageEmbeddingService()
        self.position_service = ImagePositionService()
        self.document_assoc_service = ImageCompletedDocumentAssociationService()
        self.task_assoc_service = ImageTaskAssociationService()
        self.problem_assoc_service = ImageProblemAssociationService()
        self.image_model_service = AIModelImageService()

    # ---------------------------------------------------------
    # INTERNAL HELPERS
    # ---------------------------------------------------------

    @staticmethod
    def _normalize_text(value: Optional[str], *, fallback: str = "") -> str:
        """
        Normalize text inputs and apply fallback when the incoming value is:
        - None
        - empty string
        - whitespace-only string

        This is important for upload metadata because UI forms often submit
        empty strings instead of None.
        """
        if value is None:
            return fallback

        if not isinstance(value, str):
            value = str(value)

        normalized = value.strip()
        return normalized if normalized else fallback

    def _serialize_image(self, image_obj: Any) -> Optional[Dict[str, Any]]:
        """
        Safely convert an image ORM instance into a plain dict.

        This prevents callers from depending on a live SQLAlchemy session
        after the transaction scope exits.
        """
        if image_obj is None:
            return None

        return self.image_service.serialize(image_obj)

    @staticmethod
    def _serialize_embedding(embedding_obj: Any) -> Optional[Dict[str, Any]]:
        """
        Best-effort serialization for embedding rows.
        """
        if embedding_obj is None:
            return None

        return {
            "id": getattr(embedding_obj, "id", None),
            "image_id": getattr(embedding_obj, "image_id", None),
            "model_name": getattr(embedding_obj, "model_name", None),
        }

    @staticmethod
    def _serialize_association(assoc_obj: Any) -> Optional[Dict[str, Any]]:
        """
        Best-effort generic serializer for association rows.
        """
        if assoc_obj is None:
            return None

        data: Dict[str, Any] = {}

        for attr in (
            "id",
            "image_id",
            "position_id",
            "task_id",
            "problem_id",
            "complete_document_id",
            "chunk_id",
            "confidence_score",
            "association_method",
        ):
            if hasattr(assoc_obj, attr):
                data[attr] = getattr(assoc_obj, attr, None)

        return data

    @staticmethod
    def _serialize_collection(items: Optional[List[Any]]) -> List[Dict[str, Any]]:
        serialized: List[Dict[str, Any]] = []

        for item in items or []:
            row: Dict[str, Any] = {}

            for attr in (
                "id",
                "image_id",
                "position_id",
                "task_id",
                "problem_id",
                "complete_document_id",
                "chunk_id",
                "model_name",
                "confidence_score",
                "association_method",
                "title",
                "name",
                "description",
                "file_path",
            ):
                if hasattr(item, attr):
                    row[attr] = getattr(item, attr, None)

            if row:
                serialized.append(row)
            else:
                serialized.append({"repr": repr(item)})

        return serialized

    def _build_upload_result(
        self,
        *,
        success: bool,
        file_name: str,
        file_path: Optional[str],
        image_id: Optional[int] = None,
        result: Optional[Dict[str, Any]] = None,
        error: Optional[str] = None,
        detail: Optional[str] = None,
        http_status: int = 200,
    ) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "success": success,
            "file_name": file_name,
            "file_path": file_path,
            "image_id": image_id,
            "result": result,
            "http_status": http_status,
        }

        if error:
            payload["error"] = error

        if detail:
            payload["detail"] = detail

        return payload

    # ---------------------------------------------------------
    # CREATE IMAGE
    # ---------------------------------------------------------

    @with_request_id
    def create_image(
        self,
        *,
        title: str,
        description: str,
        file_path: str,
        img_metadata: Optional[Dict[str, Any]] = None,
        request_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        rid = request_id or get_request_id()

        with self.transaction() as session:
            image = self.image_service.create(
                session=session,
                title=title,
                description=description,
                file_path=file_path,
                img_metadata=img_metadata,
                request_id=rid,
            )

            image_payload = self._serialize_image(image)

            debug_id(
                f"[ImageOrchestrator] Created image id={image_payload.get('id') if image_payload else None}",
                rid,
            )

            return {
                "status": "created",
                "image_id": image_payload.get("id") if image_payload else None,
                "image": image_payload,
            }

    # ---------------------------------------------------------
    # CREATE IMAGE + EMBEDDING
    # ---------------------------------------------------------

    @with_request_id
    def create_image_with_embedding(
        self,
        *,
        title: str,
        description: str,
        file_path: str,
        model_name: str,
        embedding: List[float],
        img_metadata: Optional[Dict[str, Any]] = None,
        request_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        rid = request_id or get_request_id()

        with self.transaction() as session:
            image = self.image_service.create(
                session=session,
                title=title,
                description=description,
                file_path=file_path,
                img_metadata=img_metadata,
                request_id=rid,
            )

            embedding_obj = self.embedding_service.add_pgvector(
                session=session,
                image_id=image.id,
                model_name=model_name,
                embedding=embedding,
                request_id=rid,
            )

            image_payload = self._serialize_image(image)
            embedding_payload = self._serialize_embedding(embedding_obj)

            return {
                "status": "created",
                "image_id": image_payload.get("id") if image_payload else None,
                "image": image_payload,
                "embedding": embedding_payload,
            }

    # ---------------------------------------------------------
    # ATTACHMENTS
    # ---------------------------------------------------------

    @with_request_id
    def attach_to_position(
        self,
        *,
        image_id: int,
        position_id: int,
        request_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        rid = request_id or get_request_id()

        with self.transaction() as session:
            assoc = self.position_service.associate(
                session=session,
                image_id=image_id,
                position_id=position_id,
            )

            assoc_payload = self._serialize_association(assoc)

            info_id(
                f"[ImageOrchestrator] Attached image_id={image_id} to position_id={position_id}",
                rid,
            )

            return {
                "status": "attached",
                "image_id": image_id,
                "position_id": position_id,
                "association": assoc_payload,
            }

    @with_request_id
    def attach_to_task(
        self,
        *,
        image_id: int,
        task_id: int,
        request_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        rid = request_id or get_request_id()

        with self.transaction() as session:
            assoc = self.task_assoc_service.associate(
                session=session,
                image_id=image_id,
                task_id=task_id,
            )

            assoc_payload = self._serialize_association(assoc)

            info_id(
                f"[ImageOrchestrator] Attached image_id={image_id} to task_id={task_id}",
                rid,
            )

            return {
                "status": "attached",
                "image_id": image_id,
                "task_id": task_id,
                "association": assoc_payload,
            }

    @with_request_id
    def attach_to_problem(
        self,
        *,
        image_id: int,
        problem_id: int,
        request_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        rid = request_id or get_request_id()

        with self.transaction() as session:
            assoc = self.problem_assoc_service.associate(
                session=session,
                image_id=image_id,
                problem_id=problem_id,
            )

            assoc_payload = self._serialize_association(assoc)

            info_id(
                f"[ImageOrchestrator] Attached image_id={image_id} to problem_id={problem_id}",
                rid,
            )

            return {
                "status": "attached",
                "image_id": image_id,
                "problem_id": problem_id,
                "association": assoc_payload,
            }

    @with_request_id
    def attach_to_complete_document(
        self,
        *,
        image_id: int,
        complete_document_id: int,
        chunk_id: Optional[int] = None,
        confidence_score: float = 0.7,
        association_method: str = "manual",
        request_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        rid = request_id or get_request_id()

        with self.transaction() as session:
            assoc = self.document_assoc_service.create_association(
                session=session,
                image_id=image_id,
                complete_document_id=complete_document_id,
                chunk_id=chunk_id,
                confidence_score=confidence_score,
                association_method=association_method,
            )

            assoc_payload = self._serialize_association(assoc)

            info_id(
                "[ImageOrchestrator] Attached image_id="
                f"{image_id} to complete_document_id={complete_document_id}",
                rid,
            )

            return {
                "status": "attached",
                "image_id": image_id,
                "complete_document_id": complete_document_id,
                "association": assoc_payload,
            }

    # ---------------------------------------------------------
    # DELETE IMAGE (SAFE CASCADE)
    # ---------------------------------------------------------

    @with_request_id
    def delete_image(
        self,
        *,
        image_id: int,
        request_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        rid = request_id or get_request_id()

        with self.transaction() as session:
            self.embedding_service.remove_all_for_image(
                session=session,
                image_id=image_id,
            )

            self.position_service.remove_all_for_image(
                session=session,
                image_id=image_id,
            )

            self.task_assoc_service.remove_all_for_image(
                session=session,
                image_id=image_id,
            )

            self.problem_assoc_service.remove_all_for_image(
                session=session,
                image_id=image_id,
            )

            self.document_assoc_service.remove_all_for_image(
                session=session,
                image_id=image_id,
            )

            removed = self.image_service.remove(
                session=session,
                image_id=image_id,
                request_id=rid,
            )

            if not removed:
                warning_id(
                    f"[ImageOrchestrator] Image not found for delete id={image_id}",
                    rid,
                )
                return {
                    "status": "not_found",
                    "image_id": image_id,
                }

            info_id(f"[ImageOrchestrator] Deleted image id={image_id}", rid)

            return {
                "status": "deleted",
                "image_id": image_id,
            }

    # ---------------------------------------------------------
    # RESOLVE FULL IMAGE GRAPH
    # ---------------------------------------------------------

    @with_request_id
    def resolve_image_graph(
        self,
        *,
        image_id: int,
        request_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        rid = request_id or get_request_id()

        with self.transaction(read_only=True) as session:
            image = self.image_service.get(
                session=session,
                image_id=image_id,
                request_id=rid,
            )

            if not image:
                return {
                    "status": "not_found",
                    "image_id": image_id,
                }

            positions = self.position_service.get_positions(
                session=session,
                image_id=image_id,
            )

            tasks = self.task_assoc_service.get_tasks_for_image(
                session=session,
                image_id=image_id,
            )

            problems = self.problem_assoc_service.get_problems_for_image(
                session=session,
                image_id=image_id,
            )

            docs = self.document_assoc_service.resolve_related_entities(
                session=session,
                image_id=image_id,
            )

            embeddings = self.embedding_service.get_all_for_image(
                session=session,
                image_id=image_id,
            )

            return {
                "status": "resolved",
                "image_id": image_id,
                "image": self._serialize_image(image),
                "positions": self._serialize_collection(positions),
                "tasks": self._serialize_collection(tasks),
                "problems": self._serialize_collection(problems),
                "documents": self._serialize_collection(docs),
                "embeddings": self._serialize_collection(embeddings),
            }

    # ---------------------------------------------------------
    # PROCESS IMAGE UPLOAD
    # ---------------------------------------------------------

    @with_request_id
    def process_upload(
        self,
        *,
        files,
        metadata: Dict[str, Any],
        request_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        rid = request_id or get_request_id()
        metadata = metadata or {}

        results: List[Dict[str, Any]] = []
        pending_embeddings: List[Dict[str, Any]] = []

        if not files:
            return {
                "status": "validation_error",
                "message": "No files provided",
                "processed": 0,
                "failed": 0,
                "results": [],
            }

        os.makedirs(DATABASE_PATH_IMAGES_FOLDER, exist_ok=True)

        # ---------------------------------------------------------
        # PHASE 1: SAVE FILES + CREATE IMAGE ROWS
        # ---------------------------------------------------------
        for file_obj in files:
            filename = secure_filename(getattr(file_obj, "filename", "") or "")

            if not filename:
                warning_id(
                    "[ImageOrchestrator] Skipping upload with empty filename",
                    rid,
                )
                results.append(
                    self._build_upload_result(
                        success=False,
                        file_name="",
                        file_path=None,
                        error="Invalid filename",
                        detail="Uploaded file has no valid filename",
                        http_status=400,
                    )
                )
                continue

            absolute_file_path = os.path.join(DATABASE_PATH_IMAGES_FOLDER, filename)
            db_relative_path: Optional[str] = None

            try:
                title = self._normalize_text(
                    metadata.get("title"),
                    fallback=Path(filename).stem,
                )
                description = self._normalize_text(
                    metadata.get("description"),
                    fallback=title,
                )

                img_metadata = metadata.get("img_metadata")
                if img_metadata is None:
                    img_metadata = {}

                file_obj.save(absolute_file_path)

                info_id(
                    f"[ImageOrchestrator] Saved image file to {absolute_file_path}",
                    rid,
                )

                db_relative_path = os.path.normpath(
                    os.path.relpath(absolute_file_path, DATABASE_DIR)
                )

                debug_id(
                    f"[ImageOrchestrator] Saving standalone image path | "
                    f"absolute='{absolute_file_path}' | relative='{db_relative_path}'",
                    rid,
                )

                created = self.create_image(
                    title=title,
                    description=description,
                    file_path=db_relative_path,
                    img_metadata=img_metadata,
                    request_id=rid,
                )

                image_payload = created.get("image") or {}
                image_id = created.get("image_id") or image_payload.get("id")

                result_payload = dict(created)

                results.append(
                    self._build_upload_result(
                        success=True,
                        file_name=filename,
                        file_path=db_relative_path,
                        image_id=image_id,
                        result=result_payload,
                        http_status=200,
                    )
                )

                if image_id:
                    pending_embeddings.append(
                        {
                            "image_id": image_id,
                            "file_name": filename,
                            "absolute_file_path": absolute_file_path,
                            "db_relative_path": db_relative_path,
                        }
                    )

                debug_id(
                    f"[ImageOrchestrator] Upload staged successfully | "
                    f"file={filename} | image_id={image_id}",
                    rid,
                )

            except ValueError as exc:
                warning_id(
                    f"[ImageOrchestrator] Validation error while processing file "
                    f"'{filename}': {exc}",
                    rid,
                )
                results.append(
                    self._build_upload_result(
                        success=False,
                        file_name=filename,
                        file_path=db_relative_path or absolute_file_path,
                        error="Validation error",
                        detail=str(exc),
                        http_status=400,
                    )
                )

            except Exception as exc:
                error_id(
                    f"[ImageOrchestrator] Image processing error for file "
                    f"'{filename}': {exc}",
                    rid,
                    exc_info=True,
                )
                results.append(
                    self._build_upload_result(
                        success=False,
                        file_name=filename,
                        file_path=db_relative_path or absolute_file_path,
                        error="Image processing failed",
                        detail=str(exc),
                        http_status=500,
                    )
                )

        # ---------------------------------------------------------
        # PHASE 2: EMBED ALL STAGED IMAGES IN BATCH
        # ---------------------------------------------------------
        if pending_embeddings:
            current_model_name: Optional[str] = None
            handler = None
            batch_key = self._normalize_text(
                metadata.get("_embedding_batch_key"),
                fallback=f"image_upload:{rid}",
            )

            try:
                current_model_name = self.image_model_service.get_current_model_name(
                    request_id=rid,
                )

                handler = self._resolve_local_image_handler(
                    model_name=current_model_name,
                    request_id=rid,
                )

                image_paths = [row["absolute_file_path"] for row in pending_embeddings]

                batch_vectors: Optional[List[Optional[List[float]]]] = None

                if handler is not None:
                    self._begin_embedding_batch_session(
                        handler=handler,
                        batch_key=batch_key,
                        request_id=rid,
                    )

                    try:
                        batch_vectors = self._get_batch_image_embeddings(
                            handler=handler,
                            image_paths=image_paths,
                            batch_key=batch_key,
                            request_id=rid,
                        )
                    finally:
                        self._end_embedding_batch_session(
                            handler=handler,
                            batch_key=batch_key,
                            request_id=rid,
                        )

                if batch_vectors is None:
                    batch_vectors = self._get_single_image_embeddings_fallback(
                        image_paths=image_paths,
                        request_id=rid,
                    )

                stored_embeddings: Dict[int, Dict[str, Any]] = {}

                with self.transaction() as session:
                    for pending_row, embedding_vector in zip(
                        pending_embeddings,
                        batch_vectors,
                    ):
                        image_id = pending_row["image_id"]

                        if embedding_vector is None:
                            warning_id(
                                f"[ImageOrchestrator] No embedding generated for "
                                f"image_id={image_id} file={pending_row['file_name']}",
                                rid,
                            )
                            continue

                        try:
                            embedding_obj = self.embedding_service.add_pgvector(
                                session=session,
                                image_id=image_id,
                                model_name=current_model_name,
                                embedding=embedding_vector,
                                request_id=rid,
                            )
                            stored_embeddings[image_id] = self._serialize_embedding(
                                embedding_obj
                            )
                        except Exception as exc:
                            warning_id(
                                f"[ImageOrchestrator] Failed storing embedding for "
                                f"image_id={image_id}: {exc}",
                                rid,
                            )

                # Patch embedding payloads back into existing results.
                for result_row in results:
                    if not result_row.get("success"):
                        continue

                    image_id = result_row.get("image_id")
                    if not image_id:
                        continue

                    embedding_payload = stored_embeddings.get(image_id)
                    if embedding_payload is not None:
                        result_payload = result_row.get("result") or {}
                        result_payload["embedding"] = embedding_payload
                        result_row["result"] = result_payload

                debug_id(
                    f"[ImageOrchestrator] Batch embedding phase complete | "
                    f"staged={len(pending_embeddings)} "
                    f"stored={len(stored_embeddings)} "
                    f"model={current_model_name}",
                    rid,
                )

            except Exception as embed_exc:
                warning_id(
                    f"[ImageOrchestrator] Batch embedding phase failed after "
                    f"image creation: {embed_exc}",
                    rid,
                )

        # ---------------------------------------------------------
        # FINAL STATUS
        # ---------------------------------------------------------
        failed = [r for r in results if not r.get("success", False)]
        processed = len(results) - len(failed)

        if not results:
            status = "validation_error"
            message = "No valid image files were provided"
        elif failed and processed:
            status = "partial_success"
            message = "Some image files failed to process"
        elif failed and not processed:
            status = "processing_error"
            message = "All image files failed to process"
        else:
            status = "success"
            message = "All image files processed successfully"

        return {
            "status": status,
            "message": message,
            "processed": processed,
            "failed": len(failed),
            "results": results,
        }

    @with_request_id
    def process_upload_concurrent(
        self,
        *,
        files,
        metadata: Dict[str, Any],
        max_workers: int = 4,
        request_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Safe first pass: keep sequential until explicit concurrency is added.

        The image/domain pipeline touches filesystem + DB + model-dependent services,
        so sequential processing remains the safest default.
        """
        debug_id(
            f"[ImageOrchestrator] process_upload_concurrent delegating to sequential "
            f"path | max_workers={max_workers}",
            request_id or get_request_id(),
        )

        return self.process_upload(
            files=files,
            metadata=metadata,
            request_id=request_id,
        )

    # ---------------------------------------------------------
    # COMPARE UPLOADED IMAGE AGAINST EXISTING IMAGE EMBEDDINGS
    # ---------------------------------------------------------

    @with_request_id
    def compare_uploaded_image(
        self,
        *,
        file_obj: Any,
        similarity_threshold: float = 0.3,
        limit: int = 10,
        cleanup_query_file: bool = True,
        request_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Compare a temporary uploaded query image against stored image embeddings.

        This does not persist the query image as a new Image row.

        Returns:
            {
                "status": "success" | "validation_error" | "processing_error",
                "message": str,
                "error": optional str,
                "image_similarity_search": [...],
                "http_status": int
            }

        Existing frontend expects:
            {
                "image_similarity_search": [...]
            }
        """
        rid = request_id or get_request_id()

        if file_obj is None:
            warning_id(
                "[ImageOrchestrator] Compare request missing file object",
                rid,
            )
            return {
                "status": "validation_error",
                "error": "No file part in the request",
                "image_similarity_search": [],
                "http_status": 400,
            }

        original_filename = getattr(file_obj, "filename", "") or ""
        safe_filename = secure_filename(original_filename)

        if not safe_filename:
            warning_id(
                "[ImageOrchestrator] Compare upload missing filename",
                rid,
            )
            return {
                "status": "validation_error",
                "error": "No selected file",
                "image_similarity_search": [],
                "http_status": 400,
            }

        file_ext = Path(safe_filename).suffix.lower()

        if file_ext not in self.COMPARE_IMAGE_EXTENSIONS:
            warning_id(
                f"[ImageOrchestrator] Compare upload rejected unsupported file type | "
                f"file='{safe_filename}' | ext='{file_ext}'",
                rid,
            )
            return {
                "status": "validation_error",
                "error": f"File type not allowed by image compare route: {file_ext.lstrip('.')}",
                "allowed_file_types": sorted(
                    ext.lstrip(".") for ext in self.COMPARE_IMAGE_EXTENSIONS
                ),
                "image_similarity_search": [],
                "http_status": 400,
            }

        query_file_path: Optional[str] = None

        try:
            query_file_path = self._save_compare_query_file(
                file_obj=file_obj,
                safe_filename=safe_filename,
                request_id=rid,
            )

            current_model_name = self.image_model_service.get_current_model_name(
                request_id=rid,
            )

            info_id(
                f"[ImageOrchestrator] Starting image compare | "
                f"file='{safe_filename}' | model='{current_model_name}' | "
                f"threshold={similarity_threshold} | limit={limit}",
                rid,
            )

            query_embedding = self.image_model_service.get_image_embedding(
                image_path=query_file_path,
                request_id=rid,
            )

            embedding_list = self._normalize_embedding_vector(query_embedding)

            if not embedding_list:
                warning_id(
                    f"[ImageOrchestrator] No query embedding generated for "
                    f"'{safe_filename}'",
                    rid,
                )
                return {
                    "status": "processing_error",
                    "error": "Failed to process the uploaded image.",
                    "image_similarity_search": [],
                    "http_status": 500,
                }

            safe_limit = max(1, min(int(limit or 10), 100))
            safe_threshold = float(similarity_threshold)

            with self.transaction(read_only=True) as session:
                search_results = DBImage.search_images(
                    session=session,
                    similarity_query_embedding=embedding_list,
                    similarity_threshold=safe_threshold,
                    embedding_model_name=current_model_name,
                    use_hybrid_ranking=True,
                    limit=safe_limit,
                )

                formatted_results = [
                    self._format_image_compare_result(row)
                    for row in search_results or []
                ]

            info_id(
                f"[ImageOrchestrator] Image compare completed | "
                f"file='{safe_filename}' | matches={len(formatted_results)}",
                rid,
            )

            return {
                "status": "success",
                "message": "Image comparison completed successfully.",
                "image_similarity_search": formatted_results,
                "http_status": 200,
            }

        except ValueError as exc:
            warning_id(
                f"[ImageOrchestrator] Image compare validation error for "
                f"'{safe_filename}': {exc}",
                rid,
            )
            return {
                "status": "validation_error",
                "error": str(exc),
                "image_similarity_search": [],
                "http_status": 400,
            }

        except Exception as exc:
            error_id(
                f"[ImageOrchestrator] Image compare failed for "
                f"'{safe_filename}': {exc}",
                rid,
                exc_info=True,
            )
            return {
                "status": "processing_error",
                "error": "An error occurred during the comparison process.",
                "detail": str(exc),
                "image_similarity_search": [],
                "http_status": 500,
            }

        finally:
            if cleanup_query_file and query_file_path:
                try:
                    if os.path.exists(query_file_path):
                        os.remove(query_file_path)
                        debug_id(
                            f"[ImageOrchestrator] Removed temporary compare file: "
                            f"{query_file_path}",
                            rid,
                        )
                except Exception as cleanup_exc:
                    warning_id(
                        f"[ImageOrchestrator] Failed to remove temporary compare file "
                        f"'{query_file_path}': {cleanup_exc}",
                        rid,
                    )

    def _save_compare_query_file(
        self,
        *,
        file_obj: Any,
        safe_filename: str,
        request_id: Optional[str] = None,
    ) -> str:
        """
        Save the uploaded compare image to a temporary compare folder.

        The query image is used only for embedding generation.
        It is not inserted into the Image table.
        """
        rid = request_id or get_request_id()

        compare_upload_dir = os.path.join(
            DATABASE_PATH_IMAGES_FOLDER,
            "_compare_uploads",
        )

        os.makedirs(compare_upload_dir, exist_ok=True)

        filename_path = Path(safe_filename)
        unique_filename = (
            f"{filename_path.stem}_{uuid.uuid4().hex}"
            f"{filename_path.suffix.lower()}"
        )

        destination_path = os.path.join(compare_upload_dir, unique_filename)

        file_obj.save(destination_path)

        info_id(
            f"[ImageOrchestrator] Saved temporary compare image | "
            f"original='{safe_filename}' | path='{destination_path}'",
            rid,
        )

        return destination_path

    @staticmethod
    def _normalize_embedding_vector(embedding: Any) -> List[float]:
        """
        Normalize embedding output into a flat list[float].

        Handles:
        - numpy arrays
        - tensors with .tolist()
        - nested lists
        - tuples
        """
        if embedding is None:
            return []

        if hasattr(embedding, "tolist"):
            embedding = embedding.tolist()

        def flatten(value: Any):
            if isinstance(value, (list, tuple)):
                for item in value:
                    yield from flatten(item)
            else:
                yield value

        try:
            return [float(x) for x in flatten(embedding)]
        except Exception as exc:
            raise ValueError(f"Invalid image embedding format: {exc}") from exc

    @staticmethod
    def _format_image_compare_result(result: Any) -> Dict[str, Any]:
        """
        Format one DBImage.search_images result for the existing frontend.

        Existing JS expects:
            id
            title
            description
            file_path
            similarity
        """
        if result is None:
            return {
                "id": None,
                "title": "",
                "description": "",
                "file_path": "",
                "similarity": 0.0,
            }

        if isinstance(result, dict):
            search_metadata = result.get("search_metadata") or {}
            similarity_score = search_metadata.get("similarity_score")

            if similarity_score is None:
                similarity_score = result.get("similarity")

            file_path = result.get("file_path") or ""

            return {
                "id": result.get("id"),
                "title": result.get("title") or "",
                "description": result.get("description") or "",
                "file_path": os.path.basename(file_path),
                "similarity": float(similarity_score) if similarity_score is not None else 0.0,
            }

        search_metadata = getattr(result, "search_metadata", None) or {}
        similarity_score = None

        if isinstance(search_metadata, dict):
            similarity_score = search_metadata.get("similarity_score")

        if similarity_score is None:
            similarity_score = getattr(result, "similarity", None)

        file_path = getattr(result, "file_path", "") or ""

        return {
            "id": getattr(result, "id", None),
            "title": getattr(result, "title", "") or "",
            "description": getattr(result, "description", "") or "",
            "file_path": os.path.basename(file_path),
            "similarity": float(similarity_score) if similarity_score is not None else 0.0,
        }

    # ---------------------------------------------------------
    # EMBEDDING HELPER METHODS
    # ---------------------------------------------------------

    @staticmethod
    def _call_with_supported_kwargs(func, **kwargs):
        """
        Call a function while only passing kwargs it actually accepts.
        """
        sig = inspect.signature(func)
        accepted = {}

        for key, value in kwargs.items():
            if key in sig.parameters:
                accepted[key] = value

        return func(**accepted)

    def _resolve_local_image_handler(
        self,
        *,
        model_name: str,
        request_id: Optional[str] = None,
    ):
        """
        Best-effort handler resolution for batch embedding support.

        We keep this defensive because AIModelImageService may evolve and
        may expose different loader methods over time.
        """
        rid = request_id or get_request_id()
        svc = self.image_model_service

        candidates = [
            "get_model_handler",
            "resolve_model_handler",
            "load_local_model",
            "_load_local_model",
        ]

        for method_name in candidates:
            method = getattr(svc, method_name, None)
            if not callable(method):
                continue

            try:
                handler = self._call_with_supported_kwargs(
                    method,
                    model_name=model_name,
                    request_id=rid,
                )
                if handler is not None:
                    debug_id(
                        f"[ImageOrchestrator] Resolved image handler via "
                        f"{method_name} | model={model_name} | "
                        f"handler_type={type(handler).__name__}",
                        rid,
                    )
                    return handler
            except Exception as exc:
                warning_id(
                    f"[ImageOrchestrator] Handler resolution method "
                    f"'{method_name}' failed for model '{model_name}': {exc}",
                    rid,
                )

        warning_id(
            f"[ImageOrchestrator] Could not resolve batch-capable handler for "
            f"model '{model_name}'",
            rid,
        )

        return None

    def _begin_embedding_batch_session(
        self,
        *,
        handler,
        batch_key: str,
        request_id: Optional[str] = None,
    ) -> None:
        rid = request_id or get_request_id()
        begin_fn = getattr(handler, "begin_document_embedding_session", None)

        if callable(begin_fn):
            try:
                self._call_with_supported_kwargs(
                    begin_fn,
                    document_key=batch_key,
                    request_id=rid,
                )
                debug_id(
                    f"[ImageOrchestrator] Began image embedding batch session | "
                    f"batch_key={batch_key}",
                    rid,
                )
            except Exception as exc:
                warning_id(
                    f"[ImageOrchestrator] Failed to begin batch session "
                    f"'{batch_key}': {exc}",
                    rid,
                )

    def _end_embedding_batch_session(
        self,
        *,
        handler,
        batch_key: str,
        request_id: Optional[str] = None,
    ) -> None:
        rid = request_id or get_request_id()
        end_fn = getattr(handler, "end_document_embedding_session", None)

        if callable(end_fn):
            try:
                self._call_with_supported_kwargs(
                    end_fn,
                    document_key=batch_key,
                    request_id=rid,
                )
                debug_id(
                    f"[ImageOrchestrator] Ended image embedding batch session | "
                    f"batch_key={batch_key}",
                    rid,
                )
            except Exception as exc:
                warning_id(
                    f"[ImageOrchestrator] Failed to end batch session "
                    f"'{batch_key}': {exc}",
                    rid,
                )

    def _get_batch_image_embeddings(
        self,
        *,
        handler,
        image_paths: List[str],
        batch_key: str,
        request_id: Optional[str] = None,
    ) -> Optional[List[Optional[List[float]]]]:
        """
        Try to use handler batch embedding if available.

        Returns:
            list of vectors, same order as image_paths, or None if unavailable.
        """
        rid = request_id or get_request_id()

        batch_fn = getattr(handler, "get_image_embeddings_batch", None)
        if not callable(batch_fn):
            debug_id(
                "[ImageOrchestrator] Handler does not expose "
                "get_image_embeddings_batch; will fall back to single-image embedding",
                rid,
            )
            return None

        try:
            vectors = self._call_with_supported_kwargs(
                batch_fn,
                images=image_paths,
                image_paths=image_paths,
                document_key=batch_key,
                request_id=rid,
            )

            if vectors is None:
                warning_id(
                    "[ImageOrchestrator] Batch embedding returned None; "
                    "falling back to singles",
                    rid,
                )
                return None

            if not isinstance(vectors, list):
                warning_id(
                    f"[ImageOrchestrator] Batch embedding returned unexpected "
                    f"type: {type(vectors)}",
                    rid,
                )
                return None

            if len(vectors) != len(image_paths):
                warning_id(
                    f"[ImageOrchestrator] Batch embedding count mismatch | "
                    f"expected={len(image_paths)} returned={len(vectors)}",
                    rid,
                )
                return None

            debug_id(
                f"[ImageOrchestrator] Batch image embeddings generated successfully | "
                f"count={len(vectors)}",
                rid,
            )

            return vectors

        except Exception as exc:
            warning_id(
                f"[ImageOrchestrator] Batch image embedding failed; "
                f"falling back to singles: {exc}",
                rid,
            )
            return None

    def _get_single_image_embeddings_fallback(
        self,
        *,
        image_paths: List[str],
        request_id: Optional[str] = None,
    ) -> List[Optional[List[float]]]:
        """
        Conservative fallback path: use existing per-image service call.
        """
        rid = request_id or get_request_id()
        vectors: List[Optional[List[float]]] = []

        for image_path in image_paths:
            try:
                vector = self.image_model_service.get_image_embedding(
                    image_path=image_path,
                    request_id=rid,
                )
                vectors.append(vector)
            except Exception as exc:
                warning_id(
                    f"[ImageOrchestrator] Single-image embedding fallback failed "
                    f"for '{image_path}': {exc}",
                    rid,
                )
                vectors.append(None)

        return vectors