# modules/orchestrators/image_orchestrator.py

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional, Dict, Any, List

from werkzeug.utils import secure_filename

from modules.orchestrators.base_orchestrator import BaseOrchestrator
from modules.configuration.log_config import (
    with_request_id,
    get_request_id,
    debug_id,
    info_id,
    warning_id,
    error_id,
)
from modules.configuration.config import DATABASE_PATH_IMAGES_FOLDER
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
    """

    def __init__(self):
        super().__init__()

        self.image_service = ImageService()
        self.embedding_service = ImageEmbeddingService()
        self.position_service = ImagePositionService()
        self.document_assoc_service = ImageCompletedDocumentAssociationService()
        self.task_assoc_service = ImageTaskAssociationService()
        self.problem_assoc_service = ImageProblemAssociationService()

    # ---------------------------------------------------------
    # INTERNAL HELPERS
    # ---------------------------------------------------------

    @staticmethod
    def _normalize_text(value: Optional[str], *, fallback: str = "") -> str:
        if value is None:
            return fallback
        if not isinstance(value, str):
            return str(value).strip()
        return value.strip()

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
                warning_id(f"[ImageOrchestrator] Image not found for delete id={image_id}", rid)
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

        if not files:
            return {
                "status": "validation_error",
                "message": "No files provided",
                "processed": 0,
                "failed": 0,
                "results": [],
            }

        os.makedirs(DATABASE_PATH_IMAGES_FOLDER, exist_ok=True)

        for file_obj in files:
            filename = secure_filename(getattr(file_obj, "filename", "") or "")

            if not filename:
                warning_id("[ImageOrchestrator] Skipping upload with empty filename", rid)
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

            file_path = os.path.join(DATABASE_PATH_IMAGES_FOLDER, filename)

            try:
                title = self._normalize_text(metadata.get("title"), fallback=Path(filename).stem)
                description = self._normalize_text(metadata.get("description"), fallback=title)

                # Pass through metadata for downstream page/document association if present
                img_metadata = metadata.get("img_metadata")
                if img_metadata is None:
                    img_metadata = {}

                file_obj.save(file_path)
                info_id(f"Saved image file to {file_path}", rid)

                created = self.create_image(
                    title=title,
                    description=description,
                    file_path=file_path,
                    img_metadata=img_metadata,
                    request_id=rid,
                )

                image_payload = created.get("image") or {}
                image_id = created.get("image_id") or image_payload.get("id")

                results.append(
                    self._build_upload_result(
                        success=True,
                        file_name=filename,
                        file_path=file_path,
                        image_id=image_id,
                        result=created,
                        http_status=200,
                    )
                )

                debug_id(
                    f"[ImageOrchestrator] Upload processed successfully | file={filename} | image_id={image_id}",
                    rid,
                )

            except ValueError as exc:
                warning_id(
                    f"[ImageOrchestrator] Validation error while processing file '{filename}': {exc}",
                    rid,
                )
                results.append(
                    self._build_upload_result(
                        success=False,
                        file_name=filename,
                        file_path=file_path,
                        error="Validation error",
                        detail=str(exc),
                        http_status=400,
                    )
                )

            except Exception as exc:
                error_id(
                    f"Image orchestrator processing error for file '{filename}': {exc}",
                    rid,
                    exc_info=True,
                )
                results.append(
                    self._build_upload_result(
                        success=False,
                        file_name=filename,
                        file_path=file_path,
                        error="Image processing failed",
                        detail=str(exc),
                        http_status=500,
                    )
                )

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
            f"[ImageOrchestrator] process_upload_concurrent delegating to sequential path | max_workers={max_workers}",
            request_id or get_request_id(),
        )

        return self.process_upload(
            files=files,
            metadata=metadata,
            request_id=request_id,
        )