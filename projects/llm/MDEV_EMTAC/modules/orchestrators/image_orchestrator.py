# modules/orchestrators/image_orchestrator.py

from typing import Optional, Dict, Any, List

from modules.orchestrators.base_orchestrator import BaseOrchestrator
from modules.configuration.log_config import (
    with_request_id,
    debug_id,
    info_id,
    error_id,
)

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
    # CREATE IMAGE
    # ---------------------------------------------------------

    @with_request_id
    def create_image(
        self,
        *,
        title: str,
        description: str,
        file_path: str,
    ) -> Dict[str, Any]:

        with self.transaction() as session:

            image = self.image_service.create(
                session=session,
                title=title,
                description=description,
                file_path=file_path,
            )

            return {
                "status": "created",
                "image": image,
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
    ) -> Dict[str, Any]:

        with self.transaction() as session:

            image = self.image_service.create(
                session=session,
                title=title,
                description=description,
                file_path=file_path,
            )

            embedding_obj = self.embedding_service.add_pgvector(
                session=session,
                image_id=image.id,
                model_name=model_name,
                embedding=embedding,
            )

            return {
                "status": "created",
                "image": image,
                "embedding": embedding_obj,
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
    ) -> Dict[str, Any]:

        with self.transaction() as session:

            assoc = self.position_service.associate(
                session=session,
                image_id=image_id,
                position_id=position_id,
            )

            return {
                "status": "attached",
                "association": assoc,
            }

    @with_request_id
    def attach_to_task(
        self,
        *,
        image_id: int,
        task_id: int,
    ) -> Dict[str, Any]:

        with self.transaction() as session:

            assoc = self.task_assoc_service.associate(
                session=session,
                image_id=image_id,
                task_id=task_id,
            )

            return {
                "status": "attached",
                "association": assoc,
            }

    @with_request_id
    def attach_to_problem(
        self,
        *,
        image_id: int,
        problem_id: int,
    ) -> Dict[str, Any]:

        with self.transaction() as session:

            assoc = self.problem_assoc_service.associate(
                session=session,
                image_id=image_id,
                problem_id=problem_id,
            )

            return {
                "status": "attached",
                "association": assoc,
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
    ) -> Dict[str, Any]:

        with self.transaction() as session:

            assoc = self.document_assoc_service.create_association(
                session=session,
                image_id=image_id,
                complete_document_id=complete_document_id,
                chunk_id=chunk_id,
                confidence_score=confidence_score,
                association_method=association_method,
            )

            return {
                "status": "attached",
                "association": assoc,
            }

    # ---------------------------------------------------------
    # DELETE IMAGE (SAFE CASCADE)
    # ---------------------------------------------------------

    @with_request_id
    def delete_image(
        self,
        *,
        image_id: int,
    ) -> Dict[str, Any]:

        with self.transaction() as session:

            # Remove embeddings
            self.embedding_service.remove_all_for_image(
                session=session,
                image_id=image_id,
            )

            # Remove position associations
            self.position_service.remove_all_for_image(
                session=session,
                image_id=image_id,
            )

            # Remove task associations
            self.task_assoc_service.remove_all_for_image(
                session=session,
                image_id=image_id,
            )

            # Remove problem associations
            self.problem_assoc_service.remove_all_for_image(
                session=session,
                image_id=image_id,
            )

            # Remove document associations
            self.document_assoc_service.remove_all_for_image(
                session=session,
                image_id=image_id,
            )

            # Remove image
            self.image_service.delete(
                session=session,
                image_id=image_id,
            )

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
    ) -> Dict[str, Any]:

        with self.transaction(read_only=True) as session:

            image = self.image_service.get(
                session=session,
                image_id=image_id,
            )

            if not image:
                return {"status": "not_found", "image_id": image_id}

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
                "image": image,
                "positions": positions,
                "tasks": tasks,
                "problems": problems,
                "documents": docs,
                "embeddings": embeddings,
            }
