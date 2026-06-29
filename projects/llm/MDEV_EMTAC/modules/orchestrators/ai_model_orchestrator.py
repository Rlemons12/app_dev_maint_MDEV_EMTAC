# modules/orchestrators/ai_model_orchestrator.py

from typing import Optional, Dict, Any

from modules.orchestrators.base_orchestrator import BaseOrchestrator

from modules.services.ai_models_service import AIModelsService
from modules.services.ai_models_embedding_service import (
    AIModelsEmbeddingService,
)
from modules.services.ai_model_image_service import (
    AIModelImageService,
)
from modules.services.ai_models_vlm_service import (
    AIModelsVLMService,
)


class AIModelOrchestrator(BaseOrchestrator):
    """
    Unified AI coordination layer.

    Responsibilities:
    - Multi-service AI coordination
    - Backend transparency
    - Unified response structure
    - Optional multimodal routing

    Note:
    - Does NOT own transactions
    - Inherits BaseOrchestrator for logging + timing consistency
    """

    def __init__(self):
        super().__init__()

        # Proper instance-based service ownership
        self.text_service = AIModelsService()
        self.embedding_service = AIModelsEmbeddingService()
        self.image_service = AIModelImageService()
        self.vlm_service = AIModelsVLMService()

    # =====================================================
    # TEXT GENERATION
    # =====================================================

    def generate_text(
        self,
        *,
        question: str,
        context: str = "",
    ) -> Dict[str, Any]:

        with self._timed("AIModelOrchestrator.generate_text"):

            self._debug("Generating text response")

            answer = self.text_service.answer(
                question=question,
                context=context,
                request_id=self._rid(),
            )

            return {
                "type": "text",
                "question": question,
                "answer": answer,
            }

    # =====================================================
    # EMBEDDING
    # =====================================================

    def embed_text(
        self,
        *,
        text: str,
    ) -> Dict[str, Any]:

        with self._timed("AIModelOrchestrator.embed_text"):

            self._debug("Generating embedding")

            vec = self.embedding_service.get_embeddings(
                text=text,
                request_id=self._rid(),
            )

            return {
                "type": "embedding",
                "dimension": len(vec),
                "vector": vec,
            }

    # =====================================================
    # IMAGE PROCESSING
    # =====================================================

    def process_image(
        self,
        *,
        image_path: str,
    ) -> Dict[str, Any]:

        with self._timed("AIModelOrchestrator.process_image"):

            self._debug(f"Processing image: {image_path}")

            result = self.image_service.process_image(
                image_path=image_path,
                request_id=self._rid(),
            )

            return {
                "type": "image",
                "result": result,
            }

    # =====================================================
    # VLM
    # =====================================================

    def describe_image_with_vlm(
        self,
        *,
        image_path: str,
    ) -> Dict[str, Any]:

        with self._timed("AIModelOrchestrator.describe_image_with_vlm"):

            self._debug(f"Describing image via VLM: {image_path}")

            description = self.vlm_service.describe_image(
                image_path=image_path,
                request_id=self._rid(),
            )

            return {
                "type": "vlm_image",
                "description": description,
            }

    def pdf_to_markdown(
        self,
        *,
        pdf_path: str,
    ) -> Dict[str, Any]:

        with self._timed("AIModelOrchestrator.pdf_to_markdown"):

            self._debug(f"Extracting markdown from PDF: {pdf_path}")

            markdown = self.vlm_service.extract_markdown_from_pdf(
                pdf_path=pdf_path,
                request_id=self._rid(),
            )

            return {
                "type": "vlm_pdf",
                "markdown": markdown,
            }

    # =====================================================
    # MULTIMODAL
    # =====================================================

    def multimodal_query(
        self,
        *,
        question: str,
        image_path: Optional[str] = None,
        context: str = "",
    ) -> Dict[str, Any]:

        with self._timed("AIModelOrchestrator.multimodal_query"):

            self._debug("Running multimodal query")

            image_description = None

            if image_path:
                image_description = self.vlm_service.describe_image(
                    image_path=image_path,
                    request_id=self._rid(),
                )

                context = f"{context}\nImage Description:\n{image_description}"

            answer = self.text_service.answer(
                question=question,
                context=context,
                request_id=self._rid(),
            )

            return {
                "type": "multimodal",
                "question": question,
                "image_description": image_description,
                "answer": answer,
            }