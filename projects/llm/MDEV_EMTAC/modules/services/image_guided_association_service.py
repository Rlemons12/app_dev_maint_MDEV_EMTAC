from __future__ import annotations

from typing import Optional, Dict, Any, List
import inspect
import os
import tempfile

from modules.configuration.log_config import (
    debug_id,
    info_id,
    warning_id,
    error_id,
    with_request_id,
    get_request_id,
)

from modules.emtacdb.emtacdb_fts import (
    Image,
    ImageCompletedDocumentAssociation,
)

from modules.services.image_extraction_service import ImageExtractionService
from modules.services.image_embedding_service import ImageEmbeddingService


class ImageGuidedAssociationService:
    """
    Guided image ingestion layer.

    Responsibilities:
      - Attempt guided extraction
      - Fallback to basic extraction
      - Generate image embeddings (ALL paths)
      - Handle synthetic VLM visual descriptions
      - Never own transactions
    """

    def __init__(self):
        self.basic = ImageExtractionService()
        self.embedding_service = ImageEmbeddingService()

    # =========================================================
    # 🔥 NEW — VLM STRUCTURED VISUAL INGESTION
    # =========================================================

    @with_request_id
    def create_from_description(
        self,
        *,
        session,
        complete_document_id: int,
        position_id: Optional[int],
        label: str,
        visual_type: str,
        description: str,
        embedding_model_service=None,
        request_id: Optional[str] = None,
    ) -> int:

        rid = request_id or get_request_id()

        try:
            # Create synthetic placeholder image file
            # (We store description as content but file_path must exist for consistency)
            temp_dir = tempfile.gettempdir()
            fake_path = os.path.join(
                temp_dir,
                f"vlm_visual_{complete_document_id}_{label}.txt",
            )

            with open(fake_path, "w", encoding="utf-8") as f:
                f.write(description)

            image = Image(
                title=label,
                description=f"{visual_type}: {description}",
                file_path=fake_path,
            )

            session.add(image)
            session.flush()

            assoc = ImageCompletedDocumentAssociation(
                image_id=image.id,
                complete_document_id=complete_document_id,
            )

            session.add(assoc)
            session.flush()

            debug_id(
                f"[VLM IMG] created synthetic image | id={image.id} | label={label}",
                rid,
            )

            # Generate embedding
            if embedding_model_service:
                try:
                    model_name = embedding_model_service.get_current_model_name(
                        request_id=rid
                    )

                    vector = embedding_model_service.get_text_embedding(
                        description,
                        request_id=rid,
                    )

                    if vector:
                        self.embedding_service.add_pgvector(
                            session=session,
                            image_id=image.id,
                            model_name=model_name,
                            embedding=vector,
                            request_id=rid,
                        )

                except Exception as e:
                    warning_id(
                        f"[VLM IMG] embedding failed image_id={image.id}: {e}",
                        rid,
                    )

            return 1

        except Exception as e:
            error_id(f"[VLM IMG] failed create_from_description: {e}", rid)
            return 0

    # =========================================================
    # EXISTING PDF EXTRACTION FLOW (UNCHANGED)
    # =========================================================

    @with_request_id
    def extract_and_associate(
        self,
        *,
        session,
        file_path: str,
        complete_document_id: int,
        position_id: Optional[int],
        embedding_model_service=None,
        request_id: Optional[str] = None,
    ) -> int:

        rid = request_id or get_request_id()
        created = 0

        def _count_images() -> int:
            return int(
                session.query(Image.id)
                .join(ImageCompletedDocumentAssociation)
                .filter(
                    ImageCompletedDocumentAssociation.complete_document_id
                    == complete_document_id
                )
                .count()
            )

        before_count = _count_images()

        # 1️⃣ Try guided extraction
        guided_fn = getattr(
            ImageCompletedDocumentAssociation,
            "guided_extraction_with_mapping",
            None,
        )

        if callable(guided_fn):
            try:
                info_id(
                    f"[GUIDED IMG] starting guided extraction doc_id={complete_document_id}",
                    rid,
                )

                payload: Dict[str, Any] = {
                    "complete_document_id": complete_document_id,
                    "position_id": position_id,
                }

                guided_sig = inspect.signature(guided_fn)
                guided_kwargs = {
                    "file_path": file_path,
                    "metadata": payload,
                    "request_id": rid,
                }

                if "session" in guided_sig.parameters:
                    guided_kwargs["session"] = session

                success, result, _ = guided_fn(**guided_kwargs)

                if success:
                    after_count = _count_images()
                    created = max(0, after_count - before_count)
                else:
                    warning_id("[GUIDED IMG] guided extraction failed", rid)

            except Exception as e:
                warning_id(f"[GUIDED IMG] guided extraction raised: {e}", rid)

        # 2️⃣ Fallback extraction
        if created == 0:
            try:
                basic_fn = getattr(self.basic, "extract_pdf_images", None)
                if callable(basic_fn):
                    basic_fn(
                        session=session,
                        file_path=file_path,
                        complete_document_id=complete_document_id,
                        position_id=position_id,
                        request_id=rid,
                    )

                    after_count = _count_images()
                    created = max(0, after_count - before_count)

            except Exception as e:
                error_id(f"[GUIDED IMG] fallback extraction failed: {e}", rid)
                return 0

        # 3️⃣ Embeddings for extracted images
        if created > 0 and embedding_model_service:

            new_images: List[Image] = (
                session.query(Image)
                .join(ImageCompletedDocumentAssociation)
                .filter(
                    ImageCompletedDocumentAssociation.complete_document_id
                    == complete_document_id
                )
                .order_by(Image.id.desc())
                .limit(created)
                .all()
            )

            model_name = embedding_model_service.get_current_model_name(
                request_id=rid
            )

            for image in new_images:
                try:
                    vector = embedding_model_service.get_image_embedding(
                        image.file_path,
                        request_id=rid,
                    )

                    if not vector:
                        continue

                    self.embedding_service.add_pgvector(
                        session=session,
                        image_id=image.id,
                        model_name=model_name,
                        embedding=vector,
                        request_id=rid,
                    )

                except Exception as e:
                    warning_id(
                        f"[GUIDED IMG] embedding failed image_id={image.id}: {e}",
                        rid,
                    )

        return created