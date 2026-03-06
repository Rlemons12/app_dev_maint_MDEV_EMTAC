from __future__ import annotations

import fitz
import os
import uuid
from typing import Optional, List
from sqlalchemy.orm import Session

from modules.configuration.config import (
    DATABASE_PATH_IMAGES_FOLDER,
    DATABASE_DIR,
)
from modules.emtacdb.emtacdb_fts import (
    Document,
    ImageCompletedDocumentAssociation,
)
from modules.services.image_service import ImageService
from modules.configuration.log_config import warning_id


# ---------------------------------------------------------
# SAFE DIRECTORY VALIDATION
# ---------------------------------------------------------
if not DATABASE_DIR:
    raise RuntimeError("DATABASE_DIR is not configured")

# Guard against accidental tuple (common trailing comma bug)
if isinstance(DATABASE_PATH_IMAGES_FOLDER, tuple):
    DATABASE_PATH_IMAGES_FOLDER = DATABASE_PATH_IMAGES_FOLDER[0]

if not isinstance(DATABASE_PATH_IMAGES_FOLDER, str):
    raise RuntimeError(
        f"DATABASE_PATH_IMAGES_FOLDER must be a string. "
        f"Got: {type(DATABASE_PATH_IMAGES_FOLDER)}"
    )

# Ensure directory exists
os.makedirs(DATABASE_PATH_IMAGES_FOLDER, exist_ok=True)


class ImageExtractionService:
    """
    Extract images from PDF and associate them to document chunks.

    HARD RULES:
    - No session creation
    - No commit/rollback
    - Orchestrator owns transaction
    """

    def __init__(self):
        self.image_service = ImageService()

    # ---------------------------------------------------------
    # SAVE IMAGE (store RELATIVE path in DB)
    # ---------------------------------------------------------
    def _save_image(
        self,
        *,
        image_bytes: bytes,
        ext: str,
        complete_document_id: int,
        page_index: int,
        img_index: int,
    ) -> str:

        filename = (
            f"{complete_document_id}_"
            f"{page_index + 1}_"
            f"{img_index + 1}_"
            f"{uuid.uuid4().hex}.{ext}"
        )

        absolute_path = os.path.join(
            DATABASE_PATH_IMAGES_FOLDER,
            filename,
        )

        # Write file
        with open(absolute_path, "wb") as f:
            f.write(image_bytes)

        # Return relative path for DB portability
        return os.path.relpath(
            absolute_path,
            start=DATABASE_DIR,
        )

    # ---------------------------------------------------------
    # MAIN EXTRACTION
    # ---------------------------------------------------------
    def extract_pdf_images(
        self,
        session: Session,
        *,
        file_path: str,
        complete_document_id: int,
        position_id: Optional[int] = None,
        request_id: Optional[str] = None,
    ) -> int:

        extracted = 0
        doc = None

        try:
            doc = fitz.open(file_path)

            chunks: List[Document] = (
                session.query(Document)
                .filter_by(complete_document_id=complete_document_id)
                .order_by(Document.id)
                .all()
            )

            if not chunks:
                return 0

            for page_index in range(len(doc)):
                page = doc[page_index]
                images = page.get_images(full=True)

                for img_index, img in enumerate(images):
                    try:
                        xref = img[0]
                        base_image = doc.extract_image(xref)

                        image_bytes = base_image["image"]
                        ext = base_image.get("ext", "png")

                        relative_path = self._save_image(
                            image_bytes=image_bytes,
                            ext=ext,
                            complete_document_id=complete_document_id,
                            page_index=page_index,
                            img_index=img_index,
                        )

                        title = f"Page {page_index + 1} Image {img_index + 1}"

                        metadata = {
                            "page_number": page_index + 1,
                            "image_index": img_index + 1,
                            "extraction_method": "basic_pdf",
                            "structure_guided": False,
                            "position_id": position_id,
                        }

                        image = self.image_service.create(
                            session=session,
                            title=title,
                            file_path=relative_path,
                            description=f"Extracted from {os.path.basename(file_path)}",
                            img_metadata=metadata,
                            request_id=request_id,
                        )

                        if not image:
                            continue

                        chunk_index = page_index % len(chunks)
                        selected_chunk = chunks[chunk_index]

                        association = ImageCompletedDocumentAssociation(
                            complete_document_id=complete_document_id,
                            image_id=image.id,
                            document_id=selected_chunk.id,
                            page_number=page_index + 1,
                            chunk_index=chunk_index,
                            association_method="basic_pdf_fallback",
                            confidence_score=0.5,
                            context_metadata=None,
                        )

                        session.add(association)
                        extracted += 1

                    except Exception as e:
                        warning_id(
                            f"[IMAGE EXTRACTION] page={page_index + 1} "
                            f"image={img_index + 1} error={e}",
                            request_id,
                        )
                        continue

        finally:
            if doc:
                doc.close()

        return extracted