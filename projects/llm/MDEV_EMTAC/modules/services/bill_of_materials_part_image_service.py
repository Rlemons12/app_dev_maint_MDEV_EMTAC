from __future__ import annotations

from typing import List, Optional

from sqlalchemy import and_

from modules.configuration.log_config import logger, with_request_id
from modules.emtacdb.emtacdb_fts import (
    Image,
    PartsPositionImageAssociation,
)


class BillOfMaterialsPartImageService:
    """
    Domain service for image records related to parts.
    """

    @with_request_id
    def get_images_for_part(self, *, session, part_id: int) -> List[Image]:
        images = (
            session.query(Image)
            .join(
                PartsPositionImageAssociation,
                PartsPositionImageAssociation.image_id == Image.id,
            )
            .filter(PartsPositionImageAssociation.part_id == part_id)
            .all()
        )
        logger.debug("Retrieved %s images for part_id=%s", len(images), part_id)
        return images

    @with_request_id
    def get_or_create_image(
        self,
        *,
        session,
        title: str,
        description: str,
        file_path: str,
    ) -> Image:
        existing_image = (
            session.query(Image)
            .filter(and_(Image.title == title, Image.description == description))
            .first()
        )

        if existing_image is not None and existing_image.file_path == file_path:
            logger.info(
                "Found existing image with matching title/description/file_path: %s",
                title,
            )
            return existing_image

        new_image = Image(
            title=title,
            description=description,
            file_path=file_path,
        )
        session.add(new_image)
        session.flush()

        logger.info(
            "Created new image record id=%s title=%s file_path=%s",
            new_image.id,
            title,
            file_path,
        )
        return new_image

    @with_request_id
    def get_image_by_id(self, *, session, image_id: int) -> Optional[Image]:
        image = session.query(Image).filter_by(id=image_id).first()
        logger.debug("Fetched image_id=%s found=%s", image_id, bool(image))
        return image

    @with_request_id
    def delete_image(self, *, session, image: Image) -> None:
        logger.info("Deleting image record id=%s", image.id)
        session.delete(image)