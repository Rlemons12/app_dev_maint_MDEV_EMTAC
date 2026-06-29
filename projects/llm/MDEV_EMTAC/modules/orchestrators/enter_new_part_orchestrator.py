from __future__ import annotations

import os
from typing import Any, Dict

from sqlalchemy import and_
from flask import send_file
import os

from modules.services.image_service import ImageService
from modules.configuration.config import (
    UPLOAD_FOLDER,
    ALLOWED_EXTENSIONS,
    BASE_DIR,
)
from modules.configuration.log_config import logger, with_request_id
from modules.orchestrators.base_orchestrator import BaseOrchestrator
from modules.services.model_service import ModelService
from modules.services.position_service import PositionService
from modules.emtacdb.emtacdb_fts import (
    Part,
    Image,
    PartsPositionImageAssociation,
)
from modules.emtacdb.utlity.main_database.database import (
    add_image_to_db,
    add_parts_position_image_association,
)


class EnterNewPartOrchestrator(BaseOrchestrator):
    """
    Orchestrator for Enter New Part workflows.

    Responsibilities:
    - Own session lifecycle
    - Assemble read-only form data
    - Own transaction for part creation workflow
    """

    def __init__(self) -> None:
        super().__init__()
        self.model_service = ModelService()
        self.position_service = PositionService()
        self.image_service = ImageService()

    @staticmethod
    def _allowed_file(filename: str) -> bool:
        return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS

    @with_request_id
    def get_part_form_data(self) -> Dict[str, Any]:
        with self.transaction(
            read_only=True,
            operation_name="EnterNewPartOrchestrator.get_part_form_data",
        ) as session:
            models = self.model_service.find(session=session, limit=10000)
            positions = self.position_service.find(session=session)

            return {
                "success": True,
                "message": "Part form data retrieved successfully.",
                "data": {
                    "models": [
                        {"id": model.id, "name": model.name}
                        for model in models
                    ],
                    "positions": [
                        {
                            "id": position.id,
                            "name": f"Position {position.id}",
                        }
                        for position in positions
                    ],
                },
                "status_code": 200,
            }

    @with_request_id
    def get_enter_part_page_data(self) -> Dict[str, Any]:
        with self.transaction(
            read_only=True,
            operation_name="EnterNewPartOrchestrator.get_enter_part_page_data",
        ) as session:
            positions = self.position_service.find(session=session)

            return {
                "success": True,
                "message": "Enter part page data retrieved successfully.",
                "data": {
                    "positions": positions,
                },
                "status_code": 200,
            }

    @with_request_id
    def submit_enter_part_form(self, data: Dict[str, Any]) -> Dict[str, Any]:
        with self.transaction(
            read_only=False,
            operation_name="EnterNewPartOrchestrator.submit_enter_part_form",
        ) as session:
            part_number = data.get("part_number")
            name = data.get("name")
            oem_mfg = data.get("oem_mfg")
            model = data.get("model")
            class_flag = data.get("class_flag")
            ud6 = data.get("ud6")
            type_value = data.get("type")
            notes = data.get("notes")
            documentation = data.get("documentation")

            if not part_number:
                return {
                    "success": False,
                    "message": "Part number is required.",
                    "status_code": 400,
                }

            new_part = Part(
                part_number=part_number,
                name=name,
                oem_mfg=oem_mfg,
                model=model,
                class_flag=class_flag,
                ud6=ud6,
                type=type_value,
                notes=notes,
                documentation=documentation,
            )

            session.add(new_part)
            session.flush()
            part_id = new_part.id
            logger.info(f"Created new part with ID: {part_id}")

            uploaded_file = data.get("part_image")
            position_id = data.get("position_id")
            image_title = data.get("image_title") or f"Image for {part_number}"
            image_description = data.get("image_description") or f"Image for part {part_number}"

            if uploaded_file and getattr(uploaded_file, "filename", ""):
                if not self._allowed_file(uploaded_file.filename):
                    return {
                        "success": False,
                        "message": "File type not allowed. Please upload jpg, jpeg, png, or gif files only.",
                        "status_code": 400,
                    }

                from werkzeug.utils import secure_filename

                filename = secure_filename(uploaded_file.filename)
                upload_folder = os.path.join(UPLOAD_FOLDER, "parts")
                os.makedirs(upload_folder, exist_ok=True)

                abs_file_path = os.path.join(upload_folder, filename)
                uploaded_file.save(abs_file_path)
                logger.info(f"Image saved to: {abs_file_path}")

                try:
                    rel_file_path = os.path.relpath(abs_file_path, BASE_DIR)

                    image_id = add_image_to_db(
                        title=image_title,
                        file_path=rel_file_path,
                        position_id=position_id,
                        description=image_description,
                    )

                    if image_id and position_id:
                        add_parts_position_image_association(
                            part_id=part_id,
                            position_id=position_id,
                            image_id=image_id,
                        )

                except Exception as exc:
                    logger.error(
                        f"Utility image insert failed, falling back to direct DB operations: {exc}",
                        exc_info=True,
                    )

                    rel_file_path = os.path.relpath(abs_file_path, BASE_DIR)

                    existing_image = session.query(Image).filter(
                        and_(
                            Image.title == image_title,
                            Image.description == image_description,
                        )
                    ).first()

                    if existing_image is not None and existing_image.file_path == rel_file_path:
                        new_image = existing_image
                    else:
                        new_image = Image(
                            title=image_title,
                            description=image_description,
                            file_path=rel_file_path,
                        )
                        session.add(new_image)
                        session.flush()

                    if position_id:
                        association = PartsPositionImageAssociation(
                            part_id=part_id,
                            position_id=position_id,
                            image_id=new_image.id,
                        )
                        session.add(association)

            return {
                "success": True,
                "message": "Part successfully entered!",
                "data": {
                    "part_id": part_id,
                },
                "status_code": 200,
            }

    @with_request_id
    def get_part_image(self, *, image_id: int):
        with self.transaction(
                read_only=True,
                operation_name="EnterNewPartOrchestrator.get_part_image",
        ) as session:
            image = self.image_service.get(
                session,
                image_id=image_id,
            )

            if not image:
                return {
                    "success": False,
                    "message": "Image not found",
                    "status_code": 404,
                }

            file_path = image.file_path
            if not file_path:
                return {
                    "success": False,
                    "message": "Image file path is missing",
                    "status_code": 404,
                }

            abs_path = file_path
            if not os.path.isabs(abs_path):
                abs_path = os.path.join(BASE_DIR, file_path)

            if not os.path.exists(abs_path):
                return {
                    "success": False,
                    "message": "Image file not found on disk",
                    "status_code": 404,
                }

            return {
                "success": True,
                "response": send_file(abs_path),
                "status_code": 200,
            }