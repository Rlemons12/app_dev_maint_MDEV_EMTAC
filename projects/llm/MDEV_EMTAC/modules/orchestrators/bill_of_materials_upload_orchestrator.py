from __future__ import annotations

import os
from typing import Any, Dict, Optional

from sqlalchemy.orm import Session
from werkzeug.utils import secure_filename

from modules.configuration.config import DB_LOADSHEET_BOMS
from modules.configuration.config_env import DatabaseConfig
from modules.configuration.log_config import logger, with_request_id
from modules.orchestrators.base_orchestrator import BaseOrchestrator
from modules.services.bill_of_materials_processing_service import (
    BillOfMaterialsProcessingService,
)
from modules.services.position_service import PositionService


class BillOfMaterialsUploadOrchestrator(BaseOrchestrator):
    """
    Orchestrator for BOM upload workflow.

    RESPONSIBILITIES:
    - Own session lifecycle / transaction boundary
    - Validate upload inputs
    - Resolve or create Position using the SAME session
    - Save uploaded file to disk
    - Call BOM processing service using orchestrator-owned session
    - Commit on success
    - Roll back on failure

    HARD RULES:
    - Orchestrator owns commit / rollback / close
    - Services never create or close sessions
    """

    def __init__(
        self,
        bom_processing_service: Optional[BillOfMaterialsProcessingService] = None,
        position_service: Optional[PositionService] = None,
        db_config: Optional[DatabaseConfig] = None,
    ) -> None:
        super().__init__()
        self.bom_processing_service = (
            bom_processing_service or BillOfMaterialsProcessingService()
        )
        self.position_service = position_service or PositionService()
        self.db_config = db_config or DatabaseConfig()

    @with_request_id
    def submit_bill_of_materials_upload(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute BOM upload workflow.

        Expected input keys:
        - file
        - image_path
        - area_id
        - equipment_group_id
        - model_id
        - asset_number_id
        - location_id
        - site_location_id

        Response shape:
        {
            "success": bool,
            "message": str,
            "data": {...},
            "status_code": int,
        }
        """
        image_path = data.get("image_path")
        uploaded_file = data.get("file")

        if not image_path:
            logger.error("Image path is required but was not provided.")
            return {
                "success": False,
                "message": "Image path is required",
                "status_code": 400,
            }

        logger.info("Image path received | image_path=%s", image_path)

        if uploaded_file is None:
            logger.error("No file part in the request.")
            return {
                "success": False,
                "message": "No file part",
                "status_code": 400,
            }

        if not getattr(uploaded_file, "filename", None):
            logger.error("No selected file in the request.")
            return {
                "success": False,
                "message": "No selected file",
                "status_code": 400,
            }

        if not self.bom_processing_service.allowed_file(uploaded_file.filename):
            logger.error(
                "Invalid file type attempted | filename=%s",
                uploaded_file.filename,
            )
            return {
                "success": False,
                "message": "Invalid file type",
                "status_code": 400,
            }

        metadata = {
            "area_id": self._to_int(data.get("area_id")),
            "equipment_group_id": self._to_int(data.get("equipment_group_id")),
            "model_id": self._to_int(data.get("model_id")),
            "asset_number_id": self._to_int(data.get("asset_number_id")),
            "location_id": self._to_int(data.get("location_id")),
            "site_location_id": self._to_int(data.get("site_location_id")),
        }

        logger.info(
            "Received Position metadata | "
            "area_id=%s | equipment_group_id=%s | model_id=%s | "
            "asset_number_id=%s | location_id=%s | site_location_id=%s",
            metadata["area_id"],
            metadata["equipment_group_id"],
            metadata["model_id"],
            metadata["asset_number_id"],
            metadata["location_id"],
            metadata["site_location_id"],
        )

        os.makedirs(DB_LOADSHEET_BOMS, exist_ok=True)

        filename = secure_filename(uploaded_file.filename)
        file_path = os.path.join(DB_LOADSHEET_BOMS, filename)

        logger.info(
            "Prepared upload target path | original_filename=%s | secure_filename=%s | file_path=%s",
            uploaded_file.filename,
            filename,
            file_path,
        )

        session: Session = self.db_config.get_main_session()

        try:
            position_id = self._resolve_or_create_position(
                session=session,
                metadata=metadata,
            )

            if position_id is None:
                logger.error("Failed to resolve or create position.")
                session.rollback()
                return {
                    "success": False,
                    "message": "Error creating position",
                    "status_code": 500,
                }

            logger.info("Position resolved successfully | position_id=%s", position_id)

            uploaded_file.save(file_path)
            logger.info("File successfully uploaded | file_path=%s", file_path)

            target_path = self.bom_processing_service.process_bom_loadsheet(
                session=session,
                file_path=file_path,
                image_path=image_path,
                position_id=position_id,
            )

            session.commit()

            logger.info(
                "BOM upload workflow completed successfully | "
                "position_id=%s | file_path=%s | target_path=%s",
                position_id,
                file_path,
                target_path,
            )

            return {
                "success": True,
                "message": "File successfully processed",
                "data": {
                    "position_id": position_id,
                    "file_path": file_path,
                    "target_path": target_path,
                },
                "status_code": 200,
            }

        except Exception as exc:
            logger.error(
                "Error during BOM upload workflow | file_path=%s | error=%s",
                file_path,
                exc,
                exc_info=True,
            )
            session.rollback()
            return {
                "success": False,
                "message": f"An error occurred while processing the file: {exc}",
                "status_code": 500,
            }

        finally:
            session.close()

    def _resolve_or_create_position(
        self,
        session: Session,
        metadata: Dict[str, Any],
    ) -> Optional[int]:
        """
        Resolve an existing Position or create a new one using the same session
        owned by this orchestrator.
        """
        logger.info("Resolving or creating position using orchestrator-owned session.")

        if hasattr(self.position_service, "resolve_from_metadata"):
            position_id = self.position_service.resolve_from_metadata(
                session=session,
                metadata=metadata,
            )
            logger.info(
                "Position resolved via PositionService.resolve_from_metadata | position_id=%s",
                position_id,
            )
            return position_id

        if hasattr(self.position_service, "get_or_create_position_from_metadata"):
            position = self.position_service.get_or_create_position_from_metadata(
                session=session,
                metadata=metadata,
            )
            position_id = getattr(position, "id", None) if position else None
            logger.info(
                "Position resolved via PositionService.get_or_create_position_from_metadata | position_id=%s",
                position_id,
            )
            return position_id

        logger.error(
            "PositionService does not expose a supported resolver method. "
            "Expected resolve_from_metadata(session=..., metadata=...) or "
            "get_or_create_position_from_metadata(session=..., metadata=...)."
        )
        return None

    @staticmethod
    def _to_int(value: Any) -> Optional[int]:
        """
        Safely normalize incoming values to int or None.
        """
        if value in (None, "", "None"):
            return None
        try:
            return int(value)
        except (TypeError, ValueError):
            return None