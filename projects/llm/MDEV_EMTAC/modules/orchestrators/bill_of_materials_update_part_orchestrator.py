from __future__ import annotations

from typing import Any, Dict, Optional

from sqlalchemy.exc import IntegrityError

from modules.configuration.config_env import get_db_config
from modules.configuration.log_config import logger, with_request_id
from modules.services.bill_of_materials_part_service import (
    BillOfMaterialsPartService,
)
from modules.services.bill_of_materials_part_image_service import (
    BillOfMaterialsPartImageService,
)
from modules.services.bill_of_materials_part_image_association_service import (
    BillOfMaterialsPartImageAssociationService,
)
from modules.services.bill_of_materials_position_service import (
    BillOfMaterialsPositionService,
)
from modules.services.file_storage_service import FileStorageService


class BillOfMaterialsUpdatePartOrchestrator:
    """
    Orchestrator for Edit Part workflows.

    Responsibilities:
    - Own session lifecycle
    - Own transaction boundaries
    - Sequence services
    - Rollback on failures
    """

    def __init__(
        self,
        shared_db_config=None,
        part_service: Optional[BillOfMaterialsPartService] = None,
        part_image_service: Optional[BillOfMaterialsPartImageService] = None,
        association_service: Optional[BillOfMaterialsPartImageAssociationService] = None,
        position_service: Optional[BillOfMaterialsPositionService] = None,
        file_storage_service: Optional[FileStorageService] = None,
    ) -> None:
        self.db_config = shared_db_config or get_db_config()
        self.part_service = part_service or BillOfMaterialsPartService()
        self.part_image_service = part_image_service or BillOfMaterialsPartImageService()
        self.association_service = (
            association_service or BillOfMaterialsPartImageAssociationService()
        )
        self.position_service = position_service or BillOfMaterialsPositionService()
        self.file_storage_service = file_storage_service or FileStorageService()

    @with_request_id
    def get_edit_part_view_data(
        self,
        *,
        part_id: int,
        search_query: str = "",
    ) -> Dict[str, Any]:
        session = self.db_config.get_main_session()

        try:
            part = self.part_service.get_part_by_id(session=session, part_id=part_id)
            if not part:
                return {
                    "success": False,
                    "message": "Part not found.",
                    "status_code": 404,
                }

            part_images = self.part_image_service.get_images_for_part(
                session=session,
                part_id=part_id,
            )
            positions = self.position_service.get_all_positions(session=session)

            return {
                "success": True,
                "message": "Edit part data loaded successfully.",
                "status_code": 200,
                "data": {
                    "part": part,
                    "part_images": part_images,
                    "positions": positions,
                    "search_query": search_query,
                },
            }
        except Exception as exc:
            logger.error(
                "Error loading edit part view data for part_id=%s: %s",
                part_id,
                exc,
                exc_info=True,
            )
            return {
                "success": False,
                "message": f"Error loading part data: {exc}",
                "status_code": 500,
            }
        finally:
            session.close()

    @with_request_id
    def search_parts(
            self,
            *,
            search_text: str,
            limit: int = 10,
            is_ajax: bool = False,
    ) -> Dict[str, Any]:
        session = self.db_config.get_main_session()

        try:
            positions = self.position_service.get_all_positions(session=session)

            normalized_search_text = (search_text or "").strip()

            if not normalized_search_text:
                return {
                    "success": False,
                    "message": "Please enter a search query.",
                    "message_html": '<div class="alert alert-info">Please enter a search query.</div>',
                    "status_code": 200,
                    "flash_category": "info",
                    "data": {
                        "positions": positions,
                        "parts": [],
                        "search_query": "",
                    },
                }

            logger.debug(
                "Searching parts with normalized query='%s', limit=%s, is_ajax=%s",
                normalized_search_text,
                limit,
                is_ajax,
            )

            # ---------------------------------------------------------
            # SEARCH STRATEGY
            # ---------------------------------------------------------
            # Part-number-like searches should NOT use PostgreSQL FTS.
            # Examples:
            #   A123
            #   100-456
            #   FB1-2
            #   MMABF
            #
            # FTS is better for longer natural-language text.
            # ---------------------------------------------------------
            use_fts = True

            has_spaces = " " in normalized_search_text
            short_query = len(normalized_search_text) <= 20
            looks_like_part_number = any(ch.isdigit() for ch in normalized_search_text)
            has_symbolic_part_chars = any(ch in "-_/" for ch in normalized_search_text)

            if short_query and (looks_like_part_number or has_symbolic_part_chars or not has_spaces):
                use_fts = False

            logger.debug(
                "Part search strategy: use_fts=%s for query='%s'",
                use_fts,
                normalized_search_text,
            )

            parts = self.part_service.search_parts(
                session=session,
                search_text=normalized_search_text,
                limit=limit,
                use_fts=use_fts,
            )

            if not parts and use_fts:
                logger.debug(
                    "No results returned from FTS search for query='%s'. Falling back to non-FTS search.",
                    normalized_search_text,
                )
                parts = self.part_service.search_parts(
                    session=session,
                    search_text=normalized_search_text,
                    limit=limit,
                    use_fts=False,
                )

            if not parts:
                return {
                    "success": False,
                    "message": "No parts found matching your search criteria.",
                    "message_html": '<div class="alert alert-info">No parts found matching your search criteria.</div>',
                    "status_code": 200,
                    "flash_category": "info",
                    "data": {
                        "positions": positions,
                        "parts": [],
                        "search_query": normalized_search_text,
                    },
                }

            redirect_part_id = None
            if len(parts) == 1 and not is_ajax:
                redirect_part_id = parts[0].id

            return {
                "success": True,
                "message": "Parts found.",
                "status_code": 200,
                "data": {
                    "positions": positions,
                    "parts": parts,
                    "search_query": normalized_search_text,
                    "redirect_part_id": redirect_part_id,
                },
            }

        except Exception as exc:
            logger.error(
                "Error searching parts with query=%s: %s",
                search_text,
                exc,
                exc_info=True,
            )
            return {
                "success": False,
                "message": f"An error occurred during search: {exc}",
                "message_html": f'<div class="alert alert-danger">An error occurred during search: {exc}</div>',
                "status_code": 500,
                "flash_category": "error",
                "data": {
                    "positions": [],
                    "parts": [],
                    "search_query": (search_text or "").strip(),
                },
            }
        finally:
            session.close()

    @with_request_id
    def update_part(self, *, payload: Dict[str, Any]) -> Dict[str, Any]:
        session = self.db_config.get_main_session()
        saved_upload: Optional[Dict[str, str]] = None

        try:
            part_id = payload["part_id"]
            part = self.part_service.get_part_by_id(session=session, part_id=part_id)

            if not part:
                return {
                    "success": False,
                    "message": "Part not found.",
                    "status_code": 404,
                    "view_data": self._build_view_data_for_failure(
                        session=session,
                        part=None,
                        part_id=part_id,
                        search_query=payload.get("search_query", ""),
                    ),
                }

            self.part_service.update_part_fields(
                part=part,
                part_fields=payload["part_fields"],
            )

            uploaded_file = payload.get("uploaded_file")
            if uploaded_file and getattr(uploaded_file, "filename", ""):
                saved_upload = self.file_storage_service.save_part_image(
                    uploaded_file=uploaded_file,
                    part_number=part.part_number or f"part_{part.id}",
                )

                image_title = (
                    payload.get("image_title") or f"Image for {part.part_number}"
                )
                image_description = (
                    payload.get("image_description")
                    or f"Image for part {part.part_number}"
                )

                image = self.part_image_service.get_or_create_image(
                    session=session,
                    title=image_title,
                    description=image_description,
                    file_path=saved_upload["relative_path"],
                )

                self.association_service.create_association(
                    session=session,
                    part_id=part.id,
                    position_id=payload.get("position_id"),
                    image_id=image.id,
                )

            remove_image_ids = payload.get("remove_image_ids", [])
            for image_id in remove_image_ids:
                association = self.association_service.get_association(
                    session=session,
                    part_id=part.id,
                    image_id=image_id,
                )

                if not association:
                    logger.warning(
                        "No association found for part_id=%s and image_id=%s",
                        part.id,
                        image_id,
                    )
                    continue

                self.association_service.delete_association(
                    session=session,
                    association=association,
                )

                remaining_count = self.association_service.count_image_associations(
                    session=session,
                    image_id=image_id,
                )

                if remaining_count == 0:
                    image = self.part_image_service.get_image_by_id(
                        session=session,
                        image_id=image_id,
                    )
                    if image:
                        absolute_path = self.file_storage_service.resolve_absolute_image_path(
                            image.file_path
                        )
                        self.file_storage_service.delete_file_if_exists(absolute_path)
                        self.part_image_service.delete_image(
                            session=session,
                            image=image,
                        )

            session.commit()

            return {
                "success": True,
                "message": "Part updated successfully!",
                "status_code": 200,
            }

        except ValueError as exc:
            session.rollback()

            if saved_upload:
                self.file_storage_service.delete_file_if_exists(
                    saved_upload["absolute_path"]
                )

            return {
                "success": False,
                "message": str(exc),
                "status_code": 400,
                "view_data": self._build_view_data_for_failure(
                    session=session,
                    part=None,
                    part_id=payload["part_id"],
                    search_query=payload.get("search_query", ""),
                ),
            }

        except IntegrityError:
            session.rollback()

            if saved_upload:
                self.file_storage_service.delete_file_if_exists(
                    saved_upload["absolute_path"]
                )

            return {
                "success": False,
                "message": "Part number must be unique.",
                "status_code": 400,
                "view_data": self._build_view_data_for_failure(
                    session=session,
                    part=None,
                    part_id=payload["part_id"],
                    search_query=payload.get("search_query", ""),
                ),
            }

        except Exception as exc:
            session.rollback()

            if saved_upload:
                self.file_storage_service.delete_file_if_exists(
                    saved_upload["absolute_path"]
                )

            logger.error(
                "Unexpected error updating part_id=%s: %s",
                payload["part_id"],
                exc,
                exc_info=True,
            )
            return {
                "success": False,
                "message": f"An error occurred: {exc}",
                "status_code": 500,
                "view_data": self._build_view_data_for_failure(
                    session=session,
                    part=None,
                    part_id=payload["part_id"],
                    search_query=payload.get("search_query", ""),
                ),
            }
        finally:
            session.close()

    @with_request_id
    def get_part_image_file_data(self, *, image_id: int) -> Dict[str, Any]:
        session = self.db_config.get_main_session()

        try:
            image = self.part_image_service.get_image_by_id(
                session=session,
                image_id=image_id,
            )

            if not image:
                return {
                    "success": False,
                    "message": "Image not found",
                    "status_code": 404,
                }

            absolute_path = self.file_storage_service.resolve_absolute_image_path(
                image.file_path
            )

            if not self.file_storage_service.file_exists(absolute_path):
                return {
                    "success": False,
                    "message": "Image file not found",
                    "status_code": 404,
                }

            mimetype = self.file_storage_service.guess_mimetype(absolute_path)

            return {
                "success": True,
                "message": "Image file loaded successfully.",
                "status_code": 200,
                "absolute_file_path": absolute_path,
                "mimetype": mimetype,
            }
        except Exception as exc:
            logger.error(
                "Error serving image_id=%s: %s",
                image_id,
                exc,
                exc_info=True,
            )
            return {
                "success": False,
                "message": "Internal Server Error",
                "status_code": 500,
            }
        finally:
            session.close()

    def _build_view_data_for_failure(
        self,
        *,
        session,
        part,
        part_id: int,
        search_query: str,
    ) -> Dict[str, Any]:
        try:
            current_part = part or self.part_service.get_part_by_id(
                session=session,
                part_id=part_id,
            )
            part_images = self.part_image_service.get_images_for_part(
                session=session,
                part_id=part_id,
            )
            positions = self.position_service.get_all_positions(session=session)

            return {
                "part": current_part,
                "part_images": part_images,
                "positions": positions,
                "search_query": search_query,
            }
        except Exception:
            return {
                "part": None,
                "part_images": [],
                "positions": [],
                "search_query": search_query,
            }