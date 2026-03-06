# modules/services/tool_image_association_service.py

from typing import Optional, List

from sqlalchemy.orm import Session

from modules.configuration.log_config import (
    error_id,
    with_request_id,
    log_timed_operation,
    get_request_id,
)

from modules.emtacdb.emtacdb_fts import (
    ToolImageAssociation,
)


class ToolImageAssociationService:
    """
    Service layer for ToolImageAssociation.

    ✔ No DatabaseConfig dependency
    ✔ No session ownership
    ✔ No commit / rollback
    ✔ Orchestrator controls transaction
    ✔ Errors propagate upward
    """

    # --------------------------------------------------
    # ASSOCIATE IMAGE ↔ TOOL
    # --------------------------------------------------

    @with_request_id
    def associate(
        self,
        session: Session,
        *,
        image_id: int,
        tool_id: int,
        description: Optional[str] = None,
    ) -> ToolImageAssociation:

        rid = get_request_id()

        try:
            with log_timed_operation(
                "ToolImageAssociationService.associate",
                rid,
            ):
                return ToolImageAssociation.associate_with_tool(
                    session,
                    image_id=image_id,
                    tool_id=tool_id,
                    description=description,
                    request_id=rid,
                )

        except Exception as e:
            error_id(
                f"Error associating image {image_id} with tool {tool_id}: {e}",
                rid,
                exc_info=True,
            )
            raise  # 🚨 do NOT swallow errors

    # --------------------------------------------------
    # GET IMAGES FOR TOOL
    # --------------------------------------------------

    @with_request_id
    def get_images_for_tool(
        self,
        session: Session,
        tool_id: int,
    ) -> List[dict]:

        rid = get_request_id()

        with log_timed_operation(
            "ToolImageAssociationService.get_images_for_tool",
            rid,
        ):
            return ToolImageAssociation.get_images_for_tool(
                session,
                tool_id=tool_id,
                request_id=rid,
            )

    # --------------------------------------------------
    # GET TOOLS FOR IMAGE
    # --------------------------------------------------

    @with_request_id
    def get_tools_for_image(
        self,
        session: Session,
        image_id: int,
    ) -> List[dict]:

        rid = get_request_id()

        with log_timed_operation(
            "ToolImageAssociationService.get_tools_for_image",
            rid,
        ):
            return ToolImageAssociation.get_tools_for_image(
                session,
                image_id=image_id,
                request_id=rid,
            )
