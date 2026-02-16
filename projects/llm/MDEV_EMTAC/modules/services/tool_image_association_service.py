from typing import Optional, List, Tuple

from sqlalchemy.orm import Session as SASession

from modules.configuration.config_env import DatabaseConfig
from modules.configuration.log_config import (
    debug_id,
    info_id,
    error_id,
    with_request_id,
    log_timed_operation,
    get_request_id,
)

from modules.emtacdb.emtacdb_fts import (
    ToolImageAssociation,
    Image,
)


class ToolImageAssociationService:
    """
    Service layer for ToolImageAssociation.

    Responsibilities:
    - Associate / dissociate images and tools
    - Query images by tool
    - Query tools by image
    """

    def __init__(self, db_config: Optional[DatabaseConfig] = None):
        self.db_config = db_config or DatabaseConfig()

    def _get_session(self, session: Optional[SASession]) -> Tuple[SASession, bool]:
        if session is not None:
            return session, False
        return self.db_config.get_main_session(), True

    @with_request_id
    def associate(
        self,
        image_id: int,
        tool_id: int,
        description: Optional[str] = None,
        session: Optional[SASession] = None,
    ) -> Optional[ToolImageAssociation]:
        rid = get_request_id()
        sess, created_here = self._get_session(session)

        try:
            with log_timed_operation("ToolImageAssociationService.associate", rid):
                return ToolImageAssociation.associate_with_tool(
                    sess,
                    image_id=image_id,
                    tool_id=tool_id,
                    description=description,
                    request_id=rid,
                )
        except Exception as e:
            error_id(f"Error associating image {image_id} with tool {tool_id}: {e}", rid, exc_info=True)
            return None
        finally:
            if created_here:
                sess.close()

    @with_request_id
    def get_images_for_tool(
        self,
        tool_id: int,
        session: Optional[SASession] = None,
    ) -> List[dict]:
        rid = get_request_id()
        sess, created_here = self._get_session(session)

        try:
            with log_timed_operation("ToolImageAssociationService.get_images_for_tool", rid):
                return ToolImageAssociation.get_images_for_tool(
                    sess,
                    tool_id=tool_id,
                    request_id=rid,
                )
        finally:
            if created_here:
                sess.close()

    @with_request_id
    def get_tools_for_image(
        self,
        image_id: int,
        session: Optional[SASession] = None,
    ) -> List[dict]:
        rid = get_request_id()
        sess, created_here = self._get_session(session)

        try:
            with log_timed_operation("ToolImageAssociationService.get_tools_for_image", rid):
                return ToolImageAssociation.get_tools_for_image(
                    sess,
                    image_id=image_id,
                    request_id=rid,
                )
        finally:
            if created_here:
                sess.close()
