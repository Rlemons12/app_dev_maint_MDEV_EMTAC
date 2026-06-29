from __future__ import annotations

from typing import Any, Dict, Optional

from modules.configuration.log_config import logger
from modules.orchestrators.user_comment_orchestrator import UserCommentOrchestrator


class UserCommentCoordinator:
    """
    Coordinator layer for user comments.

    Rules:
        - The route talks to this class.
        - This class normalizes route-level naming differences.
        - The orchestrator handles transaction/session ownership.
    """

    def __init__(
        self,
        *,
        orchestrator: Optional[UserCommentOrchestrator] = None,
    ):
        self.orchestrator = orchestrator or UserCommentOrchestrator()

    def submit_comment(
        self,
        *,
        user_id: Optional[Any],
        comment: Optional[str],
        page_url: Optional[str],
        screenshot_path: Optional[str] = None,
        request_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        logger.info(
            "[UserCommentCoordinator] submit_comment request_id=%s user_id=%s page_url=%s",
            request_id,
            user_id,
            page_url,
        )

        return self.orchestrator.submit_comment(
            user_id=user_id,
            comment=comment or "",
            page_url=page_url or "",
            screenshot_path=screenshot_path,
            request_id=request_id,
        )

    def get_comment(
        self,
        *,
        comment_id: Any,
        request_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        logger.info(
            "[UserCommentCoordinator] get_comment request_id=%s comment_id=%s",
            request_id,
            comment_id,
        )

        return self.orchestrator.get_comment(
            comment_id=comment_id,
            request_id=request_id,
        )

    def list_comments(
        self,
        *,
        user_id: Optional[Any] = None,
        page_url: Optional[str] = None,
        limit: int = 50,
        offset: int = 0,
        request_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        logger.info(
            "[UserCommentCoordinator] list_comments request_id=%s user_id=%s page_url=%s limit=%s offset=%s",
            request_id,
            user_id,
            page_url,
            limit,
            offset,
        )

        return self.orchestrator.list_comments(
            user_id=user_id,
            page_url=page_url,
            limit=limit,
            offset=offset,
            request_id=request_id,
        )