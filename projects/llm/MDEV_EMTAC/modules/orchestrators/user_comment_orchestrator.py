from __future__ import annotations

from typing import Any, Dict, Optional

from sqlalchemy.exc import SQLAlchemyError

from modules.configuration.config_env import get_db_config
from modules.configuration.log_config import logger
from modules.services.user_comment_service import UserCommentService


class UserCommentOrchestrator:
    """
    Orchestrator layer for user comments.

    Rules:
        - Owns session lifecycle.
        - Owns commit / rollback.
        - Calls service for validation and database operations.
        - Returns API-safe dictionaries to the coordinator.
    """

    def __init__(
        self,
        *,
        db_config=None,
        service: Optional[UserCommentService] = None,
    ):
        self.db_config = db_config or get_db_config()
        self.service = service or UserCommentService()

    def submit_comment(
        self,
        *,
        user_id: Optional[Any],
        comment: str,
        page_url: str,
        screenshot_path: Optional[str] = None,
        request_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        session = None

        try:
            logger.info(
                "[UserCommentOrchestrator] submit_comment started request_id=%s user_id=%s page_url=%s",
                request_id,
                user_id,
                page_url,
            )

            session = self.db_config.get_main_session()

            user_comment = self.service.create_comment(
                session=session,
                user_id=user_id,
                comment=comment,
                page_url=page_url,
                screenshot_path=screenshot_path,
            )

            session.commit()
            session.refresh(user_comment)

            payload = self.service.to_dict(user_comment)

            logger.info(
                "[UserCommentOrchestrator] submit_comment success request_id=%s comment_id=%s",
                request_id,
                payload.get("id"),
            )

            return {
                "status": "success",
                "message": "Comment saved.",
                "comment": payload,
                "request_id": request_id,
            }

        except ValueError as exc:
            if session is not None:
                session.rollback()

            logger.warning(
                "[UserCommentOrchestrator] submit_comment invalid input request_id=%s error=%s",
                request_id,
                exc,
            )

            return {
                "status": "invalid_input",
                "message": str(exc),
                "comment": None,
                "request_id": request_id,
            }

        except SQLAlchemyError as exc:
            if session is not None:
                session.rollback()

            logger.error(
                "[UserCommentOrchestrator] submit_comment database failure request_id=%s error=%s",
                request_id,
                exc,
                exc_info=True,
            )

            return {
                "status": "error",
                "message": "Database error while saving comment.",
                "comment": None,
                "request_id": request_id,
            }

        except Exception as exc:
            if session is not None:
                session.rollback()

            logger.error(
                "[UserCommentOrchestrator] submit_comment unexpected failure request_id=%s error=%s",
                request_id,
                exc,
                exc_info=True,
            )

            return {
                "status": "error",
                "message": "Unexpected error while saving comment.",
                "comment": None,
                "request_id": request_id,
            }

        finally:
            if session is not None:
                session.close()

    def get_comment(
        self,
        *,
        comment_id: Any,
        request_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        session = None

        try:
            session = self.db_config.get_main_session()

            user_comment = self.service.get_comment_by_id(
                session=session,
                comment_id=comment_id,
            )

            if user_comment is None:
                return {
                    "status": "not_found",
                    "message": "Comment not found.",
                    "comment": None,
                    "request_id": request_id,
                }

            return {
                "status": "success",
                "message": "Comment found.",
                "comment": self.service.to_dict(user_comment),
                "request_id": request_id,
            }

        except ValueError as exc:
            logger.warning(
                "[UserCommentOrchestrator] get_comment invalid input request_id=%s error=%s",
                request_id,
                exc,
            )

            return {
                "status": "invalid_input",
                "message": str(exc),
                "comment": None,
                "request_id": request_id,
            }

        except Exception as exc:
            logger.error(
                "[UserCommentOrchestrator] get_comment failure request_id=%s error=%s",
                request_id,
                exc,
                exc_info=True,
            )

            return {
                "status": "error",
                "message": "Unexpected error while loading comment.",
                "comment": None,
                "request_id": request_id,
            }

        finally:
            if session is not None:
                session.close()

    def list_comments(
        self,
        *,
        user_id: Optional[Any] = None,
        page_url: Optional[str] = None,
        limit: int = 50,
        offset: int = 0,
        request_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        session = None

        try:
            session = self.db_config.get_main_session()

            comments = self.service.list_comments(
                session=session,
                user_id=user_id,
                page_url=page_url,
                limit=limit,
                offset=offset,
            )

            comment_items = self.service.to_dict_list(comments)

            return {
                "status": "success",
                "message": "Comments loaded.",
                "comments": comment_items,
                "count": len(comment_items),
                "limit": limit,
                "offset": offset,
                "request_id": request_id,
            }

        except ValueError as exc:
            logger.warning(
                "[UserCommentOrchestrator] list_comments invalid input request_id=%s error=%s",
                request_id,
                exc,
            )

            return {
                "status": "invalid_input",
                "message": str(exc),
                "comments": [],
                "count": 0,
                "limit": limit,
                "offset": offset,
                "request_id": request_id,
            }

        except Exception as exc:
            logger.error(
                "[UserCommentOrchestrator] list_comments failure request_id=%s error=%s",
                request_id,
                exc,
                exc_info=True,
            )

            return {
                "status": "error",
                "message": "Unexpected error while loading comments.",
                "comments": [],
                "count": 0,
                "limit": limit,
                "offset": offset,
                "request_id": request_id,
            }

        finally:
            if session is not None:
                session.close()