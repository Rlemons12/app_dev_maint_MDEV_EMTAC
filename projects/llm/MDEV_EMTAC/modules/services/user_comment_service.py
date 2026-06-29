from __future__ import annotations

from typing import Any, Dict, List, Optional

from sqlalchemy import desc

from modules.configuration.log_config import logger
from modules.emtacdb.emtacdb_fts import User, UserComments


class UserCommentService:
    """
    Service layer for user comments.

    Rules:
        - No session creation here.
        - No commit or rollback here.
        - The orchestrator owns the transaction.
        - This service only validates, normalizes, queries, and creates ORM objects.
    """

    MAX_COMMENT_LENGTH = 10000
    MAX_PAGE_URL_LENGTH = 2048
    MAX_SCREENSHOT_PATH_LENGTH = 1024

    def create_comment(
        self,
        *,
        session,
        user_id: Optional[Any],
        comment: str,
        page_url: str,
        screenshot_path: Optional[str] = None,
    ) -> UserComments:
        clean_comment = self._normalize_required_text(
            value=comment,
            field_name="comment",
            max_length=self.MAX_COMMENT_LENGTH,
        )

        clean_page_url = self._normalize_required_text(
            value=page_url,
            field_name="page_url",
            max_length=self.MAX_PAGE_URL_LENGTH,
        )

        clean_screenshot_path = self._normalize_optional_text(
            value=screenshot_path,
            max_length=self.MAX_SCREENSHOT_PATH_LENGTH,
        )

        resolved_user_id = self._resolve_user_id(
            session=session,
            raw_user_id=user_id,
        )

        user_comment = UserComments(
            user_id=resolved_user_id,
            comment=clean_comment,
            page_url=clean_page_url,
            screenshot_path=clean_screenshot_path,
        )

        session.add(user_comment)
        session.flush()

        logger.info(
            "[UserCommentService] Created user comment row id=%s user_id=%s page_url=%s",
            user_comment.id,
            user_comment.user_id,
            user_comment.page_url,
        )

        return user_comment

    def get_comment_by_id(
        self,
        *,
        session,
        comment_id: Any,
    ) -> Optional[UserComments]:
        normalized_id = self._normalize_positive_int(
            value=comment_id,
            field_name="comment_id",
        )

        return (
            session.query(UserComments)
            .filter(UserComments.id == normalized_id)
            .first()
        )

    def list_comments(
        self,
        *,
        session,
        user_id: Optional[Any] = None,
        page_url: Optional[str] = None,
        limit: int = 50,
        offset: int = 0,
    ) -> List[UserComments]:
        clean_limit = self._clamp_int(limit, default=50, minimum=1, maximum=200)
        clean_offset = self._clamp_int(offset, default=0, minimum=0, maximum=100000)

        query = session.query(UserComments)

        has_user_filter = user_id not in (None, "", "anonymous", "null", "none")
        if has_user_filter:
            resolved_user_id = self._resolve_user_id(
                session=session,
                raw_user_id=user_id,
            )

            if resolved_user_id is None:
                logger.info(
                    "[UserCommentService] list_comments user filter did not resolve. raw_user_id=%s",
                    user_id,
                )
                return []

            query = query.filter(UserComments.user_id == resolved_user_id)

        clean_page_url = self._normalize_optional_text(
            value=page_url,
            max_length=self.MAX_PAGE_URL_LENGTH,
        )

        if clean_page_url:
            query = query.filter(UserComments.page_url == clean_page_url)

        return (
            query.order_by(desc(UserComments.timestamp), desc(UserComments.id))
            .offset(clean_offset)
            .limit(clean_limit)
            .all()
        )

    def to_dict(self, user_comment: UserComments) -> Dict[str, Any]:
        if hasattr(user_comment, "to_dict"):
            return user_comment.to_dict()

        return {
            "id": user_comment.id,
            "user_id": user_comment.user_id,
            "comment": user_comment.comment,
            "page_url": user_comment.page_url,
            "screenshot_path": user_comment.screenshot_path,
            "timestamp": (
                user_comment.timestamp.isoformat()
                if user_comment.timestamp
                else None
            ),
        }

    def to_dict_list(self, comments: List[UserComments]) -> List[Dict[str, Any]]:
        return [self.to_dict(item) for item in comments]

    def _resolve_user_id(
        self,
        *,
        session,
        raw_user_id: Optional[Any],
    ) -> Optional[int]:
        """
        UserComments.user_id points to users.id, which is an integer.

        This helper accepts:
            - None / anonymous values -> None
            - integer user id -> validates against users.id
            - numeric string -> validates against users.id
            - non-numeric string -> tries users.employee_id

        If nothing matches, it returns None because user_id is nullable.
        """

        if raw_user_id in (None, "", "anonymous", "null", "none"):
            return None

        raw_text = str(raw_user_id).strip()

        if not raw_text:
            return None

        if raw_text.lower() in {"anonymous", "null", "none"}:
            return None

        if raw_text.isdigit():
            user_pk = int(raw_text)
            existing_user = (
                session.query(User.id)
                .filter(User.id == user_pk)
                .first()
            )

            if existing_user:
                return user_pk

            logger.warning(
                "[UserCommentService] Numeric user_id did not match users.id. raw_user_id=%s",
                raw_user_id,
            )
            return None

        employee_user = (
            session.query(User.id)
            .filter(User.employee_id == raw_text)
            .first()
        )

        if employee_user:
            return int(employee_user.id)

        logger.warning(
            "[UserCommentService] Non-numeric user_id did not match users.employee_id. raw_user_id=%s",
            raw_user_id,
        )

        return None

    def _normalize_required_text(
        self,
        *,
        value: Any,
        field_name: str,
        max_length: int,
    ) -> str:
        if value is None:
            raise ValueError(f"{field_name} is required.")

        text_value = str(value).strip()

        if not text_value:
            raise ValueError(f"{field_name} is required.")

        if len(text_value) > max_length:
            raise ValueError(
                f"{field_name} is too long. Maximum length is {max_length} characters."
            )

        return text_value

    def _normalize_optional_text(
        self,
        *,
        value: Optional[Any],
        max_length: int,
    ) -> Optional[str]:
        if value is None:
            return None

        text_value = str(value).strip()

        if not text_value:
            return None

        if len(text_value) > max_length:
            raise ValueError(
                f"Optional text value is too long. Maximum length is {max_length} characters."
            )

        return text_value

    def _normalize_positive_int(
        self,
        *,
        value: Any,
        field_name: str,
    ) -> int:
        try:
            normalized = int(value)
        except (TypeError, ValueError):
            raise ValueError(f"{field_name} must be a valid integer.")

        if normalized <= 0:
            raise ValueError(f"{field_name} must be greater than zero.")

        return normalized

    def _clamp_int(
        self,
        value: Any,
        *,
        default: int,
        minimum: int,
        maximum: int,
    ) -> int:
        try:
            normalized = int(value)
        except (TypeError, ValueError):
            normalized = default

        if normalized < minimum:
            return minimum

        if normalized > maximum:
            return maximum

        return normalized