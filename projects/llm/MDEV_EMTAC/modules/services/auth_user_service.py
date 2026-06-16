import logging
from datetime import datetime
from typing import Optional

from modules.emtacdb.emtacdb_fts import ChatSession, User, UserLogin

logger = logging.getLogger(__name__)


class AuthUserService:
    """
    Handles normal EMTAC user authentication and UserLogin tracking.

    This service does not touch Flask request/session objects.
    """

    @staticmethod
    def authenticate_user(db_session, employee_id: str, password: str) -> Optional[User]:
        employee_id = (employee_id or "").strip()

        if not employee_id or not password:
            return None

        user = db_session.query(User).filter_by(employee_id=employee_id).first()

        if not user:
            return None

        if not user.check_password_hash(password):
            return None

        return user

    @staticmethod
    def create_chat_session(db_session, user_id: int, current_time: str) -> ChatSession:
        new_chat_session = ChatSession(
            user_id=str(user_id),
            start_time=current_time,
            last_interaction=current_time,
            session_data=[],
        )

        db_session.add(new_chat_session)
        return new_chat_session

    @staticmethod
    def create_user_login(
        db_session,
        user_id: int,
        session_tracking_id: str,
        ip_address: Optional[str],
        user_agent: Optional[str],
    ) -> UserLogin:
        user_login = UserLogin(
            user_id=user_id,
            session_id=session_tracking_id,
            ip_address=ip_address,
            user_agent=user_agent,
        )

        db_session.add(user_login)

        # Assign user_login.id before the route stores it in Flask session.
        db_session.flush()

        return user_login

    @staticmethod
    def close_user_login(db_session, login_record_id: Optional[int]) -> bool:
        if not login_record_id:
            return False

        login_record = db_session.get(UserLogin, login_record_id)

        if not login_record:
            return False

        login_record.logout_time = datetime.utcnow()
        login_record.is_active = False

        return True

    @staticmethod
    def build_user_display_name(user: User) -> str:
        display_name = " ".join(
            part
            for part in [
                getattr(user, "first_name", None),
                getattr(user, "last_name", None),
            ]
            if part
        ).strip()

        if display_name:
            return display_name

        employee_id = getattr(user, "employee_id", None)

        if employee_id:
            return str(employee_id)

        return f"User {getattr(user, 'id', '')}".strip()

    @staticmethod
    def get_login_time_string() -> str:
        return datetime.now().strftime("%Y-%m-%d %H:%M:%S")