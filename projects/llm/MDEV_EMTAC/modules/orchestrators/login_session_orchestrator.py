import logging
from typing import Optional

from modules.services.login_auth_user_service import LoginAuthUserService
from modules.services.login_session_dtos import (
    LoginRequestData,
    LoginResult,
    LogoutRequestData,
    LogoutResult,
)
from modules.services.tablet_edge.tablet_user_session_service import TabletUserSessionService


logger = logging.getLogger(__name__)


class LoginSessionOrchestrator:
    """
    Owns the login/logout transaction flow.

    Coordinator:
        Handles Flask request/session/render/redirect behavior.

    Orchestrator:
        Handles transaction sequencing and commit/rollback.

    Services:
        Handle focused database operations.
    """

    def __init__(self, db_config):
        self.db_config = db_config

    def login(
        self,
        login_data: LoginRequestData,
        session_tracking_id: str,
    ) -> LoginResult:
        db_session = self.db_config.get_main_session()

        try:
            user = LoginAuthUserService.authenticate_user(
                db_session=db_session,
                employee_id=login_data.employee_id,
                password=login_data.password,
            )

            if not user:
                db_session.rollback()
                return LoginResult(
                    success=False,
                    message="Invalid username or password",
                )

            current_time = LoginAuthUserService.get_login_time_string()

            LoginAuthUserService.create_chat_session(
                db_session=db_session,
                user_id=user.id,
                current_time=current_time,
            )

            user_login = LoginAuthUserService.create_user_login(
                db_session=db_session,
                user_id=user.id,
                session_tracking_id=session_tracking_id,
                ip_address=login_data.remote_addr,
                user_agent=login_data.user_agent,
            )

            tablet_user_session_started = False
            tablet_uid = login_data.tablet_identity.tablet_uid

            if tablet_uid:
                TabletUserSessionService.start_user_session(
                    db_session=db_session,
                    tablet_uid=tablet_uid,
                    user_id=user.id,
                    username=getattr(user, "employee_id", None),
                    display_name=LoginAuthUserService.build_user_display_name(user),
                    session_id=session_tracking_id,
                    login_ip_address=login_data.remote_addr,
                    user_agent=login_data.user_agent,
                    current_page_url=login_data.current_page_url,
                    close_existing_for_tablet=True,
                )

                tablet_user_session_started = True

                logger.info(
                    "Tablet user session tracking started. "
                    "tablet_uid=%s employee_id=%s session_tracking_id=%s",
                    tablet_uid,
                    user.employee_id,
                    session_tracking_id,
                )
            else:
                logger.info(
                    "No tablet_uid found during login for employee_id=%s. "
                    "Normal login will continue without tablet session tracking.",
                    user.employee_id,
                )

            db_session.commit()

            return LoginResult(
                success=True,
                message="Login successful",
                user_id=user.id,
                employee_id=user.employee_id,
                first_name=user.first_name,
                last_name=user.last_name,
                primary_area=user.primary_area,
                age=user.age,
                education_level=user.education_level,
                start_date=user.start_date,
                user_level=user.user_level,
                login_time=current_time,
                login_record_id=user_login.id,
                emtac_session_tracking_id=session_tracking_id,
                tablet_user_session_started=tablet_user_session_started,
            )

        except Exception as exc:
            db_session.rollback()
            logger.exception("Login orchestration failed: %s", exc)

            return LoginResult(
                success=False,
                message=f"An error occurred: {exc}",
            )

        finally:
            db_session.close()

    def logout(self, logout_data: LogoutRequestData) -> LogoutResult:
        db_session = self.db_config.get_main_session()

        try:
            tablet_uid = logout_data.tablet_identity.tablet_uid
            session_tracking_id = logout_data.emtac_session_tracking_id

            if session_tracking_id or tablet_uid:
                updated_count = TabletUserSessionService.end_user_session(
                    db_session=db_session,
                    session_id=session_tracking_id,
                    tablet_uid=tablet_uid,
                    logout_reason="user_logout",
                )

                logger.info(
                    "Tablet user session logout update complete. "
                    "updated_count=%s session_tracking_id=%s tablet_uid=%s",
                    updated_count,
                    session_tracking_id,
                    tablet_uid,
                )

            LoginAuthUserService.close_user_login(
                db_session=db_session,
                login_record_id=logout_data.login_record_id,
            )

            db_session.commit()

            return LogoutResult(
                success=True,
                message="Logout successful",
                tablet_identity=logout_data.tablet_identity,
            )

        except Exception as exc:
            db_session.rollback()
            logger.exception("Logout orchestration failed: %s", exc)

            return LogoutResult(
                success=False,
                message=f"Logout error: {exc}",
                tablet_identity=logout_data.tablet_identity,
            )

        finally:
            db_session.close()

    def touch_activity(
        self,
        login_record_id: Optional[int],
        session_tracking_id: Optional[str],
        tablet_uid: Optional[str],
        current_page_url: Optional[str],
        last_ip_address: Optional[str],
    ) -> None:
        db_session = self.db_config.get_main_session()

        try:
            LoginAuthUserService.touch_user_login(
                db_session=db_session,
                login_record_id=login_record_id,
            )

            if session_tracking_id or tablet_uid:
                TabletUserSessionService.touch_active_session(
                    db_session=db_session,
                    tablet_uid=tablet_uid,
                    session_id=session_tracking_id,
                    current_page_url=current_page_url,
                    last_ip_address=last_ip_address,
                )

            db_session.commit()

        except Exception as exc:
            db_session.rollback()
            logger.exception("Activity tracking update failed: %s", exc)

        finally:
            db_session.close()