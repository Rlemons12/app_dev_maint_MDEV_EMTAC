import logging
from datetime import date, datetime
from uuid import uuid4

from flask import flash, redirect, render_template, request, session, url_for
from modules.emtacdb.emtacdb_fts import User
from modules.emtacdb.emtacdb_fts import UserLevel
from modules.orchestrators.login_session_orchestrator import LoginSessionOrchestrator
from modules.services.login_session_dtos import (
    LoginRequestData,
    LogoutRequestData,
    TabletIdentity,
)


logger = logging.getLogger(__name__)


class LoginSessionCoordinator:
    """
    Handles Flask-specific login/logout behavior.

    This coordinator intentionally guards tablet identity so a desktop browser
    does not accidentally inherit tablet_uid/tablet_name after someone manually
    pastes a tablet login URL into the desktop browser.
    """

    TABLET_SESSION_KEYS = [
        "tablet_uid",
        "tablet_name",
    ]

    def __init__(self, db_config):
        self.orchestrator = LoginSessionOrchestrator(db_config=db_config)

    def handle_login(self):
        if self.should_clear_tablet_identity():
            self.clear_tablet_identity_from_session()

        tablet_identity = self.store_tablet_identity_from_request()

        if request.method == "GET":
            return render_template(
                "login.html",
                tablet_uid=tablet_identity.tablet_uid,
                tablet_name=tablet_identity.tablet_name,
                is_tablet_client=self.is_tablet_client(),
            )

        employee_id = request.form.get("employee_id", "").strip()
        password = request.form.get("password", "")

        if not employee_id or not password:
            flash("Invalid username or password", "error")
            return render_template(
                "login.html",
                tablet_uid=tablet_identity.tablet_uid,
                tablet_name=tablet_identity.tablet_name,
                is_tablet_client=self.is_tablet_client(),
            )

        logger.info("Login attempt for employee_id: %s", employee_id)

        session_tracking_id = self.get_or_create_emtac_session_tracking_id()

        login_data = LoginRequestData(
            employee_id=employee_id,
            password=password,
            tablet_identity=tablet_identity,
            remote_addr=request.remote_addr,
            user_agent=request.user_agent.string if request.user_agent else None,
            current_page_url=request.referrer or request.url or request.path,
        )

        result = self.orchestrator.login(
            login_data=login_data,
            session_tracking_id=session_tracking_id,
        )

        if not result.success:
            flash(result.message or "Invalid username or password", "error")
            return render_template(
                "login.html",
                tablet_uid=tablet_identity.tablet_uid,
                tablet_name=tablet_identity.tablet_name,
                is_tablet_client=self.is_tablet_client(),
            )

        self.store_authenticated_user_in_session(result)

        logger.info(
            "User %s authenticated successfully. tablet_session_started=%s",
            result.employee_id,
            result.tablet_user_session_started,
        )

        return self.redirect_after_login(result.user_level)

    def handle_logout(self):
        logger.info("Logging out user.")

        tablet_identity = self.get_tablet_identity_from_request_or_session()

        logout_data = LogoutRequestData(
            login_record_id=session.get("login_record_id"),
            emtac_session_tracking_id=session.get("emtac_session_tracking_id"),
            tablet_identity=tablet_identity,
        )

        self.orchestrator.logout(logout_data=logout_data)

        preserved_tablet_uid = tablet_identity.tablet_uid
        preserved_tablet_name = tablet_identity.tablet_name
        should_preserve_tablet_identity = self.is_tablet_client()

        session.clear()

        login_url_values = {}

        if should_preserve_tablet_identity and preserved_tablet_uid:
            session["tablet_uid"] = preserved_tablet_uid
            login_url_values["tablet_uid"] = preserved_tablet_uid

        if should_preserve_tablet_identity and preserved_tablet_name:
            session["tablet_name"] = preserved_tablet_name
            login_url_values["tablet_name"] = preserved_tablet_name

        return redirect(url_for("login_bp.login", **login_url_values))

    def handle_activity_tracking(self):
        if "user_id" not in session:
            return None

        if request.path.startswith("/static/"):
            return None

        self.orchestrator.touch_activity(
            login_record_id=session.get("login_record_id"),
            session_tracking_id=session.get("emtac_session_tracking_id"),
            tablet_uid=session.get("tablet_uid"),
            current_page_url=request.url or request.path,
            last_ip_address=request.remote_addr,
        )

        return None

    def get_or_create_emtac_session_tracking_id(self) -> str:
        existing_id = session.get("emtac_session_tracking_id")

        if existing_id:
            return existing_id

        new_id = str(uuid4())
        session["emtac_session_tracking_id"] = new_id

        return new_id

    def should_clear_tablet_identity(self) -> bool:
        """
        Allows manual cleanup using:

            /login?clear_tablet=1

        Also clears tablet identity when a desktop browser hits /login without
        a fresh tablet_uid/tablet_name from an Android/WebView client.
        """
        clear_requested = request.args.get("clear_tablet") in {"1", "true", "yes"}

        if clear_requested:
            return True

        if self.is_tablet_client():
            return False

        incoming_tablet_uid = request.values.get("tablet_uid")
        incoming_tablet_name = request.values.get("tablet_name")

        if incoming_tablet_uid or incoming_tablet_name:
            logger.warning(
                "Ignoring tablet identity on non-tablet client. user_agent=%s",
                request.user_agent.string if request.user_agent else None,
            )
            return True

        if session.get("tablet_uid") or session.get("tablet_name"):
            logger.info(
                "Clearing stale tablet identity from non-tablet browser session."
            )
            return True

        return False

    def is_tablet_client(self) -> bool:
        """
        Detects Android tablet/WebView clients.

        This prevents a normal desktop browser from accidentally becoming
        associated with a physical tablet after someone pastes a tablet URL
        into the desktop browser.
        """
        user_agent = request.user_agent.string if request.user_agent else ""
        user_agent_lower = user_agent.lower()

        android_markers = [
            "android",
            "wv",
            "webview",
            "com.example.emtactablet",
            "emtac",
        ]

        return any(marker in user_agent_lower for marker in android_markers)

    def clear_tablet_identity_from_session(self) -> None:
        for key in self.TABLET_SESSION_KEYS:
            session.pop(key, None)

    def get_tablet_identity_from_request_or_session(self) -> TabletIdentity:
        if not self.is_tablet_client():
            return TabletIdentity()

        tablet_uid = request.values.get("tablet_uid") or session.get("tablet_uid")
        tablet_name = request.values.get("tablet_name") or session.get("tablet_name")

        if tablet_uid:
            tablet_uid = str(tablet_uid).strip()

        if tablet_name:
            tablet_name = str(tablet_name).strip()

        return TabletIdentity(
            tablet_uid=tablet_uid or None,
            tablet_name=tablet_name or None,
        )

    def store_tablet_identity_from_request(self) -> TabletIdentity:
        tablet_identity = self.get_tablet_identity_from_request_or_session()

        if tablet_identity.tablet_uid:
            session["tablet_uid"] = tablet_identity.tablet_uid

        if tablet_identity.tablet_name:
            session["tablet_name"] = tablet_identity.tablet_name

        return tablet_identity

    def store_authenticated_user_in_session(self, result) -> None:
        """
        Store authenticated user data in the Flask session.

        Important:
            The frontend chatbot reads session["user_id"] through the base template.
            If user_id is missing, Q&A feedback falls back to "anonymous".

        This method stores result.user_id when present and falls back to resolving
        users.id by employee_id if needed.
        """

        resolved_user_id = self.resolve_user_id_for_session(result)

        session["login_record_id"] = result.login_record_id
        session["emtac_session_tracking_id"] = result.emtac_session_tracking_id

        session["user_id"] = str(resolved_user_id) if resolved_user_id is not None else ""
        session["employee_id"] = str(result.employee_id) if result.employee_id is not None else ""
        session["first_name"] = result.first_name or ""
        session["last_name"] = result.last_name or ""
        session["primary_area"] = result.primary_area or ""
        session["age"] = result.age
        session["education_level"] = result.education_level or ""
        session["start_date"] = self.make_session_safe(result.start_date)
        session["user_level"] = self.get_user_level_name(result.user_level)
        session["login_time"] = result.login_time

        logger.info(
            "[LoginSessionCoordinator] Stored authenticated user in session: "
            "result_user_id=%s resolved_user_id=%s session_user_id=%s "
            "employee_id=%s first_name=%s last_name=%s user_level=%s "
            "login_record_id=%s emtac_session_tracking_id=%s",
            getattr(result, "user_id", None),
            resolved_user_id,
            session.get("user_id"),
            session.get("employee_id"),
            session.get("first_name"),
            session.get("last_name"),
            session.get("user_level"),
            session.get("login_record_id"),
            session.get("emtac_session_tracking_id"),
        )
        session["login_record_id"] = result.login_record_id
        session["emtac_session_tracking_id"] = result.emtac_session_tracking_id

        session["user_id"] = result.user_id
        session["employee_id"] = result.employee_id
        session["first_name"] = result.first_name
        session["last_name"] = result.last_name
        session["primary_area"] = result.primary_area
        session["age"] = result.age
        session["education_level"] = result.education_level
        session["start_date"] = self.make_session_safe(result.start_date)
        session["user_level"] = self.get_user_level_name(result.user_level)
        session["login_time"] = result.login_time

    def make_session_safe(self, value):
        if isinstance(value, datetime):
            return value.isoformat()

        if isinstance(value, date):
            return value.isoformat()

        return value

    def get_user_level_name(self, user_level) -> str:
        if hasattr(user_level, "name"):
            return user_level.name

        if hasattr(user_level, "value"):
            return user_level.value

        return str(user_level)

    def redirect_after_login(self, user_level):
        user_level_name = self.get_user_level_name(user_level)

        if user_level_name == UserLevel.ADMIN.name:
            return redirect(url_for("admin_bp.admin_dashboard"))

        if user_level_name == UserLevel.STANDARD.name:
            return redirect(url_for("upload_image_page"))

        return redirect(url_for("index"))

    def resolve_user_id_for_session(self, result):
        """
        Resolve the database users.id value for Flask session storage.

        Priority:
            1. result.user_id from LoginResult
            2. lookup users.id by result.employee_id
            3. None
        """

        result_user_id = getattr(result, "user_id", None)

        if result_user_id is not None and str(result_user_id).strip() != "":
            return result_user_id

        employee_id = getattr(result, "employee_id", None)

        if employee_id is None or str(employee_id).strip() == "":
            logger.warning(
                "[LoginSessionCoordinator] Cannot resolve user_id because employee_id is missing. result=%s",
                result,
            )
            return None

        db_session = self.orchestrator.db_config.get_main_session()

        try:
            user = (
                db_session.query(User)
                .filter(User.employee_id == str(employee_id).strip())
                .first()
            )

            if not user:
                logger.warning(
                    "[LoginSessionCoordinator] Could not resolve user_id from employee_id=%s",
                    employee_id,
                )
                return None

            logger.info(
                "[LoginSessionCoordinator] Resolved missing user_id from employee_id. "
                "employee_id=%s user_id=%s",
                employee_id,
                user.id,
            )

            return user.id

        except Exception as exc:
            logger.exception(
                "[LoginSessionCoordinator] Failed resolving user_id from employee_id=%s: %s",
                employee_id,
                exc,
            )
            return None

        finally:
            db_session.close()