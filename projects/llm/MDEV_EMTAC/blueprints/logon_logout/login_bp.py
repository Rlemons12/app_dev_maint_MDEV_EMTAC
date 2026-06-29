import logging
from datetime import datetime

from flask import Blueprint, flash, redirect, render_template, request, url_for
from sqlalchemy import text

from modules.configuration.config_env import get_db_config
from modules.coordinators.login_session_coordinator import LoginSessionCoordinator


logger = logging.getLogger(__name__)

db_config = get_db_config()

login_bp = Blueprint("login_bp", __name__)


def _ensure_password_reset_request_table(session_db):
    session_db.execute(text("""
        CREATE TABLE IF NOT EXISTS password_reset_requests (
            id BIGSERIAL PRIMARY KEY,
            employee_id VARCHAR NOT NULL,
            request_note TEXT NULL,
            status VARCHAR NOT NULL DEFAULT 'PENDING',
            requested_at TIMESTAMP NOT NULL,
            resolved_at TIMESTAMP NULL,
            resolved_by_user_id INTEGER NULL
        )
    """))


@login_bp.route("/login", methods=["GET", "POST"])
def login():
    coordinator = LoginSessionCoordinator(db_config=db_config)
    return coordinator.handle_login()


@login_bp.route("/logout", methods=["GET", "POST"])
def logout():
    coordinator = LoginSessionCoordinator(db_config=db_config)
    return coordinator.handle_logout()


@login_bp.route("/request_password_reset", methods=["GET", "POST"])
def request_password_reset():
    if request.method == "GET":
        return render_template("request_password_reset.html")

    employee_id = (request.form.get("employee_id") or "").strip()
    request_note = (request.form.get("request_note") or "").strip()

    if not employee_id:
        flash("Employee ID is required.", "error")
        return redirect(url_for("login_bp.request_password_reset"))

    session_db = None

    try:
        session_db = db_config.get_main_session()
        _ensure_password_reset_request_table(session_db)

        existing_request = session_db.execute(
            text("""
                SELECT id
                FROM password_reset_requests
                WHERE employee_id = :employee_id
                  AND status = 'PENDING'
                ORDER BY requested_at DESC
                LIMIT 1
            """),
            {"employee_id": employee_id},
        ).fetchone()

        if existing_request:
            flash("A password reset request is already pending for that Employee ID.", "warning")
            session_db.commit()
            return redirect(url_for("login_bp.login"))

        session_db.execute(
            text("""
                INSERT INTO password_reset_requests (
                    employee_id,
                    request_note,
                    status,
                    requested_at
                )
                VALUES (
                    :employee_id,
                    :request_note,
                    'PENDING',
                    :requested_at
                )
            """),
            {
                "employee_id": employee_id,
                "request_note": request_note or None,
                "requested_at": datetime.now(),
            },
        )

        session_db.commit()

        flash("Password reset request submitted. Please contact an admin.", "success")
        return redirect(url_for("login_bp.login"))

    except Exception as e:
        if session_db:
            session_db.rollback()

        logger.error("Error submitting password reset request: %s", e, exc_info=True)
        flash(f"Error submitting password reset request: {e}", "error")
        return redirect(url_for("login_bp.request_password_reset"))

    finally:
        if session_db:
            session_db.close()


def activity_tracker():
    coordinator = LoginSessionCoordinator(db_config=db_config)
    return coordinator.handle_activity_tracking()