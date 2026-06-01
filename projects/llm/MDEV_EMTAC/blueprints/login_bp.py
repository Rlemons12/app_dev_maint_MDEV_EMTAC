import logging

from flask import Blueprint

from modules.configuration.config_env import get_db_config
from modules.coordinators.login_session_coordinator import LoginSessionCoordinator


logger = logging.getLogger(__name__)

db_config = get_db_config()

login_bp = Blueprint("login_bp", __name__)


@login_bp.route("/login", methods=["GET", "POST"])
def login():
    """
    Handles EMTAC login through the coordinator layer.

    The coordinator handles Flask request/session behavior.
    The orchestrator handles database transaction flow.
    The services handle focused database operations.
    """
    coordinator = LoginSessionCoordinator(db_config=db_config)
    return coordinator.handle_login()


@login_bp.route("/logout", methods=["GET", "POST"])
def logout():
    """
    Handles EMTAC logout through the coordinator layer.

    This also closes the tablet_edge.tablet_user_session row when tablet
    context is available.
    """
    coordinator = LoginSessionCoordinator(db_config=db_config)
    return coordinator.handle_logout()


def activity_tracker():
    """
    Updates normal UserLogin activity and tablet user session last_seen_at.

    Register this with the Flask app as a before_request hook.

    Example:
        app.before_request(activity_tracker)
    """
    coordinator = LoginSessionCoordinator(db_config=db_config)
    return coordinator.handle_activity_tracking()