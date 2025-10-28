# auth_utils.py
from flask import session, redirect, url_for, render_template, flash, abort
from functools import wraps
import logging

# Set up logging for the auth_utils module
logger = logging.getLogger(__name__)

# Define login required decorator
def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'user_id' not in session:
            logger.debug("User not logged in, redirecting to login page.")
            flash("Please log in to access this page.", "warning")
            return redirect(url_for('login_bp.login'))
        logger.debug(f"User {session.get('user_id')} is logged in.")
        return f(*args, **kwargs)
    return decorated_function


def requires_roles(*allowed_levels):
    def decorator(f):
        @wraps(f)
        def decorated_function(*args, **kwargs):
            # Check if the user is logged in
            user_level = session.get('user_level', None)
            logger.info(f"User level in session: {user_level}")

            if user_level is None:
                flash('You need to be logged in to access this page.', 'error')
                return redirect(url_for('auth_bp.login'))  # Redirect to login page if not logged in

            # Check if the user has the required level of access (case-sensitive match)
            if user_level not in allowed_levels:
                logger.info(f"Access denied for user level: {user_level}")
                flash('You do not have permission to access this page.', 'error')
                abort(403)  # Return 403 Forbidden if the user doesn't have the required role

            return f(*args, **kwargs)
        return decorated_function
    return decorator


def logout():
    # Clear session variables related to user authentication
    session.pop('user_id', None)
    session.pop('employee_id', None)
    session.pop('name', None)
    session.pop('primary_area', None)
    session.pop('age', None)
    session.pop('education_level', None)
    session.pop('start_date', None)

    # Debug statements
    print("Session variables cleared.")

    # Redirect the user to the login page
    print("Redirecting to login page.")
    return redirect(url_for('login_bp.login'))



