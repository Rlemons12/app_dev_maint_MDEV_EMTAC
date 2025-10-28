import logging
from flask import Blueprint, render_template, request, redirect, flash, session, url_for
from modules.emtacdb.emtacdb_fts import ChatSession, User, UserLevel, UserLogin
from datetime import datetime
from sqlalchemy.orm import sessionmaker, scoped_session, declarative_base
from flask_bcrypt import Bcrypt
from sqlalchemy import create_engine
from modules.configuration.config import DATABASE_URL

# Set up logging configuration
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create SQLAlchemy engine for the main database
logger.info(f"Creating SQLAlchemy engine with DATABASE_URL: {DATABASE_URL}")
engine = create_engine(DATABASE_URL)
Base = declarative_base()
db_session = scoped_session(sessionmaker(bind=engine))

login_bp = Blueprint('login_bp', __name__)

bcrypt = Bcrypt()  # Create a Flask-Bcrypt instance


@login_bp.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        employee_id = request.form['employee_id']
        password = request.form['password']

        logger.info(f"Login attempt for employee_id: {employee_id}")

        try:
            # Check if a user with the provided employee_id exists
            user = db_session.query(User).filter_by(employee_id=employee_id).first()
            logger.debug(f"User found: {user}")

            if user:
                logger.debug(f"User {user.employee_id} found. Checking password.")
                if user.check_password_hash(password):
                    logger.info(f"User {user.employee_id} authenticated successfully.")

                    # Create a chat session (keep your existing code)
                    current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                    new_chat_session = ChatSession(
                        user_id=str(user.id),
                        start_time=current_time,
                        last_interaction=current_time,
                        session_data=[]
                    )
                    db_session.add(new_chat_session)

                    # Create a user login record
                    user_login = UserLogin(
                        user_id=user.id,
                        session_id=request.cookies.get('session', ''),
                        ip_address=request.remote_addr,
                        user_agent=request.user_agent.string if request.user_agent else None
                    )
                    db_session.add(user_login)

                    # Store user login ID in session for tracking
                    session['login_record_id'] = user_login.id

                    db_session.commit()

                    # Store user information in Flask session (keep your existing code)
                    session['user_id'] = user.id
                    session['employee_id'] = user.employee_id
                    session['first_name'] = user.first_name
                    session['last_name'] = user.last_name
                    session['primary_area'] = user.primary_area
                    session['age'] = user.age
                    session['education_level'] = user.education_level
                    session['start_date'] = user.start_date
                    session['user_level'] = user.user_level.name
                    session['login_time'] = current_time

                    # Redirect based on user level (keep your existing code)
                    if user.user_level == UserLevel.ADMIN:
                        logger.info(f"Redirecting admin user {user.employee_id} to admin dashboard.")
                        return redirect(url_for('admin_bp.admin_dashboard'))
                    elif user.user_level == UserLevel.STANDARD:
                        logger.info(f"Redirecting standard user {user.employee_id} to upload_image_page.")
                        return redirect(url_for('upload_image_page'))

                    # Redirect to the main index route
                    logger.info(f"Redirecting user {user.employee_id} to the index page.")
                    return redirect(url_for('index'))
                else:
                    logger.warning(f"Failed login attempt for user {employee_id}: Incorrect password.")
                    flash("Invalid username or password", 'error')
            else:
                logger.warning(f"Failed login attempt: User {employee_id} not found.")
                flash("Invalid username or password", 'error')

        except Exception as e:
            logger.error(f"An error occurred during login attempt for user {employee_id}: {e}")
            flash(f"An error occurred: {e}", 'error')

        finally:
            db_session.remove()
            logger.debug("SQLAlchemy session removed.")

    return render_template('login.html')


@login_bp.route('/logout')
def logout():
    logger.info("Logging out user.")

    # Update UserLogin record to mark session as ended
    if 'login_record_id' in session:
        try:
            with SessionFactory() as session_db:
                login_record = session_db.query(UserLogin).get(session['login_record_id'])
                if login_record:
                    login_record.logout_time = datetime.utcnow()
                    login_record.is_active = False
                    session_db.commit()
        except Exception as e:
            logger.error(f"Error updating login record on logout: {e}")

    # Clear all user-related session data
    session.clear()

    logger.info("User session cleared. Redirecting to login page.")
    return redirect(url_for('login_bp.login'))


def activity_tracker():
    """Middleware function to track user activity and update last_activity timestamp"""
    if 'user_id' in session and 'login_record_id' in session:
        try:
            # Skip certain static file requests to reduce database load
            if request.path.startswith('/static/'):
                return

            # Use your existing db_session from login_bp
            login_record = db_session.query(UserLogin).get(session['login_record_id'])
            if login_record and login_record.is_active:
                login_record.last_activity = datetime.utcnow()
                db_session.commit()
        except Exception as e:
            # Log error but don't crash the app
            logger.error(f"Error updating activity timestamp: {e}")
        finally:
            db_session.remove()  # Use remove() for scoped_session