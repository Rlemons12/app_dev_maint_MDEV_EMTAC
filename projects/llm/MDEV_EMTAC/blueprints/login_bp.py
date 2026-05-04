import logging
from flask import Blueprint, render_template, request, redirect, flash, session, url_for
from modules.emtacdb.emtacdb_fts import ChatSession, User, UserLevel, UserLogin
from datetime import datetime
from flask_bcrypt import Bcrypt
from modules.configuration.config_env import get_db_config

# Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ✅ USE SHARED DB CONFIG (FIX)
db_config = get_db_config()

login_bp = Blueprint('login_bp', __name__)
bcrypt = Bcrypt()


@login_bp.route('/login', methods=['GET', 'POST'])
def login():
    db_session = db_config.get_main_session()  # ✅ NEW SESSION PER REQUEST

    try:
        if request.method == 'POST':
            employee_id = request.form['employee_id']
            password = request.form['password']

            logger.info(f"Login attempt for employee_id: {employee_id}")

            user = db_session.query(User).filter_by(employee_id=employee_id).first()

            if user:
                if user.check_password_hash(password):
                    logger.info(f"User {user.employee_id} authenticated successfully.")

                    current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

                    new_chat_session = ChatSession(
                        user_id=str(user.id),
                        start_time=current_time,
                        last_interaction=current_time,
                        session_data=[]
                    )
                    db_session.add(new_chat_session)

                    user_login = UserLogin(
                        user_id=user.id,
                        session_id=request.cookies.get('session', ''),
                        ip_address=request.remote_addr,
                        user_agent=request.user_agent.string if request.user_agent else None
                    )
                    db_session.add(user_login)

                    session['login_record_id'] = user_login.id

                    db_session.commit()

                    # Store session data
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

                    if user.user_level == UserLevel.ADMIN:
                        return redirect(url_for('admin_bp.admin_dashboard'))
                    elif user.user_level == UserLevel.STANDARD:
                        return redirect(url_for('upload_image_page'))

                    return redirect(url_for('index'))

                else:
                    flash("Invalid username or password", 'error')
            else:
                flash("Invalid username or password", 'error')

    except Exception as e:
        logger.error(f"Login error: {e}")
        flash(f"An error occurred: {e}", 'error')

    finally:
        db_session.close()  # ✅ IMPORTANT

    return render_template('login.html')


@login_bp.route('/logout')
def logout():
    logger.info("Logging out user.")

    if 'login_record_id' in session:
        try:
            db_session = db_config.get_main_session()

            login_record = db_session.query(UserLogin).get(session['login_record_id'])
            if login_record:
                login_record.logout_time = datetime.utcnow()
                login_record.is_active = False
                db_session.commit()

        except Exception as e:
            logger.error(f"Error updating login record on logout: {e}")
        finally:
            db_session.close()

    session.clear()
    return redirect(url_for('login_bp.login'))


def activity_tracker():
    if 'user_id' in session and 'login_record_id' in session:
        if request.path.startswith('/static/'):
            return

        db_session = db_config.get_main_session()

        try:
            login_record = db_session.query(UserLogin).get(session['login_record_id'])
            if login_record and login_record.is_active:
                login_record.last_activity = datetime.utcnow()
                db_session.commit()
        except Exception as e:
            logger.error(f"Error updating activity timestamp: {e}")
        finally:
            db_session.close()