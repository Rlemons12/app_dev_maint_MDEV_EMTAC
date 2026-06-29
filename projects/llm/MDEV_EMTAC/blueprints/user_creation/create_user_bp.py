from flask import Blueprint, render_template, request, redirect, url_for, flash
from modules.emtacdb.emtacdb_fts import UserLevel  # Import UserLevel
from datetime import datetime
from modules.configuration.log_config import logger

from modules.emtacdb.emtacdb_fts import User

create_user_bp = Blueprint('create_user_bp', __name__)


@create_user_bp.route('/create_user', methods=['GET'])
def create_user():
    return render_template('create_user.html')


@create_user_bp.route('/submit_user_creation', methods=['POST'])
def submit_user_creation():
    logger.info("============ SUBMIT USER CREATION ROUTE STARTED ============")
    try:
        logger.info(">>> ROUTE HIT: submit_user_creation")

        # Extract form data
        employee_id = request.form['employee_id']
        first_name = request.form['first_name']
        last_name = request.form['last_name']
        current_shift = request.form['current_shift']
        primary_area = request.form['primary_area']

        # Convert age to integer if provided
        age_str = request.form.get('age', None)
        age = None
        if age_str and age_str.strip():
            try:
                age = int(age_str)
                logger.debug(f"Converted age '{age_str}' to {age}")
            except ValueError:
                logger.warning(f"Could not convert age '{age_str}' to integer")

        education_level = request.form.get('education_level', None)
        start_date_str = request.form.get('start_date', None)
        password = request.form['password']

        # Get user preferences
        text_to_voice = request.form.get('text_to_voice', 'default')
        voice_to_text = request.form.get('voice_to_text', 'default')

        # Convert start_date to datetime if provided
        start_date = None
        if start_date_str and start_date_str.strip():
            try:
                start_date = datetime.strptime(start_date_str, '%Y-%m-%d')
                logger.debug(f"Converted start_date: {start_date}")
            except ValueError:
                logger.warning(f"Invalid date format: {start_date_str}")

        logger.info(f"About to call User.create_new_user for employee_id={employee_id}")

        # Test direct database access to isolate the issue
        from modules.configuration.config_env import DatabaseConfig
        db_config = DatabaseConfig()
        session = db_config.get_main_session()
        logger.debug(f"Got database session: {session}")

        # Call the user creation method with try/except to catch any errors
        try:
            logger.info("Calling User.create_new_user...")
            success, message = User.create_new_user(
                employee_id=employee_id,
                first_name=first_name,
                last_name=last_name,
                password=password,
                current_shift=current_shift,
                primary_area=primary_area,
                age=age,
                education_level=education_level,
                start_date=start_date,
                text_to_voice=text_to_voice,
                voice_to_text=voice_to_text
            )
            logger.info(f"User.create_new_user returned: success={success}, message={message}")
        except Exception as e:
            logger.error(f"EXCEPTION IN USER CREATION: {str(e)}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            success = False
            message = f"Error creating user: {str(e)}"

        # Handle the result
        if success:
            logger.info(f"User creation successful: {message}")
            flash(message, "success")
            logger.info(f"User {employee_id} created successfully. Redirecting to login.")
            return redirect(url_for('login_bp.login'))
        else:
            logger.error(f"User creation failed: {message}")
            flash(message, "error")
            logger.error(f"Failed to create user {employee_id}: {message}")
            return redirect(url_for('create_user_bp.create_user'))

    except Exception as e:
        logger.error(f"UNHANDLED EXCEPTION IN ROUTE: {str(e)}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        flash(f"An error occurred: {str(e)}", "error")
        return redirect(url_for('create_user_bp.create_user'))

    finally:
        logger.info("============ SUBMIT USER CREATION ROUTE FINISHED ============")
