import os
import base64
from flask import Blueprint, request, jsonify, session
from modules.emtacdb.emtacdb_fts import User, UserComments  # Import UserComments model
from modules.configuration.config_env import DatabaseConfig
from modules.configuration.config import COMMENT_IMAGES_FOLDER
from sqlalchemy.orm.exc import NoResultFound
from datetime import datetime
from modules.configuration.log_config import logger, info_id, warning_id, error_id, debug_id, with_request_id, log_timed_operation

# Initialize the database config
db_config = DatabaseConfig()

comment_pop_up_bp = Blueprint('comment_pop_up_bp', __name__)

# Ensure the COMMENT_IMAGES_FOLDER exists
if not os.path.exists(COMMENT_IMAGES_FOLDER):
    os.makedirs(COMMENT_IMAGES_FOLDER)
    info_id(f"Created folder: {COMMENT_IMAGES_FOLDER}")

@comment_pop_up_bp.route('/submit-comment', methods=['POST'])
@with_request_id
def submit_comment():
    comment = request.form.get('comment')
    page_url = request.form.get('page_url')
    employee_id = session.get('employee_id')  # Retrieve employee_id from session
    base64_image = request.form.get('imageData')  # Get base64-encoded image data

    info_id(f"Submitting comment from user: {employee_id} for page: {page_url}")

    # Return an error if comment or page_url are missing
    if not comment or not page_url:
        warning_id("Comment or page URL missing")
        return jsonify({"error": "Comment and page URL are required."}), 400

    # Open a new session for the main database
    with log_timed_operation("Database session"):
        session_db = db_config.get_main_session()

    # Retrieve the user based on employee_id
    user_id = None  # Default to None for anonymous comments
    if employee_id:
        try:
            with log_timed_operation("User lookup"):
                user = session_db.query(User).filter_by(employee_id=employee_id).one()
                user_id = user.id  # Set the user_id if the user exists
                info_id(f"User {employee_id} found with ID: {user_id}")
        except NoResultFound:
            warning_id(f"No user found with employee_id: {employee_id}")
            # Proceed without user association if no result found

    screenshot_relative_path = None

    # Process image if provided
    if base64_image:
        with log_timed_operation("Image processing"):
            try:
                # Decode the base64 image and save it as a file
                image_data = base64.b64decode(base64_image.split(',')[1])  # Remove 'data:image/...;base64,' prefix
                image_name = f"{datetime.utcnow().strftime('%Y%m%d%H%M%S')}.png"
                # Use relative path for the database, full path for saving
                screenshot_relative_path = os.path.join('comment_images', image_name).replace("\\", "/")  # Ensure forward slashes
                screenshot_full_path = os.path.join(COMMENT_IMAGES_FOLDER, image_name)  # Full path for saving

                with open(screenshot_full_path, 'wb') as f:
                    f.write(image_data)
                info_id(f"Screenshot saved at {screenshot_full_path}")
            except Exception as e:
                error_id(f"Failed to save image: {e}")
                return jsonify({"error": f"Failed to save image: {e}"}), 500

    # Create a new UserComments entry
    try:
        with log_timed_operation("Comment submission"):
            new_comment = UserComments(
                user_id=user_id,
                comment=comment,
                page_url=page_url,
                screenshot_path=screenshot_relative_path,  # Store the relative path in the database
                timestamp=datetime.utcnow()
            )

            # Add the comment to the session and commit
            session_db.add(new_comment)
            session_db.commit()
            info_id("Comment submitted successfully")

    except Exception as e:
        error_id(f"Error while saving comment: {e}")
        session_db.rollback()
        return jsonify({"error": f"Failed to submit comment: {e}"}), 500

    finally:
        # Close the session after completing the transaction
        session_db.close()
        debug_id("Database session closed")

    return jsonify({"message": "Comment submitted successfully!"}), 200