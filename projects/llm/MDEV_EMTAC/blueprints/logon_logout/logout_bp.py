from flask import Blueprint, render_template, request, redirect, flash, session,redirect, url_for

logout_bp = Blueprint('logout_bp', __name__)

@logout_bp.route('/logout_bp', methods=['GET', 'POST'])
def logout():
    # Clear session variables related to user authentication
    session_keys = ['user_id', 'employee_id', 'name', 'primary_area', 
                    'age', 'education_level', 'start_date']
    for key in session_keys:
        session.pop(key, None)

    if request.method == 'POST':
        return '', 204  # Return no content for AJAX request
    else:
        return redirect(url_for('login_bp.login'))  # Redirect to login page for regular logout

