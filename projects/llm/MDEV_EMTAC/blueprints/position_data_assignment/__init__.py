# blueprints/position_data_assignment/__init__.py

from flask import Blueprint

# Initialize the Blueprint
position_data_assignment_bp = Blueprint('position_data_assignment_bp', __name__,
    template_folder='../../templates/position_data_assignment', static_folder='../../static')

from .position_data_assignment_tool_management import *
from .position_data_assignment import *


