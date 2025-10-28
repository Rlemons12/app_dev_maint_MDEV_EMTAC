# blueprints/bill_of_materials/__init__.py
from flask import Blueprint

# Create the blueprint
update_part_bp = Blueprint('update_part_bp', __name__, template_folder=
                                                                    '../../templates/bill_of_materials/bom_partials')

# Import routes from submodules
from .update_part_bp import *  # Import all from update_part.py