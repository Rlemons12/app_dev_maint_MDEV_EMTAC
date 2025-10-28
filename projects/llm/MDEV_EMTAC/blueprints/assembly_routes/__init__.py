from flask import Blueprint

# Define the blueprint
assembly_model_bp = Blueprint(
    'assembly_routes',
    __name__,
    template_folder='../../templates/assembly_module'
)


# Import routes from associated files
from .assembly_models_view import *
from .submit_assembly import *
