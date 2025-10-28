#__init__.py
from flask import Blueprint
from modules.tool_module.forms.tool_tool_form import ToolForm
# Create the blueprint
tool_blueprint_bp = Blueprint('tool_routes', __name__, template_folder='../../templates/tool_templates')

# Import routes from submodules
from .tool_add import *
from .tool_search import *
from .tool_get_data import get_tool_manufacturers,get_tool_categories  # Import the renamed module
from .tool_manage_category import *
from .tool_manage_manufacturer import *
from .search_tool import *
