from __future__ import annotations

from flask import Blueprint, request, render_template

from modules.configuration.log_config import logger, with_request_id
from modules.coordinators.bill_of_materials_coordinator import BillOfMaterialsCoordinator
from utilities.auth_utils import login_required


search_bill_of_material_bp = Blueprint(
    "search_bill_of_material_bp",
    __name__,
    template_folder="templates",
)

coordinator = BillOfMaterialsCoordinator()


@search_bill_of_material_bp.route("/tool_search", methods=["GET", "POST"])
@login_required
@with_request_id
def search_bill_of_material():
    """
    Search bill of materials / parts by position-related filters.

    Preserves legacy route name:
        /tool_search
    """
    logger.info("Accessed route: /tool_search | method=%s", request.method)

    source = request.form if request.method == "POST" else request.args

    result = coordinator.search_bill_of_materials(
        form_data=source,
    )

    template_name = result.get(
        "template_name",
        "bill_of_materials/partials/bom_search_message.html",
    )
    context = result.get("context", {})

    return render_template(template_name, **context), result.get("status_code", 200)