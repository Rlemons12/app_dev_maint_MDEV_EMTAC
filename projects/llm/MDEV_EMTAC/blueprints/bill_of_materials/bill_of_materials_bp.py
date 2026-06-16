# blueprints/bill_of_materials/bill_of_materials_bp.py
from __future__ import annotations

from flask import Blueprint, request, redirect, url_for, flash, render_template, jsonify

from utilities.auth_utils import login_required
from modules.configuration.log_config import logger, with_request_id
from modules.coordinators.bill_of_materials_coordinator import BillOfMaterialsCoordinator

bill_of_materials_bp = Blueprint(
    "bill_of_materials_bp",
    __name__,
    template_folder="templates",
)

upload_coordinator = BillOfMaterialsCoordinator()


def _is_ajax_request() -> bool:
    return (
        request.form.get("ajax") == "true"
        or request.args.get("ajax") == "true"
        or request.args.get("ajax") == "1"
        or request.headers.get("X-Requested-With") == "XMLHttpRequest"
    )


@bill_of_materials_bp.route("", methods=["GET", "POST"])
@login_required
@with_request_id
def bill_of_materials():
    logger.info("Accessed /bill_of_materials route")

    if request.method == "GET":
        return render_template("bill_of_materials/bill_of_materials.html")

    logger.info("Received BOM upload POST request")

    result = upload_coordinator.submit_bill_of_materials_upload(
        form_data=request.form,
        files=request.files,
    )

    if _is_ajax_request():
        status_code = 200 if result.get("success") else int(result.get("status_code", 400))
        return jsonify(result), status_code

    if result.get("success"):
        flash(result.get("message", "File successfully processed"), "success")
    else:
        flash(result.get("message", "An error occurred"), "error")

    redirect_endpoint = result.get("redirect_endpoint")
    redirect_values = result.get("redirect_values", {})

    if redirect_endpoint:
        return redirect(url_for(redirect_endpoint, **redirect_values))

    return redirect(url_for("bill_of_materials_bp.bill_of_materials"))