from __future__ import annotations

from flask import Blueprint, jsonify

from utilities.auth_utils import login_required
from modules.configuration.log_config import logger, with_request_id
from modules.coordinators.bill_of_materials_coordinator import BillOfMaterialsCoordinator

bill_of_materials_data_bp = Blueprint(
    "bill_of_materials_data_bp",
    __name__,
)


@bill_of_materials_data_bp.route("/get_bom_list_data", methods=["GET"])
@login_required
@with_request_id
def get_bom_list_data():
    logger.info("Route hit: /get_bom_list_data")

    lookup_coordinator = BillOfMaterialsCoordinator()
    result = lookup_coordinator.get_bom_list_data()

    status_code = result.get("status_code", 200)
    payload = {k: v for k, v in result.items() if k != "status_code"}

    return jsonify(payload), status_code


@bill_of_materials_data_bp.route("/get_parts_position_data", methods=["GET"])
@login_required
@with_request_id
def get_parts_position_data():
    logger.info("Route hit: /get_parts_position_data")

    lookup_coordinator = BillOfMaterialsCoordinator()
    result = lookup_coordinator.get_parts_position_data()

    status_code = result.get("status_code", 200)
    payload = {k: v for k, v in result.items() if k != "status_code"}

    return jsonify(payload), status_code