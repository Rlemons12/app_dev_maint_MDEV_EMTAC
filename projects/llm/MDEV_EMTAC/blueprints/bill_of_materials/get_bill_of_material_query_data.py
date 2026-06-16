from __future__ import annotations

from flask import Blueprint, jsonify, render_template

from modules.configuration.log_config import logger, with_request_id
from modules.coordinators.bill_of_materials_coordinator import BillOfMaterialsCoordinator

get_bill_of_material_query_data_bp = Blueprint(
    "get_bill_of_material_query_data_bp",
    __name__,
)

coordinator = BillOfMaterialsCoordinator()


@get_bill_of_material_query_data_bp.route("/get_parts_position_data", methods=["GET"])
@with_request_id
def get_parts_position_data():
    logger.info("Received request to fetch Parts Position data")

    result = coordinator.get_parts_position_data()

    if not result["success"]:
        logger.error("Failed to fetch Parts Position data: %s", result["message"])
        return jsonify({"error": result["message"]}), result["status_code"]

    logger.info("Successfully fetched Parts Position data")
    return jsonify(result["data"]), 200


@get_bill_of_material_query_data_bp.route("/search_bill_of_material", methods=["GET"])
@with_request_id
def filter_parts_position():
    logger.info("Rendering search_bill_of_material page")
    return render_template("search_bill_of_material.html")