from __future__ import annotations

from flask import (
    jsonify,
    render_template,
    request,
    redirect,
    url_for,
    flash,
    send_file,
)

from blueprints.bill_of_materials import update_part_bp
from modules.configuration.log_config import logger, with_request_id
from modules.coordinators.bill_of_materials_coordinator import BillOfMaterialsCoordinator

coordinator = BillOfMaterialsCoordinator()


@update_part_bp.route("/edit_part/<int:part_id>", methods=["GET", "POST"])
@with_request_id
def edit_part(part_id: int):
    logger.info("Starting edit_part route for part_id=%s", part_id)

    if request.method == "POST":
        payload = coordinator.build_update_request(request=request, part_id=part_id)
        result = coordinator.update_part(payload)

        if payload["is_ajax"]:
            return jsonify(
                {
                    "success": result["success"],
                    "message": result["message"],
                }
            ), result["status_code"]

        if result["success"]:
            flash(result["message"], "success")
            return redirect(url_for("update_part_bp.edit_part", part_id=part_id))

        flash(result["message"], "error")

        if result.get("view_data"):
            return render_template(
                "bill_of_materials/bom_partials/edit_part.html",
                part=result["view_data"].get("part"),
                part_images=result["view_data"].get("part_images", []),
                positions=result["view_data"].get("positions", []),
                search_query=result["view_data"].get("search_query", ""),
            )

        return redirect(url_for("update_part_bp.search_part"))

    payload = coordinator.build_edit_view_request(request=request, part_id=part_id)

    if payload["is_ajax"]:
        return redirect(url_for("update_part_bp.edit_part_ajax", part_id=part_id))

    result = coordinator.get_edit_part_view_data(payload)

    if not result["success"]:
        flash(result["message"], "error")
        return redirect(url_for("update_part_bp.search_part"))

    return render_template(
        "bill_of_materials/bom_partials/edit_part.html",
        part=result["data"].get("part"),
        part_images=result["data"].get("part_images", []),
        positions=result["data"].get("positions", []),
        search_query=result["data"].get("search_query", ""),
    )


@update_part_bp.route("/part_image/<int:image_id>")
@with_request_id
def serve_part_image(image_id: int):
    logger.info("Starting serve_part_image route for image_id=%s", image_id)

    payload = coordinator.build_image_request(image_id=image_id)
    result = coordinator.get_part_image_file_data(payload)

    if not result["success"]:
        return result["message"], result["status_code"]

    return send_file(
        result["absolute_file_path"],
        mimetype=result["mimetype"],
    )


@update_part_bp.route("/search_part", methods=["GET"])
@with_request_id
def search_part():
    logger.info("Starting search_part route")

    payload = coordinator.build_search_request(request=request)
    result = coordinator.search_parts(payload)

    if payload["is_ajax"]:
        if result["success"]:
            return render_template(
                "bill_of_materials/bom_partials/search_parts_results.html",
                parts=result["data"].get("parts", []),
                search_query=result["data"].get("search_query", ""),
            )
        return result["message_html"], result["status_code"]

    if result["success"] and result["data"].get("redirect_part_id"):
        return redirect(
            url_for(
                "update_part_bp.edit_part",
                part_id=result["data"]["redirect_part_id"],
            )
        )

    if result["success"]:
        return render_template(
            "bill_of_materials/bom_partials/search_parts_results.html",
            parts=result["data"].get("parts", []),
            search_query=result["data"].get("search_query", ""),
            positions=result["data"].get("positions", []),
        )

    if result["message"]:
        flash(result["message"], result.get("flash_category", "info"))

    return render_template(
        "bill_of_materials/bom_partials/edit_part.html",
        part=None,
        part_images=[],
        positions=result.get("data", {}).get("positions", []),
        search_query=payload["search_query"],
    )


@update_part_bp.route("/search_part_ajax", methods=["GET"])
@with_request_id
def search_part_ajax():
    logger.info("Starting search_part_ajax route")

    payload = coordinator.build_ajax_search_request(request=request)
    result = coordinator.search_parts(payload)

    if result["success"]:
        return render_template(
            "bill_of_materials/bom_partials/search_parts_results.html",
            parts=result["data"].get("parts", []),
            search_query=result["data"].get("search_query", ""),
        )

    return result["message_html"], result["status_code"]


@update_part_bp.route("/edit_part_ajax/<int:part_id>", methods=["GET"])
@with_request_id
def edit_part_ajax(part_id: int):
    logger.info("Starting edit_part_ajax route for part_id=%s", part_id)

    payload = coordinator.build_edit_view_request(request=request, part_id=part_id)
    result = coordinator.get_edit_part_view_data(payload)

    if not result["success"]:
        return (
            f'<div class="alert alert-danger">{result["message"]}</div>',
            result["status_code"],
        )

    try:
        html = render_template(
            "bill_of_materials/bom_partials/edit_part.html",
            part=result["data"].get("part"),
            part_images=result["data"].get("part_images", []),
            positions=result["data"].get("positions", []),
            search_query=result["data"].get("search_query", ""),
        )
        return html
    except Exception as exc:
        logger.error(
            "Error rendering edit_part_ajax template for part_id=%s: %s",
            part_id,
            exc,
            exc_info=True,
        )
        return (
            f'<div class="alert alert-danger">Error rendering part form: {exc}</div>',
            500,
        )