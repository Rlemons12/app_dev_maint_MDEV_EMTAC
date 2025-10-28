# search_bill_of_material.route.py

from flask import Blueprint, request, render_template
from models import Position, PartsPositionImageAssociation, Part, db
import concurrent.futures

# Define the Blueprint
search_bill_of_material = Blueprint('search_bill_of_material', __name__, template_folder='templates')


def get_parts_for_position(position_id):
    """Retrieve parts for a given position."""
    parts = []
    parts_position_images = db.session.query(PartsPositionImageAssociation).filter_by(position_id=position_id).all()

    for ppi in parts_position_images:
        part = db.session.query(Part).filter_by(id=ppi.part_id).first()
        if part:
            parts.append(part)

    return parts


@search_bill_of_material.route('/tool_search', methods=['GET'])
def search_bill_of_material():
    model = request.args.get('model')
    asset_number = request.args.get('asset_number')
    location = request.args.get('location')

    # Start building the query for Position
    query = db.session.query(Position)

    if model:
        query = query.filter(Position.model_id == int(model))
    elif asset_number:
        query = query.filter(Position.asset_number_id == int(asset_number))
    elif location:
        query = query.filter(Position.location_id == int(location))

    positions = query.all()

    if not positions:
        return "No positions found matching the search criteria", 404

    # Use ThreadPoolExecutor to fetch parts concurrently for each position
    parts = []
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = {executor.submit(get_parts_for_position, position.id): position for position in positions}

        for future in concurrent.futures.as_completed(futures):
            parts += future.result()

    if parts:
        return render_template('parts_results.html', parts=parts)
    else:
        return "No parts found for the selected positions", 404
