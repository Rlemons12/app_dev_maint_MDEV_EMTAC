import logging
from flask import Blueprint, send_file, request, redirect, url_for, flash, render_template, session as flask_session
from modules.emtacdb.emtacdb_fts import (
    PartsPositionImageAssociation, Position, Part, Image, BOMResult, AssetNumber
)
from modules.configuration.config_env import DatabaseConfig
from modules.configuration.config import DATABASE_DIR
from modules.configuration.log_config import logger
import json
import os

# Instantiate the database configuration
db_config = DatabaseConfig()

# Blueprint setup (define this before using it)
create_bill_of_material_bp = Blueprint('create_bill_of_material_bp', __name__)

def serve_bom_image(session, image_id):
    logger.info(f"Entered serve_bom_image with image_id: {image_id}")
    try:
        image = session.query(Image).filter_by(id=image_id).first()
        if image:
            logger.debug(f"Image found: {image.title} with file_path: {image.file_path}")
            file_path = os.path.join(DATABASE_DIR, image.file_path)
            logger.debug(f"Constructed file path: {file_path}")
            if os.path.exists(file_path):
                logger.info(f"File exists. Serving file: {file_path}")
                return send_file(file_path, mimetype='image/jpeg', as_attachment=False)
            else:
                logger.error(f"File not found at path: {file_path}")
                return "Image file not found", 404
        else:
            logger.error(f"No image found with ID: {image_id}")
            return "Image not found", 404
    except Exception as e:
        logger.exception(f"Exception in serve_bom_image: {e}")
        return "Internal Server Error", 500

@create_bill_of_material_bp.route('/bom_serve_image/<int:image_id>')
def bom_serve_image_route(image_id):
    logger.debug(f"Route /bom_serve_image accessed with image_id: {image_id}")
    db_session = db_config.get_main_session()
    try:
        response = serve_bom_image(db_session, image_id)
        logger.debug(f"Response from serve_bom_image: {response}")
        return response
    except Exception as e:
        logger.exception(f"Error in bom_serve_image_route for image_id {image_id}: {e}")
        flash(f"Error serving image {image_id}", "error")
        return "Image not found", 404
    finally:
        db_session.close()
        logger.debug("Database session closed in bom_serve_image_route.")


@create_bill_of_material_bp.route('/create_bill_of_material', methods=['GET', 'POST'])
def create_bill_of_material():
    logger.info("Entered create_bill_of_material route.")

    # Handle GET requests
    if request.method == 'GET':
        logger.info("GET request to create_bill_of_material, rendering form template.")
        return render_template('bill_of_materials/bill_of_materials.html')

    # Handle POST requests
    db_session = db_config.get_main_session()
    try:
        logger.debug("Retrieving form data for create_bill_of_material.")
        area_id = request.form.get('area')
        equipment_group_id = request.form.get('equipment_group')
        model_id = request.form.get('model')
        asset_number_id = request.form.get('asset_number')
        location_id = request.form.get('location')
        logger.debug(f"Form data received: area_id={area_id}, equipment_group_id={equipment_group_id}, "
                     f"model_id={model_id}, asset_number_id={asset_number_id}, location_id={location_id}")

        # Start building the query
        query = db_session.query(Position)

        if area_id:
            query = query.filter(Position.area_id == int(area_id))
            logger.debug(f"Filtered by area_id: {area_id}")
        if equipment_group_id:
            query = query.filter(Position.equipment_group_id == int(equipment_group_id))
            logger.debug(f"Filtered by equipment_group_id: {equipment_group_id}")
        if model_id:
            query = query.filter(Position.model_id == int(model_id))
            logger.debug(f"Filtered by model_id: {model_id}")
        if asset_number_id:
            query = query.filter(Position.asset_number_id == int(asset_number_id))
            logger.debug(f"Filtered by asset_number_id: {asset_number_id}")
        if location_id:
            query = query.filter(Position.location_id == int(location_id))
            logger.debug(f"Filtered by location_id: {location_id}")

        positions = query.all()
        logger.info(f"Number of positions found: {len(positions)}")

        if not positions:
            logger.warning("No matching positions found for create_bill_of_material.")
            flash('No matching positions found for the provided input.', 'error')
            return render_template('bill_of_materials/bill_of_materials.html')

        results = []
        for position in positions:
            logger.debug(f"Processing position with ID: {position.id}")
            parts_images = db_session.query(PartsPositionImageAssociation).filter_by(position_id=position.id).all()
            logger.debug(f"Found {len(parts_images)} parts/images associations for position ID: {position.id}")
            for association in parts_images:
                part = db_session.query(Part).filter_by(id=association.part_id).first()
                if part:
                    store_bom_results(db_session, part_id=part.id, position_id=position.id,
                                      image_id=association.image_id, description="Sample description")
                    results.append({'part_id': part.id, 'image_id': association.image_id})
                    logger.debug(f"Stored BOM result for part ID: {part.id}, position ID: {position.id}, "
                                 f"image ID: {association.image_id}")
                else:
                    logger.error(f"Part not found with ID: {association.part_id}")

        db_session.commit()
        logger.info("BOM results committed successfully.")

        flask_session['results'] = json.dumps(results)
        flask_session['model_id'] = model_id
        flask_session['asset_number_id'] = asset_number_id
        flask_session['location_id'] = location_id
        logger.debug("BOM results stored in session.")

        logger.info("Redirecting to view_bill_of_material route with index 0.")
        return redirect(url_for('create_bill_of_material_bp.view_bill_of_material', index=0))
    except Exception as e:
        logger.exception(f"Exception in create_bill_of_material: {e}")
        flash(f'An error occurred: {str(e)}', 'error')
        db_session.rollback()
        return render_template('bill_of_materials/bill_of_materials.html')
    finally:
        db_session.close()
        logger.debug("Database session closed in create_bill_of_material.")

@create_bill_of_material_bp.route('/view_bill_of_material', methods=['GET'])
def view_bill_of_material():
    """
    View the bill of material results with pagination.
    Redirects to a standalone results page.
    """
    logger.info("Entered view_bill_of_material route.")
    db_session = db_config.get_main_session()
    try:
        # Get pagination parameters
        index = request.args.get('index', 0, type=int)
        per_page = request.args.get('per_page', 4, type=int)
        logger.debug(f"Pagination parameters: index={index}, per_page={per_page}")

        # Query BOM results with pagination
        query = db_session.query(BOMResult).order_by(BOMResult.id.desc()).offset(index).limit(per_page)
        results = query.all()
        total_results = db_session.query(BOMResult).count()
        logger.info(f"Retrieved {len(results)} BOM results out of total {total_results}")

        # If no results, redirect back with a message
        if not results:
            logger.warning("No BOM results found in the database.")
            flash('No results found. Please try creating a Bill of Materials first.', 'warning')
            return redirect(url_for('create_bill_of_material_bp.create_bill_of_material'))

        # Process results to get parts and images
        parts_and_images = []
        for result in results:
            logger.debug(f"Processing BOM result with part_id: {result.part_id}")
            part = db_session.query(Part).filter_by(id=result.part_id).first()

            if not part:
                logger.error(f"Part not found for BOM result with part_id: {result.part_id}")
                continue

            image = None
            if result.image_id is not None:
                image = db_session.query(Image).filter_by(id=result.image_id).first()
                if not image:
                    logger.warning(f"Image not found for BOM result with image_id: {result.image_id}")

            # Add to results list
            parts_and_images.append({
                'part': part,
                'image': image,
                'description': result.description
            })

        # Calculate pagination
        next_index = index + per_page if index + per_page < total_results else None
        prev_index = index - per_page if index - per_page >= 0 else None
        logger.debug(f"Pagination: next_index={next_index}, prev_index={prev_index}")

        # Debug information to help troubleshoot
        debug_info = {
            'result_count': len(results),
            'parts_count': len(parts_and_images),
            'total_results': total_results
        }

        # Additional context data from session
        model_id = flask_session.get('model_id')
        asset_number_id = flask_session.get('asset_number_id')
        location_id = flask_session.get('location_id')

        # Get descriptive names if possible
        model_name = None
        asset_number = None
        location_name = None

        if asset_number_id:
            try:
                asset_obj = db_session.query(AssetNumber).filter_by(id=asset_number_id).first()
                if asset_obj:
                    asset_number = asset_obj.number
            except Exception as e:
                logger.error(f"Error getting asset number: {e}")

        # Log rendering parameters
        logger.info(f"Rendering template with {len(parts_and_images)} parts/images")

        # Return the standalone results page
        return render_template(
            'bill_of_materials/bom_partials/bill_of_material_results_partial.html',
            index=index,
            parts_and_images=parts_and_images,
            per_page=per_page,
            total=total_results,
            next_index=next_index,
            prev_index=prev_index,
            debug_info=debug_info,
            model_name=model_name,
            asset_number=asset_number,
            location_name=location_name
        )

    except Exception as e:
        logger.exception(f"Exception in view_bill_of_material: {e}")
        flash(f'An error occurred: {str(e)}', 'error')
        return redirect(url_for('create_bill_of_material_bp.create_bill_of_material'))
    finally:
        db_session.close()
        logger.debug("Database session closed in view_bill_of_material.")

@create_bill_of_material_bp.route('/bom_general_search', methods=['POST'])
def bom_general_search():
    logger.info("Entered bom_general_search route.")
    db_session = db_config.get_main_session()

    try:
        clear_bom_results()

        # 1) GRAB INPUTS
        general_asset_number = request.form.get('general_asset_number', '').strip()
        general_location     = request.form.get('general_location', '').strip()
        logger.debug(f"General search parameters: asset_number='{general_asset_number}', location='{general_location}'")

        # 2) ASSET-NUMBER-BASED SEARCH
        positions = []
        if general_asset_number:
            asset_ids = AssetNumber.get_ids_by_number(db_session, general_asset_number)
            logger.debug(f"AssetNumber IDs: {asset_ids}")

            if not asset_ids:
                flash('No Asset Number found matching the provided input.', 'error')
                return render_template('bill_of_materials/bill_of_materials.html')

            # primary lookup by asset_number_id
            positions = (
                db_session
                .query(Position)
                .filter(Position.asset_number_id.in_(asset_ids))
                .all()
            )
            logger.info(f"Found {len(positions)} positions by asset_number.")

            # fallback to model if needed
            if not positions:
                logger.info("FALLBACK: No positions for the asset—trying by model")
                model_id = AssetNumber.get_model_id_by_asset_number_id(db_session, asset_ids[0])
                if model_id:
                    logger.info(f"Falling back to model_id={model_id}")
                    positions = (
                        db_session
                        .query(Position)
                        .filter(Position.model_id == model_id)
                        .all()
                    )
                    logger.info(f"Found {len(positions)} positions by model.")
        else:
            # no asset number → all positions that have *some* asset_number
            positions = (
                db_session
                .query(Position)
                .filter(Position.asset_number_id.isnot(None))
                .all()
            )
            logger.debug("Asset number not provided; using all positions with an asset_number.")

        # 3) LOCATION FILTER (only if user supplied one)
        if general_location:
            location_records = (
                db_session
                .query(Location)
                .filter(Location.name == general_location)
                .all()
            )
            if not location_records:
                flash('No Location found matching the provided input.', 'error')
                return render_template('bill_of_materials/bill_of_materials.html')

            location_ids = [loc.id for loc in location_records]
            logger.debug(f"Filtering positions by location_ids: {location_ids}")
            positions = [pos for pos in positions if pos.location_id in location_ids]
            logger.debug(f"Positions after location filter: {len(positions)}")
        # otherwise: leave *all* positions (including those with NULL location_id)

        # 4) NO POSITIONS → ERROR
        logger.info(f"Total positions after filters: {len(positions)}")
        if not positions:
            flash('No results found for the given Asset Number or Location.', 'error')
            return render_template('bill_of_materials/bill_of_materials.html')

        # 5) BUILD BOM RESULTS
        results = []
        for position in positions:
            logger.debug(f"Processing position ID: {position.id}")
            associations = (
                db_session
                .query(PartsPositionImageAssociation)
                .filter_by(position_id=position.id)
                .all()
            )
            for assoc in associations:
                part = db_session.query(Part).get(assoc.part_id)
                if not part:
                    logger.error(f"Part not found with ID: {assoc.part_id}")
                    continue

                store_bom_results(
                    db_session,
                    part_id=part.id,
                    position_id=position.id,
                    image_id=assoc.image_id,
                    description="General search result"
                )
                results.append({
                    'part_id': part.id,
                    'image_id': assoc.image_id
                })
                logger.debug(f"Stored BOM result for part ID: {part.id}")

        # 6) COMMIT & REDIRECT
        db_session.commit()
        logger.info("General search BOM results committed successfully.")

        flask_session['results']              = json.dumps(results)
        flask_session['general_asset_number'] = general_asset_number
        flask_session['general_location']     = general_location

        return redirect(url_for('create_bill_of_material_bp.view_bill_of_material', index=0))

    except Exception as e:
        logger.exception(f"Exception in bom_general_search: {e}")
        db_session.rollback()
        flash(f'An error occurred during general search: {e}', 'error')
        return render_template('bill_of_materials/bill_of_materials.html')

    finally:
        db_session.close()
        logger.debug("Database session closed in bom_general_search.")

def store_bom_results(session, part_id, position_id, image_id=None, description=None):
    try:
        result = BOMResult(part_id=part_id, position_id=position_id, image_id=image_id, description=description)
        session.add(result)
        logger.info(f"Stored BOM result: part_id={part_id}, position_id={position_id}, "
                    f"image_id={image_id}, description='{description}'")
    except Exception as e:
        logger.exception(f"Failed to store BOM result for part_id={part_id}: {e}")
        raise

def clear_bom_results():
    logger.info("Initiating clearing of BOM results.")
    session = db_config.get_main_session()
    try:
        deleted_count = session.query(BOMResult).delete()
        session.commit()
        logger.info(f"Cleared BOM results successfully, deleted {deleted_count} record(s).")
    except Exception as e:
        logger.exception(f"Failed to clear BOM results: {e}")
        session.rollback()
    finally:
        session.close()
        logger.debug("Database session closed in clear_bom_results.")

@create_bill_of_material_bp.route('/debug_bom_results')
def debug_bom_results():
    """Debug route to check BOM results in the database."""
    if not flask_session.get('user_level') == 'ADMIN':
        flash('Admin access required', 'error')
        return redirect(url_for('create_bill_of_material_bp.create_bill_of_material'))

    db_session = db_config.get_main_session()
    try:
        # Get all BOM results
        results = db_session.query(BOMResult).all()
        count = len(results)

        # Get sample data for first 10 results
        sample_data = []
        for result in results[:10]:
            part = db_session.query(Part).filter_by(id=result.part_id).first()
            position = db_session.query(Position).filter_by(id=result.position_id).first()

            part_info = {'id': result.part_id, 'exists': part is not None}
            if part:
                part_info.update({
                    'part_number': part.part_number,
                    'name': part.name
                })

            position_info = {'id': result.position_id, 'exists': position is not None}
            if position:
                position_info.update({
                    'name': getattr(position, 'name', 'N/A')
                })

            sample_data.append({
                'id': result.id,
                'part': part_info,
                'position': position_info,
                'image_id': result.image_id,
                'description': result.description
            })

        # Get session data
        session_data = {
            'results': flask_session.get('results'),
            'model_id': flask_session.get('model_id'),
            'asset_number_id': flask_session.get('asset_number_id'),
            'location_id': flask_session.get('location_id'),
            'general_asset_number': flask_session.get('general_asset_number'),
            'general_location': flask_session.get('general_location')
        }

        return render_template(
            'debug_bom_results.html',
            count=count,
            sample_data=sample_data,
            session_data=session_data
        )
    except Exception as e:
        logger.exception(f"Exception in debug_bom_results: {e}")
        flash(f'Debug error: {str(e)}', 'error')
        return str(e), 500
    finally:
        db_session.close()

@create_bill_of_material_bp.route('/debug_clear_bom_results')
def debug_clear_bom_results():
    """Debug route to clear all BOM results."""
    if not flask_session.get('user_level') == 'ADMIN':
        flash('Admin access required', 'error')
        return redirect(url_for('create_bill_of_material_bp.create_bill_of_material'))

    try:
        clear_bom_results()
        flash('All BOM results cleared', 'success')
    except Exception as e:
        logger.exception(f"Exception in debug_clear_bom_results: {e}")
        flash(f'Error clearing BOM results: {str(e)}', 'error')

    return redirect(url_for('create_bill_of_material_bp.debug_bom_results'))