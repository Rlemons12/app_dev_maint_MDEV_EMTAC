from flask import Blueprint, jsonify
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, scoped_session
from sqlalchemy.ext.declarative import declarative_base
from modules.configuration.config import DATABASE_URL
from modules.emtacdb.emtacdb_fts import Area, EquipmentGroup, Model, AssetNumber, Location, CompleteDocument, Image

get_search_tsg_list_data_bp = Blueprint('get_search_tsg_list_data_bp', __name__)

# Create SQLAlchemy engine
engine = create_engine(DATABASE_URL)
Base = declarative_base()
Session = scoped_session(sessionmaker(bind=engine))  # Use scoped_session here

@get_search_tsg_list_data_bp.route('/get_search_tsg_list_data_bp')
def get_list_data():
    # Create a session
    session = Session()

    try:
        print(f'get_search_tsg_list_data_bp Query the database to get all areas, equipment groups, models, asset numbers, locations, and documents')
        areas = session.query(Area).all()
        equipment_groups = session.query(EquipmentGroup).all()
        models = session.query(Model).all()
        asset_numbers = session.query(AssetNumber).all()
        locations = session.query(Location).all()
        documents = session.query(CompleteDocument).all()
        images = session.query(Image).all()  # Fetch images from the database

        # Convert queried data to a list of dictionaries for JSON serialization
        areas_list = [{'id': area.id, 'name': area.name} for area in areas]
        equipment_groups_list = [{'id': equipment_group.id, 'name': equipment_group.name, 'area_id': equipment_group.area_id} for equipment_group in equipment_groups]
        models_list = [{'id': model.id, 'name': model.name, 'equipment_group_id': model.equipment_group_id} for model in models]
        asset_numbers_list = [{'id': number.id, 'number': number.number, 'model_id': number.model_id} for number in asset_numbers]
        locations_list = [{'id': location.id, 'name': location.name, 'model_id': location.model_id} for location in locations]
        documents_list = [{'id': doc.id, 'title': doc.title, 'file_path': doc.file_path} for doc in documents]
        images_list = [{'id': image.id, 'title': image.title, 'description': image.description} for image in images]  # Convert images to a list of dictionaries

    except Exception as e:
        print("An error occurred:", e)
        session.rollback()
        raise e

    finally:
        # Close the session
        session.close()

    # Combine all the lists into a single dictionary
    data = {
        'areas': areas_list,
        'equipment_groups': equipment_groups_list,
        'models': models_list,
        'asset_numbers': asset_numbers_list,
        'locations': locations_list,
        'documents': documents_list,
        'images': images_list  # Include images in the data dictionary
    }

    # Print out the documents list
    print("Documents List:")
    for document in documents_list:
        print(document)

    # Print out Image list
    print("Image List:")
    for image in images_list:
        print(image)


    # Return the data as JSON
    return jsonify(data)

