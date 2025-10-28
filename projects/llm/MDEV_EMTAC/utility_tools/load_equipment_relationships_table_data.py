import pandas as pd
from sqlalchemy import create_engine, func
from sqlalchemy.orm import sessionmaker, scoped_session
import os
from datetime import datetime

from modules.configuration.config import BASE_DIR, DATABASE_URL, DB_LOADSHEET
from modules.emtacdb.emtacdb_fts import Area, EquipmentGroup, Model, AssetNumber, Location

"""from auditlog import AuditLog, log_delete, log_insert, log_update
from emtac_revision_control_db import (
    VersionInfo, revision_control_engine, RevisionControlSession, SiteLocationSnapshot, PositionSnapshot, AreaSnapshot, EquipmentGroupSnapshot, ModelSnapshot,
    AssetNumberSnapshot, PartSnapshot, ImageSnapshot, ImageEmbeddingSnapshot, DrawingSnapshot,
    DocumentSnapshot, CompleteDocumentSnapshot, ProblemSnapshot, TaskSnapshot,
    DrawingPartAssociationSnapshot, PartProblemAssociationSnapshot, PartTaskAssociationSnapshot,
    PartsPositionImageAssociationSnapshot, DrawingProblemAssociationSnapshot, DrawingTaskAssociationSnapshot,
    ProblemPositionAssociationSnapshot, CompleteDocumentProblemAssociationSnapshot,
    CompleteDocumentTaskAssociationSnapshot, ImageProblemAssociationSnapshot,
    ImageTaskAssociationSnapshot, ImagePositionAssociationSnapshot, DrawingPositionAssociationSnapshot,
    CompletedDocumentPositionAssociationSnapshot, ImageCompletedDocumentAssociationSnapshot
)
from snapshot_utils import (
    get_latest_version_info, add_version_info,
    create_sitlocation_snapshot, create_position_snapshot, create_area_snapshot, create_equipment_group_snapshot,
    create_model_snapshot, create_asset_number_snapshot, create_part_snapshot, create_image_snapshot,
    create_image_embedding_snapshot, create_drawing_snapshot, create_document_snapshot,
    create_complete_document_snapshot, create_problem_snapshot, create_task_snapshot,
    create_drawing_part_association_snapshot, create_part_problem_association_snapshot,
    create_part_task_association_snapshot, create_drawing_problem_association_snapshot,
    create_drawing_task_association_snapshot, create_problem_position_association_snapshot,
    create_complete_document_problem_association_snapshot, create_complete_document_task_association_snapshot,
    create_image_problem_association_snapshot, create_image_task_association_snapshot,
    create_image_position_association_snapshot, create_drawing_position_association_snapshot,
    create_completed_document_position_association_snapshot, create_image_completed_document_association_snapshot,
    create_parts_position_association_snapshot
)"""

# Initialize logging
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Create SQLAlchemy engine for main database
engine = create_engine(DATABASE_URL)
Session = scoped_session(sessionmaker(bind=engine))
session = Session()

# Attach event listeners for logging and snapshots
"""listen(Area, 'after_insert', log_insert, retval=False)
listen(Area, 'after_update', log_update, retval=False)
listen(Area, 'after_delete', log_delete, retval=False)
listen(EquipmentGroup, 'after_insert', log_insert, retval=False)
listen(EquipmentGroup, 'after_update', log_update, retval=False)
listen(EquipmentGroup, 'after_delete', log_delete, retval=False)
listen(Model, 'after_insert', log_insert, retval=False)
listen(Model, 'after_update', log_update, retval=False)
listen(Model, 'after_delete', log_delete, retval=False)
listen(AssetNumber, 'after_insert', log_insert, retval=False)
listen(AssetNumber, 'after_update', log_update, retval=False)
listen(AssetNumber, 'after_delete', log_delete, retval=False)
listen(Location, 'after_insert', log_insert, retval=False)
listen(Location, 'after_update', log_update, retval=False)
listen(Location, 'after_delete', log_delete, retval=False)

# Create SQLAlchemy engine for revision control session
revision_control_engine = create_engine(f'sqlite:///{REVISION_CONTROL_DB_PATH}')
RevisionControlSession = scoped_session(sessionmaker(bind=revision_control_engine))"""

def delete_duplicates(session, model, attribute):
    # Find duplicate records based on the specified attribute
    duplicates = session.query(getattr(model, attribute), func.count()).group_by(getattr(model, attribute)).having(func.count() > 1)
    
    # Iterate over duplicate records and keep one instance while deleting the rest
    for attr_value, count in duplicates:
        records = session.query(model).filter(getattr(model, attribute) == attr_value).all()
        for record in records[1:]:  # Keep the first instance, delete the rest
            session.delete(record)

def backup_database_relationships(session):
    """
    Function to create a backup of the database.
    """
    try:
        # Define the directory to store backup Excel files
        backup_directory = os.path.join(BASE_DIR, "Database", "DB_LOADSHEETS_BACKUP")
        
        # Create the backup directory if it doesn't exist
        if not os.path.exists(backup_directory):
            os.makedirs(backup_directory)

        # Get the current date and time for the timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Create the Excel file name with the timestamp
        excel_file_name = f"equipment_relationships_table_data_database_backup_{timestamp}.xlsx"
        excel_file_path = os.path.join(backup_directory, excel_file_name)

        # Extract data from each table and create DataFrames
        area_data = [(area.name, area.description) for area in session.query(Area).all()]
        equipment_group_data = [(group.name, group.area_id) for group in session.query(EquipmentGroup).all()]
        model_data = [(model.name, model.description, model.equipment_group_id) for model in session.query(Model).all()]
        asset_number_data = [(asset.number, asset.model_id, asset.description) for asset in session.query(AssetNumber).all()]
        location_data = [(location.name, location.model_id) for location in session.query(Location).all()]

        # Create DataFrames from the extracted data
        df_area = pd.DataFrame(area_data, columns=['name', 'description'])
        df_equipment_group = pd.DataFrame(equipment_group_data, columns=['name', 'area_id'])
        df_model = pd.DataFrame(model_data, columns=['name', 'description', 'equipment_group_id'])
        df_asset_number = pd.DataFrame(asset_number_data, columns=['number', 'model_id', 'description'])
        df_location = pd.DataFrame(location_data, columns=['name', 'model_id'])

        # Write DataFrames to the Excel file
        with pd.ExcelWriter(excel_file_path) as writer:
            df_area.to_excel(writer, sheet_name='Area', index=False)
            df_equipment_group.to_excel(writer, sheet_name='EquipmentGroup', index=False)
            df_model.to_excel(writer, sheet_name='Model', index=False)
            df_asset_number.to_excel(writer, sheet_name='AssetNumber', index=False)
            df_location.to_excel(writer, sheet_name='Location', index=False)

        print("Database backup created successfully:", excel_file_name)
    except Exception as e:
        print("Error creating database backup:", e)

def upload_data_from_excel(file_path, engine):
    # Load Excel file into pandas DataFrame
    print("Loading 'Area' DataFrame...")
    df_area = pd.read_excel(file_path, sheet_name='Area')
    print("Number of rows in 'Area' DataFrame:", len(df_area))
    print("Number of columns in 'Area' DataFrame:", len(df_area.columns))

    print("Loading 'EquipmentGroup' DataFrame...")
    df_equipment_group = pd.read_excel(file_path, sheet_name='EquipmentGroup')
    print("Number of rows in 'EquipmentGroup' DataFrame:", len(df_equipment_group))
    print("Number of columns in 'EquipmentGroup' DataFrame:", len(df_equipment_group.columns))

    print("Loading 'Model' DataFrame...")
    df_model = pd.read_excel(file_path, sheet_name='Model')
    print("Number of rows in 'Model' DataFrame:", len(df_model))
    print("Number of columns in 'Model' DataFrame:", len(df_model.columns))
    print("Column names in 'Model' DataFrame:", df_model.columns)

    print("Loading 'AssetNumber' DataFrame...")
    df_asset_number = pd.read_excel(file_path, sheet_name='AssetNumber')
    print("Number of rows in 'AssetNumber' DataFrame:", len(df_asset_number))
    print("Number of columns in 'AssetNumber' DataFrame:", len(df_asset_number.columns))

    print("Loading 'Location' DataFrame...")
    df_location = pd.read_excel(file_path, sheet_name='Location')
    print("Number of rows in 'Location' DataFrame:", len(df_location))
    print("Number of columns in 'Location' DataFrame:", len(df_location.columns))
    print("Column names in 'Location' DataFrame:", df_location.columns)

    # Create session
    session = Session()

    try:
        # Backup the database before making any changes
        backup_database_relationships(session)
        
        # Insert or update data into 'Area' table
        for _, row in df_area.iterrows():
            # Strip leading and trailing spaces from the area name
            area_name = row['name'].strip()
            area = session.query(Area).filter_by(name=area_name).first()
            if area:
                area.description = row['description']
            else:
                area = Area(name=area_name, description=row['description'])
                session.add(area)

        # Insert or update data into 'EquipmentGroup' table
        for _, row in df_equipment_group.iterrows():
            if 'area_id' in df_equipment_group.columns:
                area_id = row['area_id']
            else:
                area_id = None
            # Strip leading and trailing spaces from the equipment group name
            equipment_group_name = row['name'].strip()
            equipment_group = session.query(EquipmentGroup).filter_by(name=equipment_group_name).first()
            if equipment_group:
                equipment_group.area_id = area_id
            else:
                equipment_group = EquipmentGroup(name=equipment_group_name, area_id=area_id)
                session.add(equipment_group)

        # Insert or update data into 'Model' table
        for _, row in df_model.iterrows():
            if 'equipment_group_id' in df_model.columns:
                equipment_group_id = row['equipment_group_id']
            else:
                equipment_group_id = None
            # Strip leading and trailing spaces from the model name
            model_name = row['name'].strip()
            model = session.query(Model).filter_by(name=model_name).first()
            if model:
                model.description = row['description']
                model.equipment_group_id = equipment_group_id
            else:
                equipment_group = session.query(EquipmentGroup).filter_by(id=equipment_group_id).first()
                model = Model(name=model_name, description=row['description'], equipment_group=equipment_group)
                session.add(model)
                
        # Insert or update data into 'AssetNumber' table
        for _, row in df_asset_number.iterrows():
            if 'model_id' in df_asset_number.columns:
                model_id = row['model_id']
            else:
                model_id = None
            # Strip leading and trailing spaces from the asset number
            asset_number_name = row['number'].strip()
            asset_number = session.query(AssetNumber).filter_by(number=asset_number_name).first()
            if asset_number:
                asset_number.model_id = model_id
                asset_number.description = row['description']
            else:
                asset_number = AssetNumber(number=asset_number_name, model_id=model_id, description=row['description'])
                session.add(asset_number)

        # Insert or update data into 'Location' table
        print(f'inserting into location table')
        for index, row in df_location.iterrows():
            print("Processing row:", index)
            if 'model_id' in df_location.columns:
                model_id = row['model_id']
            else:
                model_id = None
            # Strip leading and trailing spaces from the location name
            location_name = row['name'].strip()
            print("Location name:", location_name)
            location = session.query(Location).filter_by(name=location_name).first()
            if location:
                location.model_id = model_id
            else:
                location = Location(name=location_name, model_id=model_id)
                session.add(location)

        delete_duplicates(session, Area, 'name')
        delete_duplicates(session, EquipmentGroup, 'name')
        delete_duplicates(session, Model, 'name')
        delete_duplicates(session, AssetNumber, 'number')
        delete_duplicates(session, Location, 'name')

        # Commit the session
        session.commit()
        print("Data uploaded successfully!")

        # Add version info and create snapshots
        with revision_control_session() as rev_session:
            new_version = VersionInfo(version_number=1, description="Initial version")
            rev_session.add(new_version)
            rev_session.commit()

            for area in session.query(Area).all():
                create_snapshot(area, rev_session, AreaSnapshot)
            for equipment_group in session.query(EquipmentGroup).all():
                create_snapshot(equipment_group, rev_session, EquipmentGroupSnapshot)
            for model in session.query(Model).all():
                create_snapshot(model, rev_session, ModelSnapshot)
            for asset_number in session.query(AssetNumber).all():
                create_snapshot(asset_number, rev_session, AssetNumberSnapshot)
            for location in session.query(Location).all():
                create_snapshot(location, rev_session, LocationSnapshot)

    except Exception as e:
        print("An error occurred:", e)
        session.rollback()
    finally:
        session.close()

if __name__ == "__main__":
    # Debugging print statement to verify DB_LOADSHEET value
    print("DB_LOADSHEET:", DB_LOADSHEET)
    
    # Provide the new name for your Excel file
    excel_file_path = os.path.join(DB_LOADSHEET, "load_equipment_relationships_table_data.xlsx")

    # Call the function to upload data
    upload_data_from_excel(excel_file_path, engine)
