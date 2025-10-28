import pandas as pd
import sys
import os
from datetime import datetime
import logging

# Ensure the parent directory is in the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from modules.configuration.config import DB_LOADSHEET, DB_LOADSHEETS_BACKUP
from modules.emtacdb.emtacdb_fts import (Area, EquipmentGroup, Model, AssetNumber, Location, Problem, Task,
                                         ProblemPositionAssociation,
                                         load_config_from_db)
from modules.emtacdb.utlity.main_database.database import create_position, split_text_into_chunks
from plugins.ai_modules import generate_embedding
from modules.configuration.config_env import DatabaseConfig

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Initialize DatabaseConfig
db_config = DatabaseConfig()
MainSession = db_config.get_main_session()

# Load the current AI and embedding model configurations from the database
current_ai_model, current_embedding_model = load_config_from_db()

def backup_database():
    session = MainSession  # Directly use MainSession as it is a scoped session
    try:
        # Define the directory to store backup Excel files
        backup_directory = os.path.join(DB_LOADSHEETS_BACKUP)

        # Create the backup directory if it doesn't exist
        if not os.path.exists(backup_directory):
            os.makedirs(backup_directory)

        # Get the current date and time for the timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Create the Excel file name with the timestamp
        excel_file_name = f"tsg_database_backup_{timestamp}.xlsx"
        excel_file_path = os.path.join(backup_directory, excel_file_name)

        # Extract data from each table and create DataFrames
        area_data = [(area.name, area.description) for area in session.query(Area).all()]
        equipment_group_data = [(group.name, group.area_id) for group in session.query(EquipmentGroup).all()]
        model_data = [(model.name, model.description, model.equipment_group_id) for model in session.query(Model).all()]
        asset_number_data = [(asset.number, asset.model_id, asset.description) for asset in session.query(AssetNumber).all()]
        location_data = [(location.name, location.model_id) for location in session.query(Location).all()]
        problem_data = [(problem.name, problem.description) for problem in session.query(Problem).all()]
        solution_data = [(solution.description, solution.problem_id) for solution in session.query(Task).all()]

        # Create DataFrames from the extracted data
        df_area = pd.DataFrame(area_data, columns=['name', 'description'])
        df_equipment_group = pd.DataFrame(equipment_group_data, columns=['name', 'area_id'])
        df_model = pd.DataFrame(model_data, columns=['name', 'description', 'equipment_group_id'])
        df_asset_number = pd.DataFrame(asset_number_data, columns=['number', 'model_id', 'description'])
        df_location = pd.DataFrame(location_data, columns=['name', 'model_id'])
        df_problem = pd.DataFrame(problem_data, columns=['name', 'description'])
        df_solution = pd.DataFrame(solution_data, columns=['description', 'problem_id'])

        # Write DataFrames to the Excel file
        with pd.ExcelWriter(excel_file_path) as writer:
            df_area.to_excel(writer, sheet_name='Area', index=False)
            df_equipment_group.to_excel(writer, sheet_name='EquipmentGroup', index=False)
            df_model.to_excel(writer, sheet_name='Model', index=False)
            df_asset_number.to_excel(writer, sheet_name='AssetNumber', index=False)
            df_location.to_excel(writer, sheet_name='Location', index=False)
            df_problem.to_excel(writer, sheet_name='Problem', index=False)
            df_solution.to_excel(writer, sheet_name='Solution', index=False)

        logger.info("Database backup created successfully: %s", excel_file_name)
    except Exception as e:
        logger.error("Error creating database backup: %s", e)

def add_tsg_loadsheet_to_document_table_db(file_path, area_data, equipment_group_data, model_data, asset_number_data,
                                           location_data, problem_data, solution_data):
    session = MainSession  # Directly use MainSession as it is a scoped session
    try:
        logger.info(f"Reading Excel file: {file_path}")
        # Read the Excel file
        df = pd.read_excel(file_path)

        # Log the column names found in the Excel file
        logger.info(f"Columns in the Excel file: {df.columns}")

        # Check if the necessary columns are present
        required_columns = ['area', 'equipmentgroup', 'model', 'asset', 'location', 'problem', 'solution']
        if not all(column in df.columns for column in required_columns):
            logger.error("Excel file is missing required columns.")
            return None, False

        # Initialize document_id for return value
        document_id = None

        # Iterate through each row in the dataframe
        for index, row in df.iterrows():
            # Extract data from the row
            area_name = row['area'].strip() if pd.notna(row['area']) else None
            equipment_group_name = row['equipmentgroup'].strip() if pd.notna(row['equipmentgroup']) else None
            asset_number = row['asset'].strip() if pd.notna(row['asset']) else None
            location_name = row['location'].strip() if pd.notna(row['location']) else None
            model_name = row['model'].strip() if pd.notna(row['model']) else None
            problem_desc = row['problem'].strip() if pd.notna(row['problem']) else None
            solution_desc = row['solution'].strip() if pd.notna(row['solution']) else None

            logger.info(
                f"Processing row {index + 1} - Problem: {problem_desc}, Area: {area_name}, EquipmentGroup: {equipment_group_name}, Asset: {asset_number}, Location: {location_name}, Model: {model_name}, Solution: {solution_desc}")

            # Query Area table to find matching area_id
            area = session.query(Area).filter(Area.name == area_name).first()
            if area:
                area_id = area.id
            else:
                logger.warning("Area '%s' not found.", area_name)
                continue

            # Query EquipmentGroup table to find matching equipment_group_id
            equipment_group = session.query(EquipmentGroup).filter(EquipmentGroup.name == equipment_group_name).first()
            if equipment_group:
                equipment_group_id = equipment_group.id
            else:
                logger.warning("Equipment group '%s' not found.", equipment_group_name)
                continue

            # Query AssetNumber table to find matching asset_number_id (if provided)
            asset_number_id = None
            if asset_number:
                asset = session.query(AssetNumber).filter(AssetNumber.number == asset_number).first()
                if asset:
                    asset_number_id = asset.id
                else:
                    logger.warning("Asset number '%s' not found.", asset_number)

            # Query Location table to find matching location_id
            location = session.query(Location).filter(Location.name == location_name).first()
            if location:
                location_id = location.id
            else:
                logger.warning("Location '%s' not found.", location_name)
                continue

            # Query Model table to find matching model_id
            model = session.query(Model).filter(Model.name == model_name).first()
            if model:
                model_id = model.id
            else:
                logger.warning("Model '%s' not found.", model_name)
                continue

            # Create a new position if required
            position_id = create_position(area_id, equipment_group_id, model_id, asset_number_id, location_id, None,
                                          session)
            if not position_id:
                logger.error("Failed to create or retrieve position.")
                continue

            # Check if the problem description already exists in the Problem table
            existing_problem = session.query(Problem).filter(Problem.description == problem_desc).first()
            if existing_problem:
                logger.info("Problem with description '%s' already exists.", problem_desc)
                continue

            # Create a new entry in the Problem table
            new_problem = Problem(name=problem_desc, description=problem_desc)
            session.add(new_problem)
            session.commit()

            # Insert solution into Solution table if it exists and is valid
            if solution_desc:
                new_solution = Task(description=solution_desc, problem_id=new_problem.id)
                session.add(new_solution)
                session.commit()
            else:
                logger.warning("Invalid solution description for problem '%s'. Skipping solution insertion.", problem_desc)

            # Create a new entry in the ProblemPositionAssociation table
            problem_position_association = ProblemPositionAssociation(problem_id=new_problem.id, position_id=position_id)
            session.add(problem_position_association)
            session.commit()

            # Concatenate extracted text for embedding
            extracted_text = f"{problem_desc} {area_name} {equipment_group_name} {asset_number} {location_name} {model_name} {solution_desc}"

            logger.info("Splitting extracted text into chunks.")
            # Split text into chunks and process each chunk
            text_chunks = split_text_into_chunks(extracted_text)
            for i, chunk in enumerate(text_chunks):
                padded_chunk = ' '.join(split_text_into_chunks(chunk, pad_token="", max_words=150))

                if current_embedding_model != "NoEmbeddingModel":
                    embeddings = generate_embedding(padded_chunk, current_embedding_model)
                    if embeddings is None:
                        logger.warning(f"Failed to generate embedding for chunk {i + 1} of document: {file_path}")
                    else:
                        store_embedding(document_id, embeddings, current_embedding_model)
                        logger.info(f"Generated and stored embedding for chunk {i + 1} of document: {file_path}")
                else:
                    logger.info(f"No embedding generated for chunk {i + 1} of document: {file_path} because no model is selected.")

        logger.info(f"Successfully processed file: {file_path}")
        return document_id, True
    except Exception as e:
        logger.error(f"An error occurred while adding document from Excel: {e}")
        return None, False

# Main script logic

# Backup the database
backup_database()

# Extract data from each table and create DataFrames
session = MainSession  # Directly use MainSession as it is a scoped session
try:
    area_data = [(area.name, area.description) for area in session.query(Area).all()]
    equipment_group_data = [(group.name, group.area_id) for group in session.query(EquipmentGroup).all()]
    model_data = [(model.name, model.description, model.equipment_group_id) for model in session.query(Model).all()]
    asset_number_data = [(asset.number, asset.model_id, asset.description) for asset in session.query(AssetNumber).all()]
    location_data = [(location.name, location.model_id) for location in session.query(Location).all()]
    problem_data = [(problem.name, problem.description) for problem in session.query(Problem).all()]
    solution_data = [(solution.description, solution.problem_id) for solution in session.query(Task).all()]
finally:
    session.close()  # Ensure the session is closed after the operation

# Load data from load sheet
load_sheet_path = os.path.join(DB_LOADSHEET, "tsg_load_sheet.xlsx")

# Add the load sheet data to the Document table
document_id, success = add_tsg_loadsheet_to_document_table_db(load_sheet_path, area_data, equipment_group_data,
                                                              model_data, asset_number_data, location_data,
                                                              problem_data, solution_data)
if success:
    logger.info(f"Document added with ID: {document_id}")
else:
    logger.error("Failed to add document from Excel")
