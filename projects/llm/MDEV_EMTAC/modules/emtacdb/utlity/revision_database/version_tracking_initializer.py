# version_tracking_initializer.py
import os
import logging
from concurrent.futures import ThreadPoolExecutor
from sqlalchemy import create_engine, inspect, event
from sqlalchemy.orm import sessionmaker, scoped_session
from auditlog import AuditLog
from modules.emtacdb.emtacdb_fts import (
    SiteLocation, Position, Area, EquipmentGroup, Model, AssetNumber, Part, Image, ImageEmbedding, Drawing, Document,
    CompleteDocument, Problem, Task, DrawingPartAssociation, PartProblemAssociation, PartTaskAssociation,
    DrawingProblemAssociation, DrawingTaskAssociation, ProblemPositionAssociation, CompleteDocumentProblemAssociation,
    CompleteDocumentTaskAssociation, ImageProblemAssociation, ImageTaskAssociation, ImagePositionAssociation,
    DrawingPositionAssociation, CompletedDocumentPositionAssociation, ImageCompletedDocumentAssociation,
    PartsPositionImageAssociation
)

from modules.emtacdb.emtac_revision_control_db import (
    VersionInfo, RevisionControlBase, SiteLocationSnapshot, PositionSnapshot, AreaSnapshot, EquipmentGroupSnapshot, ModelSnapshot,
    AssetNumberSnapshot, PartSnapshot, ImageSnapshot, ImageEmbeddingSnapshot, DrawingSnapshot,
    DocumentSnapshot, CompleteDocumentSnapshot, ProblemSnapshot, TaskSnapshot,
    DrawingPartAssociationSnapshot, PartProblemAssociationSnapshot, PartTaskAssociationSnapshot,
    DrawingProblemAssociationSnapshot, DrawingTaskAssociationSnapshot,
    ProblemPositionAssociationSnapshot, CompleteDocumentProblemAssociationSnapshot,
    CompleteDocumentTaskAssociationSnapshot, ImageProblemAssociationSnapshot,
    ImageTaskAssociationSnapshot, ImagePositionAssociationSnapshot, DrawingPositionAssociationSnapshot,
    CompletedDocumentPositionAssociationSnapshot, ImageCompletedDocumentAssociationSnapshot
)
from snapshot_utils import (
    create_sitlocation_snapshot, create_position_snapshot, create_snapshot,
    create_area_snapshot, create_equipment_group_snapshot, create_model_snapshot, create_asset_number_snapshot,
    create_part_snapshot, create_image_snapshot, create_image_embedding_snapshot, create_drawing_snapshot,
    create_document_snapshot, create_complete_document_snapshot, create_problem_snapshot, create_task_snapshot,
    create_drawing_part_association_snapshot, create_part_problem_association_snapshot, create_part_task_association_snapshot,
    create_drawing_problem_association_snapshot, create_drawing_task_association_snapshot, create_problem_position_association_snapshot,
    create_complete_document_problem_association_snapshot, create_complete_document_task_association_snapshot,
    create_image_problem_association_snapshot, create_image_task_association_snapshot, create_image_position_association_snapshot,
    create_drawing_position_association_snapshot, create_completed_document_position_association_snapshot, create_image_completed_document_association_snapshot,
    create_parts_position_association_snapshot
)

from modules.configuration.config import DATABASE_PATH, REVISION_CONTROL_DB_PATH

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Create engines and sessions
main_engine = create_engine(f'sqlite:///{DATABASE_PATH}')
revision_control_engine = create_engine(f'sqlite:///{REVISION_CONTROL_DB_PATH}')
MainSession = scoped_session(sessionmaker(bind=main_engine))
RevisionControlSession = scoped_session(sessionmaker(bind=revision_control_engine))

# Log database connection info
logger.info(f"Attempting to connect to main database at '{DATABASE_PATH}'")
logger.info(f"Attempting to connect to revision control database at '{REVISION_CONTROL_DB_PATH}'")

# Correct the session initialization
main_session = MainSession()
revision_session = RevisionControlSession()  # Correct session creation

def initialize_snapshots(main_session, revision_control_session):
    try:
        inspector = inspect(main_engine)
        tables = inspector.get_table_names()
        logger.info(f"Tables in the main database '{DATABASE_PATH}': {tables}")

        expected_tables = [
            'site_location', 'position', 'area', 'equipment_group', 'model', 'asset_number', 'part',
            'image', 'image_embedding', 'drawing', 'document', 'complete_document', 'problem', 'solution',
            'drawing_part_association', 'part_problem_association', 'part_solution_association',
            'drawing_problem_association', 'drawing_solution_association', 'problem_position_association',
            'complete_document_problem_association', 'complete_document_solution_association',
            'image_problem_association', 'image_solution_association', 'image_position_association',
            'drawing_position_association', 'completed_document_position_association', 'image_completed_document_association',
            'audit_log'
        ]

        missing_tables = [table for table in expected_tables if table not in tables]
        if missing_tables:
            logger.error(f"The following expected tables are missing: {missing_tables}")
            return

        entities_to_snapshot = [
            (SiteLocation, SiteLocationSnapshot),
            (Position, PositionSnapshot),
            (Area, AreaSnapshot),
            (EquipmentGroup, EquipmentGroupSnapshot),
            (Model, ModelSnapshot),
            (AssetNumber, AssetNumberSnapshot),
            (Part, PartSnapshot),
            (Image, ImageSnapshot),
            (ImageEmbedding, ImageEmbeddingSnapshot),
            (Drawing, DrawingSnapshot),
            (Document, DocumentSnapshot),
            (CompleteDocument, CompleteDocumentSnapshot),
            (Problem, ProblemSnapshot),
            (Task, TaskSnapshot),
            (DrawingPartAssociation, DrawingPartAssociationSnapshot),
            (PartProblemAssociation, PartProblemAssociationSnapshot),
            (PartTaskAssociation, PartTaskAssociationSnapshot),
            (DrawingProblemAssociation, DrawingProblemAssociationSnapshot),
            (DrawingTaskAssociation, DrawingTaskAssociationSnapshot),
            (ProblemPositionAssociation, ProblemPositionAssociationSnapshot),
            (CompleteDocumentProblemAssociation, CompleteDocumentProblemAssociationSnapshot),
            (CompleteDocumentTaskAssociation, CompleteDocumentTaskAssociationSnapshot),
            (ImageProblemAssociation, ImageProblemAssociationSnapshot),
            (ImageTaskAssociation, ImageTaskAssociationSnapshot),
            (ImagePositionAssociation, ImagePositionAssociationSnapshot),
            (DrawingPositionAssociation, DrawingPositionAssociationSnapshot),
            (CompletedDocumentPositionAssociation, CompletedDocumentPositionAssociationSnapshot),
            (ImageCompletedDocumentAssociation, ImageCompletedDocumentAssociationSnapshot)
        ]

        for entity, snapshot_class in entities_to_snapshot:
            instances = main_session.query(entity).all()
            for instance in instances:
                create_snapshot(instance, snapshot_class, revision_control_session)

        logger.info("All initial snapshots created successfully.")

    except Exception as e:
        logger.error(f"An error occurred while creating snapshots: {e}")
    finally:
        main_session.close()
        revision_control_session.close()

def insert_initial_version(session):
    try:
        existing_versions = session.query(VersionInfo).count()
        if existing_versions > 0:
            logger.info("Version tracking already initialized.")
            return

        version_number = int(input("Enter the initial version number (integer): "))
        description = input("Enter the description for the initial version: ")

        initial_version = VersionInfo(version_number=version_number, description=description)
        session.add(initial_version)
        session.commit()
        logger.info(f"Inserted initial version: {version_number} with description: '{description}'")
    except ValueError:
        logger.error("Invalid input. Please enter a valid integer for the version number.")
    except Exception as e:
        logger.error(f"An error occurred: {e}")

def create_all_snapshots(main_session, revision_control_session):
    current_version = revision_control_session.query(VersionInfo).order_by(VersionInfo.id.desc()).first()
    if not current_version:
        logger.error("No version found. Cannot create snapshots.")
        return

    confirm = input("Do you want to create a snapshot of all tables? (yes/no): ").strip().lower()
    if confirm != 'yes':
        logger.info("Snapshot creation aborted by the user.")
        return

    logger.info("Starting snapshot creation for all tables.")

    try:
        for instance in main_session.query(SiteLocation).all():
            create_sitlocation_snapshot(instance, revision_control_session)
        
        for instance in main_session.query(Position).all():
            create_position_snapshot(instance, revision_control_session)
        
        for instance in main_session.query(Area).all():
            create_area_snapshot(instance, revision_control_session)
        
        for instance in main_session.query(EquipmentGroup).all():
            create_equipment_group_snapshot(instance, revision_control_session)
        
        for instance in main_session.query(Model).all():
            create_model_snapshot(instance, revision_control_session)
        
        for instance in main_session.query(AssetNumber).all():
            create_asset_number_snapshot(instance, revision_control_session)
        
        for instance in main_session.query(Part).all():
            create_part_snapshot(instance, revision_control_session)
        
        for instance in main_session.query(Image).all():
            create_image_snapshot(instance, revision_control_session)
        
        for instance in main_session.query(ImageEmbedding).all():
            create_image_embedding_snapshot(instance, revision_control_session)
        
        for instance in main_session.query(Drawing).all():
            create_drawing_snapshot(instance, revision_control_session)
        
        for instance in main_session.query(Document).all():
            create_document_snapshot(instance, revision_control_session)
        
        for instance in main_session.query(CompleteDocument).all():
            create_complete_document_snapshot(instance, revision_control_session)
        
        for instance in main_session.query(Problem).all():
            create_problem_snapshot(instance, revision_control_session)
        
        for instance in main_session.query(Task).all():
            create_task_snapshot(instance, revision_control_session)
        
        for instance in main_session.query(DrawingPartAssociation).all():
            create_drawing_part_association_snapshot(instance, revision_control_session)
        
        for instance in main_session.query(PartsPositionImageAssociation).all():
            create_parts_position_association_snapshot(instance, revision_control_session)

        for instance in main_session.query(PartProblemAssociation).all():
            create_part_problem_association_snapshot(instance, revision_control_session)
        
        for instance in main_session.query(PartTaskAssociation).all():
            create_part_task_association_snapshot(instance, revision_control_session)
        
        for instance in main_session.query(DrawingProblemAssociation).all():
            create_drawing_problem_association_snapshot(instance, revision_control_session)
        
        for instance in main_session.query(DrawingTaskAssociation).all():
            create_drawing_task_association_snapshot(instance, revision_control_session)
        
        for instance in main_session.query(ProblemPositionAssociation).all():
            create_problem_position_association_snapshot(instance, revision_control_session)
        
        for instance in main_session.query(CompleteDocumentProblemAssociation).all():
            create_complete_document_problem_association_snapshot(instance, revision_control_session)
        
        for instance in main_session.query(CompleteDocumentTaskAssociation).all():
            create_complete_document_task_association_snapshot(instance, revision_control_session)
        
        for instance in main_session.query(ImageProblemAssociation).all():
            create_image_problem_association_snapshot(instance, revision_control_session)
        
        for instance in main_session.query(ImageTaskAssociation).all():
            create_image_task_association_snapshot(instance, revision_control_session)
        
        for instance in main_session.query(PartsPositionImageAssociation).all():
            create_parts_position_association_snapshot(instance, revision_control_session)
        
        for instance in main_session.query(ImagePositionAssociation).all():
            create_image_position_association_snapshot(instance, revision_control_session)
        
        for instance in main_session.query(DrawingPositionAssociation).all():
            create_drawing_position_association_snapshot(instance, revision_control_session)
        
        for instance in main_session.query(CompletedDocumentPositionAssociation).all():
            create_completed_document_position_association_snapshot(instance, revision_control_session)
        
        for instance in main_session.query(ImageCompletedDocumentAssociation).all():
            create_image_completed_document_association_snapshot(instance, revision_control_session)

        logger.info("All snapshots created successfully.")
    except Exception as e:
        logger.error(f"An error occurred during snapshot creation: {e}")

# Function to set SQLite PRAGMA settings
def set_sqlite_pragmas(dbapi_connection, connection_record):
    cursor = dbapi_connection.cursor()
    cursor.execute('PRAGMA synchronous = OFF;')
    cursor.execute('PRAGMA journal_mode = MEMORY;')
    cursor.execute('PRAGMA temp_store = MEMORY;')
    cursor.execute('PRAGMA cache_size = -64000;')
    cursor.close()

# Apply PRAGMA settings for both engines
event.listen(main_engine, 'connect', set_sqlite_pragmas)
event.listen(revision_control_engine, 'connect', set_sqlite_pragmas)

# Function to list all tables for debugging purposes
def list_tables(engine):
    inspector = inspect(engine)
    tables = inspector.get_table_names()
    logger.info(f"Tables in the current database: {tables}")

# Function to create missing tables
def create_missing_tables(engine, base):
    inspector = inspect(engine)
    existing_tables = inspector.get_table_names()
    all_tables = base.metadata.tables.keys()
    missing_tables = set(all_tables) - set(existing_tables)
    
    if missing_tables:
        logger.info(f"Missing tables found: {missing_tables}. Creating them.")
        base.metadata.create_all(engine, tables=[base.metadata.tables[table] for table in missing_tables])
        logger.info("Missing tables created successfully.")
    else:
        logger.info("No missing tables found. All tables are up-to-date.")

# Function to create snapshots concurrently
def create_snapshots_concurrently(main_session, revision_control_session):
    logger.info("Starting concurrent snapshot creation process.")
    
    try:
        num_workers = max(1, os.cpu_count() - 1)  # Use all cores minus one or at least one
        logger.info(f"Number of workers used for concurrent processing: {num_workers}")

        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            future = executor.submit(create_all_snapshots, main_session, revision_control_session)
            # Wait for the future to complete and handle exceptions if any
            try:
                future.result()  # This will raise an exception if the callable raised one
                logger.info("Snapshot creation completed successfully.")
            except Exception as e:
                logger.error(f"Error during snapshot creation: {e}")
                
    except Exception as e:
        logger.error(f"An error occurred in the concurrent snapshot creation process: {e}")

    logger.info("Concurrent snapshot creation process finished.")

if __name__ == "__main__":
    # Initialize the database if it doesn't exist
    if not os.path.exists(REVISION_CONTROL_DB_PATH):
        logger.info(f"Database '{REVISION_CONTROL_DB_PATH}' does not exist. Creating new database.")
        logger.debug("Creating audit_log table first...")
        AuditLog.__table__.create(bind=revision_control_engine, checkfirst=True)
        logger.debug("audit_log table created.")
        logger.debug("Creating all other tables...")
        RevisionControlBase.metadata.create_all(revision_control_engine)
        logger.debug("All other tables created.")
    else:
        logger.info(f"Database '{REVISION_CONTROL_DB_PATH}' already exists.")
        # Ensure all tables are created or updated
        create_missing_tables(revision_control_engine, RevisionControlBase)
        # Ensure audit_log table exists
        AuditLog.__table__.create(bind=revision_control_engine, checkfirst=True)
        logger.info("audit_log table creation checked/completed.")

    # Correct the session initialization
    main_session = MainSession()
    revision_session = RevisionControlSession()  # Correct session creation

    try:
        # Prompt for initial version data and insert
        insert_initial_version(revision_session)

        # Create initial snapshots (this includes user confirmation within the function)
        create_snapshots_concurrently(main_session, revision_session)
    except Exception as e:
        # Ensure that the logger is defined before usage
        logger.error(f"An unexpected error occurred: {e}")
    finally:
        # Close the sessions
        main_session.close()
        revision_session.close()

    logger.info("Script execution completed.")
