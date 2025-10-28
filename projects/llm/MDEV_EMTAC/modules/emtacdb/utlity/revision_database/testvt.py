import os
import logging
from concurrent.futures import ThreadPoolExecutor
from sqlalchemy import inspect
from modules.emtacdb.emtacdb_fts import (
    SiteLocation, SiteLocationSnapshot, Position, PositionSnapshot, Area, AreaSnapshot,
    EquipmentGroup, EquipmentGroupSnapshot, Model, ModelSnapshot, AssetNumber, AssetNumberSnapshot,
    Part, PartSnapshot, Image, ImageSnapshot, ImageEmbedding, ImageEmbeddingSnapshot, Drawing, DrawingSnapshot,
    Document, DocumentSnapshot, CompleteDocument, CompleteDocumentSnapshot, Problem, ProblemSnapshot,
    Task, TaskSnapshot, DrawingPartAssociation, DrawingPartAssociationSnapshot,
    PartProblemAssociation, PartProblemAssociationSnapshot, PartTaskAssociation, PartTaskAssociationSnapshot,
    DrawingProblemAssociation, DrawingProblemAssociationSnapshot, DrawingTaskAssociation, DrawingTaskAssociationSnapshot,
    ProblemPositionAssociation, ProblemPositionAssociationSnapshot, CompleteDocumentProblemAssociation,
    CompleteDocumentProblemAssociationSnapshot, CompleteDocumentTaskAssociation, CompleteDocumentTaskAssociationSnapshot,
    ImageProblemAssociation, ImageProblemAssociationSnapshot, ImageTaskAssociation, ImageTaskAssociationSnapshot,
    ImagePositionAssociation, ImagePositionAssociationSnapshot, DrawingPositionAssociation, DrawingPositionAssociationSnapshot,
    CompletedDocumentPositionAssociation, CompletedDocumentPositionAssociationSnapshot, ImageCompletedDocumentAssociation,
    ImageCompletedDocumentAssociationSnapshot, PartsPositionImageAssociation
)
from modules.emtacdb.emtac_revision_control_db import (
    VersionInfo, RevisionControlBase
)
from modules.configuration.config_env import DatabaseConfig  # Import DatabaseConfig

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Initialize database configuration
db_config = DatabaseConfig()

# Correct session and base initialization
main_session = db_config.get_main_session()
revision_session = db_config.get_revision_control_session()

def create_all_tables():
    """Ensure that all tables are created before proceeding."""
    logger.info(f"Ensuring all tables exist in '{db_config.revision_control_db_path}'")
    RevisionControlBase.metadata.create_all(db_config.revision_control_engine)
    logger.info("All tables have been created or verified as existing.")

def initialize_snapshots(main_session, revision_control_session):
    try:
        inspector = inspect(db_config.main_engine)
        tables = inspector.get_table_names()
        logger.info(f"Tables in the main database '{db_config.main_database_url}': {tables}")

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
            create_solution_snapshot(instance, revision_control_session)
        
        for instance in main_session.query(DrawingPartAssociation).all():
            create_drawing_part_association_snapshot(instance, revision_control_session)
        
        for instance in main_session.query(PartsPositionImageAssociation).all():
            create_parts_position_association_snapshot(instance, revision_control_session)

        for instance in main_session.query(PartProblemAssociation).all():
            create_part_problem_association_snapshot(instance, revision_control_session)
        
        for instance in main_session.query(PartTaskAssociation).all():
            create_part_solution_association_snapshot(instance, revision_control_session)
        
        for instance in main_session.query(DrawingProblemAssociation).all():
            create_drawing_problem_association_snapshot(instance, revision_control_session)
        
        for instance in main_session.query(DrawingTaskAssociation).all():
            create_drawing_solution_association_snapshot(instance, revision_control_session)
        
        for instance in main_session.query(ProblemPositionAssociation).all():
            create_problem_position_association_snapshot(instance, revision_control_session)
        
        for instance in main_session.query(CompleteDocumentProblemAssociation).all():
            create_complete_document_problem_association_snapshot(instance, revision_control_session)
        
        for instance in main_session.query(CompleteDocumentTaskAssociation).all():
            create_complete_document_solution_association_snapshot(instance, revision_control_session)
        
        for instance in main_session.query(ImageProblemAssociation).all():
            create_image_problem_association_snapshot(instance, revision_control_session)
        
        for instance in main_session.query(ImageTaskAssociation).all():
            create_image_solution_association_snapshot(instance, revision_control_session)
        
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
    # Create all tables to ensure the database is initialized properly
    create_all_tables()

    try:
        insert_initial_version(revision_session)
        create_snapshots_concurrently(main_session, revision_session)
    except Exception as e:
        logger.error(f"An unexpected error occurred: {e}")
    finally:
        main_session.close()
        revision_session.close()

    logger.info("Script execution completed.")
