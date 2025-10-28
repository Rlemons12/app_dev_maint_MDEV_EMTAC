import os
import logging
from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import scoped_session, sessionmaker
from modules.emtacdb.emtac_revision_control_db import (
    SiteLocationSnapshot, PositionSnapshot, AreaSnapshot, EquipmentGroupSnapshot, ModelSnapshot,
    AssetNumberSnapshot, PartSnapshot, ImageSnapshot, ImageEmbeddingSnapshot, DrawingSnapshot,
    DocumentSnapshot, CompleteDocumentSnapshot, ProblemSnapshot, TaskSnapshot,
    DrawingPartAssociationSnapshot, PartProblemAssociationSnapshot, PartTaskAssociationSnapshot,
    PartsPositionImageAssociationSnapshot, DrawingProblemAssociationSnapshot, DrawingTaskAssociationSnapshot,
    ProblemPositionAssociationSnapshot, CompleteDocumentProblemAssociationSnapshot,
    CompleteDocumentTaskAssociationSnapshot, ImageProblemAssociationSnapshot,
    ImageTaskAssociationSnapshot, ImagePositionAssociationSnapshot, DrawingPositionAssociationSnapshot,
    CompletedDocumentPositionAssociationSnapshot, ImageCompletedDocumentAssociationSnapshot, VersionInfo
)
from modules.configuration.config import DATABASE_DIR

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Revision control database configuration
REVISION_CONTROL_DB_PATH = os.path.join(DATABASE_DIR, 'emtac_revision_control_db.db')
revision_control_engine = create_engine(f'sqlite:///{REVISION_CONTROL_DB_PATH}')
RevisionControlBase = declarative_base()
RevisionControlSession = scoped_session(sessionmaker(bind=revision_control_engine))  # Use distinct name
revision_control_session = RevisionControlSession()

# Create a scoped session for revision control operations
rev_control_Session = scoped_session(sessionmaker(bind=revision_control_engine))
rev_control_session = rev_control_Session()  # Initialize a session for immediate use

def get_latest_version_info():
    try:
        # Explicitly use revision_control_session
        result = revision_control_session.query(VersionInfo).order_by(VersionInfo.id.desc()).limit(1).offset(0).all()
        for row in result:
            print(row.version_number, row.description)
    except Exception as e:
        revision_control_session.rollback()
        logger.error(f"An error occurred: {e}")
    finally:
        revision_control_session.close()

def add_version_info(version_number, description):
    try:
        new_version = VersionInfo(version_number=version_number, description=description)
        revision_control_session.add(new_version)
        revision_control_session.commit()
        logger.info("New version info added successfully.")
    except Exception as e:
        revision_control_session.rollback()
        logger.error(f"An error occurred: {e}")
    finally:
        revision_control_session.close()

# Ensure the correct session is used for version_info operations
get_latest_version_info()
add_version_info(version_number=1, description="Initial version")

def create_snapshot(instance, session, SnapshotClass):
    data = instance.__dict__.copy()
    data.pop('_sa_instance_state', None)
    snapshot = SnapshotClass(**data)
    
    current_version = session.query(VersionInfo).order_by(VersionInfo.id.desc()).first()
    if current_version:
        snapshot.version_id = current_version.id

    session.add(snapshot)
    session.commit()
    logger.info(f"Created snapshot for {SnapshotClass.__tablename__}: {data}")

# Snapshot creation functions for each entity
def create_sitlocation_snapshot(instance, session):
    logger.info(f"Creating snapshot for SiteLocation: {instance.id}")
    create_snapshot(instance, session, SiteLocationSnapshot)

def create_position_snapshot(instance, session):
    logger.info(f"Creating snapshot for Position: {instance.id}")
    create_snapshot(instance, session, PositionSnapshot)

def create_area_snapshot(instance, session):
    logger.info(f"Creating snapshot for Area: {instance.id}")
    create_snapshot(instance, session, AreaSnapshot)

def create_equipment_group_snapshot(instance, session):
    logger.info(f"Creating snapshot for EquipmentGroup: {instance.id}")
    create_snapshot(instance, session, EquipmentGroupSnapshot)

def create_model_snapshot(instance, session):
    logger.info(f"Creating snapshot for Model: {instance.id}")
    create_snapshot(instance, session, ModelSnapshot)

def create_asset_number_snapshot(instance, session):
    logger.info(f"Creating snapshot for AssetNumber: {instance.id}")
    create_snapshot(instance, session, AssetNumberSnapshot)

def create_part_snapshot(instance, session):
    logger.info(f"Creating snapshot for Part: {instance.id}")
    create_snapshot(instance, session, PartSnapshot)

def create_image_snapshot(instance, session):
    logger.info(f"Creating snapshot for Image: {instance.id}")
    create_snapshot(instance, session, ImageSnapshot)

def create_image_embedding_snapshot(instance, session):
    logger.info(f"Creating snapshot for ImageEmbedding: {instance.id}")
    create_snapshot(instance, session, ImageEmbeddingSnapshot)

def create_drawing_snapshot(instance, session):
    logger.info(f"Creating snapshot for Drawing: {instance.id}")
    create_snapshot(instance, session, DrawingSnapshot)

def create_document_snapshot(instance, session):
    logger.info(f"Creating snapshot for Document: {instance.id}")
    create_snapshot(instance, session, DocumentSnapshot)

def create_complete_document_snapshot(instance, session):
    logger.info(f"Creating snapshot for CompleteDocument: {instance.id}")
    create_snapshot(instance, session, CompleteDocumentSnapshot)

def create_problem_snapshot(instance, session):
    logger.info(f"Creating snapshot for Problem: {instance.id}")
    create_snapshot(instance, session, ProblemSnapshot)

def create_task_snapshot(instance, session):
    logger.info(f"Creating snapshot for Task: {instance.id}")
    create_snapshot(instance, session, TaskSnapshot)

# Snapshot creation functions for junction tables
def create_drawing_part_association_snapshot(instance, session):
    logger.info(f"Creating snapshot for DrawingPartAssociation: {instance.id}")
    create_snapshot(instance, session, DrawingPartAssociationSnapshot)

def create_parts_position_association_snapshot(instance, session):
    logger.info(f"Creating snapshot for PartsPositionImageAssociation: {instance.id}")
    create_snapshot(instance, session, PartsPositionImageAssociationSnapshot)

def create_part_problem_association_snapshot(instance, session):
    logger.info(f"Creating snapshot for PartProblemAssociation: {instance.id}")
    create_snapshot(instance, session, PartProblemAssociationSnapshot)

def create_part_task_association_snapshot(instance, session):
    logger.info(f"Creating snapshot for PartTaskAssociation: {instance.id}")
    create_snapshot(instance, session, PartTaskAssociationSnapshot)

def create_drawing_problem_association_snapshot(instance, session):
    logger.info(f"Creating snapshot for DrawingProblemAssociation: {instance.id}")
    create_snapshot(instance, session, DrawingProblemAssociationSnapshot)

def create_drawing_task_association_snapshot(instance, session):
    logger.info(f"Creating snapshot for DrawingTaskAssociation: {instance.id}")
    create_snapshot(instance, session, DrawingTaskAssociationSnapshot)

def create_problem_position_association_snapshot(instance, session):
    logger.info(f"Creating snapshot for ProblemPositionAssociation: {instance.id}")
    create_snapshot(instance, session, ProblemPositionAssociationSnapshot)

def create_complete_document_problem_association_snapshot(instance, session):
    logger.info(f"Creating snapshot for CompleteDocumentProblemAssociation: {instance.id}")
    create_snapshot(instance, session, CompleteDocumentProblemAssociationSnapshot)

def create_complete_document_task_association_snapshot(instance, session):
    logger.info(f"Creating snapshot for CompleteDocumentTaskAssociation: {instance.id}")
    create_snapshot(instance, session, CompleteDocumentTaskAssociationSnapshot)

def create_image_problem_association_snapshot(instance, session):
    logger.info(f"Creating snapshot for ImageProblemAssociation: {instance.id}")
    create_snapshot(instance, session, ImageProblemAssociationSnapshot)

def create_image_task_association_snapshot(instance, session):
    logger.info(f"Creating snapshot for ImageTaskAssociation: {instance.id}")
    create_snapshot(instance, session, ImageTaskAssociationSnapshot)

def create_image_position_association_snapshot(instance, session):
    logger.info(f"Creating snapshot for ImagePositionAssociation: {instance.id}")
    create_snapshot(instance, session, ImagePositionAssociationSnapshot)

def create_drawing_position_association_snapshot(instance, session):
    logger.info(f"Creating snapshot for DrawingPositionAssociation: {instance.id}")
    create_snapshot(instance, session, DrawingPositionAssociationSnapshot)

def create_completed_document_position_association_snapshot(instance, session):
    logger.info(f"Creating snapshot for CompletedDocumentPositionAssociation: {instance.id}")
    create_snapshot(instance, session, CompletedDocumentPositionAssociationSnapshot)

def create_image_completed_document_association_snapshot(instance, session):
    logger.info(f"Creating snapshot for ImageCompletedDocumentAssociation: {instance.id}")
    create_snapshot(instance, session, ImageCompletedDocumentAssociationSnapshot)
