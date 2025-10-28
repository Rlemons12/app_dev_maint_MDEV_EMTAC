# emtac_revision_control_db.py
# Todo: Remove from project. Function is now handled by setup_manager.py
import os
from datetime import datetime
from sqlalchemy import (DateTime, Column, ForeignKey, Integer, LargeBinary, String, Float, Text, create_engine)
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship, scoped_session, sessionmaker
from modules.configuration.config import DATABASE_DIR
import logging

# Initialize logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Revision control database configuration
REVISION_CONTROL_DB_PATH = os.path.join(DATABASE_DIR, 'emtac_revision_control_db.db')
revision_control_engine = create_engine(f'sqlite:///{REVISION_CONTROL_DB_PATH}')
RevisionControlBase = declarative_base()
RevisionControlSession = scoped_session(sessionmaker(bind=revision_control_engine))  # Use distinct name
revision_control_session = RevisionControlSession()



class VersionInfo(RevisionControlBase):
    __tablename__ = 'version_info'
    __table_args__ = {'extend_existing': True}
    id = Column(Integer, primary_key=True, autoincrement=True)
    version_number = Column(Integer, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    description = Column(String, nullable=True)

class SiteLocationSnapshot(RevisionControlBase):
    __tablename__ = 'site_location_snapshot'
    snapshot_id = Column(Integer, primary_key=True, autoincrement=True)
    id = Column(Integer, nullable=False)
    title = Column(String, nullable=False)
    room_number = Column(String, nullable=False)
    version_id = Column(Integer, ForeignKey('version_info.id'), nullable=False)
    version = relationship("VersionInfo")

class PositionSnapshot(RevisionControlBase):
    __tablename__ = 'position_snapshot'
    snapshot_id = Column(Integer, primary_key=True, autoincrement=True)
    id = Column(Integer, nullable=False)
    area_id = Column(Integer, nullable=True)
    equipment_group_id = Column(Integer, nullable=True)
    model_id = Column(Integer, nullable=True)
    asset_number_id = Column(Integer, nullable=True)
    location_id = Column(Integer, nullable=True)
    site_location_id = Column(Integer, nullable=True)
    version_id = Column(Integer, ForeignKey('version_info.id'), nullable=False)
    version = relationship("VersionInfo")

class AreaSnapshot(RevisionControlBase):
    __tablename__ = 'area_snapshot'
    snapshot_id = Column(Integer, primary_key=True, autoincrement=True)
    id = Column(Integer, nullable=False)
    name = Column(String, nullable=False)
    description = Column(String)
    version_id = Column(Integer, ForeignKey('version_info.id'), nullable=False)
    version = relationship("VersionInfo")

class EquipmentGroupSnapshot(RevisionControlBase):
    __tablename__ = 'equipment_group_snapshot'
    snapshot_id = Column(Integer, primary_key=True, autoincrement=True)
    id = Column(Integer, nullable=False)
    name = Column(String, nullable=False)
    area_id = Column(Integer, nullable=False)
    version_id = Column(Integer, ForeignKey('version_info.id'), nullable=False)
    version = relationship("VersionInfo")

class ModelSnapshot(RevisionControlBase):
    __tablename__ = 'model_snapshot'
    snapshot_id = Column(Integer, primary_key=True, autoincrement=True)
    id = Column(Integer, nullable=False)
    name = Column(String, nullable=False)
    description = Column(String)
    equipment_group_id = Column(Integer, nullable=True)
    version_id = Column(Integer, ForeignKey('version_info.id'), nullable=False)
    version = relationship("VersionInfo")

class AssetNumberSnapshot(RevisionControlBase):
    __tablename__ = 'asset_number_snapshot'
    snapshot_id = Column(Integer, primary_key=True, autoincrement=True)
    id = Column(Integer, nullable=False)
    number = Column(String, nullable=False)
    description = Column(String)
    model_id = Column(Integer, nullable=True)
    version_id = Column(Integer, ForeignKey('version_info.id'), nullable=False)
    version = relationship("VersionInfo")

class LocationSnapshot(RevisionControlBase):
    __tablename__ = 'location_snapshot'
    snapshot_id = Column(Integer, primary_key=True, autoincrement=True)
    id = Column(Integer, nullable=False)
    name = Column(String, nullable=False)
    model_id = Column(Integer, nullable=True)
    version_id = Column(Integer, ForeignKey('version_info.id'), nullable=False)
    version = relationship("VersionInfo")

class PartSnapshot(RevisionControlBase):
    __tablename__ = 'part_snapshot'
    snapshot_id = Column(Integer, primary_key=True, autoincrement=True)
    id = Column(Integer, nullable=False)
    part_number = Column(String)
    name = Column(String)
    oem_mfg = Column(String)
    model = Column(String)
    class_flag = Column(String)
    ud6 = Column(String)
    type = Column(String)
    notes = Column(String)
    documentation = Column(String)
    version_id = Column(Integer, ForeignKey('version_info.id'), nullable=False)
    version = relationship("VersionInfo")

class ImageSnapshot(RevisionControlBase):
    __tablename__ = 'image_snapshot'
    snapshot_id = Column(Integer, primary_key=True, autoincrement=True)
    id = Column(Integer, nullable=False)
    title = Column(String, nullable=False)
    description = Column(String, nullable=False)
    file_path = Column(String, nullable=False)
    version_id = Column(Integer, ForeignKey('version_info.id'), nullable=False)
    version = relationship("VersionInfo")

class ImageEmbeddingSnapshot(RevisionControlBase):
    __tablename__ = 'image_embedding_snapshot'
    snapshot_id = Column(Integer, primary_key=True, autoincrement=True)
    id = Column(Integer, nullable=False)
    image_id = Column(Integer, nullable=False)
    model_name = Column(String, nullable=False)
    model_embedding = Column(LargeBinary, nullable=False)
    version_id = Column(Integer, ForeignKey('version_info.id'), nullable=False)
    version = relationship("VersionInfo")

class DrawingSnapshot(RevisionControlBase):
    __tablename__ = 'drawing_snapshot'
    snapshot_id = Column(Integer, primary_key=True, autoincrement=True)
    id = Column(Integer, nullable=False)
    drw_equipment_name = Column(String)
    drw_number = Column(String)
    drw_name = Column(String)
    drw_revision = Column(String)
    drw_spare_part_number = Column(String)
    file_path = Column(String, nullable=False)
    version_id = Column(Integer, ForeignKey('version_info.id'), nullable=False)
    version = relationship("VersionInfo")

class DocumentSnapshot(RevisionControlBase):
    __tablename__ = 'document_snapshot'
    snapshot_id = Column(Integer, primary_key=True, autoincrement=True)
    id = Column(Integer, nullable=False)
    name = Column(String, nullable=False)
    file_path = Column(String, nullable=False)
    content = Column(String)
    complete_document_id = Column(Integer)
    embedding = Column(LargeBinary)
    rev = Column(String)
    version_id = Column(Integer, ForeignKey('version_info.id'), nullable=False)
    version = relationship("VersionInfo")

class CompleteDocumentSnapshot(RevisionControlBase):
    __tablename__ = 'complete_document_snapshot'
    snapshot_id = Column(Integer, primary_key=True, autoincrement=True)
    id = Column(Integer, nullable=False)
    title = Column(String)
    file_path = Column(String)
    content = Column(Text)
    rev = Column(String)
    version_id = Column(Integer, ForeignKey('version_info.id'), nullable=False)
    version = relationship("VersionInfo")

class ProblemSnapshot(RevisionControlBase):
    __tablename__ = 'problem_snapshot'
    snapshot_id = Column(Integer, primary_key=True, autoincrement=True)
    id = Column(Integer, nullable=False)
    name = Column(String, nullable=False)
    description = Column(String, nullable=False)
    version_id = Column(Integer, ForeignKey('version_info.id'), nullable=False)
    version = relationship("VersionInfo")

class TaskSnapshot(RevisionControlBase):
    __tablename__ = 'task_snapshot'
    snapshot_id = Column(Integer, primary_key=True, autoincrement=True)
    id = Column(Integer, nullable=False)
    description = Column(String, nullable=False)
    problem_id = Column(Integer)
    version_id = Column(Integer, ForeignKey('version_info.id'), nullable=False)
    version = relationship("VersionInfo")

class PowerPointSnapshot(RevisionControlBase):
    __tablename__ = 'powerpoint_snapshot'
    snapshot_id = Column(Integer, primary_key=True, autoincrement=True)
    id = Column(Integer, nullable=False)
    title = Column(String, nullable=False)
    ppt_file_path = Column(String, nullable=False)
    pdf_file_path = Column(String, nullable=False)
    description = Column(String, nullable=True)
    complete_document_id = Column(Integer, nullable=True)
    rev = Column(String, nullable=True)
    version_id = Column(Integer, ForeignKey('version_info.id'), nullable=False)
    version = relationship("VersionInfo")

# Junction Table Snapshots
class DrawingPartAssociationSnapshot(RevisionControlBase):
    __tablename__ = 'drawing_part_association_snapshot'
    snapshot_id = Column(Integer, primary_key=True, autoincrement=True)
    id = Column(Integer, nullable=False)
    drawing_id = Column(Integer, nullable=False)
    part_id = Column(Integer, nullable=False)
    version_id = Column(Integer, ForeignKey('version_info.id'), nullable=False)
    version = relationship("VersionInfo")

class PartProblemAssociationSnapshot(RevisionControlBase):
    __tablename__ = 'part_problem_association_snapshot'
    snapshot_id = Column(Integer, primary_key=True, autoincrement=True)
    id = Column(Integer, nullable=False)
    part_id = Column(Integer, nullable=False)
    problem_id = Column(Integer, nullable=False)
    version_id = Column(Integer, ForeignKey('version_info.id'), nullable=False)
    version = relationship("VersionInfo")

class PartTaskAssociationSnapshot(RevisionControlBase):
    __tablename__ = 'part_task_association_snapshot'
    snapshot_id = Column(Integer, primary_key=True, autoincrement=True)
    id = Column(Integer, nullable=False)
    part_id = Column(Integer, nullable=False)
    task_id = Column(Integer, nullable=False)
    version_id = Column(Integer, ForeignKey('version_info.id'), nullable=False)
    version = relationship("VersionInfo")

class DrawingProblemAssociationSnapshot(RevisionControlBase):
    __tablename__ = 'drawing_problem_association_snapshot'
    snapshot_id = Column(Integer, primary_key=True, autoincrement=True)
    id = Column(Integer, nullable=False)
    drawing_id = Column(Integer, nullable=False)
    problem_id = Column(Integer, nullable=False)
    version_id = Column(Integer, ForeignKey('version_info.id'), nullable=False)
    version = relationship("VersionInfo")

class DrawingTaskAssociationSnapshot(RevisionControlBase):
    __tablename__ = 'drawing_task_association_snapshot'
    snapshot_id = Column(Integer, primary_key=True, autoincrement=True)
    id = Column(Integer, nullable=False)
    drawing_id = Column(Integer, nullable=False)
    task_id = Column(Integer, nullable=False)
    version_id = Column(Integer, ForeignKey('version_info.id'), nullable=False)
    version = relationship("VersionInfo")

class BillOfMaterialSnapshot(RevisionControlBase):
    __tablename__ = 'bill_of_material_snapshot'
    snapshot_id = Column(Integer, primary_key=True, autoincrement=True)
    id = Column(Integer, nullable=False)
    part_id = Column(Integer, nullable=False)
    position_id = Column(Integer, nullable=False)
    image_id = Column(Integer, nullable=True)
    quantity = Column(Float, nullable=False)
    comment = Column(String)
    version_id = Column(Integer, ForeignKey('version_info.id'), nullable=False)
    version = relationship("VersionInfo")

class ProblemPositionAssociationSnapshot(RevisionControlBase):
    __tablename__ = 'problem_position_association_snapshot'
    snapshot_id = Column(Integer, primary_key=True, autoincrement=True)
    id = Column(Integer, nullable=False)
    problem_id = Column(Integer, nullable=False)
    position_id = Column(Integer, nullable=False)
    version_id = Column(Integer, ForeignKey('version_info.id'), nullable=False)
    version = relationship("VersionInfo")

class CompleteDocumentProblemAssociationSnapshot(RevisionControlBase):
    __tablename__ = 'complete_document_problem_association_snapshot'
    snapshot_id = Column(Integer, primary_key=True, autoincrement=True)
    id = Column(Integer, nullable=False)
    complete_document_id = Column(Integer, nullable=False)
    problem_id = Column(Integer, nullable=False)
    version_id = Column(Integer, ForeignKey('version_info.id'), nullable=False)
    version = relationship("VersionInfo")

class CompleteDocumentTaskAssociationSnapshot(RevisionControlBase):
    __tablename__ = 'complete_document_task_association_snapshot'
    snapshot_id = Column(Integer, primary_key=True, autoincrement=True)
    id = Column(Integer, nullable=False)
    complete_document_id = Column(Integer, nullable=False)
    task_id = Column(Integer, nullable=False)
    version_id = Column(Integer, ForeignKey('version_info.id'), nullable=False)
    version = relationship("VersionInfo")

class ImageProblemAssociationSnapshot(RevisionControlBase):
    __tablename__ = 'image_problem_association_snapshot'
    snapshot_id = Column(Integer, primary_key=True, autoincrement=True)
    id = Column(Integer, nullable=False)
    image_id = Column(Integer, nullable=False)
    problem_id = Column(Integer, nullable=False)
    version_id = Column(Integer, ForeignKey('version_info.id'), nullable=False)
    version = relationship("VersionInfo")

class ImageTaskAssociationSnapshot(RevisionControlBase):
    __tablename__ = 'image_task_association_snapshot'
    snapshot_id = Column(Integer, primary_key=True, autoincrement=True)
    id = Column(Integer, nullable=False)
    image_id = Column(Integer, nullable=False)
    task_id = Column(Integer, nullable=False)
    version_id = Column(Integer, ForeignKey('version_info.id'), nullable=False)
    version = relationship("VersionInfo")

class PartsPositionImageAssociationSnapshot(RevisionControlBase):
    __tablename__ = 'part_position_snapshot'
    snapshot_id = Column(Integer, primary_key=True, autoincrement=True)
    id = Column(Integer, nullable=False)
    part_id = Column(Integer, nullable=False)
    position_id = Column(Integer, nullable=False)
    version_id = Column(Integer, ForeignKey('version_info.id'), nullable=False)
    version = relationship("VersionInfo")

class ImagePositionAssociationSnapshot(RevisionControlBase):
    __tablename__ = 'image_position_association_snapshot'
    snapshot_id = Column(Integer, primary_key=True, autoincrement=True)
    id = Column(Integer, nullable=False)
    image_id = Column(Integer, nullable=False)
    position_id = Column(Integer, nullable=False)
    version_id = Column(Integer, ForeignKey('version_info.id'), nullable=False)
    version = relationship("VersionInfo")

class DrawingPositionAssociationSnapshot(RevisionControlBase):
    __tablename__ = 'drawing_position_association_snapshot'
    snapshot_id = Column(Integer, primary_key=True, autoincrement=True)
    id = Column(Integer, nullable=False)
    drawing_id = Column(Integer, nullable=False)
    position_id = Column(Integer, nullable=False)
    version_id = Column(Integer, ForeignKey('version_info.id'), nullable=False)
    version = relationship("VersionInfo")

class CompletedDocumentPositionAssociationSnapshot(RevisionControlBase):
    __tablename__ = 'completed_document_position_association_snapshot'
    snapshot_id = Column(Integer, primary_key=True, autoincrement=True)
    id = Column(Integer, nullable=False)
    complete_document_id = Column(Integer, nullable=False)
    position_id = Column(Integer, nullable=False)
    version_id = Column(Integer, ForeignKey('version_info.id'), nullable=False)
    version = relationship("VersionInfo")

class ImageCompletedDocumentAssociationSnapshot(RevisionControlBase):
    __tablename__ = 'image_completed_document_association_snapshot'
    snapshot_id = Column(Integer, primary_key=True, autoincrement=True)
    id = Column(Integer, nullable=False)
    complete_document_id = Column(Integer, nullable=False)
    image_id = Column(Integer, nullable=False)
    version_id = Column(Integer, ForeignKey('version_info.id'), nullable=False)
    version = relationship("VersionInfo")
    
    
