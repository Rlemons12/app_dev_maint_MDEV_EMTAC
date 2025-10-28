import os
from sqlalchemy.orm import scoped_session
from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
import logging
from modules.configuration.config import DATABASE_DIR

# Set up logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Revision control database configuration
REVISION_CONTROL_DB_PATH = os.path.join(DATABASE_DIR, 'emtac_revision_control_db.db')
revision_control_engine = create_engine(f'sqlite:///{REVISION_CONTROL_DB_PATH}')
RevisionControlBase = declarative_base()
RevisionControlSession = scoped_session(sessionmaker(bind=revision_control_engine))  # Use distinct name
revision_control_session = RevisionControlSession()

# Register event listeners for all entities
def register_event_listeners():
    # SiteLocation events
    """log_event_listeners('SiteLocation')
    event.listen(SiteLocation, 'after_insert', lambda m, c, t: log_insert(m, c, t, SiteLocationSnapshot  , RevisionControlSession()))
    logger.info("Event listener for 'after_insert' on SiteLocation has been set up.")
    event.listen(SiteLocation, 'after_update', lambda m, c, t: log_update(m, c, t, SiteLocationSnapshot  , RevisionControlSession()))
    logger.info("Event listener for 'after_update' on SiteLocation has been set up.")
    event.listen(SiteLocation, 'after_delete', lambda m, c, t: log_delete(m, c, t, SiteLocationSnapshot  , RevisionControlSession()))
    logger.info("Event listener for 'after_delete' on SiteLocation has been set up.")

    # Position events
    log_event_listeners('Position')
    event.listen(Position, 'after_insert', lambda m, c, t: log_insert(m, c, t, PositionSnapshot  , RevisionControlSession()))
    logger.info("Event listener for 'after_insert' on Position has been set up.")
    event.listen(Position, 'after_update', lambda m, c, t: log_update(m, c, t, PositionSnapshot  , RevisionControlSession()))
    logger.info("Event listener for 'after_update' on Position has been set up.")
    event.listen(Position, 'after_delete', lambda m, c, t: log_delete(m, c, t, PositionSnapshot  , RevisionControlSession()))
    logger.info("Event listener for 'after_delete' on Position has been set up.")

    # Area events
    log_event_listeners('Area')
    event.listen(Area, 'after_insert', lambda m, c, t: log_insert(m, c, t, AreaSnapshot  , RevisionControlSession()))
    logger.info("Event listener for 'after_insert' on Area has been set up.")
    event.listen(Area, 'after_update', lambda m, c, t: log_update(m, c, t, AreaSnapshot  , RevisionControlSession()))
    logger.info("Event listener for 'after_update' on Area has been set up.")
    event.listen(Area, 'after_delete', lambda m, c, t: log_delete(m, c, t, AreaSnapshot  , RevisionControlSession()))
    logger.info("Event listener for 'after_delete' on Area has been set up.")

    # EquipmentGroup events
    log_event_listeners('EquipmentGroup')
    event.listen(EquipmentGroup, 'after_insert', lambda m, c, t: log_insert(m, c, t, EquipmentGroupSnapshot  , RevisionControlSession()))
    logger.info("Event listener for 'after_insert' on EquipmentGroup has been set up.")
    event.listen(EquipmentGroup, 'after_update', lambda m, c, t: log_update(m, c, t, EquipmentGroupSnapshot  , RevisionControlSession()))
    logger.info("Event listener for 'after_update' on EquipmentGroup has been set up.")
    event.listen(EquipmentGroup, 'after_delete', lambda m, c, t: log_delete(m, c, t, EquipmentGroupSnapshot  , RevisionControlSession()))
    logger.info("Event listener for 'after_delete' on EquipmentGroup has been set up.")

    # Model events
    log_event_listeners('Model')
    event.listen(Model, 'after_insert', lambda m, c, t: log_insert(m, c, t, ModelSnapshot  , RevisionControlSession()))
    logger.info("Event listener for 'after_insert' on Model has been set up.")
    event.listen(Model, 'after_update', lambda m, c, t: log_update(m, c, t, ModelSnapshot  , RevisionControlSession()))
    logger.info("Event listener for 'after_update' on Model has been set up.")
    event.listen(Model, 'after_delete', lambda m, c, t: log_delete(m, c, t, ModelSnapshot  , RevisionControlSession()))
    logger.info("Event listener for 'after_delete' on Model has been set up.")

    # AssetNumber events
    log_event_listeners('AssetNumber')
    event.listen(AssetNumber, 'after_insert', lambda m, c, t: log_insert(m, c, t, AssetNumberSnapshot  , RevisionControlSession()))
    logger.info("Event listener for 'after_insert' on AssetNumber has been set up.")
    event.listen(AssetNumber, 'after_update', lambda m, c, t: log_update(m, c, t, AssetNumberSnapshot  , RevisionControlSession()))
    logger.info("Event listener for 'after_update' on AssetNumber has been set up.")
    event.listen(AssetNumber, 'after_delete', lambda m, c, t: log_delete(m, c, t, AssetNumberSnapshot  , RevisionControlSession()))
    logger.info("Event listener for 'after_delete' on AssetNumber has been set up.")

    # Location events
    log_event_listeners('Location')
    event.listen(Location, 'after_insert', lambda m, c, t: log_insert(m, c, t, LocationSnapshot  , RevisionControlSession()))
    logger.info("Event listener for 'after_insert' on Location has been set up.")
    event.listen(Location, 'after_update', lambda m, c, t: log_update(m, c, t, LocationSnapshot  , RevisionControlSession()))
    logger.info("Event listener for 'after_update' on Location has been set up.")
    event.listen(Location, 'after_delete', lambda m, c, t: log_delete(m, c, t, LocationSnapshot  , RevisionControlSession()))
    logger.info("Event listener for 'after_delete' on Location has been set up.")

    # Part events
    log_event_listeners('Part')
    event.listen(Part, 'after_insert', lambda m, c, t: log_insert(m, c, t, PartSnapshot  , RevisionControlSession()))
    logger.info("Event listener for 'after_insert' on Part has been set up.")
    event.listen(Part, 'after_update', lambda m, c, t: log_update(m, c, t, PartSnapshot  , RevisionControlSession()))
    logger.info("Event listener for 'after_update' on Part has been set up.")
    event.listen(Part, 'after_delete', lambda m, c, t: log_delete(m, c, t, PartSnapshot  , RevisionControlSession()))
    logger.info("Event listener for 'after_delete' on Part has been set up.")

    # Image events
    log_event_listeners('Image')
    event.listen(Image, 'after_insert', lambda m, c, t: log_insert(m, c, t, ImageSnapshot  , RevisionControlSession()))
    logger.info("Event listener for 'after_insert' on Image has been set up.")
    event.listen(Image, 'after_update', lambda m, c, t: log_update(m, c, t, ImageSnapshot  , RevisionControlSession()))
    logger.info("Event listener for 'after_update' on Image has been set up.")
    event.listen(Image, 'after_delete', lambda m, c, t: log_delete(m, c, t, ImageSnapshot  , RevisionControlSession()))
    logger.info("Event listener for 'after_delete' on Image has been set up.")

    # ImageEmbedding events
    log_event_listeners('ImageEmbedding')
    event.listen(ImageEmbedding, 'after_insert', lambda m, c, t: log_insert(m, c, t, ImageEmbeddingSnapshot  , RevisionControlSession()))
    logger.info("Event listener for 'after_insert' on ImageEmbedding has been set up.")
    event.listen(ImageEmbedding, 'after_update', lambda m, c, t: log_update(m, c, t, ImageEmbeddingSnapshot  , RevisionControlSession()))
    logger.info("Event listener for 'after_update' on ImageEmbedding has been set up.")
    event.listen(ImageEmbedding, 'after_delete', lambda m, c, t: log_delete(m, c, t, ImageEmbeddingSnapshot  , RevisionControlSession()))
    logger.info("Event listener for 'after_delete' on ImageEmbedding has been set up.")

    # Drawing events
    log_event_listeners('Drawing')
    event.listen(Drawing, 'after_insert', lambda m, c, t: log_insert(m, c, t, DrawingSnapshot  , RevisionControlSession()))
    logger.info("Event listener for 'after_insert' on Drawing has been set up.")
    event.listen(Drawing, 'after_update', lambda m, c, t: log_update(m, c, t, DrawingSnapshot  , RevisionControlSession()))
    logger.info("Event listener for 'after_update' on Drawing has been set up.")
    event.listen(Drawing, 'after_delete', lambda m, c, t: log_delete(m, c, t, DrawingSnapshot  , RevisionControlSession()))
    logger.info("Event listener for 'after_delete' on Drawing has been set up.")
    
    # Document events
    log_event_listeners('Document')
    event.listen(Document, 'after_insert', lambda m, c, t: log_insert(m, c, t, DocumentSnapshot, RevisionControlSession()))
    logger.info("Event listener for 'after_insert' on Document has been set up.")
    event.listen(Document, 'after_update', lambda m, c, t: log_update(m, c, t, DocumentSnapshot, RevisionControlSession()))
    logger.info("Event listener for 'after_update' on Document has been set up.")
    event.listen(Document, 'after_delete', lambda m, c, t: log_delete(m, c, t, DocumentSnapshot, RevisionControlSession()))
    logger.info("Event listener for 'after_delete' on Document has been set up.")

    # CompleteDocument events
    log_event_listeners('CompleteDocument')
    event.listen(CompleteDocument, 'after_insert', lambda m, c, t: log_insert(m, c, t, CompleteDocumentSnapshot, RevisionControlSession()))
    logger.info("Event listener for 'after_insert' on CompleteDocument has been set up.")
    event.listen(CompleteDocument, 'after_update', lambda m, c, t: log_update(m, c, t, CompleteDocumentSnapshot, RevisionControlSession()))
    logger.info("Event listener for 'after_update' on CompleteDocument has been set up.")
    event.listen(CompleteDocument, 'after_delete', lambda m, c, t: log_delete(m, c, t, CompleteDocumentSnapshot, RevisionControlSession()))
    logger.info("Event listener for 'after_delete' on CompleteDocument has been set up.")
    

    # Problem events
    log_event_listeners('Problem')
    event.listen(Problem, 'after_insert', lambda m, c, t: log_insert(m, c, t, ProblemSnapshot  , RevisionControlSession()))
    logger.info("Event listener for 'after_insert' on Problem has been set up.")
    event.listen(Problem, 'after_update', lambda m, c, t: log_update(m, c, t, ProblemSnapshot  , RevisionControlSession()))
    logger.info("Event listener for 'after_update' on Problem has been set up.")
    event.listen(Problem, 'after_delete', lambda m, c, t: log_delete(m, c, t, ProblemSnapshot  , RevisionControlSession()))
    logger.info("Event listener for 'after_delete' on Problem has been set up.")

    # Solution events
    log_event_listeners('Solution')
    event.listen(Solution, 'after_insert', lambda m, c, t: log_insert(m, c, t, SolutionSnapshot  , RevisionControlSession()))
    logger.info("Event listener for 'after_insert' on Solution has been set up.")
    event.listen(Solution, 'after_update', lambda m, c, t: log_update(m, c, t, SolutionSnapshot  , RevisionControlSession()))
    logger.info("Event listener for 'after_update' on Solution has been set up.")
    event.listen(Solution, 'after_delete', lambda m, c, t: log_delete(m, c, t, SolutionSnapshot  , RevisionControlSession()))
    logger.info("Event listener for 'after_delete' on Solution has been set up.")

    # PowerPoint events
    log_event_listeners('PowerPoint')
    event.listen(PowerPoint, 'after_insert', lambda m, c, t: log_insert(m, c, t, PowerPointSnapshot  , RevisionControlSession()))
    logger.info("Event listener for 'after_insert' on PowerPoint has been set up.")
    event.listen(PowerPoint, 'after_update', lambda m, c, t: log_update(m, c, t, PowerPointSnapshot  , RevisionControlSession()))
    logger.info("Event listener for 'after_update' on PowerPoint has been set up.")
    event.listen(PowerPoint, 'after_delete', lambda m, c, t: log_delete(m, c, t, PowerPointSnapshot  , RevisionControlSession()))
    logger.info("Event listener for 'after_delete' on PowerPoint has been set up.")

    # Junction Table Snapshots
    # DrawingPartAssociation events
    log_event_listeners('DrawingPartAssociation')
    event.listen(DrawingPartAssociation, 'after_insert', lambda m, c, t: log_insert(m, c, t, DrawingPartAssociationSnapshot  , RevisionControlSession()))
    logger.info("Event listener for 'after_insert' on DrawingPartAssociation has been set up.")
    event.listen(DrawingPartAssociation, 'after_update', lambda m, c, t: log_update(m, c, t, DrawingPartAssociationSnapshot  , RevisionControlSession()))
    logger.info("Event listener for 'after_update' on DrawingPartAssociation has been set up.")
    event.listen(DrawingPartAssociation, 'after_delete', lambda m, c, t: log_delete(m, c, t, DrawingPartAssociationSnapshot  , RevisionControlSession()))
    logger.info("Event listener for 'after_delete' on DrawingPartAssociation has been set up.")

    # PartProblemAssociation events
    log_event_listeners('PartProblemAssociation')
    event.listen(PartProblemAssociation, 'after_insert', lambda m, c, t: log_insert(m, c, t, PartProblemAssociationSnapshot  , RevisionControlSession()))
    logger.info("Event listener for 'after_insert' on PartProblemAssociation has been set up.")
    event.listen(PartProblemAssociation, 'after_update', lambda m, c, t: log_update(m, c, t, PartProblemAssociationSnapshot  , RevisionControlSession()))
    logger.info("Event listener for 'after_update' on PartProblemAssociation has been set up.")
    event.listen(PartProblemAssociation, 'after_delete', lambda m, c, t: log_delete(m, c, t, PartProblemAssociationSnapshot  , RevisionControlSession()))
    logger.info("Event listener for 'after_delete' on PartProblemAssociation has been set up.")

    # PartSolutionAssociation events
    log_event_listeners('PartSolutionAssociation')
    event.listen(PartSolutionAssociation, 'after_insert', lambda m, c, t: log_insert(m, c, t, PartSolutionAssociationSnapshot  , RevisionControlSession()))
    logger.info("Event listener for 'after_insert' on PartSolutionAssociation has been set up.")
    event.listen(PartSolutionAssociation, 'after_update', lambda m, c, t: log_update(m, c, t, PartSolutionAssociationSnapshot  , RevisionControlSession()))
    logger.info("Event listener for 'after_update' on PartSolutionAssociation has been set up.")
    event.listen(PartSolutionAssociation, 'after_delete', lambda m, c, t: log_delete(m, c, t, PartSolutionAssociationSnapshot  , RevisionControlSession()))
    logger.info("Event listener for 'after_delete' on PartSolutionAssociation has been set up.")

    # DrawingProblemAssociation events
    log_event_listeners('DrawingProblemAssociation')
    event.listen(DrawingProblemAssociation, 'after_insert', lambda m, c, t: log_insert(m, c, t, DrawingProblemAssociationSnapshot  , RevisionControlSession()))
    logger.info("Event listener for 'after_insert' on DrawingProblemAssociation has been set up.")
    event.listen(DrawingProblemAssociation, 'after_update', lambda m, c, t: log_update(m, c, t, DrawingProblemAssociationSnapshot  , RevisionControlSession()))
    logger.info("Event listener for 'after_update' on DrawingProblemAssociation has been set up.")
    event.listen(DrawingProblemAssociation, 'after_delete', lambda m, c, t: log_delete(m, c, t, DrawingProblemAssociationSnapshot  , RevisionControlSession()))
    logger.info("Event listener for 'after_delete' on DrawingProblemAssociation has been set up.")

    # DrawingSolutionAssociation events
    log_event_listeners('DrawingSolutionAssociation')
    event.listen(DrawingSolutionAssociation, 'after_insert', lambda m, c, t: log_insert(m, c, t, DrawingSolutionAssociationSnapshot  , RevisionControlSession()))
    logger.info("Event listener for 'after_insert' on DrawingSolutionAssociation has been set up.")
    event.listen(DrawingSolutionAssociation, 'after_update', lambda m, c, t: log_update(m, c, t, DrawingSolutionAssociationSnapshot  , RevisionControlSession()))
    logger.info("Event listener for 'after_update' on DrawingSolutionAssociation has been set up.")
    event.listen(DrawingSolutionAssociation, 'after_delete', lambda m, c, t: log_delete(m, c, t, DrawingSolutionAssociationSnapshot  , RevisionControlSession()))
    logger.info("Event listener for 'after_delete' on DrawingSolutionAssociation has been set up.")

    # BillOfMaterial events
    log_event_listeners('BillOfMaterial')
    event.listen(BillOfMaterial, 'after_insert', lambda m, c, t: log_insert(m, c, t, BillOfMaterialSnapshot  , RevisionControlSession()))
    logger.info("Event listener for 'after_insert' on BillOfMaterial has been set up.")
    event.listen(BillOfMaterial, 'after_update', lambda m, c, t: log_update(m, c, t, BillOfMaterialSnapshot  , RevisionControlSession()))
    logger.info("Event listener for 'after_update' on BillOfMaterial has been set up.")
    event.listen(BillOfMaterial, 'after_delete', lambda m, c, t: log_delete(m, c, t, BillOfMaterialSnapshot  , RevisionControlSession()))
    logger.info("Event listener for 'after_delete' on BillOfMaterial has been set up.")

    # ProblemPositionAssociation events
    log_event_listeners('ProblemPositionAssociation')
    event.listen(ProblemPositionAssociation, 'after_insert', lambda m, c, t: log_insert(m, c, t, ProblemPositionAssociationSnapshot  , RevisionControlSession()))
    logger.info("Event listener for 'after_insert' on ProblemPositionAssociation has been set up.")
    event.listen(ProblemPositionAssociation, 'after_update', lambda m, c, t: log_update(m, c, t, ProblemPositionAssociationSnapshot  , RevisionControlSession()))
    logger.info("Event listener for 'after_update' on ProblemPositionAssociation has been set up.")
    event.listen(ProblemPositionAssociation, 'after_delete', lambda m, c, t: log_delete(m, c, t, ProblemPositionAssociationSnapshot  , RevisionControlSession()))
    logger.info("Event listener for 'after_delete' on ProblemPositionAssociation has been set up.")

    # CompleteDocumentProblemAssociation events
    log_event_listeners('CompleteDocumentProblemAssociation')
    event.listen(CompleteDocumentProblemAssociation, 'after_insert', lambda m, c, t: log_insert(m, c, t, CompleteDocumentProblemAssociationSnapshot  , RevisionControlSession()))
    logger.info("Event listener for 'after_insert' on CompleteDocumentProblemAssociation has been set up.")
    event.listen(CompleteDocumentProblemAssociation, 'after_update', lambda m, c, t: log_update(m, c, t, CompleteDocumentProblemAssociationSnapshot  , RevisionControlSession()))
    logger.info("Event listener for 'after_update' on CompleteDocumentProblemAssociation has been set up.")
    event.listen(CompleteDocumentProblemAssociation, 'after_delete', lambda m, c, t: log_delete(m, c, t, CompleteDocumentProblemAssociationSnapshot  , RevisionControlSession()))
    logger.info("Event listener for 'after_delete' on CompleteDocumentProblemAssociation has been set up.")

    # CompleteDocumentSolutionAssociation events
    log_event_listeners('CompleteDocumentSolutionAssociation')
    event.listen(CompleteDocumentSolutionAssociation, 'after_insert', lambda m, c, t: log_insert(m, c, t, CompleteDocumentSolutionAssociationSnapshot  , RevisionControlSession()))
    logger.info("Event listener for 'after_insert' on CompleteDocumentSolutionAssociation has been set up.")
    event.listen(CompleteDocumentSolutionAssociation, 'after_update', lambda m, c, t: log_update(m, c, t, CompleteDocumentSolutionAssociationSnapshot  , RevisionControlSession()))
    logger.info("Event listener for 'after_update' on CompleteDocumentSolutionAssociation has been set up.")
    event.listen(CompleteDocumentSolutionAssociation, 'after_delete', lambda m, c, t: log_delete(m, c, t, CompleteDocumentSolutionAssociationSnapshot  , RevisionControlSession()))
    logger.info("Event listener for 'after_delete' on CompleteDocumentSolutionAssociation has been set up.")

    # ImageProblemAssociation events
    log_event_listeners('ImageProblemAssociation')
    event.listen(ImageProblemAssociation, 'after_insert', lambda m, c, t: log_insert(m, c, t, ImageProblemAssociationSnapshot  , RevisionControlSession()))
    logger.info("Event listener for 'after_insert' on ImageProblemAssociation has been set up.")
    event.listen(ImageProblemAssociation, 'after_update', lambda m, c, t: log_update(m, c, t, ImageProblemAssociationSnapshot  , RevisionControlSession()))
    logger.info("Event listener for 'after_update' on ImageProblemAssociation has been set up.")
    event.listen(ImageProblemAssociation, 'after_delete', lambda m, c, t: log_delete(m, c, t, ImageProblemAssociationSnapshot  , RevisionControlSession()))
    logger.info("Event listener for 'after_delete' on ImageProblemAssociation has been set up.")

    # ImageSolutionAssociation events
    log_event_listeners('ImageSolutionAssociation')
    event.listen(ImageSolutionAssociation, 'after_insert', lambda m, c, t: log_insert(m, c, t, ImageSolutionAssociationSnapshot  , RevisionControlSession()))
    logger.info("Event listener for 'after_insert' on ImageSolutionAssociation has been set up.")
    event.listen(ImageSolutionAssociation, 'after_update', lambda m, c, t: log_update(m, c, t, ImageSolutionAssociationSnapshot  , RevisionControlSession()))
    logger.info("Event listener for 'after_update' on ImageSolutionAssociation has been set up.")
    event.listen(ImageSolutionAssociation, 'after_delete', lambda m, c, t: log_delete(m, c, t, ImageSolutionAssociationSnapshot  , RevisionControlSession()))
    logger.info("Event listener for 'after_delete' on ImageSolutionAssociation has been set up.")

    # PartsPositionImageAssociation events
    log_event_listeners('PartsPositionImageAssociation')
    event.listen(PartsPositionImageAssociation, 'after_insert', lambda m, c, t: log_insert(m, c, t, PartsPositionImageAssociationSnapshot  , RevisionControlSession()))
    logger.info("Event listener for 'after_insert' on PartsPositionImageAssociation has been set up.")
    event.listen(PartsPositionImageAssociation, 'after_update', lambda m, c, t: log_update(m, c, t, PartsPositionImageAssociationSnapshot  , RevisionControlSession()))
    logger.info("Event listener for 'after_update' on PartsPositionImageAssociation has been set up.")
    event.listen(PartsPositionImageAssociation, 'after_delete', lambda m, c, t: log_delete(m, c, t, PartsPositionImageAssociationSnapshot  , RevisionControlSession()))
    logger.info("Event listener for 'after_delete' on PartsPositionImageAssociation has been set up.")

    # ImagePositionAssociation events
    log_event_listeners('ImagePositionAssociation')
    event.listen(ImagePositionAssociation, 'after_insert', lambda m, c, t: log_insert(m, c, t, ImagePositionAssociationSnapshot  , RevisionControlSession()))
    logger.info("Event listener for 'after_insert' on ImagePositionAssociation has been set up.")
    event.listen(ImagePositionAssociation, 'after_update', lambda m, c, t: log_update(m, c, t, ImagePositionAssociationSnapshot  , RevisionControlSession()))
    logger.info("Event listener for 'after_update' on ImagePositionAssociation has been set up.")
    event.listen(ImagePositionAssociation, 'after_delete', lambda m, c, t: log_delete(m, c, t, ImagePositionAssociationSnapshot  , RevisionControlSession()))
    logger.info("Event listener for 'after_delete' on ImagePositionAssociation has been set up.")

    # DrawingPositionAssociation events
    log_event_listeners('DrawingPositionAssociation')
    event.listen(DrawingPositionAssociation, 'after_insert', lambda m, c, t: log_insert(m, c, t, DrawingPositionAssociationSnapshot  , RevisionControlSession()))
    logger.info("Event listener for 'after_insert' on DrawingPositionAssociation has been set up.")
    event.listen(DrawingPositionAssociation, 'after_update', lambda m, c, t: log_update(m, c, t, DrawingPositionAssociationSnapshot  , RevisionControlSession()))
    logger.info("Event listener for 'after_update' on DrawingPositionAssociation has been set up.")
    event.listen(DrawingPositionAssociation, 'after_delete', lambda m, c, t: log_delete(m, c, t, DrawingPositionAssociationSnapshot  , RevisionControlSession()))
    logger.info("Event listener for 'after_delete' on DrawingPositionAssociation has been set up.")

    # CompletedDocumentPositionAssociation events
    log_event_listeners('CompletedDocumentPositionAssociation')
    event.listen(CompletedDocumentPositionAssociation, 'after_insert', lambda m, c, t: log_insert(m, c, t, CompletedDocumentPositionAssociationSnapshot  , RevisionControlSession()))
    logger.info("Event listener for 'after_insert' on CompletedDocumentPositionAssociation has been set up.")
    event.listen(CompletedDocumentPositionAssociation, 'after_update', lambda m, c, t: log_update(m, c, t, CompletedDocumentPositionAssociationSnapshot  , RevisionControlSession()))
    logger.info("Event listener for 'after_update' on CompletedDocumentPositionAssociation has been set up.")
    event.listen(CompletedDocumentPositionAssociation, 'after_delete', lambda m, c, t: log_delete(m, c, t, CompletedDocumentPositionAssociationSnapshot  , RevisionControlSession()))
    logger.info("Event listener for 'after_delete' on CompletedDocumentPositionAssociation has been set up.")

    # ImageCompletedDocumentAssociation events
    log_event_listeners('ImageCompletedDocumentAssociation')
    event.listen(ImageCompletedDocumentAssociation, 'after_insert', lambda m, c, t: log_insert(m, c, t, ImageCompletedDocumentAssociationSnapshot  , RevisionControlSession()))
    logger.info("Event listener for 'after_insert' on ImageCompletedDocumentAssociation has been set up.")
    event.listen(ImageCompletedDocumentAssociation, 'after_update', lambda m, c, t: log_update(m, c, t, ImageCompletedDocumentAssociationSnapshot  , RevisionControlSession()))
    logger.info("Event listener for 'after_update' on ImageCompletedDocumentAssociation has been set up.")
    event.listen(ImageCompletedDocumentAssociation, 'after_delete', lambda m, c, t: log_delete(m, c, t, ImageCompletedDocumentAssociationSnapshot  , RevisionControlSession()))
    logger.info("Event listener for 'after_delete' on ImageCompletedDocumentAssociation has been set up.")"""