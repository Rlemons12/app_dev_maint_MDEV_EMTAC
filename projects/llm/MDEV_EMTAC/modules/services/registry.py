"""
AUTO-GENERATED SERVICE REGISTRY

DO NOT EDIT MANUALLY.
Run build_service_registry.py to regenerate.
"""

from typing import Optional

from modules.services.ai_model_image_service import AIModelImageService
from modules.services.ai_models_embedding_service import AIModelsEmbeddingService
from modules.services.ai_models_service import AIModelsService
from modules.services.ai_models_vlm_service import AIModelsVLMService
from modules.services.area_service import AreaService
from modules.services.assembly_view_service import AssemblyViewService
from modules.services.assetnumber_service import AssetNumberService
from modules.services.building_service import BuildingService
#from modules.services.campus_service import CampusService
#from modules.services.chunk_search_service import ChunkAssociationSearchExtendedService
from modules.services.chunk_search_service import ChunkAssociationSearchService
from modules.services.complete_document_service import CompleteDocumentService
from modules.services.completed_document_position_service import CompletedDocumentPositionService
from modules.services.component_assembly_service import ComponentAssemblyService
from modules.services.document_embedding_service import DocumentEmbeddingService
from modules.services.document_processing_service import DocumentProcessingService
from modules.services.document_service import DocumentService
from modules.services.document_ui_projection_service import DocumentUIProjectionService
from modules.services.drawing_part_association_service import DrawingPartAssociationService
from modules.services.drawing_position_association_service import DrawingPositionAssociationService
from modules.services.drawing_service import DrawingService
from modules.services.equipmentgroup_service import EquipmentGroupService
from modules.services.image_completed_document_association_service import ImageCompletedDocumentAssociationService
from modules.services.image_embedding_service import ImageEmbeddingService
from modules.services.image_position_association_service import ImagePositionAssociationService
from modules.services.image_position_service import ImagePositionService
from modules.services.image_service import ImageService
from modules.services.intent_classifier_service import IntentClassifierService
from modules.services.location_service import LocationService
from modules.services.model_service import ModelService
from modules.services.part_service import PartService
from modules.services.parts_position_image_service import PartsPositionImageService
from modules.services.position_service import PositionService
from modules.services.problem_position_association_service import ProblemPositionAssociationService
from modules.services.problem_service import ProblemService
from modules.services.site_location_service import SiteLocationService
from modules.services.subassembly_service import SubassemblyService
from modules.services.task_position_association_service import TaskPositionAssociationService
from modules.services.task_tool_association_service import TaskToolAssociationService
from modules.services.tool_image_association_service import ToolImageAssociationService
from modules.services.tool_position_association_service import ToolPositionAssociationService
from modules.services.tool_service import ToolService
from modules.services.troubleshooting_service import TroubleshootingService


class ServiceRegistry:

    def __init__(self):
        self._ai_model_image = None
        self._ai_models_embedding = None
        self._ai_models = None
        self._ai_models_vlm = None
        self._area = None
        self._assembly_view = None
        self._asset_number = None
        self._building = None
        self._campus = None
        self._chunk_association_search_extended = None
        self._chunk_association_search = None
        self._complete_document = None
        self._completed_document_position = None
        self._component_assembly = None
        self._document_embedding = None
        self._document_processing = None
        self._document = None
        self._document_ui_projection = None
        self._drawing_part_association = None
        self._drawing_position_association = None
        self._drawing = None
        self._equipment_group = None
        self._image_completed_document_association = None
        self._image_embedding = None
        self._image_position_association = None
        self._image_position = None
        self._image = None
        self._intent_classifier = None
        self._location = None
        self._model = None
        self._part = None
        self._parts_position_image = None
        self._position = None
        self._problem_position_association = None
        self._problem = None
        self._site_location = None
        self._subassembly = None
        self._task_position_association = None
        self._task_tool_association = None
        self._tool_image_association = None
        self._tool_position_association = None
        self._tool = None
        self._troubleshooting = None


    @property
    def ai_model_image(self) -> AIModelImageService:
        if self._ai_model_image is None:
            self._ai_model_image = AIModelImageService()
        return self._ai_model_image

    @property
    def ai_models_embedding(self) -> AIModelsEmbeddingService:
        if self._ai_models_embedding is None:
            self._ai_models_embedding = AIModelsEmbeddingService()
        return self._ai_models_embedding

    @property
    def ai_models(self) -> AIModelsService:
        if self._ai_models is None:
            self._ai_models = AIModelsService()
        return self._ai_models

    @property
    def ai_models_vlm(self) -> AIModelsVLMService:
        if self._ai_models_vlm is None:
            self._ai_models_vlm = AIModelsVLMService()
        return self._ai_models_vlm

    @property
    def area(self) -> AreaService:
        if self._area is None:
            self._area = AreaService()
        return self._area

    @property
    def assembly_view(self) -> AssemblyViewService:
        if self._assembly_view is None:
            self._assembly_view = AssemblyViewService()
        return self._assembly_view

    @property
    def asset_number(self) -> AssetNumberService:
        if self._asset_number is None:
            self._asset_number = AssetNumberService()
        return self._asset_number

    @property
    def building(self) -> BuildingService:
        if self._building is None:
            self._building = BuildingService()
        return self._building

    @property
    def campus(self) -> CampusService:
        if self._campus is None:
            self._campus = CampusService()
        return self._campus

    @property
    def chunk_association_search_extended(self) -> ChunkAssociationSearchExtendedService:
        if self._chunk_association_search_extended is None:
            self._chunk_association_search_extended = ChunkAssociationSearchExtendedService()
        return self._chunk_association_search_extended

    @property
    def chunk_association_search(self) -> ChunkAssociationSearchService:
        if self._chunk_association_search is None:
            self._chunk_association_search = ChunkAssociationSearchService()
        return self._chunk_association_search

    @property
    def complete_document(self) -> CompleteDocumentService:
        if self._complete_document is None:
            self._complete_document = CompleteDocumentService()
        return self._complete_document

    @property
    def completed_document_position(self) -> CompletedDocumentPositionService:
        if self._completed_document_position is None:
            self._completed_document_position = CompletedDocumentPositionService()
        return self._completed_document_position

    @property
    def component_assembly(self) -> ComponentAssemblyService:
        if self._component_assembly is None:
            self._component_assembly = ComponentAssemblyService()
        return self._component_assembly

    @property
    def document_embedding(self) -> DocumentEmbeddingService:
        if self._document_embedding is None:
            self._document_embedding = DocumentEmbeddingService()
        return self._document_embedding

    @property
    def document_processing(self) -> DocumentProcessingService:
        if self._document_processing is None:
            self._document_processing = DocumentProcessingService()
        return self._document_processing

    @property
    def document(self) -> DocumentService:
        if self._document is None:
            self._document = DocumentService()
        return self._document

    @property
    def document_ui_projection(self) -> DocumentUIProjectionService:
        if self._document_ui_projection is None:
            self._document_ui_projection = DocumentUIProjectionService()
        return self._document_ui_projection

    @property
    def drawing_part_association(self) -> DrawingPartAssociationService:
        if self._drawing_part_association is None:
            self._drawing_part_association = DrawingPartAssociationService()
        return self._drawing_part_association

    @property
    def drawing_position_association(self) -> DrawingPositionAssociationService:
        if self._drawing_position_association is None:
            self._drawing_position_association = DrawingPositionAssociationService()
        return self._drawing_position_association

    @property
    def drawing(self) -> DrawingService:
        if self._drawing is None:
            self._drawing = DrawingService()
        return self._drawing

    @property
    def equipment_group(self) -> EquipmentGroupService:
        if self._equipment_group is None:
            self._equipment_group = EquipmentGroupService()
        return self._equipment_group

    @property
    def image_completed_document_association(self) -> ImageCompletedDocumentAssociationService:
        if self._image_completed_document_association is None:
            self._image_completed_document_association = ImageCompletedDocumentAssociationService()
        return self._image_completed_document_association

    @property
    def image_embedding(self) -> ImageEmbeddingService:
        if self._image_embedding is None:
            self._image_embedding = ImageEmbeddingService()
        return self._image_embedding

    @property
    def image_position_association(self) -> ImagePositionAssociationService:
        if self._image_position_association is None:
            self._image_position_association = ImagePositionAssociationService()
        return self._image_position_association

    @property
    def image_position(self) -> ImagePositionService:
        if self._image_position is None:
            self._image_position = ImagePositionService()
        return self._image_position

    @property
    def image(self) -> ImageService:
        if self._image is None:
            self._image = ImageService()
        return self._image

    @property
    def intent_classifier(self) -> IntentClassifierService:
        if self._intent_classifier is None:
            self._intent_classifier = IntentClassifierService()
        return self._intent_classifier

    @property
    def location(self) -> LocationService:
        if self._location is None:
            self._location = LocationService()
        return self._location

    @property
    def model(self) -> ModelService:
        if self._model is None:
            self._model = ModelService()
        return self._model

    @property
    def part(self) -> PartService:
        if self._part is None:
            self._part = PartService()
        return self._part

    @property
    def parts_position_image(self) -> PartsPositionImageService:
        if self._parts_position_image is None:
            self._parts_position_image = PartsPositionImageService()
        return self._parts_position_image

    @property
    def position(self) -> PositionService:
        if self._position is None:
            self._position = PositionService()
        return self._position

    @property
    def problem_position_association(self) -> ProblemPositionAssociationService:
        if self._problem_position_association is None:
            self._problem_position_association = ProblemPositionAssociationService()
        return self._problem_position_association

    @property
    def problem(self) -> ProblemService:
        if self._problem is None:
            self._problem = ProblemService()
        return self._problem

    @property
    def site_location(self) -> SiteLocationService:
        if self._site_location is None:
            self._site_location = SiteLocationService()
        return self._site_location

    @property
    def subassembly(self) -> SubassemblyService:
        if self._subassembly is None:
            self._subassembly = SubassemblyService()
        return self._subassembly

    @property
    def task_position_association(self) -> TaskPositionAssociationService:
        if self._task_position_association is None:
            self._task_position_association = TaskPositionAssociationService()
        return self._task_position_association

    @property
    def task_tool_association(self) -> TaskToolAssociationService:
        if self._task_tool_association is None:
            self._task_tool_association = TaskToolAssociationService()
        return self._task_tool_association

    @property
    def tool_image_association(self) -> ToolImageAssociationService:
        if self._tool_image_association is None:
            self._tool_image_association = ToolImageAssociationService()
        return self._tool_image_association

    @property
    def tool_position_association(self) -> ToolPositionAssociationService:
        if self._tool_position_association is None:
            self._tool_position_association = ToolPositionAssociationService()
        return self._tool_position_association

    @property
    def tool(self) -> ToolService:
        if self._tool is None:
            self._tool = ToolService()
        return self._tool

    @property
    def troubleshooting(self) -> TroubleshootingService:
        if self._troubleshooting is None:
            self._troubleshooting = TroubleshootingService()
        return self._troubleshooting




