# services/__init__.py
from modules.services.part_service import PartService

from .area_service import AreaService
from .equipmentgroup_service import EquipmentGroupService
from .model_service import ModelService
from .assetnumber_service import AssetNumberService
from .location_service import LocationService
from .site_location_service import SiteLocationService
from .subassembly_service import SubassemblyService
from .component_assembly_service import ComponentAssemblyService
from .assembly_view_service import AssemblyViewService
from .position_service import PositionService
from .part_service import PartService
from .image_service import ImageService
from .image_embedding_service import ImageEmbeddingService
from .drawing_service import DrawingService
from .document_service import DocumentService
from .drawing_part_association_service import DrawingPartAssociationService  # NEW


class DBServices:
    """Purpose: Encapsulates business logic and operations that act on data.Service = Act / change / orchestrate.
        Façade providing a single entry point for all services."""

    def __init__(self, db_config=None):
        self.areas = AreaService(db_config)
        self.equipment_groups = EquipmentGroupService(db_config)
        self.models = ModelService(db_config)
        self.asset_numbers = AssetNumberService(db_config)
        self.locations = LocationService(db_config)
        self.site_locations = SiteLocationService(db_config)
        self.subassemblies = SubassemblyService(db_config)
        self.component_assemblies = ComponentAssemblyService(db_config)
        self.assembly_views = AssemblyViewService(db_config)
        self.positions = PositionService(db_config)
        self.parts = PartService(db_config)
        self.images = ImageService(db_config)
        self.image_embeddings = ImageEmbeddingService(db_config)
        self.drawings = DrawingService(db_config)
        self.documents = DocumentService(db_config)
        self.drawing_part_associations = DrawingPartAssociationService(db_config)  # NEW


__all__ = [
    "AreaService",
    "EquipmentGroupService",
    "ModelService",
    "AssetNumberService",
    "LocationService",
    "SiteLocationService",
    "SubassemblyService",
    "ComponentAssemblyService",
    "AssemblyViewService",
    "PositionService",
    "PartService",
    "ImageService",
    "ImageEmbeddingService",
    "DrawingService",
    "DocumentService",
    "DrawingPartAssociationService",  # NEW
]
