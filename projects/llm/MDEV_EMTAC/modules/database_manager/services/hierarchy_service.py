from __future__ import annotations
from typing import Optional
from sqlalchemy.orm import Session
from modules.emtacdb.emtacdb_fts import (
    Position, Campus, Building, SiteLocation, Area, EquipmentGroup, Model,
    AssetNumber, Location, Subassembly, ComponentAssembly, AssemblyView
)
from .campus_service import CampusService
from .building_service import BuildingService
from .site_location_service import SiteLocationService
from .area_service import AreaService
from .equipment_group_service import EquipmentGroupService
from .model_service import ModelService
from .asset_number_service import AssetNumberService
from .location_service import LocationService
from .subassembly_service import SubassemblyService
from .component_assembly_service import ComponentAssemblyService
from .assembly_view_service import AssemblyViewService
from .position_service import PositionService

class HierarchyService:
    MODELS_MAP = {
        "Campus": Campus, "Building": Building, "SiteLocation": SiteLocation, "Area": Area,
        "EquipmentGroup": EquipmentGroup, "Model": Model, "AssetNumber": AssetNumber,
        "Location": Location, "Subassembly": Subassembly, "ComponentAssembly": ComponentAssembly,
        "AssemblyView": AssemblyView,
    }
    ENTITY_SERVICE_MAP = {
        "campus": CampusService, "building": BuildingService, "site_location": SiteLocationService,
        "area": AreaService, "equipment_group": EquipmentGroupService, "model": ModelService,
        "asset_number": AssetNumberService, "location": LocationService, "subassembly": SubassemblyService,
        "component_assembly": ComponentAssemblyService, "assembly_view": AssemblyViewService,
    }

    @staticmethod
    def get_dependent_items(session: Session, parent_type: str, parent_id: int, child_type: str | None = None) -> list:
        if not parent_id:
            return []
        parent_config = Position.HIERARCHY.get(parent_type)
        if not parent_config:
            return []
        if "child_types" in parent_config:
            if child_type:
                for child_config in parent_config["child_types"]:
                    if child_config.get("next_level") == child_type:
                        return HierarchyService._fetch(session, child_config, parent_id)
                return []
            return HierarchyService._fetch(session, parent_config["child_types"][0], parent_id)
        return HierarchyService._fetch(session, parent_config, parent_id)

    @staticmethod
    def _fetch(session: Session, config: dict, parent_id: int) -> list:
        model_name = config.get("model")
        filter_field = config.get("filter_field")
        order_field = config.get("order_field")
        if not all([model_name, filter_field, order_field]):
            return []
        model_cls = HierarchyService.MODELS_MAP.get(model_name) if isinstance(model_name, str) else model_name
        if model_cls is None:
            return []
        return session.query(model_cls).filter_by(**{filter_field: parent_id}).order_by(getattr(model_cls, order_field)).all()

    @staticmethod
    def get_next_level_type(current_level: str):
        config = Position.HIERARCHY.get(current_level)
        if not config:
            return None
        if "child_types" in config:
            return config["child_types"][0].get("next_level")
        return config.get("next_level")

    @staticmethod
    def find_related_entities(session: Session, entity_type: str, identifier, is_id: bool = True):
        service_cls = HierarchyService.ENTITY_SERVICE_MAP.get(entity_type)
        if not service_cls or not hasattr(service_cls, "get_by_id"):
            return None
        if is_id:
            entity = service_cls.get_by_id(session, int(identifier))
        else:
            return service_cls.search(session, name=str(identifier))[0] if service_cls.search(session, name=str(identifier)) else None
        if entity is None:
            return None
        if hasattr(service_cls, "find_related_entities"):
            return service_cls.find_related_entities(session, identifier, is_id=is_id)
        return {"entity": entity, "upward": {}, "downward": {}}

    @staticmethod
    def get_positions_by_hierarchy(session: Session, **filters):
        return PositionService.search(session, **filters)
