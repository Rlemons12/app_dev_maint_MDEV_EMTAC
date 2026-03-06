# modules/orchestrators/position_orchestrator.py

from typing import Optional, Dict, Any

from modules.orchestrators.base_orchestrator import BaseOrchestrator
from modules.configuration.log_config import with_request_id

from modules.services.position_service import PositionService
from modules.services.area_service import AreaService
from modules.services.campus_service import CampusService
from modules.services.building_service import BuildingService
from modules.services.equipment_group_service import EquipmentGroupService
from modules.services.model_service import ModelService
from modules.services.location_service import LocationService
from modules.services.subassembly_service import SubassemblyService
from modules.services.component_assembly_service import ComponentAssemblyService
from modules.services.assembly_view_service import AssemblyViewService
from modules.services.asset_number_service import AssetNumberService


class PositionOrchestrator(BaseOrchestrator):

    def __init__(self):
        super().__init__()

        self.position_service = PositionService()
        self.area_service = AreaService()
        self.campus_service = CampusService()
        self.building_service = BuildingService()
        self.equipment_group_service = EquipmentGroupService()
        self.model_service = ModelService()
        self.location_service = LocationService()
        self.subassembly_service = SubassemblyService()
        self.component_assembly_service = ComponentAssemblyService()
        self.assembly_view_service = AssemblyViewService()
        self.asset_number_service = AssetNumberService()

    # ---------------------------------------------------------
    # FULL HIERARCHY RESOLUTION
    # ---------------------------------------------------------

    @with_request_id
    def resolve_from_metadata(self, metadata: Dict[str, Any]) -> Optional[int]:

        if not metadata:
            return None

        with self.transaction() as session:

            # ---------------- AREA ----------------

            area = None
            if metadata.get("area_name"):
                area = self.area_service.find_or_create(
                    session=session,
                    name=metadata["area_name"],
                )
            elif metadata.get("area_id"):
                area = self.area_service.get(
                    session=session,
                    area_id=metadata["area_id"],
                )

            # ---------------- EQUIPMENT GROUP ----------------

            equipment_group = None
            if metadata.get("equipment_group_name") and area:
                equipment_group = self.equipment_group_service.find_or_create(
                    session=session,
                    name=metadata["equipment_group_name"],
                    area_id=area.id,
                )
            elif metadata.get("equipment_group_id"):
                equipment_group = self.equipment_group_service.get(
                    session=session,
                    equipment_group_id=metadata["equipment_group_id"],
                )

            # ---------------- MODEL ----------------

            model = None
            if metadata.get("model_name") and equipment_group:
                model = self.model_service.find_or_create(
                    session=session,
                    name=metadata["model_name"],
                    equipment_group_id=equipment_group.id,
                )
            elif metadata.get("model_id"):
                model = self.model_service.get(
                    session=session,
                    model_id=metadata["model_id"],
                )

            # ---------------- ASSET NUMBER ----------------

            asset_number = None

            if metadata.get("asset_number_number") and model:
                asset_number = self.asset_number_service.save(
                    session=session,
                    number=metadata["asset_number_number"],
                    model_id=model.id,
                    description=metadata.get("asset_number_description"),
                )

            elif metadata.get("asset_number_id"):
                asset_number = self.asset_number_service.get(
                    session=session,
                    asset_number_id=metadata["asset_number_id"],
                )

            # ---------------- LOCATION ----------------

            location = None
            if metadata.get("location_name") and model:
                location = self.location_service.find_or_create(
                    session=session,
                    name=metadata["location_name"],
                    model_id=model.id,
                )
            elif metadata.get("location_id"):
                location = self.location_service.get(
                    session=session,
                    location_id=metadata["location_id"],
                )

            # ---------------- SUBASSEMBLY ----------------

            subassembly = None
            if metadata.get("subassembly_name") and location:
                subassembly = self.subassembly_service.find_or_create(
                    session=session,
                    name=metadata["subassembly_name"],
                    location_id=location.id,
                )
            elif metadata.get("subassembly_id"):
                subassembly = self.subassembly_service.get(
                    session=session,
                    subassembly_id=metadata["subassembly_id"],
                )

            # ---------------- COMPONENT ASSEMBLY ----------------

            component_assembly = None
            if metadata.get("component_assembly_name") and subassembly:
                component_assembly = self.component_assembly_service.find_or_create(
                    session=session,
                    name=metadata["component_assembly_name"],
                    subassembly_id=subassembly.id,
                )
            elif metadata.get("component_assembly_id"):
                component_assembly = self.component_assembly_service.get(
                    session=session,
                    component_assembly_id=metadata["component_assembly_id"],
                )

            # ---------------- ASSEMBLY VIEW ----------------

            assembly_view = None
            if metadata.get("assembly_view_name") and component_assembly:
                assembly_view = self.assembly_view_service.find_or_create(
                    session=session,
                    name=metadata["assembly_view_name"],
                    component_assembly_id=component_assembly.id,
                    asset_number_id=asset_number.id if asset_number else None,
                )
            elif metadata.get("assembly_view_id"):
                assembly_view = self.assembly_view_service.get(
                    session=session,
                    assembly_view_id=metadata["assembly_view_id"],
                )

            # ---------------- CAMPUS ----------------

            campus = None
            if metadata.get("campus_name"):
                campus = self.campus_service.find_or_create(
                    session=session,
                    name=metadata["campus_name"],
                )
            elif metadata.get("campus_id"):
                campus = self.campus_service.get(
                    session=session,
                    campus_id=metadata["campus_id"],
                )

            # ---------------- BUILDING ----------------

            building = None
            if metadata.get("building_name") and campus:
                building = self.building_service.find_or_create(
                    session=session,
                    name=metadata["building_name"],
                    campus_id=campus.id,
                )
            elif metadata.get("building_id"):
                building = self.building_service.get(
                    session=session,
                    building_id=metadata["building_id"],
                )

            # ---------------- POSITION ----------------

            position_id = self.position_service.add_to_db(
                session=session,
                area_id=area.id if area else None,
                equipment_group_id=equipment_group.id if equipment_group else None,
                model_id=model.id if model else None,
                location_id=location.id if location else None,
                subassembly_id=subassembly.id if subassembly else None,
                component_assembly_id=component_assembly.id if component_assembly else None,
                assembly_view_id=assembly_view.id if assembly_view else None,
                campus_id=campus.id if campus else None,
                building_id=building.id if building else None,
                asset_number_id=asset_number.id if asset_number else None,
                site_location_id=metadata.get("site_location_id"),
            )

            return position_id