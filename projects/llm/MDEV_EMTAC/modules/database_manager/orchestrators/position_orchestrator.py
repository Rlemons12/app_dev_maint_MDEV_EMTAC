from __future__ import annotations

from typing import Any, Optional

from sqlalchemy.exc import SQLAlchemyError

from modules.configuration.config_env import get_db_config
from modules.configuration.log_config import DatabaseMaintLogManager
from modules.database_manager.services import (
    CampusService,
    BuildingService,
    SiteLocationService,
    AreaService,
    EquipmentGroupService,
    ModelService,
    AssetNumberService,
    LocationService,
    SubassemblyService,
    ComponentAssemblyService,
    AssemblyViewService,
    PositionService,
    HierarchyService,
)


class PositionOrchestrator:
    """
    Owns session lifecycle and transaction boundaries for position-related workflows.

    Rules:
    - Services do NOT commit/rollback
    - This orchestrator DOES commit/rollback
    - This orchestrator can create or resolve the full hierarchy ending in Position
    - This orchestrator can bulk-build Position rows from existing hierarchy data
    """

    def __init__(
        self,
        db_config=None,
        db_log_manager: DatabaseMaintLogManager | None = None,
        log_run_dir=None,
        log_to_console: bool = False,
    ):
        self.db_config = db_config or get_db_config()
        self.db_log_manager = db_log_manager or DatabaseMaintLogManager(
            run_dir=log_run_dir,
            run_name="position_orchestrator",
            to_console=log_to_console,
        )
        self.logger = self.db_log_manager.logger

    # -------------------------------------------------------------------------
    # Internal helpers
    # -------------------------------------------------------------------------
    @staticmethod
    def _clean_str(value: Any) -> Optional[str]:
        if value is None:
            return None
        value = str(value).strip()
        return value if value else None

    @staticmethod
    def _build_result(
        success: bool,
        message: str,
        data: Optional[dict] = None,
        errors: Optional[list[str]] = None,
    ) -> dict:
        return {
            "success": success,
            "message": message,
            "data": data or {},
            "errors": errors or [],
        }

    def close(self) -> None:
        """
        Explicitly close the dedicated maintenance logger handlers if needed.
        Useful for scripts/tests.
        """
        if self.db_log_manager:
            self.db_log_manager.close()

    # -------------------------------------------------------------------------
    # Bulk position build workflow
    # -------------------------------------------------------------------------
    def build_positions_from_existing_data(
        self,
        *,
        create_asset_only_positions: bool = True,
        include_model_level_locations: bool = True,
    ) -> dict:
        """
        Build Position rows from existing related hierarchy data already in the DB.

        Intended use:
        - after bootstrap loads of area / equipment_group / model / asset_number / location
        - before image/drawing association rebuilds

        Rules are implemented in PositionService.build_positions_from_existing_data(...)

        Default behavior:
        - create asset-only positions with location_id=None
        - create model-level generic positions for locations where asset_number_id is NULL
        - create asset-specific positions for locations tied to a specific asset_number_id
        """
        session = self.db_config.get_main_session()

        with self.db_log_manager.timed_operation("build_positions_from_existing_data"):
            try:
                self.logger.info(
                    "Building positions from existing data "
                    "(create_asset_only_positions=%s, include_model_level_locations=%s)",
                    create_asset_only_positions,
                    include_model_level_locations,
                )

                result = PositionService.build_positions_from_existing_data(
                    session,
                    create_asset_only_positions=create_asset_only_positions,
                    include_model_level_locations=include_model_level_locations,
                )

                if not result.get("success", False):
                    session.rollback()
                    self.logger.warning(
                        "PositionService returned unsuccessful result during bulk build: %s",
                        result.get("message"),
                    )
                    return self._build_result(
                        False,
                        result.get("message", "Position build failed."),
                        data=result.get("data", {}),
                        errors=result.get("errors", []),
                    )

                session.commit()

                totals = result.get("data", {}).get("totals", {})
                self.logger.info(
                    "Bulk position build completed successfully. "
                    "positions_created=%s, positions_reused=%s, errors=%s",
                    totals.get("positions_created", 0),
                    totals.get("positions_reused", 0),
                    totals.get("errors", 0),
                )

                return self._build_result(
                    True,
                    result.get("message", "Position build from existing data completed successfully."),
                    data=result.get("data", {}),
                    errors=result.get("errors", []),
                )

            except SQLAlchemyError as exc:
                session.rollback()
                self.logger.exception("Failed to build positions from existing data: %s", exc)
                return self._build_result(
                    False,
                    "Failed to build positions from existing data.",
                    errors=[str(exc)],
                )
            except Exception as exc:
                session.rollback()
                self.logger.exception(
                    "Unexpected error in build_positions_from_existing_data: %s",
                    exc,
                )
                return self._build_result(
                    False,
                    "Unexpected error while building positions from existing data.",
                    errors=[str(exc)],
                )
            finally:
                session.close()

    # -------------------------------------------------------------------------
    # Main full hierarchy workflow
    # -------------------------------------------------------------------------
    def create_or_resolve_position_hierarchy(
        self,
        *,
        campus_name: str | None = None,
        campus_description: str | None = None,
        campus_city: str | None = None,
        campus_state: str | None = None,
        campus_country: str | None = None,
        building_name: str | None = None,
        building_description: str | None = None,
        building_address: str | None = None,
        site_location_title: str | None = None,
        site_location_room_number: str | None = None,
        site_location_area: str | None = None,
        area_name: str | None = None,
        area_description: str | None = None,
        equipment_group_name: str | None = None,
        equipment_group_description: str | None = None,
        model_name: str | None = None,
        model_description: str | None = None,
        asset_number_value: str | None = None,
        asset_number_description: str | None = None,
        location_name: str | None = None,
        location_description: str | None = None,
        subassembly_name: str | None = None,
        subassembly_description: str | None = None,
        component_assembly_name: str | None = None,
        component_assembly_description: str | None = None,
        assembly_view_name: str | None = None,
        assembly_view_description: str | None = None,
    ) -> dict:
        """
        Creates or resolves the full hierarchy and returns the final Position.

        Only the provided levels are created/resolved.
        Missing lower levels are allowed.
        """
        session = self.db_config.get_main_session()

        with self.db_log_manager.timed_operation("create_or_resolve_position_hierarchy"):
            try:
                created = {}

                campus = None
                if self._clean_str(campus_name):
                    self.logger.info("Resolving campus '%s'", self._clean_str(campus_name))
                    campus = CampusService.find_or_create(
                        session,
                        name=self._clean_str(campus_name),
                        description=self._clean_str(campus_description),
                        city=self._clean_str(campus_city),
                        state=self._clean_str(campus_state),
                        country=self._clean_str(campus_country),
                    )
                    created["campus"] = {"id": campus.id, "name": campus.name}

                building = None
                if self._clean_str(building_name) and campus:
                    self.logger.info(
                        "Resolving building '%s' for campus_id=%s",
                        self._clean_str(building_name),
                        campus.id,
                    )
                    building = BuildingService.find_or_create(
                        session,
                        name=self._clean_str(building_name),
                        campus_id=campus.id,
                        description=self._clean_str(building_description),
                        address=self._clean_str(building_address),
                    )
                    created["building"] = {"id": building.id, "name": building.name}

                site_location = None
                if self._clean_str(site_location_title):
                    self.logger.info(
                        "Resolving site location '%s'",
                        self._clean_str(site_location_title),
                    )
                    site_location = SiteLocationService.find_or_create(
                        session,
                        title=self._clean_str(site_location_title),
                        room_number=self._clean_str(site_location_room_number) or "Unknown",
                        site_area=self._clean_str(site_location_area) or "General",
                        building_id=building.id if building else None,
                    )
                    created["site_location"] = {
                        "id": site_location.id,
                        "title": site_location.title,
                    }

                area = None
                if self._clean_str(area_name):
                    self.logger.info("Resolving area '%s'", self._clean_str(area_name))
                    area = AreaService.find_or_create(
                        session,
                        name=self._clean_str(area_name),
                        description=self._clean_str(area_description),
                    )
                    created["area"] = {"id": area.id, "name": area.name}

                equipment_group = None
                if self._clean_str(equipment_group_name) and area:
                    self.logger.info(
                        "Resolving equipment group '%s' for area_id=%s",
                        self._clean_str(equipment_group_name),
                        area.id,
                    )
                    equipment_group = EquipmentGroupService.find_or_create(
                        session,
                        name=self._clean_str(equipment_group_name),
                        area_id=area.id,
                        description=self._clean_str(equipment_group_description),
                    )
                    created["equipment_group"] = {
                        "id": equipment_group.id,
                        "name": equipment_group.name,
                    }

                model = None
                if self._clean_str(model_name) and equipment_group:
                    self.logger.info(
                        "Resolving model '%s' for equipment_group_id=%s",
                        self._clean_str(model_name),
                        equipment_group.id,
                    )
                    model = ModelService.find_or_create(
                        session,
                        name=self._clean_str(model_name),
                        equipment_group_id=equipment_group.id,
                        description=self._clean_str(model_description),
                    )
                    created["model"] = {"id": model.id, "name": model.name}

                asset_number = None
                if self._clean_str(asset_number_value) and model:
                    self.logger.info(
                        "Resolving asset number '%s' for model_id=%s",
                        self._clean_str(asset_number_value),
                        model.id,
                    )
                    asset_number = AssetNumberService.find_or_create(
                        session,
                        number=self._clean_str(asset_number_value),
                        model_id=model.id,
                        description=self._clean_str(asset_number_description),
                    )
                    created["asset_number"] = {
                        "id": asset_number.id,
                        "number": asset_number.number,
                    }

                location = None
                if self._clean_str(location_name) and model:
                    self.logger.info(
                        "Resolving location '%s' for model_id=%s",
                        self._clean_str(location_name),
                        model.id,
                    )
                    location = LocationService.find_or_create(
                        session,
                        name=self._clean_str(location_name),
                        model_id=model.id,
                        asset_number_id=asset_number.id if asset_number else None,
                        description=self._clean_str(location_description),
                    )
                    created["location"] = {"id": location.id, "name": location.name}

                subassembly = None
                if self._clean_str(subassembly_name) and location:
                    self.logger.info(
                        "Resolving subassembly '%s' for location_id=%s",
                        self._clean_str(subassembly_name),
                        location.id,
                    )
                    subassembly = SubassemblyService.find_or_create(
                        session,
                        name=self._clean_str(subassembly_name),
                        location_id=location.id,
                        asset_number_id=asset_number.id if asset_number else None,
                        description=self._clean_str(subassembly_description),
                    )
                    created["subassembly"] = {
                        "id": subassembly.id,
                        "name": subassembly.name,
                    }

                component_assembly = None
                if self._clean_str(component_assembly_name) and subassembly:
                    self.logger.info(
                        "Resolving component assembly '%s' for subassembly_id=%s",
                        self._clean_str(component_assembly_name),
                        subassembly.id,
                    )
                    component_assembly = ComponentAssemblyService.find_or_create(
                        session,
                        name=self._clean_str(component_assembly_name),
                        subassembly_id=subassembly.id,
                        asset_number_id=asset_number.id if asset_number else None,
                        description=self._clean_str(component_assembly_description),
                    )
                    created["component_assembly"] = {
                        "id": component_assembly.id,
                        "name": component_assembly.name,
                    }

                assembly_view = None
                if self._clean_str(assembly_view_name) and component_assembly:
                    self.logger.info(
                        "Resolving assembly view '%s' for component_assembly_id=%s",
                        self._clean_str(assembly_view_name),
                        component_assembly.id,
                    )
                    assembly_view = AssemblyViewService.find_or_create(
                        session,
                        name=self._clean_str(assembly_view_name),
                        component_assembly_id=component_assembly.id,
                        asset_number_id=asset_number.id if asset_number else None,
                        description=self._clean_str(assembly_view_description),
                    )
                    created["assembly_view"] = {
                        "id": assembly_view.id,
                        "name": assembly_view.name,
                    }

                self.logger.info("Resolving final position record")
                position = PositionService.find_or_create(
                    session,
                    area_id=area.id if area else None,
                    equipment_group_id=equipment_group.id if equipment_group else None,
                    model_id=model.id if model else None,
                    asset_number_id=asset_number.id if asset_number else None,
                    location_id=location.id if location else None,
                    subassembly_id=subassembly.id if subassembly else None,
                    component_assembly_id=component_assembly.id if component_assembly else None,
                    assembly_view_id=assembly_view.id if assembly_view else None,
                    site_location_id=site_location.id if site_location else None,
                    campus_id=campus.id if campus else None,
                    building_id=building.id if building else None,
                )

                session.commit()

                created["position"] = {
                    "id": position.id,
                    "area_id": position.area_id,
                    "equipment_group_id": position.equipment_group_id,
                    "model_id": position.model_id,
                    "asset_number_id": position.asset_number_id,
                    "location_id": position.location_id,
                    "subassembly_id": position.subassembly_id,
                    "component_assembly_id": position.component_assembly_id,
                    "assembly_view_id": position.assembly_view_id,
                    "site_location_id": position.site_location_id,
                    "campus_id": position.campus_id,
                    "building_id": position.building_id,
                }

                self.logger.info(
                    "Position hierarchy created/resolved successfully. position_id=%s",
                    position.id,
                )

                return self._build_result(
                    True,
                    "Position hierarchy created/resolved successfully.",
                    data=created,
                )

            except SQLAlchemyError as exc:
                session.rollback()
                self.logger.exception("Failed to create/resolve position hierarchy: %s", exc)
                return self._build_result(
                    False,
                    "Failed to create/resolve position hierarchy.",
                    errors=[str(exc)],
                )
            except Exception as exc:
                session.rollback()
                self.logger.exception(
                    "Unexpected error in create_or_resolve_position_hierarchy: %s",
                    exc,
                )
                return self._build_result(
                    False,
                    "Unexpected error while creating/resolving position hierarchy.",
                    errors=[str(exc)],
                )
            finally:
                session.close()

    # -------------------------------------------------------------------------
    # Position CRUD / lookup
    # -------------------------------------------------------------------------
    def get_position_by_id(self, position_id: int) -> dict:
        session = self.db_config.get_main_session()

        with self.db_log_manager.timed_operation("get_position_by_id"):
            try:
                self.logger.info("Retrieving position_id=%s", position_id)
                position = PositionService.get_by_id(session, position_id)
                if not position:
                    self.logger.warning("Position id=%s was not found", position_id)
                    return self._build_result(
                        False,
                        f"Position id={position_id} was not found.",
                    )

                return self._build_result(
                    True,
                    "Position retrieved successfully.",
                    data={
                        "position": {
                            "id": position.id,
                            "area_id": position.area_id,
                            "equipment_group_id": position.equipment_group_id,
                            "model_id": position.model_id,
                            "asset_number_id": position.asset_number_id,
                            "location_id": position.location_id,
                            "subassembly_id": position.subassembly_id,
                            "component_assembly_id": position.component_assembly_id,
                            "assembly_view_id": position.assembly_view_id,
                            "site_location_id": position.site_location_id,
                            "campus_id": position.campus_id,
                            "building_id": position.building_id,
                        }
                    },
                )

            except Exception as exc:
                self.logger.exception("Failed to get position by id: %s", exc)
                return self._build_result(
                    False,
                    "Failed to retrieve position.",
                    errors=[str(exc)],
                )
            finally:
                session.close()

    def search_positions(self, **filters) -> dict:
        session = self.db_config.get_main_session()

        with self.db_log_manager.timed_operation("search_positions"):
            try:
                self.logger.info("Searching positions with filters=%s", filters)
                positions = PositionService.search(session, **filters)
                return self._build_result(
                    True,
                    "Positions retrieved successfully.",
                    data={
                        "count": len(positions),
                        "positions": [
                            {
                                "id": p.id,
                                "area_id": p.area_id,
                                "equipment_group_id": p.equipment_group_id,
                                "model_id": p.model_id,
                                "asset_number_id": p.asset_number_id,
                                "location_id": p.location_id,
                                "subassembly_id": p.subassembly_id,
                                "component_assembly_id": p.component_assembly_id,
                                "assembly_view_id": p.assembly_view_id,
                                "site_location_id": p.site_location_id,
                                "campus_id": p.campus_id,
                                "building_id": p.building_id,
                            }
                            for p in positions
                        ],
                    },
                )
            except Exception as exc:
                self.logger.exception("Failed to search positions: %s", exc)
                return self._build_result(
                    False,
                    "Failed to search positions.",
                    errors=[str(exc)],
                )
            finally:
                session.close()

    def update_position(self, position_id: int, **kwargs) -> dict:
        session = self.db_config.get_main_session()

        with self.db_log_manager.timed_operation("update_position"):
            try:
                self.logger.info(
                    "Updating position_id=%s with fields=%s",
                    position_id,
                    list(kwargs.keys()),
                )
                position = PositionService.update(session, position_id, **kwargs)
                if not position:
                    session.rollback()
                    self.logger.warning("Position id=%s was not found for update", position_id)
                    return self._build_result(
                        False,
                        f"Position id={position_id} was not found.",
                    )

                session.commit()

                self.logger.info("Position id=%s updated successfully", position_id)

                return self._build_result(
                    True,
                    "Position updated successfully.",
                    data={
                        "position": {
                            "id": position.id,
                            "area_id": position.area_id,
                            "equipment_group_id": position.equipment_group_id,
                            "model_id": position.model_id,
                            "asset_number_id": position.asset_number_id,
                            "location_id": position.location_id,
                            "subassembly_id": position.subassembly_id,
                            "component_assembly_id": position.component_assembly_id,
                            "assembly_view_id": position.assembly_view_id,
                            "site_location_id": position.site_location_id,
                            "campus_id": position.campus_id,
                            "building_id": position.building_id,
                        }
                    },
                )

            except SQLAlchemyError as exc:
                session.rollback()
                self.logger.exception("Failed to update position: %s", exc)
                return self._build_result(
                    False,
                    "Failed to update position.",
                    errors=[str(exc)],
                )
            finally:
                session.close()

    def delete_position(self, position_id: int) -> dict:
        session = self.db_config.get_main_session()

        with self.db_log_manager.timed_operation("delete_position"):
            try:
                self.logger.info("Deleting position_id=%s", position_id)
                deleted = PositionService.delete(session, position_id)
                if not deleted:
                    session.rollback()
                    self.logger.warning("Position id=%s was not found for deletion", position_id)
                    return self._build_result(
                        False,
                        f"Position id={position_id} was not found.",
                    )

                session.commit()

                self.logger.info("Position id=%s deleted successfully", position_id)

                return self._build_result(
                    True,
                    f"Position id={position_id} deleted successfully.",
                )

            except SQLAlchemyError as exc:
                session.rollback()
                self.logger.exception("Failed to delete position: %s", exc)
                return self._build_result(
                    False,
                    "Failed to delete position.",
                    errors=[str(exc)],
                )
            finally:
                session.close()

    # -------------------------------------------------------------------------
    # Hierarchy helpers
    # -------------------------------------------------------------------------
    def get_position_related_entities(self, position_id: int) -> dict:
        session = self.db_config.get_main_session()

        with self.db_log_manager.timed_operation("get_position_related_entities"):
            try:
                self.logger.info("Retrieving related entities for position_id=%s", position_id)
                position = PositionService.get_by_id(session, position_id)
                if not position:
                    self.logger.warning(
                        "Position id=%s was not found while retrieving related entities",
                        position_id,
                    )
                    return self._build_result(
                        False,
                        f"Position id={position_id} was not found.",
                    )

                return self._build_result(
                    True,
                    "Position related entities retrieved successfully.",
                    data={
                        "position": {
                            "id": position.id,
                            "area": position.area,
                            "equipment_group": position.equipment_group,
                            "model": position.model,
                            "asset_number": position.asset_number,
                            "location": position.location,
                            "subassembly": position.subassembly,
                            "component_assembly": position.component_assembly,
                            "assembly_view": position.assembly_view,
                            "site_location": position.site_location,
                            "campus": position.campus,
                            "building": position.building,
                        }
                    },
                )

            except Exception as exc:
                self.logger.exception("Failed to get position related entities: %s", exc)
                return self._build_result(
                    False,
                    "Failed to retrieve position related entities.",
                    errors=[str(exc)],
                )
            finally:
                session.close()

    def get_dependent_items(
        self,
        *,
        parent_type: str,
        parent_id: int,
        child_type: str | None = None,
    ) -> dict:
        session = self.db_config.get_main_session()

        with self.db_log_manager.timed_operation("get_dependent_items"):
            try:
                self.logger.info(
                    "Getting dependent items for parent_type=%s parent_id=%s child_type=%s",
                    parent_type,
                    parent_id,
                    child_type,
                )
                items = HierarchyService.get_dependent_items(
                    session,
                    parent_type=parent_type,
                    parent_id=parent_id,
                    child_type=child_type,
                )

                return self._build_result(
                    True,
                    "Dependent items retrieved successfully.",
                    data={
                        "parent_type": parent_type,
                        "parent_id": parent_id,
                        "child_type": child_type,
                        "count": len(items),
                        "items": items,
                    },
                )

            except Exception as exc:
                self.logger.exception("Failed to get dependent items: %s", exc)
                return self._build_result(
                    False,
                    "Failed to retrieve dependent items.",
                    errors=[str(exc)],
                )
            finally:
                session.close()

    def get_positions_by_hierarchy(self, **filters) -> dict:
        session = self.db_config.get_main_session()

        with self.db_log_manager.timed_operation("get_positions_by_hierarchy"):
            try:
                self.logger.info("Getting positions by hierarchy with filters=%s", filters)
                positions = HierarchyService.get_positions_by_hierarchy(session, **filters)

                return self._build_result(
                    True,
                    "Hierarchy positions retrieved successfully.",
                    data={
                        "count": len(positions),
                        "positions": positions,
                    },
                )

            except Exception as exc:
                self.logger.exception("Failed to get positions by hierarchy: %s", exc)
                return self._build_result(
                    False,
                    "Failed to retrieve positions by hierarchy.",
                    errors=[str(exc)],
                )
            finally:
                session.close()