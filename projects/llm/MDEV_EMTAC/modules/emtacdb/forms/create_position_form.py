# modules/emtacdb/forms/create_position_form.py

from flask_wtf import FlaskForm
from wtforms import SubmitField, StringField, TextAreaField
from wtforms.validators import Optional
from wtforms_sqlalchemy.fields import QuerySelectField

from modules.emtacdb.emtacdb_fts import (Campus,Building, Area,EquipmentGroup,Model,AssetNumber,Location,Subassembly,
                                         ComponentAssembly,AssemblyView,SiteLocation,Position,)
from modules.configuration.log_config import logger
from modules.configuration.config_env import DatabaseConfig

database_config = DatabaseConfig()

# ----------------------------------------------------------------------
# Default Campus / Building IDs
# If user does not explicitly pick a Campus/Building, these will be used.
# You can change them later or set them to None to disable automatic defaults.
# ----------------------------------------------------------------------
DEFAULT_CAMPUS_ID = 1
DEFAULT_BUILDING_ID = 1

def cnp_form_create_position(
    campus_id,
    building_id,
    area_id,
    equipment_group_id,
    model_id,
    asset_number_id,
    location_id,
    site_location_id,
    subassembly_id,
    component_assembly_id,
    assembly_view_id,
    session,
):
    """
    Create or reuse a Position row based on the combination of hierarchy objects.

    - ALL IDs are optional.
    - Will reuse existing Position if identical combination exists.
    - If campus_id or building_id are missing but can be inferred from SiteLocation,
      that inference will be performed.
    """

    try:
        logger.debug(
            "cnp_form_create_position called with: "
            "campus_id=%s, building_id=%s, area_id=%s, equipment_group_id=%s, "
            "model_id=%s, asset_number_id=%s, location_id=%s, site_location_id=%s, "
            "subassembly_id=%s, component_assembly_id=%s, assembly_view_id=%s",
            campus_id, building_id,
            area_id, equipment_group_id,
            model_id, asset_number_id,
            location_id, site_location_id,
            subassembly_id, component_assembly_id, assembly_view_id,
        )

        # ------------------------------------------------------------
        # AUTO-RESOLVE campus_id / building_id FROM SiteLocation IF:
        # - form did not explicitly provide them
        # - SiteLocation knows them
        # ------------------------------------------------------------
        if site_location_id and (campus_id is None or building_id is None):
            site = session.query(SiteLocation).filter_by(id=site_location_id).first()

            if site:
                if building_id is None and hasattr(site, "building_id"):
                    building_id = site.building_id
                    logger.debug(
                        "Auto-resolved building_id=%s from site_location_id=%s",
                        building_id, site_location_id
                    )

                if campus_id is None and hasattr(site, "building") and site.building:
                    campus_id = site.building.campus_id
                    logger.debug(
                        "Auto-resolved campus_id=%s from building_id=%s",
                        campus_id, building_id
                    )

        # ------------------------------------------------------------
        # Delegate to Position.add_to_db()
        # This handles FK uniqueness, reuse, creation, commit, return ID.
        # ------------------------------------------------------------
        pos_id = Position.add_to_db(
            session=session,
            campus_id=campus_id,
            building_id=building_id,
            area_id=area_id,
            equipment_group_id=equipment_group_id,
            model_id=model_id,
            asset_number_id=asset_number_id,
            location_id=location_id,
            site_location_id=site_location_id,
            subassembly_id=subassembly_id,
            component_assembly_id=component_assembly_id,
            assembly_view_id=assembly_view_id,
        )

        logger.debug("Position created or reused with ID=%s", pos_id)
        return pos_id

    except Exception as e:
        logger.error("Error in cnp_form_create_position: %s", e, exc_info=True)
        session.rollback()
        raise


class CreatePositionForm(FlaskForm):
    """
    Form for creating a Position, with optional creation of any missing
    hierarchy nodes (Campus, Building, Area, EquipmentGroup, Model,
    AssetNumber, Location, Subassembly, ComponentAssembly, AssemblyView,
    SiteLocation).

    Design goals:
    - Allow minimal positions (e.g., only Campus/Building + Area).
    - Allow deep positions (all the way down to AssemblyView).
    - Avoid duplicates by reusing existing rows where possible.
    """

    # --- CAMPUS ---
    campus = QuerySelectField(
        label="Campus",
        query_factory=lambda: [],
        allow_blank=True,
        blank_text="Select a Campus",
        validators=[Optional()],
        get_label="name",
        render_kw={
            "id": "campusDropdown",
            "data-toggle-input": "create_position_form-campus_input",
        },
    )
    campus_input = StringField(
        label="New Campus",
        validators=[Optional()],
        render_kw={
            "id": "campusInput",
            "placeholder": "Enter new Campus if not listed",
        },
    )

    # --- BUILDING ---
    building = QuerySelectField(
        label="Building",
        query_factory=lambda: [],
        allow_blank=True,
        blank_text="Select a Building",
        validators=[Optional()],
        get_label="name",
        render_kw={
            "id": "buildingDropdown",
            "data-toggle-input": "create_position_form-building_input",
        },
    )
    building_input = StringField(
        label="New Building",
        validators=[Optional()],
        render_kw={
            "id": "buildingInput",
            "placeholder": "Enter new Building if not listed",
        },
    )

    # --- AREA ---
    area = QuerySelectField(
        label="Area",
        query_factory=lambda: [],
        allow_blank=True,
        blank_text="Select an Area",
        validators=[Optional()],
        get_label="name",
        render_kw={
            "id": "areaDropdown",
            "data-toggle-input": "create_position_form-area_input",
        },
    )
    area_input = StringField(
        label="New Area",
        validators=[Optional()],
        render_kw={
            "id": "areaInput",
            "placeholder": "Enter new Area if not listed",
        },
    )

    # --- EQUIPMENT GROUP ---
    equipment_group = QuerySelectField(
        label="Equipment Group",
        query_factory=lambda: [],
        allow_blank=True,
        blank_text="Select an Equipment Group",
        validators=[Optional()],
        get_label="name",
        render_kw={
            "id": "equipmentGroupDropdown",
            "data-toggle-input": "create_position_form-equipment_group_input",
        },
    )
    equipment_group_input = StringField(
        label="New Equipment Group",
        validators=[Optional()],
        render_kw={
            "id": "equipmentGroupInput",
            "placeholder": "Enter new Equipment Group if not listed",
        },
    )

    # --- MODEL (with description) ---
    model = QuerySelectField(
        label="Model",
        query_factory=lambda: [],
        allow_blank=True,
        blank_text="Select a Model",
        validators=[Optional()],
        get_label="name",
        render_kw={
            "id": "modelDropdown",
            "data-toggle-input": "create_position_form-model_input",
        },
    )
    model_input = StringField(
        label="New Model",
        validators=[Optional()],
        render_kw={
            "id": "modelInput",
            "placeholder": "Enter new Model if not listed",
        },
    )
    model_description = TextAreaField(
        label="Model Description",
        validators=[Optional()],
        render_kw={
            "id": "modelDescription",
            "placeholder": "Enter description for new Model if not listed",
        },
    )

    # --- ASSET NUMBER ---
    asset_number = QuerySelectField(
        label="Asset Number",
        query_factory=lambda: [],
        allow_blank=True,
        blank_text="Select an Asset Number",
        validators=[Optional()],
        get_label=lambda asset: (
            f"{asset.number} - {asset.description}"
            if asset.description
            else asset.number
        ),
        render_kw={
            "id": "assetNumberDropdown",
            "data-toggle-input": "create_position_form-asset_number_input",
        },
    )
    asset_number_input = StringField(
        label="New Asset Number",
        validators=[Optional()],
        render_kw={
            "id": "assetNumberInput",
            "placeholder": "Enter new Asset Number if not listed",
        },
    )
    asset_number_description = TextAreaField(
        label="Asset Number Description",
        validators=[Optional()],
        render_kw={
            "id": "assetNumberDescription",
            "placeholder": "Enter description for new Asset Number if not listed",
        },
    )

    # --- LOCATION ---
    location = QuerySelectField(
        label="Location",
        query_factory=lambda: [],
        allow_blank=True,
        blank_text="Select a Location",
        validators=[Optional()],
        get_label="name",
        render_kw={
            "id": "locationDropdown",
            "data-toggle-input": "create_position_form-location_input",
        },
    )
    location_input = StringField(
        label="New Location",
        validators=[Optional()],
        render_kw={
            "id": "locationInput",
            "placeholder": "Enter new Location if not listed",
        },
    )

    # --- SUBASSEMBLY ---
    subassembly = QuerySelectField(
        label="Subassembly",
        query_factory=lambda: [],
        allow_blank=True,
        blank_text="Select a Subassembly",
        validators=[Optional()],
        get_label="name",
        render_kw={
            "id": "assemblyDropdown",
            "data-toggle-input": "create_position_form-assembly_input",
        },
    )
    subassembly_input = StringField(
        label="New Subassembly",
        validators=[Optional()],
        render_kw={
            "id": "assemblyInput",
            "placeholder": "Enter new Subassembly if not listed",
        },
    )

    # --- COMPONENT SUBASSEMBLY (ComponentAssembly) ---
    component_assembly = QuerySelectField(
        label="Component Subassembly",
        query_factory=lambda: [],
        allow_blank=True,
        blank_text="Select a Component Subassembly",
        validators=[Optional()],
        get_label="name",
        render_kw={
            "id": "componentAssemblyDropdown",
            "data-toggle-input": "create_position_form-component_assembly_input",
        },
    )
    component_assembly_input = StringField(
        label="New Component Subassembly",
        validators=[Optional()],
        render_kw={
            "id": "componentAssemblyInput",
            "placeholder": "Enter new Component Subassembly if not listed",
        },
    )

    # --- SUBASSEMBLY VIEW (AssemblyView) ---
    assembly_view = QuerySelectField(
        label="Subassembly View",
        query_factory=lambda: [],
        allow_blank=True,
        blank_text="Select a Subassembly View",
        validators=[Optional()],
        get_label="name",
        render_kw={
            "id": "assemblyViewDropdown",
            "data-toggle-input": "create_position_form-assembly_view_input",
        },
    )
    assembly_view_input = StringField(
        label="New Subassembly View",
        validators=[Optional()],
        render_kw={
            "id": "assemblyViewInput",
            "placeholder": "Enter new Subassembly View if not listed",
        },
    )

    # --- SITE LOCATION (title + room) ---
    site_location = QuerySelectField(
        label="Site Location",
        query_factory=lambda: [],
        allow_blank=True,
        blank_text="Select a Site Location",
        validators=[Optional()],
        get_label=lambda site_location: (
            f"{site_location.title} - Room {site_location.room_number}"
        ),
        render_kw={
            "id": "siteLocationDropdown",
            "data-toggle-input": "create_position_form-site_location_input",
        },
    )
    site_location_input = StringField(
        label="New Site Location Title",
        validators=[Optional()],
        render_kw={
            "id": "siteLocationInput",
            "placeholder": "Enter new Site Location title if not listed",
        },
    )
    site_location_room = StringField(
        label="Room Number",
        validators=[Optional()],
        render_kw={
            "id": "siteLocationRoom",
            "placeholder": "Enter Room Number",
        },
    )

    submit = SubmitField("Create Position")

    # ------------------------------------------------------------------
    # Custom validation: make sure we don't specify BOTH existing + new
    # for the same conceptual field.
    # ------------------------------------------------------------------
    def validate(self, *args, **kwargs):
        if not super(CreatePositionForm, self).validate(*args, **kwargs):
            return False

        success = True
        field_pairs = [
            (self.campus, self.campus_input, "Campus"),
            (self.building, self.building_input, "Building"),
            (self.area, self.area_input, "Area"),
            (self.equipment_group, self.equipment_group_input, "Equipment Group"),
            (self.model, self.model_input, "Model"),
            (self.asset_number, self.asset_number_input, "Asset Number"),
            (self.location, self.location_input, "Location"),
            (self.subassembly, self.subassembly_input, "Subassembly"),
            (
                self.component_assembly,
                self.component_assembly_input,
                "Component Subassembly",
            ),
            (self.assembly_view, self.assembly_view_input, "Subassembly View"),
            (self.site_location, self.site_location_input, "Site Location"),
        ]

        for select_field, input_field, field_name in field_pairs:
            if select_field.data and input_field.data:
                msg = (
                    f"Please provide either a selected {field_name} or a new "
                    f"{field_name}, not both."
                )
                select_field.errors.append(msg)
                input_field.errors.append(msg)
                success = False

        return success

    # ------------------------------------------------------------------
    # Hook to populate the dropdowns from a live session.
    # ------------------------------------------------------------------
    def set_query_factories(self, session):
        # Campus & Building
        self.campus.query_factory = (
            lambda: session.query(Campus).order_by(Campus.name).all()
        )
        self.building.query_factory = (
            lambda: session.query(Building).order_by(Building.name).all()
        )

        # Existing hierarchy
        self.area.query_factory = lambda: session.query(Area).order_by(Area.name).all()
        self.equipment_group.query_factory = (
            lambda: session.query(EquipmentGroup)
            .order_by(EquipmentGroup.name)
            .all()
        )
        self.model.query_factory = (
            lambda: session.query(Model).order_by(Model.name).all()
        )
        self.asset_number.query_factory = (
            lambda: session.query(AssetNumber)
            .order_by(AssetNumber.number)
            .all()
        )
        self.location.query_factory = (
            lambda: session.query(Location).order_by(Location.name).all()
        )
        self.subassembly.query_factory = (
            lambda: session.query(Subassembly).order_by(Subassembly.name).all()
        )
        self.component_assembly.query_factory = (
            lambda: session.query(ComponentAssembly)
            .order_by(ComponentAssembly.name)
            .all()
        )
        self.assembly_view.query_factory = (
            lambda: session.query(AssemblyView)
            .order_by(AssemblyView.name)
            .all()
        )
        self.site_location.query_factory = (
            lambda: session.query(SiteLocation)
            .order_by(SiteLocation.title, SiteLocation.room_number)
            .all()
        )

    # ------------------------------------------------------------------
    # Main save() – creates any needed hierarchy rows and then delegates
    # Position creation to cnp_form_create_position.
    # ------------------------------------------------------------------
    def save(self, session, cnp_form_create_position):
        logger.debug("=== Starting save() in CreatePositionForm ===")

        # --------------------------------------------------------------
        # 0. CAMPUS (with default fallback)
        # --------------------------------------------------------------
        if self.campus_input.data:
            name = self.campus_input.data
            logger.debug("New Campus input provided: %s", name)
            # Use Campus.find_or_create to avoid duplicates
            campus_obj = Campus.find_or_create(
                session,
                name=name,
                request_id=None,  # or pass through if you have one
            )
            campus_id = campus_obj.id
            logger.debug("Using Campus (created or existing) with ID: %s", campus_id)
        elif self.campus.data:
            campus_id = self.campus.data.id
            logger.debug("Using selected Campus with ID: %s", campus_id)
        else:
            # Fall back to default campus if configured and present
            campus_id = None
            if DEFAULT_CAMPUS_ID is not None:
                default_campus = session.get(Campus, DEFAULT_CAMPUS_ID)
                if default_campus:
                    campus_id = DEFAULT_CAMPUS_ID
                    logger.debug(
                        "Using DEFAULT_CAMPUS_ID: %s for Campus", DEFAULT_CAMPUS_ID
                    )
                else:
                    logger.warning(
                        "DEFAULT_CAMPUS_ID=%s not found in DB; campus_id=None",
                        DEFAULT_CAMPUS_ID,
                    )
            else:
                logger.debug("No Campus provided and DEFAULT_CAMPUS_ID is None.")

        # --------------------------------------------------------------
        # 1. BUILDING (with default fallback; must be linked to a Campus)
        # --------------------------------------------------------------
        if self.building_input.data:
            name = self.building_input.data
            logger.debug("New Building input provided: %s", name)

            if campus_id is None:
                raise ValueError(
                    "A Campus (selected or default) is required to create a Building."
                )

            building_obj = Building.find_or_create(
                session,
                name=name,
                campus_id=campus_id,
                request_id=None,
            )
            building_id = building_obj.id
            logger.debug(
                "Using Building (created or existing) with ID: %s", building_id
            )
        elif self.building.data:
            building_id = self.building.data.id
            # force campus alignment
            if campus_id is None:
                campus_id = self.building.data.campus_id
            logger.debug("Using selected Building with ID: %s", building_id)
        else:
            building_id = None
            if DEFAULT_BUILDING_ID is not None:
                default_building = session.get(Building, DEFAULT_BUILDING_ID)
                if default_building:
                    building_id = DEFAULT_BUILDING_ID
                    logger.debug(
                        "Using DEFAULT_BUILDING_ID: %s for Building",
                        DEFAULT_BUILDING_ID,
                    )
                else:
                    logger.warning(
                        "DEFAULT_BUILDING_ID=%s not found in DB; building_id=None",
                        DEFAULT_BUILDING_ID,
                    )
            else:
                logger.debug("No Building provided and DEFAULT_BUILDING_ID is None.")

        # --------------------------------------------------------------
        # 2. AREA
        # --------------------------------------------------------------
        if self.area_input.data:
            logger.debug("New Area input provided: %s", self.area_input.data)
            new_area = Area(name=self.area_input.data)
            session.add(new_area)
            session.commit()
            area_id = new_area.id
            logger.debug("Created new Area with ID: %s", area_id)
        elif self.area.data:
            area_id = self.area.data.id
            logger.debug("Using selected Area with ID: %s", area_id)
        else:
            area_id = None
            logger.debug("No Area provided.")

        # --------------------------------------------------------------
        # 3. EQUIPMENT GROUP (linked to Area via area_id)
        # --------------------------------------------------------------
        if self.equipment_group_input.data:
            logger.debug(
                "New Equipment Group input provided: %s",
                self.equipment_group_input.data,
            )
            new_eq_group = EquipmentGroup(
                name=self.equipment_group_input.data,
                area_id=area_id,
            )
            session.add(new_eq_group)
            session.commit()
            equipment_group_id = new_eq_group.id
            logger.debug(
                "Created new EquipmentGroup with ID: %s (Area ID: %s)",
                equipment_group_id,
                area_id,
            )
        elif self.equipment_group.data:
            equipment_group_id = self.equipment_group.data.id
            logger.debug(
                "Using selected EquipmentGroup with ID: %s",
                equipment_group_id,
            )
        else:
            equipment_group_id = None
            logger.debug("No EquipmentGroup provided.")

        # --------------------------------------------------------------
        # 4. MODEL (optionally linked to EquipmentGroup)
        # --------------------------------------------------------------
        if self.model_input.data:
            logger.debug("New Model input provided: %s", self.model_input.data)
            new_model = Model(
                name=self.model_input.data,
                description=self.model_description.data or "",
                equipment_group_id=equipment_group_id,
            )
            session.add(new_model)
            session.commit()
            model_id = new_model.id
            logger.debug(
                "Created new Model with ID: %s (EquipmentGroup ID: %s)",
                model_id,
                equipment_group_id,
            )
        elif self.model.data:
            model_id = self.model.data.id
            logger.debug("Using selected Model with ID: %s", model_id)
        else:
            model_id = None
            logger.debug("No Model provided.")

        # --------------------------------------------------------------
        # 5. ASSET NUMBER (linked to Model)
        # --------------------------------------------------------------
        if self.asset_number_input.data:
            logger.debug(
                "New Asset Number input provided: %s",
                self.asset_number_input.data,
            )
            new_asset_number = AssetNumber(
                number=self.asset_number_input.data,
                description=self.asset_number_description.data or "",
                model_id=model_id,
            )
            session.add(new_asset_number)
            session.commit()
            asset_number_id = new_asset_number.id
            logger.debug(
                "Created new AssetNumber with ID: %s (linked to Model ID: %s)",
                asset_number_id,
                model_id,
            )
        elif self.asset_number.data:
            asset_number_id = self.asset_number.data.id
            logger.debug(
                "Using selected AssetNumber with ID: %s", asset_number_id
            )
        else:
            asset_number_id = None
            logger.debug("No AssetNumber provided.")

        # --------------------------------------------------------------
        # 6. LOCATION (linked to Model, optionally asset-specific)
        # --------------------------------------------------------------
        if self.location_input.data:
            logger.debug(
                "New Location input provided: %s", self.location_input.data
            )
            new_location = Location(
                name=self.location_input.data,
                description="",
                model_id=model_id,
                # If an AssetNumber is chosen, we treat this as an asset-specific location
                asset_number_id=asset_number_id,
            )
            session.add(new_location)
            session.commit()
            location_id = new_location.id
            logger.debug(
                "Created new Location with ID: %s (Model ID: %s, AssetNumber ID: %s)",
                location_id,
                model_id,
                asset_number_id,
            )
        elif self.location.data:
            location_id = self.location.data.id
            logger.debug("Using selected Location with ID: %s", location_id)
        else:
            location_id = None
            logger.debug("No Location provided.")

        # --------------------------------------------------------------
        # 7. SUBASSEMBLY (linked to Location, optionally asset-specific)
        # --------------------------------------------------------------
        if self.subassembly_input.data:
            logger.debug(
                "New Subassembly input provided: %s",
                self.subassembly_input.data,
            )
            new_subassembly = Subassembly(
                name=self.subassembly_input.data,
                description="",
                location_id=location_id,
                # Asset-specific if an AssetNumber is selected
                asset_number_id=asset_number_id,
            )
            session.add(new_subassembly)
            session.commit()
            subassembly_id = new_subassembly.id
            logger.debug(
                "Created new Subassembly with ID: %s (Location ID: %s, AssetNumber ID: %s)",
                subassembly_id,
                location_id,
                asset_number_id,
            )
        elif self.subassembly.data:
            subassembly_id = self.subassembly.data.id
            logger.debug("Using selected Subassembly with ID: %s", subassembly_id)
        else:
            subassembly_id = None
            logger.debug("No Subassembly provided.")

        # --------------------------------------------------------------
        # 8. COMPONENT SUBASSEMBLY (linked to Subassembly, optionally asset-specific)
        # --------------------------------------------------------------
        if self.component_assembly_input.data:
            logger.debug(
                "New Component Subassembly input provided: %s",
                self.component_assembly_input.data,
            )
            new_component_assembly = ComponentAssembly(
                name=self.component_assembly_input.data,
                description="",
                subassembly_id=subassembly_id,
                # Asset-specific if an AssetNumber is selected
                asset_number_id=asset_number_id,
            )
            session.add(new_component_assembly)
            session.commit()
            component_assembly_id = new_component_assembly.id
            logger.debug(
                "Created new Component Subassembly with ID: %s "
                "(Subassembly ID: %s, AssetNumber ID: %s)",
                component_assembly_id,
                subassembly_id,
                asset_number_id,
            )
        elif self.component_assembly.data:
            component_assembly_id = self.component_assembly.data.id
            logger.debug(
                "Using selected Component Subassembly with ID: %s",
                component_assembly_id,
            )
        else:
            component_assembly_id = None
            logger.debug("No Component Subassembly provided.")

        # --------------------------------------------------------------
        # 9. SUBASSEMBLY VIEW (linked to Component Subassembly, optionally asset-specific)
        # --------------------------------------------------------------
        if self.assembly_view_input.data:
            if component_assembly_id is None:
                # We still enforce that a view cannot exist without a component assembly
                raise ValueError(
                    "A Component Subassembly is required to create a new Subassembly View."
                )

            logger.debug(
                "New Subassembly View input provided: %s",
                self.assembly_view_input.data,
            )
            new_assembly_view = AssemblyView(
                name=self.assembly_view_input.data,
                description="",
                component_assembly_id=component_assembly_id,
                # Asset-specific if an AssetNumber is selected
                asset_number_id=asset_number_id,
            )
            session.add(new_assembly_view)
            session.commit()
            assembly_view_id = new_assembly_view.id
            logger.debug(
                "Created new Subassembly View with ID: %s "
                "(Component Subassembly ID: %s, AssetNumber ID: %s)",
                assembly_view_id,
                component_assembly_id,
                asset_number_id,
            )
        elif self.assembly_view.data:
            assembly_view_id = self.assembly_view.data.id
            logger.debug(
                "Using selected Subassembly View with ID: %s", assembly_view_id
            )
        else:
            assembly_view_id = None
            logger.debug("No Subassembly View provided.")

        # --------------------------------------------------------------
        # 10. SITE LOCATION (find-or-create on title+room)
        # --------------------------------------------------------------
        if self.site_location_input.data:
            title = self.site_location_input.data
            room = self.site_location_room.data or ""
            logger.debug(
                "New Site Location input provided: title=%s room=%s", title, room
            )

            existing_site = (
                session.query(SiteLocation)
                .filter(
                    SiteLocation.title == title,
                    SiteLocation.room_number == room,
                )
                .first()
            )
            if existing_site:
                site_location_id = existing_site.id
                logger.debug(
                    "Using existing SiteLocation with ID: %s", site_location_id
                )
            else:
                new_site_location = SiteLocation(
                    title=title,
                    room_number=room,
                    site_area="",  # or self.site_area_input if added later
                    building_id=building_id,
                )
                session.add(new_site_location)
                session.commit()
                site_location_id = new_site_location.id
                logger.debug(
                    "Created new Site Location with ID: %s", site_location_id
                )
        elif self.site_location.data:
            site_location_id = self.site_location.data.id
            # OPTIONAL: ensure building/campus align with the site
            if building_id is None and self.site_location.data.building_id:
                building_id = self.site_location.data.building_id

            if campus_id is None and self.site_location.data.building:
                campus_id = self.site_location.data.building.campus_id
            logger.debug(
                "Using selected Site Location with ID: %s", site_location_id
            )
        else:
            site_location_id = None
            logger.debug("No Site Location provided.")

        logger.debug(
            (
                "Final IDs - Campus: %s, Building: %s, Area: %s, "
                "EquipmentGroup: %s, Model: %s, AssetNumber: %s, "
                "Location: %s, SiteLocation: %s, Subassembly: %s, "
                "ComponentSubassembly: %s, SubassemblyView: %s"
            ),
            campus_id,
            building_id,
            area_id,
            equipment_group_id,
            model_id,
            asset_number_id,
            location_id,
            site_location_id,
            subassembly_id,
            component_assembly_id,
            assembly_view_id,
        )

        # --------------------------------------------------------------
        # 11. Create / reuse Position using helper
        # --------------------------------------------------------------
        pos_id = cnp_form_create_position(
            campus_id=campus_id,
            building_id=building_id,
            area_id=area_id,
            equipment_group_id=equipment_group_id,
            model_id=model_id,
            asset_number_id=asset_number_id,
            location_id=location_id,
            site_location_id=site_location_id,
            subassembly_id=subassembly_id,
            component_assembly_id=component_assembly_id,
            assembly_view_id=assembly_view_id,
            session=session,
        )

        logger.debug("Created / reused Position with ID: %s", pos_id)
        logger.debug("=== Finished save() in CreatePositionForm ===")
        return pos_id

