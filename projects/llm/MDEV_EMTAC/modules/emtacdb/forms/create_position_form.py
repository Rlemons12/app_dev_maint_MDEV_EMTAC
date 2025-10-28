# modules/emtacdb/forms/create_position_form.py

from flask_wtf import FlaskForm
from wtforms import SubmitField, StringField, TextAreaField
from wtforms.validators import Optional, DataRequired
from wtforms_sqlalchemy.fields import QuerySelectField
from modules.emtacdb.emtacdb_fts import (
    Area, EquipmentGroup, Model, AssetNumber, Location,
    Subassembly, ComponentAssembly, AssemblyView, SiteLocation, Position
)
from modules.configuration.log_config import logger
from modules.configuration.config_env import DatabaseConfig

database_config = DatabaseConfig()

def cnp_form_create_position(area_id, equipment_group_id, model_id, asset_number_id,
                             location_id, site_location_id, subassembly_id,
                             component_assembly_id, assembly_view_id, session):
    try:
        logger.debug(
            "Entering cnp_form_create_position with IDs: Area=%s, EquipmentGroup=%s, Model=%s, AssetNumber=%s, "
            "Location=%s, SiteLocation=%s, Subassembly=%s, ComponentAssembly=%s, AssemblyView=%s",
            area_id, equipment_group_id, model_id, asset_number_id, location_id,
            site_location_id, subassembly_id, component_assembly_id, assembly_view_id
        )

        # Retrieve the related objects by their IDs (if provided)
        area_entity = session.query(Area).filter_by(id=area_id).first() if area_id else None
        equipment_group_entity = session.query(EquipmentGroup).filter_by(id=equipment_group_id).first() if equipment_group_id else None
        model_entity = session.query(Model).filter_by(id=model_id).first() if model_id else None
        asset_number_entity = session.query(AssetNumber).filter_by(id=asset_number_id).first() if asset_number_id else None
        location_entity = session.query(Location).filter_by(id=location_id).first() if location_id else None
        site_location_entity = session.query(SiteLocation).filter_by(id=site_location_id).first() if site_location_id else None
        subassembly_entity = session.query(Subassembly).filter_by(id=subassembly_id).first() if subassembly_id else None
        component_assembly_entity = session.query(ComponentAssembly).filter_by(id=component_assembly_id).first() if component_assembly_id else None
        assembly_view_entity = session.query(AssemblyView).filter_by(id=assembly_view_id).first() if assembly_view_id else None

        logger.debug(
            "Retrieved related objects: Area=%s, EquipmentGroup=%s, Model=%s, AssetNumber=%s, Location=%s, "
            "SiteLocation=%s, Subassembly=%s, ComponentAssembly=%s, AssemblyView=%s",
            area_entity, equipment_group_entity, model_entity, asset_number_entity,
            location_entity, site_location_entity, subassembly_entity, component_assembly_entity,
            assembly_view_entity
        )

        # Check for an existing Position with the same relationships.
        existing_position = session.query(Position).filter_by(
            area=area_entity,
            equipment_group=equipment_group_entity,
            model=model_entity,
            asset_number=asset_number_entity,
            location=location_entity,
            site_location=site_location_entity,
            subassembly=subassembly_entity,
            component_assembly=component_assembly_entity,
            assembly_view=assembly_view_entity
        ).first()

        if existing_position:
            logger.debug("Found existing Position with ID: %s", existing_position.id)
            return existing_position.id
        else:
            new_position = Position(
                area=area_entity,
                equipment_group=equipment_group_entity,
                model=model_entity,
                asset_number=asset_number_entity,
                location=location_entity,
                site_location=site_location_entity,
                subassembly=subassembly_entity,
                component_assembly=component_assembly_entity,
                assembly_view=assembly_view_entity
            )
            session.add(new_position)
            session.commit()
            logger.debug("Created new Position with ID: %s", new_position.id)
            return new_position.id

    except Exception as e:
        logger.error("Error in cnp_form_create_position: %s", str(e), exc_info=True)
        session.rollback()
        raise e



class CreatePositionForm(FlaskForm):
    # --- Define fields for each related table ---
    # AREA
    area = QuerySelectField(
        label="Area",
        query_factory=lambda: [],
        allow_blank=True,
        blank_text="Select an Area",
        validators=[Optional()],
        get_label="name",
        render_kw={"id": "areaDropdown", "data-toggle-input": "create_position_form-area_input"}
    )
    area_input = StringField(
        label="New Area",
        validators=[Optional()],
        render_kw={"id": "areaInput", "placeholder": "Enter new Area if not listed"}
    )

    # EQUIPMENT GROUP
    equipment_group = QuerySelectField(
        label="Equipment Group",
        query_factory=lambda: [],
        allow_blank=True,
        blank_text="Select an Equipment Group",
        validators=[Optional()],
        get_label="name",
        render_kw={"id": "equipmentGroupDropdown", "data-toggle-input": "create_position_form-equipment_group_input"}
    )
    equipment_group_input = StringField(
        label="New Equipment Group",
        validators=[Optional()],
        render_kw={"id": "equipmentGroupInput", "placeholder": "Enter new Equipment Group if not listed"}
    )

    # MODEL (with description)
    model = QuerySelectField(
        label="Model",
        query_factory=lambda: [],
        allow_blank=True,
        blank_text="Select a Model",
        validators=[Optional()],
        get_label="name",
        render_kw={"id": "modelDropdown", "data-toggle-input": "create_position_form-model_input"}
    )
    model_input = StringField(
        label="New Model",
        validators=[Optional()],
        render_kw={"id": "modelInput", "placeholder": "Enter new Model if not listed"}
    )
    model_description = TextAreaField(
        label="Model Description",
        validators=[Optional()],
        render_kw={"id": "modelDescription", "placeholder": "Enter description for new Model if not listed"}
    )

    # ASSET NUMBER
    asset_number = QuerySelectField(
        label="Asset Number",
        query_factory=lambda: [],
        allow_blank=True,
        blank_text="Select an Asset Number",
        validators=[Optional()],
        get_label=lambda asset: f"{asset.number} - {asset.description}" if asset.description else asset.number,
        render_kw={"id": "assetNumberDropdown", "data-toggle-input": "create_position_form-asset_number_input"}
    )
    asset_number_input = StringField(
        label="New Asset Number",
        validators=[Optional()],
        render_kw={"id": "assetNumberInput", "placeholder": "Enter new Asset Number if not listed"}
    )
    asset_number_description = TextAreaField(
        label="Asset Number Description",
        validators=[Optional()],
        render_kw={"id": "assetNumberDescription", "placeholder": "Enter description for new Asset Number if not listed"}
    )

    # LOCATION
    location = QuerySelectField(
        label="Location",
        query_factory=lambda: [],
        allow_blank=True,
        blank_text="Select a Location",
        validators=[Optional()],
        get_label="name",
        render_kw={"id": "locationDropdown", "data-toggle-input": "create_position_form-location_input"}
    )
    location_input = StringField(
        label="New Location",
        validators=[Optional()],
        render_kw={"id": "locationInput", "placeholder": "Enter new Location if not listed"}
    )

    # SUBASSEMBLY (formerly Assembly)
    subassembly = QuerySelectField(
        label="Subassembly",
        query_factory=lambda: [],
        allow_blank=True,
        blank_text="Select a Subassembly",
        validators=[Optional()],
        get_label="name",
        render_kw={"id": "assemblyDropdown", "data-toggle-input": "create_position_form-assembly_input"}
    )
    subassembly_input = StringField(
        label="New Subassembly",
        validators=[Optional()],
        render_kw={"id": "assemblyInput", "placeholder": "Enter new Subassembly if not listed"}
    )

    # COMPONENT SUBASSEMBLY (formerly Component Assembly)
    component_assembly = QuerySelectField(
        label="Component Subassembly",
        query_factory=lambda: [],
        allow_blank=True,
        blank_text="Select a Component Subassembly",
        validators=[Optional()],
        get_label="name",
        render_kw={"id": "componentAssemblyDropdown", "data-toggle-input": "create_position_form-component_assembly_input"}
    )
    component_assembly_input = StringField(
        label="New Component Subassembly",
        validators=[Optional()],
        render_kw={"id": "componentAssemblyInput", "placeholder": "Enter new Component Subassembly if not listed"}
    )

    # SUBASSEMBLY VIEW (formerly Assembly View)
    assembly_view = QuerySelectField(
        label="Subassembly View",
        query_factory=lambda: [],
        allow_blank=True,
        blank_text="Select a Subassembly View",
        validators=[Optional()],
        get_label="name",
        render_kw={"id": "assemblyViewDropdown", "data-toggle-input": "create_position_form-assembly_view_input"}
    )
    assembly_view_input = StringField(
        label="New Subassembly View",
        validators=[Optional()],
        render_kw={"id": "assemblyViewInput", "placeholder": "Enter new Subassembly View if not listed"}
    )

    # SITE LOCATION (custom: title and room number)
    site_location = QuerySelectField(
        label="Site Location",
        query_factory=lambda: [],
        allow_blank=True,
        blank_text="Select a Site Location",
        validators=[Optional()],
        get_label=lambda site_location: f"{site_location.title} - Room {site_location.room_number}",
        render_kw={"id": "siteLocationDropdown", "data-toggle-input": "create_position_form-site_location_input"}
    )
    site_location_input = StringField(
        label="New Site Location Title",
        validators=[Optional()],
        render_kw={"id": "siteLocationInput", "placeholder": "Enter new Site Location title if not listed"}
    )
    site_location_room = StringField(
        label="Room Number",
        validators=[Optional()],
        render_kw={"id": "siteLocationRoom", "placeholder": "Enter Room Number"}
    )

    submit = SubmitField("Create Position")

    # Custom Validators: Ensure only one of (existing OR new) is provided per field.
    def validate(self, *args, **kwargs):
        if not super(CreatePositionForm, self).validate(*args, **kwargs):
            return False
        success = True
        field_pairs = [
            (self.area, self.area_input, "Area"),
            (self.equipment_group, self.equipment_group_input, "Equipment Group"),
            (self.model, self.model_input, "Model"),
            (self.asset_number, self.asset_number_input, "Asset Number"),
            (self.location, self.location_input, "Location"),
            (self.subassembly, self.subassembly_input, "Subassembly"),
            (self.component_assembly, self.component_assembly_input, "Component Subassembly"),
            (self.assembly_view, self.assembly_view_input, "Subassembly View"),
            (self.site_location, self.site_location_input, "Site Location"),
        ]
        for select_field, input_field, field_name in field_pairs:
            if select_field.data and input_field.data:
                msg = f"Please provide either a selected {field_name} or a new {field_name}, not both."
                select_field.errors.append(msg)
                input_field.errors.append(msg)
                success = False
        return success

    def set_query_factories(self, session):
        self.area.query_factory = lambda: session.query(Area).order_by(Area.name).all()
        self.equipment_group.query_factory = lambda: session.query(EquipmentGroup).order_by(EquipmentGroup.name).all()
        self.model.query_factory = lambda: session.query(Model).order_by(Model.name).all()
        self.asset_number.query_factory = lambda: session.query(AssetNumber).order_by(AssetNumber.number).all()
        self.location.query_factory = lambda: session.query(Location).order_by(Location.name).all()
        self.subassembly.query_factory = lambda: session.query(Subassembly).order_by(Subassembly.name).all()
        self.component_assembly.query_factory = lambda: session.query(ComponentAssembly).order_by(ComponentAssembly.name).all()
        self.assembly_view.query_factory = lambda: session.query(AssemblyView).order_by(AssemblyView.name).all()
        self.site_location.query_factory = lambda: session.query(SiteLocation).order_by(SiteLocation.title, SiteLocation.room_number).all()

    def save(self, session, cnp_form_create_position):
        logger.debug("=== Starting save() in CreatePositionForm ===")

        # 1. Process AREA
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

        # 2. Process EQUIPMENT GROUP (linked to Area via area_id)
        if self.equipment_group_input.data:
            logger.debug("New Equipment Group input provided: %s", self.equipment_group_input.data)
            new_eq_group = EquipmentGroup(
                name=self.equipment_group_input.data,
                area_id=area_id
            )
            session.add(new_eq_group)
            session.commit()
            equipment_group_id = new_eq_group.id
            logger.debug("Created new EquipmentGroup with ID: %s (Area ID: %s)", equipment_group_id, area_id)
        elif self.equipment_group.data:
            equipment_group_id = self.equipment_group.data.id
            logger.debug("Using selected EquipmentGroup with ID: %s", equipment_group_id)
        else:
            equipment_group_id = None
            logger.debug("No EquipmentGroup provided.")

        # 3. Process MODEL (optionally linking to EquipmentGroup)
        if self.model_input.data:
            logger.debug("New Model input provided: %s", self.model_input.data)
            new_model = Model(
                name=self.model_input.data,
                description=self.model_description.data or "",
                equipment_group_id=equipment_group_id
            )
            session.add(new_model)
            session.commit()
            model_id = new_model.id
            logger.debug("Created new Model with ID: %s (EquipmentGroup ID: %s)", model_id, equipment_group_id)
        elif self.model.data:
            model_id = self.model.data.id
            logger.debug("Using selected Model with ID: %s", model_id)
        else:
            model_id = None
            logger.debug("No Model provided.")

        # 4. Process ASSET NUMBER
        if self.asset_number_input.data:
            logger.debug("New Asset Number input provided: %s", self.asset_number_input.data)
            new_asset_number = AssetNumber(
                number=self.asset_number_input.data,
                description=self.asset_number_description.data or "",
                model_id=model_id  # Link AssetNumber to Model
            )
            session.add(new_asset_number)
            session.commit()
            asset_number_id = new_asset_number.id
            logger.debug("Created new AssetNumber with ID: %s (linked to Model ID: %s)", asset_number_id, model_id)
        elif self.asset_number.data:
            asset_number_id = self.asset_number.data.id
            logger.debug("Using selected AssetNumber with ID: %s", asset_number_id)
        else:
            asset_number_id = None
            logger.debug("No AssetNumber provided.")

        # 5. Process LOCATION (optionally linking to Model)
        if self.location_input.data:
            logger.debug("New Location input provided: %s", self.location_input.data)
            new_location = Location(
                name=self.location_input.data,
                description="",
                model_id=model_id
            )
            session.add(new_location)
            session.commit()
            location_id = new_location.id
            logger.debug("Created new Location with ID: %s (linked to Model ID: %s)", location_id, model_id)
        elif self.location.data:
            location_id = self.location.data.id
            logger.debug("Using selected Location with ID: %s", location_id)
        else:
            location_id = None
            logger.debug("No Location provided.")

        # 6. Process SUBASSEMBLY (linking to Location)
        if self.subassembly_input.data:
            logger.debug("New Subassembly input provided: %s", self.subassembly_input.data)
            new_subassembly = Subassembly(
                name=self.subassembly_input.data,
                description="",
                location_id=location_id
            )
            session.add(new_subassembly)
            session.commit()
            subassembly_id = new_subassembly.id
            logger.debug("Created new Subassembly with ID: %s (linked to Location ID: %s)", subassembly_id, location_id)
        elif self.subassembly.data:
            subassembly_id = self.subassembly.data.id
            logger.debug("Using selected Subassembly with ID: %s", subassembly_id)
        else:
            subassembly_id = None
            logger.debug("No Subassembly provided.")

        # 7. Process COMPONENT SUBASSEMBLY (linking to Subassembly)
        if self.component_assembly_input.data:
            logger.debug("New Component Subassembly input provided: %s", self.component_assembly_input.data)
            new_component_assembly = ComponentAssembly(
                name=self.component_assembly_input.data,
                description="",
                subassembly_id=subassembly_id  # Link to Subassembly via subassembly_id
            )
            session.add(new_component_assembly)
            session.commit()
            component_assembly_id = new_component_assembly.id
            logger.debug("Created new Component Subassembly with ID: %s (linked to Subassembly ID: %s)", component_assembly_id, subassembly_id)
        elif self.component_assembly.data:
            component_assembly_id = self.component_assembly.data.id
            logger.debug("Using selected Component Subassembly with ID: %s", component_assembly_id)
        else:
            component_assembly_id = None
            logger.debug("No Component Subassembly provided.")

        # 8. Process SUBASSEMBLY VIEW (linking to Component Subassembly)
        if self.assembly_view_input.data:
            if component_assembly_id is None:
                raise ValueError("A Component Subassembly is required to create a new Subassembly View.")
            logger.debug("New Subassembly View input provided: %s", self.assembly_view_input.data)
            new_assembly_view = AssemblyView(
                name=self.assembly_view_input.data,
                description="",
                component_assembly_id=component_assembly_id
            )
            session.add(new_assembly_view)
            session.commit()
            assembly_view_id = new_assembly_view.id
            logger.debug("Created new Subassembly View with ID: %s (linked to Component Subassembly ID: %s)", assembly_view_id, component_assembly_id)
        elif self.assembly_view.data:
            assembly_view_id = self.assembly_view.data.id
            logger.debug("Using selected Subassembly View with ID: %s", assembly_view_id)
        else:
            assembly_view_id = None
            logger.debug("No Subassembly View provided.")

        # 9. Process SITE LOCATION
        if self.site_location_input.data:
            logger.debug("New Site Location input provided: %s", self.site_location_input.data)
            new_site_location = SiteLocation(
                title=self.site_location_input.data,
                room_number=self.site_location_room.data or ""
            )
            session.add(new_site_location)
            session.commit()
            site_location_id = new_site_location.id
            logger.debug("Created new Site Location with ID: %s", site_location_id)
        elif self.site_location.data:
            site_location_id = self.site_location.data.id
            logger.debug("Using selected Site Location with ID: %s", site_location_id)
        else:
            site_location_id = None
            logger.debug("No Site Location provided.")

        logger.debug(
            "Final IDs - Area: %s, EquipmentGroup: %s, Model: %s, AssetNumber: %s, Location: %s, SiteLocation: %s, Subassembly: %s, ComponentSubassembly: %s, SubassemblyView: %s",
            area_id, equipment_group_id, model_id, asset_number_id, location_id, site_location_id, subassembly_id,
            component_assembly_id, assembly_view_id
        )

        # Call the helper function to create the Position record.
        pos_id = cnp_form_create_position(
            area_id=area_id,
            equipment_group_id=equipment_group_id,
            model_id=model_id,
            asset_number_id=asset_number_id,
            location_id=location_id,
            site_location_id=site_location_id,
            subassembly_id=subassembly_id,           # Updated parameter name
            component_assembly_id=component_assembly_id,
            assembly_view_id=assembly_view_id,
            session=session
        )
        logger.debug("Created Position with ID: %s", pos_id)
        logger.debug("=== Finished save() in CreatePositionForm ===")
        return pos_id





