from flask_wtf import FlaskForm
from wtforms import SubmitField
from wtforms.validators import Optional
from wtforms_sqlalchemy.fields import QuerySelectField
from modules.emtacdb.emtacdb_fts import (Area, EquipmentGroup, Model, AssetNumber, Location, Subassembly,
                                         ComponentAssembly, AssemblyView, SiteLocation)
from flask import current_app  # To access the app's config

class PositionForm(FlaskForm):
    area = QuerySelectField(
        label="Area",
        query_factory=lambda: current_app.config['db_config'].get_main_session().query(Area).order_by(Area.name).all(),
        allow_blank=True,
        blank_text="Select an Area",
        validators=[Optional()],
        get_label="name"
    )

    equipment_group = QuerySelectField(
        label="Equipment Group",
        query_factory=lambda: current_app.config['db_config'].get_main_session().query(EquipmentGroup).order_by(EquipmentGroup.name).all(),
        allow_blank=True,
        blank_text="Select an Equipment Group",
        validators=[Optional()],
        get_label="name"
    )

    model = QuerySelectField(
        label="Model",
        query_factory=lambda: current_app.config['db_config'].get_main_session().query(Model).order_by(Model.name).all(),
        allow_blank=True,
        blank_text="Select a Model",
        validators=[Optional()],
        get_label="name"
    )

    asset_number = QuerySelectField(
        label="Asset Number",
        query_factory=lambda: current_app.config['db_config'].get_main_session().query(AssetNumber).order_by(AssetNumber.number).all(),
        allow_blank=True,
        blank_text="Select an Asset Number",
        validators=[Optional()],
        get_label="number"
    )

    location = QuerySelectField(
        label="Location",
        query_factory=lambda: current_app.config['db_config'].get_main_session().query(Location).order_by(Location.name).all(),
        allow_blank=True,
        blank_text="Select a Location",
        validators=[Optional()],
        get_label="name"
    )

    subassembly = QuerySelectField(  # Kept for actual Subassembly
        label="Subassembly",
        query_factory=lambda: current_app.config['db_config'].get_main_session().query(Subassembly).order_by(Subassembly.name).all(),
        allow_blank=True,
        blank_text="Select a Subassembly",
        validators=[Optional()],
        get_label="name"
    )

    component_assembly = QuerySelectField(  # Renamed from `assembly`
        label="Component Assembly",
        query_factory=lambda: current_app.config['db_config'].get_main_session().query(ComponentAssembly).order_by(ComponentAssembly.name).all(),
        allow_blank=True,
        blank_text="Select a Component Assembly",
        validators=[Optional()],
        get_label="name"
    )

    assembly_view = QuerySelectField(
        label="Assembly View",
        query_factory=lambda: current_app.config['db_config'].get_main_session().query(AssemblyView).order_by(AssemblyView.name).all(),
        allow_blank=True,
        blank_text="Select an Assembly View",
        validators=[Optional()],
        get_label="name"
    )

    site_location = QuerySelectField(
        label="Site Location",
        query_factory=lambda: current_app.config['db_config']
        .get_main_session()
        .query(SiteLocation)
        .order_by(SiteLocation.title, SiteLocation.room_number)
        .all(),
        allow_blank=True,
        blank_text="Select a Site Location",
        validators=[Optional()],
        get_label=lambda site_location: f"{site_location.title} - Room {site_location.room_number}"
    )

    submit = SubmitField("Submit")
