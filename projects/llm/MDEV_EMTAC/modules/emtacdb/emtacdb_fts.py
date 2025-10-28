
from sqlalchemy.dialects.postgresql import TSVECTOR, JSONB
from pgvector.sqlalchemy import Vector
from dataclasses import dataclass
from sqlalchemy.dialects.postgresql import UUID
from pgvector.sqlalchemy import Vector
from sqlalchemy import Index,func
from typing import List, Dict, Any, Tuple, Optional
import time
import threading
from sqlalchemy.dialects.postgresql import UUID
import uuid
import shutil
from datetime import datetime
import pythoncom
from docx2pdf import convert  # or whatever conversion library you're using
import re
import pandas as pd
import fitz  # PyMuPDF
from fuzzywuzzy import fuzz
import unicodedata
import tempfile
import json
from sqlalchemy import or_, and_
from sqlalchemy.orm import relationship, joinedload
from typing import Dict, Any, List, Optional, Tuple, Union
from flask import send_file, jsonify, request, abort, flash, redirect, url_for, render_template
from sqlalchemy import desc, asc
from werkzeug.utils import secure_filename
import mimetypes
from PIL import Image as PILImage
from sqlalchemy import (DateTime, Column, ForeignKey, Integer, JSON, LargeBinary,
                       Enum, Boolean, String, create_engine, text, Float,
                       Text, UniqueConstraint, and_, Table)

import openai
import spacy
from fuzzywuzzy import process
from werkzeug.security import check_password_hash, generate_password_hash
from sqlalchemy.dialects.postgresql import ARRAY, JSON
from pgvector.sqlalchemy import Vector
from sqlalchemy import (DateTime, Column, ForeignKey, Integer, JSON, LargeBinary, Enum as SqlEnum, Boolean,
                        String, create_engine, text, Float, Text, UniqueConstraint, and_, Table)
from enum import Enum as PyEnum  # Import Enum and alias it as PyEnum
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.ext.mutable import MutableList
from sqlalchemy.orm import declarative_base, configure_mappers, relationship, scoped_session, sessionmaker
from modules.configuration.config import (OPENAI_API_KEY, BASE_DIR, COPY_FILES, DATABASE_URL, DATABASE_PATH,
                                          DATABASE_PATH_IMAGES_FOLDER, CURRENT_EMBEDDING_MODEL, DATABASE_DOC,
                                          DATABASE_DIR, TEMPORARY_UPLOAD_FILES)
from modules.configuration.base import Base
from modules.configuration.log_config import *
from modules.configuration.config_env import DatabaseConfig
from flask import g  # Required for access to g.request_id in the methods
from functools import wraps  # Required if you need to recreate with_request_id
from modules.emtacdb.utlity.system_manager import SystemResourceManager
from plugins import generate_embedding, CLIPModelHandler
from plugins.ai_modules import ModelsConfig
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np

# Configure mappers (must be called after all ORM classes are defined)
configure_mappers()

# Load the English language model
nlp = spacy.load('en_core_web_sm')

# Set your OpenAI API key
openai.api_key = OPENAI_API_KEY

# Constants for chunk size and model name
CHUNK_SIZE = 8000
MODEL_NAME = "text-embedding-ada-002"


# Update your engine configuration
engine = create_engine(
    DATABASE_URL,
    pool_size=10,
    max_overflow=20,
    pool_pre_ping=True,  # Good for PostgreSQL
    # Remove SQLite-specific connect_args
)

Session = scoped_session(sessionmaker(bind=engine))
session = Session


# Revision control database configuration
"""REVISION_CONTROL_DB_PATH = os.path.join(DATABASE_DIR, 'emtac_revision_control_db.db')
revision_control_engine = create_engine(
    f'sqlite:///{REVISION_CONTROL_DB_PATH}',
    pool_size=10,            # Set a small pool size
    max_overflow=20,         # Allow up to 10 additional connections
    connect_args={"check_same_thread": False}  # Needed for SQLite when using threading
)

RevisionControlBase = declarative_base()  # This is correctly defined
RevisionControlSession = scoped_session(sessionmaker(bind=revision_control_engine))  # Use distinct name
revision_control_session = RevisionControlSession()"""

class VersionInfo(Base):
    __tablename__ = 'version_info'
    __table_args__ = {'extend_existing': True}
    id = Column(Integer, primary_key=True, autoincrement=True)
    version_number = Column(Integer, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    description = Column(String, nullable=True)

"""
ACADEMIC CONTENT MAPPING SYSTEM
===============================

This module repurposes our equipment management database schema to create an academic content 
organization system. The hierarchical nature of our equipment schema maps perfectly to the 
hierarchical organization of academic knowledge.

MAPPING OVERVIEW:
----------------
Our equipment hierarchy tables are mapped to academic concepts as follows:

1. Area → Academic Field
   Represents broad academic disciplines like Physics, Mathematics, Chemistry, etc.
   Example: "Physics" as an Area

2. EquipmentGroup → Subject
   Represents major subjects within an academic field.
   Example: "Mechanics" as an EquipmentGroup within "Physics" Area

3. Model → Branch/Subdiscipline
   Represents specialized branches or subdisciplines within a subject.
   Example: "Classical Mechanics" as a Model within "Mechanics" EquipmentGroup

4. AssetNumber → Specific Book/Resource
   Represents individual academic resources like textbooks or reference materials.
   Example: "Feynman Lectures Vol. 1" as an AssetNumber within "Classical Mechanics" Model

5. Location → Chapter
   Represents main divisions within a book or resource.
   Example: "Chapter 1: Atoms in Motion" as a Location within a book

6. Subassembly → Section
   Represents major sections within a chapter.
   Example: "1.1 Introduction to Atomic Theory" as a Subassembly within Chapter 1

7. ComponentAssembly → Subsection/Topic
   Represents specific topics or subsections within a section.
   Example: "1.1.1 Dalton's Atomic Theory" as a ComponentAssembly within Section 1.1

8. AssemblyView → Specific Concept/Figure/Example
   Represents individual concepts, illustrations, or examples within a topic.
   Example: "Figure 1: Dalton's Atomic Symbols" as an AssemblyView within Topic 1.1.1

INTEGRATION WITH POSITION TABLE:
------------------------------
The Position table serves as the central connection point, linking entities across all levels
of the academic hierarchy. This enables navigation through the knowledge structure and allows
for querying relationships between academic concepts at different levels.

USAGE EXAMPLES:
--------------
1. Creating a physics textbook with chapters and sections:
   - Create an Area for "Physics"
   - Create an EquipmentGroup for "Mechanics" within Physics
   - Create a Model for "Classical Mechanics" within Mechanics
   - Create an AssetNumber for "Principles of Physics" textbook
   - Create Locations for each chapter
   - Create Subassemblies for sections within chapters
   - Create ComponentAssemblies for subsections
   - Create AssemblyViews for specific examples or figures
   - Use Position to connect all these entities together

2. Finding all chapters in a specific book:
   - Use Position.get_dependent_items(session, 'model', model_id, child_type='location')

3. Finding all topics within a specific chapter section:
   - Use Subassembly.find_related_entities(session, section_id).get('downward', {}).get('component_assemblies', [])

BENEFITS:
--------
- Reuses existing database schema and navigation logic
- Maintains consistent hierarchical organization
- Allows for flexible academic content modeling
- Supports complex queries across the knowledge hierarchy
- Integrates with existing application infrastructure

NOTE:
----
While we're repurposing equipment tables for academic content, the underlying logic
of hierarchical navigation remains the same. This approach allows us to leverage our
existing codebase while expanding its functionality to new domains.
"""
# Main Tables
class SiteLocation(Base):
    __tablename__ = 'site_location'
    id = Column(Integer, primary_key=True)
    title = Column(String, nullable=False)
    room_number = Column(String, nullable=False)
    site_area = Column(String, nullable=False)
    
    position = relationship('Position', back_populates="site_location")

    @classmethod
    @with_request_id
    def add_site_location(cls, session, title, room_number, site_area, request_id=None):
        """
        Add a new site location to the database.

        Args:
            session: SQLAlchemy database session
            title (str): Title of the site location
            room_number (str): Room number of the site location
            site_area (str): Site area of the site location
            request_id (str, optional): Unique identifier for the request

        Returns:
            SiteLocation: The newly created site location object
        """
        new_site_location = cls(
            title=title,
            room_number=room_number,
            site_area=site_area
        )

        session.add(new_site_location)
        session.commit()

        logger.info(f"Created new site location: '{title}' in room {room_number}, area {site_area}")
        return new_site_location

    @classmethod
    @with_request_id
    def delete_site_location(cls, session, site_location_id, request_id=None):
        """
        Delete a site location from the database.

        Args:
            session: SQLAlchemy database session
            site_location_id (int): ID of the site location to delete
            request_id (str, optional): Unique identifier for the request

        Returns:
            bool: True if deletion was successful, False if site location not found
        """
        site_location = session.query(cls).filter(cls.id == site_location_id).first()

        if site_location:
            session.delete(site_location)
            session.commit()
            logger.info(f"Deleted site location ID {site_location_id}")
            return True
        else:
            logger.warning(f"Failed to delete site location ID {site_location_id} - not found")
            return False

    @classmethod
    @with_request_id
    def find_related_entities(cls, session, identifier, is_id=True, request_id=None):
        """
        Find all related entities for a site location.

        Args:
            session: SQLAlchemy database session
            identifier: Either site location ID (int) or title (str)
            is_id (bool): If True, identifier is an ID, otherwise it's a title
            request_id (str, optional): Unique identifier for the request

        Returns:
            dict: Dictionary containing:
                - 'site_location': The found site location object
                - 'downward': Dictionary containing:
                    - 'positions': List of all positions at this site location
        """
        # Find the site location
        if is_id:
            site_location = session.query(cls).filter(cls.id == identifier).first()
        else:
            site_location = session.query(cls).filter(cls.title == identifier).first()

        if not site_location:
            logger.warning(f"Site location not found for identifier: {identifier}")
            return None

        # Going downward in the hierarchy
        downward = {
            'positions': site_location.position
        }

        logger.info(f"Found related entities for site location ID {site_location.id}")
        return {
            'site_location': site_location,
            'downward': downward
        }

    @classmethod
    @with_request_id
    def find_or_create(cls, session, title, room_number="Unknown", site_area="General", request_id=None):
        """
        Find a SiteLocation by title, or create it if it doesn't exist.

        Args:
            session: SQLAlchemy database session
            title (str): Title of the site location
            room_number (str): Room number (default "Unknown")
            site_area (str): Site area (default "General")
            request_id (str, optional): Unique identifier for the request

        Returns:
            SiteLocation: The found or newly created site location object
        """
        site_location = session.query(cls).filter_by(title=title).first()

        if site_location:
            logger.info(f"Found existing site location '{title}'", extra={'request_id': request_id})
        else:
            site_location = cls(
                title=title,
                room_number=room_number,
                site_area=site_area
            )
            session.add(site_location)
            session.commit()
            logger.info(f"Created new site location '{title}' with room '{room_number}' and area '{site_area}'",
                        extra={'request_id': request_id})

        return site_location

class Position(Base):
    __tablename__ = 'position'
    id = Column(Integer, primary_key=True)
    area_id = Column(Integer, ForeignKey('area.id'), nullable=True)
    equipment_group_id = Column(Integer, ForeignKey('equipment_group.id'), nullable=True)
    model_id = Column(Integer, ForeignKey('model.id'), nullable=True)
    asset_number_id = Column(Integer, ForeignKey('asset_number.id'), nullable=True)
    location_id = Column(Integer, ForeignKey('location.id'), nullable=True)
    subassembly_id = Column(Integer, ForeignKey('subassembly.id'), nullable=True)
    component_assembly_id = Column(Integer, ForeignKey('component_assembly.id'), nullable=True)
    assembly_view_id = Column(Integer, ForeignKey('assembly_view.id'), nullable=True)
    site_location_id = Column(Integer, ForeignKey('site_location.id'), nullable=True)

    area = relationship("Area", back_populates="position")
    equipment_group = relationship("EquipmentGroup", back_populates="position")
    model = relationship("Model", back_populates="position")
    asset_number = relationship("AssetNumber", back_populates="position")
    location = relationship("Location", back_populates="position")
    """bill_of_material = relationship("BillOfMaterial", back_populates="position")"""
    part_position_image = relationship("PartsPositionImageAssociation", back_populates="position")
    image_position_association = relationship("ImagePositionAssociation", back_populates="position")
    drawing_position = relationship("DrawingPositionAssociation", back_populates="position")
    problem_position = relationship("ProblemPositionAssociation", back_populates="position")
    completed_document_position_association = relationship("CompletedDocumentPositionAssociation", back_populates="position")
    site_location = relationship("SiteLocation", back_populates="position")
    position_tasks = relationship("TaskPositionAssociation", back_populates="position", cascade="all, delete-orphan")
    tool_position_association = relationship("ToolPositionAssociation", back_populates="position")
    subassembly = relationship("Subassembly", back_populates="position")
    component_assembly = relationship("ComponentAssembly", back_populates="position")
    assembly_view = relationship("AssemblyView", back_populates="position")

    # Hierarchy definition
    # Define HIERARCHY using string names instead of direct class references
    HIERARCHY = {
        'area': {
            'model': 'EquipmentGroup',
            'filter_field': 'area_id',
            'order_field': 'name',
            'next_level': 'equipment_group'
        },
        'equipment_group': {
            'model': 'Model',
            'filter_field': 'equipment_group_id',
            'order_field': 'name',
            'next_level': 'model'
        },
        'model': {
            # Models have two potential child types - asset_number and location
            'child_types': [
                {
                    'model': 'AssetNumber',
                    'filter_field': 'model_id',
                    'order_field': 'number',
                    'next_level': 'asset_number'
                },
                {
                    'model': 'Location',
                    'filter_field': 'model_id',
                    'order_field': 'name',
                    'next_level': 'location'
                }
            ]
        },
        'location': {
            'model': 'Subassembly',
            'filter_field': 'location_id',
            'order_field': 'name',
            'next_level': 'subassembly'
        },
        'subassembly': {
            'model': 'ComponentAssembly',
            'filter_field': 'subassembly_id',
            'order_field': 'name',
            'next_level': 'component_assembly'
        },
        'component_assembly': {
            'model': 'AssemblyView',
            'filter_field': 'component_assembly_id',
            'order_field': 'name',
            'next_level': 'assembly_view'
        }
    }

    # Model mapping - defined once for efficiency
    MODELS_MAP = None

    @classmethod
    @with_request_id
    def get_dependent_items(cls, session, parent_type, parent_id, child_type=None):
        """
        Generic method to get dependent items based on parent type and ID.

        Args:
            session: SQLAlchemy session
            parent_type: The type of the parent (e.g., 'area', 'equipment_group')
            parent_id: The ID of the parent
            child_type: Optional, to specify which child type to return when parent has multiple child types

        Returns:
            List of dependent items
        """
        if not parent_id:
            return []

        # Get parent configuration from hierarchy
        parent_config = cls.HIERARCHY.get(parent_type)
        if not parent_config:
            return []

        # Handle parents with multiple child types
        if 'child_types' in parent_config:
            if child_type:
                # Find the specific child type configuration
                for child_config in parent_config['child_types']:
                    if child_config.get('next_level') == child_type:
                        return cls._fetch_dependent_items(session, child_config, parent_id)
                return []
            else:
                # Return the first child type by default
                return cls._fetch_dependent_items(session, parent_config['child_types'][0], parent_id)
        else:
            # Standard single child type
            return cls._fetch_dependent_items(session, parent_config, parent_id)

    @staticmethod
    def _fetch_dependent_items(session, config, parent_id):
        """
        Helper method to fetch dependent items based on configuration.

        Args:
            session: SQLAlchemy session
            config: Configuration dictionary with model, filter_field, order_field
            parent_id: The ID of the parent

        Returns:
            List of dependent items
        """
        model_name = config.get('model')
        filter_field = config.get('filter_field')
        order_field = config.get('order_field')

        if not all([model_name, filter_field, order_field]):
            return []

        # Get the actual model class from its name
        if isinstance(model_name, str):
            # Use globals() to find the class by name
            model = globals().get(model_name)
            if not model:
                # Alternative approach - if globals() doesn't work, you can use a mapping
                models_map = {
                    'EquipmentGroup': EquipmentGroup,
                    'Model': Model,
                    'AssetNumber': AssetNumber,
                    'Location': Location,
                    'Subassembly': Subassembly,
                    'ComponentAssembly': ComponentAssembly,
                    'AssemblyView': AssemblyView,
                    'SiteLocation': SiteLocation
                }
                model = models_map.get(model_name)
                if not model:
                    return []
        else:
            model = model_name  # Already a class

        query = session.query(model).filter_by(**{filter_field: parent_id})

        # Apply ordering
        order_attr = getattr(model, order_field)
        query = query.order_by(order_attr)

        return query.all()

    @classmethod
    @with_request_id
    def get_next_level_type(cls, current_level):
        """Get the next level type in the hierarchy"""
        config = cls.HIERARCHY.get(current_level)
        if not config:
            return None

        if 'child_types' in config:
            # Return the first child type by default
            return config['child_types'][0].get('next_level')
        else:
            return config.get('next_level')

    @classmethod
    @with_request_id
    def add_to_db(cls, session=None, area_id=None, equipment_group_id=None, model_id=None, asset_number_id=None,
                  location_id=None, subassembly_id=None, component_assembly_id=None, assembly_view_id=None,
                  site_location_id=None):
        """
        Get-or-create a Position with exactly these FK values.
        If `session` is None, uses DatabaseConfig().get_main_session().
        Returns the Position ID (integer) of the new or existing position.
        """
        # 1) ensure we have a session
        if session is None:
            session = DatabaseConfig().get_main_session()

        # 2) log input parameters - FIXED
        debug_id(
            f"add_to_db called with area_id={area_id}, equipment_group_id={equipment_group_id}, "
            f"model_id={model_id}, asset_number_id={asset_number_id}, location_id={location_id}, "
            f"subassembly_id={subassembly_id}, component_assembly_id={component_assembly_id}, "
            f"assembly_view_id={assembly_view_id}, site_location_id={site_location_id}"
        )

        # 3) build filter dict
        filters = {
            "area_id": area_id,
            "equipment_group_id": equipment_group_id,
            "model_id": model_id,
            "asset_number_id": asset_number_id,
            "location_id": location_id,
            "subassembly_id": subassembly_id,
            "component_assembly_id": component_assembly_id,
            "assembly_view_id": assembly_view_id,
            "site_location_id": site_location_id,
        }

        try:
            # 4) try to find an existing row
            existing = session.query(cls).filter_by(**filters).first()
            if existing:
                info_id("Found existing Position id=%s", existing.id)
                return existing.id

            # 5) not found → create new
            position = cls(**filters)
            session.add(position)
            session.commit()
            info_id("Created new Position id=%s", position.id)
            return position.id

        except SQLAlchemyError as e:
            session.rollback()
            error_id("Failed to add_or_get Position: %s", e, exc_info=True)
            raise

    @classmethod
    @with_request_id
    def get_corresponding_position_ids(cls, session=None, area_id=None, equipment_group_id=None,
                                       model_id=None, asset_number_id=None, location_id=None,
                                       request_id='no_request_id'):
        """
        Search for corresponding Position IDs based on the provided filters with request ID logging.

        Args:
            session: SQLAlchemy session (Optional)
            area_id: ID of the area (optional)
            equipment_group_id: ID of the equipment group (optional)
            model_id: ID of the model (optional)
            asset_number_id: ID of the asset number (optional)
            location_id: ID of the location (optional)
            request_id: Unique identifier for the request

        Returns:
            List of Position IDs that match the criteria
        """
        # Ensure a session is available, if not use DatabaseConfig to get it
        if session is None:
            session = DatabaseConfig().get_main_session()

        # Log input parameters with request ID
        logging.info(
            f"[{request_id}] get_corresponding_position_ids called with "
            f"area_id={area_id}, equipment_group_id={equipment_group_id}, "
            f"model_id={model_id}, asset_number_id={asset_number_id}, "
            f"location_id={location_id}"
        )

        try:
            # Start by fetching the root-level positions based on hierarchy
            positions = cls._get_positions_by_hierarchy(
                session,
                area_id=area_id,
                equipment_group_id=equipment_group_id,
                model_id=model_id,
                asset_number_id=asset_number_id,
                location_id=location_id,
                request_id=request_id
            )

            # Extract Position IDs
            position_ids = [position.id for position in positions]

            # Log the result
            logging.info(f"[{request_id}] Retrieved {len(position_ids)} Position IDs")
            return position_ids

        except SQLAlchemyError as e:
            # Log any errors encountered during the query
            logging.error(
                f"[{request_id}] Error in get_corresponding_position_ids: {str(e)}",
                exc_info=True
            )
            raise

    @classmethod
    @with_request_id
    def _get_positions_by_hierarchy(cls, session, area_id=None, equipment_group_id=None, model_id=None,
                                    asset_number_id=None, location_id=None):
        """
        Helper method to fetch positions based on hierarchical filters.

        Args:
            session: SQLAlchemy session
            area_id, equipment_group_id, model_id, asset_number_id, location_id: IDs for filtering

        Returns:
            List of Position objects that match the criteria
        """
        # Building the filter dynamically based on input parameters
        filters = {}
        if area_id:
            filters['area_id'] = area_id
        if equipment_group_id:
            filters['equipment_group_id'] = equipment_group_id
        if model_id:
            filters['model_id'] = model_id
        if asset_number_id:
            filters['asset_number_id'] = asset_number_id
        if location_id:
            filters['location_id'] = location_id

        # Log the filter parameters
        debug_id(f"Filtering Positions with filters: {filters}", request_id=g.request_id)

        try:
            # Query the Position table based on the filters
            query = session.query(Position).filter_by(**filters)

            # Log the query execution
            info_id(f"Executing query for positions with {len(filters)} filters.", request_id=g.request_id)

            # Return the positions matching the filter
            positions = query.all()

            # Log the result
            info_id(f"Retrieved {len(positions)} positions.", request_id=g.request_id)
            return positions

        except SQLAlchemyError as e:
            # Log any errors encountered during the query
            error_id(f"Error in _get_positions_by_hierarchy: {str(e)}", exc_info=True, request_id=g.request_id)
            raise

class Area(Base):
    __tablename__ = 'area'

    id = Column(Integer, primary_key=True)
    name = Column(String, nullable=False)
    description = Column(String)

    equipment_group = relationship("EquipmentGroup", back_populates="area")
    position = relationship("Position", back_populates="area")

    @classmethod
    @with_request_id
    def add(cls, session: Session, name: str, description: str = None, logger=None):
        """
        Add a new Area to the database.
        Returns the created Area instance, or None if failed.
        """
        try:
            area = cls(name=name, description=description)
            session.add(area)
            session.commit()
            if logger:
                logger.info(f"Added Area: {name}")
            return area
        except SQLAlchemyError as e:
            session.rollback()
            if logger:
                logger.error(f"Failed to add Area: {e}")
            return None

    @classmethod
    @with_request_id
    def delete(cls, session: Session, area_id: int, logger=None):
        """
        Delete an Area by ID.
        Returns True if deleted, False if not found or failed.
        """
        try:
            area = session.query(cls).get(area_id)
            if area:
                session.delete(area)
                session.commit()
                if logger:
                    logger.info(f"Deleted Area id={area_id}")
                return True
            else:
                if logger:
                    logger.warning(f"Area id={area_id} not found for deletion")
                return False
        except SQLAlchemyError as e:
            session.rollback()
            if logger:
                logger.error(f"Failed to delete Area id={area_id}: {e}")
            return False

    @classmethod
    @with_request_id
    def search(cls, session: Session, name: str = None, description: str = None):
        """
        Search for Areas by name and/or description.
        Returns a list of Area instances matching the criteria.
        """
        query = session.query(cls)
        if name:
            query = query.filter(cls.name.ilike(f"%{name}%"))
        if description:
            query = query.filter(cls.description.ilike(f"%{description}%"))
        return query.all()

class EquipmentGroup(Base):
    __tablename__ = 'equipment_group'

    id = Column(Integer, primary_key=True)
    name = Column(String, nullable=False)
    area_id = Column(Integer, ForeignKey('area.id'))
    description = Column(String,nullable=True)
    
    area = relationship("Area", back_populates="equipment_group") 
    model = relationship("Model", back_populates="equipment_group")
    position = relationship("Position", back_populates="equipment_group")

    @classmethod
    @with_request_id
    def add_equipment_group(cls, session, name, area_id, description=None, request_id=None):
        """
        Add a new equipment group to the database.

        Args:
            session: SQLAlchemy database session
            name (str): Name of the equipment group
            area_id (int): ID of the area this equipment group belongs to
            description (str, optional): Description of the equipment group
            request_id (str, optional): Unique identifier for the request

        Returns:
            EquipmentGroup: The newly created equipment group object
        """
        new_equipment_group = cls(
            name=name,
            area_id=area_id,
            description=description
        )

        session.add(new_equipment_group)
        session.commit()

        return new_equipment_group

    @classmethod
    @with_request_id
    def delete_equipment_group(cls, session, equipment_group_id, request_id=None):
        """
        Delete an equipment group from the database.

        Args:
            session: SQLAlchemy database session
            equipment_group_id (int): ID of the equipment group to delete
            request_id (str, optional): Unique identifier for the request

        Returns:
            bool: True if deletion was successful, False if equipment group not found
        """
        equipment_group = session.query(cls).filter(cls.id == equipment_group_id).first()

        if equipment_group:
            session.delete(equipment_group)
            session.commit()
            return True
        else:
            return False

    @classmethod
    @with_request_id
    def find_related_entities(cls, session, identifier, is_id=True, request_id=None):
        """
        Find all related entities for an equipment group, traversing both up and down
        the hierarchy: Area → EquipmentGroup → Model → (AssetNumber, Location, Position).

        Args:
            session: SQLAlchemy database session
            identifier: Either equipment_group ID (int) or name (str)
            is_id (bool): If True, identifier is an ID, otherwise it's a name
            request_id (str, optional): Unique identifier for the request

        Returns:
            dict: Dictionary containing:
                - 'equipment_group': The found equipment group object
                - 'upward': Dictionary containing 'area' the equipment group belongs to
                - 'downward': Dictionary containing:
                    - 'models': List of all models belonging to this equipment group
                    - 'positions': List of all positions directly related to this equipment group
        """
        # Find the equipment group
        if is_id:
            equipment_group = session.query(cls).filter(cls.id == identifier).first()
        else:
            equipment_group = session.query(cls).filter(cls.name == identifier).first()

        if not equipment_group:
            return None

        # Going upward in the hierarchy
        upward = {
            'area': equipment_group.area
        }

        # Going downward in the hierarchy
        downward = {
            'models': equipment_group.model,
            'positions': equipment_group.position
        }

        # Collecting more detailed information from models if needed
        model_details = []
        for model in equipment_group.model:
            model_info = {
                'id': model.id,
                'name': model.name,
                'description': model.description,
                'asset_numbers': model.asset_number,
                'locations': model.location,
                'positions': model.position
            }
            model_details.append(model_info)

        downward['model_details'] = model_details

        return {
            'equipment_group': equipment_group,
            'upward': upward,
            'downward': downward
        }

class Model(Base):
    __tablename__ = 'model'

    id = Column(Integer, primary_key=True)
    name = Column(String, nullable=False)
    description = Column(String,nullable=True)
    equipment_group_id = Column(Integer, ForeignKey('equipment_group.id'))
    
    equipment_group = relationship("EquipmentGroup", back_populates="model")
    asset_number = relationship("AssetNumber", back_populates="model")
    location = relationship("Location", back_populates="model")
    position = relationship("Position", back_populates="model")

    @classmethod
    @with_request_id
    def search_models(cls, session, query, limit=10):
        """
        Searches for models that match the provided query using a case-insensitive
        partial match on the name field. Useful for autocomplete or dynamic search interfaces.

        Parameters:
            session: SQLAlchemy session object used for querying.
            query: The partial model name input by the user.
            limit: Maximum number of results to return (default is 10).

        Returns:
            A list of dictionaries, each containing details about a model:
              - id: The model's unique identifier.
              - name: The model's name.
              - description: The model's description.
              - equipment_group_id: The associated equipment group ID.
            If no records match, an empty list is returned.
        """
        logger.info("========== MODEL AUTOCOMPLETE SEARCH ==========")
        logger.debug(f"Initiating search for models with query: '{query}'")

        try:
            if not query:
                logger.debug("Empty query received; returning empty result set.")
                return []

            search_pattern = f"%{query}%"
            logger.debug(f"Using search pattern: '{search_pattern}'")

            results = session.query(cls).filter(cls.name.ilike(search_pattern)).limit(limit).all()

            if results:
                models = []
                for model in results:
                    model_details = {
                        "id": model.id,
                        "name": model.name,
                        "description": model.description,
                        "equipment_group_id": model.equipment_group_id
                    }
                    models.append(model_details)
                    logger.debug(f"Found model: {model_details}")
                logger.info(f"Found {len(models)} model(s) matching query '{query}'.")
                return models
            else:
                logger.warning(f"No models found matching query '{query}'.")
                return []
        except Exception as e:
            logger.error(f"Error searching for models with query '{query}': {e}")
            import traceback
            logger.error(traceback.format_exc())
            return []
        finally:
            logger.info("========== MODEL AUTOCOMPLETE SEARCH COMPLETE ==========")

    @classmethod
    @with_request_id
    def add_model(cls, session, name, equipment_group_id, description=None, request_id=None):
        """
        Add a new model to the database.

        Args:
            session: SQLAlchemy database session
            name (str): Name of the model
            equipment_group_id (int): ID of the equipment group this model belongs to
            description (str, optional): Description of the model
            request_id (str, optional): Unique identifier for the request

        Returns:
            Model: The newly created model object
        """
        new_model = cls(
            name=name,
            equipment_group_id=equipment_group_id,
            description=description
        )

        session.add(new_model)
        session.commit()

        return new_model

    @classmethod
    @with_request_id
    def delete_model(cls, session, model_id, request_id=None):
        """
        Delete a model from the database.

        Args:
            session: SQLAlchemy database session
            model_id (int): ID of the model to delete
            request_id (str, optional): Unique identifier for the request

        Returns:
            bool: True if deletion was successful, False if model not found
        """
        model = session.query(cls).filter(cls.id == model_id).first()

        if model:
            session.delete(model)
            session.commit()
            return True
        else:
            return False

    @classmethod
    @with_request_id
    def find_related_entities(cls, session, identifier, is_id=True, request_id=None):
        """
        Find all related entities for a model, traversing both up and down
        the hierarchy: Area → EquipmentGroup → Model → (AssetNumber, Location, Position).

        Args:
            session: SQLAlchemy database session
            identifier: Either model ID (int) or name (str)
            is_id (bool): If True, identifier is an ID, otherwise it's a name
            request_id (str, optional): Unique identifier for the request

        Returns:
            dict: Dictionary containing:
                - 'model': The found model object
                - 'upward': Dictionary containing 'equipment_group' and 'area'
                - 'downward': Dictionary containing:
                    - 'asset_numbers': List of all asset numbers belonging to this model
                    - 'locations': List of all locations for this model
                    - 'positions': List of all positions related to this model
        """
        # Find the model
        if is_id:
            model = session.query(cls).filter(cls.id == identifier).first()
        else:
            model = session.query(cls).filter(cls.name == identifier).first()

        if not model:
            return None

        # Going upward in the hierarchy
        upward = {
            'equipment_group': model.equipment_group,
            'area': model.equipment_group.area if model.equipment_group else None
        }

        # Going downward in the hierarchy
        downward = {
            'asset_numbers': model.asset_number,
            'locations': model.location,
            'positions': model.position
        }

        return {
            'model': model,
            'upward': upward,
            'downward': downward
        }

class AssetNumber(Base):
    __tablename__ = 'asset_number'

    id = Column(Integer, primary_key=True)
    number = Column(String, nullable=False)
    description = Column(String)
    model_id = Column(Integer, ForeignKey('model.id'))

    model = relationship("Model", back_populates="asset_number")
    position = relationship("Position", back_populates="asset_number")

    @classmethod
    @with_request_id
    def get_ids_by_number(cls, session, number):
        """Retrieve all AssetNumber IDs that match the given number."""
        logger.info(f"========== ASSET NUMBER SEARCH ==========")
        logger.debug(f"Querying AssetNumber IDs for number: '{number}'")

        try:
            # Log the search pattern being used
            logger.debug(f"Using exact match search pattern for number: '{number}'")

            # Execute the query
            results = session.query(cls.id).filter(cls.number == number).all()

            # Extract IDs from the results
            ids = [id_ for (id_,) in results]

            # Log detailed information about the results
            if ids:
                logger.info(f"Found {len(ids)} AssetNumbers with number '{number}': {ids}")
                for i, asset_id in enumerate(ids):
                    try:
                        # Get more details about each asset found
                        asset = session.query(cls).filter(cls.id == asset_id).first()
                        if asset:
                            logger.debug(f"Asset #{i + 1}: ID={asset_id}, Number={asset.number}, " +
                                         f"Description={asset.description or 'None'}, Model ID={asset.model_id}")

                            # Get model info if available
                            if asset.model_id:
                                model = session.query(Model).filter(Model.id == asset.model_id).first()
                                if model:
                                    logger.debug(f"  -> Model: ID={model.id}, Name={model.name}")
                    except Exception as e:
                        logger.warning(f"Error getting details for asset ID {asset_id}: {e}")
            else:
                logger.warning(f"No AssetNumbers found with number '{number}'")

            return ids
        except Exception as e:
            logger.error(f"Error querying AssetNumbers by number '{number}': {e}")
            import traceback
            logger.error(traceback.format_exc())
            return []
        finally:
            logger.info(f"========== ASSET NUMBER SEARCH COMPLETE ==========")

    @classmethod
    @with_request_id
    def get_model_id_by_asset_number_id(cls, session, asset_number_id):
        """
        Given an asset_number_id, returns the associated model_id.

        Parameters:
            session: SQLAlchemy session object used for querying.
            asset_number_id: The id of the AssetNumber record.

        Returns:
            The model_id associated with the asset_number, or None if not found.
        """
        logger.info(f"========== GETTING MODEL FOR ASSET ID {asset_number_id} ==========")
        logger.debug(f"Querying AssetNumber for asset_number_id: {asset_number_id}")

        try:
            # First try to get the full asset record for more detailed logging
            asset = session.query(cls).filter(cls.id == asset_number_id).first()
            if asset:
                logger.debug(f"Found asset: ID={asset.id}, Number={asset.number}, " +
                             f"Description={asset.description or 'None'}, Model ID={asset.model_id}")
                model_id = asset.model_id
            else:
                # Fallback to just getting the model_id directly
                logger.debug(f"Asset not found, querying only for the model_id")
                model_id = session.query(cls.model_id).filter(cls.id == asset_number_id).scalar()

            if model_id is not None:
                logger.info(f"Found model_id: {model_id} for asset_number_id: {asset_number_id}")

                # Get model details for better logging
                try:
                    model = session.query(Model).filter(Model.id == model_id).first()
                    if model:
                        logger.debug(f"Model details: ID={model.id}, Name={model.name}, " +
                                     f"Equipment Group ID={model.equipment_group_id}")
                except Exception as e:
                    logger.warning(f"Error getting model details: {e}")
            else:
                logger.warning(f"No AssetNumber found with id: {asset_number_id}")

            return model_id
        except Exception as e:
            logger.error(f"Error getting model_id for asset_number_id {asset_number_id}: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return None
        finally:
            logger.info(f"========== MODEL SEARCH COMPLETE ==========")

    @classmethod
    @with_request_id
    def get_equipment_group_id_by_asset_number_id(cls, session, asset_number_id):
        """
        Given an asset_number_id, retrieves the equipment_group id that is associated with its model.

        This method works in two steps:
          1. It joins the AssetNumber table with the Model table (using AssetNumber.model_id).
          2. It selects the 'equipment_group_id' field from Model, which holds the id of the associated EquipmentGroup.

        Parameters:
            session: SQLAlchemy session object used for querying.
            asset_number_id: The id of the AssetNumber record.

        Returns:
            The equipment_group id if found, otherwise None.
        """
        logger.info(f"========== GETTING EQUIPMENT GROUP FOR ASSET ID {asset_number_id} ==========")
        logger.debug(f"Querying for equipment_group id using asset_number_id: {asset_number_id}")

        try:
            # Try to get the model_id first for more detailed logging
            model_id = cls.get_model_id_by_asset_number_id(session, asset_number_id)

            if model_id is not None:
                logger.debug(f"Found model_id: {model_id} for asset_number_id: {asset_number_id}")

                # Query directly using the model ID for better performance
                equipment_group_id = session.query(Model.equipment_group_id).filter(Model.id == model_id).scalar()

                if equipment_group_id is not None:
                    logger.info(f"Found equipment_group_id: {equipment_group_id} via model_id {model_id}")

                    # Get equipment group details for better logging
                    try:
                        group = session.query(EquipmentGroup).filter(EquipmentGroup.id == equipment_group_id).first()
                        if group:
                            logger.debug(f"Equipment Group details: ID={group.id}, Name={group.name}, " +
                                         f"Area ID={group.area_id}")
                    except Exception as e:
                        logger.warning(f"Error getting equipment group details: {e}")
                else:
                    logger.warning(f"No equipment_group_id found for model_id: {model_id}")

                    # Fall back to the join method
                    logger.debug(f"Falling back to join query method")
                    equipment_group_id = (
                        session.query(Model.equipment_group_id)
                        .join(Model, Model.id == cls.model_id)
                        .filter(cls.id == asset_number_id)
                        .scalar()
                    )
            else:
                # If we couldn't get the model_id, use the join method directly
                logger.debug(f"No model_id found, using join query method directly")
                equipment_group_id = (
                    session.query(Model.equipment_group_id)
                    .join(Model, Model.id == cls.model_id)
                    .filter(cls.id == asset_number_id)
                    .scalar()
                )

            if equipment_group_id is not None:
                logger.info(
                    f"Final result: Found equipment_group_id: {equipment_group_id} for asset_number_id: {asset_number_id}")
            else:
                logger.warning(f"Final result: No EquipmentGroup found for asset_number_id: {asset_number_id}")

            return equipment_group_id
        except Exception as e:
            logger.error(f"Error getting equipment_group_id for asset_number_id {asset_number_id}: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return None
        finally:
            logger.info(f"========== EQUIPMENT GROUP SEARCH COMPLETE ==========")

    @classmethod
    @with_request_id
    def get_area_id_by_asset_number_id(cls, session, asset_number_id):
        """
        Given an asset_number_id, retrieves the associated area_id.

        This method performs a series of joins:
          1. Join Area to EquipmentGroup on Area.id equals EquipmentGroup.area_id.
          2. Join EquipmentGroup to Model on EquipmentGroup.id equals Model.equipment_group_id.
          3. Join Model to AssetNumber on Model.id equals AssetNumber.model_id.
          4. Filter by the specified asset_number_id to ultimately extract the Area.id.

        Parameters:
            session: SQLAlchemy session object used for querying.
            asset_number_id: The id of the AssetNumber record.

        Returns:
            The area_id associated with the asset_number, or None if no matching record is found.
        """
        logger.info(f"========== GETTING AREA FOR ASSET ID {asset_number_id} ==========")
        logger.debug(f"Querying for area_id using asset_number_id: {asset_number_id}")

        try:
            # Try to get the equipment_group_id first for more detailed logging
            equipment_group_id = cls.get_equipment_group_id_by_asset_number_id(session, asset_number_id)

            if equipment_group_id is not None:
                logger.debug(f"Found equipment_group_id: {equipment_group_id} for asset_number_id: {asset_number_id}")

                # Query directly using the equipment group ID for better performance
                area_id = session.query(EquipmentGroup.area_id).filter(EquipmentGroup.id == equipment_group_id).scalar()

                if area_id is not None:
                    logger.info(f"Found area_id: {area_id} via equipment_group_id {equipment_group_id}")

                    # Get area details for better logging
                    try:
                        area = session.query(Area).filter(Area.id == area_id).first()
                        if area:
                            logger.debug(f"Area details: ID={area.id}, Name={area.name}")
                    except Exception as e:
                        logger.warning(f"Error getting area details: {e}")
                else:
                    logger.warning(f"No area_id found for equipment_group_id: {equipment_group_id}")

                    # Fall back to the join method
                    logger.debug(f"Falling back to join query method")
                    area_id = (
                        session.query(Area.id)
                        .join(EquipmentGroup, EquipmentGroup.area_id == Area.id)
                        .join(Model, Model.equipment_group_id == EquipmentGroup.id)
                        .join(AssetNumber, AssetNumber.model_id == Model.id)
                        .filter(AssetNumber.id == asset_number_id)
                        .scalar()
                    )
            else:
                # If we couldn't get the equipment_group_id, use the join method directly
                logger.debug(f"No equipment_group_id found, using join query method directly")
                area_id = (
                    session.query(Area.id)
                    .join(EquipmentGroup, EquipmentGroup.area_id == Area.id)
                    .join(Model, Model.equipment_group_id == EquipmentGroup.id)
                    .join(AssetNumber, AssetNumber.model_id == Model.id)
                    .filter(AssetNumber.id == asset_number_id)
                    .scalar()
                )

            if area_id is not None:
                logger.info(f"Final result: Found area_id: {area_id} for asset_number_id: {asset_number_id}")
            else:
                logger.warning(f"Final result: No area found for asset_number_id: {asset_number_id}")

            return area_id
        except Exception as e:
            logger.error(f"Error getting area_id for asset_number_id {asset_number_id}: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return None
        finally:
            logger.info(f"========== AREA SEARCH COMPLETE ==========")

    @classmethod
    @with_request_id
    def get_position_ids_by_asset_number_id(cls, session, asset_number_id):
        """
        Given an asset_number_id, retrieves all Position IDs that reference this asset_number.

        This method performs a query on the Position table where the asset_number_id
        matches the provided value. It returns a list of Position.id values.

        Parameters:
            session: SQLAlchemy session object used for querying.
            asset_number_id: The id value of the AssetNumber record.

        Returns:
            A list of Position IDs associated with the given asset_number_id.
            If no matching positions are found, an empty list is returned.
        """
        logger.info(f"========== GETTING POSITIONS FOR ASSET ID {asset_number_id} ==========")
        logger.debug(f"Querying for all Position IDs with asset_number_id: {asset_number_id}")

        try:
            # Get the asset details for more context in logging
            asset = session.query(cls).filter(cls.id == asset_number_id).first()
            if asset:
                logger.debug(f"Asset details: ID={asset.id}, Number={asset.number}, " +
                             f"Description={asset.description or 'None'}, Model ID={asset.model_id}")

            # Execute the query to get positions
            results = session.query(Position.id).filter(Position.asset_number_id == asset_number_id).all()
            position_ids = [pos_id for (pos_id,) in results]

            # Log detailed information about the results
            if position_ids:
                logger.info(
                    f"Found {len(position_ids)} Position(s) for asset_number_id: {asset_number_id}: {position_ids}")

                # Log details about each position
                for i, pos_id in enumerate(position_ids):
                    try:
                        position = session.query(Position).filter(Position.id == pos_id).first()
                        if position:
                            logger.debug(f"Position #{i + 1}: ID={pos_id}, " +
                                         f"Area ID={position.area_id}, " +
                                         f"Group ID={position.equipment_group_id}, " +
                                         f"Model ID={position.model_id}, " +
                                         f"Location ID={position.location_id}")

                            # Try to get location name for more context
                            if position.location_id:
                                location = session.query(Location).filter(Location.id == position.location_id).first()
                                if location:
                                    logger.debug(f"  -> Location: ID={location.id}, Name={location.name}")
                    except Exception as e:
                        logger.warning(f"Error getting details for position ID {pos_id}: {e}")
            else:
                logger.warning(f"No Positions found for asset_number_id: {asset_number_id}")

            return position_ids
        except Exception as e:
            logger.error(f"Error getting position_ids for asset_number_id {asset_number_id}: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return []
        finally:
            logger.info(f"========== POSITION SEARCH COMPLETE ==========")

    @classmethod
    @with_request_id
    def search_asset_numbers(cls, session, query, limit=10):
        """
        Searches for asset numbers that match the provided query using
        a case-insensitive partial match. Useful for autocomplete or dynamic
        search interfaces.

        Parameters:
            session: SQLAlchemy session object used for querying.
            query: The partial asset number string input by the user.
            limit: Maximum number of results to return (default is 10).

        Returns:
            A list of dictionaries, each containing details about an asset:
              - id: The asset's unique identifier.
              - number: The asset number.
              - description: The asset description.
              - model_id: The associated model ID.
            If no records match, an empty list is returned.
        """
        logger.info("========== ASSET NUMBER AUTOCOMPLETE SEARCH ==========")
        logger.debug(f"Initiating search for asset numbers with query: '{query}'")

        try:
            # If the query is empty, just return an empty list early
            if not query:
                logger.debug("Empty query received; returning empty result set.")
                return []

            # Create a search pattern for a partial, case-insensitive match.
            search_pattern = f"%{query}%"
            logger.debug(f"Using search pattern: '{search_pattern}'")

            # Query for matching asset numbers; you can adjust the limit as needed.
            results = session.query(cls).filter(cls.number.ilike(search_pattern)).limit(limit).all()

            if results:
                assets = []
                # Loop through the found results to build a structured list with detailed logging.
                for asset in results:
                    asset_details = {
                        "id": asset.id,
                        "number": asset.number,
                        "description": asset.description,
                        "model_id": asset.model_id
                    }
                    assets.append(asset_details)
                    logger.debug(f"Found asset: {asset_details}")

                logger.info(f"Found {len(assets)} asset(s) matching query '{query}'.")
                return assets
            else:
                logger.warning(f"No assets found matching query '{query}'.")
                return []
        except Exception as e:
            logger.error(f"Error searching for asset numbers with query '{query}': {e}")
            import traceback
            logger.error(traceback.format_exc())
            return []
        finally:
            logger.info("========== ASSET NUMBER AUTOCOMPLETE SEARCH COMPLETE ==========")

    @classmethod
    @with_request_id
    def add_asset_number(cls, session, number, model_id, description=None, request_id=None):
        """
        Add a new asset number to the database.

        Args:
            session: SQLAlchemy database session
            number (str): Asset number
            model_id (int): ID of the model this asset number belongs to
            description (str, optional): Description of the asset number
            request_id (str, optional): Unique identifier for the request

        Returns:
            AssetNumber: The newly created asset number object
        """
        new_asset_number = cls(
            number=number,
            model_id=model_id,
            description=description
        )

        session.add(new_asset_number)
        session.commit()

        logger.info(f"Created new asset number: {number} for model ID {model_id}")
        return new_asset_number

    @classmethod
    @with_request_id
    def delete_asset_number(cls, session, asset_number_id, request_id=None):
        """
        Delete an asset number from the database.

        Args:
            session: SQLAlchemy database session
            asset_number_id (int): ID of the asset number to delete
            request_id (str, optional): Unique identifier for the request

        Returns:
            bool: True if deletion was successful, False if asset number not found
        """
        asset_number = session.query(cls).filter(cls.id == asset_number_id).first()

        if asset_number:
            session.delete(asset_number)
            session.commit()
            logger.info(f"Deleted asset number ID {asset_number_id}")
            return True
        else:
            logger.warning(f"Failed to delete asset number ID {asset_number_id} - not found")
            return False

    @classmethod
    @with_request_id
    def find_related_entities(cls, session, identifier, is_id=True, request_id=None):
        """
        Find all related entities for an asset number, traversing both up and down
        the hierarchy: Area → EquipmentGroup → Model → AssetNumber → Position.

        Args:
            session: SQLAlchemy database session
            identifier: Either asset_number ID (int) or number (str)
            is_id (bool): If True, identifier is an ID, otherwise it's a number
            request_id (str, optional): Unique identifier for the request

        Returns:
            dict: Dictionary containing:
                - 'asset_number': The found asset number object
                - 'upward': Dictionary containing 'model', 'equipment_group', and 'area'
                - 'downward': Dictionary containing:
                    - 'positions': List of all positions related to this asset number
        """
        # Find the asset number
        if is_id:
            asset_number = session.query(cls).filter(cls.id == identifier).first()
        else:
            asset_number = session.query(cls).filter(cls.number == identifier).first()

        if not asset_number:
            logger.warning(f"Asset number not found for identifier: {identifier}")
            return None

        # Going upward in the hierarchy
        upward = {
            'model': asset_number.model,
            'equipment_group': asset_number.model.equipment_group if asset_number.model else None,
            'area': asset_number.model.equipment_group.area if asset_number.model and asset_number.model.equipment_group else None
        }

        # Going downward in the hierarchy
        downward = {
            'positions': asset_number.position
        }

        logger.info(f"Found related entities for asset number ID {asset_number.id}")
        return {
            'asset_number': asset_number,
            'upward': upward,
            'downward': downward
        }
    
class Location(Base):
    __tablename__ = 'location'

    id = Column(Integer, primary_key=True)
    name = Column(String, nullable=False)    
    model_id = Column(Integer, ForeignKey('model.id'))
    description = Column(String, nullable=True)
    
    model = relationship("Model", back_populates="location")
    position = relationship("Position", back_populates="location")
    subassembly = relationship("Subassembly", back_populates="location")

    @classmethod
    @with_request_id
    def add_location(cls, session, name, model_id, description=None, request_id=None):
        """
        Add a new location to the database.

        Args:
            session: SQLAlchemy database session
            name (str): Name of the location
            model_id (int): ID of the model this location belongs to
            description (str, optional): Description of the location
            request_id (str, optional): Unique identifier for the request

        Returns:
            Location: The newly created location object
        """
        new_location = cls(
            name=name,
            model_id=model_id,
            description=description
        )

        session.add(new_location)
        session.commit()

        logger.info(f"Created new location: {name} for model ID {model_id}")
        return new_location

    @classmethod
    @with_request_id
    def delete_location(cls, session, location_id, request_id=None):
        """
        Delete a location from the database.

        Args:
            session: SQLAlchemy database session
            location_id (int): ID of the location to delete
            request_id (str, optional): Unique identifier for the request

        Returns:
            bool: True if deletion was successful, False if location not found
        """
        location = session.query(cls).filter(cls.id == location_id).first()

        if location:
            session.delete(location)
            session.commit()
            logger.info(f"Deleted location ID {location_id}")
            return True
        else:
            logger.warning(f"Failed to delete location ID {location_id} - not found")
            return False

    @classmethod
    @with_request_id
    def find_related_entities(cls, session, identifier, is_id=True, request_id=None):
        """
        Find all related entities for a location, traversing both up and down
        the hierarchy: Area → EquipmentGroup → Model → Location → (Position, Subassembly).

        Args:
            session: SQLAlchemy database session
            identifier: Either location ID (int) or name (str)
            is_id (bool): If True, identifier is an ID, otherwise it's a name
            request_id (str, optional): Unique identifier for the request

        Returns:
            dict: Dictionary containing:
                - 'location': The found location object
                - 'upward': Dictionary containing 'model', 'equipment_group', and 'area'
                - 'downward': Dictionary containing:
                    - 'positions': List of all positions related to this location
                    - 'subassemblies': List of all subassemblies related to this location
        """
        # Find the location
        if is_id:
            location = session.query(cls).filter(cls.id == identifier).first()
        else:
            location = session.query(cls).filter(cls.name == identifier).first()

        if not location:
            logger.warning(f"Location not found for identifier: {identifier}")
            return None

        # Going upward in the hierarchy
        upward = {
            'model': location.model,
            'equipment_group': location.model.equipment_group if location.model else None,
            'area': location.model.equipment_group.area if location.model and location.model.equipment_group else None
        }

        # Going downward in the hierarchy
        downward = {
            'positions': location.position,
            'subassemblies': location.subassembly
        }

        logger.info(f"Found related entities for location ID {location.id}")
        return {
            'location': location,
            'upward': upward,
            'downward': downward
        }

#class's dealing with machine subassemblies.

class Subassembly(Base):
    __tablename__ = 'subassembly'
    id = Column(Integer, primary_key=True)
    name = Column(String, nullable=True)
    location_id = Column(Integer, ForeignKey('location.id'))
    description = Column(String, nullable=True)

    # Relationships
    location = relationship("Location", back_populates="subassembly")
    component_assembly = relationship("ComponentAssembly", back_populates="subassembly")
    position = relationship("Position", back_populates="subassembly")

    @classmethod
    @with_request_id
    def add_subassembly(cls, session, name, location_id, description=None, request_id=None):
        """Add a new Subassembly to the database."""
        new_sub = cls(name=name, location_id=location_id, description=description)
        session.add(new_sub)
        session.commit()
        logger.info(f"Created Subassembly '{name}' under Location {location_id}")
        return new_sub

    @classmethod
    @with_request_id
    def delete_subassembly(cls, session, subassembly_id, request_id=None):
        """Delete a Subassembly by ID."""
        sub = session.query(cls).filter_by(id=subassembly_id).first()
        if sub:
            session.delete(sub)
            session.commit()
            logger.info(f"Deleted Subassembly id={subassembly_id}")
            return True
        return False

    @classmethod
    @with_request_id
    def search(cls, session, name=None, location_id=None):
        """Search Subassemblies by name or location_id."""
        query = session.query(cls)
        if name:
            query = query.filter(cls.name.ilike(f"%{name}%"))
        if location_id:
            query = query.filter(cls.location_id == location_id)
        return query.all()

    @classmethod
    @with_request_id
    def find_or_create(cls, session, name, location_id, description=None, request_id=None):
        """Find Subassembly by name + location, or create it."""
        sub = session.query(cls).filter_by(name=name, location_id=location_id).first()
        if sub:
            return sub
        sub = cls(name=name, location_id=location_id, description=description)
        session.add(sub)
        session.commit()
        return sub

    @classmethod
    @with_request_id
    def find_related_entities(cls, session, subassembly_id, request_id=None):
        """Traverse relationships: upward (location), downward (component assemblies, positions)."""
        sub = session.query(cls).filter_by(id=subassembly_id).first()
        if not sub:
            return None
        return {
            "subassembly": sub,
            "upward": {"location": sub.location},
            "downward": {"component_assemblies": sub.component_assembly, "positions": sub.position}
        }

class ComponentAssembly(Base):
    __tablename__ = 'component_assembly'
    id = Column(Integer, primary_key=True)
    name = Column(String, nullable=True)
    description = Column(String, nullable=True)
    subassembly_id = Column(Integer, ForeignKey('subassembly.id'), nullable=False)

    # Relationships
    subassembly = relationship("Subassembly", back_populates="component_assembly")
    assembly_view = relationship("AssemblyView", back_populates="component_assembly")
    position = relationship("Position", back_populates="component_assembly")

    @classmethod
    @with_request_id
    def add_component(cls, session, name, subassembly_id, description=None, request_id=None):
        """Add a new ComponentAssembly."""
        comp = cls(name=name, subassembly_id=subassembly_id, description=description)
        session.add(comp)
        session.commit()
        logger.info(f"Created ComponentAssembly '{name}' under Subassembly {subassembly_id}")
        return comp

    @classmethod
    @with_request_id
    def delete_component(cls, session, component_id, request_id=None):
        """Delete a ComponentAssembly by ID."""
        comp = session.query(cls).filter_by(id=component_id).first()
        if comp:
            session.delete(comp)
            session.commit()
            logger.info(f"Deleted ComponentAssembly id={component_id}")
            return True
        return False

    @classmethod
    @with_request_id
    def search(cls, session, name=None, subassembly_id=None):
        """Search ComponentAssemblies by name or subassembly_id."""
        query = session.query(cls)
        if name:
            query = query.filter(cls.name.ilike(f"%{name}%"))
        if subassembly_id:
            query = query.filter(cls.subassembly_id == subassembly_id)
        return query.all()

    @classmethod
    @with_request_id
    def find_or_create(cls, session, name, subassembly_id, description=None, request_id=None):
        """Find a ComponentAssembly or create it if missing."""
        comp = session.query(cls).filter_by(name=name, subassembly_id=subassembly_id).first()
        if comp:
            return comp
        comp = cls(name=name, subassembly_id=subassembly_id, description=description)
        session.add(comp)
        session.commit()
        return comp

    @classmethod
    @with_request_id
    def find_related_entities(cls, session, component_id, request_id=None):
        """Traverse relationships: upward (subassembly), downward (assembly views, positions)."""
        comp = session.query(cls).filter_by(id=component_id).first()
        if not comp:
            return None
        return {
            "component_assembly": comp,
            "upward": {"subassembly": comp.subassembly},
            "downward": {"assembly_views": comp.assembly_view, "positions": comp.position}
        }

class AssemblyView(Base):  # TODO rename to ComponentView
    __tablename__ = 'assembly_view'
    id = Column(Integer, primary_key=True)
    name = Column(String, nullable=True)
    description = Column(String, nullable=True)
    component_assembly_id = Column(Integer, ForeignKey('component_assembly.id'), nullable=False)

    # Relationships
    component_assembly = relationship("ComponentAssembly", back_populates="assembly_view")
    position = relationship("Position", back_populates="assembly_view")

    @classmethod
    @with_request_id
    def add_view(cls, session, name, component_assembly_id, description=None, request_id=None):
        """Add a new AssemblyView (ComponentView)."""
        view = cls(name=name, component_assembly_id=component_assembly_id, description=description)
        session.add(view)
        session.commit()
        logger.info(f"Created AssemblyView '{name}' under ComponentAssembly {component_assembly_id}")
        return view

    @classmethod
    @with_request_id
    def delete_view(cls, session, view_id, request_id=None):
        """Delete an AssemblyView by ID."""
        view = session.query(cls).filter_by(id=view_id).first()
        if view:
            session.delete(view)
            session.commit()
            logger.info(f"Deleted AssemblyView id={view_id}")
            return True
        return False

    @classmethod
    @with_request_id
    def search(cls, session, name=None, component_assembly_id=None):
        """Search AssemblyViews by name or component_assembly_id."""
        query = session.query(cls)
        if name:
            query = query.filter(cls.name.ilike(f"%{name}%"))
        if component_assembly_id:
            query = query.filter(cls.component_assembly_id == component_assembly_id)
        return query.all()

    @classmethod
    @with_request_id
    def find_or_create(cls, session, name, component_assembly_id, description=None, request_id=None):
        """Find an AssemblyView by name + component_assembly, or create it."""
        view = session.query(cls).filter_by(name=name, component_assembly_id=component_assembly_id).first()
        if view:
            return view
        view = cls(name=name, component_assembly_id=component_assembly_id, description=description)
        session.add(view)
        session.commit()
        return view

    @classmethod
    @with_request_id
    def find_related_entities(cls, session, view_id, request_id=None):
        """Traverse relationships: upward (component assembly), downward (positions)."""
        view = session.query(cls).filter_by(id=view_id).first()
        if not view:
            return None
        return {
            "assembly_view": view,
            "upward": {"component_assembly": view.component_assembly},
            "downward": {"positions": view.position}
        }

class Part(Base):
    __tablename__ = 'part'

    id = Column(Integer, primary_key=True)
    part_number = Column(String, unique=True)  # MP2=ITEMNUM, SPC= Item Number
    name = Column(String)  # MP2=DESCRIPTION, SPC= Description
    oem_mfg = Column(String)  # MP2=OEMMFG, SPC= Manufacturer
    model = Column(String)  # MP2=MODEL, SPC= MFG Part Number
    class_flag = Column(String)  # MP2=Class Flag SPC= Category
    ud6 = Column(String)  # MP2=UD6
    type = Column(String)  # MP2=TYPE
    notes = Column(String)  # MP2=Notes, SPC= Long Description
    documentation = Column(String)  # MP2=Specifications

    # FTS Column - will be populated by trigger
    search_vector = Column(TSVECTOR)

    part_position_image = relationship("PartsPositionImageAssociation", back_populates="part")
    part_problem = relationship("PartProblemAssociation", back_populates="part")
    part_task = relationship("PartTaskAssociation", back_populates="part")
    drawing_part = relationship("DrawingPartAssociation", back_populates="part")

    __table_args__ = (UniqueConstraint('part_number', name='_part_number_uc'),)

    @classmethod
    @with_request_id
    def add_part_to_db(cls,
                       session,
                       part_number: str,
                       name: Optional[str] = None,
                       oem_mfg: Optional[str] = None,
                       model: Optional[str] = None,
                       class_flag: Optional[str] = None,
                       ud6: Optional[str] = None,
                       type_: Optional[str] = None,
                       notes: Optional[str] = None,
                       documentation: Optional[str] = None,
                       request_id: Optional[str] = None) -> int:
        """
        Get-or-create a Part by part_number. Always returns the Part ID.

        Args:
            session: SQLAlchemy session
            part_number (str): Unique part number
            Other fields: Optional values for creation

        Returns:
            int: ID of the existing or newly created Part
        """
        try:
            part = session.query(cls).filter_by(part_number=part_number).first()
            if part:
                info_id(f"Found existing Part '{part_number}' (id={part.id})", request_id)
                return part.id

            # Create new Part
            part = cls(
                part_number=part_number,
                name=name,
                oem_mfg=oem_mfg,
                model=model,
                class_flag=class_flag,
                ud6=ud6,
                type=type_,
                notes=notes,
                documentation=documentation
            )
            session.add(part)
            session.commit()
            info_id(f"Created new Part '{part_number}' (id={part.id})", request_id)
            return part.id

        except SQLAlchemyError as e:
            session.rollback()
            error_id(f"Part.add_part_to_db failed: {e}", request_id)
            raise

    @classmethod
    @with_request_id
    def search(cls,
               search_text: Optional[str] = None,
               fields: Optional[List[str]] = None,
               exact_match: bool = False,
               use_fts: bool = True,  # New parameter to enable/disable FTS
               part_id: Optional[int] = None,
               part_number: Optional[str] = None,
               name: Optional[str] = None,
               oem_mfg: Optional[str] = None,
               model: Optional[str] = None,
               class_flag: Optional[str] = None,
               ud6: Optional[str] = None,
               type_: Optional[str] = None,
               notes: Optional[str] = None,
               documentation: Optional[str] = None,
               limit: int = 100,
               request_id: Optional[str] = None,
               session: Optional[Session] = None) -> List['Part']:
        """
        Comprehensive search method for Part objects with flexible search options.
        Now includes PostgreSQL Full Text Search (FTS) capabilities.

        Args:
            search_text: Text to search for across specified fields
            fields: List of field names to search in. If None, searches in default fields
                   (part_number, name, oem_mfg, model)
            exact_match: If True, performs exact matching instead of partial matching
            use_fts: If True and search_text provided, uses PostgreSQL FTS for better performance
            part_id: Optional ID to filter by
            part_number: Optional part_number to filter by
            name: Optional name to filter by
            oem_mfg: Optional oem_mfg to filter by
            model: Optional model to filter by
            class_flag: Optional class_flag to filter by
            ud6: Optional ud6 to filter by
            type_: Optional type to filter by (using type_ to avoid Python keyword conflict)
            notes: Optional notes to filter by
            documentation: Optional documentation to filter by
            limit: Maximum number of results to return (default 100)
            request_id: Optional request ID for tracking this operation in logs
            session: Optional SQLAlchemy session. If None, a new session will be created

        Returns:
            List of Part objects matching the search criteria, ordered by relevance when using FTS
        """
        # Get or use the provided request_id
        rid = request_id or get_request_id()

        # Get a database session if one wasn't provided
        db_config = DatabaseConfig()
        session_provided = session is not None
        if not session_provided:
            session = db_config.get_main_session()
            debug_id(f"Created new database session for Part.search", rid)

        # Log the search operation with request ID
        search_params = {
            'search_text': search_text,
            'fields': fields,
            'exact_match': exact_match,
            'use_fts': use_fts,
            'part_id': part_id,
            'part_number': part_number,
            'name': name,
            'oem_mfg': oem_mfg,
            'model': model,
            'class_flag': class_flag,
            'ud6': ud6,
            'type_': type_,
            'notes': notes,
            'documentation': documentation,
            'limit': limit
        }
        # Filter out None values for cleaner logging
        logged_params = {k: v for k, v in search_params.items() if v is not None}
        debug_id(f"Starting Part.search with parameters: {logged_params}", rid)

        try:
            # Use the timed operation context manager with request ID
            with log_timed_operation("Part.search", rid):
                # Start with the base query
                query = session.query(cls)
                filters = []
                use_fts_ranking = False

                # Process search_text across multiple fields if provided
                if search_text:
                    search_text = search_text.strip()
                    if search_text:
                        # Check if we should use FTS
                        if use_fts and hasattr(cls, 'search_vector'):
                            debug_id(f"Using PostgreSQL FTS for text search: '{search_text}'", rid)
                            try:
                                # Use PostgreSQL FTS with ranking
                                ts_query = func.plainto_tsquery('english', search_text)

                                # Add FTS filter
                                filters.append(cls.search_vector.op('@@')(ts_query))

                                # Add ranking for ordering
                                query = query.add_columns(
                                    func.ts_rank(cls.search_vector, ts_query).label('rank')
                                )
                                use_fts_ranking = True

                            except Exception as fts_error:
                                debug_id(f"FTS search failed, falling back to ILIKE: {fts_error}", rid)
                                use_fts = False

                        # Fallback to traditional ILIKE search if FTS not available or failed
                        if not use_fts:
                            # Default fields to search in if none specified
                            if fields is None or len(fields) == 0:
                                fields = ['part_number', 'name', 'oem_mfg', 'model']

                            debug_id(f"Using ILIKE search for text '{search_text}' in fields: {fields}", rid)

                            # Create field-specific filters
                            text_filters = []
                            for field_name in fields:
                                if hasattr(cls, field_name):
                                    field = getattr(cls, field_name)
                                    if exact_match:
                                        text_filters.append(field == search_text)
                                    else:
                                        text_filters.append(field.ilike(f"%{search_text}%"))

                            # Add the combined text search filter if we have any
                            if text_filters:
                                filters.append(or_(*text_filters))

                # Add filters for specific fields if provided
                if part_id is not None:
                    debug_id(f"Adding filter for part_id: {part_id}", rid)
                    filters.append(cls.id == part_id)

                if part_number is not None:
                    debug_id(f"Adding filter for part_number: {part_number}", rid)
                    if exact_match:
                        filters.append(cls.part_number == part_number)
                    else:
                        filters.append(cls.part_number.ilike(f"%{part_number}%"))

                if name is not None:
                    debug_id(f"Adding filter for name: {name}", rid)
                    if exact_match:
                        filters.append(cls.name == name)
                    else:
                        filters.append(cls.name.ilike(f"%{name}%"))

                if oem_mfg is not None:
                    debug_id(f"Adding filter for oem_mfg: {oem_mfg}", rid)
                    if exact_match:
                        filters.append(cls.oem_mfg == oem_mfg)
                    else:
                        filters.append(cls.oem_mfg.ilike(f"%{oem_mfg}%"))

                if model is not None:
                    debug_id(f"Adding filter for model: {model}", rid)
                    if exact_match:
                        filters.append(cls.model == model)
                    else:
                        filters.append(cls.model.ilike(f"%{model}%"))

                if class_flag is not None:
                    debug_id(f"Adding filter for class_flag: {class_flag}", rid)
                    if exact_match:
                        filters.append(cls.class_flag == class_flag)
                    else:
                        filters.append(cls.class_flag.ilike(f"%{class_flag}%"))

                if ud6 is not None:
                    debug_id(f"Adding filter for ud6: {ud6}", rid)
                    if exact_match:
                        filters.append(cls.ud6 == ud6)
                    else:
                        filters.append(cls.ud6.ilike(f"%{ud6}%"))

                if type_ is not None:
                    debug_id(f"Adding filter for type: {type_}", rid)
                    if exact_match:
                        filters.append(cls.type == type_)
                    else:
                        filters.append(cls.type.ilike(f"%{type_}%"))

                if notes is not None:
                    debug_id(f"Adding filter for notes: {notes}", rid)
                    if exact_match:
                        filters.append(cls.notes == notes)
                    else:
                        filters.append(cls.notes.ilike(f"%{notes}%"))

                if documentation is not None:
                    debug_id(f"Adding filter for documentation: {documentation}", rid)
                    if exact_match:
                        filters.append(cls.documentation == documentation)
                    else:
                        filters.append(cls.documentation.ilike(f"%{documentation}%"))

                # Apply all filters with AND logic if we have any
                if filters:
                    query = query.filter(and_(*filters))

                # Apply ordering - use FTS ranking if available, otherwise default ordering
                if use_fts_ranking:
                    query = query.order_by(text('rank DESC'))
                else:
                    # Default ordering by part_number for consistent results
                    query = query.order_by(cls.part_number)

                # Apply limit
                query = query.limit(limit)

                # Execute query and handle results
                if use_fts_ranking:
                    # Extract just the Part objects from the ranked results
                    raw_results = query.all()
                    results = [result[0] for result in raw_results]  # result[0] is the Part object
                    debug_id(f"FTS search completed, found {len(results)} ranked results", rid)
                else:
                    results = query.all()
                    debug_id(f"Standard search completed, found {len(results)} results", rid)

                return results

        except Exception as e:
            error_id(f"Error in Part.search: {str(e)}", rid)
            # Re-raise the exception after logging it
            raise
        finally:
            # Close the session if we created it
            if not session_provided:
                session.close()
                debug_id(f"Closed database session for Part.search", rid)

    @classmethod
    @with_request_id
    def fts_search(cls,
                   search_text: str,
                   limit: int = 100,
                   request_id: Optional[str] = None,
                   session: Optional[Session] = None) -> List[tuple]:
        """
        Dedicated Full Text Search method that returns parts with their relevance scores.

        Args:
            search_text: Text to search for using PostgreSQL FTS
            limit: Maximum number of results to return
            request_id: Optional request ID for tracking
            session: Optional SQLAlchemy session

        Returns:
            List of tuples: (Part object, relevance_score)
        """
        rid = request_id or get_request_id()

        db_config = DatabaseConfig()
        session_provided = session is not None
        if not session_provided:
            session = db_config.get_main_session()

        debug_id(f"Starting FTS search for: '{search_text}'", rid)

        try:
            ts_query = func.plainto_tsquery('english', search_text)
            rank = func.ts_rank(cls.search_vector, ts_query)

            results = session.query(cls, rank.label('relevance')) \
                .filter(cls.search_vector.op('@@')(ts_query)) \
                .order_by(rank.desc()) \
                .limit(limit) \
                .all()

            debug_id(f"FTS search found {len(results)} results", rid)
            return [(result[0], float(result[1])) for result in results]

        except Exception as e:
            error_id(f"Error in Part.fts_search: {str(e)}", rid)
            raise
        finally:
            if not session_provided:
                session.close()

    @classmethod
    @with_request_id
    def get_by_id(cls, part_id: int, request_id: Optional[str] = None, session: Optional[Session] = None) -> Optional[
        'Part']:
        """
        Get a part by its ID.

        Args:
            part_id: ID of the part to retrieve
            request_id: Optional request ID for tracking this operation in logs
            session: Optional SQLAlchemy session. If None, a new session will be created

        Returns:
            Part object if found, None otherwise
        """
        # Get or use the provided request_id
        rid = request_id or get_request_id()

        # Get a database session if one wasn't provided
        db_config = DatabaseConfig()
        session_provided = session is not None
        if not session_provided:
            session = db_config.get_main_session()
            debug_id(f"Created new database session for Part.get_by_id", rid)

        debug_id(f"Getting part with ID: {part_id}", rid)
        try:
            part = session.query(cls).filter(cls.id == part_id).first()
            if part:
                debug_id(f"Found part: {part.part_number} (ID: {part_id})", rid)
            else:
                debug_id(f"No part found with ID: {part_id}", rid)
            return part
        except Exception as e:
            error_id(f"Error retrieving part with ID {part_id}: {str(e)}", rid)
            return None
        finally:
            # Close the session if we created it
            if not session_provided:
                session.close()
                debug_id(f"Closed database session for Part.get_by_id", rid)

class Image(Base):
    __tablename__ = 'image'

    id = Column(Integer, primary_key=True)
    title = Column(String, nullable=False)
    description = Column(String, nullable=False)
    file_path = Column(String, nullable=False)
    img_metadata = Column(JSON)

    parts_position_image = relationship("PartsPositionImageAssociation", back_populates="image")
    image_problem = relationship("ImageProblemAssociation", back_populates="image")
    image_task = relationship("ImageTaskAssociation", back_populates="image")
    image_completed_document_association = relationship("ImageCompletedDocumentAssociation", back_populates="image")
    image_embedding = relationship("ImageEmbedding", back_populates="image")
    image_position_association = relationship("ImagePositionAssociation", back_populates="image")
    tool_image_association = relationship("ToolImageAssociation", back_populates="image", cascade="all, delete-orphan")

    @classmethod
    @with_request_id
    def add_to_db(cls, session, title, file_path, description, position_id=None, complete_document_id=None,
                  metadata=None,
                  request_id=None):
        import shutil
        import os
        from modules.database_manager.db_manager import PostgreSQLDatabaseManager
        from modules.configuration.config import DATABASE_PATH_IMAGES_FOLDER, DATABASE_DIR
        try:
            if session is None:
                db_config = DatabaseConfig()
                session = db_config.get_main_session().__enter__()

            # Create a database manager instance for committing
            db_manager = PostgreSQLDatabaseManager(session=session, request_id=request_id)

            # Create a unique destination path by combining the directory and a unique filename
            original_filename = os.path.basename(file_path)
            base_name, ext = os.path.splitext(original_filename)
            destination_filename = f"{title.replace(' ', '_')}_{base_name}{ext}"  # Use title to make filename meaningful

            # FIXED: Create both absolute path (for file operations) and relative path (for database)
            destination_absolute_path = os.path.join(DATABASE_PATH_IMAGES_FOLDER, destination_filename)
            destination_relative_path = os.path.join("DB_IMAGES",
                                                     destination_filename)  # Store relative path like before

            # Ensure the destination directory exists
            os.makedirs(DATABASE_PATH_IMAGES_FOLDER, exist_ok=True)

            shutil.copy(file_path, destination_absolute_path)
            debug_id(f"Copied '{file_path}' -> '{destination_absolute_path}'", request_id)

            # FIXED: Store relative path in database (like the old system)
            image = cls(
                title=title,
                description=description,
                file_path=destination_relative_path,  #  Store relative path like: DB_IMAGES\filename.png
                img_metadata=metadata if metadata else {}
            )
            session.add(image)
            session.flush()

            debug_id(f"Added image to session: {title}, id={image.id}, stored path: {destination_relative_path}",
                     request_id)

            # Conditionally handle embedding generation and associations if associated with a document OR position
            if complete_document_id is not None or position_id is not None:
                debug_id(f"Generating embedding for image {image.id}", request_id)

                # Add this code in Image.add_to_db() method after session.flush() and before the commit

                # Create position association if position_id is provided
                if position_id is not None:
                    # Check if association already exists
                    existing_association = session.query(ImagePositionAssociation).filter(
                        and_(ImagePositionAssociation.image_id == image.id,
                             ImagePositionAssociation.position_id == position_id)
                    ).first()

                    if existing_association is None:
                        # Create the association
                        image_position_association = ImagePositionAssociation(
                            image_id=image.id,
                            position_id=position_id
                        )
                        session.add(image_position_association)
                        debug_id(f"Added position association for image {image.id} with position_id {position_id}",
                                 request_id)
                    else:
                        debug_id(
                            f"Position association already exists for image {image.id} with position_id {position_id}",
                            request_id)

                # Only create document association if complete_document_id is provided
                if complete_document_id is not None:
                    debug_id(
                        f"Added enhanced document association for image {image.id} with complete_document_id {complete_document_id}",
                        request_id)
            else:
                debug_id(f"Skipping embedding and associations for standalone image {image.id}", request_id)

            # Use the database manager's commit_with_retry
            if not db_manager.commit_with_retry():
                raise Exception("Failed to commit image to database")

            debug_id(f"Committed image to database: {title}, id={image.id}", request_id)
            return image.id
        except Exception as e:
            error_id(f"Failed to add image to database: {e}", request_id)
            if session is None:
                session.__exit__(None, None, None)
            return None

    # =============================================================================
    # Also fix the serve_file method to handle relative paths correctly
    # =============================================================================

    @classmethod
    @with_request_id
    def _add_to_db_internal(cls, session, title, file_path, description, position_id=None, complete_document_id=None,
                            metadata=None, request_id=None):
        """
        Updated internal method with pgvector embedding support.
        """
        try:
            db_config = DatabaseConfig()
            if hasattr(db_config, 'is_postgresql') and db_config.is_postgresql:
                debug_id("Using PostgreSQL database with pgvector support", request_id)
            else:
                debug_id("Using SQLite database", request_id)
                try:
                    session.execute(text("PRAGMA busy_timeout = 30000"))
                except Exception as e:
                    warning_id(f"Could not set SQLite busy timeout: {e}", request_id)

            # Check for existing image
            existing_image = session.query(cls).filter(
                and_(cls.title == title, cls.description == description)
            ).first()

            if existing_image is not None and existing_image.file_path == file_path:
                info_id(f"Image already exists: {title}", request_id)
                return existing_image

            # Copy file to destination
            _, ext = os.path.splitext(file_path)
            dest_name = f"{title}{ext}"
            dest_rel = os.path.join("DB_IMAGES", dest_name)
            dest_abs = os.path.join(DATABASE_DIR, "DB_IMAGES", dest_name)
            os.makedirs(os.path.dirname(dest_abs), exist_ok=True)
            shutil.copy2(file_path, dest_abs)
            debug_id(f"Copied '{file_path}' -> '{dest_abs}'", request_id)

            # Create image record with enhanced metadata support
            img_metadata = metadata or {}
            new_image = cls(
                title=title,
                description=description,
                file_path=dest_rel,
                img_metadata=img_metadata
            )
            session.add(new_image)
            session.flush()
            debug_id(f"Added image to session: {title}, id={new_image.id}", request_id)

            # Generate embedding with pgvector support
            try:
                model_handler = ModelsConfig.load_image_model()
                info_id(f"Using model handler: {type(model_handler).__name__}", request_id)
                absolute_file_path = os.path.join(DATABASE_DIR, new_image.file_path)
                image = PILImage.open(absolute_file_path).convert("RGB")

                if model_handler.is_valid_image(image):
                    model_embedding = model_handler.get_image_embedding(image)
                    model_name = type(model_handler).__name__

                    if model_name and model_embedding is not None:
                        # Check for existing embedding
                        existing_embedding = session.query(ImageEmbedding).filter(
                            and_(ImageEmbedding.image_id == new_image.id, ImageEmbedding.model_name == model_name)
                        ).first()

                        if existing_embedding is None:
                            # Convert numpy array to list
                            if hasattr(model_embedding, 'tolist'):
                                embedding_list = model_embedding.tolist()
                            elif isinstance(model_embedding, np.ndarray):
                                embedding_list = model_embedding.flatten().tolist()
                            else:
                                embedding_list = list(model_embedding)

                            # Try pgvector first, fallback to legacy
                            try:
                                image_embedding = ImageEmbedding.create_with_pgvector(
                                    image_id=new_image.id,
                                    model_name=model_name,
                                    embedding=embedding_list
                                )
                                session.add(image_embedding)
                                debug_id(f"Added pgvector image embedding for {title}", request_id)
                            except Exception as pgvector_error:
                                warning_id(f"pgvector creation failed, using legacy: {pgvector_error}", request_id)
                                image_embedding = ImageEmbedding.create_with_legacy(
                                    image_id=new_image.id,
                                    model_name=model_name,
                                    embedding=embedding_list
                                )
                                session.add(image_embedding)
                                debug_id(f"Added legacy image embedding for {title}", request_id)
                else:
                    info_id(f"Image {title} not valid for embedding", request_id)
            except Exception as e:
                warning_id(f"Embedding failed for {title}: {e}", request_id)

            # Create position association (existing logic)
            if position_id:
                existing_association = session.query(ImagePositionAssociation).filter(
                    and_(ImagePositionAssociation.image_id == new_image.id,
                         ImagePositionAssociation.position_id == position_id)
                ).first()
                if existing_association is None:
                    image_position_association = ImagePositionAssociation(
                        image_id=new_image.id,
                        position_id=position_id
                    )
                    session.add(image_position_association)
                    debug_id(f"Added position association for {title}", request_id)

            # Create structure-guided document association
            if complete_document_id:
                cls._create_enhanced_document_association(
                    session, new_image.id, complete_document_id, metadata, request_id
                )

            cls.commit_with_retry(session, retries=5, delay=1, request_id=request_id)
            debug_id(f"Committed image to database: {title}, id={new_image.id}", request_id)
            return new_image

        except Exception as e:
            error_id(f"Critical error in _add_to_db_internal: {e}", request_id, exc_info=True)
            try:
                session.rollback()
            except:
                pass
            raise

    @classmethod
    def _create_enhanced_document_association(cls, session, image_id, complete_document_id, metadata, request_id):
        """
        Create enhanced ImageCompletedDocumentAssociation with structure-guided data.
        """
        try:
            # Check for existing association
            existing_association = session.query(ImageCompletedDocumentAssociation).filter(
                and_(ImageCompletedDocumentAssociation.image_id == image_id,
                     ImageCompletedDocumentAssociation.complete_document_id == complete_document_id)
            ).first()

            if existing_association is None:
                # Extract structure-guided metadata
                structure_metadata = metadata or {}

                # Create enhanced association with structure-guided fields
                association = ImageCompletedDocumentAssociation(
                    image_id=image_id,
                    complete_document_id=complete_document_id,
                    # NEW: Structure-guided fields
                    page_number=structure_metadata.get('page_number'),
                    chunk_index=structure_metadata.get('image_index', 0),
                    association_method=structure_metadata.get('association_method', 'structure_guided'),
                    confidence_score=structure_metadata.get('confidence_score', 0.8),
                    context_metadata=json.dumps({
                        'structure_guided': structure_metadata.get('structure_guided', True),
                        'content_type': structure_metadata.get('content_type', 'image'),
                        'bbox': structure_metadata.get('bbox'),
                        'estimated_size': structure_metadata.get('estimated_size'),
                        'created_at': datetime.now().isoformat(),
                        'processing_method': structure_metadata.get('processing_method', 'enhanced_add_to_db')
                    })
                )

                session.add(association)
                debug_id(f"Added enhanced document association for image {image_id}", request_id)
            else:
                debug_id(f"Document association already exists for image {image_id}", request_id)

        except Exception as e:
            warning_id(f"Failed to create enhanced document association: {e}", request_id)
            # Fallback to basic association
            try:
                basic_association = ImageCompletedDocumentAssociation(
                    image_id=image_id,
                    complete_document_id=complete_document_id
                )
                session.add(basic_association)
                debug_id(f"Added basic document association for image {image_id}", request_id)
            except Exception as fallback_error:
                error_id(f"Failed to create even basic association: {fallback_error}", request_id)

    @classmethod
    @with_request_id
    def create_with_tool_association(cls, session, title, file_path, tool, description="", request_id=None):
        """
        Enhanced version with database compatibility.
        """
        rid = request_id or get_request_id()

        # Create the image using the enhanced add_to_db
        new_image = cls.add_to_db(session, title, file_path, description, request_id=rid)

        # Check for existing association
        existing_assoc = session.query(ToolImageAssociation).filter(
            and_(ToolImageAssociation.tool_id == tool.id,
                 ToolImageAssociation.image_id == new_image.id)
        ).first()

        if existing_assoc:
            info_id(f"Tool-image association already exists for tool ID {tool.id} and image ID {new_image.id}", rid)
            return new_image, existing_assoc

        # Create new association
        tool_image_assoc = ToolImageAssociation(
            tool_id=tool.id,
            image_id=new_image.id,
            description="Primary uploaded tool image"
        )
        session.add(tool_image_assoc)

        # Enhanced commit with retry
        cls.commit_with_retry(session, retries=3, delay=1, request_id=rid)

        return new_image, tool_image_assoc

    @with_request_id
    def generate_embedding(self, session, model_handler, request_id=None):
        """
        Updated embedding generation with pgvector support.
        """
        rid = request_id or get_request_id()

        try:
            # Convert the relative file path back to an absolute path if needed
            if os.path.isabs(self.file_path):
                absolute_file_path = self.file_path
            else:
                absolute_file_path = os.path.join(DATABASE_DIR, self.file_path)

            info_id(f"Opening image: {absolute_file_path}", rid)

            # Open the image using the absolute file path
            image = PILImage.open(absolute_file_path).convert("RGB")

            info_id("Checking if image is valid for the model", rid)
            if not model_handler.is_valid_image(image):
                info_id(f"Image does not meet the required dimensions or aspect ratio.", rid)
                return False

            info_id("Generating image embedding", rid)
            model_embedding = model_handler.get_image_embedding(image)
            model_name = model_handler.__class__.__name__

            if model_embedding is None:
                error_id("Failed to generate embedding", rid)
                return False

            # Check if the embedding already exists
            existing_embedding = session.query(ImageEmbedding).filter(
                and_(ImageEmbedding.image_id == self.id, ImageEmbedding.model_name == model_name)
            ).first()

            if existing_embedding is None:
                # Create new embedding with pgvector support
                info_id(f"Creating a new ImageEmbedding entry for image ID {self.id}", rid)

                # Convert numpy array to list
                if hasattr(model_embedding, 'tolist'):
                    embedding_list = model_embedding.tolist()
                elif isinstance(model_embedding, np.ndarray):
                    embedding_list = model_embedding.flatten().tolist()
                else:
                    embedding_list = list(model_embedding)

                # Try pgvector first, fallback to legacy
                try:
                    image_embedding = ImageEmbedding.create_with_pgvector(
                        image_id=self.id,
                        model_name=model_name,
                        embedding=embedding_list
                    )
                    session.add(image_embedding)
                    info_id(f"Created pgvector embedding for image ID {self.id}", rid)
                except Exception as pgvector_error:
                    warning_id(f"pgvector creation failed, using legacy: {pgvector_error}", rid)
                    image_embedding = ImageEmbedding.create_with_legacy(
                        image_id=self.id,
                        model_name=model_name,
                        embedding=embedding_list
                    )
                    session.add(image_embedding)
                    info_id(f"Created legacy embedding for image ID {self.id}", rid)

                # Enhanced commit with retry
                Image.commit_with_retry(session, retries=3, delay=1, request_id=rid)
            else:
                # Optionally migrate existing embedding to pgvector
                if existing_embedding.get_storage_type() == 'legacy':
                    info_id(f"Migrating existing embedding to pgvector for image ID {self.id}", rid)
                    if existing_embedding.migrate_to_pgvector():
                        session.add(existing_embedding)
                        Image.commit_with_retry(session, retries=3, delay=1, request_id=rid)
                        info_id(f"Successfully migrated embedding to pgvector", rid)
                    else:
                        warning_id(f"Failed to migrate embedding to pgvector", rid)

            return True

        except Exception as e:
            error_id(f"Error generating embedding: {e}", rid, exc_info=True)
            return False

    # =====================================================
    # NEW: STRUCTURE-GUIDED QUERY METHODS
    # =====================================================

    @classmethod
    @with_request_id
    def get_images_with_chunk_context(cls, session, complete_document_id, request_id=None):
        """
        Get all images for a document with their associated chunk context.
        This method supports the new structure-guided associations.
        """
        try:
            from modules.emtacdb.emtacdb_fts import Document

            # Query images with their associations and chunk context
            result = session.query(
                cls.id.label('image_id'),
                cls.title.label('image_title'),
                cls.file_path.label('image_path'),
                cls.description.label('image_description'),
                cls.img_metadata.label('image_metadata'),
                Document.id.label('chunk_id'),
                Document.name.label('chunk_name'),
                Document.content.label('chunk_content'),
                ImageCompletedDocumentAssociation.page_number,
                ImageCompletedDocumentAssociation.chunk_index,
                ImageCompletedDocumentAssociation.confidence_score,
                ImageCompletedDocumentAssociation.association_method,
                ImageCompletedDocumentAssociation.context_metadata
            ).select_from(cls).join(
                ImageCompletedDocumentAssociation,
                cls.id == ImageCompletedDocumentAssociation.image_id
            ).outerjoin(
                Document,
                ImageCompletedDocumentAssociation.document_id == Document.id
            ).filter(
                ImageCompletedDocumentAssociation.complete_document_id == complete_document_id
            ).order_by(
                ImageCompletedDocumentAssociation.page_number,
                ImageCompletedDocumentAssociation.chunk_index
            ).all()

            images_with_context = []
            for row in result:
                # Parse context metadata
                context_metadata = {}
                if row.context_metadata:
                    try:
                        context_metadata = json.loads(row.context_metadata)
                    except:
                        pass

                image_context = {
                    'image_id': row.image_id,
                    'image_title': row.image_title,
                    'image_path': row.image_path,
                    'image_description': row.image_description,
                    'image_metadata': row.image_metadata,
                    'chunk_id': row.chunk_id,
                    'chunk_name': row.chunk_name,
                    'chunk_content': row.chunk_content,
                    'page_number': row.page_number,
                    'chunk_index': row.chunk_index,
                    'confidence_score': row.confidence_score,
                    'association_method': row.association_method,
                    'context_metadata': context_metadata,
                    'view_url': f'/add_document/image/{row.image_id}',
                    'has_chunk_association': row.chunk_id is not None,
                    'structure_guided': context_metadata.get('structure_guided', False),
                    'content_type': context_metadata.get('content_type', 'image'),
                    'chunk_preview': row.chunk_content[:200] + "..." if row.chunk_content and len(
                        row.chunk_content) > 200 else row.chunk_content
                }
                images_with_context.append(image_context)

            info_id(f"Retrieved {len(images_with_context)} images with chunk context", request_id)
            return images_with_context

        except Exception as e:
            error_id(f"Failed to get images with chunk context: {e}", request_id)
            return []

    @classmethod
    @with_request_id
    def search_by_chunk_text(cls, session, search_text, complete_document_id=None,
                             confidence_threshold=0.5, request_id=None):
        """
        Search for images by their associated chunk text content.
        Uses the new structure-guided associations.
        """
        try:
            from modules.emtacdb.emtacdb_fts import Document, CompleteDocument

            query = session.query(
                cls.id.label('image_id'),
                cls.title.label('image_title'),
                cls.file_path.label('image_path'),
                cls.description.label('image_description'),
                Document.content.label('chunk_content'),
                Document.name.label('chunk_name'),
                CompleteDocument.title.label('document_title'),
                ImageCompletedDocumentAssociation.confidence_score,
                ImageCompletedDocumentAssociation.page_number,
                ImageCompletedDocumentAssociation.association_method
            ).select_from(cls).join(
                ImageCompletedDocumentAssociation,
                cls.id == ImageCompletedDocumentAssociation.image_id
            ).join(
                Document,
                ImageCompletedDocumentAssociation.document_id == Document.id
            ).join(
                CompleteDocument,
                ImageCompletedDocumentAssociation.complete_document_id == CompleteDocument.id
            ).filter(
                Document.content.ilike(f"%{search_text}%")
            )

            # Optional filters
            if complete_document_id:
                query = query.filter(CompleteDocument.id == complete_document_id)

            if confidence_threshold:
                query = query.filter(ImageCompletedDocumentAssociation.confidence_score >= confidence_threshold)

            # Order by confidence and relevance
            query = query.order_by(
                ImageCompletedDocumentAssociation.confidence_score.desc(),
                ImageCompletedDocumentAssociation.page_number
            )

            results = query.all()

            search_results = []
            for row in results:
                # Highlight the search term in chunk content
                highlighted_content = cls._highlight_search_term(row.chunk_content, search_text)

                search_results.append({
                    'image_id': row.image_id,
                    'image_title': row.image_title,
                    'image_path': row.image_path,
                    'image_description': row.image_description,
                    'chunk_content': row.chunk_content,
                    'chunk_name': row.chunk_name,
                    'document_title': row.document_title,
                    'confidence': row.confidence_score,
                    'page_number': row.page_number,
                    'association_method': row.association_method,
                    'highlighted_content': highlighted_content,
                    'view_url': f'/add_document/image/{row.image_id}'
                })

            info_id(f"Found {len(search_results)} images matching '{search_text}'", request_id)
            return search_results

        except Exception as e:
            error_id(f"Failed to search images by chunk text: {e}", request_id)
            return []

    @classmethod
    @with_request_id
    def get_association_statistics(cls, session, complete_document_id=None, request_id=None):
        """
        Get statistics about image-chunk associations, especially structure-guided ones.
        Useful for monitoring and debugging the new association system.
        """
        try:
            from sqlalchemy import func

            # Base query for associations
            base_query = session.query(ImageCompletedDocumentAssociation)

            if complete_document_id:
                base_query = base_query.filter(
                    ImageCompletedDocumentAssociation.complete_document_id == complete_document_id
                )

            # Total associations
            total_associations = base_query.count()

            # Structure-guided associations
            structure_guided = base_query.filter(
                ImageCompletedDocumentAssociation.association_method == 'structure_guided'
            ).count()

            # High confidence associations
            high_confidence = base_query.filter(
                ImageCompletedDocumentAssociation.confidence_score >= 0.8
            ).count()

            # Average confidence score
            avg_confidence = session.query(
                func.avg(ImageCompletedDocumentAssociation.confidence_score)
            ).filter(
                ImageCompletedDocumentAssociation.complete_document_id == complete_document_id
                if complete_document_id else True
            ).scalar() or 0

            # Associations by method
            method_stats = session.query(
                ImageCompletedDocumentAssociation.association_method,
                func.count(ImageCompletedDocumentAssociation.id)
            ).filter(
                ImageCompletedDocumentAssociation.complete_document_id == complete_document_id
                if complete_document_id else True
            ).group_by(
                ImageCompletedDocumentAssociation.association_method
            ).all()

            # Associations by page
            page_stats = session.query(
                ImageCompletedDocumentAssociation.page_number,
                func.count(ImageCompletedDocumentAssociation.id)
            ).filter(
                ImageCompletedDocumentAssociation.complete_document_id == complete_document_id
                if complete_document_id else True
            ).group_by(
                ImageCompletedDocumentAssociation.page_number
            ).order_by(
                ImageCompletedDocumentAssociation.page_number
            ).all()

            stats = {
                'total_associations': total_associations,
                'structure_guided_count': structure_guided,
                'high_confidence_count': high_confidence,
                'average_confidence': float(avg_confidence),
                'structure_guided_percentage': (
                            structure_guided / total_associations * 100) if total_associations > 0 else 0,
                'high_confidence_percentage': (
                            high_confidence / total_associations * 100) if total_associations > 0 else 0,
                'associations_by_method': dict(method_stats),
                'associations_by_page': dict(page_stats)
            }

            info_id(f"Association statistics: {stats['total_associations']} total, "
                    f"{stats['structure_guided_percentage']:.1f}% structure-guided, "
                    f"{stats['high_confidence_percentage']:.1f}% high-confidence", request_id)

            return stats

        except Exception as e:
            error_id(f"Failed to get association statistics: {e}", request_id)
            return {
                'total_associations': 0,
                'structure_guided_count': 0,
                'high_confidence_count': 0,
                'average_confidence': 0,
                'structure_guided_percentage': 0,
                'high_confidence_percentage': 0,
                'associations_by_method': {},
                'associations_by_page': {}
            }

    @classmethod
    def _highlight_search_term(cls, content, search_term):
        """Simple text highlighting for search results."""
        if not content or not search_term:
            return content

        import re
        # Case-insensitive highlighting
        pattern = re.compile(re.escape(search_term), re.IGNORECASE)
        highlighted = pattern.sub(f"<mark>{search_term}</mark>", content)
        return highlighted

    # =====================================================
    # DATABASE-AGNOSTIC HELPER METHODS
    # =====================================================

    @classmethod
    @with_request_id
    def commit_with_retry(cls, session, retries=3, delay=0.5, request_id=None):
        """
        Database-agnostic commit with retry logic.
        """
        db_config = DatabaseConfig()

        for attempt in range(retries):
            try:
                session.commit()
                if attempt > 0:
                    debug_id(f"Commit succeeded on attempt {attempt + 1}", request_id)
                return True

            except Exception as e:
                error_msg = str(e).lower()

                # Handle different database-specific errors
                if "pending rollback" in error_msg:
                    warning_id(f"Session rolled back, clearing state for retry {attempt + 1}/{retries}", request_id)
                    try:
                        session.rollback()
                    except:
                        pass

                    if attempt < retries - 1:
                        time.sleep(delay)
                        delay = min(delay * 1.2, 2)
                    else:
                        error_id(f"Session rollback error after {retries} retries", request_id)
                        raise RuntimeError(f"Session in invalid state after {retries} retries")

                elif ("database is locked" in error_msg or "locked" in error_msg or
                      "deadlock" in error_msg or "timeout" in error_msg):
                    warning_id(f"Database contention, retry {attempt + 1}/{retries} in {delay}s", request_id)

                    try:
                        session.rollback()
                    except:
                        pass

                    if attempt < retries - 1:
                        time.sleep(delay)
                        delay = min(delay * 1.2, 2)
                    else:
                        error_id(f"Database contention after {retries} retries", request_id)
                        raise RuntimeError(f"Database contention after {retries} retries")
                else:
                    error_id(f"Non-retryable database error on attempt {attempt + 1}: {e}", request_id)
                    try:
                        session.rollback()
                    except:
                        pass
                    raise

        return False

    @classmethod
    @with_request_id
    def monitor_processing_session(cls, session, operation_name="image_operation", request_id=None):
        """
        Database-agnostic session monitor.
        """
        import time

        class SessionMonitor:
            def __init__(self, session, operation_name, request_id):
                self.session = session
                self.operation_name = operation_name
                self.request_id = request_id
                self.start_time = None

            def __enter__(self):
                self.start_time = time.time()
                info_id(f"Starting monitored operation: {self.operation_name}", self.request_id)

                try:
                    db_config = DatabaseConfig()
                    if hasattr(db_config, 'is_postgresql') and db_config.is_postgresql:
                        # PostgreSQL - no PRAGMA commands needed
                        debug_id("Using PostgreSQL for monitored session", self.request_id)
                    else:
                        # SQLite - set busy timeout
                        self.session.execute(text("PRAGMA busy_timeout = 30000"))
                        debug_id("Set SQLite busy timeout for monitored session", self.request_id)
                except Exception as e:
                    warning_id(f"Could not configure session timeout: {e}", self.request_id)

                return self.session

            def __exit__(self, exc_type, exc_val, exc_tb):
                elapsed = time.time() - self.start_time

                if exc_type is None:
                    info_id(f"Completed monitored operation: {self.operation_name} in {elapsed:.2f}s", self.request_id)
                else:
                    error_id(f"Failed monitored operation: {self.operation_name} after {elapsed:.2f}s: {exc_val}",
                             self.request_id)

                    # Database-agnostic error detection
                    error_str = str(exc_val).lower()
                    if "database is locked" in error_str or "timeout" in error_str or "deadlock" in error_str:
                        warning_id(f"Database contention during {self.operation_name}", self.request_id)

        return SessionMonitor(session, operation_name, request_id)

    @classmethod
    @with_request_id
    def serve_file(cls, image_id, request_id=None):
        """
        Enhanced serve_file method with proper relative path handling.
        Fixed to work with your specific path configuration.
        """
        from flask import send_file
        from modules.configuration.config import DATABASE_DIR
        from modules.configuration.log_config import info_id, debug_id, error_id

        rid = request_id or get_request_id()
        info_id(f"Attempting to retrieve image with ID: {image_id}", rid)

        db_config = DatabaseConfig()
        try:
            with cls.monitor_processing_session(
                    db_config.get_main_session(),
                    "serve_image_file",
                    rid
            ) as session:

                image = session.query(cls).filter_by(id=image_id).first()

                if not image:
                    error_id(f"No image found in database with ID: {image_id}", rid)
                    return False, "Image not found", 404

                debug_id(f"Found image: {image.title}, stored path: {image.file_path}", rid)

                # FIXED: Handle your specific path structure correctly
                if os.path.isabs(image.file_path):
                    # If it's already an absolute path, use it directly
                    file_path = image.file_path
                    debug_id(f"Using absolute path: {file_path}", rid)
                else:
                    # For relative paths like "DB_IMAGES\filename.png"
                    # Join with DATABASE_DIR to get the full path
                    file_path = os.path.join(DATABASE_DIR, image.file_path)
                    debug_id(f"Constructed path from relative: {DATABASE_DIR} + {image.file_path} = {file_path}", rid)

                # Normalize the path to handle both forward and backward slashes
                file_path = os.path.normpath(file_path)
                debug_id(f"Normalized path: {file_path}", rid)

                if not os.path.exists(file_path):
                    error_id(f"File not found on disk: {file_path}", rid)

                    # Additional debugging: check if the DB_IMAGES directory exists
                    db_images_dir = os.path.join(DATABASE_DIR, 'DB_IMAGES')
                    debug_id(f"DB_IMAGES directory exists: {os.path.exists(db_images_dir)}", rid)
                    if os.path.exists(db_images_dir):
                        try:
                            files_in_dir = os.listdir(db_images_dir)[:5]  # Show first 5 files
                            debug_id(f"Sample files in DB_IMAGES: {files_in_dir}", rid)
                        except Exception as list_error:
                            debug_id(f"Error listing DB_IMAGES directory: {list_error}", rid)

                    # Try alternative path constructions as fallback
                    alternative_paths = [
                        # Try just the filename in DB_IMAGES
                        os.path.join(DATABASE_DIR, 'DB_IMAGES', os.path.basename(image.file_path)),
                        # Try normalized path with forward slashes
                        os.path.join(DATABASE_DIR, image.file_path.replace('\\', '/')),
                        # Try normalized path with backslashes
                        os.path.join(DATABASE_DIR, image.file_path.replace('/', '\\')),
                    ]

                    for alt_path in alternative_paths:
                        alt_path = os.path.normpath(alt_path)
                        debug_id(f"Trying alternative path: {alt_path}", rid)
                        if os.path.exists(alt_path):
                            file_path = alt_path
                            info_id(f"Found file at alternative path: {alt_path}", rid)
                            break
                    else:
                        # If none of the alternatives worked
                        return False, "Image file not found", 404

                info_id(f"Serving file: {file_path}", rid)

                # Determine mimetype based on file extension
                _, ext = os.path.splitext(file_path)  # FIXED: syntax error here
                mimetype_map = {
                    '.jpg': 'image/jpeg',
                    '.jpeg': 'image/jpeg',
                    '.png': 'image/png',
                    '.gif': 'image/gif',
                    '.bmp': 'image/bmp',
                    '.webp': 'image/webp',
                    '.tiff': 'image/tiff',
                    '.tif': 'image/tiff',
                    '.svg': 'image/svg+xml'
                }
                mimetype = mimetype_map.get(ext.lower(), 'image/jpeg')
                debug_id(f"Using mimetype: {mimetype} for extension: {ext}", rid)

                response = send_file(file_path, mimetype=mimetype, as_attachment=False)
                info_id(f"Successfully served image {image_id}", rid)
                return True, response, 200

        except Exception as e:
            error_id(f"Unhandled error while serving image with ID {image_id}: {e}", rid, exc_info=True)
            return False, "Internal Server Error", 500

    # =====================================================
    # COMPREHENSIVE SEARCH METHODS
    # =====================================================

    @classmethod
    @with_request_id
    def search_images(cls, session, title=None, description=None, position_id=None,
                      tool_id=None, task_id=None, problem_id=None, completed_document_id=None,
                      area_id=None, equipment_group_id=None, model_id=None, asset_number_id=None,
                      location_id=None, subassembly_id=None, component_assembly_id=None,
                      assembly_view_id=None, site_location_id=None,
                      # NEW: pgvector parameters
                      similarity_query_embedding=None, similarity_threshold=0.7,
                      embedding_model_name="CLIPModelHandler", use_hybrid_ranking=True,
                      enable_similarity_boost=True,
                      limit=50, request_id=None):
        """
        Enhanced dynamic search method for Images with comprehensive filtering capabilities.
        UPDATED: Now includes pgvector similarity search alongside existing association searches.

        New Parameters:
            similarity_query_embedding: Optional embedding vector for similarity search
            similarity_threshold: Minimum similarity score (0.0-1.0) for pgvector results
            embedding_model_name: Model name for embedding-based search
            use_hybrid_ranking: Combine similarity scores with traditional relevance
            enable_similarity_boost: Boost ranking for images with high similarity scores

        Returns:
            List of image dictionaries, optionally enhanced with similarity scores
        """
        from modules.configuration.log_config import info_id, debug_id, error_id, warning_id

        rid = request_id or get_request_id()

        info_id("========== ENHANCED DYNAMIC IMAGE SEARCH WITH PGVECTOR ==========", rid)
        debug_id(f"Search parameters: title='{title}', description='{description}', "
                 f"position_id={position_id}, tool_id={tool_id}, task_id={task_id}, "
                 f"problem_id={problem_id}, completed_document_id={completed_document_id}, "
                 f"area_id={area_id}, equipment_group_id={equipment_group_id}, "
                 f"model_id={model_id}, asset_number_id={asset_number_id}, "
                 f"location_id={location_id}, subassembly_id={subassembly_id}, "
                 f"component_assembly_id={component_assembly_id}, assembly_view_id={assembly_view_id}, "
                 f"site_location_id={site_location_id}, "
                 f"similarity_query={'provided' if similarity_query_embedding else 'none'}, "
                 f"use_hybrid_ranking={use_hybrid_ranking}, limit={limit}", rid)

        try:
            # Collect all image IDs from different search paths
            all_image_ids = set()
            similarity_scores = {}  # Track similarity scores for ranking
            search_path_sources = {}  # Track which search path found each image

            # ======================================================================
            # PHASE 1: PGVECTOR SIMILARITY SEARCH (NEW!)
            # ======================================================================
            if similarity_query_embedding is not None:
                debug_id("=== PHASE 1: pgvector similarity search ===", rid)
                try:
                    # Use ImageEmbedding's search method for similarity
                    similar_results = ImageEmbedding.search_similar_images(
                        session=session,
                        query_embedding=similarity_query_embedding,
                        model_name=embedding_model_name,
                        limit=limit * 3,  # Get more candidates for filtering
                        similarity_threshold=similarity_threshold
                    )

                    similarity_image_ids = {result['image_id'] for result in similar_results}
                    all_image_ids.update(similarity_image_ids)

                    # Store similarity scores and source tracking
                    for result in similar_results:
                        image_id = result['image_id']
                        similarity_scores[image_id] = result['similarity']
                        search_path_sources[image_id] = search_path_sources.get(image_id, []) + ['similarity']

                    debug_id(f"Found {len(similarity_image_ids)} images via pgvector similarity search", rid)

                except Exception as e:
                    error_id(f"pgvector similarity search failed: {e}", rid)
                    # Continue with traditional search methods

            # ======================================================================
            # PHASE 2: TRADITIONAL SEARCH PATHS (Your existing logic)
            # ======================================================================

            # Text-based search conditions
            text_conditions = []
            if title:
                text_conditions.append(cls.title.ilike(f"%{title}%"))
                debug_id(f"Added title filter: '{title}'", rid)
            if description:
                text_conditions.append(cls.description.ilike(f"%{description}%"))
                debug_id(f"Added description filter: '{description}'", rid)

            # SEARCH PATH 1: Direct position associations
            if position_id or any([area_id, equipment_group_id, model_id, asset_number_id, location_id,
                                   subassembly_id, component_assembly_id, assembly_view_id, site_location_id]):

                debug_id("=== SEARCH PATH 1: Direct position associations ===", rid)

                # Build query for direct position associations
                direct_query = session.query(cls.id).select_from(cls)

                # Apply text filters
                if text_conditions:
                    from sqlalchemy import and_
                    direct_query = direct_query.filter(and_(*text_conditions))

                # Join with position associations
                direct_query = direct_query.join(ImagePositionAssociation, cls.id == ImagePositionAssociation.image_id)
                direct_query = direct_query.join(Position, ImagePositionAssociation.position_id == Position.id)

                # Apply hierarchy filters
                hierarchy_conditions = []
                if position_id:
                    hierarchy_conditions.append(ImagePositionAssociation.position_id == position_id)
                    debug_id(f"Added direct position_id filter: {position_id}", rid)
                if area_id:
                    hierarchy_conditions.append(Position.area_id == area_id)
                    debug_id(f"Added direct area_id filter: {area_id}", rid)
                if equipment_group_id:
                    hierarchy_conditions.append(Position.equipment_group_id == equipment_group_id)
                    debug_id(f"Added direct equipment_group_id filter: {equipment_group_id}", rid)
                if model_id:
                    hierarchy_conditions.append(Position.model_id == model_id)
                    debug_id(f"Added direct model_id filter: {model_id}", rid)
                if asset_number_id:
                    hierarchy_conditions.append(Position.asset_number_id == asset_number_id)
                    debug_id(f"Added direct asset_number_id filter: {asset_number_id}", rid)
                if location_id:
                    hierarchy_conditions.append(Position.location_id == location_id)
                    debug_id(f"Added direct location_id filter: {location_id}", rid)
                if subassembly_id:
                    hierarchy_conditions.append(Position.subassembly_id == subassembly_id)
                    debug_id(f"Added direct subassembly_id filter: {subassembly_id}", rid)
                if component_assembly_id:
                    hierarchy_conditions.append(Position.component_assembly_id == component_assembly_id)
                    debug_id(f"Added direct component_assembly_id filter: {component_assembly_id}", rid)
                if assembly_view_id:
                    hierarchy_conditions.append(Position.assembly_view_id == assembly_view_id)
                    debug_id(f"Added direct assembly_view_id filter: {assembly_view_id}", rid)
                if site_location_id:
                    hierarchy_conditions.append(Position.site_location_id == site_location_id)
                    debug_id(f"Added direct site_location_id filter: {site_location_id}", rid)

                if hierarchy_conditions:
                    from sqlalchemy import and_
                    direct_query = direct_query.filter(and_(*hierarchy_conditions))

                # Execute direct query and collect IDs
                direct_results = direct_query.distinct().all()
                direct_ids = {result.id for result in direct_results}
                all_image_ids.update(direct_ids)

                # Track search source
                for img_id in direct_ids:
                    search_path_sources[img_id] = search_path_sources.get(img_id, []) + ['direct_position']

                debug_id(f"Found {len(direct_ids)} images via direct position associations", rid)

            # SEARCH PATH 2: Parts Position associations
            if any([area_id, equipment_group_id, model_id, asset_number_id, location_id,
                    subassembly_id, component_assembly_id, assembly_view_id, site_location_id]):

                debug_id("=== SEARCH PATH 2: Parts position associations ===", rid)

                # Build query for parts position associations
                parts_query = session.query(cls.id).select_from(cls)

                # Apply text filters
                if text_conditions:
                    from sqlalchemy import and_
                    parts_query = parts_query.filter(and_(*text_conditions))

                # Join with parts position associations
                parts_query = parts_query.join(PartsPositionImageAssociation,
                                               cls.id == PartsPositionImageAssociation.image_id)
                parts_query = parts_query.join(Position, PartsPositionImageAssociation.position_id == Position.id)

                # Apply hierarchy filters (same as direct position associations)
                parts_hierarchy_conditions = []
                if area_id:
                    parts_hierarchy_conditions.append(Position.area_id == area_id)
                if equipment_group_id:
                    parts_hierarchy_conditions.append(Position.equipment_group_id == equipment_group_id)
                if model_id:
                    parts_hierarchy_conditions.append(Position.model_id == model_id)
                if asset_number_id:
                    parts_hierarchy_conditions.append(Position.asset_number_id == asset_number_id)
                if location_id:
                    parts_hierarchy_conditions.append(Position.location_id == location_id)
                if subassembly_id:
                    parts_hierarchy_conditions.append(Position.subassembly_id == subassembly_id)
                if component_assembly_id:
                    parts_hierarchy_conditions.append(Position.component_assembly_id == component_assembly_id)
                if assembly_view_id:
                    parts_hierarchy_conditions.append(Position.assembly_view_id == assembly_view_id)
                if site_location_id:
                    parts_hierarchy_conditions.append(Position.site_location_id == site_location_id)

                if parts_hierarchy_conditions:
                    from sqlalchemy import and_
                    parts_query = parts_query.filter(and_(*parts_hierarchy_conditions))

                # Execute parts query and collect IDs
                parts_results = parts_query.distinct().all()
                parts_ids = {result.id for result in parts_results}
                all_image_ids.update(parts_ids)

                # Track search source
                for img_id in parts_ids:
                    search_path_sources[img_id] = search_path_sources.get(img_id, []) + ['parts_position']

                debug_id(f"Found {len(parts_ids)} images via parts position associations", rid)

            # SEARCH PATH 3: Completed document associations
            if any([area_id, equipment_group_id, model_id, asset_number_id, location_id,
                    subassembly_id, component_assembly_id, assembly_view_id, site_location_id]):

                debug_id("=== SEARCH PATH 3: Completed document associations ===", rid)

                # Build query for document associations
                doc_query = session.query(cls.id).select_from(cls)

                # Apply text filters
                if text_conditions:
                    from sqlalchemy import and_
                    doc_query = doc_query.filter(and_(*text_conditions))

                # Join through document associations
                doc_query = doc_query.join(ImageCompletedDocumentAssociation,
                                           cls.id == ImageCompletedDocumentAssociation.image_id)
                doc_query = doc_query.join(CompleteDocument,
                                           ImageCompletedDocumentAssociation.complete_document_id == CompleteDocument.id)
                doc_query = doc_query.join(CompletedDocumentPositionAssociation,
                                           CompleteDocument.id == CompletedDocumentPositionAssociation.complete_document_id)
                doc_query = doc_query.join(Position,
                                           CompletedDocumentPositionAssociation.position_id == Position.id)

                # Apply hierarchy filters (same as direct path)
                doc_hierarchy_conditions = []
                if area_id:
                    doc_hierarchy_conditions.append(Position.area_id == area_id)
                if equipment_group_id:
                    doc_hierarchy_conditions.append(Position.equipment_group_id == equipment_group_id)
                if model_id:
                    doc_hierarchy_conditions.append(Position.model_id == model_id)
                if asset_number_id:
                    doc_hierarchy_conditions.append(Position.asset_number_id == asset_number_id)
                if location_id:
                    doc_hierarchy_conditions.append(Position.location_id == location_id)
                if subassembly_id:
                    doc_hierarchy_conditions.append(Position.subassembly_id == subassembly_id)
                if component_assembly_id:
                    doc_hierarchy_conditions.append(Position.component_assembly_id == component_assembly_id)
                if assembly_view_id:
                    doc_hierarchy_conditions.append(Position.assembly_view_id == assembly_view_id)
                if site_location_id:
                    doc_hierarchy_conditions.append(Position.site_location_id == site_location_id)

                if doc_hierarchy_conditions:
                    from sqlalchemy import and_
                    doc_query = doc_query.filter(and_(*doc_hierarchy_conditions))

                # Execute document query and collect IDs
                doc_results = doc_query.distinct().all()
                doc_ids = {result.id for result in doc_results}
                all_image_ids.update(doc_ids)

                # Track search source
                for img_id in doc_ids:
                    search_path_sources[img_id] = search_path_sources.get(img_id, []) + ['document_association']

                debug_id(f"Found {len(doc_ids)} images via completed document associations", rid)

            # SEARCH PATH 4: Direct association searches (tools, tasks, problems, specific documents)
            if tool_id:
                debug_id("=== SEARCH PATH 4a: Tool associations ===", rid)
                tool_query = session.query(cls.id).select_from(cls)
                if text_conditions:
                    from sqlalchemy import and_
                    tool_query = tool_query.filter(and_(*text_conditions))
                tool_query = tool_query.join(ToolImageAssociation, cls.id == ToolImageAssociation.image_id)
                tool_query = tool_query.filter(ToolImageAssociation.tool_id == tool_id)
                tool_results = tool_query.distinct().all()
                tool_ids = {result.id for result in tool_results}
                all_image_ids.update(tool_ids)

                # Track search source
                for img_id in tool_ids:
                    search_path_sources[img_id] = search_path_sources.get(img_id, []) + ['tool']

                debug_id(f"Added tool_id filter: {tool_id} - found {len(tool_ids)} images", rid)

            if task_id:
                debug_id("=== SEARCH PATH 4b: Task associations ===", rid)
                task_query = session.query(cls.id).select_from(cls)
                if text_conditions:
                    from sqlalchemy import and_
                    task_query = task_query.filter(and_(*text_conditions))
                task_query = task_query.join(ImageTaskAssociation, cls.id == ImageTaskAssociation.image_id)
                task_query = task_query.filter(ImageTaskAssociation.task_id == task_id)
                task_results = task_query.distinct().all()
                task_ids = {result.id for result in task_results}
                all_image_ids.update(task_ids)

                # Track search source
                for img_id in task_ids:
                    search_path_sources[img_id] = search_path_sources.get(img_id, []) + ['task']

                debug_id(f"Added task_id filter: {task_id} - found {len(task_ids)} images", rid)

            if problem_id:
                debug_id("=== SEARCH PATH 4c: Problem associations ===", rid)
                problem_query = session.query(cls.id).select_from(cls)
                if text_conditions:
                    from sqlalchemy import and_
                    problem_query = problem_query.filter(and_(*text_conditions))
                problem_query = problem_query.join(ImageProblemAssociation, cls.id == ImageProblemAssociation.image_id)
                problem_query = problem_query.filter(ImageProblemAssociation.problem_id == problem_id)
                problem_results = problem_query.distinct().all()
                problem_ids = {result.id for result in problem_results}
                all_image_ids.update(problem_ids)

                # Track search source
                for img_id in problem_ids:
                    search_path_sources[img_id] = search_path_sources.get(img_id, []) + ['problem']

                debug_id(f"Added problem_id filter: {problem_id} - found {len(problem_ids)} images", rid)

            if completed_document_id:
                debug_id("=== SEARCH PATH 4d: Specific document associations ===", rid)
                cdoc_query = session.query(cls.id).select_from(cls)
                if text_conditions:
                    from sqlalchemy import and_
                    cdoc_query = cdoc_query.filter(and_(*text_conditions))
                cdoc_query = cdoc_query.join(ImageCompletedDocumentAssociation,
                                             cls.id == ImageCompletedDocumentAssociation.image_id)
                cdoc_query = cdoc_query.filter(
                    ImageCompletedDocumentAssociation.complete_document_id == completed_document_id)
                cdoc_results = cdoc_query.distinct().all()
                cdoc_ids = {result.id for result in cdoc_results}
                all_image_ids.update(cdoc_ids)

                # Track search source
                for img_id in cdoc_ids:
                    search_path_sources[img_id] = search_path_sources.get(img_id, []) + ['specific_document']

                debug_id(f"Added completed_document_id filter: {completed_document_id} - found {len(cdoc_ids)} images",
                         rid)

            # SEARCH PATH 5: Text-only search (no associations)
            if text_conditions and not any([position_id, tool_id, task_id, problem_id, completed_document_id,
                                            area_id, equipment_group_id, model_id, asset_number_id, location_id,
                                            subassembly_id, component_assembly_id, assembly_view_id, site_location_id]):
                debug_id("=== SEARCH PATH 5: Text-only search ===", rid)
                from sqlalchemy import and_
                text_query = session.query(cls.id).filter(and_(*text_conditions))
                text_results = text_query.distinct().limit(limit).all()
                text_ids = {result.id for result in text_results}
                all_image_ids.update(text_ids)

                # Track search source
                for img_id in text_ids:
                    search_path_sources[img_id] = search_path_sources.get(img_id, []) + ['text_only']

                debug_id(f"Found {len(text_ids)} images via text-only search", rid)

            # If no search criteria provided, return recent images
            if not all_image_ids and not any(
                    [title, description, position_id, tool_id, task_id, problem_id, completed_document_id,
                     area_id, equipment_group_id, model_id, asset_number_id, location_id,
                     subassembly_id, component_assembly_id, assembly_view_id, site_location_id,
                     similarity_query_embedding]):
                debug_id("=== FALLBACK: Recent images ===", rid)
                warning_id("No search criteria provided, returning recent images", rid)
                recent_query = session.query(cls.id).order_by(cls.id.desc()).limit(limit)
                recent_results = recent_query.all()
                all_image_ids = {result.id for result in recent_results}

                # Track search source
                for img_id in all_image_ids:
                    search_path_sources[img_id] = ['recent']

            # ======================================================================
            # PHASE 3: INTELLIGENT RANKING AND FILTERING
            # ======================================================================

            debug_id("=== PHASE 3: Ranking and filtering ===", rid)

            # Apply intelligent ranking
            if use_hybrid_ranking and similarity_scores:
                # Hybrid ranking: combine similarity with search path relevance
                ranked_image_ids = cls._calculate_hybrid_ranking(
                    all_image_ids, similarity_scores, search_path_sources, limit
                )
                debug_id("Applied hybrid ranking (similarity + relevance)", rid)
            elif similarity_scores:
                # Pure similarity ranking
                ranked_image_ids = sorted(
                    all_image_ids,
                    key=lambda x: similarity_scores.get(x, 0),
                    reverse=True
                )[:limit]
                debug_id("Applied pure similarity ranking", rid)
            else:
                # Traditional ranking (by ID descending for recency)
                ranked_image_ids = sorted(all_image_ids, reverse=True)[:limit]
                debug_id("Applied traditional recency ranking", rid)

            info_id(
                f"Found {len(all_image_ids)} total unique images, returning {len(ranked_image_ids)} after ranking and limit",
                rid)

            # ======================================================================
            # PHASE 4: BUILD ENHANCED RESPONSE
            # ======================================================================

            debug_id("=== PHASE 4: Building enhanced response ===", rid)

            # Get full image objects while preserving ranking order
            if ranked_image_ids:
                images_query = session.query(cls).filter(cls.id.in_(ranked_image_ids))
                images_by_id = {img.id: img for img in images_query.all()}
                ordered_images = [images_by_id[img_id] for img_id in ranked_image_ids if img_id in images_by_id]
            else:
                ordered_images = []

            # Build enhanced detailed response
            images = []
            for image in ordered_images:
                try:
                    image_details = {
                        "id": image.id,
                        "title": image.title,
                        "description": image.description,
                        "file_path": image.file_path,
                        "img_metadata": image.img_metadata,
                        "view_url": f"/add_document/image/{image.id}",
                        "associations": cls._get_enhanced_image_associations(session, image.id, rid),
                        # NEW: Enhanced search metadata
                        "search_metadata": {
                            "search_paths": search_path_sources.get(image.id, []),
                            "found_via_similarity": image.id in similarity_scores,
                            "similarity_score": similarity_scores.get(image.id),
                            "search_relevance": len(search_path_sources.get(image.id, [])),
                            # Higher = found via more paths
                            "has_pgvector_embedding": cls._has_pgvector_embedding(session, image.id,
                                                                                  embedding_model_name)
                        }
                    }

                    images.append(image_details)
                    debug_id(f"Processed image: ID={image.id}, Title='{image.title}', "
                             f"Paths={search_path_sources.get(image.id, [])}, "
                             f"Similarity={similarity_scores.get(image.id, 'N/A')}", rid)

                except Exception as e:
                    error_id(f"Error processing image ID {image.id}: {e}", rid, exc_info=True)
                    continue

            # Add search summary to response
            search_summary = {
                "total_found": len(all_image_ids),
                "returned": len(images),
                "used_similarity_search": len(similarity_scores) > 0,
                "similarity_results": len(similarity_scores),
                "traditional_results": len(all_image_ids) - len(similarity_scores),
                "search_paths_used": list(set([path for paths in search_path_sources.values() for path in paths])),
                "hybrid_ranking_applied": use_hybrid_ranking and len(similarity_scores) > 0
            }

            info_id(f"Search complete: {search_summary}", rid)

            # Optionally attach search summary to first result for debugging
            if images and len(images) > 0:
                images[0]["_search_summary"] = search_summary

            return images

        except Exception as e:
            error_id(f"Error in enhanced search_images: {e}", rid, exc_info=True)
            return []
        finally:
            info_id("========== ENHANCED DYNAMIC IMAGE SEARCH WITH PGVECTOR COMPLETE ==========", rid)

    @classmethod
    @with_request_id
    def search(cls, search_text=None, fields=None, image_id=None, title=None, description=None,
               file_path=None, position_id=None, tool_id=None, complete_document_id=None,
               exact_match=False, limit=20, offset=0, sort_by='id', sort_order='asc',
               request_id=None, session=None):
        """
        Simple search interface that delegates to the more powerful search_images method.
        This provides compatibility with the expected search interface.
        """
        try:
            # Create session if not provided
            local_session = None
            if session is None:
                db_config = DatabaseConfig()
                local_session = db_config.get_main_session()
                session = local_session

            try:
                # Use the existing powerful search_images method
                results = cls.search_images(
                    session=session,
                    title=title,
                    description=description,
                    position_id=position_id,
                    tool_id=tool_id,
                    completed_document_id=complete_document_id,
                    limit=limit,
                    request_id=request_id
                )

                # Convert the rich image dictionaries back to Image objects for compatibility
                image_objects = []
                if results:
                    image_ids = [result['id'] for result in results]
                    image_objects = session.query(cls).filter(cls.id.in_(image_ids)).all()

                    # Maintain the order from search_images
                    id_to_image = {img.id: img for img in image_objects}
                    image_objects = [id_to_image[img_id] for img_id in image_ids if img_id in id_to_image]

                info_id(f"Image.search found {len(image_objects)} results (via search_images)", request_id)
                return image_objects

            finally:
                if local_session:
                    local_session.close()

        except Exception as e:
            error_id(f"Error in Image.search: {e}", request_id, exc_info=True)
            return []

    @classmethod
    def _calculate_hybrid_ranking(cls, image_ids, similarity_scores, search_path_sources, limit):
        """
        Calculate intelligent hybrid ranking combining similarity and relevance factors.

        Args:
            image_ids: Set of image IDs to rank
            similarity_scores: Dict of image_id -> similarity_score
            search_path_sources: Dict of image_id -> list of search paths that found it
            limit: Maximum results to return

        Returns:
            List of image IDs in optimal ranking order
        """

        def calculate_composite_score(image_id):
            # Base similarity score (0.0 to 1.0)
            similarity = similarity_scores.get(image_id, 0.0)

            # Search path relevance boost
            search_paths = search_path_sources.get(image_id, [])
            path_count = len(search_paths)

            # Different search paths have different relevance weights
            path_weights = {
                'similarity': 1.0,  # Base pgvector similarity
                'direct_position': 0.8,  # Direct position associations are very relevant
                'tool': 0.9,  # Tool associations are highly relevant
                'task': 0.9,  # Task associations are highly relevant
                'problem': 0.9,  # Problem associations are highly relevant
                'specific_document': 0.85,  # Specific document requests are relevant
                'parts_position': 0.7,  # Parts associations are moderately relevant
                'document_association': 0.6,  # Document associations are somewhat relevant
                'text_only': 0.3,  # Text-only matches are less relevant
                'recent': 0.1  # Recent fallback is least relevant
            }

            # Calculate relevance boost based on search paths
            relevance_boost = sum(path_weights.get(path, 0.5) for path in search_paths) / max(len(search_paths), 1)

            # Multiple search paths finding the same image = higher confidence
            multi_path_boost = min(path_count * 0.1, 0.3)  # Cap at 0.3

            # Combine factors with weights
            composite_score = (
                    similarity * 0.6 +  # 60% similarity
                    relevance_boost * 0.3 +  # 30% search path relevance
                    multi_path_boost * 0.1  # 10% multi-path confidence
            )

            return composite_score

        # Sort by composite score and return top results
        ranked_ids = sorted(image_ids, key=calculate_composite_score, reverse=True)
        return ranked_ids[:limit]

    @classmethod
    def _has_pgvector_embedding(cls, session, image_id, model_name="CLIPModelHandler"):
        """
        Quick check if an image has a pgvector embedding available.

        Args:
            session: Database session
            image_id: ID of the image to check
            model_name: Embedding model name to check for

        Returns:
            bool: True if pgvector embedding exists
        """
        try:
            embedding = session.query(ImageEmbedding).filter(
                ImageEmbedding.image_id == image_id,
                ImageEmbedding.model_name == model_name,
                ImageEmbedding.embedding_vector.isnot(None)
            ).first()

            return embedding is not None
        except Exception:
            return False

    @classmethod
    @with_request_id
    def _get_enhanced_image_associations(cls, session, image_id, request_id=None):
        """
        Enhanced helper method to get all associations for an image, including structure-guided ones.
        """
        from modules.configuration.log_config import debug_id, error_id

        rid = request_id or get_request_id()
        associations = {}

        try:
            # Get position associations
            position_assocs = session.query(ImagePositionAssociation).filter(
                ImagePositionAssociation.image_id == image_id).all()
            associations['positions'] = [{'position_id': assoc.position_id} for assoc in position_assocs]

            # Get tool associations
            tool_assocs = session.query(ToolImageAssociation).filter(
                ToolImageAssociation.image_id == image_id).all()
            associations['tools'] = [{'tool_id': assoc.tool_id, 'description': assoc.description}
                                     for assoc in tool_assocs]

            # Get task associations
            task_assocs = session.query(ImageTaskAssociation).filter(
                ImageTaskAssociation.image_id == image_id).all()
            associations['tasks'] = [{'task_id': assoc.task_id} for assoc in task_assocs]

            # Get problem associations
            problem_assocs = session.query(ImageProblemAssociation).filter(
                ImageProblemAssociation.image_id == image_id).all()
            associations['problems'] = [{'problem_id': assoc.problem_id} for assoc in problem_assocs]

            # ENHANCED: Get completed document associations with structure-guided data
            doc_assocs = session.query(ImageCompletedDocumentAssociation).filter(
                ImageCompletedDocumentAssociation.image_id == image_id).all()

            enhanced_doc_assocs = []
            for assoc in doc_assocs:
                context_metadata = {}
                if assoc.context_metadata:
                    try:
                        context_metadata = json.loads(assoc.context_metadata)
                    except:
                        pass

                enhanced_doc_assocs.append({
                    'document_id': assoc.complete_document_id,
                    'page_number': assoc.page_number,
                    'chunk_index': assoc.chunk_index,
                    'association_method': assoc.association_method,
                    'confidence_score': assoc.confidence_score,
                    'structure_guided': context_metadata.get('structure_guided', False),
                    'content_type': context_metadata.get('content_type', 'image')
                })

            associations['completed_documents'] = enhanced_doc_assocs

            # Get parts position associations
            parts_assocs = session.query(PartsPositionImageAssociation).filter(
                PartsPositionImageAssociation.image_id == image_id).all()
            associations['parts_positions'] = [{'part_id': assoc.part_id, 'position_id': assoc.position_id}
                                               for assoc in parts_assocs]

            debug_id(f"Retrieved enhanced associations for image {image_id}: "
                     f"{len(associations.get('positions', []))} positions, "
                     f"{len(associations.get('tools', []))} tools, "
                     f"{len(associations.get('tasks', []))} tasks, "
                     f"{len(associations.get('problems', []))} problems, "
                     f"{len(associations.get('completed_documents', []))} documents", rid)

        except Exception as e:
            error_id(f"Error getting enhanced associations for image {image_id}: {e}", rid, exc_info=True)

        return associations

    @classmethod
    @with_request_id
    def search_similar_images_by_embedding(cls, session, query_embedding, model_name="CLIPModelHandler",
                                           limit=10, similarity_threshold=0.7, request_id=None):
        """
        New method to search for similar images using pgvector similarity search.
        """
        rid = request_id or get_request_id()

        try:
            info_id(f"Searching for similar images using {model_name} with threshold {similarity_threshold}", rid)

            # Convert embedding to list if necessary
            if hasattr(query_embedding, 'tolist'):
                embedding_list = query_embedding.tolist()
            elif isinstance(query_embedding, np.ndarray):
                embedding_list = query_embedding.flatten().tolist()
            else:
                embedding_list = list(query_embedding)

            # Use the ImageEmbedding's search method
            similar_images = ImageEmbedding.search_similar_images(
                session=session,
                query_embedding=embedding_list,
                model_name=model_name,
                limit=limit,
                similarity_threshold=similarity_threshold
            )

            # Enhance results with full image data
            enhanced_results = []
            for result in similar_images:
                image = session.query(cls).filter_by(id=result['image_id']).first()
                if image:
                    enhanced_result = {
                        **result,
                        'image': {
                            'id': image.id,
                            'title': image.title,
                            'description': image.description,
                            'file_path': image.file_path,
                            'metadata': image.img_metadata,
                            'view_url': f'/add_document/image/{image.id}'
                        }
                    }
                    enhanced_results.append(enhanced_result)

            info_id(f"Found {len(enhanced_results)} similar images", rid)
            return enhanced_results

        except Exception as e:
            error_id(f"Error in similarity search: {e}", rid, exc_info=True)
            return []

    @classmethod
    @with_request_id
    def find_similar_images(cls, session, reference_image_id, model_name="CLIPModelHandler",
                            limit=10, similarity_threshold=0.7, request_id=None):
        """
        New method to find images similar to a reference image using pgvector.
        """
        rid = request_id or get_request_id()

        try:
            info_id(f"Finding images similar to image ID {reference_image_id}", rid)

            # Use the ImageEmbedding's find similar method
            similar_images = ImageEmbedding.find_similar_to_image(
                session=session,
                image_id=reference_image_id,
                model_name=model_name,
                limit=limit,
                exclude_self=True
            )

            # Enhance results with full image data
            enhanced_results = []
            for result in similar_images:
                image = session.query(cls).filter_by(id=result['image_id']).first()
                if image:
                    enhanced_result = {
                        **result,
                        'image': {
                            'id': image.id,
                            'title': image.title,
                            'description': image.description,
                            'file_path': image.file_path,
                            'metadata': image.img_metadata,
                            'view_url': f'/add_document/image/{image.id}'
                        }
                    }
                    enhanced_results.append(enhanced_result)

            info_id(f"Found {len(enhanced_results)} images similar to reference image", rid)
            return enhanced_results

        except Exception as e:
            error_id(f"Error finding similar images: {e}", rid, exc_info=True)
            return []

    @classmethod
    @with_request_id
    def migrate_all_embeddings_to_pgvector(cls, session, request_id=None):
        """
        Utility method to migrate all existing image embeddings to pgvector format.
        """
        rid = request_id or get_request_id()

        try:
            info_id("Starting migration of all image embeddings to pgvector", rid)

            # Get all embeddings that are not in pgvector format
            legacy_embeddings = session.query(ImageEmbedding).filter(
                ImageEmbedding.embedding_vector.is_(None),
                ImageEmbedding.model_embedding.isnot(None)
            ).all()

            total_count = len(legacy_embeddings)
            migrated_count = 0
            failed_count = 0

            info_id(f"Found {total_count} legacy embeddings to migrate", rid)

            for embedding in legacy_embeddings:
                try:
                    if embedding.migrate_to_pgvector():
                        migrated_count += 1
                        debug_id(f"Migrated embedding {embedding.id} for image {embedding.image_id}", rid)
                    else:
                        failed_count += 1
                        warning_id(f"Failed to migrate embedding {embedding.id} for image {embedding.image_id}", rid)

                    # Commit in batches of 50
                    if (migrated_count + failed_count) % 50 == 0:
                        session.commit()
                        info_id(f"Batch commit: {migrated_count} migrated, {failed_count} failed", rid)

                except Exception as e:
                    failed_count += 1
                    error_id(f"Error migrating embedding {embedding.id}: {e}", rid)

            # Final commit
            session.commit()

            info_id(f"Migration complete: {migrated_count} successfully migrated, {failed_count} failed", rid)

            return {
                'total': total_count,
                'migrated': migrated_count,
                'failed': failed_count,
                'success_rate': (migrated_count / total_count * 100) if total_count > 0 else 0
            }

        except Exception as e:
            error_id(f"Error in bulk migration: {e}", rid, exc_info=True)
            session.rollback()
            return {'total': 0, 'migrated': 0, 'failed': 0, 'success_rate': 0}

    @classmethod
    @with_request_id
    def setup_pgvector_indexes(cls, session, request_id=None):
        """
        Setup pgvector indexes for optimal similarity search performance.
        """
        rid = request_id or get_request_id()

        try:
            info_id("Setting up pgvector indexes for image embeddings", rid)

            if ImageEmbedding.create_pgvector_indexes(session):
                info_id("Successfully created pgvector indexes", rid)
                return True
            else:
                warning_id("Failed to create some pgvector indexes", rid)
                return False

        except Exception as e:
            error_id(f"Error setting up pgvector indexes: {e}", rid, exc_info=True)
            return False

    @classmethod
    @with_request_id
    def get_embedding_statistics(cls, session, request_id=None):
        """
        Get statistics about image embeddings and their storage formats.
        """
        rid = request_id or get_request_id()

        try:
            from sqlalchemy import func

            # Total embeddings
            total_embeddings = session.query(ImageEmbedding).count()

            # pgvector embeddings
            pgvector_embeddings = session.query(ImageEmbedding).filter(
                ImageEmbedding.embedding_vector.isnot(None)
            ).count()

            # Legacy embeddings
            legacy_embeddings = session.query(ImageEmbedding).filter(
                ImageEmbedding.model_embedding.isnot(None),
                ImageEmbedding.embedding_vector.is_(None)
            ).count()

            # Both formats
            both_formats = session.query(ImageEmbedding).filter(
                ImageEmbedding.embedding_vector.isnot(None),
                ImageEmbedding.model_embedding.isnot(None)
            ).count()

            # Models breakdown
            model_stats = session.query(
                ImageEmbedding.model_name,
                func.count(ImageEmbedding.id).label('count')
            ).group_by(ImageEmbedding.model_name).all()

            stats = {
                'total_embeddings': total_embeddings,
                'pgvector_embeddings': pgvector_embeddings,
                'legacy_embeddings': legacy_embeddings,
                'both_formats': both_formats,
                'pgvector_percentage': (pgvector_embeddings / total_embeddings * 100) if total_embeddings > 0 else 0,
                'legacy_percentage': (legacy_embeddings / total_embeddings * 100) if total_embeddings > 0 else 0,
                'models': {model: count for model, count in model_stats}
            }

            info_id(f"Embedding statistics: {total_embeddings} total, "
                    f"{pgvector_embeddings} pgvector ({stats['pgvector_percentage']:.1f}%), "
                    f"{legacy_embeddings} legacy ({stats['legacy_percentage']:.1f}%)", rid)

            return stats

        except Exception as e:
            error_id(f"Error getting embedding statistics: {e}", rid, exc_info=True)
            return {}

class ImageEmbedding(Base):
    """
    Enhanced image embedding model supporting both legacy LargeBinary and modern pgvector storage.

    This class intelligently handles both storage formats:
    - Legacy: model_embedding (LargeBinary) for backward compatibility
    - Modern: embedding_vector (pgvector) for better performance

    The class automatically determines which storage format to use based on what's available.
    """
    __tablename__ = 'image_embedding'

    id = Column(Integer, primary_key=True)
    image_id = Column(Integer, ForeignKey('image.id'))
    model_name = Column(String, nullable=False)

    # Legacy storage (keep for backward compatibility)
    model_embedding = Column(LargeBinary, nullable=True)  # Made nullable

    # Modern pgvector storage (add this column)
    # CLIP embeddings are typically 512 dimensions, adjust as needed
    embedding_vector = Column(Vector(512), nullable=True)  # Made nullable

    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # Relationships
    image = relationship("Image", back_populates="image_embedding")

    def __repr__(self):
        storage_type = self.get_storage_type()
        return f"<ImageEmbedding(image_id={self.image_id}, model={self.model_name}, storage={storage_type})>"

    @property
    def embedding_as_list(self) -> List[float]:
        """
        Get embedding as Python list, automatically handling both storage formats.

        Returns:
            List[float]: Embedding vector as list of floats
        """
        # Try pgvector first (preferred format)
        if self.embedding_vector is not None:
            try:
                # Convert pgvector to list
                vector_str = str(self.embedding_vector)
                if vector_str.startswith('[') and vector_str.endswith(']'):
                    vector_str = vector_str[1:-1]  # Remove brackets
                return [float(x.strip()) for x in vector_str.split(',') if x.strip()]
            except Exception as e:
                print(f"Warning: Error parsing pgvector format: {e}")

        # Fallback to legacy LargeBinary format
        if self.model_embedding is not None:
            try:
                # Handle numpy array stored as bytes
                import numpy as np
                return np.frombuffer(self.model_embedding, dtype=np.float32).tolist()
            except Exception as e:
                try:
                    # Fallback to JSON format
                    return json.loads(self.model_embedding.decode('utf-8'))
                except Exception as e2:
                    print(f"Warning: Error parsing legacy embedding format: {e}, {e2}")

        return []

    @embedding_as_list.setter
    def embedding_as_list(self, value: List[float]):
        """
        Set embedding from Python list, preferring pgvector storage.

        Args:
            value: List of floats representing the embedding
        """
        if not isinstance(value, list) or not value:
            raise ValueError("Embedding must be a non-empty list of floats")

        # Store in pgvector format (preferred)
        try:
            self.embedding_vector = value
        except Exception as e:
            print(f"Warning: Could not store in pgvector format: {e}")
            # Fallback to legacy format (numpy array as bytes)
            import numpy as np
            self.model_embedding = np.array(value, dtype=np.float32).tobytes()

    def get_storage_type(self) -> str:
        """
        Determine which storage format is being used.

        Returns:
            str: 'pgvector', 'legacy', 'both', or 'none'
        """
        has_pgvector = self.embedding_vector is not None
        has_legacy = self.model_embedding is not None

        if has_pgvector and has_legacy:
            return 'both'
        elif has_pgvector:
            return 'pgvector'
        elif has_legacy:
            return 'legacy'
        else:
            return 'none'

    def migrate_to_pgvector(self) -> bool:
        """
        Migrate legacy LargeBinary embedding to pgvector format.
        Fixed to handle numpy arrays stored as binary data.

        Returns:
            bool: True if migration successful, False otherwise
        """
        if self.embedding_vector is not None:
            return True  # Already in pgvector format

        if self.model_embedding is None:
            return False  # No data to migrate

        try:
            import numpy as np

            # Try to parse as numpy array first (CLIP embeddings are stored this way)
            try:
                legacy_data = np.frombuffer(self.model_embedding, dtype=np.float32).tolist()
                print(f"Successfully parsed numpy array with {len(legacy_data)} dimensions")
            except Exception as numpy_error:
                print(f"Numpy parsing failed: {numpy_error}")

                # Fallback to JSON format (for document embeddings)
                try:
                    legacy_data = json.loads(self.model_embedding.decode('utf-8'))
                    print(f"Successfully parsed JSON with {len(legacy_data)} dimensions")
                except Exception as json_error:
                    print(f"JSON parsing also failed: {json_error}")
                    return False

            # Validate the data
            if not isinstance(legacy_data, list) or len(legacy_data) == 0:
                print(
                    f"Invalid embedding data: type={type(legacy_data)}, length={len(legacy_data) if hasattr(legacy_data, '__len__') else 'N/A'}")
                return False

            # Store in pgvector format
            self.embedding_vector = legacy_data

            print(f"Successfully migrated embedding with {len(legacy_data)} dimensions to pgvector")
            return True

        except Exception as e:
            print(f"Error migrating image embedding to pgvector: {e}")
            return False

    def get_embedding_stats(self) -> dict:
        """
        Get statistics about the embedding.

        Returns:
            dict: Statistics including dimensions, storage type, etc.
        """
        embedding = self.embedding_as_list

        return {
            'dimensions': len(embedding),
            'storage_type': self.get_storage_type(),
            'model_name': self.model_name,
            'image_id': self.image_id,
            'has_data': len(embedding) > 0,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'updated_at': self.updated_at.isoformat() if self.updated_at else None
        }

    @classmethod
    def create_with_pgvector(cls, image_id: int, model_name: str, embedding: List[float], **kwargs):
        """
        Create a new ImageEmbedding using pgvector storage.

        Args:
            image_id: ID of the associated image
            model_name: Name of the embedding model
            embedding: List of floats representing the embedding
            **kwargs: Additional keyword arguments

        Returns:
            ImageEmbedding: New instance with pgvector storage
        """
        instance = cls(
            image_id=image_id,
            model_name=model_name,
            embedding_vector=embedding,
            **kwargs
        )
        return instance

    @classmethod
    def create_with_legacy(cls, image_id: int, model_name: str, embedding: List[float], **kwargs):
        """
        Create a new ImageEmbedding using legacy LargeBinary storage.

        Args:
            image_id: ID of the associated image
            model_name: Name of the embedding model
            embedding: List of floats representing the embedding
            **kwargs: Additional keyword arguments

        Returns:
            ImageEmbedding: New instance with legacy storage
        """
        import numpy as np
        instance = cls(
            image_id=image_id,
            model_name=model_name,
            model_embedding=np.array(embedding, dtype=np.float32).tobytes(),
            **kwargs
        )
        return instance

    def cosine_similarity(self, other_embedding: List[float]) -> Optional[float]:
        """
        Calculate cosine similarity with another embedding.

        Args:
            other_embedding: List of floats to compare against

        Returns:
            float: Cosine similarity score (0-1), or None if calculation fails
        """
        current_embedding = self.embedding_as_list

        if not current_embedding or not other_embedding:
            return None

        if len(current_embedding) != len(other_embedding):
            return None

        try:
            import math

            dot_product = sum(a * b for a, b in zip(current_embedding, other_embedding))
            norm_a = math.sqrt(sum(a * a for a in current_embedding))
            norm_b = math.sqrt(sum(b * b for b in other_embedding))

            if norm_a == 0 or norm_b == 0:
                return 0.0

            return dot_product / (norm_a * norm_b)
        except Exception as e:
            print(f"Error calculating cosine similarity: {e}")
            return None

    def to_dict(self) -> dict:
        """
        Convert embedding to dictionary for JSON serialization.

        Returns:
            dict: Dictionary representation of the embedding
        """
        return {
            'id': self.id,
            'image_id': self.image_id,
            'model_name': self.model_name,
            'embedding': self.embedding_as_list,
            'storage_type': self.get_storage_type(),
            'dimensions': len(self.embedding_as_list),
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'updated_at': self.updated_at.isoformat() if self.updated_at else None
        }

    @classmethod
    def create_pgvector_indexes(cls, session):
        """
        Create optimized indexes for pgvector similarity search.

        Args:
            session: SQLAlchemy session

        Returns:
            bool: True if indexes created successfully
        """
        from sqlalchemy import text

        indexes = [
            # HNSW index for cosine similarity
            """
            CREATE INDEX IF NOT EXISTS idx_image_embedding_cosine 
            ON image_embedding 
            USING hnsw (embedding_vector vector_cosine_ops)
            WITH (m = 16, ef_construction = 64);
            """,

            # HNSW index for L2 distance
            """
            CREATE INDEX IF NOT EXISTS idx_image_embedding_l2 
            ON image_embedding 
            USING hnsw (embedding_vector vector_l2_ops)
            WITH (m = 16, ef_construction = 64);
            """,

            # Regular indexes
            "CREATE INDEX IF NOT EXISTS idx_image_embedding_model ON image_embedding (model_name);",
            "CREATE INDEX IF NOT EXISTS idx_image_embedding_image_id ON image_embedding (image_id);",
            "CREATE INDEX IF NOT EXISTS idx_image_embedding_created ON image_embedding (created_at);"
        ]

        try:
            for index_sql in indexes:
                session.execute(text(index_sql))
            session.commit()
            print("pgvector indexes created successfully for image embeddings")
            return True
        except Exception as e:
            session.rollback()
            print(f"Error creating pgvector indexes for image embeddings: {e}")
            return False

    @classmethod
    def search_similar_images(cls, session, query_embedding: List[float],
                              model_name: str = None, limit: int = 10,
                              similarity_threshold: float = 0.5) -> List[Dict]:
        """
        Search for images similar to the query embedding using pgvector.

        Args:
            session: Database session
            query_embedding: Embedding vector to search for
            model_name: Filter by specific model (optional)
            limit: Maximum number of results
            similarity_threshold: Minimum similarity score

        Returns:
            List of dictionaries with image info and similarity scores
        """
        from modules.emtacdb.emtacdb_fts import Image

        try:
            # Build query
            query = session.query(
                cls,
                Image,
                (1 - cls.embedding_vector.cosine_distance(query_embedding)).label('similarity')
            ).join(Image, cls.image_id == Image.id)

            # Filter by model if specified
            if model_name:
                query = query.filter(cls.model_name == model_name)

            # Filter by similarity threshold and order by similarity
            results = query.filter(
                cls.embedding_vector.cosine_distance(query_embedding) <= (2 * (1 - similarity_threshold))
            ).order_by(
                cls.embedding_vector.cosine_distance(query_embedding)
            ).limit(limit).all()

            # Format results
            formatted_results = []
            for embedding, image, similarity in results:
                formatted_results.append({
                    'image_id': image.id,
                    'image_title': image.title,
                    'image_path': image.file_path,
                    'similarity': float(similarity),
                    'model_name': embedding.model_name,
                    'embedding_id': embedding.id
                })

            return formatted_results

        except Exception as e:
            print(f"Error in image similarity search: {e}")
            return []

    @classmethod
    def find_similar_to_image(cls, session, image_id: int, model_name: str = None,
                              limit: int = 10, exclude_self: bool = True) -> List[Dict]:
        """
        Find images similar to a specific image using its stored embedding.

        Args:
            session: Database session
            image_id: ID of the reference image
            model_name: Embedding model to use
            limit: Maximum number of results
            exclude_self: Whether to exclude the reference image from results

        Returns:
            List of similar images with similarity scores
        """
        try:
            # Get the reference image's embedding
            ref_embedding = session.query(cls).filter_by(
                image_id=image_id,
                model_name=model_name
            ).first()

            if not ref_embedding or not ref_embedding.embedding_vector:
                return []

            # Use the existing search method
            results = cls.search_similar_images(
                session,
                ref_embedding.embedding_as_list,
                model_name,
                limit + (1 if exclude_self else 0)
            )

            # Remove the reference image if requested
            if exclude_self:
                results = [r for r in results if r['image_id'] != image_id][:limit]

            return results

        except Exception as e:
            print(f"Error finding similar images for image {image_id}: {e}")
            return []

class DrawingType(Enum):
    ELECTRICAL = "Electrical"
    MECHANICAL = "Mechanical"
    PIPING = "Piping"
    INSTRUMENTATION = "Instrumentation"
    CIVIL = "Civil"
    STRUCTURAL = "Structural"
    PROCESS = "Process"
    ASSEMBLY = "Assembly"
    DETAIL = "Detail"
    SCHEMATIC = "Schematic"
    LAYOUT = "Layout"
    OTHER = "Other"

class Drawing(Base):
    __tablename__ = 'drawing'

    id = Column(Integer, primary_key=True)
    drw_equipment_name = Column(String)
    drw_number = Column(String)
    drw_name = Column(String)
    drw_revision = Column(String)
    drw_spare_part_number = Column(String)
    drw_type = Column(String, default="Other")  # New field for drawing type
    file_path = Column(String, nullable=False)

    drawing_position = relationship("DrawingPositionAssociation", back_populates="drawing")
    drawing_problem = relationship("DrawingProblemAssociation", back_populates="drawing")
    drawing_task = relationship("DrawingTaskAssociation", back_populates="drawing")
    drawing_part = relationship("DrawingPartAssociation", back_populates="drawing")

    @classmethod
    @with_request_id
    def search(cls,
               search_text: Optional[str] = None,
               fields: Optional[List[str]] = None,
               exact_match: bool = False,
               drawing_id: Optional[int] = None,
               drw_equipment_name: Optional[str] = None,
               drw_number: Optional[str] = None,
               drw_name: Optional[str] = None,
               drw_revision: Optional[str] = None,
               drw_spare_part_number: Optional[str] = None,
               drw_type: Optional[str] = None,  # New parameter for drawing type
               file_path: Optional[str] = None,
               limit: int = 100,
               request_id: Optional[str] = None,
               session: Optional[Session] = None) -> List['Drawing']:
        """
        Comprehensive search method for Drawing objects with flexible search options.

        Args:
            search_text: Text to search for across specified fields
            fields: List of field names to search in. If None, searches in default fields
                   (drw_number, drw_name, drw_equipment_name, drw_spare_part_number, drw_type)
            exact_match: If True, performs exact matching instead of partial matching
            drawing_id: Optional ID to filter by
            drw_equipment_name: Optional equipment name to filter by
            drw_number: Optional drawing number to filter by
            drw_name: Optional drawing name to filter by
            drw_revision: Optional revision to filter by
            drw_spare_part_number: Optional spare part number to filter by
            drw_type: Optional drawing type to filter by (e.g., 'Electrical', 'Mechanical')
            file_path: Optional file path to filter by
            limit: Maximum number of results to return (default 100)
            request_id: Optional request ID for tracking this operation in logs
            session: Optional SQLAlchemy session. If None, a new session will be created

        Returns:
            List of Drawing objects matching the search criteria
        """
        # Get or use the provided request_id
        rid = request_id or get_request_id()

        # Get a database session if one wasn't provided
        db_config = DatabaseConfig()
        session_provided = session is not None
        if not session_provided:
            session = db_config.get_main_session()
            debug_id(f"Created new database session for Drawing.search", rid)

        # Log the search operation with request ID
        search_params = {
            'search_text': search_text,
            'fields': fields,
            'exact_match': exact_match,
            'drawing_id': drawing_id,
            'drw_equipment_name': drw_equipment_name,
            'drw_number': drw_number,
            'drw_name': drw_name,
            'drw_revision': drw_revision,
            'drw_spare_part_number': drw_spare_part_number,
            'drw_type': drw_type,  # Include new parameter in logging
            'file_path': file_path,
            'limit': limit
        }
        # Filter out None values for cleaner logging
        logged_params = {k: v for k, v in search_params.items() if v is not None}
        debug_id(f"Starting Drawing.search with parameters: {logged_params}", rid)

        try:
            # Use the timed operation context manager with request ID
            with log_timed_operation("Drawing.search", rid):
                # Start with the base query
                query = session.query(cls)
                filters = []

                # Process search_text across multiple fields if provided
                if search_text:
                    search_text = search_text.strip()
                    if search_text:
                        # Default fields to search in if none specified (now includes drw_type)
                        if fields is None or len(fields) == 0:
                            fields = ['drw_number', 'drw_name', 'drw_equipment_name', 'drw_spare_part_number',
                                      'drw_type']

                        debug_id(f"Searching for text '{search_text}' in fields: {fields}", rid)

                        # Create field-specific filters
                        text_filters = []
                        for field_name in fields:
                            if hasattr(cls, field_name):
                                field = getattr(cls, field_name)
                                if exact_match:
                                    text_filters.append(field == search_text)
                                else:
                                    text_filters.append(field.ilike(f"%{search_text}%"))

                        # Add the combined text search filter if we have any
                        if text_filters:
                            filters.append(or_(*text_filters))

                # Add filters for specific fields if provided
                if drawing_id is not None:
                    debug_id(f"Adding filter for drawing_id: {drawing_id}", rid)
                    filters.append(cls.id == drawing_id)

                if drw_equipment_name is not None:
                    debug_id(f"Adding filter for drw_equipment_name: {drw_equipment_name}", rid)
                    if exact_match:
                        filters.append(cls.drw_equipment_name == drw_equipment_name)
                    else:
                        filters.append(cls.drw_equipment_name.ilike(f"%{drw_equipment_name}%"))

                if drw_number is not None:
                    debug_id(f"Adding filter for drw_number: {drw_number}", rid)
                    if exact_match:
                        filters.append(cls.drw_number == drw_number)
                    else:
                        filters.append(cls.drw_number.ilike(f"%{drw_number}%"))

                if drw_name is not None:
                    debug_id(f"Adding filter for drw_name: {drw_name}", rid)
                    if exact_match:
                        filters.append(cls.drw_name == drw_name)
                    else:
                        filters.append(cls.drw_name.ilike(f"%{drw_name}%"))

                if drw_revision is not None:
                    debug_id(f"Adding filter for drw_revision: {drw_revision}", rid)
                    if exact_match:
                        filters.append(cls.drw_revision == drw_revision)
                    else:
                        filters.append(cls.drw_revision.ilike(f"%{drw_revision}%"))

                if drw_spare_part_number is not None:
                    debug_id(f"Adding filter for drw_spare_part_number: {drw_spare_part_number}", rid)
                    if exact_match:
                        filters.append(cls.drw_spare_part_number == drw_spare_part_number)
                    else:
                        filters.append(cls.drw_spare_part_number.ilike(f"%{drw_spare_part_number}%"))

                # New filter for drawing type
                if drw_type is not None:
                    debug_id(f"Adding filter for drw_type: {drw_type}", rid)
                    if exact_match:
                        filters.append(cls.drw_type == drw_type)
                    else:
                        filters.append(cls.drw_type.ilike(f"%{drw_type}%"))

                if file_path is not None:
                    debug_id(f"Adding filter for file_path: {file_path}", rid)
                    if exact_match:
                        filters.append(cls.file_path == file_path)
                    else:
                        filters.append(cls.file_path.ilike(f"%{file_path}%"))

                # Apply all filters with AND logic if we have any
                if filters:
                    query = query.filter(and_(*filters))

                # Apply limit
                query = query.limit(limit)

                # Execute query and log results
                results = query.all()
                debug_id(f"Drawing.search completed, found {len(results)} results", rid)
                return results

        except Exception as e:
            error_id(f"Error in Drawing.search: {str(e)}", rid)
            # Re-raise the exception after logging it
            raise
        finally:
            # Close the session if we created it
            if not session_provided:
                session.close()
                debug_id(f"Closed database session for Drawing.search", rid)

    @classmethod
    @with_request_id
    def get_by_id(cls, drawing_id: int, request_id: Optional[str] = None, session: Optional[Session] = None) -> \
            Optional['Drawing']:
        """
        Get a drawing by its ID.

        Args:
            drawing_id: ID of the drawing to retrieve
            request_id: Optional request ID for tracking this operation in logs
            session: Optional SQLAlchemy session. If None, a new session will be created

        Returns:
            Drawing object if found, None otherwise
        """
        # Get or use the provided request_id
        rid = request_id or get_request_id()

        # Get a database session if one wasn't provided
        db_config = DatabaseConfig()
        session_provided = session is not None
        if not session_provided:
            session = db_config.get_main_session()
            debug_id(f"Created new database session for Drawing.get_by_id", rid)

        debug_id(f"Getting drawing with ID: {drawing_id}", rid)
        try:
            drawing = session.query(cls).filter(cls.id == drawing_id).first()
            if drawing:
                debug_id(f"Found drawing: {drawing.drw_number} (ID: {drawing_id})", rid)
            else:
                debug_id(f"No drawing found with ID: {drawing_id}", rid)
            return drawing
        except Exception as e:
            error_id(f"Error retrieving drawing with ID {drawing_id}: {str(e)}", rid)
            return None
        finally:
            # Close the session if we created it
            if not session_provided:
                session.close()
                debug_id(f"Closed database session for Drawing.get_by_id", rid)

    @classmethod
    @with_request_id
    def search_and_format(cls, search_text=None, fields=None, exact_match=False, drawing_id=None,
                          drw_equipment_name=None, drw_number=None, drw_name=None, drw_revision=None,
                          drw_spare_part_number=None, drw_type=None, file_path=None, limit=100,
                          request_id=None, session=None):
        """
        Search for drawings and return formatted results ready for API response.

        Args:
            (same parameters as the search method, now including drw_type)

        Returns:
            Dictionary with entity_type and results ready for API response,
            or fallback to legacy search if needed
        """
        # Get or create session
        session_provided = session is not None
        if not session_provided:
            db_config = DatabaseConfig()
            session = db_config.get_main_session()

        try:
            # First try the regular search
            results = cls.search(
                search_text=search_text,
                fields=fields,
                exact_match=exact_match,
                drawing_id=drawing_id,
                drw_equipment_name=drw_equipment_name,
                drw_number=drw_number,
                drw_name=drw_name,
                drw_revision=drw_revision,
                drw_spare_part_number=drw_spare_part_number,
                drw_type=drw_type,  # Include new parameter
                file_path=file_path,
                limit=limit,
                request_id=request_id,
                session=session
            )

            if results:
                # Format the results (now includes drawing type)
                drawing_results = []
                for drawing in results:
                    drawing_results.append({
                        'id': drawing.id,
                        'number': drawing.drw_number,
                        'name': drawing.drw_name,
                        'equipment_name': drawing.drw_equipment_name,
                        'revision': drawing.drw_revision,
                        'spare_part_number': drawing.drw_spare_part_number,
                        'type': drawing.drw_type,  # Include type in results
                        'file_path': drawing.file_path,
                        'url': f"/drawings/view/{drawing.id}"
                    })

                return {
                    "entity_type": "drawing",
                    "results": drawing_results
                }

            # If no results and we have a drawing number, try legacy search
            if drw_number:
                try:
                    from modules.emtacdb.search_drawing_by_number_bp import search_drawing_by_number
                    legacy_result = search_drawing_by_number(session, drw_number)

                    if legacy_result and isinstance(legacy_result, list) and len(legacy_result) > 0:
                        # Format legacy results
                        drawing_results = []
                        for drawing in legacy_result:
                            drawing_results.append({
                                'id': drawing['id'],
                                'number': drawing['number'],
                                'name': drawing.get('name', ''),
                                'type': drawing.get('type', 'Unknown'),  # Handle legacy type field
                                'url': f"/drawings/view/{drawing['id']}"
                            })

                        return {
                            "entity_type": "drawing",
                            "results": drawing_results
                        }
                except ImportError:
                    # Legacy search not available, continue with no results response
                    pass

            # No results from either search method
            return {
                "entity_type": "response",
                "results": [{"text": "No drawings found matching your criteria."}]
            }

        except Exception as e:
            # Log the error
            error_id(f"Error in Drawing.search_and_format: {str(e)}", request_id)
            return {
                "entity_type": "error",
                "results": [{"text": f"Error searching for drawings: {str(e)}"}]
            }
        finally:
            # Close session if we created it
            if not session_provided and session:
                session.close()

    @classmethod
    @with_request_id
    def search_by_asset_number(cls, asset_number_value, request_id=None, session=None):
        """
        Search for drawings related to a specific asset number.

        Args:
            asset_number_value: The asset number to search by
            request_id: Optional request ID for tracking
            session: Optional SQLAlchemy session

        Returns:
            List of Drawing objects associated with the asset number
        """
        # Get or create session
        session_provided = session is not None
        if not session_provided:
            db_config = DatabaseConfig()
            session = db_config.get_main_session()

        try:
            # Step 1: Find asset number ID(s)
            asset_number_ids = AssetNumber.get_ids_by_number(session, asset_number_value)

            if not asset_number_ids:
                return []

            # Step 2: Find position IDs for these asset numbers
            position_ids = []
            for asset_id in asset_number_ids:
                pos_ids = AssetNumber.get_position_ids_by_asset_number_id(session, asset_id)
                position_ids.extend(pos_ids)

            if not position_ids:
                return []

            # Step 3: Find drawings for these positions
            drawings = []
            for pos_id in position_ids:
                drawing_results = DrawingPositionAssociation.get_drawings_by_position(
                    session=session,
                    position_id=pos_id,
                    request_id=request_id
                )
                drawings.extend(drawing_results)

            # Remove duplicates
            unique_drawings = {drawing.id: drawing for drawing in drawings}
            return list(unique_drawings.values())

        except Exception as e:
            error_id(f"Error in Drawing.search_by_asset_number: {str(e)}", request_id)
            return []
        finally:
            if not session_provided:
                session.close()

    @classmethod
    def get_available_types(cls):
        """
        Get all available drawing types.

        Returns:
            List of available drawing type values
        """
        return [dtype.value for dtype in DrawingType]

    @classmethod
    @with_request_id
    def search_by_type(cls, drawing_type: str, request_id: Optional[str] = None,
                       session: Optional[Session] = None) -> List['Drawing']:
        """
        Search for drawings by their type.

        Args:
            drawing_type: The type of drawing to search for
            request_id: Optional request ID for tracking
            session: Optional SQLAlchemy session

        Returns:
            List of Drawing objects of the specified type
        """
        return cls.search(drw_type=drawing_type, request_id=request_id, session=session)

class Document(Base):
    """Your existing Document model with added image association relationship."""
    __tablename__ = 'document'

    id = Column(Integer, primary_key=True)
    name = Column(String, nullable=False)
    file_path = Column(String, nullable=False)
    content = Column(String)
    complete_document_id = Column(Integer, ForeignKey('complete_document.id'))
    embedding = Column(LargeBinary)
    rev = Column(String, nullable=False, default="R0")
    doc_metadata = Column(JSON)

    # Existing relationships
    embeddings = relationship("DocumentEmbedding", back_populates="document")
    complete_document = relationship("CompleteDocument", back_populates="document")

    # NEW: Image associations relationship
    image_associations = relationship("ImageCompletedDocumentAssociation", back_populates="document_chunk")

    # NEW: Helper methods for image-chunk associations
    @classmethod
    @with_request_id
    def get_images_for_chunk(cls, chunk_id, request_id=None):
        """Get all images associated with this text chunk."""
        try:
            db_config = DatabaseConfig()
            with db_config.main_session() as session:
                # Query images through the association table
                result = session.query(Image, ImageCompletedDocumentAssociation).join(
                    ImageCompletedDocumentAssociation,
                    Image.id == ImageCompletedDocumentAssociation.image_id
                ).filter(
                    ImageCompletedDocumentAssociation.document_id == chunk_id
                ).all()

                return [{
                    'image_id': img.id,
                    'image_title': img.title,
                    'image_filepath': img.filepath,
                    'association_confidence': assoc.confidence_score,
                    'page_number': assoc.page_number,
                    'context_metadata': assoc.context_metadata
                } for img, assoc in result]

        except Exception as e:
            error_id(f"Failed to get images for chunk {chunk_id}: {e}", request_id)
            return []

    @classmethod
    @with_request_id
    def find_chunks_with_images(cls, complete_document_id, request_id=None):
        """Find all chunks in a document that have associated images."""
        try:
            db_config = DatabaseConfig()
            with db_config.main_session() as session:
                result = session.query(cls).join(
                    ImageCompletedDocumentAssociation,
                    cls.id == ImageCompletedDocumentAssociation.document_id
                ).filter(
                    cls.complete_document_id == complete_document_id
                ).distinct().all()

                return result

        except Exception as e:
            error_id(f"Failed to find chunks with images: {e}", request_id)
            return []

    # Enhance your existing create_fts_table method to handle chunk searching
    @classmethod
    @with_request_id
    def create_fts_table(cls):
        """Enhanced FTS table creation with file_path + image-chunk search support."""
        db_config = DatabaseConfig()

        with db_config.main_session() as session:
            try:
                # Enable extensions
                session.execute(text("CREATE EXTENSION IF NOT EXISTS pg_trgm"))
                session.commit()
                session.execute(text("CREATE EXTENSION IF NOT EXISTS unaccent"))
                session.commit()

                # Create enhanced FTS table that includes file_path
                session.execute(text("""
                                     CREATE TABLE IF NOT EXISTS documents_fts
                                     (
                                         id
                                         SERIAL
                                         PRIMARY
                                         KEY,
                                         title
                                         TEXT
                                         NOT
                                         NULL,
                                         content
                                         TEXT,
                                         file_path
                                         TEXT,
                                         chunk_id
                                         INTEGER, -- Links to document.id (chunk)
                                         complete_document_id
                                         INTEGER, -- Links to complete_document.id
                                         has_images
                                         BOOLEAN
                                         DEFAULT
                                         FALSE,   -- Quick flag for chunks with images
                                         search_vector
                                         TSVECTOR,
                                         created_at
                                         TIMESTAMP
                                         DEFAULT
                                         NOW
                                     (
                                     ),
                                         updated_at TIMESTAMP DEFAULT NOW
                                     (
                                     ),
                                         UNIQUE
                                     (
                                         title,
                                         file_path
                                     )
                                         )
                                     """))
                session.commit()

                # Create indexes
                index_statements = [
                    "CREATE INDEX IF NOT EXISTS idx_documents_fts_search_vector ON documents_fts USING gin(search_vector)",
                    "CREATE INDEX IF NOT EXISTS idx_documents_fts_title ON documents_fts(title)",
                    "CREATE INDEX IF NOT EXISTS idx_documents_fts_chunk_id ON documents_fts(chunk_id)",
                    "CREATE INDEX IF NOT EXISTS idx_documents_fts_complete_doc_id ON documents_fts(complete_document_id)",
                    "CREATE INDEX IF NOT EXISTS idx_documents_fts_has_images ON documents_fts(has_images)",
                    "CREATE INDEX IF NOT EXISTS idx_documents_fts_title_gin ON documents_fts USING gin(title gin_trgm_ops)",
                    "CREATE INDEX IF NOT EXISTS idx_documents_fts_content_gin ON documents_fts USING gin(content gin_trgm_ops)",
                    "CREATE INDEX IF NOT EXISTS idx_documents_fts_filepath_gin ON documents_fts USING gin(file_path gin_trgm_ops)"
                ]

                for stmt in index_statements:
                    session.execute(text(stmt))
                    session.commit()

                # Trigger function includes file_path
                session.execute(text("""
                                     CREATE
                                     OR REPLACE FUNCTION update_search_vector() RETURNS trigger AS $$
                                     BEGIN
                        NEW.search_vector
                                     := to_tsvector(
                            'english',
                            COALESCE(NEW.title, '') || ' ' ||
                            COALESCE(NEW.content, '') || ' ' ||
                            COALESCE(NEW.file_path, '')
                        );
                        NEW.updated_at
                                     := NOW();

                        -- Check if this chunk has associated images
                        IF
                                     NEW.chunk_id IS NOT NULL THEN
                                     SELECT EXISTS(SELECT 1
                                                   FROM image_completed_document_association
                                                   WHERE document_id = NEW.chunk_id)
                                     INTO NEW.has_images;
                                     END IF;

                                     RETURN NEW;
                                     END;
                    $$
                                     LANGUAGE plpgsql
                                     """))
                session.commit()

                # Create trigger
                session.execute(text("DROP TRIGGER IF EXISTS search_vector_update ON documents_fts"))
                session.commit()
                session.execute(text("""
                                     CREATE TRIGGER search_vector_update
                                         BEFORE INSERT OR
                                     UPDATE ON documents_fts
                                         FOR EACH ROW EXECUTE FUNCTION update_search_vector()
                                     """))
                session.commit()

                print("✅ Enhanced PostgreSQL FTS table created with file_path + image-chunk support")
                return True

            except Exception as e:
                session.rollback()
                print(f"❌ Failed to create enhanced FTS table: {e}")
                return False


class DocumentEmbedding(Base):
    """
    Enhanced embedding model supporting variable dimensions and rich metadata.

    Updated to work with migrated table structure that includes:
    - Flexible pgvector storage (no dimension constraints)
    - actual_dimensions column for fast querying
    - embedding_metadata JSONB column for rich model information
    """
    __tablename__ = 'document_embedding'

    id = Column(Integer, primary_key=True)
    document_id = Column(Integer, ForeignKey('document.id'), nullable=False)
    model_name = Column(String, nullable=False)

    # Legacy storage (backward compatibility)
    model_embedding = Column(LargeBinary, nullable=True)

    # Modern flexible pgvector storage (no dimension constraint)
    embedding_vector = Column(Vector, nullable=True)  # No dimension limit!

    # New enhanced columns from migration
    actual_dimensions = Column(Integer, nullable=True)
    embedding_metadata = Column(JSONB, nullable=True)

    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow, nullable=True)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=True)

    # Relationships
    document = relationship("Document", back_populates="embeddings")

    def __repr__(self):
        storage_type = self.get_storage_type()
        dims = self.actual_dimensions or "unknown"
        return f"<DocumentEmbedding(doc_id={self.document_id}, model={self.model_name}, dims={dims}, storage={storage_type})>"

    @property
    def embedding_as_list(self) -> List[float]:
        """
        Get embedding as Python list, automatically handling both storage formats.
        Optimized to reuse existing database session when possible.
        """
        # Try pgvector first (preferred format)
        if self.embedding_vector is not None:
            try:
                # Convert pgvector to list
                vector_str = str(self.embedding_vector).strip()

                # Check if data is truncated (contains ...)
                if '...' in vector_str:
                    # Try to get the actual data from the database
                    return self._get_full_embedding_from_db()

                # Handle bracket format: [1.0,2.0,3.0]
                if vector_str.startswith('[') and vector_str.endswith(']'):
                    vector_str = vector_str[1:-1]  # Remove brackets
                    embedding_list = [float(x.strip()) for x in vector_str.split(',') if x.strip()]

                # Handle escaped newlines and spaces
                elif any(seq in vector_str for seq in ['\\n', '  ', '\t']):
                    # Replace escaped newlines with actual newlines
                    vector_str = vector_str.replace('\\n', '\n').replace('\\t', ' ')

                    # Split on any whitespace and filter out empty strings
                    import re
                    values = re.split(r'\s+', vector_str.strip())
                    # Filter out empty strings and ellipsis
                    values = [x for x in values if x.strip() and x.strip() != '...' and x.strip() != '']
                    embedding_list = [float(x) for x in values]

                # Handle single space-separated line: "1.0 2.0 3.0"
                else:
                    values = vector_str.split()
                    # Filter out ellipsis and empty values
                    values = [x for x in values if x.strip() and x.strip() != '...' and x.strip() != '']
                    embedding_list = [float(x) for x in values]

                # Validate we have a reasonable number of dimensions
                if embedding_list and len(embedding_list) > 100:  # OpenAI embeddings should have 1536 dimensions
                    # Auto-update actual_dimensions if not set
                    if not self.actual_dimensions or self.actual_dimensions != len(embedding_list):
                        self.actual_dimensions = len(embedding_list)

                    return embedding_list
                else:
                    # Fall through to try database retrieval
                    return self._get_full_embedding_from_db()

            except Exception as e:
                # Try to get full data from database
                try:
                    return self._get_full_embedding_from_db()
                except Exception as db_error:
                    logger.debug(f"Failed to get full embedding from DB: {db_error}")

        # Fallback to legacy LargeBinary format
        if self.model_embedding is not None:
            try:
                # Try numpy array first (CLIP embeddings)
                import numpy as np
                embedding_list = np.frombuffer(self.model_embedding, dtype=np.float32).tolist()

                if embedding_list and len(embedding_list) > 100:
                    # Auto-update actual_dimensions if not set
                    if not self.actual_dimensions or self.actual_dimensions != len(embedding_list):
                        self.actual_dimensions = len(embedding_list)
                    return embedding_list

            except Exception as numpy_error:
                try:
                    # Fallback to JSON format (document embeddings)
                    embedding_list = json.loads(self.model_embedding.decode('utf-8'))

                    if embedding_list and len(embedding_list) > 100:
                        # Auto-update actual_dimensions if not set
                        if not self.actual_dimensions or self.actual_dimensions != len(embedding_list):
                            self.actual_dimensions = len(embedding_list)
                        return embedding_list

                except Exception as json_error:
                    logger.debug(f"Error parsing legacy formats for embedding {getattr(self, 'id', 'unknown')}")

        return []

    def _get_full_embedding_from_db(self):
        """
        Get the full embedding data directly from the database when the cached version is truncated.
        Optimized to reuse session from current context when possible.
        """
        try:
            # Try to reuse existing session from SQLAlchemy context
            from sqlalchemy.orm import object_session
            session = object_session(self)

            if session:
                # Use existing session - much faster
                result = session.execute(
                    text("SELECT embedding_vector FROM document_embedding WHERE id = :embedding_id"),
                    {"embedding_id": self.id}
                ).fetchone()
            else:
                # Fall back to creating new session
                from modules.configuration.config_env import DatabaseConfig
                db_config = DatabaseConfig()
                with db_config.main_session() as new_session:
                    result = new_session.execute(
                        text("SELECT embedding_vector FROM document_embedding WHERE id = :embedding_id"),
                        {"embedding_id": self.id}
                    ).fetchone()

            if result and result[0] is not None:
                full_vector_str = str(result[0]).strip()

                # Parse the full vector string
                if full_vector_str.startswith('[') and full_vector_str.endswith(']'):
                    vector_str = full_vector_str[1:-1]  # Remove brackets
                    embedding_list = [float(x.strip()) for x in vector_str.split(',') if x.strip()]
                else:
                    # Handle space-separated format
                    import re
                    values = re.split(r'\s+', full_vector_str.replace('\\n', '\n').replace('\\t', ' '))
                    embedding_list = [float(x) for x in values if x.strip() and x.strip() != '...' and x.strip() != '']

                logger.debug(f"Retrieved full embedding from DB: {len(embedding_list)} dimensions")
                return embedding_list

        except Exception as e:
            logger.debug(f"Error retrieving full embedding from database: {e}")

        return []

    def _get_full_embedding_from_db(self):
        """
        Get the full embedding data directly from the database when the cached version is truncated.
        """
        try:
            from modules.configuration.config_env import DatabaseConfig

            db_config = DatabaseConfig()
            with db_config.main_session() as session:
                # Query the full embedding vector directly
                result = session.execute(
                    text("SELECT embedding_vector FROM document_embedding WHERE id = :embedding_id"),
                    {"embedding_id": self.id}
                ).fetchone()

                if result and result[0] is not None:
                    full_vector_str = str(result[0]).strip()

                    # Parse the full vector string
                    if full_vector_str.startswith('[') and full_vector_str.endswith(']'):
                        vector_str = full_vector_str[1:-1]  # Remove brackets
                        embedding_list = [float(x.strip()) for x in vector_str.split(',') if x.strip()]
                    else:
                        # Handle space-separated format
                        import re
                        values = re.split(r'\s+', full_vector_str.replace('\\n', '\n').replace('\\t', ' '))
                        embedding_list = [float(x) for x in values if
                                          x.strip() and x.strip() != '...' and x.strip() != '']

                    logger.debug(f"Retrieved full embedding from DB: {len(embedding_list)} dimensions")
                    return embedding_list

        except Exception as e:
            logger.error(f"Error retrieving full embedding from database: {e}")

        return []

    @embedding_as_list.setter
    def embedding_as_list(self, value: List[float]):
        """
        Set embedding from Python list, automatically updating metadata and dimensions.
        """
        if not isinstance(value, list) or not value:
            raise ValueError("Embedding must be a non-empty list of floats")

        # Set actual dimensions
        self.actual_dimensions = len(value)

        # Update or create metadata
        if not self.embedding_metadata:
            self.embedding_metadata = {}

        self.embedding_metadata.update({
            'dimensions': len(value),
            'model_type': self._infer_model_type(len(value)),
            'last_updated': datetime.utcnow().isoformat(),
            'storage_method': 'enhanced_pgvector'
        })

        # Store in pgvector format (preferred)
        try:
            self.embedding_vector = value
            logger.debug(f"Stored embedding in pgvector format: {len(value)} dimensions")
        except Exception as e:
            logger.warning(f"Could not store in pgvector format: {e}")
            # Fallback to legacy format
            self.model_embedding = json.dumps(value).encode('utf-8')
            if self.embedding_metadata:
                self.embedding_metadata['storage_method'] = 'legacy_fallback'

    def _infer_model_type(self, dimensions: int) -> str:
        """
        Infer model type based on embedding dimensions and model name
        """
        model_name_lower = (self.model_name or '').lower()

        # Check model name patterns first
        if 'openai' in model_name_lower or 'text-embedding' in model_name_lower:
            return f'openai_{dimensions}d'
        elif 'tinyllama' in model_name_lower:
            return f'tinyllama_{dimensions}d'
        elif any(x in model_name_lower for x in ['sentence', 'transformer', 'mini', 'mpnet']):
            return f'sentence_transformer_{dimensions}d'
        elif 'clip' in model_name_lower:
            return f'clip_{dimensions}d'

        # Fall back to dimension-based inference
        dimension_mapping = {
            384: 'sentence_transformers_mini',  # all-MiniLM-L6-v2
            768: 'sentence_transformers_base',  # all-mpnet-base-v2
            1024: 'sentence_transformers_large',  # larger sentence transformers
            1536: 'openai_text_embedding_3_small',  # OpenAI small
            3072: 'openai_text_embedding_3_large',  # OpenAI large
            512: 'clip_base',  # CLIP base model
        }

        return dimension_mapping.get(dimensions, f'custom_{dimensions}d')

    def get_storage_type(self) -> str:
        """
        Determine which storage format is being used.
        """
        has_pgvector = self.embedding_vector is not None
        has_legacy = self.model_embedding is not None

        if has_pgvector and has_legacy:
            return 'both'
        elif has_pgvector:
            return 'pgvector'
        elif has_legacy:
            return 'legacy'
        else:
            return 'none'

    def get_embedding_stats(self) -> dict:
        """
        Get comprehensive statistics about the embedding.
        """
        embedding = self.embedding_as_list

        return {
            'id': self.id,
            'document_id': self.document_id,
            'model_name': self.model_name,
            'dimensions': len(embedding),
            'actual_dimensions': self.actual_dimensions,
            'storage_type': self.get_storage_type(),
            'has_data': len(embedding) > 0,
            'metadata': self.embedding_metadata or {},
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'updated_at': self.updated_at.isoformat() if self.updated_at else None,
            'model_info': self._get_model_info()
        }

    def _get_model_info(self) -> dict:
        """Get inferred information about the model and compatibility"""
        if not self.actual_dimensions:
            return {'status': 'no_embedding'}

        return {
            'inferred_type': self._infer_model_type(self.actual_dimensions),
            'dimension_category': self._get_dimension_category(),
            'openai_compatible': self.actual_dimensions in [1536, 3072],
            'sentence_transformers_compatible': self.actual_dimensions in [384, 768, 1024],
            'clip_compatible': self.actual_dimensions in [512, 768],
            'storage_efficiency': 'pgvector' if self.embedding_vector else 'legacy'
        }

    def _get_dimension_category(self) -> str:
        """Categorize embedding by dimension size"""
        if not self.actual_dimensions:
            return 'unknown'
        elif self.actual_dimensions < 400:
            return 'compact'  # 384d models
        elif self.actual_dimensions < 800:
            return 'standard'  # 768d models
        elif self.actual_dimensions < 1600:
            return 'large'  # 1536d OpenAI
        else:
            return 'extra_large'  # 3072d+ models

    def cosine_similarity(self, other_embedding: List[float]) -> Optional[float]:
        """
        Calculate cosine similarity with another embedding.
        Includes dimension compatibility check.
        """
        current_embedding = self.embedding_as_list

        if not current_embedding or not other_embedding:
            return None

        if len(current_embedding) != len(other_embedding):
            logger.warning(f"Dimension mismatch: {len(current_embedding)} vs {len(other_embedding)}")
            return None

        try:
            # Manual cosine similarity calculation
            import math

            dot_product = sum(a * b for a, b in zip(current_embedding, other_embedding))
            norm_a = math.sqrt(sum(a * a for a in current_embedding))
            norm_b = math.sqrt(sum(b * b for b in other_embedding))

            if norm_a == 0 or norm_b == 0:
                return 0.0

            return dot_product / (norm_a * norm_b)
        except Exception as e:
            logger.error(f"Error calculating cosine similarity: {e}")
            return None

    def to_dict(self) -> dict:
        """
        Convert embedding to dictionary for JSON serialization.
        """
        return {
            'id': self.id,
            'document_id': self.document_id,
            'model_name': self.model_name,
            'embedding': self.embedding_as_list,
            'actual_dimensions': self.actual_dimensions,
            'storage_type': self.get_storage_type(),
            'metadata': self.embedding_metadata or {},
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'updated_at': self.updated_at.isoformat() if self.updated_at else None
        }

    # Factory methods for creating embeddings
    @classmethod
    def create_with_pgvector(cls, document_id: int, model_name: str, embedding: List[float], **kwargs):
        """
        Create a new DocumentEmbedding using pgvector storage.
        """
        instance = cls(
            document_id=document_id,
            model_name=model_name,
            **kwargs
        )
        # Use the setter to automatically handle dimensions and metadata
        instance.embedding_as_list = embedding
        return instance

    @classmethod
    def create_with_legacy(cls, document_id: int, model_name: str, embedding: List[float], **kwargs):
        """
        Create a new DocumentEmbedding using legacy LargeBinary storage.
        """
        instance = cls(
            document_id=document_id,
            model_name=model_name,
            model_embedding=json.dumps(embedding).encode('utf-8'),
            actual_dimensions=len(embedding),
            **kwargs
        )

        # Set metadata for legacy storage
        instance.embedding_metadata = {
            'dimensions': len(embedding),
            'model_type': instance._infer_model_type(len(embedding)),
            'storage_method': 'legacy_direct',
            'created': datetime.utcnow().isoformat()
        }

        return instance

    @classmethod
    def create_tinyllama_embedding(cls, document_id: int, model_name: str, embedding: List[float], **kwargs):
        """
        Create a TinyLlama embedding with appropriate metadata.
        """
        instance = cls(
            document_id=document_id,
            model_name=model_name,
            **kwargs
        )

        # Use the setter to handle everything automatically
        instance.embedding_as_list = embedding

        # Add TinyLlama-specific metadata
        if not instance.embedding_metadata:
            instance.embedding_metadata = {}

        instance.embedding_metadata.update({
            'model_family': 'tinyllama',
            'sentence_transformer_base': True,
            'creation_method': 'sentence_transformer_direct'
        })

        return instance

    # Migration and maintenance methods
    def migrate_to_pgvector(self) -> bool:
        """
        Migrate legacy LargeBinary embedding to pgvector format.
        Enhanced to update new columns.
        """
        if self.embedding_vector is not None:
            return True  # Already in pgvector format

        if self.model_embedding is None:
            return False  # No data to migrate

        try:
            # Get embedding as list (this will try both formats)
            embedding_list = self.embedding_as_list

            if not embedding_list:
                return False

            # Store in pgvector format using the setter
            self.embedding_as_list = embedding_list

            # Update metadata to reflect migration
            if not self.embedding_metadata:
                self.embedding_metadata = {}

            self.embedding_metadata.update({
                'migration_timestamp': datetime.utcnow().isoformat(),
                'migrated_from': 'legacy_binary',
                'storage_method': 'migrated_to_pgvector'
            })

            logger.info(f"Successfully migrated embedding {self.id} to pgvector format")
            return True

        except Exception as e:
            logger.error(f"Error migrating embedding {self.id} to pgvector: {e}")
            return False

    @classmethod
    def create_pgvector_indexes(cls, session):
        """
        Create optimized indexes for the new table structure.
        Enhanced for variable dimensions.
        """
        from sqlalchemy import text

        indexes = [
            # Basic indexes
            "CREATE INDEX IF NOT EXISTS idx_doc_embedding_doc_id ON document_embedding (document_id);",
            "CREATE INDEX IF NOT EXISTS idx_doc_embedding_model ON document_embedding (model_name);",
            "CREATE INDEX IF NOT EXISTS idx_doc_embedding_dims ON document_embedding (actual_dimensions);",
            "CREATE INDEX IF NOT EXISTS idx_doc_embedding_created ON document_embedding (created_at);",
            "CREATE INDEX IF NOT EXISTS idx_doc_embedding_metadata ON document_embedding USING gin (embedding_metadata);",

            # Dimension-specific HNSW indexes for common sizes
            """
            CREATE INDEX IF NOT EXISTS idx_doc_embedding_vector_cosine_384d 
            ON document_embedding 
            USING hnsw (embedding_vector vector_cosine_ops)
            WHERE actual_dimensions = 384
            WITH (m = 16, ef_construction = 64);
            """,

            """
            CREATE INDEX IF NOT EXISTS idx_doc_embedding_vector_cosine_768d 
            ON document_embedding 
            USING hnsw (embedding_vector vector_cosine_ops)
            WHERE actual_dimensions = 768
            WITH (m = 16, ef_construction = 64);
            """,

            """
            CREATE INDEX IF NOT EXISTS idx_doc_embedding_vector_cosine_1536d 
            ON document_embedding 
            USING hnsw (embedding_vector vector_cosine_ops)
            WHERE actual_dimensions = 1536
            WITH (m = 16, ef_construction = 64);
            """,

            # L2 distance indexes
            """
            CREATE INDEX IF NOT EXISTS idx_doc_embedding_vector_l2_384d 
            ON document_embedding 
            USING hnsw (embedding_vector vector_l2_ops)
            WHERE actual_dimensions = 384
            WITH (m = 16, ef_construction = 64);
            """,

            """
            CREATE INDEX IF NOT EXISTS idx_doc_embedding_vector_l2_768d 
            ON document_embedding 
            USING hnsw (embedding_vector vector_l2_ops)
            WHERE actual_dimensions = 768
            WITH (m = 16, ef_construction = 64);
            """,

            """
            CREATE INDEX IF NOT EXISTS idx_doc_embedding_vector_l2_1536d 
            ON document_embedding 
            USING hnsw (embedding_vector vector_l2_ops)
            WHERE actual_dimensions = 1536
            WITH (m = 16, ef_construction = 64);
            """,
        ]

        successful_indexes = 0
        for index_sql in indexes:
            try:
                session.execute(text(index_sql))
                session.commit()
                successful_indexes += 1
                logger.debug(f"Created index successfully")
            except Exception as e:
                session.rollback()
                logger.warning(f"Index creation failed (may already exist): {e}")

        logger.info(f"Successfully created {successful_indexes} indexes for document_embedding table")
        return successful_indexes > 0


    @with_request_id
    def _find_most_relevant_document_chunk(cls, question, model_name=None, session=None, request_id=None):
        """
        Find the most relevant document chunk for a given question using vector similarity.

        Optimized for performance with:
        1. pgvector search (fastest)
        2. Fallback to batched processing
        3. Query-side filtering
        4. Simple LRU caching

        Args:
            question (str): The user's question like "how large is the moon"
            model_name (str, optional): Embedding model to use
            session (Session, optional): Database session
            request_id (str, optional): Request ID for logging

        Returns:
            Document: The most relevant text chunk as a Document object, or None
        """
        from plugins.ai_modules import generate_embedding, ModelsConfig
        from modules.configuration.config_env import DatabaseConfig
        import numpy as np

        # Get model name
        if model_name is None:
            model_name = ModelsConfig.get_config_value('embedding', 'CURRENT_MODEL', 'NoEmbeddingModel')

        if model_name == "NoEmbeddingModel":
            logger.info("Embeddings are disabled. Returning None for chunk search.")
            return None

        # Session management
        session_created = False
        if session is None:
            db_config = DatabaseConfig()
            session = db_config.get_main_session()
            session_created = True

        try:
            # Check cache first
            cache_key = f"chunk:{question}:{model_name}"
            if hasattr(cls._find_most_relevant_document_chunk,
                       'cache') and cache_key in cls._find_most_relevant_document_chunk.cache:
                logger.info("Using cached chunk result")
                cached_doc_id = cls._find_most_relevant_document_chunk.cache[cache_key]
                if cached_doc_id is None:
                    return None
                chunk = session.query(Document).get(cached_doc_id)
                if chunk:
                    logger.info(f"Retrieved cached chunk: Document ID {cached_doc_id}")
                return chunk

            # Generate embedding for the question
            question_embedding = generate_embedding(question, model_name)
            if not question_embedding:
                logger.info("No embeddings generated for question. Returning None.")
                cls._cache_chunk_result(cache_key, None)
                return None

            # Try pgvector first (much faster for chunk search!)
            try:
                chunk = cls._search_chunks_with_pgvector(session, question_embedding, model_name)
                if chunk:
                    logger.info(f"pgvector found relevant chunk: Document ID {chunk.id}")
                    cls._cache_chunk_result(cache_key, chunk.id)
                    return chunk
            except Exception as e:
                logger.warning(f"pgvector chunk search failed, falling back to batch processing: {e}")

            # Fallback to batch processing method
            chunk = cls._search_chunks_with_batch_processing(session, question, question_embedding, model_name,
                                                             cache_key)
            return chunk

        except Exception as e:
            logger.error(f"Error in chunk search: {e}")
            return None
        finally:
            if session_created and session:
                session.close()

    @with_request_id
    def _search_chunks_with_pgvector(cls, session, question_embedding, model_name):
        """Fast pgvector search for document chunks - preferred method"""
        from sqlalchemy import text

        query_vector_str = '[' + ','.join(map(str, question_embedding)) + ']'

        # Search for chunks with content (not empty chunks)
        search_query = text("""
            SELECT 
                de.document_id,
                d.content,
                d.name as chunk_name,
                cd.title as document_title,
                1 - (de.embedding_vector <=> :query_vector) AS similarity
            FROM document_embedding de
            JOIN document d ON de.document_id = d.id
            LEFT JOIN complete_document cd ON d.complete_document_id = cd.id
            WHERE de.model_name = :model_name
              AND de.embedding_vector IS NOT NULL
              AND d.content IS NOT NULL
              AND LENGTH(d.content) > 50
              AND (1 - (de.embedding_vector <=> :query_vector)) >= 0.01
            ORDER BY de.embedding_vector <=> :query_vector ASC
            LIMIT 1
        """)

        result = session.execute(search_query, {
            'query_vector': query_vector_str,
            'model_name': model_name
        }).fetchone()

        if result:
            doc_id, content, chunk_name, doc_title, similarity = result
            logger.info(f"pgvector found chunk {doc_id} with similarity {similarity:.4f} from '{doc_title}'")

            chunk = session.query(Document).get(doc_id)
            if chunk:
                # Attach similarity metadata
                chunk._similarity_score = float(similarity)
                chunk._search_metadata = {
                    'method': 'pgvector',
                    'similarity': float(similarity),
                    'document_title': doc_title,
                    'chunk_name': chunk_name
                }
            return chunk

        return None

    @classmethod
    def _cache_chunk_result(cls, cache_key, doc_id):
        """Cache chunk search results"""
        if not hasattr(cls._find_most_relevant_document_chunk, 'cache'):
            cls._find_most_relevant_document_chunk.cache = {}

        cache = cls._find_most_relevant_document_chunk.cache
        if len(cache) > 100:  # Keep cache size reasonable
            # Remove oldest entry (simple LRU)
            oldest_key = next(iter(cache))
            del cache[oldest_key]

        cache[cache_key] = doc_id
        logger.debug(f"Cached chunk result: {cache_key} -> {doc_id}")

def get_embeddings_by_model_and_dimensions(session, model_pattern: str = None, dimensions: int = None,
                                           limit: int = 100):
    """
    Query embeddings by model pattern and/or dimensions
    """
    query = session.query(DocumentEmbedding)

    if model_pattern:
        query = query.filter(DocumentEmbedding.model_name.ilike(f'%{model_pattern}%'))

    if dimensions:
        query = query.filter(DocumentEmbedding.actual_dimensions == dimensions)

    return query.limit(limit).all()
def get_dimension_statistics(session):
    """
    Get statistics about embedding dimensions in the database
    """
    from sqlalchemy import func

    results = session.query(
        DocumentEmbedding.actual_dimensions,
        func.count(DocumentEmbedding.id).label('count'),
        func.array_agg(DocumentEmbedding.model_name.distinct()).label('models')
    ).filter(
        DocumentEmbedding.actual_dimensions.isnot(None)
    ).group_by(
        DocumentEmbedding.actual_dimensions
    ).order_by(
        DocumentEmbedding.actual_dimensions
    ).all()

    stats = {}
    for result in results:
        stats[result.actual_dimensions] = {
            'count': result.count,
            'models': list(set(result.models)) if result.models else []
        }

    return stats

class CompleteDocument(Base):
    """
    Modern document model with robust PostgreSQL database handling.
    Streamlined for PostgreSQL-only operations with enhanced performance.
    Now includes image extraction capabilities aligned with Image class and DocumentStructureManager.
    """

    __tablename__ = 'complete_document'

    # Core fields
    id = Column(Integer, primary_key=True)
    title = Column(String, nullable=False)
    file_path = Column(String)
    content = Column(Text)
    rev = Column(String, nullable=False, default="R0")

    # Relationships
    document = relationship("Document", back_populates="complete_document")
    completed_document_position_association = relationship(
        "CompletedDocumentPositionAssociation",
        back_populates="complete_document"
    )
    powerpoint = relationship("PowerPoint", back_populates="complete_document")
    image_completed_document_association = relationship(
        "ImageCompletedDocumentAssociation",
        back_populates="complete_document"
    )
    complete_document_problem = relationship(
        "CompleteDocumentProblemAssociation",
        back_populates="complete_document"
    )
    complete_document_task = relationship(
        "CompleteDocumentTaskAssociation",
        back_populates="complete_document"
    )

    def __repr__(self):
        return f"<CompleteDocument(id={self.id}, title='{self.title}', rev='{self.rev}')>"

    # =====================================================
    # SIMPLE MODEL REFERENCES (SAME MODULE)
    # =====================================================

    @classmethod
    def _get_position_class(cls):
        """Get Position class from same module."""
        return globals().get('Position')

    @classmethod
    def _get_site_location_class(cls):
        """Get SiteLocation class from same module."""
        return globals().get('SiteLocation')

    @classmethod
    def _get_association_class(cls):
        """Get CompletedDocumentPositionAssociation class from same module."""
        return globals().get('CompletedDocumentPositionAssociation')

    @classmethod
    def _get_document_class(cls):
        """Get Document class from same module."""
        return globals().get('Document')

    @classmethod
    def _get_image_class(cls):
        """Get Image class from same module."""
        return globals().get('Image')

    @classmethod
    def _get_image_association_class(cls):
        """Get ImageCompletedDocumentAssociation class from same module."""
        return globals().get('ImageCompletedDocumentAssociation')

    # =====================================================
    # PUBLIC API - 3 MAIN METHODS (Enhanced for PostgreSQL)
    # =====================================================

    @classmethod
    @with_request_id
    def process_upload(cls, files, metadata, request_id=None):
        """
        Main upload processing method - now optimized for PostgreSQL with concurrent processing.
        """
        valid_files = [f for f in files if f.filename.strip()]
        if not valid_files:
            warning_id("No valid files provided", request_id)
            return False, {"error": "No valid files provided"}, 400

        info_id(f"Processing {len(valid_files)} files using PostgreSQL with concurrent processing", request_id)

        try:
            position_id = cls._create_position(metadata, request_id)

            # Always use concurrent processing for PostgreSQL (enhanced performance)
            return cls._process_concurrent(valid_files, metadata, position_id, request_id)

        except Exception as e:
            error_id(f"Upload failed: {e}", request_id)
            return False, {"error": str(e)}, 500

    @classmethod
    @with_request_id
    def search_documents(cls, query, limit=50, request_id=None):
        """Search documents using PostgreSQL full-text search with ranking."""
        if not query or not query.strip():
            return []

        db_config = DatabaseConfig()

        with db_config.main_session() as session:
            return cls._search_postgresql(session, query, limit, request_id)

    @classmethod
    @with_request_id
    def find_similar(cls, document_id, threshold=0.3, limit=10, request_id=None):
        """Find documents similar to the given document using PostgreSQL similarity functions."""
        db_config = DatabaseConfig()

        with db_config.main_session() as session:
            source = session.query(cls).filter_by(id=document_id).first()
            if not source:
                return []

            return cls._similar_postgresql(session, source, threshold, limit, request_id)

    @classmethod
    def dynamic_search(cls, session, **filters):
        """
        Dynamic search with explicit relationship handling for CompleteDocument.
        This approach explicitly handles the common search patterns used in the application.
        """
        from sqlalchemy import and_

        query = session.query(cls)
        filter_conditions = []

        # Track if we need to join with position-related tables
        needs_position_join = False
        needs_association_join = False

        for key, value in filters.items():
            if key.startswith('completed_document_position_association__position__'):
                # This is a position-related search through the association
                needs_association_join = True
                needs_position_join = True

                # Extract the position attribute name
                position_attr = key.replace('completed_document_position_association__position__', '')

                # Get the Position class
                PositionClass = cls._get_position_class()
                if PositionClass and hasattr(PositionClass, position_attr):
                    attr = getattr(PositionClass, position_attr)
                    if isinstance(value, str):
                        condition = attr.ilike(f"%{value}%")
                    else:
                        condition = attr == value
                    filter_conditions.append(condition)

            elif key.startswith('completed_document_position_association__'):
                # This is an association-related search
                needs_association_join = True

                # Extract the association attribute name
                assoc_attr = key.replace('completed_document_position_association__', '')

                # Get the Association class
                AssociationClass = cls._get_association_class()
                if AssociationClass and hasattr(AssociationClass, assoc_attr):
                    attr = getattr(AssociationClass, assoc_attr)
                    if isinstance(value, str):
                        condition = attr.ilike(f"%{value}%")
                    else:
                        condition = attr == value
                    filter_conditions.append(condition)

            elif key in ['title', 'content', 'file_path', 'rev']:
                # Direct attributes on CompleteDocument
                attr = getattr(cls, key, None)
                if attr:
                    if isinstance(value, str):
                        condition = attr.ilike(f"%{value}%")
                    else:
                        condition = attr == value
                    filter_conditions.append(condition)

            else:
                # Handle other relationship patterns or direct attributes
                if hasattr(cls, key):
                    attr = getattr(cls, key)
                    if isinstance(value, str):
                        condition = attr.ilike(f"%{value}%")
                    else:
                        condition = attr == value
                    filter_conditions.append(condition)

        # Add necessary joins
        if needs_association_join:
            AssociationClass = cls._get_association_class()
            if AssociationClass:
                query = query.join(AssociationClass, cls.id == AssociationClass.complete_document_id)

                if needs_position_join:
                    PositionClass = cls._get_position_class()
                    if PositionClass:
                        query = query.join(PositionClass, AssociationClass.position_id == PositionClass.id)

        # Apply all filter conditions
        if filter_conditions:
            query = query.filter(and_(*filter_conditions))

        # Add distinct to avoid duplicates from joins
        query = query.distinct()

        return query.all()

    # =====================================================
    # PROCESSING STRATEGIES (PostgreSQL Optimized) - FIXED
    # =====================================================

    @classmethod
    @with_request_id
    def _process_concurrent(cls, files, metadata, position_id, request_id):
        max_workers = min(len(files), 4)
        info_id(f"PostgreSQL: Using {max_workers} concurrent workers for optimal performance", request_id)
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(cls._process_file, f, metadata, position_id, request_id): f
                for f in files
            }
            results = []
            for future in as_completed(futures):
                file = futures[future]
                try:
                    success, error, status = future.result()
                    results.append((file.filename, success, error, status))
                    if success:
                        debug_id(f"SUCCESS: {file.filename}", request_id)
                    else:
                        # FIXED: Safe error handling
                        error_msg = error.get('error', 'Unknown error') if isinstance(error, dict) else str(error)
                        warning_id(f"FAILED: {file.filename} - {error_msg}", request_id)
                except Exception as e:
                    error_id(f"Error processing {file.filename}: {e}", request_id)
                    results.append((file.filename, False, {"error": str(e)}, 500))

            if all(r[1] for r in results):
                cls._optimize_postgresql(request_id)
                info_id(f"PostgreSQL processing complete: {len(files)}/{len(files)} successful", request_id)
                return True, None, 200

            errors = [r[2] for r in results if not r[1]]
            success_count = sum(r[1] for r in results)
            info_id(f"PostgreSQL processing complete: {success_count}/{len(files)} successful", request_id)
            # FIXED: Safe max calculation
            error_statuses = [r[3] for r in results if not r[1]]
            max_status = max(error_statuses) if error_statuses else 500
            return False, {"errors": errors}, max_status

    @classmethod
    @with_request_id
    def _process_file(cls, file, metadata, position_id, request_id):
        """Enhanced file processing with correct DATABASE_DOC storage location."""
        try:
            # Import the proper database configuration
            from modules.configuration.config import DATABASE_DIR

            filename = secure_filename(file.filename)

            # FIXED: Use DATABASE_DOC instead of generic Uploads folder
            DATABASE_DOC = os.path.join(DATABASE_DIR, 'DB_DOC')
            os.makedirs(DATABASE_DOC, exist_ok=True)
            file_path = os.path.join(DATABASE_DOC, filename)

            # Handle filename conflicts
            counter = 1
            original_file_path = file_path
            while os.path.exists(file_path):
                name, ext = os.path.splitext(filename)
                file_path = os.path.join(DATABASE_DOC, f"{name}_{counter}{ext}")
                counter += 1

            file.save(file_path)
            info_id(f"Saved file to: {file_path}", request_id)

            title = metadata.get('title') or cls._clean_filename(filename)
            content = cls._extract_content(file_path, request_id)

            if not content:
                warning_id(f"No content extracted from {filename}", request_id)
                return False, {"error": f"No text content in {filename}"}, 400

            # Save document and get ID
            document_id = cls._save_document_and_get_id(title, file_path, content, position_id, request_id)
            if not document_id:
                error_id(f"Failed to save document {filename}", request_id)
                return False, {"error": f"Failed to save document {filename}"}, 500

            # Extract images with guided association
            try:
                extracted_count = cls._extract_images_with_guided_association(
                    file_path=file_path,
                    document_id=document_id,
                    position_id=position_id,
                    request_id=request_id
                )

                if extracted_count > 0:
                    info_id(f"Created {extracted_count} intelligent image-chunk associations for {filename}",
                            request_id)
            except Exception as img_error:
                warning_id(f"Image extraction failed for {filename}: {img_error}", request_id)

            return True, None, 200

        except Exception as e:
            error_id(f"File processing failed for {file.filename}: {e}", request_id)
            return False, {"error": str(e)}, 500

    @classmethod
    @with_request_id
    def _extract_images_with_guided_association(cls, file_path, document_id, position_id=None, request_id=None):
        """
        FIXED: Use the aligned ImageCompletedDocumentAssociation.guided_extraction_with_mapping method.
        """
        try:
            info_id(f"Starting guided image extraction for document {document_id}", request_id)

            # Get the ImageCompletedDocumentAssociation class
            AssociationClass = cls._get_image_association_class()
            if not AssociationClass:
                warning_id("ImageCompletedDocumentAssociation class not available, falling back to basic extraction",
                           request_id)
                return cls._extract_images_basic(file_path, document_id, position_id, request_id)

            # Prepare metadata for the association method
            metadata = {
                'complete_document_id': document_id,
                'position_id': position_id
            }

            # Use the aligned guided extraction method
            success, result, status = AssociationClass.guided_extraction_with_mapping(
                file_path=file_path,
                metadata=metadata,
                request_id=request_id
            )

            if success:
                associations_created = result.get('associations_created', 0)
                info_id(f"Guided extraction successful: {associations_created} associations created", request_id)
                return associations_created
            else:
                warning_id(f"Guided extraction failed: {result.get('error', 'Unknown error')}", request_id)
                # Fallback to basic extraction
                return cls._extract_images_basic(file_path, document_id, position_id, request_id)

        except Exception as e:
            error_id(f"Error in guided image extraction: {e}", request_id)
            # Fallback to basic extraction
            return cls._extract_images_basic(file_path, document_id, position_id, request_id)

    @classmethod
    @with_request_id
    def _extract_images_basic(cls, file_path, document_id, position_id, request_id):
        """FIXED: Fallback basic image extraction with proper session management."""
        ext = os.path.splitext(file_path)[1].lower()

        if ext == '.pdf':
            return cls._extract_pdf_images(file_path, document_id, position_id, request_id)
        elif ext in ['.docx', '.doc']:
            return cls._extract_docx_images(file_path, document_id, position_id, request_id)
        else:
            debug_id(f"Image extraction not supported for {ext} files", request_id)
            return 0

    # =====================================================
    # IMAGE EXTRACTION METHODS (FIXED)
    # =====================================================

    @classmethod
    @with_request_id
    def _extract_pdf_images(cls, file_path, document_id, position_id, request_id):
        try:
            import fitz  # PyMuPDF

            ImageClass = cls._get_image_class()
            if ImageClass is None:
                error_id("Image class not available, skipping image extraction", request_id)
                return 0

            extracted_count = 0
            doc = fitz.open(file_path)
            file_name = os.path.splitext(os.path.basename(file_path))[0]

            db_config = DatabaseConfig()
            with db_config.main_session() as session:
                db_manager = PostgreSQLDatabaseManager(session=session, request_id=request_id)
                chunks = session.query(Document).filter_by(complete_document_id=document_id).order_by(Document.id).all()

                for page_num in range(len(doc)):
                    page = doc[page_num]
                    img_list = page.get_images(full=True)

                    page_chunks = [(i, chunk) for i, chunk in enumerate(chunks) if
                                   chunk.doc_metadata and chunk.doc_metadata.get("page_number") == page_num]
                    if not page_chunks:
                        warning_id(f"No chunks found for page {page_num + 1}, skipping image association", request_id)
                        continue

                    for img_index, img in enumerate(img_list):
                        try:
                            xref = img[0]
                            base_image = doc.extract_image(xref)
                            image_bytes = base_image["image"]
                            ext = base_image.get("ext", "jpg")

                            with tempfile.NamedTemporaryFile(suffix=f'.{ext}', delete=False) as tmp:
                                tmp.write(image_bytes)
                                temp_path = tmp.name

                            title = f"{file_name} - Page {page_num + 1} Image {img_index + 1}"

                            chunk_index, nearest_chunk = page_chunks[img_index % len(page_chunks)]
                            debug_id(
                                f"Selected chunk for image on page {page_num + 1}: chunk_index={chunk_index}, chunk_id={nearest_chunk.id}",
                                request_id)

                            metadata = {
                                'page_number': page_num,
                                'image_index': img_index,
                                'extraction_method': 'basic_pdf',
                                'structure_guided': False
                            }

                            image_id = ImageClass.add_to_db(
                                session=session,
                                title=title,
                                file_path=temp_path,
                                description=f"Basic extraction from {os.path.basename(file_path)}",
                                position_id=position_id,
                                complete_document_id=document_id,
                                metadata=metadata,
                                request_id=request_id
                            )

                            if image_id is not None:
                                assoc = ImageCompletedDocumentAssociation(
                                    complete_document_id=document_id,
                                    image_id=image_id,
                                    document_id=nearest_chunk.id,
                                    page_number=page_num,
                                    chunk_index=chunk_index,
                                    association_method='basic_pdf',
                                    confidence_score=0.5,
                                    context_metadata={'extraction_method': 'basic'}
                                )
                                session.add(assoc)
                                extracted_count += 1
                                debug_id(f"Associated image {image_id} with chunk {nearest_chunk.id}", request_id)

                            try:
                                os.unlink(temp_path)
                            except:
                                pass

                        except Exception as e:
                            error_id(f"Error extracting image {img_index + 1} on page {page_num + 1}: {e}", request_id)
                            continue

                db_manager.commit_with_retry()
                doc.close()
                info_id(f"Extracted {extracted_count} images from PDF", request_id)
                return extracted_count

        except Exception as e:
            error_id(f"PDF image extraction failed: {e}", request_id)
            return 0

    @classmethod
    @with_request_id
    def _extract_docx_images(cls, file_path, document_id, position_id, request_id):
        """FIXED: Extract DOCX images using Image.add_to_db method with proper session management."""
        try:
            ImageClass = cls._get_image_class()
            if ImageClass is None:
                error_id("Image class not available, skipping image extraction", request_id)
                return 0

            extracted_count = 0
            file_name = os.path.splitext(os.path.basename(file_path))[0]

            with zipfile.ZipFile(file_path, 'r') as docx_zip:
                image_files = [f for f in docx_zip.namelist() if f.startswith('word/media/')]

                info_id(f"Found {len(image_files)} images in DOCX file", request_id)

                for idx, img_path in enumerate(image_files):
                    try:
                        # Extract image data
                        image_data = docx_zip.read(img_path)

                        # Determine extension
                        original_ext = os.path.splitext(img_path)[1].lower()
                        if not original_ext:
                            original_ext = '.png'

                        # Create temp file
                        with tempfile.NamedTemporaryFile(suffix=original_ext, delete=False) as temp_file:
                            temp_file.write(image_data)
                            temp_path = temp_file.name

                        title = f"{file_name} - Image {idx + 1}"

                        # Enhanced metadata for DOCX extraction
                        metadata = {
                            'image_index': idx,
                            'extraction_method': 'basic_docx',
                            'structure_guided': False,
                            'original_path': img_path
                        }

                        # FIXED: Use Image.add_to_db with session=None
                        success = ImageClass.add_to_db(
                            session=None,  # Let Image.add_to_db create its own session
                            title=title,
                            file_path=temp_path,
                            description=f"Basic extraction from {os.path.basename(file_path)}",
                            position_id=position_id,
                            complete_document_id=document_id,
                            metadata=metadata,
                            request_id=request_id
                        )

                        if success:
                            extracted_count += 1
                            debug_id(f"Successfully processed DOCX image {idx + 1}", request_id)

                        # Clean up temp file
                        try:
                            os.unlink(temp_path)
                        except:
                            pass

                    except Exception as e:
                        error_id(f"Error extracting DOCX image {idx + 1}: {e}", request_id)
                        continue

            info_id(f"Extracted {extracted_count} images from DOCX", request_id)
            return extracted_count

        except Exception as e:
            error_id(f"DOCX image extraction failed: {e}", request_id)
            return 0

    # =====================================================
    # DATABASE OPERATIONS (PostgreSQL Optimized)
    # =====================================================

    @classmethod
    @with_request_id
    def _save_document_and_get_id(cls, title, file_path, content, position_id, request_id):
        """Save document and return the document ID for image associations."""
        db_config = DatabaseConfig()

        with db_config.main_session() as session:
            try:
                # Set PostgreSQL-specific optimizations
                session.execute(text("SET work_mem = '256MB'"))
                session.execute(text("SET maintenance_work_mem = '512MB'"))

                # Upsert document
                document_id = cls._upsert_document(session, title, file_path, content)

                # Create associations
                cls._create_associations(session, document_id, position_id)

                # Add to PostgreSQL search index
                cls._add_to_search_safe(session, title, content, request_id)

                # Create chunks
                cls._create_chunks(session, document_id, title, content, file_path)

                session.commit()
                debug_id(f"Saved document to PostgreSQL with ID {document_id}: {title}", request_id)
                return document_id

            except Exception as e:
                session.rollback()
                error_id(f"PostgreSQL database save failed for {title}: {e}", request_id)
                return None

    @classmethod
    def _upsert_document(cls, session, title, file_path, content):
        """Create or update document with correct relative path storage."""
        try:
            from modules.configuration.config import DATABASE_DIR

            # Ensure content is properly encoded as UTF-8 string
            if content:
                if isinstance(content, bytes):
                    content = content.decode('utf-8', errors='replace')
                else:
                    content = content.encode('utf-8', errors='replace').decode('utf-8')

            # Ensure title is also properly encoded
            if isinstance(title, bytes):
                title = title.decode('utf-8', errors='replace')
            else:
                title = title.encode('utf-8', errors='replace').decode('utf-8')

            # Calculate the correct relative path from DATABASE_DIR
            DATABASE_DOC = os.path.join(DATABASE_DIR, 'DB_DOC')
            if file_path.startswith(DATABASE_DOC):
                # Store path relative to DATABASE_DOC (just the filename)
                relative_path = os.path.relpath(file_path, DATABASE_DOC)
            else:
                # Fallback: store relative to DATABASE_DIR
                relative_path = os.path.relpath(file_path, DATABASE_DIR)

            # First try to find existing document
            existing = session.query(cls).filter_by(title=title).first()

            if existing:
                # Update existing document
                if existing.content != content:
                    existing.content = content
                    existing.file_path = relative_path  # Store correct relative path
                    rev_num = int(existing.rev[1:]) + 1 if existing.rev.startswith('R') else 1
                    existing.rev = f"R{rev_num}"
                session.flush()
                return existing.id
            else:
                # Create new document
                doc = cls(
                    title=title,
                    file_path=relative_path,  # Store correct relative path
                    content=content,
                    rev="R0"
                )
                session.add(doc)
                session.flush()
                return doc.id

        except Exception as e:
            # Enhanced error handling for encoding issues
            error_msg = str(e)
            if 'codec' in error_msg.lower() or 'encode' in error_msg.lower():
                debug_id(f"Encoding error in upsert, attempting content sanitization: {e}")

                # Sanitize content by removing problematic characters
                if content:
                    content = unicodedata.normalize('NFKD', content)
                    content = content.encode('ascii', errors='replace').decode('ascii')

                if title:
                    title = unicodedata.normalize('NFKD', title)
                    title = title.encode('ascii', errors='replace').decode('ascii')

                try:
                    # Retry with sanitized content
                    doc = cls(
                        title=f"{title}_{uuid.uuid4().hex[:8]}",
                        file_path=relative_path,  # Use the correct relative path
                        content=content,
                        rev="R0"
                    )
                    session.add(doc)
                    session.flush()
                    return doc.id
                except Exception as retry_error:
                    debug_id(f"Retry with sanitized content also failed: {retry_error}")

            # Final fallback to unique title if there's still a conflict
            debug_id(f"PostgreSQL upsert fallback for {title}: {e}")
            doc = cls(
                title=f"{title}_{uuid.uuid4().hex[:8]}",  # Make unique
                file_path=relative_path,  # Use the correct relative path
                content="[Content encoding error - original content could not be saved]",
                rev="R0"
            )
            session.add(doc)
            session.flush()
            return doc.id

    @classmethod
    @with_request_id
    def _optimize_postgresql(cls, request_id):
        """PostgreSQL optimization that integrates with the database manager patterns."""
        try:
            # Use the PostgreSQL optimization from db_manager
            from modules.database_manager.db_manager import PostgreSQLDatabaseManager
            PostgreSQLDatabaseManager._optimize_database(request_id)
        except Exception as e:
            debug_id(f"PostgreSQL optimization skipped: {e}", request_id)

    # =====================================================
    # ADDITIONAL ALIGNED METHODS
    # =====================================================

    @classmethod
    @with_request_id
    def get_images_with_chunk_context(cls, document_id, request_id=None):
        """
        ALIGNED: Get all images for a document with their associated chunk context.
        Uses the Image class's enhanced query methods.
        """
        try:
            ImageClass = cls._get_image_class()
            if not ImageClass:
                error_id("Image class not available", request_id)
                return []

            db_config = DatabaseConfig()
            with db_config.main_session() as session:
                return ImageClass.get_images_with_chunk_context(session, document_id, request_id)

        except Exception as e:
            error_id(f"Failed to get images with chunk context: {e}", request_id)
            return []

    @classmethod
    @with_request_id
    def search_images_by_chunk_text(cls, search_text, document_id=None, confidence_threshold=0.5, request_id=None):
        """
        ALIGNED: Search for images by their associated chunk text content.
        Uses the Image class's enhanced search methods.
        """
        try:
            ImageClass = cls._get_image_class()
            if not ImageClass:
                error_id("Image class not available", request_id)
                return []

            db_config = DatabaseConfig()
            with db_config.main_session() as session:
                return ImageClass.search_by_chunk_text(
                    session, search_text, document_id, confidence_threshold, request_id
                )

        except Exception as e:
            error_id(f"Failed to search images by chunk text: {e}", request_id)
            return []

    @classmethod
    @with_request_id
    def get_association_statistics(cls, document_id, request_id=None):
        """
        ALIGNED: Get statistics about image-chunk associations.
        Uses the Image class's enhanced statistics methods.
        """
        try:
            ImageClass = cls._get_image_class()
            if not ImageClass:
                error_id("Image class not available", request_id)
                return {}

            db_config = DatabaseConfig()
            with db_config.main_session() as session:
                return ImageClass.get_association_statistics(session, document_id, request_id)

        except Exception as e:
            error_id(f"Failed to get association statistics: {e}", request_id)
            return {}

    # =====================================================
    # EXISTING METHODS (FIXED)
    # =====================================================

    @classmethod
    def _create_associations(cls, session, document_id, position_id):
        """FIXED: Create document-position associations with PostgreSQL optimization and null check."""
        AssociationClass = cls._get_association_class()

        if AssociationClass is None:
            debug_id("Association class not available, skipping", None)
            return

        # FIXED: Add null check for position_id
        if position_id is None:
            debug_id("No position_id provided, skipping association creation", None)
            return

        # Check if association already exists using PostgreSQL-optimized query
        existing = session.query(AssociationClass).filter_by(
            complete_document_id=document_id,
            position_id=position_id
        ).first()

        if not existing:
            assoc = AssociationClass(
                complete_document_id=document_id,
                position_id=position_id
            )
            session.add(assoc)

    @classmethod
    @with_request_id
    def _add_to_search_safe(cls, session, title, content, request_id):
        """Add document to PostgreSQL FTS index with enhanced Unicode safety."""
        try:
            # Ensure content is properly encoded
            if isinstance(title, bytes):
                title = title.decode('utf-8', errors='replace')
            if isinstance(content, bytes):
                content = content.decode('utf-8', errors='replace')

            # Truncate very long content for FTS to avoid memory issues
            if content and len(content) > 1000000:  # 1MB limit
                content = content[:1000000]
                debug_id("Truncated content for FTS indexing", request_id)

            # Use a savepoint for PostgreSQL FTS to avoid transaction abort
            savepoint = session.begin_nested()
            try:
                # Enhanced PostgreSQL FTS with Unicode safety
                sql = text("""
                    INSERT INTO documents_fts (title, content, search_vector)
                    VALUES (:title, :content, to_tsvector('english', :title || ' ' || :content))
                    ON CONFLICT (title) DO UPDATE SET
                        content = EXCLUDED.content,
                        search_vector = EXCLUDED.search_vector,
                        updated_at = CURRENT_TIMESTAMP
                """)
                session.execute(sql, {'title': title, 'content': content})
                savepoint.commit()
                debug_id("Added to PostgreSQL FTS with Unicode safety", request_id)
            except Exception as fts_error:
                savepoint.rollback()
                # Try with ASCII-only content as fallback
                try:
                    safe_title = unicodedata.normalize('NFKD', title).encode('ascii', errors='replace').decode('ascii')
                    safe_content = unicodedata.normalize('NFKD', content).encode('ascii', errors='replace').decode(
                        'ascii')

                    savepoint = session.begin_nested()
                    sql = text("""
                        INSERT INTO documents_fts (title, content, search_vector)
                        VALUES (:title, :content, to_tsvector('english', :title || ' ' || :content))
                        ON CONFLICT (title) DO UPDATE SET
                            content = EXCLUDED.content,
                            search_vector = EXCLUDED.search_vector,
                            updated_at = CURRENT_TIMESTAMP
                    """)
                    session.execute(sql, {'title': safe_title, 'content': safe_content})
                    savepoint.commit()
                    debug_id("Added to PostgreSQL FTS with ASCII fallback", request_id)
                except Exception as ascii_error:
                    savepoint.rollback()
                    debug_id(f"PostgreSQL FTS completely skipped: {ascii_error}", request_id)
        except Exception as e:
            debug_id(f"Search indexing completely skipped: {e}", request_id)

    @classmethod
    @with_request_id
    def _create_chunks(cls, session, document_id, title, content, file_path=None, request_id=None):
        debug_id("Starting _create_chunks", request_id)
        DocumentClass = cls._get_document_class()

        if DocumentClass is None:
            debug_id("Document class not available, skipping chunk creation", request_id)
            return

        try:
            if not file_path:
                parent_doc = session.query(cls).filter_by(id=document_id).first()
                file_path = parent_doc.file_path if parent_doc else "unknown"

            doc = fitz.open(file_path)
            chunk_objects = []
            chunk_counter = 0

            for page_num in range(len(doc)):
                page = doc[page_num]
                page_text = page.get_text()

                # Split the page text into smaller chunks of 150 words
                page_chunks = cls._split_text(page_text, 150)

                for page_chunk in page_chunks:
                    if not page_chunk.strip():
                        continue

                    chunk_counter += 1
                    doc_chunk = DocumentClass(
                        name=f"{title} - Chunk {chunk_counter}",
                        file_path=file_path,
                        content=page_chunk.strip(),
                        complete_document_id=document_id,
                        rev="R0",
                        doc_metadata={"page_number": page_num}
                    )
                    chunk_objects.append(doc_chunk)

            doc.close()

            if chunk_objects:
                session.add_all(chunk_objects)
                session.flush()
                cls._generate_embeddings_for_chunks(session, chunk_objects)
                debug_id(f"Created {len(chunk_objects)} document chunks with embeddings", request_id)
            else:
                debug_id("No chunks created", request_id)

        except Exception as e:
            debug_id(f"Chunk creation skipped: {e}", request_id)

    @classmethod
    @with_request_id
    def _generate_embeddings_for_chunks(cls, session, chunk_objects, request_id=None):
        """
        Updated method to generate embeddings for document chunks using pgvector storage.
        Now uses the enhanced DocumentEmbedding class with pgvector support.
        """
        debug_id("Starting _generate_embeddings_for_chunks with pgvector support", request_id)

        try:
            # Import the necessary modules
            from plugins.ai_modules import generate_embedding, ModelsConfig

            # Get current embedding model name
            current_embedding_model = ModelsConfig.get_config_value('embedding', 'CURRENT_MODEL')

            if current_embedding_model == "NoEmbeddingModel":
                debug_id("Embedding generation disabled, skipping", request_id)
                return

            debug_id(f"Generating embeddings for {len(chunk_objects)} chunks using {current_embedding_model}",
                     request_id)

            # Process each chunk
            for chunk in chunk_objects:
                try:
                    # Generate embedding for this chunk
                    embeddings = generate_embedding(chunk.content, current_embedding_model)

                    if embeddings:
                        # Store using the enhanced DocumentEmbedding with pgvector
                        # FIXED: Use _store_embedding instead of _store_embedding_pgvector
                        success = cls._store_embedding(
                            session, chunk.id, embeddings, current_embedding_model, request_id
                        )

                        if success:
                            debug_id(f"Generated and stored pgvector embedding for chunk: {chunk.name}", request_id)
                        else:
                            debug_id(f"Failed to store embedding for chunk: {chunk.name}", request_id)
                    else:
                        debug_id(f"No embedding generated for chunk: {chunk.name}", request_id)

                except Exception as e:
                    debug_id(f"Error generating embedding for chunk {chunk.name}: {e}", request_id)
                    import traceback
                    debug_id(f"Traceback: {traceback.format_exc()}", request_id)
                    continue

        except Exception as e:
            debug_id(f"Embedding generation for chunks failed: {e}", request_id)
            import traceback
            debug_id(f"Traceback: {traceback.format_exc()}", request_id)

    @classmethod
    @with_request_id
    def _store_embedding(cls, session, document_id, embeddings, model_name, request_id=None):
        """
        Store embeddings using the enhanced DocumentEmbedding class with pgvector support.
        Updated to use pgvector while maintaining the original method name.

        Args:
            session: Database session
            document_id: ID of the document chunk
            embeddings: List of embedding values
            model_name: Name of the embedding model
            request_id: Request ID for logging

        Returns:
            bool: Success status
        """
        try:
            # Import DocumentEmbedding class
            DocumentEmbeddingClass = cls._get_document_embedding_class()

            if DocumentEmbeddingClass is None:
                error_id("DocumentEmbedding class not available", request_id)
                return False

            if embeddings is None or len(embeddings) == 0:
                warning_id(f"No embeddings to store for document ID {document_id}", request_id)
                return False

            # Use PostgreSQL savepoint for transaction safety
            savepoint = session.begin_nested()
            try:
                # Check if embedding already exists
                existing = session.query(DocumentEmbeddingClass).filter_by(
                    document_id=document_id,
                    model_name=model_name
                ).first()

                if existing:
                    # Update existing embedding using the enhanced property
                    existing.embedding_as_list = embeddings
                    debug_id(f"Updated existing pgvector embedding for document ID {document_id}", request_id)
                else:
                    # Create new embedding using the enhanced factory method
                    document_embedding = DocumentEmbeddingClass.create_with_pgvector(
                        document_id=document_id,
                        model_name=model_name,
                        embedding=embeddings
                    )
                    session.add(document_embedding)
                    debug_id(f"Created new pgvector embedding for document ID {document_id}", request_id)

                session.flush()  # Flush within savepoint
                savepoint.commit()  # Commit savepoint
                return True

            except Exception as savepoint_error:
                savepoint.rollback()  # Rollback only the savepoint
                error_id(f"Savepoint rolled back for pgvector embedding storage: {savepoint_error}", request_id)
                raise

        except Exception as e:
            error_id(f"Error storing pgvector embedding for document {document_id}: {e}", request_id)
            import traceback
            debug_id(f"Traceback: {traceback.format_exc()}", request_id)
            return False

    @classmethod
    def _get_document_embedding_class(cls):
        """Get DocumentEmbedding class from same module."""
        return globals().get('DocumentEmbedding')

    @classmethod
    @with_request_id
    def search_similar_by_embedding(cls, query_text, limit=10, threshold=0.7, request_id=None):
        """
        Enhanced similarity search using pgvector embeddings with cosine similarity.

        Args:
            query_text: Text to search for
            limit: Maximum number of results
            threshold: Minimum similarity threshold (0.0 to 1.0)
            request_id: Request ID for logging

        Returns:
            List of similar documents with similarity scores
        """
        try:
            from plugins.ai_modules import generate_embedding, ModelsConfig

            # Generate embedding for query text
            current_embedding_model = ModelsConfig.get_config_value('embedding', 'CURRENT_MODEL')

            if current_embedding_model == "NoEmbeddingModel":
                warning_id("Embedding search disabled - no embedding model available", request_id)
                return []

            query_embeddings = generate_embedding(query_text, current_embedding_model)

            if not query_embeddings:
                warning_id("Failed to generate embeddings for query text", request_id)
                return []

            db_config = DatabaseConfig()
            with db_config.main_session() as session:
                return cls._search_by_pgvector_similarity(
                    session, query_embeddings, current_embedding_model, limit, threshold, request_id
                )

        except Exception as e:
            error_id(f"Embedding similarity search failed: {e}", request_id)
            return []

    @classmethod
    @with_request_id
    def get_embedding_statistics(cls, request_id=None):
        """
        Get statistics about embeddings in the database.

        Args:
            request_id: Request ID for logging

        Returns:
            dict: Embedding statistics
        """
        try:
            DocumentEmbeddingClass = cls._get_document_embedding_class()
            if DocumentEmbeddingClass is None:
                error_id("DocumentEmbedding class not available", request_id)
                return {}

            db_config = DatabaseConfig()
            with db_config.main_session() as session:

                total_embeddings = session.query(DocumentEmbeddingClass).count()

                pgvector_embeddings = session.query(DocumentEmbeddingClass).filter(
                    DocumentEmbeddingClass.embedding_vector.isnot(None)
                ).count()

                legacy_embeddings = session.query(DocumentEmbeddingClass).filter(
                    DocumentEmbeddingClass.model_embedding.isnot(None),
                    DocumentEmbeddingClass.embedding_vector.is_(None)
                ).count()

                # Get model distribution
                from sqlalchemy import func
                model_stats = session.query(
                    DocumentEmbeddingClass.model_name,
                    func.count(DocumentEmbeddingClass.id).label('count')
                ).group_by(DocumentEmbeddingClass.model_name).all()

                statistics = {
                    'total_embeddings': total_embeddings,
                    'pgvector_embeddings': pgvector_embeddings,
                    'legacy_embeddings': legacy_embeddings,
                    'pgvector_percentage': (
                                pgvector_embeddings / total_embeddings * 100) if total_embeddings > 0 else 0,
                    'models': {model: count for model, count in model_stats}
                }

                info_id(f"Embedding statistics: {pgvector_embeddings}/{total_embeddings} using pgvector", request_id)
                return statistics

        except Exception as e:
            error_id(f"Failed to get embedding statistics: {e}", request_id)
            return {}

    @classmethod
    @with_request_id
    def create_pgvector_indexes(cls, request_id=None):
        """
        Create optimized pgvector indexes for embedding similarity search.

        Args:
            request_id: Request ID for logging

        Returns:
            bool: Success status
        """
        try:
            DocumentEmbeddingClass = cls._get_document_embedding_class()
            if DocumentEmbeddingClass is None:
                error_id("DocumentEmbedding class not available", request_id)
                return False

            db_config = DatabaseConfig()
            with db_config.main_session() as session:
                success = DocumentEmbeddingClass.create_pgvector_indexes(session)

                if success:
                    info_id("pgvector indexes created successfully for DocumentEmbedding", request_id)
                else:
                    warning_id("Failed to create some pgvector indexes", request_id)

                return success

        except Exception as e:
            error_id(f"Failed to create pgvector indexes: {e}", request_id)
            return False

    @classmethod
    @with_request_id
    def get_images_with_chunk_context(cls, document_id, request_id=None):
        """
        Get all images for a document with their associated chunk context.
        Uses the Image class's enhanced query methods.
        """
        try:
            ImageClass = cls._get_image_class()
            if not ImageClass:
                error_id("Image class not available", request_id)
                return []

            db_config = DatabaseConfig()
            with db_config.main_session() as session:
                return ImageClass.get_images_with_chunk_context(session, document_id, request_id)

        except Exception as e:
            error_id(f"Failed to get images with chunk context: {e}", request_id)
            return []

    @classmethod
    @with_request_id
    def search_images_by_chunk_text(cls, search_text, document_id=None, confidence_threshold=0.5, request_id=None):
        """
        Search for images by their associated chunk text content.
        Uses the Image class's enhanced search methods.
        """
        try:
            ImageClass = cls._get_image_class()
            if not ImageClass:
                error_id("Image class not available", request_id)
                return []

            db_config = DatabaseConfig()
            with db_config.main_session() as session:
                return ImageClass.search_by_chunk_text(
                    session, search_text, document_id, confidence_threshold, request_id
                )

        except Exception as e:
            error_id(f"Failed to search images by chunk text: {e}", request_id)
            return []

    @classmethod
    @with_request_id
    def get_association_statistics(cls, document_id, request_id=None):
        """
        Get statistics about image-chunk associations.
        Uses the Image class's enhanced statistics methods.
        """
        try:
            ImageClass = cls._get_image_class()
            if not ImageClass:
                error_id("Image class not available", request_id)
                return {}

            db_config = DatabaseConfig()
            with db_config.main_session() as session:
                return ImageClass.get_association_statistics(session, document_id, request_id)

        except Exception as e:
            error_id(f"Failed to get association statistics: {e}", request_id)
            return {}

    # =====================================================
    # SEARCH IMPLEMENTATIONS (PostgreSQL Only)
    # =====================================================

    @classmethod
    def _search_postgresql(cls, session, query, limit, request_id):
        """Enhanced PostgreSQL full-text search with ranking and highlighting."""
        try:
            # Enhanced search with better ranking and highlighting
            sql = text("""
                SELECT cd.id, cd.title, cd.file_path, cd.rev,
                       ts_rank_cd(fts.search_vector, plainto_tsquery('english', :query)) as rank,
                       ts_headline('english', cd.content, plainto_tsquery('english', :query), 
                                  'MaxWords=50, MinWords=10, StartSel=<mark>, StopSel=</mark>') as highlight
                FROM complete_document cd
                JOIN documents_fts fts ON cd.title = fts.title
                WHERE fts.search_vector @@ plainto_tsquery('english', :query)
                ORDER BY rank DESC, cd.id DESC
                LIMIT :limit
            """)

            result = session.execute(sql, {'query': query, 'limit': limit})

            results = []
            for row in result:
                results.append({
                    'id': row[0],
                    'title': row[1],
                    'file_path': row[2],
                    'rev': row[3],
                    'relevance': float(row[4]),
                    'highlight': row[5]
                })

            info_id(f"PostgreSQL enhanced search found {len(results)} results", request_id)
            return results

        except Exception as e:
            error_id(f"PostgreSQL search failed: {e}", request_id)
            return []

    @classmethod
    def _similar_postgresql(cls, session, source, threshold, limit, request_id):
        """Enhanced PostgreSQL similarity search using pg_trgm extension."""
        try:
            # Enable pg_trgm extension if not already enabled
            try:
                session.execute(text("CREATE EXTENSION IF NOT EXISTS pg_trgm"))
            except:
                pass  # Extension might already exist

            # Enhanced similarity search with better algorithms
            sql = text("""
                SELECT cd.id, cd.title, cd.file_path, cd.rev,
                       similarity(cd.title, :source_title) as title_sim,
                       similarity(cd.content, :source_content) as content_sim,
                       GREATEST(
                           similarity(cd.title, :source_title),
                           similarity(cd.content, :source_content) * 0.7
                       ) as combined_sim
                FROM complete_document cd
                WHERE cd.id != :source_id 
                  AND (
                      similarity(cd.title, :source_title) >= :threshold
                      OR similarity(cd.content, :source_content) >= :threshold
                  )
                ORDER BY combined_sim DESC
                LIMIT :limit
            """)

            result = session.execute(sql, {
                'source_title': source.title,
                'source_content': source.content[:1000] if source.content else '',  # Limit content for performance
                'source_id': source.id,
                'threshold': threshold,
                'limit': limit
            })

            similar_docs = []
            for row in result:
                similar_docs.append({
                    'id': row[0],
                    'title': row[1],
                    'file_path': row[2],
                    'rev': row[3],
                    'title_similarity': float(row[4]),
                    'content_similarity': float(row[5]),
                    'combined_similarity': float(row[6])
                })

            info_id(f"PostgreSQL enhanced similarity search found {len(similar_docs)} results", request_id)
            return similar_docs

        except Exception as e:
            error_id(f"PostgreSQL similarity search failed: {e}", request_id)
            return []

    # =====================================================
    # UTILITY METHODS (PostgreSQL Optimized)
    # =====================================================

    @classmethod
    @with_request_id
    def _create_position(cls, metadata, request_id):
        """Create position record from metadata using PostgreSQL."""
        PositionClass = cls._get_position_class()
        SiteLocationClass = cls._get_site_location_class()

        if PositionClass is None:
            error_id("Position class not available", request_id)
            return cls._create_position_fallback(metadata, request_id)

        db_config = DatabaseConfig()

        with db_config.main_session() as session:
            # Create site location if provided
            site_location_id = None
            if metadata.get('site_location') and SiteLocationClass:
                site_loc = SiteLocationClass(
                    title=metadata['site_location'],
                    room_number=metadata.get('room_number', 'Unknown')
                )
                session.add(site_loc)
                session.flush()
                site_location_id = site_loc.id

            # Create position
            position = PositionClass(
                area_id=cls._safe_int(metadata.get('area')),
                equipment_group_id=cls._safe_int(metadata.get('equipment_group')),
                model_id=cls._safe_int(metadata.get('model')),
                asset_number_id=cls._safe_int(metadata.get('asset_number')),
                location_id=cls._safe_int(metadata.get('location')),
                site_location_id=site_location_id
            )

            session.add(position)
            session.commit()

            info_id(f"Created position {position.id} in PostgreSQL", request_id)
            return position.id

    @classmethod
    @with_request_id
    def _create_position_fallback(cls, metadata, request_id):
        """Fallback position creation using PostgreSQL raw SQL."""
        db_config = DatabaseConfig()

        with db_config.main_session() as session:
            try:
                # PostgreSQL-specific position creation
                sql = text("""
                    INSERT INTO position (area_id, equipment_group_id, model_id, asset_number_id, location_id)
                    VALUES (:area_id, :equipment_group_id, :model_id, :asset_number_id, :location_id)
                    RETURNING id
                """)

                result = session.execute(sql, {
                    'area_id': cls._safe_int(metadata.get('area')),
                    'equipment_group_id': cls._safe_int(metadata.get('equipment_group')),
                    'model_id': cls._safe_int(metadata.get('model')),
                    'asset_number_id': cls._safe_int(metadata.get('asset_number')),
                    'location_id': cls._safe_int(metadata.get('location'))
                })

                position_id = result.fetchone()[0]
                session.commit()
                info_id(f"Created position {position_id} using PostgreSQL fallback method", request_id)
                return position_id

            except Exception as e:
                error_id(f"PostgreSQL fallback position creation failed: {e}", request_id)
                # Return a default position ID (you may need to adjust this)
                return 1

    @classmethod
    @with_request_id
    def _extract_content(cls, file_path, request_id):
        """Extract text content from file - convert DOCX to PDF first if needed."""
        ext = os.path.splitext(file_path)[1].lower()

        try:
            if ext == '.pdf':
                return cls._extract_pdf_text(file_path, request_id)
            elif ext == '.txt':
                return cls._extract_txt_text(file_path, request_id)
            elif ext == '.docx':
                # Convert DOCX to PDF first, then use existing PDF processing
                info_id("Converting DOCX to PDF for processing", request_id)
                pdf_path = cls._convert_docx_to_pdf(file_path, request_id)
                if pdf_path:
                    # Use existing PDF text extraction
                    text = cls._extract_pdf_text(pdf_path, request_id)
                    # Clean up temporary PDF
                    cls._cleanup_temp_file(pdf_path, request_id)
                    return text
                else:
                    error_id("DOCX to PDF conversion failed", request_id)
                    return None
            else:
                warning_id(f"Unsupported file type: {ext}", request_id)
                return None

        except Exception as e:
            error_id(f"Content extraction failed for {file_path}: {e}", request_id)
            return None

    @classmethod
    @with_request_id
    def _extract_pdf_text(cls, pdf_path, request_id):
        """Extract text from PDF using PyMuPDF with proper Unicode handling."""
        try:
            import fitz  # PyMuPDF

            text = ""
            with fitz.open(pdf_path) as doc:
                for page in doc:
                    page_text = page.get_text()

                    # Ensure proper Unicode handling
                    if isinstance(page_text, bytes):
                        page_text = page_text.decode('utf-8', errors='replace')

                    text += page_text + "\n"

            # Additional Unicode normalization
            if text:
                # Normalize Unicode characters
                text = unicodedata.normalize('NFKC', text)

                # SAFE logging - count non-ASCII chars without logging the actual characters
                non_ascii_chars = [char for char in text if ord(char) > 127]
                if non_ascii_chars:
                    unique_count = len(set(non_ascii_chars))
                    debug_id(f"Found {unique_count} unique non-ASCII characters in extracted PDF text", request_id)
                else:
                    debug_id("All characters in PDF text are ASCII-compatible", request_id)

            debug_id(f"Extracted {len(text)} characters from PDF with Unicode normalization", request_id)
            return text.strip()

        except ImportError:
            error_id("PyMuPDF (fitz) not available for PDF processing", request_id)
            return None
        except Exception as e:
            error_id(f"PDF text extraction failed: {e}", request_id)
            return None

    @classmethod
    @with_request_id
    def _extract_txt_text(cls, txt_path, request_id):
        """Extract text from plain text file."""
        try:
            with open(txt_path, 'r', encoding='utf-8') as f:
                text = f.read()

            debug_id(f"Extracted {len(text)} characters from TXT", request_id)
            return text.strip()

        except Exception as e:
            error_id(f"TXT extraction failed: {e}", request_id)
            return None

    @classmethod
    @with_request_id
    def _convert_docx_to_pdf(cls, docx_path, request_id):
        """Convert DOCX to PDF using docx2pdf library."""
        try:
            from docx2pdf import convert
            import pythoncom

            # Initialize COM for Word automation (Windows)
            pythoncom.CoInitialize()
            debug_id("COM initialization successful", request_id)

            # Create temporary PDF path
            temp_dir = tempfile.mkdtemp()
            base_name = os.path.splitext(os.path.basename(docx_path))[0]
            pdf_path = os.path.join(temp_dir, f"{base_name}.pdf")

            debug_id(f"Converting DOCX to PDF: {docx_path} -> {pdf_path}", request_id)

            # Convert DOCX to PDF
            convert(docx_path, pdf_path)

            if os.path.exists(pdf_path):
                info_id(f"Successfully converted DOCX to PDF: {pdf_path}", request_id)
                return pdf_path
            else:
                error_id("PDF conversion failed - output file not created", request_id)
                return None

        except ImportError:
            error_id("docx2pdf not installed. Run: pip install docx2pdf", request_id)
            return None
        except Exception as e:
            error_id(f"DOCX to PDF conversion failed: {e}", request_id)
            return None
        finally:
            try:
                pythoncom.CoUninitialize()
            except:
                pass

    @classmethod
    @with_request_id
    def _cleanup_temp_file(cls, file_path, request_id):
        """Clean up temporary files and directories."""
        try:
            if file_path and os.path.exists(file_path):
                # Remove the file
                os.remove(file_path)
                # Remove the temporary directory if empty
                temp_dir = os.path.dirname(file_path)
                if os.path.exists(temp_dir) and not os.listdir(temp_dir):
                    os.rmdir(temp_dir)
                debug_id(f"Cleaned up temporary file: {file_path}", request_id)
        except Exception as e:
            warning_id(f"Failed to cleanup temp file {file_path}: {e}", request_id)

    @staticmethod
    def _clean_filename(filename):
        """Generate clean title from filename."""
        if not filename:
            return "Untitled Document"

        # Remove extension and clean up
        title = os.path.splitext(filename)[0]
        title = re.sub(r'[_\-]+', ' ', title)
        title = ' '.join(word.capitalize() for word in title.split() if word)

        return title or "Untitled Document"

    @staticmethod
    def _safe_int(value):
        """Convert to int safely or return None."""
        if value and str(value).strip().isdigit():
            return int(value)
        return None

    @staticmethod
    def _split_text(text, max_words):
        """Split text into chunks of specified word count."""
        if not text:
            return []

        words = text.split()
        chunks = []

        for i in range(0, len(words), max_words):
            chunk = ' '.join(words[i:i + max_words])
            if chunk.strip():
                chunks.append(chunk)

        return chunks

    @classmethod
    @with_request_id
    def serve_file(cls, document_id, download=None, request_id=None):
        """
        Enhanced serve_file method for CompleteDocument with DATABASE_DOC support.
        Returns a tuple (success, response, status_code) for Flask compatibility.
        """
        # Import locally to avoid circular dependencies
        from flask import send_file
        from modules.configuration.config import DATABASE_DIR

        # Get or generate request_id
        rid = request_id or get_request_id()

        info_id(f"Attempting to retrieve document with ID: {document_id}", rid)

        # Create session using DatabaseConfig
        db_config = DatabaseConfig()
        try:
            with db_config.main_session() as session:

                # Query for the document
                document = session.query(cls).filter_by(id=document_id).first()

                if not document:
                    error_id(f"No document found in database with ID: {document_id}", rid)
                    return False, "Document not found", 404

                debug_id(f"Found document: {document.title}, File path: {document.file_path}", rid)

                # Define the correct DATABASE_DOC path
                DATABASE_DOC = os.path.join(DATABASE_DIR, 'DB_DOC')

                # Build potential file paths in order of preference
                potential_paths = []

                if document.file_path:
                    # Clean the file path
                    clean_path = document.file_path.strip().replace('\\', '/')

                    # 1. PRIORITY: Handle the specific case where DB_DOC is already in the path
                    if clean_path.startswith('DB_DOC/') or clean_path.startswith('DB_DOC\\'):
                        # Path already includes DB_DOC, so join with DATABASE_DIR instead
                        potential_paths.append(os.path.join(DATABASE_DIR, clean_path))
                        # Also try just the filename in DATABASE_DOC
                        potential_paths.append(os.path.join(DATABASE_DOC, os.path.basename(clean_path)))
                    else:
                        # Normal case: DATABASE_DOC directory (correct location)
                        # Handle both filename-only and relative paths
                        potential_paths.append(os.path.join(DATABASE_DOC, os.path.basename(document.file_path)))
                        if document.file_path != os.path.basename(document.file_path):
                            # If it's a relative path, try it relative to DATABASE_DOC
                            potential_paths.append(os.path.join(DATABASE_DOC, clean_path))

                    # 2. If file_path is absolute, try it directly
                    if os.path.isabs(document.file_path):
                        potential_paths.append(document.file_path)

                    # 3. LEGACY SUPPORT: Old storage locations (for backwards compatibility)
                    clean_legacy_path = clean_path.lstrip('/\\\\')
                    potential_paths.extend([
                        # Try relative to DATABASE_DIR (for paths like "DB_DOC\file.pdf")
                        os.path.join(DATABASE_DIR, clean_legacy_path),
                        # Try relative to current working directory
                        os.path.join(os.getcwd(), document.file_path),
                        # Try old uploads directories
                        os.path.join(os.getcwd(), "Uploads", os.path.basename(clean_legacy_path)),
                        os.path.join(os.getcwd(), "uploads", os.path.basename(clean_legacy_path)),
                        os.path.join(DATABASE_DIR, "uploads", os.path.basename(clean_legacy_path)),
                        os.path.join(DATABASE_DIR, "documents", os.path.basename(clean_legacy_path))
                    ])

                debug_id(f"Database file_path: '{document.file_path}'", rid)
                debug_id(f"DATABASE_DOC: {DATABASE_DOC}", rid)
                debug_id(f"Checking {len(potential_paths)} potential file paths", rid)

                # Find the first existing file
                file_path = None
                for i, path in enumerate(potential_paths):
                    debug_id(f"Checking path {i + 1}: {path}", rid)
                    if os.path.exists(path):
                        file_path = path
                        if i == 0:
                            info_id(f"Found file in correct location: {path}", rid)
                        elif i == 1:
                            info_id(f"Found file using DATABASE_DIR join: {path}", rid)
                        else:
                            warning_id(f"Found file in legacy location {i + 1}: {path} - consider fixing database path",
                                       rid)
                        break

                if not file_path:
                    error_id(f"Document file not found in any location. Expected in: {DATABASE_DOC}", rid)
                    debug_id(f"Searched paths: {potential_paths[:3]}... (and {len(potential_paths) - 3} more)", rid)
                    return False, "Document file not found", 404

                info_id(f"Serving document file: {file_path}", rid)

                # Determine mimetype based on file extension
                _, ext = os.path.splitext(file_path)
                mimetype_map = {
                    '.pdf': 'application/pdf',
                    '.doc': 'application/msword',
                    '.docx': 'application/vnd.openxmlformats-officedocument.wordprocessingml.document',
                    '.xls': 'application/vnd.ms-excel',
                    '.xlsx': 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
                    '.ppt': 'application/vnd.ms-powerpoint',
                    '.pptx': 'application/vnd.openxmlformats-officedocument.presentationml.presentation',
                    '.txt': 'text/plain',
                    '.rtf': 'application/rtf',
                    '.csv': 'text/csv',
                    '.xml': 'application/xml',
                    '.json': 'application/json'
                }
                mimetype = mimetype_map.get(ext.lower(), 'application/octet-stream')

                debug_id(f"Using mimetype: {mimetype} for extension: {ext}", rid)

                # Determine disposition based on download parameter and file type
                if download is True:
                    # Force download
                    as_attachment = True
                    debug_id("Forcing download (as_attachment=True)", rid)
                elif download is False:
                    # Force inline viewing
                    as_attachment = False
                    debug_id("Forcing inline view (as_attachment=False)", rid)
                else:
                    # Auto-detect based on file type - PDFs and text files can be viewed inline
                    as_attachment = ext.lower() not in ['.pdf', '.txt', '.csv', '.xml', '.json']
                    debug_id(f"Auto-detect disposition: as_attachment={as_attachment} for {ext}", rid)

                response = send_file(file_path, mimetype=mimetype, as_attachment=as_attachment)
                return True, response, 200

        except Exception as e:
            error_id(f"Unhandled error while serving document with ID {document_id}: {e}", rid, exc_info=True)
            return False, "Internal Server Error", 500

    @classmethod
    @with_request_id
    def _search_by_pgvector_similarity(cls, session, query_embeddings, model_name, limit, threshold, request_id=None):
        """
        Perform similarity search using pgvector operators.

        Args:
            session: Database session
            query_embeddings: Query embedding vector
            model_name: Embedding model name
            limit: Maximum results
            threshold: Similarity threshold
            request_id: Request ID for logging

        Returns:
            List of similar documents with scores
        """
        try:
            from sqlalchemy import text, func

            DocumentEmbeddingClass = cls._get_document_embedding_class()
            if DocumentEmbeddingClass is None:
                error_id("DocumentEmbedding class not available", request_id)
                return []

            # Convert query embeddings to pgvector format
            query_vector_str = '[' + ','.join(map(str, query_embeddings)) + ']'

            # Use pgvector cosine similarity operator (<=>)
            similarity_query = text("""
                  SELECT 
                      cd.id,
                      cd.title,
                      cd.file_path,
                      cd.rev,
                      de.embedding_vector <=> :query_vector AS distance,
                      1 - (de.embedding_vector <=> :query_vector) AS similarity,
                      de.model_name,
                      de.created_at
                  FROM complete_document cd
                  JOIN document d ON cd.id = d.complete_document_id
                  JOIN document_embedding de ON d.id = de.document_id
                  WHERE de.model_name = :model_name
                    AND de.embedding_vector IS NOT NULL
                    AND (1 - (de.embedding_vector <=> :query_vector)) >= :threshold
                  ORDER BY de.embedding_vector <=> :query_vector ASC
                  LIMIT :limit
              """)

            result = session.execute(similarity_query, {
                'query_vector': query_vector_str,
                'model_name': model_name,
                'threshold': threshold,
                'limit': limit
            })

            similar_docs = []
            for row in result:
                similar_docs.append({
                    'id': row[0],
                    'title': row[1],
                    'file_path': row[2],
                    'rev': row[3],
                    'distance': float(row[4]),
                    'similarity': float(row[5]),
                    'model_name': row[6],
                    'created_at': row[7].isoformat() if row[7] else None
                })

            info_id(f"pgvector similarity search found {len(similar_docs)} results above threshold {threshold}",
                    request_id)
            return similar_docs

        except Exception as e:
            error_id(f"pgvector similarity search failed: {e}", request_id)
            return []

    # ADD these missing methods to your CompleteDocument class in emtacdb_fts.py

    # Add this method to the CompleteDocument class
    @classmethod
    @with_request_id
    def search_by_text(
            cls,
            query,
            session=None,
            limit: int = 25,  # NEW: accept limit from callers
            threshold: float = None,  # NEW: accept 'threshold' (ignored here, for compat)
            similarity_threshold: int = 70,  # kept for compat with older code
            with_links: bool = False,
            request_id=None,
            **kwargs  # NEW: swallow any future kwargs safely
    ):
        """
        Search documents by text content using PostgreSQL full-text search.

        Args:
            query: Search query string
            session: Optional database session
            limit: Max results to return (compat with UnifiedSearch)
            threshold: Optional FTS score threshold (not used here; kept for compat)
            similarity_threshold: Back-compat arg (not used by FTS)
            with_links: Whether to return HTML links or document objects
            request_id: Request ID for logging
        """
        try:
            # Create session if not provided
            local_session = None
            if session is None:
                db_config = DatabaseConfig()
                local_session = db_config.get_main_session()
                session = local_session

            try:
                # Use PostgreSQL full-text search if available
                # (was hardcoded 50; now honor 'limit')
                results = cls._search_postgresql(session, query, limit, request_id)

                if results:
                    if with_links:
                        # Return HTML links (cap display to 10 to keep UI tidy)
                        html_links = []
                        for result in results[:10]:
                            title = result.get('title', 'Untitled')
                            doc_id = result.get('id')
                            highlight = result.get('highlight', '')
                            link = f'<a href="/complete_document/{doc_id}" target="_blank">{title}</a>'
                            if highlight:
                                link += f'<br><small>{highlight}</small>'
                            html_links.append(link)
                        return '<br><br>'.join(html_links)
                    else:
                        # Return document objects, preserving rank order
                        doc_ids = [r['id'] for r in results]
                        documents = session.query(cls).filter(cls.id.in_(doc_ids)).all()
                        doc_dict = {doc.id: doc for doc in documents}
                        sorted_docs = [doc_dict[i] for i in doc_ids if i in doc_dict]
                        info_id(f"Found {len(sorted_docs)} documents matching '{query}'", request_id)
                        return sorted_docs
                else:
                    # Fallback to simple text search
                    return cls._fallback_text_search(session, query, with_links, request_id)

            finally:
                if local_session:
                    local_session.close()

        except Exception as e:
            error_id(f"Error in search_by_text: {e}", request_id, exc_info=True)
            return [] if not with_links else "No documents found"

    @classmethod
    @with_request_id
    def _fallback_text_search(cls, session, query, with_links=False, request_id=None):
        """
        Fallback text search using simple ILIKE when FTS is not available.
        """
        try:
            search_term = f"%{query}%"

            # Search in title and content
            documents = session.query(cls).filter(
                (cls.title.ilike(search_term)) |
                (cls.content.ilike(search_term))
            ).limit(10).all()

            if with_links:
                # Return HTML links
                if documents:
                    html_links = []
                    for doc in documents:
                        link = f'<a href="/complete_document/{doc.id}" target="_blank">{doc.title}</a>'
                        html_links.append(link)
                    return '<br><br>'.join(html_links)
                else:
                    return "No documents found"
            else:
                # Return document objects
                info_id(f"Fallback search found {len(documents)} documents matching '{query}'", request_id)
                return documents

        except Exception as e:
            error_id(f"Error in fallback text search: {e}", request_id, exc_info=True)
            return [] if not with_links else "No documents found"

class Problem(Base):
    __tablename__ = 'problem'

    id = Column(Integer, primary_key=True)
    name = Column(String, nullable=False)
    description = Column(String, nullable=False)

    # Relationships
    solutions = relationship("Solution", back_populates="problem")  # One-to-many with solutions
    problem_position = relationship("ProblemPositionAssociation", back_populates="problem")
    image_problem = relationship("ImageProblemAssociation", back_populates="problem")
    complete_document_problem = relationship("CompleteDocumentProblemAssociation", back_populates="problem")
    drawing_problem = relationship("DrawingProblemAssociation", back_populates="problem")
    part_problem = relationship("PartProblemAssociation", back_populates="problem")

    # ----------------------------
    # CREATE / ADD
    # ----------------------------
    @classmethod
    @with_request_id
    def add_to_db(cls, session, name: str, description: str, request_id=None) -> int:
        """
        Add or get a Problem by name. Always returns the Problem ID.
        """
        try:
            problem = session.query(cls).filter_by(name=name).first()
            if problem:
                info_id(f"Found existing Problem '{name}' (id={problem.id})", request_id)
                return problem.id

            problem = cls(name=name, description=description)
            session.add(problem)
            session.commit()
            info_id(f"Created new Problem '{name}' (id={problem.id})", request_id)
            return problem.id
        except Exception as e:
            session.rollback()
            error_id(f"Problem.add_to_db failed: {e}", request_id)
            raise

    # ----------------------------
    # RETRIEVE
    # ----------------------------
    @classmethod
    @with_request_id
    def get_by_id(cls, session, problem_id: int, request_id=None):
        """Retrieve a Problem by its ID."""
        try:
            return session.query(cls).filter_by(id=problem_id).first()
        except Exception as e:
            error_id(f"Problem.get_by_id failed: {e}", request_id)
            return None

    @classmethod
    @with_request_id
    def list_all(cls, session, limit: int = 100, request_id=None):
        """List all Problems with optional limit."""
        try:
            return session.query(cls).limit(limit).all()
        except Exception as e:
            error_id(f"Problem.list_all failed: {e}", request_id)
            return []

    # ----------------------------
    # UPDATE
    # ----------------------------
    @classmethod
    @with_request_id
    def update_problem(cls, session, problem_id: int, name: str = None, description: str = None, request_id=None) -> bool:
        """Update an existing Problem's fields."""
        try:
            problem = session.query(cls).filter_by(id=problem_id).first()
            if not problem:
                warning_id(f"Problem {problem_id} not found for update", request_id)
                return False

            if name:
                problem.name = name
            if description:
                problem.description = description

            session.commit()
            info_id(f"Updated Problem {problem_id}", request_id)
            return True
        except Exception as e:
            session.rollback()
            error_id(f"Problem.update_problem failed: {e}", request_id)
            return False

    # ----------------------------
    # DELETE
    # ----------------------------
    @classmethod
    @with_request_id
    def delete_problem(cls, session, problem_id: int, request_id=None) -> bool:
        """Delete a Problem by ID."""
        try:
            problem = session.query(cls).filter_by(id=problem_id).first()
            if not problem:
                warning_id(f"Problem {problem_id} not found for deletion", request_id)
                return False

            session.delete(problem)
            session.commit()
            info_id(f"Deleted Problem {problem_id}", request_id)
            return True
        except Exception as e:
            session.rollback()
            error_id(f"Problem.delete_problem failed: {e}", request_id)
            return False

    # ----------------------------
    # SEARCH
    # ----------------------------
    @classmethod
    @with_request_id
    def search(cls, session, text: str, exact: bool = False, limit: int = 50, request_id=None):
        """Search Problems by name or description."""
        try:
            query = session.query(cls)
            if exact:
                results = query.filter((cls.name == text) | (cls.description == text)).limit(limit).all()
            else:
                results = query.filter(
                    (cls.name.ilike(f"%{text}%")) | (cls.description.ilike(f"%{text}%"))
                ).limit(limit).all()

            debug_id(f"Problem.search for '{text}' returned {len(results)} results", request_id)
            return results
        except Exception as e:
            error_id(f"Problem.search failed: {e}", request_id)
            return []

    # ----------------------------
    # RELATIONSHIP HELPERS
    # ----------------------------
    def get_solutions(self):
        """Return all solutions linked to this Problem."""
        return self.solutions

    def get_positions(self):
        """Return all position associations for this Problem."""
        return self.problem_position

    def get_images(self):
        """Return all image associations for this Problem."""
        return self.image_problem

    def get_documents(self):
        """Return all document associations for this Problem."""
        return self.complete_document_problem

    def get_drawings(self):
        """Return all drawing associations for this Problem."""
        return self.drawing_problem

    def get_parts(self):
        """Return all part associations for this Problem."""
        return self.part_problem

    # ----------------------------
    # STATS
    # ----------------------------
    @classmethod
    @with_request_id
    def count(cls, session, request_id=None) -> int:
        """Count total number of Problems."""
        try:
            return session.query(cls).count()
        except Exception as e:
            error_id(f"Problem.count failed: {e}", request_id)
            return 0


class Solution(Base):
    __tablename__ = 'solution'

    id = Column(Integer, primary_key=True)
    name = Column(String, nullable=False)  # Solution name
    description = Column(String, nullable=False)
    problem_id = Column(Integer, ForeignKey('problem.id'))

    # Relationships
    problem = relationship("Problem", back_populates="solutions")
    task_solutions = relationship("TaskSolutionAssociation", back_populates="solution", cascade="all, delete-orphan")

class Task(Base):
    __tablename__ = 'task'

    id = Column(Integer, primary_key=True)
    name = Column(String, nullable=False)
    description = Column(String, nullable=False)

    # Relationships
    task_positions = relationship("TaskPositionAssociation", back_populates="task", cascade="all, delete-orphan")
    task_solutions = relationship("TaskSolutionAssociation", back_populates="task", cascade="all, delete-orphan")
    image_task = relationship("ImageTaskAssociation", back_populates="task")
    complete_document_task = relationship("CompleteDocumentTaskAssociation", back_populates="task")
    drawing_task = relationship("DrawingTaskAssociation", back_populates="task")
    part_task = relationship("PartTaskAssociation", back_populates="task")
    tool_tasks = relationship("TaskToolAssociation", back_populates="task", cascade="all, delete-orphan")

class TaskSolutionAssociation(Base):
    __tablename__ = 'task_solution_association'

    id = Column(Integer, primary_key=True)
    task_id = Column(Integer, ForeignKey('task.id'))
    solution_id = Column(Integer, ForeignKey('solution.id'))

    # Relationships
    task = relationship("Task", back_populates="task_solutions")
    solution = relationship("Solution", back_populates="task_solutions")

class PowerPoint(Base):
    __tablename__ = 'powerpoint'

    id = Column(Integer, primary_key=True)
    title = Column(String, nullable=False)
    ppt_file_path = Column(String, nullable=False)
    pdf_file_path = Column(String, nullable=False)
    description = Column(String, nullable=True)
    complete_document_id = Column(Integer, ForeignKey('complete_document.id'))
    rev = Column(String, nullable=True)

    complete_document = relationship("CompleteDocument", back_populates="powerpoint")

    def __init__(self, title, ppt_file_path, pdf_file_path, complete_document_id, description=None):
        self.title = title
        self.ppt_file_path = ppt_file_path
        self.pdf_file_path = pdf_file_path
        self.complete_document_id = complete_document_id
        self.description = description

# Junction Classes 
class DrawingPartAssociation(Base):
    __tablename__ = 'drawing_part'
    id = Column(Integer, primary_key=True)
    drawing_id = Column(Integer, ForeignKey('drawing.id'))
    part_id = Column(Integer, ForeignKey('part.id'))
    
    drawing = relationship("Drawing", back_populates="drawing_part")
    part = relationship("Part", back_populates="drawing_part")

    # ======================================================
    # SINGLE ADD OPERATION
    # ======================================================
    @classmethod
    @with_request_id
    def add(cls,
            drawing_id: int,
            part_id: int,
            session: Optional[Session] = None,
            request_id: Optional[str] = None) -> Optional[int]:
        """
        Create a new Drawing-Part association if it does not already exist.

        Args:
            drawing_id: ID of the drawing
            part_id: ID of the part
            session: Optional SQLAlchemy session (new session created if not provided)
            request_id: Optional request ID for logging

        Returns:
            int: ID of the created association, or None if it already existed or failed
        """
        rid = request_id or get_request_id()
        db_config = DatabaseConfig()
        session_provided = session is not None
        if not session_provided:
            session = db_config.get_main_session()

        try:
            # Check if association already exists
            existing = session.query(cls).filter_by(
                drawing_id=drawing_id,
                part_id=part_id
            ).first()

            if existing:
                debug_id(f"Association already exists: Drawing {drawing_id} ↔ Part {part_id}", rid)
                return existing.id

            assoc = cls(drawing_id=drawing_id, part_id=part_id)
            session.add(assoc)
            session.commit()

            info_id(f"Created DrawingPartAssociation: Drawing {drawing_id} ↔ Part {part_id}", rid)
            return assoc.id

        except Exception as e:
            session.rollback()
            error_id(f"Failed to create DrawingPartAssociation: {e}", rid, exc_info=True)
            return None
        finally:
            if not session_provided:
                session.close()


    @classmethod
    @with_request_id
    def get_parts_by_drawing(cls,
                             drawing_id: Optional[int] = None,
                             drw_equipment_name: Optional[str] = None,
                             drw_number: Optional[str] = None,
                             drw_name: Optional[str] = None,
                             drw_revision: Optional[str] = None,
                             drw_spare_part_number: Optional[str] = None,
                             part_id: Optional[int] = None,
                             part_number: Optional[str] = None,
                             part_name: Optional[str] = None,
                             oem_mfg: Optional[str] = None,
                             model: Optional[str] = None,
                             class_flag: Optional[str] = None,
                             exact_match: bool = False,
                             limit: int = 100,
                             request_id: Optional[str] = None,
                             session: Optional[Session] = None) -> List['Part']:
        """
        Get parts associated with drawings based on flexible search criteria.

        Args:
            drawing_id: Optional drawing ID to filter by
            drw_equipment_name, drw_number, drw_name, drw_revision, drw_spare_part_number:
                Optional drawing attributes to filter by
            part_id: Optional part ID to filter by
            part_number, part_name, oem_mfg, model, class_flag:
                Optional part attributes to filter by
            exact_match: If True, performs exact matching instead of partial matching for string fields
            limit: Maximum number of results to return (default 100)
            request_id: Optional request ID for tracking this operation in logs
            session: Optional SQLAlchemy session. If None, a new session will be created

        Returns:
            List of Part objects matching the search criteria
        """
        # Get or use the provided request_id
        rid = request_id or get_request_id()

        # Get a database session if one wasn't provided
        db_config = DatabaseConfig()
        session_provided = session is not None
        if not session_provided:
            session = db_config.get_main_session()
            debug_id(f"Created new database session for Drawing.get_parts_by_drawing", rid)

        # Log the search operation with request ID
        search_params = {
            'drawing_id': drawing_id,
            'drw_equipment_name': drw_equipment_name,
            'drw_number': drw_number,
            'drw_name': drw_name,
            'drw_revision': drw_revision,
            'drw_spare_part_number': drw_spare_part_number,
            'part_id': part_id,
            'part_number': part_number,
            'part_name': part_name,
            'oem_mfg': oem_mfg,
            'model': model,
            'class_flag': class_flag,
            'exact_match': exact_match,
            'limit': limit
        }
        # Filter out None values for cleaner logging
        logged_params = {k: v for k, v in search_params.items() if v is not None}
        debug_id(f"Starting Drawing.get_parts_by_drawing with parameters: {logged_params}", rid)

        try:
            # Use the timed operation context manager with request ID
            with log_timed_operation("Drawing.get_parts_by_drawing", rid):


                # Start with a query that joins Part and DrawingPartAssociation
                query = session.query(Part).join(DrawingPartAssociation).join(cls)

                # Apply drawing filters
                if drawing_id is not None:
                    query = query.filter(cls.id == drawing_id)

                if drw_equipment_name is not None:
                    if exact_match:
                        query = query.filter(cls.drw_equipment_name == drw_equipment_name)
                    else:
                        query = query.filter(cls.drw_equipment_name.ilike(f"%{drw_equipment_name}%"))

                if drw_number is not None:
                    if exact_match:
                        query = query.filter(cls.drw_number == drw_number)
                    else:
                        query = query.filter(cls.drw_number.ilike(f"%{drw_number}%"))

                if drw_name is not None:
                    if exact_match:
                        query = query.filter(cls.drw_name == drw_name)
                    else:
                        query = query.filter(cls.drw_name.ilike(f"%{drw_name}%"))

                if drw_revision is not None:
                    if exact_match:
                        query = query.filter(cls.drw_revision == drw_revision)
                    else:
                        query = query.filter(cls.drw_revision.ilike(f"%{drw_revision}%"))

                if drw_spare_part_number is not None:
                    if exact_match:
                        query = query.filter(cls.drw_spare_part_number == drw_spare_part_number)
                    else:
                        query = query.filter(cls.drw_spare_part_number.ilike(f"%{drw_spare_part_number}%"))

                # Apply part filters
                if part_id is not None:
                    query = query.filter(Part.id == part_id)

                if part_number is not None:
                    if exact_match:
                        query = query.filter(Part.part_number == part_number)
                    else:
                        query = query.filter(Part.part_number.ilike(f"%{part_number}%"))

                if part_name is not None:
                    if exact_match:
                        query = query.filter(Part.name == part_name)
                    else:
                        query = query.filter(Part.name.ilike(f"%{part_name}%"))

                if oem_mfg is not None:
                    if exact_match:
                        query = query.filter(Part.oem_mfg == oem_mfg)
                    else:
                        query = query.filter(Part.oem_mfg.ilike(f"%{oem_mfg}%"))

                if model is not None:
                    if exact_match:
                        query = query.filter(Part.model == model)
                    else:
                        query = query.filter(Part.model.ilike(f"%{model}%"))

                if class_flag is not None:
                    if exact_match:
                        query = query.filter(Part.class_flag == class_flag)
                    else:
                        query = query.filter(Part.class_flag.ilike(f"%{class_flag}%"))

                # Make results distinct to avoid duplicates
                query = query.distinct()

                # Apply limit
                query = query.limit(limit)

                # Execute query and log results
                results = query.all()
                debug_id(f"Drawing.get_parts_by_drawing completed, found {len(results)} parts", rid)
                return results

        except Exception as e:
            error_id(f"Error in Drawing.get_parts_by_drawing: {str(e)}", rid, exc_info=True)
            return []
        finally:
            # Close the session if we created it
            if not session_provided:
                session.close()
                debug_id(f"Closed database session for Drawing.get_parts_by_drawing", rid)

    @classmethod
    @with_request_id
    def get_drawings_by_part(cls,
                             part_id: Optional[int] = None,
                             part_number: Optional[str] = None,
                             part_name: Optional[str] = None,
                             oem_mfg: Optional[str] = None,
                             model: Optional[str] = None,
                             class_flag: Optional[str] = None,
                             drawing_id: Optional[int] = None,
                             drw_equipment_name: Optional[str] = None,
                             drw_number: Optional[str] = None,
                             drw_name: Optional[str] = None,
                             drw_revision: Optional[str] = None,
                             drw_spare_part_number: Optional[str] = None,
                             exact_match: bool = False,
                             limit: int = 100,
                             request_id: Optional[str] = None,
                             session: Optional[Session] = None) -> List['Drawing']:
        """
        Get drawings associated with parts based on flexible search criteria.

        Args:
            part_id: Optional part ID to filter by
            part_number, part_name, oem_mfg, model, class_flag:
                Optional part attributes to filter by
            drawing_id: Optional drawing ID to filter by
            drw_equipment_name, drw_number, drw_name, drw_revision, drw_spare_part_number:
                Optional drawing attributes to filter by
            exact_match: If True, performs exact matching instead of partial matching for string fields
            limit: Maximum number of results to return (default 100)
            request_id: Optional request ID for tracking this operation in logs
            session: Optional SQLAlchemy session. If None, a new session will be created

        Returns:
            List of Drawing objects matching the search criteria
        """
        # Get or use the provided request_id
        rid = request_id or get_request_id()

        # Get a database session if one wasn't provided
        db_config = DatabaseConfig()
        session_provided = session is not None
        if not session_provided:
            session = db_config.get_main_session()
            debug_id(f"Created new database session for Part.get_drawings_by_part", rid)

        # Log the search operation with request ID
        search_params = {
            'part_id': part_id,
            'part_number': part_number,
            'part_name': part_name,
            'oem_mfg': oem_mfg,
            'model': model,
            'class_flag': class_flag,
            'drawing_id': drawing_id,
            'drw_equipment_name': drw_equipment_name,
            'drw_number': drw_number,
            'drw_name': drw_name,
            'drw_revision': drw_revision,
            'drw_spare_part_number': drw_spare_part_number,
            'exact_match': exact_match,
            'limit': limit
        }
        # Filter out None values for cleaner logging
        logged_params = {k: v for k, v in search_params.items() if v is not None}
        debug_id(f"Starting Part.get_drawings_by_part with parameters: {logged_params}", rid)

        try:
            # Use the timed operation context manager with request ID
            with log_timed_operation("Part.get_drawings_by_part", rid):


                # Start with a query that joins Drawing and DrawingPartAssociation
                query = session.query(Drawing).join(DrawingPartAssociation).join(cls)

                # Apply part filters
                if part_id is not None:
                    query = query.filter(cls.id == part_id)

                if part_number is not None:
                    if exact_match:
                        query = query.filter(cls.part_number == part_number)
                    else:
                        query = query.filter(cls.part_number.ilike(f"%{part_number}%"))

                if part_name is not None:
                    if exact_match:
                        query = query.filter(cls.name == part_name)
                    else:
                        query = query.filter(cls.name.ilike(f"%{part_name}%"))

                if oem_mfg is not None:
                    if exact_match:
                        query = query.filter(cls.oem_mfg == oem_mfg)
                    else:
                        query = query.filter(cls.oem_mfg.ilike(f"%{oem_mfg}%"))

                if model is not None:
                    if exact_match:
                        query = query.filter(cls.model == model)
                    else:
                        query = query.filter(cls.model.ilike(f"%{model}%"))

                if class_flag is not None:
                    if exact_match:
                        query = query.filter(cls.class_flag == class_flag)
                    else:
                        query = query.filter(cls.class_flag.ilike(f"%{class_flag}%"))

                # Apply drawing filters
                if drawing_id is not None:
                    query = query.filter(Drawing.id == drawing_id)

                if drw_equipment_name is not None:
                    if exact_match:
                        query = query.filter(Drawing.drw_equipment_name == drw_equipment_name)
                    else:
                        query = query.filter(Drawing.drw_equipment_name.ilike(f"%{drw_equipment_name}%"))

                if drw_number is not None:
                    if exact_match:
                        query = query.filter(Drawing.drw_number == drw_number)
                    else:
                        query = query.filter(Drawing.drw_number.ilike(f"%{drw_number}%"))

                if drw_name is not None:
                    if exact_match:
                        query = query.filter(Drawing.drw_name == drw_name)
                    else:
                        query = query.filter(Drawing.drw_name.ilike(f"%{drw_name}%"))

                if drw_revision is not None:
                    if exact_match:
                        query = query.filter(Drawing.drw_revision == drw_revision)
                    else:
                        query = query.filter(Drawing.drw_revision.ilike(f"%{drw_revision}%"))

                if drw_spare_part_number is not None:
                    if exact_match:
                        query = query.filter(Drawing.drw_spare_part_number == drw_spare_part_number)
                    else:
                        query = query.filter(Drawing.drw_spare_part_number.ilike(f"%{drw_spare_part_number}%"))

                # Make results distinct to avoid duplicates
                query = query.distinct()

                # Apply limit
                query = query.limit(limit)

                # Execute query and log results
                results = query.all()
                debug_id(f"Part.get_drawings_by_part completed, found {len(results)} drawings", rid)
                return results

        except Exception as e:
            error_id(f"Error in Part.get_drawings_by_part: {str(e)}", rid, exc_info=True)
            return []
        finally:
            # Close the session if we created it
            if not session_provided:
                session.close()
                debug_id(f"Closed database session for Part.get_drawings_by_part", rid)

    # ======================================================
    # BULK OPERATIONS
    # ======================================================
    @classmethod
    @with_request_id
    def bulk_link(cls,
                  drawing_ids: Optional[List[int]] = None,
                  part_ids: Optional[List[int]] = None,
                  session: Optional[Session] = None,
                  request_id: Optional[str] = None) -> int:
        """
        Bulk link parts and drawings in one transaction.

        You can:
        - Pass a single drawing_id with multiple part_ids
        - Pass a single part_id with multiple drawing_ids

        Args:
            drawing_ids: List of drawing IDs (or a single-element list if linking one drawing to many parts)
            part_ids: List of part IDs (or a single-element list if linking one part to many drawings)
            session: Optional session (if not provided, a new one is created)
            request_id: For logging

        Returns:
            int: Number of new associations created
        """
        rid = request_id or get_request_id()
        db_config = DatabaseConfig()
        session_provided = session is not None
        if not session_provided:
            session = db_config.get_main_session()

        created_count = 0
        try:
            if not drawing_ids or not part_ids:
                warning_id("bulk_link requires both drawing_ids and part_ids", rid)
                return 0

            # Ensure both are lists
            if isinstance(drawing_ids, int):
                drawing_ids = [drawing_ids]
            if isinstance(part_ids, int):
                part_ids = [part_ids]

            # Loop through all combinations
            for d_id in drawing_ids:
                for p_id in part_ids:
                    existing = session.query(cls).filter_by(drawing_id=d_id, part_id=p_id).first()
                    if not existing:
                        assoc = cls(drawing_id=d_id, part_id=p_id)
                        session.add(assoc)
                        created_count += 1
                        debug_id(f"Prepared link Drawing {d_id} ↔ Part {p_id}", rid)
                    else:
                        debug_id(f"Skipped existing link Drawing {d_id} ↔ Part {p_id}", rid)

            if created_count > 0:
                session.commit()
                info_id(f"bulk_link created {created_count} new associations", rid)
            else:
                debug_id("bulk_link found all associations already exist", rid)

            return created_count

        except Exception as e:
            session.rollback()
            error_id(f"bulk_link failed: {e}", rid, exc_info=True)
            return 0
        finally:
            if not session_provided:
                session.close()

class PartProblemAssociation(Base):
    __tablename__ = 'part_problem'
    id = Column(Integer, primary_key=True)
    part_id = Column(Integer, ForeignKey('part.id'))
    problem_id = Column(Integer, ForeignKey('problem.id'))
    
    part = relationship("Part", back_populates="part_problem")
    problem = relationship("Problem", back_populates="part_problem")

class TaskPositionAssociation(Base):
    __tablename__ = 'task_position'

    id = Column(Integer, primary_key=True)
    task_id = Column(Integer, ForeignKey('task.id'), nullable=False)
    position_id = Column(Integer, ForeignKey('position.id'), nullable=False)

    # Define relationships
    task = relationship("Task", back_populates="task_positions")
    position = relationship("Position", back_populates="position_tasks")

    @classmethod
    @with_request_id
    def get_positions_by_task_id(cls, session=None, task_id=None, name=None, description=None,
                                 area_id=None, equipment_group_id=None, model_id=None,
                                 asset_number_id=None, location_id=None, subassembly_id=None,
                                 component_assembly_id=None, assembly_view_id=None, site_location_id=None):
        """
        Get all positions associated with a specific task or set of task criteria.

        Args:
            session: SQLAlchemy session (Optional)
            task_id: ID of the task (Optional)
            name: Filter by task name (Optional)
            description: Filter by task description (Optional)
            area_id, equipment_group_id, etc.: Position hierarchy filters (Optional)

        Returns:
            List of Position objects matching the criteria
        """
        if session is None:
            session = DatabaseConfig().get_main_session()

        try:


            # Start with a query to get task(s)
            task_query = session.query(cls)

            # Apply task filters if provided
            if task_id is not None:
                task_query = task_query.filter(cls.id == task_id)
            if name is not None:
                task_query = task_query.filter(cls.name.like(f"%{name}%"))
            if description is not None:
                task_query = task_query.filter(cls.description.like(f"%{description}%"))

            tasks = task_query.all()
            if not tasks:
                return []

            task_ids = [t.id for t in tasks]

            # Start with a query that joins Position and TaskPositionAssociation
            query = session.query(Position).join(TaskPositionAssociation)

            # Filter by task IDs
            query = query.filter(TaskPositionAssociation.task_id.in_(task_ids))

            # Apply position hierarchy filters if provided
            if area_id is not None:
                query = query.filter(Position.area_id == area_id)
            if equipment_group_id is not None:
                query = query.filter(Position.equipment_group_id == equipment_group_id)
            if model_id is not None:
                query = query.filter(Position.model_id == model_id)
            if asset_number_id is not None:
                query = query.filter(Position.asset_number_id == asset_number_id)
            if location_id is not None:
                query = query.filter(Position.location_id == location_id)
            if subassembly_id is not None:
                query = query.filter(Position.subassembly_id == subassembly_id)
            if component_assembly_id is not None:
                query = query.filter(Position.component_assembly_id == component_assembly_id)
            if assembly_view_id is not None:
                query = query.filter(Position.assembly_view_id == assembly_view_id)
            if site_location_id is not None:
                query = query.filter(Position.site_location_id == site_location_id)

            # Make results distinct in case multiple tasks point to same position
            query = query.distinct()

            debug_id(f"Getting positions with filters: task_id={task_id}, name={name}, description={description}")
            positions = query.all()
            info_id(f"Found {len(positions)} positions matching the criteria")

            return positions

        except SQLAlchemyError as e:
            error_id(f"Error getting positions: {str(e)}", exc_info=True)
            return []

    @classmethod
    @with_request_id
    def get_tasks_by_position_id(cls, session=None, position_id=None, name=None, description=None,
                                 area_id=None, equipment_group_id=None, model_id=None,
                                 asset_number_id=None, location_id=None, subassembly_id=None,
                                 component_assembly_id=None, assembly_view_id=None, site_location_id=None):
        """
        Get all tasks associated with a specific position or set of position criteria.

        Args:
            session: SQLAlchemy session (Optional)
            position_id: ID of the position (Optional)
            name: Filter tasks by name (Optional)
            description: Filter tasks by description (Optional)
            area_id, equipment_group_id, etc.: Position hierarchy filters (Optional)

        Returns:
            List of Task objects matching the criteria
        """
        if session is None:
            session = DatabaseConfig().get_main_session()

        try:


            # Start with a query that joins Task and TaskPositionAssociation
            query = session.query(Task).join(TaskPositionAssociation)

            # If position_id is provided, filter by that specific position
            if position_id:
                query = query.filter(TaskPositionAssociation.position_id == position_id)
            else:
                # If no position_id but position hierarchy filters are provided
                position_filters = {}
                if area_id is not None:
                    position_filters['area_id'] = area_id
                if equipment_group_id is not None:
                    position_filters['equipment_group_id'] = equipment_group_id
                if model_id is not None:
                    position_filters['model_id'] = model_id
                if asset_number_id is not None:
                    position_filters['asset_number_id'] = asset_number_id
                if location_id is not None:
                    position_filters['location_id'] = location_id
                if subassembly_id is not None:
                    position_filters['subassembly_id'] = subassembly_id
                if component_assembly_id is not None:
                    position_filters['component_assembly_id'] = component_assembly_id
                if assembly_view_id is not None:
                    position_filters['assembly_view_id'] = assembly_view_id
                if site_location_id is not None:
                    position_filters['site_location_id'] = site_location_id

                if position_filters:
                    # Get position IDs matching the criteria
                    positions = session.query(cls).filter_by(**position_filters).all()
                    position_ids = [p.id for p in positions]

                    if not position_ids:
                        return []  # No positions match the criteria

                    query = query.filter(TaskPositionAssociation.position_id.in_(position_ids))

            # Apply task-specific filters if provided
            if name is not None:
                query = query.filter(Task.name.like(f"%{name}%"))
            if description is not None:
                query = query.filter(Task.description.like(f"%{description}%"))

            # Make results distinct in case same task appears in multiple positions
            query = query.distinct()

            debug_id(f"Getting tasks with filters: position_id={position_id}, name={name}, description={description}")
            tasks = query.all()
            info_id(f"Found {len(tasks)} tasks matching the criteria")

            return tasks

        except SQLAlchemyError as e:
            error_id(f"Error getting tasks: {str(e)}", exc_info=True)
            return []

    @classmethod
    @with_request_id
    def associate_task_position(cls,
                                task_id: int,
                                position_id: int,
                                request_id: Optional[str] = None,
                                session: Optional[Session] = None) -> Optional['TaskPositionAssociation']:
        """
        Associate a task with a position.

        Args:
            task_id: ID of the task to associate
            position_id: ID of the position to associate
            request_id: Optional request ID for tracking this operation in logs
            session: Optional SQLAlchemy session. If None, a new session will be created

        Returns:
            The created TaskPositionAssociation object if successful, None otherwise
        """
        # Get or use the provided request_id
        rid = request_id or get_request_id()

        # Get a database session if one wasn't provided
        db_config = DatabaseConfig()
        session_provided = session is not None
        if not session_provided:
            session = db_config.get_main_session()
            debug_id(f"Created new database session for TaskPositionAssociation.associate_task_position", rid)

        # Log the operation with request ID
        debug_id(
            f"Starting TaskPositionAssociation.associate_task_position with parameters: task_id={task_id}, position_id={position_id}",
            rid)

        try:
            # Use the timed operation context manager with request ID
            with log_timed_operation("TaskPositionAssociation.associate_task_position", rid):


                # Check if task exists
                task = session.query(Task).filter(Task.id == task_id).first()
                if not task:
                    error_id(
                        f"Error in TaskPositionAssociation.associate_task_position: Task with ID {task_id} not found",
                        rid)
                    return None

                # Check if position exists
                position = session.query(Position).filter(Position.id == position_id).first()
                if not position:
                    error_id(
                        f"Error in TaskPositionAssociation.associate_task_position: Position with ID {position_id} not found",
                        rid)
                    return None

                # Check if association already exists
                existing = session.query(cls).filter(
                    cls.task_id == task_id,
                    cls.position_id == position_id
                ).first()

                if existing:
                    debug_id(f"Association between task {task_id} and position {position_id} already exists", rid)
                    return existing

                # Create new association
                association = cls(task_id=task_id, position_id=position_id)
                session.add(association)

                # Commit if we created the session
                if not session_provided:
                    session.commit()
                    debug_id(f"Committed new association between task {task_id} and position {position_id}", rid)

                return association

        except Exception as e:
            error_id(f"Error in TaskPositionAssociation.associate_task_position: {str(e)}", rid, exc_info=True)
            if not session_provided:
                session.rollback()
                debug_id(f"Rolled back transaction in TaskPositionAssociation.associate_task_position", rid)
            return None
        finally:
            # Close the session if we created it
            if not session_provided:
                session.close()
                debug_id(f"Closed database session for TaskPositionAssociation.associate_task_position", rid)

    @classmethod
    @with_request_id
    def dissociate_task_position(cls,
                                 task_id: int,
                                 position_id: int,
                                 request_id: Optional[str] = None,
                                 session: Optional[Session] = None) -> bool:
        """
        Remove an association between a task and a position.

        Args:
            task_id: ID of the task to dissociate
            position_id: ID of the position to dissociate
            request_id: Optional request ID for tracking this operation in logs
            session: Optional SQLAlchemy session. If None, a new session will be created

        Returns:
            True if the association was removed, False otherwise
        """
        # Get or use the provided request_id
        rid = request_id or get_request_id()

        # Get a database session if one wasn't provided
        db_config = DatabaseConfig()
        session_provided = session is not None
        if not session_provided:
            session = db_config.get_main_session()
            debug_id(f"Created new database session for TaskPositionAssociation.dissociate_task_position", rid)

        # Log the operation with request ID
        debug_id(
            f"Starting TaskPositionAssociation.dissociate_task_position with parameters: task_id={task_id}, position_id={position_id}",
            rid)

        try:
            # Use the timed operation context manager with request ID
            with log_timed_operation("TaskPositionAssociation.dissociate_task_position", rid):
                # Find the association
                association = session.query(cls).filter(
                    cls.task_id == task_id,
                    cls.position_id == position_id
                ).first()

                if not association:
                    debug_id(f"No association found between task {task_id} and position {position_id}", rid)
                    return False

                # Delete the association
                session.delete(association)

                # Commit if we created the session
                if not session_provided:
                    session.commit()
                    debug_id(f"Removed association between task {task_id} and position {position_id}", rid)

                return True

        except Exception as e:
            error_id(f"Error in TaskPositionAssociation.dissociate_task_position: {str(e)}", rid, exc_info=True)
            if not session_provided:
                session.rollback()
                debug_id(f"Rolled back transaction in TaskPositionAssociation.dissociate_task_position", rid)
            return False
        finally:
            # Close the session if we created it
            if not session_provided:
                session.close()
                debug_id(f"Closed database session for TaskPositionAssociation.dissociate_task_position", rid)

    @classmethod
    @with_request_id
    def associate_multiple_tasks_to_position(cls,
                                             task_ids: List[int],
                                             position_id: int,
                                             request_id: Optional[str] = None,
                                             session: Optional[Session] = None) -> Dict[int, bool]:
        """
        Associate multiple tasks with a single position.

        Args:
            task_ids: List of task IDs to associate
            position_id: ID of the position to associate with all tasks
            request_id: Optional request ID for tracking this operation in logs
            session: Optional SQLAlchemy session. If None, a new session will be created

        Returns:
            Dictionary mapping task IDs to success status (True if associated, False if failed)
        """
        # Get or use the provided request_id
        rid = request_id or get_request_id()

        # Get a database session if one wasn't provided
        db_config = DatabaseConfig()
        session_provided = session is not None
        if not session_provided:
            session = db_config.get_main_session()
            debug_id(f"Created new database session for TaskPositionAssociation.associate_multiple_tasks_to_position",
                     rid)

        # Log the operation with request ID
        debug_id(f"Starting TaskPositionAssociation.associate_multiple_tasks_to_position with parameters: "
                 f"task_ids={task_ids}, position_id={position_id}", rid)

        results = {}
        try:
            # Check if position exists

            position = session.query(Position).filter(Position.id == position_id).first()
            if not position:
                error_id(f"Error in TaskPositionAssociation.associate_multiple_tasks_to_position: "
                         f"Position with ID {position_id} not found", rid)
                return {task_id: False for task_id in task_ids}

            # Process each task
            for task_id in task_ids:
                try:
                    association = cls.associate_task_position(
                        task_id=task_id,
                        position_id=position_id,
                        request_id=rid,
                        session=session
                    )
                    results[task_id] = association is not None
                except Exception as e:
                    error_id(f"Error associating task {task_id} with position {position_id}: {str(e)}", rid)
                    results[task_id] = False

            # Commit if we created the session
            if not session_provided:
                session.commit()
                debug_id(f"Committed all associations in associate_multiple_tasks_to_position", rid)

            # Log summary
            success_count = sum(1 for success in results.values() if success)
            debug_id(
                f"Successfully associated {success_count} out of {len(task_ids)} tasks with position {position_id}",
                rid)

            return results

        except Exception as e:
            error_id(f"Error in TaskPositionAssociation.associate_multiple_tasks_to_position: {str(e)}", rid,
                     exc_info=True)
            if not session_provided:
                session.rollback()
                debug_id(f"Rolled back transaction in TaskPositionAssociation.associate_multiple_tasks_to_position",
                         rid)
            return {task_id: False for task_id in task_ids}
        finally:
            # Close the session if we created it
            if not session_provided:
                session.close()
                debug_id(f"Closed database session for TaskPositionAssociation.associate_multiple_tasks_to_position",
                         rid)

    @classmethod
    @with_request_id
    def associate_task_to_multiple_positions(cls,
                                             task_id: int,
                                             position_ids: List[int],
                                             request_id: Optional[str] = None,
                                             session: Optional[Session] = None) -> Dict[int, bool]:
        """
        Associate a single task with multiple positions.

        Args:
            task_id: ID of the task to associate
            position_ids: List of position IDs to associate with the task
            request_id: Optional request ID for tracking this operation in logs
            session: Optional SQLAlchemy session. If None, a new session will be created

        Returns:
            Dictionary mapping position IDs to success status (True if associated, False if failed)
        """
        # Get or use the provided request_id
        rid = request_id or get_request_id()

        # Get a database session if one wasn't provided
        db_config = DatabaseConfig()
        session_provided = session is not None
        if not session_provided:
            session = db_config.get_main_session()
            debug_id(f"Created new database session for TaskPositionAssociation.associate_task_to_multiple_positions",
                     rid)

        # Log the operation with request ID
        debug_id(f"Starting TaskPositionAssociation.associate_task_to_multiple_positions with parameters: "
                 f"task_id={task_id}, position_ids={position_ids}", rid)

        results = {}
        try:
            # Check if task exists

            task = session.query(Task).filter(Task.id == task_id).first()
            if not task:
                error_id(f"Error in TaskPositionAssociation.associate_task_to_multiple_positions: "
                         f"Task with ID {task_id} not found", rid)
                return {position_id: False for position_id in position_ids}

            # Process each position
            for position_id in position_ids:
                try:
                    association = cls.associate_task_position(
                        task_id=task_id,
                        position_id=position_id,
                        request_id=rid,
                        session=session
                    )
                    results[position_id] = association is not None
                except Exception as e:
                    error_id(f"Error associating task {task_id} with position {position_id}: {str(e)}", rid)
                    results[position_id] = False

            # Commit if we created the session
            if not session_provided:
                session.commit()
                debug_id(f"Committed all associations in associate_task_to_multiple_positions", rid)

            # Log summary
            success_count = sum(1 for success in results.values() if success)
            debug_id(
                f"Successfully associated task {task_id} with {success_count} out of {len(position_ids)} positions",
                rid)

            return results

        except Exception as e:
            error_id(f"Error in TaskPositionAssociation.associate_task_to_multiple_positions: {str(e)}", rid,
                     exc_info=True)
            if not session_provided:
                session.rollback()
                debug_id(f"Rolled back transaction in TaskPositionAssociation.associate_task_to_multiple_positions",
                         rid)
            return {position_id: False for position_id in position_ids}
        finally:
            # Close the session if we created it
            if not session_provided:
                session.close()
                debug_id(f"Closed database session for TaskPositionAssociation.associate_task_to_multiple_positions",
                         rid)

class PartTaskAssociation(Base):
    __tablename__ = 'part_task'

    id = Column(Integer, primary_key=True)
    part_id = Column(Integer, ForeignKey('part.id'))
    task_id = Column(Integer, ForeignKey('task.id'))  # Corrected foreign key

    part = relationship("Part", back_populates="part_task")
    task = relationship("Task", back_populates="part_task")

    @classmethod
    @with_request_id
    def get_tasks_by_part(cls,
                          part_id: Optional[int] = None,
                          part_number: Optional[str] = None,
                          name: Optional[str] = None,
                          oem_mfg: Optional[str] = None,
                          model: Optional[str] = None,
                          task_id: Optional[int] = None,
                          task_name: Optional[str] = None,
                          task_description: Optional[str] = None,
                          exact_match: bool = False,
                          limit: int = 100,
                          request_id: Optional[str] = None,
                          session: Optional[Session] = None) -> List['Task']:
        """
        Get tasks associated with parts based on flexible search criteria.

        Args:
            part_id: Optional part ID to filter by
            part_number, name, oem_mfg, model: Optional part attributes to filter by
            task_id: Optional task ID to filter by
            task_name: Optional task name to filter by
            task_description: Optional task description to filter by
            exact_match: If True, performs exact matching instead of partial matching for string fields
            limit: Maximum number of results to return (default 100)
            request_id: Optional request ID for tracking this operation in logs
            session: Optional SQLAlchemy session. If None, a new session will be created

        Returns:
            List of Task objects matching the search criteria
        """
        # Get or use the provided request_id
        rid = request_id or get_request_id()

        # Get a database session if one wasn't provided
        db_config = DatabaseConfig()
        session_provided = session is not None
        if not session_provided:
            session = db_config.get_main_session()
            debug_id(f"Created new database session for Part.get_tasks_by_part", rid)

        # Log the search operation with request ID
        search_params = {
            'part_id': part_id,
            'part_number': part_number,
            'name': name,
            'oem_mfg': oem_mfg,
            'model': model,
            'task_id': task_id,
            'task_name': task_name,
            'task_description': task_description,
            'exact_match': exact_match,
            'limit': limit
        }
        # Filter out None values for cleaner logging
        logged_params = {k: v for k, v in search_params.items() if v is not None}
        debug_id(f"Starting Part.get_tasks_by_part with parameters: {logged_params}", rid)

        try:
            # Use the timed operation context manager with request ID
            with log_timed_operation("Part.get_tasks_by_part", rid):


                # Start with a query that joins Task and PartTaskAssociation
                query = session.query(Task).join(PartTaskAssociation).join(Part)

                # Apply part filters
                if part_id is not None:
                    query = query.filter(Part.id == part_id)

                if part_number is not None:
                    if exact_match:
                        query = query.filter(Part.part_number == part_number)
                    else:
                        query = query.filter(Part.part_number.ilike(f"%{part_number}%"))

                if name is not None:
                    if exact_match:
                        query = query.filter(Part.name == name)
                    else:
                        query = query.filter(Part.name.ilike(f"%{name}%"))

                if oem_mfg is not None:
                    if exact_match:
                        query = query.filter(Part.oem_mfg == oem_mfg)
                    else:
                        query = query.filter(Part.oem_mfg.ilike(f"%{oem_mfg}%"))

                if model is not None:
                    if exact_match:
                        query = query.filter(Part.model == model)
                    else:
                        query = query.filter(Part.model.ilike(f"%{model}%"))

                # Apply task filters
                if task_id is not None:
                    query = query.filter(Task.id == task_id)

                if task_name is not None:
                    if exact_match:
                        query = query.filter(Task.name == task_name)
                    else:
                        query = query.filter(Task.name.ilike(f"%{task_name}%"))

                if task_description is not None:
                    if exact_match:
                        query = query.filter(Task.description == task_description)
                    else:
                        query = query.filter(Task.description.ilike(f"%{task_description}%"))

                # Make results distinct to avoid duplicates
                query = query.distinct()

                # Apply limit
                query = query.limit(limit)

                # Execute query and log results
                results = query.all()
                debug_id(f"Part.get_tasks_by_part completed, found {len(results)} tasks", rid)
                return results

        except Exception as e:
            error_id(f"Error in Part.get_tasks_by_part: {str(e)}", rid, exc_info=True)
            return []
        finally:
            # Close the session if we created it
            if not session_provided:
                session.close()
                debug_id(f"Closed database session for Part.get_tasks_by_part", rid)

    @classmethod
    @with_request_id
    def get_parts_by_task(cls,
                          task_id: Optional[int] = None,
                          task_name: Optional[str] = None,
                          task_description: Optional[str] = None,
                          part_id: Optional[int] = None,
                          part_number: Optional[str] = None,
                          part_name: Optional[str] = None,
                          oem_mfg: Optional[str] = None,
                          model: Optional[str] = None,
                          class_flag: Optional[str] = None,
                          exact_match: bool = False,
                          limit: int = 100,
                          request_id: Optional[str] = None,
                          session: Optional[Session] = None) -> List['Part']:
        """
        Get parts associated with tasks based on flexible search criteria.

        Args:
            task_id: Optional task ID to filter by
            task_name: Optional task name to filter by
            task_description: Optional task description to filter by
            part_id: Optional part ID to filter by
            part_number, part_name, oem_mfg, model, class_flag: Optional part attributes to filter by
            exact_match: If True, performs exact matching instead of partial matching for string fields
            limit: Maximum number of results to return (default 100)
            request_id: Optional request ID for tracking this operation in logs
            session: Optional SQLAlchemy session. If None, a new session will be created

        Returns:
            List of Part objects matching the search criteria
        """
        # Get or use the provided request_id
        rid = request_id or get_request_id()

        # Get a database session if one wasn't provided
        db_config = DatabaseConfig()
        session_provided = session is not None
        if not session_provided:
            session = db_config.get_main_session()
            debug_id(f"Created new database session for Task.get_parts_by_task", rid)

        # Log the search operation with request ID
        search_params = {
            'task_id': task_id,
            'task_name': task_name,
            'task_description': task_description,
            'part_id': part_id,
            'part_number': part_number,
            'part_name': part_name,
            'oem_mfg': oem_mfg,
            'model': model,
            'class_flag': class_flag,
            'exact_match': exact_match,
            'limit': limit
        }
        # Filter out None values for cleaner logging
        logged_params = {k: v for k, v in search_params.items() if v is not None}
        debug_id(f"Starting Task.get_parts_by_task with parameters: {logged_params}", rid)

        try:
            # Use the timed operation context manager with request ID
            with log_timed_operation("Task.get_parts_by_task", rid):


                # Start with a query that joins Part and PartTaskAssociation
                query = session.query(Part).join(PartTaskAssociation).join(Task)

                # Apply task filters
                if task_id is not None:
                    query = query.filter(Task.id == task_id)

                if task_name is not None:
                    if exact_match:
                        query = query.filter(Task.name == task_name)
                    else:
                        query = query.filter(Task.name.ilike(f"%{task_name}%"))

                if task_description is not None:
                    if exact_match:
                        query = query.filter(Task.description == task_description)
                    else:
                        query = query.filter(Task.description.ilike(f"%{task_description}%"))

                # Apply part filters
                if part_id is not None:
                    query = query.filter(Part.id == part_id)

                if part_number is not None:
                    if exact_match:
                        query = query.filter(Part.part_number == part_number)
                    else:
                        query = query.filter(Part.part_number.ilike(f"%{part_number}%"))

                if part_name is not None:
                    if exact_match:
                        query = query.filter(Part.name == part_name)
                    else:
                        query = query.filter(Part.name.ilike(f"%{part_name}%"))

                if oem_mfg is not None:
                    if exact_match:
                        query = query.filter(Part.oem_mfg == oem_mfg)
                    else:
                        query = query.filter(Part.oem_mfg.ilike(f"%{oem_mfg}%"))

                if model is not None:
                    if exact_match:
                        query = query.filter(Part.model == model)
                    else:
                        query = query.filter(Part.model.ilike(f"%{model}%"))

                if class_flag is not None:
                    if exact_match:
                        query = query.filter(Part.class_flag == class_flag)
                    else:
                        query = query.filter(Part.class_flag.ilike(f"%{class_flag}%"))

                # Make results distinct to avoid duplicates
                query = query.distinct()

                # Apply limit
                query = query.limit(limit)

                # Execute query and log results
                results = query.all()
                debug_id(f"Task.get_parts_by_task completed, found {len(results)} parts", rid)
                return results

        except Exception as e:
            error_id(f"Error in Task.get_parts_by_task: {str(e)}", rid, exc_info=True)
            return []
        finally:
            # Close the session if we created it
            if not session_provided:
                session.close()
                debug_id(f"Closed database session for Task.get_parts_by_task", rid)

    @classmethod
    @with_request_id
    def associate_part_with_task(cls, session=None, part_id=None, task_id=None, request_id=None):
        """
        Create an association between a part and a task if it doesn't already exist.

        Args:
            session: SQLAlchemy session (optional)
            part_id: ID of the part to associate
            task_id: ID of the task to associate
            request_id: Optional request ID for tracking

        Returns:
            The PartTaskAssociation instance (existing or new)
        """
        # Get or use the provided request_id
        rid = request_id or get_request_id()

        # Get a database session if one wasn't provided
        db_config = DatabaseConfig()
        session_provided = session is not None
        if not session_provided:
            session = db_config.get_main_session()
            debug_id(f"Created new database session for PartTaskAssociation.associate_part_with_task", rid)

        debug_id(f"Associating part ID {part_id} with task ID {task_id}", rid)

        try:


            # Check if the part and task exist
            part = session.query(Part).get(part_id)
            task = session.query(Task).get(task_id)

            if not part:
                error_id(f"Part with ID {part_id} not found", rid)
                return None

            if not task:
                error_id(f"Task with ID {task_id} not found", rid)
                return None

            # Check if the association already exists
            existing = session.query(cls).filter(
                and_(
                    cls.part_id == part_id,
                    cls.task_id == task_id
                )
            ).first()

            if existing:
                debug_id(f"Association between part ID {part_id} and task ID {task_id} already exists", rid)
                return existing

            # Create new association
            association = cls(
                part_id=part_id,
                task_id=task_id
            )
            session.add(association)
            session.flush()

            debug_id(f"Created new association between part ID {part_id} and task ID {task_id}", rid)
            return association

        except Exception as e:
            error_id(f"Error associating part with task: {str(e)}", rid, exc_info=True)
            return None
        finally:
            # Close the session if we created it
            if not session_provided and session:
                session.close()
                debug_id(f"Closed database session for PartTaskAssociation.associate_part_with_task", rid)

class DrawingTaskAssociation(Base):
    __tablename__ = 'drawing_task'
    id = Column(Integer, primary_key=True)
    drawing_id = Column(Integer, ForeignKey('drawing.id'))
    task_id = Column(Integer, ForeignKey('task.id'))

    drawing = relationship("Drawing", back_populates="drawing_task")
    task = relationship("Task", back_populates="drawing_task")

    @classmethod
    def get_tasks_by_drawing(cls,
                             drawing_id: Optional[int] = None,
                             drw_equipment_name: Optional[str] = None,
                             drw_number: Optional[str] = None,
                             drw_name: Optional[str] = None,
                             drw_revision: Optional[str] = None,
                             drw_spare_part_number: Optional[str] = None,
                             file_path: Optional[str] = None,
                             task_id: Optional[int] = None,
                             task_name: Optional[str] = None,
                             task_description: Optional[str] = None,
                             exact_match: bool = False,
                             limit: int = 100,
                             request_id: Optional[str] = None,
                             session: Optional[Session] = None) -> List['Task']:
        """
        Get tasks associated with drawings based on flexible search criteria.

        Args:
            drawing_id: Optional drawing ID to filter by
            drw_equipment_name, drw_number, drw_name, drw_revision, drw_spare_part_number, file_path:
                Optional drawing attributes to filter by
            task_id: Optional task ID to filter by
            task_name: Optional task name to filter by
            task_description: Optional task description to filter by
            exact_match: If True, performs exact matching instead of partial matching for string fields
            limit: Maximum number of results to return (default 100)
            request_id: Optional request ID for tracking this operation in logs
            session: Optional SQLAlchemy session. If None, a new session will be created

        Returns:
            List of Task objects matching the search criteria
        """
        # Get or use the provided request_id
        rid = request_id or get_request_id()

        # Get a database session if one wasn't provided
        db_config = DatabaseConfig()
        session_provided = session is not None
        if not session_provided:
            session = db_config.get_main_session()
            debug_id(f"Created new database session for Drawing.get_tasks_by_drawing", rid)

        # Log the search operation with request ID
        search_params = {
            'drawing_id': drawing_id,
            'drw_equipment_name': drw_equipment_name,
            'drw_number': drw_number,
            'drw_name': drw_name,
            'drw_revision': drw_revision,
            'drw_spare_part_number': drw_spare_part_number,
            'file_path': file_path,
            'task_id': task_id,
            'task_name': task_name,
            'task_description': task_description,
            'exact_match': exact_match,
            'limit': limit
        }
        # Filter out None values for cleaner logging
        logged_params = {k: v for k, v in search_params.items() if v is not None}
        debug_id(f"Starting Drawing.get_tasks_by_drawing with parameters: {logged_params}", rid)

        try:
            # Use the timed operation context manager with request ID
            with log_timed_operation("Drawing.get_tasks_by_drawing", rid):


                # Start with a query that joins Task and DrawingTaskAssociation
                query = session.query(Task).join(DrawingTaskAssociation).join(cls)

                # Apply drawing filters
                if drawing_id is not None:
                    query = query.filter(cls.id == drawing_id)

                if drw_equipment_name is not None:
                    if exact_match:
                        query = query.filter(cls.drw_equipment_name == drw_equipment_name)
                    else:
                        query = query.filter(cls.drw_equipment_name.ilike(f"%{drw_equipment_name}%"))

                if drw_number is not None:
                    if exact_match:
                        query = query.filter(cls.drw_number == drw_number)
                    else:
                        query = query.filter(cls.drw_number.ilike(f"%{drw_number}%"))

                if drw_name is not None:
                    if exact_match:
                        query = query.filter(cls.drw_name == drw_name)
                    else:
                        query = query.filter(cls.drw_name.ilike(f"%{drw_name}%"))

                if drw_revision is not None:
                    if exact_match:
                        query = query.filter(cls.drw_revision == drw_revision)
                    else:
                        query = query.filter(cls.drw_revision.ilike(f"%{drw_revision}%"))

                if drw_spare_part_number is not None:
                    if exact_match:
                        query = query.filter(cls.drw_spare_part_number == drw_spare_part_number)
                    else:
                        query = query.filter(cls.drw_spare_part_number.ilike(f"%{drw_spare_part_number}%"))

                if file_path is not None:
                    if exact_match:
                        query = query.filter(cls.file_path == file_path)
                    else:
                        query = query.filter(cls.file_path.ilike(f"%{file_path}%"))

                # Apply task filters
                if task_id is not None:
                    query = query.filter(Task.id == task_id)

                if task_name is not None:
                    if exact_match:
                        query = query.filter(Task.name == task_name)
                    else:
                        query = query.filter(Task.name.ilike(f"%{task_name}%"))

                if task_description is not None:
                    if exact_match:
                        query = query.filter(Task.description == task_description)
                    else:
                        query = query.filter(Task.description.ilike(f"%{task_description}%"))

                # Make results distinct to avoid duplicates
                query = query.distinct()

                # Apply limit
                query = query.limit(limit)

                # Execute query and log results
                results = query.all()
                debug_id(f"Drawing.get_tasks_by_drawing completed, found {len(results)} tasks", rid)
                return results

        except Exception as e:
            error_id(f"Error in Drawing.get_tasks_by_drawing: {str(e)}", rid, exc_info=True)
            return []
        finally:
            # Close the session if we created it
            if not session_provided:
                session.close()
                debug_id(f"Closed database session for Drawing.get_tasks_by_drawing", rid)

    @classmethod
    def get_drawings_by_task(cls,
                             task_id: Optional[int] = None,
                             task_name: Optional[str] = None,
                             task_description: Optional[str] = None,
                             drawing_id: Optional[int] = None,
                             drw_equipment_name: Optional[str] = None,
                             drw_number: Optional[str] = None,
                             drw_name: Optional[str] = None,
                             drw_revision: Optional[str] = None,
                             drw_spare_part_number: Optional[str] = None,
                             file_path: Optional[str] = None,
                             exact_match: bool = False,
                             limit: int = 100,
                             request_id: Optional[str] = None,
                             session: Optional[Session] = None) -> List['Drawing']:
        """
        Get drawings associated with tasks based on flexible search criteria.

        Args:
            task_id: Optional task ID to filter by
            task_name: Optional task name to filter by
            task_description: Optional task description to filter by
            drawing_id: Optional drawing ID to filter by
            drw_equipment_name, drw_number, drw_name, drw_revision, drw_spare_part_number, file_path:
                Optional drawing attributes to filter by
            exact_match: If True, performs exact matching instead of partial matching for string fields
            limit: Maximum number of results to return (default 100)
            request_id: Optional request ID for tracking this operation in logs
            session: Optional SQLAlchemy session. If None, a new session will be created

        Returns:
            List of Drawing objects matching the search criteria
        """
        # Get or use the provided request_id
        rid = request_id or get_request_id()

        # Get a database session if one wasn't provided
        db_config = DatabaseConfig()
        session_provided = session is not None
        if not session_provided:
            session = db_config.get_main_session()
            debug_id(f"Created new database session for Task.get_drawings_by_task", rid)

        # Log the search operation with request ID
        search_params = {
            'task_id': task_id,
            'task_name': task_name,
            'task_description': task_description,
            'drawing_id': drawing_id,
            'drw_equipment_name': drw_equipment_name,
            'drw_number': drw_number,
            'drw_name': drw_name,
            'drw_revision': drw_revision,
            'drw_spare_part_number': drw_spare_part_number,
            'file_path': file_path,
            'exact_match': exact_match,
            'limit': limit
        }
        # Filter out None values for cleaner logging
        logged_params = {k: v for k, v in search_params.items() if v is not None}
        debug_id(f"Starting Task.get_drawings_by_task with parameters: {logged_params}", rid)

        try:
            # Use the timed operation context manager with request ID
            with log_timed_operation("Task.get_drawings_by_task", rid):


                # Start with a query that joins Drawing and DrawingTaskAssociation
                query = session.query(Drawing).join(DrawingTaskAssociation).join(cls)

                # Apply task filters
                if task_id is not None:
                    query = query.filter(cls.id == task_id)

                if task_name is not None:
                    if exact_match:
                        query = query.filter(cls.name == task_name)
                    else:
                        query = query.filter(cls.name.ilike(f"%{task_name}%"))

                if task_description is not None:
                    if exact_match:
                        query = query.filter(cls.description == task_description)
                    else:
                        query = query.filter(cls.description.ilike(f"%{task_description}%"))

                # Apply drawing filters
                if drawing_id is not None:
                    query = query.filter(Drawing.id == drawing_id)

                if drw_equipment_name is not None:
                    if exact_match:
                        query = query.filter(Drawing.drw_equipment_name == drw_equipment_name)
                    else:
                        query = query.filter(Drawing.drw_equipment_name.ilike(f"%{drw_equipment_name}%"))

                if drw_number is not None:
                    if exact_match:
                        query = query.filter(Drawing.drw_number == drw_number)
                    else:
                        query = query.filter(Drawing.drw_number.ilike(f"%{drw_number}%"))

                if drw_name is not None:
                    if exact_match:
                        query = query.filter(Drawing.drw_name == drw_name)
                    else:
                        query = query.filter(Drawing.drw_name.ilike(f"%{drw_name}%"))

                if drw_revision is not None:
                    if exact_match:
                        query = query.filter(Drawing.drw_revision == drw_revision)
                    else:
                        query = query.filter(Drawing.drw_revision.ilike(f"%{drw_revision}%"))

                if drw_spare_part_number is not None:
                    if exact_match:
                        query = query.filter(Drawing.drw_spare_part_number == drw_spare_part_number)
                    else:
                        query = query.filter(Drawing.drw_spare_part_number.ilike(f"%{drw_spare_part_number}%"))

                if file_path is not None:
                    if exact_match:
                        query = query.filter(Drawing.file_path == file_path)
                    else:
                        query = query.filter(Drawing.file_path.ilike(f"%{file_path}%"))

                # Make results distinct to avoid duplicates
                query = query.distinct()

                # Apply limit
                query = query.limit(limit)

                # Execute query and log results
                results = query.all()
                debug_id(f"Task.get_drawings_by_task completed, found {len(results)} drawings", rid)
                return results

        except Exception as e:
            error_id(f"Error in Task.get_drawings_by_task: {str(e)}", rid, exc_info=True)
            return []
        finally:
            # Close the session if we created it
            if not session_provided:
                session.close()
                debug_id(f"Closed database session for Task.get_drawings_by_task", rid)

    @classmethod
    @with_request_id
    def associate_drawing_with_task(cls, session, drawing_id, task_id, request_id=None):
        """
        Create an association between a drawing and a task if it doesn't already exist.

        Args:
            session: SQLAlchemy session
            drawing_id: ID of the drawing to associate
            task_id: ID of the task to associate
            request_id: Optional request ID for tracking

        Returns:
            The DrawingTaskAssociation instance (existing or new)
        """


        # Get or use the provided request_id
        rid = request_id or get_request_id()

        debug_id(f"Associating drawing ID {drawing_id} with task ID {task_id}", rid)

        try:
            # Check if the drawing and task exist
            drawing = session.query(Drawing).get(drawing_id)
            task = session.query(cls).get(task_id)

            if not drawing:
                error_id(f"Drawing with ID {drawing_id} not found", rid)
                return None

            if not task:
                error_id(f"Task with ID {task_id} not found", rid)
                return None

            # Check if the association already exists
            existing = session.query(DrawingTaskAssociation).filter(
                and_(
                    DrawingTaskAssociation.drawing_id == drawing_id,
                    DrawingTaskAssociation.task_id == task_id
                )
            ).first()

            if existing:
                debug_id(f"Association between drawing ID {drawing_id} and task ID {task_id} already exists", rid)
                return existing

            # Create new association
            association = DrawingTaskAssociation(
                drawing_id=drawing_id,
                task_id=task_id
            )
            session.add(association)
            session.flush()

            debug_id(f"Created new association between drawing ID {drawing_id} and task ID {task_id}", rid)
            return association

        except Exception as e:
            error_id(f"Error associating drawing with task: {str(e)}", rid, exc_info=True)
            return None

class ImageTaskAssociation(Base):

    __tablename__ = 'image_task'

    id = Column(Integer, primary_key=True)
    image_id = Column(Integer, ForeignKey('image.id'))
    task_id = Column(Integer, ForeignKey('task.id'))  # Corrected foreign key

    image = relationship("Image", back_populates="image_task")
    task = relationship("Task", back_populates="image_task")

    @classmethod
    @with_request_id
    def get_tasks_by_image(cls,
                           image_id: Optional[int] = None,
                           title: Optional[str] = None,
                           description: Optional[str] = None,
                           file_path: Optional[str] = None,
                           task_id: Optional[int] = None,
                           task_name: Optional[str] = None,
                           task_description: Optional[str] = None,
                           exact_match: bool = False,
                           limit: int = 100,
                           request_id: Optional[str] = None,
                           session: Optional[Session] = None) -> List['Task']:
        """
        Get tasks associated with images based on flexible search criteria.

        Args:
            image_id: Optional image ID to filter by
            title: Optional image title to filter by
            description: Optional image description to filter by
            file_path: Optional file path to filter by
            task_id: Optional task ID to filter by
            task_name: Optional task name to filter by
            task_description: Optional task description to filter by
            exact_match: If True, performs exact matching instead of partial matching for string fields
            limit: Maximum number of results to return (default 100)
            request_id: Optional request ID for tracking this operation in logs
            session: Optional SQLAlchemy session. If None, a new session will be created

        Returns:
            List of Task objects matching the search criteria
        """
        # Get or use the provided request_id
        rid = request_id or get_request_id()

        # Get a database session if one wasn't provided
        db_config = DatabaseConfig()
        session_provided = session is not None
        if not session_provided:
            session = db_config.get_main_session()
            debug_id(f"Created new database session for Image.get_tasks_by_image", rid)

        # Log the search operation with request ID
        search_params = {
            'image_id': image_id,
            'title': title,
            'description': description,
            'file_path': file_path,
            'task_id': task_id,
            'task_name': task_name,
            'task_description': task_description,
            'exact_match': exact_match,
            'limit': limit
        }
        # Filter out None values for cleaner logging
        logged_params = {k: v for k, v in search_params.items() if v is not None}
        debug_id(f"Starting Image.get_tasks_by_image with parameters: {logged_params}", rid)

        try:
            # Use the timed operation context manager with request ID
            with log_timed_operation("Image.get_tasks_by_image", rid):


                # Start with a query that joins Task and ImageTaskAssociation
                query = session.query(Task).join(ImageTaskAssociation).join(cls)

                # Apply image filters
                if image_id is not None:
                    query = query.filter(cls.id == image_id)

                if title is not None:
                    if exact_match:
                        query = query.filter(cls.title == title)
                    else:
                        query = query.filter(cls.title.ilike(f"%{title}%"))

                if description is not None:
                    if exact_match:
                        query = query.filter(cls.description == description)
                    else:
                        query = query.filter(cls.description.ilike(f"%{description}%"))

                if file_path is not None:
                    if exact_match:
                        query = query.filter(cls.file_path == file_path)
                    else:
                        query = query.filter(cls.file_path.ilike(f"%{file_path}%"))

                # Apply task filters
                if task_id is not None:
                    query = query.filter(Task.id == task_id)

                if task_name is not None:
                    if exact_match:
                        query = query.filter(Task.name == task_name)
                    else:
                        query = query.filter(Task.name.ilike(f"%{task_name}%"))

                if task_description is not None:
                    if exact_match:
                        query = query.filter(Task.description == task_description)
                    else:
                        query = query.filter(Task.description.ilike(f"%{task_description}%"))

                # Make results distinct to avoid duplicates
                query = query.distinct()

                # Apply limit
                query = query.limit(limit)

                # Execute query and log results
                results = query.all()
                debug_id(f"Image.get_tasks_by_image completed, found {len(results)} tasks", rid)
                return results

        except Exception as e:
            error_id(f"Error in Image.get_tasks_by_image: {str(e)}", rid, exc_info=True)
            return []
        finally:
            # Close the session if we created it
            if not session_provided:
                session.close()
                debug_id(f"Closed database session for Image.get_tasks_by_image", rid)

    @classmethod
    @with_request_id
    def get_images_by_task(cls,
                           task_id: Optional[int] = None,
                           task_name: Optional[str] = None,
                           task_description: Optional[str] = None,
                           image_id: Optional[int] = None,
                           title: Optional[str] = None,
                           description: Optional[str] = None,
                           file_path: Optional[str] = None,
                           exact_match: bool = False,
                           limit: int = 100,
                           request_id: Optional[str] = None,
                           session: Optional[Session] = None) -> List['Image']:
        """
        Get images associated with tasks based on flexible search criteria.

        Args:
            task_id: Optional task ID to filter by
            task_name: Optional task name to filter by
            task_description: Optional task description to filter by
            image_id: Optional image ID to filter by
            title: Optional image title to filter by
            description: Optional image description to filter by
            file_path: Optional file path to filter by
            exact_match: If True, performs exact matching instead of partial matching for string fields
            limit: Maximum number of results to return (default 100)
            request_id: Optional request ID for tracking this operation in logs
            session: Optional SQLAlchemy session. If None, a new session will be created

        Returns:
            List of Image objects matching the search criteria
        """
        # Get or use the provided request_id
        rid = request_id or get_request_id()

        # Get a database session if one wasn't provided
        db_config = DatabaseConfig()
        session_provided = session is not None
        if not session_provided:
            session = db_config.get_main_session()
            debug_id(f"Created new database session for Task.get_images_by_task", rid)

        # Log the search operation with request ID
        search_params = {
            'task_id': task_id,
            'task_name': task_name,
            'task_description': task_description,
            'image_id': image_id,
            'title': title,
            'description': description,
            'file_path': file_path,
            'exact_match': exact_match,
            'limit': limit
        }
        # Filter out None values for cleaner logging
        logged_params = {k: v for k, v in search_params.items() if v is not None}
        debug_id(f"Starting Task.get_images_by_task with parameters: {logged_params}", rid)

        try:
            # Use the timed operation context manager with request ID
            with log_timed_operation("Task.get_images_by_task", rid):


                # Start with a query that joins Image and ImageTaskAssociation
                query = session.query(Image).join(ImageTaskAssociation).join(cls)

                # Apply task filters
                if task_id is not None:
                    query = query.filter(cls.id == task_id)

                if task_name is not None:
                    if exact_match:
                        query = query.filter(cls.name == task_name)
                    else:
                        query = query.filter(cls.name.ilike(f"%{task_name}%"))

                if task_description is not None:
                    if exact_match:
                        query = query.filter(cls.description == task_description)
                    else:
                        query = query.filter(cls.description.ilike(f"%{task_description}%"))

                # Apply image filters
                if image_id is not None:
                    query = query.filter(Image.id == image_id)

                if title is not None:
                    if exact_match:
                        query = query.filter(Image.title == title)
                    else:
                        query = query.filter(Image.title.ilike(f"%{title}%"))

                if description is not None:
                    if exact_match:
                        query = query.filter(Image.description == description)
                    else:
                        query = query.filter(Image.description.ilike(f"%{description}%"))

                if file_path is not None:
                    if exact_match:
                        query = query.filter(Image.file_path == file_path)
                    else:
                        query = query.filter(Image.file_path.ilike(f"%{file_path}%"))

                # Make results distinct to avoid duplicates
                query = query.distinct()

                # Apply limit
                query = query.limit(limit)

                # Execute query and log results
                results = query.all()
                debug_id(f"Task.get_images_by_task completed, found {len(results)} images", rid)
                return results

        except Exception as e:
            error_id(f"Error in Task.get_images_by_task: {str(e)}", rid, exc_info=True)
            return []
        finally:
            # Close the session if we created it
            if not session_provided:
                session.close()
                debug_id(f"Closed database session for Task.get_images_by_task", rid)

    @classmethod
    @with_request_id
    def associate_image_task(cls,
                             image_id: int,
                             task_id: int,
                             request_id: Optional[str] = None,
                             session: Optional[Session] = None) -> Optional['ImageTaskAssociation']:
        """
        Associate an image with a task.

        Args:
            image_id: ID of the image to associate
            task_id: ID of the task to associate
            request_id: Optional request ID for tracking this operation in logs
            session: Optional SQLAlchemy session. If None, a new session will be created

        Returns:
            The created ImageTaskAssociation object if successful, None otherwise
        """
        # Get or use the provided request_id
        rid = request_id or get_request_id()

        # Get a database session if one wasn't provided
        db_config = DatabaseConfig()
        session_provided = session is not None
        if not session_provided:
            session = db_config.get_main_session()
            debug_id(f"Created new database session for ImageTaskAssociation.associate_image_task", rid)

        # Log the operation with request ID
        debug_id(
            f"Starting ImageTaskAssociation.associate_image_task with parameters: image_id={image_id}, task_id={task_id}",
            rid)

        try:
            # Use the timed operation context manager with request ID
            with log_timed_operation("ImageTaskAssociation.associate_image_task", rid):


                # Check if image exists
                image = session.query(Image).filter(Image.id == image_id).first()
                if not image:
                    error_id(f"Error in ImageTaskAssociation.associate_image_task: Image with ID {image_id} not found",
                             rid)
                    return None

                # Check if task exists
                task = session.query(Task).filter(Task.id == task_id).first()
                if not task:
                    error_id(f"Error in ImageTaskAssociation.associate_image_task: Task with ID {task_id} not found",
                             rid)
                    return None

                # Check if association already exists
                existing = session.query(cls).filter(
                    cls.image_id == image_id,
                    cls.task_id == task_id
                ).first()

                if existing:
                    debug_id(f"Association between image {image_id} and task {task_id} already exists", rid)
                    return existing

                # Create new association
                association = cls(image_id=image_id, task_id=task_id)
                session.add(association)

                # Commit if we created the session
                if not session_provided:
                    session.commit()
                    debug_id(f"Committed new association between image {image_id} and task {task_id}", rid)

                return association

        except Exception as e:
            error_id(f"Error in ImageTaskAssociation.associate_image_task: {str(e)}", rid, exc_info=True)
            if not session_provided:
                session.rollback()
                debug_id(f"Rolled back transaction in ImageTaskAssociation.associate_image_task", rid)
            return None
        finally:
            # Close the session if we created it
            if not session_provided:
                session.close()
                debug_id(f"Closed database session for ImageTaskAssociation.associate_image_task", rid)

class TaskToolAssociation(Base):
    __tablename__ = 'tool_task'

    id = Column(Integer, primary_key=True)
    tool_id = Column(Integer, ForeignKey('tool.id'), nullable=False)
    task_id = Column(Integer, ForeignKey('task.id'), nullable=False)

    # Relationships
    tool = relationship("Tool", back_populates="tool_tasks")
    task = relationship("Task", back_populates="tool_tasks")

    @classmethod
    @with_request_id
    def associate_task_tool(cls,
                            task_id: int,
                            tool_id: int,
                            request_id: Optional[str] = None,
                            session: Optional[Session] = None) -> Optional['TaskToolAssociation']:
        """
        Associate a task with a tool.

        Args:
            task_id: ID of the task to associate
            tool_id: ID of the tool to associate
            request_id: Optional request ID for tracking this operation in logs
            session: Optional SQLAlchemy session. If None, a new session will be created

        Returns:
            The created TaskToolAssociation object if successful, None otherwise
        """
        # Get or use the provided request_id
        rid = request_id or get_request_id()

        # Get a database session if one wasn't provided
        db_config = DatabaseConfig()
        session_provided = session is not None
        if not session_provided:
            session = db_config.get_main_session()
            debug_id(f"Created new database session for TaskToolAssociation.associate_task_tool", rid)

        # Log the operation with request ID
        debug_id(
            f"Starting TaskToolAssociation.associate_task_tool with parameters: task_id={task_id}, tool_id={tool_id}",
            rid)

        try:
            # Use the timed operation context manager with request ID
            with log_timed_operation("TaskToolAssociation.associate_task_tool", rid):


                # Check if task exists
                task = session.query(Task).filter(Task.id == task_id).first()
                if not task:
                    error_id(f"Error in TaskToolAssociation.associate_task_tool: Task with ID {task_id} not found", rid)
                    return None

                # Check if tool exists
                tool = session.query(Tool).filter(Tool.id == tool_id).first()
                if not tool:
                    error_id(f"Error in TaskToolAssociation.associate_task_tool: Tool with ID {tool_id} not found", rid)
                    return None

                # Check if association already exists
                existing = session.query(cls).filter(
                    cls.task_id == task_id,
                    cls.tool_id == tool_id
                ).first()

                if existing:
                    debug_id(f"Association between task {task_id} and tool {tool_id} already exists", rid)
                    return existing

                # Create new association
                association = cls(task_id=task_id, tool_id=tool_id)
                session.add(association)

                # Commit if we created the session
                if not session_provided:
                    session.commit()
                    debug_id(f"Committed new association between task {task_id} and tool {tool_id}", rid)

                return association

        except Exception as e:
            error_id(f"Error in TaskToolAssociation.associate_task_tool: {str(e)}", rid, exc_info=True)
            if not session_provided:
                session.rollback()
                debug_id(f"Rolled back transaction in TaskToolAssociation.associate_task_tool", rid)
            return None
        finally:
            # Close the session if we created it
            if not session_provided:
                session.close()
                debug_id(f"Closed database session for TaskToolAssociation.associate_task_tool", rid)

    @classmethod
    @with_request_id
    def dissociate_task_tool(cls,
                             task_id: int,
                             tool_id: int,
                             request_id: Optional[str] = None,
                             session: Optional[Session] = None) -> bool:
        """
        Remove an association between a task and a tool.

        Args:
            task_id: ID of the task to dissociate
            tool_id: ID of the tool to dissociate
            request_id: Optional request ID for tracking this operation in logs
            session: Optional SQLAlchemy session. If None, a new session will be created

        Returns:
            True if the association was removed, False otherwise
        """
        # Get or use the provided request_id
        rid = request_id or get_request_id()

        # Get a database session if one wasn't provided
        db_config = DatabaseConfig()
        session_provided = session is not None
        if not session_provided:
            session = db_config.get_main_session()
            debug_id(f"Created new database session for TaskToolAssociation.dissociate_task_tool", rid)

        # Log the operation with request ID
        debug_id(
            f"Starting TaskToolAssociation.dissociate_task_tool with parameters: task_id={task_id}, tool_id={tool_id}",
            rid)

        try:
            # Use the timed operation context manager with request ID
            with log_timed_operation("TaskToolAssociation.dissociate_task_tool", rid):
                # Find the association
                association = session.query(cls).filter(
                    cls.task_id == task_id,
                    cls.tool_id == tool_id
                ).first()

                if not association:
                    debug_id(f"No association found between task {task_id} and tool {tool_id}", rid)
                    return False

                # Delete the association
                session.delete(association)

                # Commit if we created the session
                if not session_provided:
                    session.commit()
                    debug_id(f"Removed association between task {task_id} and tool {tool_id}", rid)

                return True

        except Exception as e:
            error_id(f"Error in TaskToolAssociation.dissociate_task_tool: {str(e)}", rid, exc_info=True)
            if not session_provided:
                session.rollback()
                debug_id(f"Rolled back transaction in TaskToolAssociation.dissociate_task_tool", rid)
            return False
        finally:
            # Close the session if we created it
            if not session_provided:
                session.close()
                debug_id(f"Closed database session for TaskToolAssociation.dissociate_task_tool", rid)

    @classmethod
    @with_request_id
    def get_tools_by_task(cls,
                          task_id: Optional[int] = None,
                          task_name: Optional[str] = None,
                          task_description: Optional[str] = None,
                          tool_id: Optional[int] = None,
                          tool_name: Optional[str] = None,
                          tool_type: Optional[str] = None,
                          tool_material: Optional[str] = None,
                          tool_size: Optional[str] = None,
                          exact_match: bool = False,
                          limit: int = 100,
                          request_id: Optional[str] = None,
                          session: Optional[Session] = None) -> List['Tool']:
        """
        Get tools associated with tasks based on flexible search criteria.

        Args:
            task_id: Optional task ID to filter by
            task_name: Optional task name to filter by
            task_description: Optional task description to filter by
            tool_id: Optional tool ID to filter by
            tool_name: Optional tool name to filter by
            tool_type: Optional tool type to filter by
            tool_material: Optional tool material to filter by
            tool_size: Optional tool size to filter by
            exact_match: If True, performs exact matching instead of partial matching for string fields
            limit: Maximum number of results to return (default 100)
            request_id: Optional request ID for tracking this operation in logs
            session: Optional SQLAlchemy session. If None, a new session will be created

        Returns:
            List of Tool objects matching the search criteria
        """
        # Get or use the provided request_id
        rid = request_id or get_request_id()

        # Get a database session if one wasn't provided
        db_config = DatabaseConfig()
        session_provided = session is not None
        if not session_provided:
            session = db_config.get_main_session()
            debug_id(f"Created new database session for TaskToolAssociation.get_tools_by_task", rid)

        # Log the search operation with request ID
        search_params = {
            'task_id': task_id,
            'task_name': task_name,
            'task_description': task_description,
            'tool_id': tool_id,
            'tool_name': tool_name,
            'tool_type': tool_type,
            'tool_material': tool_material,
            'tool_size': tool_size,
            'exact_match': exact_match,
            'limit': limit
        }
        # Filter out None values for cleaner logging
        logged_params = {k: v for k, v in search_params.items() if v is not None}
        debug_id(f"Starting TaskToolAssociation.get_tools_by_task with parameters: {logged_params}", rid)

        try:
            # Use the timed operation context manager with request ID
            with log_timed_operation("TaskToolAssociation.get_tools_by_task", rid):


                # Start with a query that joins Tool and TaskToolAssociation
                query = session.query(Tool).join(cls, Tool.id == cls.tool_id).join(Task, Task.id == cls.task_id)

                # Apply task filters
                if task_id is not None:
                    query = query.filter(Task.id == task_id)

                if task_name is not None:
                    if exact_match:
                        query = query.filter(Task.name == task_name)
                    else:
                        query = query.filter(Task.name.ilike(f"%{task_name}%"))

                if task_description is not None:
                    if exact_match:
                        query = query.filter(Task.description == task_description)
                    else:
                        query = query.filter(Task.description.ilike(f"%{task_description}%"))

                # Apply tool filters
                if tool_id is not None:
                    query = query.filter(Tool.id == tool_id)

                if tool_name is not None:
                    if exact_match:
                        query = query.filter(Tool.name == tool_name)
                    else:
                        query = query.filter(Tool.name.ilike(f"%{tool_name}%"))

                if tool_type is not None:
                    if exact_match:
                        query = query.filter(Tool.type == tool_type)
                    else:
                        query = query.filter(Tool.type.ilike(f"%{tool_type}%"))

                if tool_material is not None:
                    if exact_match:
                        query = query.filter(Tool.material == tool_material)
                    else:
                        query = query.filter(Tool.material.ilike(f"%{tool_material}%"))

                if tool_size is not None:
                    if exact_match:
                        query = query.filter(Tool.size == tool_size)
                    else:
                        query = query.filter(Tool.size.ilike(f"%{tool_size}%"))

                # Make results distinct to avoid duplicates
                query = query.distinct()

                # Apply limit
                query = query.limit(limit)

                # Execute query and log results
                results = query.all()
                debug_id(f"TaskToolAssociation.get_tools_by_task completed, found {len(results)} tools", rid)
                return results

        except Exception as e:
            error_id(f"Error in TaskToolAssociation.get_tools_by_task: {str(e)}", rid, exc_info=True)
            return []
        finally:
            # Close the session if we created it
            if not session_provided:
                session.close()
                debug_id(f"Closed database session for TaskToolAssociation.get_tools_by_task", rid)

    @classmethod
    @with_request_id
    def get_tasks_by_tool(cls,
                          tool_id: Optional[int] = None,
                          tool_name: Optional[str] = None,
                          tool_type: Optional[str] = None,
                          tool_material: Optional[str] = None,
                          tool_size: Optional[str] = None,
                          task_id: Optional[int] = None,
                          task_name: Optional[str] = None,
                          task_description: Optional[str] = None,
                          exact_match: bool = False,
                          limit: int = 100,
                          request_id: Optional[str] = None,
                          session: Optional[Session] = None) -> List['Task']:
        """
        Get tasks associated with tools based on flexible search criteria.

        Args:
            tool_id: Optional tool ID to filter by
            tool_name: Optional tool name to filter by
            tool_type: Optional tool type to filter by
            tool_material: Optional tool material to filter by
            tool_size: Optional tool size to filter by
            task_id: Optional task ID to filter by
            task_name: Optional task name to filter by
            task_description: Optional task description to filter by
            exact_match: If True, performs exact matching instead of partial matching for string fields
            limit: Maximum number of results to return (default 100)
            request_id: Optional request ID for tracking this operation in logs
            session: Optional SQLAlchemy session. If None, a new session will be created

        Returns:
            List of Task objects matching the search criteria
        """
        # Get or use the provided request_id
        rid = request_id or get_request_id()

        # Get a database session if one wasn't provided
        db_config = DatabaseConfig()
        session_provided = session is not None
        if not session_provided:
            session = db_config.get_main_session()
            debug_id(f"Created new database session for TaskToolAssociation.get_tasks_by_tool", rid)

        # Log the search operation with request ID
        search_params = {
            'tool_id': tool_id,
            'tool_name': tool_name,
            'tool_type': tool_type,
            'tool_material': tool_material,
            'tool_size': tool_size,
            'task_id': task_id,
            'task_name': task_name,
            'task_description': task_description,
            'exact_match': exact_match,
            'limit': limit
        }
        # Filter out None values for cleaner logging
        logged_params = {k: v for k, v in search_params.items() if v is not None}
        debug_id(f"Starting TaskToolAssociation.get_tasks_by_tool with parameters: {logged_params}", rid)

        try:
            # Use the timed operation context manager with request ID
            with log_timed_operation("TaskToolAssociation.get_tasks_by_tool", rid):


                # Start with a query that joins Task and TaskToolAssociation
                query = session.query(Task).join(cls, Task.id == cls.task_id).join(Tool, Tool.id == cls.tool_id)

                # Apply tool filters
                if tool_id is not None:
                    query = query.filter(Tool.id == tool_id)

                if tool_name is not None:
                    if exact_match:
                        query = query.filter(Tool.name == tool_name)
                    else:
                        query = query.filter(Tool.name.ilike(f"%{tool_name}%"))

                if tool_type is not None:
                    if exact_match:
                        query = query.filter(Tool.type == tool_type)
                    else:
                        query = query.filter(Tool.type.ilike(f"%{tool_type}%"))

                if tool_material is not None:
                    if exact_match:
                        query = query.filter(Tool.material == tool_material)
                    else:
                        query = query.filter(Tool.material.ilike(f"%{tool_material}%"))

                if tool_size is not None:
                    if exact_match:
                        query = query.filter(Tool.size == tool_size)
                    else:
                        query = query.filter(Tool.size.ilike(f"%{tool_size}%"))

                # Apply task filters
                if task_id is not None:
                    query = query.filter(Task.id == task_id)

                if task_name is not None:
                    if exact_match:
                        query = query.filter(Task.name == task_name)
                    else:
                        query = query.filter(Task.name.ilike(f"%{task_name}%"))

                if task_description is not None:
                    if exact_match:
                        query = query.filter(Task.description == task_description)
                    else:
                        query = query.filter(Task.description.ilike(f"%{task_description}%"))

                # Make results distinct to avoid duplicates
                query = query.distinct()

                # Apply limit
                query = query.limit(limit)

                # Execute query and log results
                results = query.all()
                debug_id(f"TaskToolAssociation.get_tasks_by_tool completed, found {len(results)} tasks", rid)
                return results

        except Exception as e:
            error_id(f"Error in TaskToolAssociation.get_tasks_by_tool: {str(e)}", rid, exc_info=True)
            return []
        finally:
            # Close the session if we created it
            if not session_provided:
                session.close()
                debug_id(f"Closed database session for TaskToolAssociation.get_tasks_by_tool", rid)

class DrawingProblemAssociation(Base):
    __tablename__ = 'drawing_problem'
    id = Column(Integer, primary_key=True)
    drawing_id = Column(Integer, ForeignKey('drawing.id'))
    problem_id = Column(Integer, ForeignKey('problem.id'))
    
    drawing = relationship("Drawing", back_populates="drawing_problem")
    problem = relationship("Problem", back_populates="drawing_problem")

class BillOfMaterial(Base):
    __tablename__ = 'bill_of_material'
    id = Column(Integer, primary_key=True)
    part_position_image_id = Column(Integer, ForeignKey('part_position_image.id'))  # Corrected line
    """part_id = Column(Integer, ForeignKey('part.id'))
    position_id = Column(Integer, ForeignKey('position.id'))"""
    quantity = Column(Float, nullable=False)  # Corrected to Float
    comment = Column(String)

    part_position_image = relationship("PartsPositionImageAssociation", back_populates="bill_of_material")
    """part = relationship("Part", back_populates="bill_of_material")
    position = relationship("Position", back_populates="bill_of_material")
    image = relationship("Image", back_populates="bill_of_material")"""

class ProblemPositionAssociation(Base):
    __tablename__ = 'problem_position'
    id = Column(Integer, primary_key=True)
    problem_id = Column(Integer, ForeignKey('problem.id'))
    position_id = Column(Integer, ForeignKey('position.id'))

    problem = relationship("Problem", back_populates="problem_position")
    position = relationship("Position", back_populates="problem_position")

    @classmethod
    @with_request_id
    def add_to_db(cls, session=None, problem_id=None, position_id=None):
        """
        Get-or-create a ProblemPositionAssociation with the specified problem_id and position_id.
        If `session` is None, uses DatabaseConfig().get_main_session().
        Returns the ProblemPositionAssociation instance (new or existing).
        """
        # 1) ensure we have a session
        if session is None:
            session = DatabaseConfig().get_main_session()

        # 2) log input parameters
        debug_id(
            "add_to_db called with problem_id=%s, position_id=%s",
            problem_id, position_id
        )

        # Check for required parameters
        if problem_id is None or position_id is None:
            error_id("Both problem_id and position_id must be provided")
            raise ValueError("Both problem_id and position_id must be provided")

        # 3) build filter dict
        filters = {
            "problem_id": problem_id,
            "position_id": position_id,
        }

        try:
            # 4) try to find an existing row
            existing = session.query(cls).filter_by(**filters).first()
            if existing:
                info_id("Found existing ProblemPositionAssociation id=%s", existing.id)
                return existing

            # 5) not found → create new
            association = cls(**filters)
            session.add(association)
            session.commit()
            info_id("Created new ProblemPositionAssociation id=%s", association.id)
            return association

        except SQLAlchemyError as e:
            session.rollback()
            error_id("Failed to add_or_get ProblemPositionAssociation: %s", e, exc_info=True)
            raise

    @classmethod
    @with_request_id
    def get_positions_for_problem(cls, session, problem_id):
        """
        Get all positions associated with a specific problem.

        Args:
            session: SQLAlchemy session
            problem_id: ID of the problem

        Returns:
            List of Position objects associated with the problem
        """
        if session is None:
            session = DatabaseConfig().get_main_session()

        if problem_id is None:
            error_id("problem_id must be provided")
            return []

        try:
            # Query for all associations with this problem_id
            associations = session.query(cls).filter_by(problem_id=problem_id).all()

            # Extract the positions
            positions = [assoc.position for assoc in associations if assoc.position]

            info_id(f"Retrieved {len(positions)} positions for problem_id={problem_id}")
            return positions

        except SQLAlchemyError as e:
            error_id(f"Error retrieving positions for problem_id={problem_id}: {str(e)}", exc_info=True)
            return []

    @classmethod
    @with_request_id
    def get_problems_for_position(cls, session, position_id):
        """
        Get all problems associated with a specific position.

        Args:
            session: SQLAlchemy session
            position_id: ID of the position

        Returns:
            List of Problem objects associated with the position
        """
        if session is None:
            session = DatabaseConfig().get_main_session()

        if position_id is None:
            error_id("position_id must be provided")
            return []

        try:
            # Query for all associations with this position_id
            associations = session.query(cls).filter_by(position_id=position_id).all()

            # Extract the problems
            problems = [assoc.problem for assoc in associations if assoc.problem]

            info_id(f"Retrieved {len(problems)} problems for position_id={position_id}")
            return problems

        except SQLAlchemyError as e:
            error_id(f"Error retrieving problems for position_id={position_id}: {str(e)}", exc_info=True)
            return []

    @classmethod
    @with_request_id
    def delete_association(cls, session, problem_id=None, position_id=None, association_id=None):
        """
        Delete a problem-position association.
        Can delete by providing either the association_id or both problem_id and position_id.

        Args:
            session: SQLAlchemy session
            problem_id: ID of the problem (optional if association_id is provided)
            position_id: ID of the position (optional if association_id is provided)
            association_id: ID of the association (optional if both problem_id and position_id are provided)

        Returns:
            Boolean indicating success
        """
        if session is None:
            session = DatabaseConfig().get_main_session()

        try:
            if association_id:
                # Delete by association ID
                association = session.query(cls).filter_by(id=association_id).first()
                if association:
                    session.delete(association)
                    session.commit()
                    info_id(f"Deleted ProblemPositionAssociation id={association_id}")
                    return True
                else:
                    warning_id(f"No ProblemPositionAssociation found with id={association_id}")
                    return False
            elif problem_id and position_id:
                # Delete by problem_id and position_id
                association = session.query(cls).filter_by(
                    problem_id=problem_id, position_id=position_id).first()
                if association:
                    session.delete(association)
                    session.commit()
                    info_id(
                        f"Deleted ProblemPositionAssociation with problem_id={problem_id}, position_id={position_id}")
                    return True
                else:
                    warning_id(
                        f"No ProblemPositionAssociation found with problem_id={problem_id}, position_id={position_id}")
                    return False
            else:
                error_id("Either association_id or both problem_id and position_id must be provided")
                return False

        except SQLAlchemyError as e:
            session.rollback()
            error_id(f"Error deleting ProblemPositionAssociation: {str(e)}", exc_info=True)
            return False

    @classmethod
    @with_request_id
    def get_positions_for_problem_by_hierarchy(cls, session, problem_id, level_filters=None):
        """
        Get positions associated with a problem filtered by hierarchy levels.

        Args:
            session: SQLAlchemy session
            problem_id: ID of the problem
            level_filters: Dictionary with level names as keys and IDs as values
                           e.g., {'area_id': 1, 'equipment_group_id': 2}

        Returns:
            List of Position objects matching the criteria
        """
        if session is None:
            session = DatabaseConfig().get_main_session()

        if problem_id is None:
            error_id("problem_id must be provided")
            return []

        try:
            # Start with base query for the problem
            query = session.query(Position).join(
                cls, Position.id == cls.position_id
            ).filter(cls.problem_id == problem_id)

            # Apply hierarchy filters if provided
            if level_filters and isinstance(level_filters, dict):
                for field, value in level_filters.items():
                    if hasattr(Position, field) and value is not None:
                        query = query.filter(getattr(Position, field) == value)

            positions = query.all()
            info_id(f"Retrieved {len(positions)} positions for problem_id={problem_id} with hierarchy filters")
            return positions

        except SQLAlchemyError as e:
            error_id(f"Error in get_positions_for_problem_by_hierarchy: {str(e)}", exc_info=True)
            return []

class CompleteDocumentProblemAssociation(Base):
    __tablename__ = 'complete_document_problem'
    
    id = Column(Integer, primary_key=True)
    complete_document_id = Column(Integer, ForeignKey('complete_document.id'))
    problem_id = Column(Integer, ForeignKey('problem.id'))
    
    complete_document = relationship("CompleteDocument", back_populates="complete_document_problem")
    problem = relationship("Problem", back_populates="complete_document_problem")
    
class CompleteDocumentTaskAssociation(Base):
    __tablename__ = 'complete_document_task'
    
    id = Column(Integer, primary_key=True)
    complete_document_id = Column(Integer, ForeignKey('complete_document.id'))
    task_id = Column(Integer, ForeignKey('task.id'))
    
    complete_document = relationship("CompleteDocument", back_populates="complete_document_task")
    task = relationship("Task", back_populates="complete_document_task")

    @classmethod
    @with_request_id
    def associate_complete_document_task(cls,
                                         complete_document_id: int,
                                         task_id: int,
                                         request_id: Optional[str] = None,
                                         session: Optional[Session] = None) -> Optional[
        'CompleteDocumentTaskAssociation']:
        """
        Associate a complete document with a task.

        Args:
            complete_document_id: ID of the complete document to associate
            task_id: ID of the task to associate
            request_id: Optional request ID for tracking this operation in logs
            session: Optional SQLAlchemy session. If None, a new session will be created

        Returns:
            The created CompleteDocumentTaskAssociation object if successful, None otherwise
        """
        # Get or use the provided request_id
        rid = request_id or get_request_id()

        # Get a database session if one wasn't provided
        db_config = DatabaseConfig()
        session_provided = session is not None
        if not session_provided:
            session = db_config.get_main_session()
            debug_id(
                f"Created new database session for CompleteDocumentTaskAssociation.associate_complete_document_task",
                rid)

        # Log the operation with request ID
        debug_id(
            f"Starting CompleteDocumentTaskAssociation.associate_complete_document_task with parameters: complete_document_id={complete_document_id}, task_id={task_id}",
            rid)

        try:
            # Use the timed operation context manager with request ID
            with log_timed_operation("CompleteDocumentTaskAssociation.associate_complete_document_task", rid):


                # Check if complete document exists
                complete_document = session.query(CompleteDocument).filter(
                    CompleteDocument.id == complete_document_id).first()
                if not complete_document:
                    error_id(
                        f"Error in CompleteDocumentTaskAssociation.associate_complete_document_task: CompleteDocument with ID {complete_document_id} not found",
                        rid)
                    return None

                # Check if task exists
                task = session.query(Task).filter(Task.id == task_id).first()
                if not task:
                    error_id(
                        f"Error in CompleteDocumentTaskAssociation.associate_complete_document_task: Task with ID {task_id} not found",
                        rid)
                    return None

                # Check if association already exists
                existing = session.query(cls).filter(
                    cls.complete_document_id == complete_document_id,
                    cls.task_id == task_id
                ).first()

                if existing:
                    debug_id(
                        f"Association between complete document {complete_document_id} and task {task_id} already exists",
                        rid)
                    return existing

                # Create new association
                association = cls(complete_document_id=complete_document_id, task_id=task_id)
                session.add(association)

                # Commit if we created the session
                if not session_provided:
                    session.commit()
                    debug_id(
                        f"Committed new association between complete document {complete_document_id} and task {task_id}",
                        rid)

                return association

        except Exception as e:
            error_id(f"Error in CompleteDocumentTaskAssociation.associate_complete_document_task: {str(e)}", rid,
                     exc_info=True)
            if not session_provided:
                session.rollback()
                debug_id(f"Rolled back transaction in CompleteDocumentTaskAssociation.associate_complete_document_task",
                         rid)
            return None
        finally:
            # Close the session if we created it
            if not session_provided:
                session.close()
                debug_id(
                    f"Closed database session for CompleteDocumentTaskAssociation.associate_complete_document_task",
                    rid)

    @classmethod
    @with_request_id
    def dissociate_complete_document_task(cls,
                                          complete_document_id: int,
                                          task_id: int,
                                          request_id: Optional[str] = None,
                                          session: Optional[Session] = None) -> bool:
        """
        Remove an association between a complete document and a task.

        Args:
            complete_document_id: ID of the complete document to dissociate
            task_id: ID of the task to dissociate
            request_id: Optional request ID for tracking this operation in logs
            session: Optional SQLAlchemy session. If None, a new session will be created

        Returns:
            True if the association was removed, False otherwise
        """
        # Get or use the provided request_id
        rid = request_id or get_request_id()

        # Get a database session if one wasn't provided
        db_config = DatabaseConfig()
        session_provided = session is not None
        if not session_provided:
            session = db_config.get_main_session()
            debug_id(
                f"Created new database session for CompleteDocumentTaskAssociation.dissociate_complete_document_task",
                rid)

        # Log the operation with request ID
        debug_id(
            f"Starting CompleteDocumentTaskAssociation.dissociate_complete_document_task with parameters: complete_document_id={complete_document_id}, task_id={task_id}",
            rid)

        try:
            # Use the timed operation context manager with request ID
            with log_timed_operation("CompleteDocumentTaskAssociation.dissociate_complete_document_task", rid):
                # Find the association
                association = session.query(cls).filter(
                    cls.complete_document_id == complete_document_id,
                    cls.task_id == task_id
                ).first()

                if not association:
                    debug_id(
                        f"No association found between complete document {complete_document_id} and task {task_id}",
                        rid)
                    return False

                # Delete the association
                session.delete(association)

                # Commit if we created the session
                if not session_provided:
                    session.commit()
                    debug_id(f"Removed association between complete document {complete_document_id} and task {task_id}",
                             rid)

                return True

        except Exception as e:
            error_id(f"Error in CompleteDocumentTaskAssociation.dissociate_complete_document_task: {str(e)}", rid,
                     exc_info=True)
            if not session_provided:
                session.rollback()
                debug_id(
                    f"Rolled back transaction in CompleteDocumentTaskAssociation.dissociate_complete_document_task",
                    rid)
            return False
        finally:
            # Close the session if we created it
            if not session_provided:
                session.close()
                debug_id(
                    f"Closed database session for CompleteDocumentTaskAssociation.dissociate_complete_document_task",
                    rid)

    @classmethod
    @with_request_id
    def get_tasks_by_complete_document(cls,
                                       complete_document_id: Optional[int] = None,
                                       title: Optional[str] = None,
                                       file_path: Optional[str] = None,
                                       rev: Optional[str] = None,
                                       task_id: Optional[int] = None,
                                       task_name: Optional[str] = None,
                                       task_description: Optional[str] = None,
                                       exact_match: bool = False,
                                       limit: int = 100,
                                       request_id: Optional[str] = None,
                                       session: Optional[Session] = None) -> List['Task']:
        """
        Get tasks associated with complete documents based on flexible search criteria.

        Args:
            complete_document_id: Optional complete document ID to filter by
            title: Optional document title to filter by
            file_path: Optional file path to filter by
            rev: Optional revision to filter by
            task_id: Optional task ID to filter by
            task_name: Optional task name to filter by
            task_description: Optional task description to filter by
            exact_match: If True, performs exact matching instead of partial matching for string fields
            limit: Maximum number of results to return (default 100)
            request_id: Optional request ID for tracking this operation in logs
            session: Optional SQLAlchemy session. If None, a new session will be created

        Returns:
            List of Task objects matching the search criteria
        """
        # Get or use the provided request_id
        rid = request_id or get_request_id()

        # Get a database session if one wasn't provided
        db_config = DatabaseConfig()
        session_provided = session is not None
        if not session_provided:
            session = db_config.get_main_session()
            debug_id(f"Created new database session for CompleteDocumentTaskAssociation.get_tasks_by_complete_document",
                     rid)

        # Log the search operation with request ID
        search_params = {
            'complete_document_id': complete_document_id,
            'title': title,
            'file_path': file_path,
            'rev': rev,
            'task_id': task_id,
            'task_name': task_name,
            'task_description': task_description,
            'exact_match': exact_match,
            'limit': limit
        }
        # Filter out None values for cleaner logging
        logged_params = {k: v for k, v in search_params.items() if v is not None}
        debug_id(
            f"Starting CompleteDocumentTaskAssociation.get_tasks_by_complete_document with parameters: {logged_params}",
            rid)

        try:
            # Use the timed operation context manager with request ID
            with log_timed_operation("CompleteDocumentTaskAssociation.get_tasks_by_complete_document", rid):


                # Start with a query that joins Task and CompleteDocumentTaskAssociation
                query = session.query(Task).join(cls, Task.id == cls.task_id).join(CompleteDocument,
                                                                                   CompleteDocument.id == cls.complete_document_id)

                # Apply complete document filters
                if complete_document_id is not None:
                    query = query.filter(CompleteDocument.id == complete_document_id)

                if title is not None:
                    if exact_match:
                        query = query.filter(CompleteDocument.title == title)
                    else:
                        query = query.filter(CompleteDocument.title.ilike(f"%{title}%"))

                if file_path is not None:
                    if exact_match:
                        query = query.filter(CompleteDocument.file_path == file_path)
                    else:
                        query = query.filter(CompleteDocument.file_path.ilike(f"%{file_path}%"))

                if rev is not None:
                    if exact_match:
                        query = query.filter(CompleteDocument.rev == rev)
                    else:
                        query = query.filter(CompleteDocument.rev.ilike(f"%{rev}%"))

                # Apply task filters
                if task_id is not None:
                    query = query.filter(Task.id == task_id)

                if task_name is not None:
                    if exact_match:
                        query = query.filter(Task.name == task_name)
                    else:
                        query = query.filter(Task.name.ilike(f"%{task_name}%"))

                if task_description is not None:
                    if exact_match:
                        query = query.filter(Task.description == task_description)
                    else:
                        query = query.filter(Task.description.ilike(f"%{task_description}%"))

                # Make results distinct to avoid duplicates
                query = query.distinct()

                # Apply limit
                query = query.limit(limit)

                # Execute query and log results
                results = query.all()
                debug_id(
                    f"CompleteDocumentTaskAssociation.get_tasks_by_complete_document completed, found {len(results)} tasks",
                    rid)
                return results

        except Exception as e:
            error_id(f"Error in CompleteDocumentTaskAssociation.get_tasks_by_complete_document: {str(e)}", rid,
                     exc_info=True)
            return []
        finally:
            # Close the session if we created it
            if not session_provided:
                session.close()
                debug_id(f"Closed database session for CompleteDocumentTaskAssociation.get_tasks_by_complete_document",
                         rid)

    @classmethod
    @with_request_id
    def get_complete_documents_by_task(cls,
                                       task_id: Optional[int] = None,
                                       task_name: Optional[str] = None,
                                       task_description: Optional[str] = None,
                                       complete_document_id: Optional[int] = None,
                                       title: Optional[str] = None,
                                       file_path: Optional[str] = None,
                                       rev: Optional[str] = None,
                                       exact_match: bool = False,
                                       limit: int = 100,
                                       request_id: Optional[str] = None,
                                       session: Optional[Session] = None) -> List['CompleteDocument']:
        """
        Get complete documents associated with tasks based on flexible search criteria.

        Args:
            task_id: Optional task ID to filter by
            task_name: Optional task name to filter by
            task_description: Optional task description to filter by
            complete_document_id: Optional complete document ID to filter by
            title: Optional document title to filter by
            file_path: Optional file path to filter by
            rev: Optional revision to filter by
            exact_match: If True, performs exact matching instead of partial matching for string fields
            limit: Maximum number of results to return (default 100)
            request_id: Optional request ID for tracking this operation in logs
            session: Optional SQLAlchemy session. If None, a new session will be created

        Returns:
            List of CompleteDocument objects matching the search criteria
        """
        # Get or use the provided request_id
        rid = request_id or get_request_id()

        # Get a database session if one wasn't provided
        db_config = DatabaseConfig()
        session_provided = session is not None
        if not session_provided:
            session = db_config.get_main_session()
            debug_id(f"Created new database session for CompleteDocumentTaskAssociation.get_complete_documents_by_task",
                     rid)

        # Log the search operation with request ID
        search_params = {
            'task_id': task_id,
            'task_name': task_name,
            'task_description': task_description,
            'complete_document_id': complete_document_id,
            'title': title,
            'file_path': file_path,
            'rev': rev,
            'exact_match': exact_match,
            'limit': limit
        }
        # Filter out None values for cleaner logging
        logged_params = {k: v for k, v in search_params.items() if v is not None}
        debug_id(
            f"Starting CompleteDocumentTaskAssociation.get_complete_documents_by_task with parameters: {logged_params}",
            rid)

        try:
            # Use the timed operation context manager with request ID
            with log_timed_operation("CompleteDocumentTaskAssociation.get_complete_documents_by_task", rid):


                # Start with a query that joins CompleteDocument and CompleteDocumentTaskAssociation
                query = session.query(CompleteDocument).join(cls, CompleteDocument.id == cls.complete_document_id).join(
                    Task, Task.id == cls.task_id)

                # Apply task filters
                if task_id is not None:
                    query = query.filter(Task.id == task_id)

                if task_name is not None:
                    if exact_match:
                        query = query.filter(Task.name == task_name)
                    else:
                        query = query.filter(Task.name.ilike(f"%{task_name}%"))

                if task_description is not None:
                    if exact_match:
                        query = query.filter(Task.description == task_description)
                    else:
                        query = query.filter(Task.description.ilike(f"%{task_description}%"))

                # Apply complete document filters
                if complete_document_id is not None:
                    query = query.filter(CompleteDocument.id == complete_document_id)

                if title is not None:
                    if exact_match:
                        query = query.filter(CompleteDocument.title == title)
                    else:
                        query = query.filter(CompleteDocument.title.ilike(f"%{title}%"))

                if file_path is not None:
                    if exact_match:
                        query = query.filter(CompleteDocument.file_path == file_path)
                    else:
                        query = query.filter(CompleteDocument.file_path.ilike(f"%{file_path}%"))

                if rev is not None:
                    if exact_match:
                        query = query.filter(CompleteDocument.rev == rev)
                    else:
                        query = query.filter(CompleteDocument.rev.ilike(f"%{rev}%"))

                # Make results distinct to avoid duplicates
                query = query.distinct()

                # Apply limit
                query = query.limit(limit)

                # Execute query and log results
                results = query.all()
                debug_id(
                    f"CompleteDocumentTaskAssociation.get_complete_documents_by_task completed, found {len(results)} complete documents",
                    rid)
                return results

        except Exception as e:
            error_id(f"Error in CompleteDocumentTaskAssociation.get_complete_documents_by_task: {str(e)}", rid,
                     exc_info=True)
            return []
        finally:
            # Close the session if we created it
            if not session_provided:
                session.close()
                debug_id(f"Closed database session for CompleteDocumentTaskAssociation.get_complete_documents_by_task",
                         rid)

class ImageProblemAssociation(Base):
    __tablename__ = 'image_problem'
    id = Column(Integer, primary_key=True)
    image_id = Column(Integer, ForeignKey('image.id'))
    problem_id = Column(Integer, ForeignKey('problem.id'))
    
    image = relationship("Image", back_populates="image_problem")
    problem = relationship("Problem", back_populates="image_problem")

class PartsPositionImageAssociation(Base):
    __tablename__ = 'part_position_image'
    id = Column(Integer, primary_key=True)
    part_id = Column(Integer, ForeignKey('part.id'))
    position_id = Column(Integer, ForeignKey('position.id'))
    image_id = Column(Integer, ForeignKey('image.id'))

    part = relationship("Part", back_populates="part_position_image")
    position = relationship("Position", back_populates="part_position_image")
    image = relationship("Image", back_populates="parts_position_image")
    bill_of_material = relationship("BillOfMaterial", back_populates="part_position_image")

    @classmethod
    @with_request_id
    def search(cls, session=None, **filters):
        """
        Search the 'part_position_image' table based on the provided filters.

        Args:
            session: SQLAlchemy session (optional).
            filters: A dictionary of filter parameters (e.g., part_id, position_id, image_id).

        Returns:
            List of matching 'PartPositionImageAssociation' objects.
        """
        if session is None:
            session = DatabaseConfig().get_main_session()

        # Get the request ID for logging
        request_id = get_request_id()

        # Log the start of the search operation
        info_id(f"Starting search with filters: {filters}", request_id=request_id)

        # Start with the base query
        query = session.query(cls)

        try:
            # Apply filters dynamically
            if filters:
                for field, value in filters.items():
                    if value is not None:  # Only apply non-None filters
                        query = query.filter(getattr(cls, field) == value)

            # Execute the query and log the result
            results = query.all()

            # Log the number of results found
            info_id(f"Search returned {len(results)} result(s) for filters: {filters}", request_id=request_id)

            return results
        except SQLAlchemyError as e:
            # Log the error
            error_id(f"Error during search operation with filters {filters}: {e}", request_id=request_id, exc_info=True)
            raise

    @classmethod
    @with_request_id
    def get_corresponding_position_ids(cls, session, area_id=None, equipment_group_id=None, model_id=None,
                                       asset_number_id=None, location_id=None):
        """
        Search for corresponding Position IDs based on the provided filters.
        Traverses the hierarchy and retrieves matching Position IDs.

        Args:
            session: SQLAlchemy session
            area_id: ID of the area (optional)
            equipment_group_id: ID of the equipment group (optional)
            model_id: ID of the model (optional)
            asset_number_id: ID of the asset number (optional)
            location_id: ID of the location (optional)

        Returns:
            List of Position IDs that match the criteria
        """
        # Get the request ID for logging
        request_id = get_request_id()

        # Log the start of the operation
        info_id(f"Starting get_corresponding_position_ids with filters: "
                f"area_id={area_id}, equipment_group_id={equipment_group_id}, "
                f"model_id={model_id}, asset_number_id={asset_number_id}, "
                f"location_id={location_id}", request_id=request_id)

        # Start by fetching the root-level positions based on area_id (or first level in hierarchy)
        try:
            positions = cls._get_positions_by_hierarchy(session, area_id=area_id,
                                                        equipment_group_id=equipment_group_id,
                                                        model_id=model_id,
                                                        asset_number_id=asset_number_id,
                                                        location_id=location_id)
            position_ids = [position.id for position in positions]

            # Log the number of Position IDs found
            info_id(f"Found {len(position_ids)} Position IDs for the given filters", request_id=request_id)

            return position_ids
        except SQLAlchemyError as e:
            error_id(f"Error during get_corresponding_position_ids with filters "
                     f"area_id={area_id}, equipment_group_id={equipment_group_id}, "
                     f"model_id={model_id}, asset_number_id={asset_number_id}, "
                     f"location_id={location_id}: {e}", request_id=request_id, exc_info=True)
            raise

    @classmethod
    @with_request_id
    def _get_positions_by_hierarchy(cls, session, area_id=None, equipment_group_id=None, model_id=None,
                                    asset_number_id=None, location_id=None):
        """
        Helper method to fetch positions based on hierarchical filters.

        Args:
            session: SQLAlchemy session
            area_id, equipment_group_id, model_id, asset_number_id, location_id: IDs for filtering

        Returns:
            List of Position objects that match the criteria
        """
        # Get the request ID for logging
        request_id = get_request_id()

        # Building the filter dynamically based on input parameters
        filters = {}
        if area_id:
            filters['area_id'] = area_id
        if equipment_group_id:
            filters['equipment_group_id'] = equipment_group_id
        if model_id:
            filters['model_id'] = model_id
        if asset_number_id:
            filters['asset_number_id'] = asset_number_id
        if location_id:
            filters['location_id'] = location_id

        try:
            # Log the filter being applied
            info_id(f"Applying filters to query: {filters}", request_id=request_id)

            # Query the Position table based on the filters
            query = session.query(Position).filter_by(**filters)

            # Execute and return the results
            positions = query.all()

            # Log the number of results
            info_id(f"Found {len(positions)} positions for the given filters", request_id=request_id)

            return positions
        except SQLAlchemyError as e:
            error_id(f"Error during _get_positions_by_hierarchy with filters {filters}: {e}", request_id=request_id,
                     exc_info=True)
            raise

class ImagePositionAssociation(Base):
    __tablename__ = 'image_position_association'
    id = Column(Integer, primary_key=True)
    image_id = Column(Integer, ForeignKey('image.id'))
    position_id = Column(Integer, ForeignKey('position.id'))
    
    image = relationship("Image", back_populates="image_position_association")
    position = relationship("Position", back_populates="image_position_association")

    @classmethod
    @with_request_id
    def associate_image_position(cls,
                                 image_id: int,
                                 position_id: int,
                                 request_id: Optional[str] = None,
                                 session: Optional[Session] = None) -> Optional['ImagePositionAssociation']:
        """
        Associate an image with a position.

        Args:
            image_id: ID of the image to associate
            position_id: ID of the position to associate
            request_id: Optional request ID for tracking this operation in logs
            session: Optional SQLAlchemy session. If None, a new session will be created

        Returns:
            The created ImagePositionAssociation object if successful, None otherwise
        """
        # Get or use the provided request_id
        rid = request_id or get_request_id()

        # Get a database session if one wasn't provided
        db_config = DatabaseConfig()
        session_provided = session is not None
        if not session_provided:
            session = db_config.get_main_session()
            debug_id(f"Created new database session for ImagePositionAssociation.associate_image_position", rid)

        # Log the operation with request ID
        debug_id(
            f"Starting ImagePositionAssociation.associate_image_position with parameters: image_id={image_id}, position_id={position_id}",
            rid)

        try:
            # Use the timed operation context manager with request ID
            with log_timed_operation("ImagePositionAssociation.associate_image_position", rid):


                # Check if image exists
                image = session.query(Image).filter(Image.id == image_id).first()
                if not image:
                    error_id(
                        f"Error in ImagePositionAssociation.associate_image_position: Image with ID {image_id} not found",
                        rid)
                    return None

                # Check if position exists
                position = session.query(Position).filter(Position.id == position_id).first()
                if not position:
                    error_id(
                        f"Error in ImagePositionAssociation.associate_image_position: Position with ID {position_id} not found",
                        rid)
                    return None

                # Check if association already exists
                existing = session.query(cls).filter(
                    cls.image_id == image_id,
                    cls.position_id == position_id
                ).first()

                if existing:
                    debug_id(f"Association between image {image_id} and position {position_id} already exists", rid)
                    return existing

                # Create new association
                association = cls(image_id=image_id, position_id=position_id)
                session.add(association)

                # Commit if we created the session
                if not session_provided:
                    session.commit()
                    debug_id(f"Committed new association between image {image_id} and position {position_id}", rid)

                return association

        except Exception as e:
            error_id(f"Error in ImagePositionAssociation.associate_image_position: {str(e)}", rid, exc_info=True)
            if not session_provided:
                session.rollback()
                debug_id(f"Rolled back transaction in ImagePositionAssociation.associate_image_position", rid)
            return None
        finally:
            # Close the session if we created it
            if not session_provided:
                session.close()
                debug_id(f"Closed database session for ImagePositionAssociation.associate_image_position", rid)

    @classmethod
    @with_request_id
    def dissociate_image_position(cls,
                                  image_id: int,
                                  position_id: int,
                                  request_id: Optional[str] = None,
                                  session: Optional[Session] = None) -> bool:
        """
        Remove an association between an image and a position.

        Args:
            image_id: ID of the image to dissociate
            position_id: ID of the position to dissociate
            request_id: Optional request ID for tracking this operation in logs
            session: Optional SQLAlchemy session. If None, a new session will be created

        Returns:
            True if the association was removed, False otherwise
        """
        # Get or use the provided request_id
        rid = request_id or get_request_id()

        # Get a database session if one wasn't provided
        db_config = DatabaseConfig()
        session_provided = session is not None
        if not session_provided:
            session = db_config.get_main_session()
            debug_id(f"Created new database session for ImagePositionAssociation.dissociate_image_position", rid)

        # Log the operation with request ID
        debug_id(
            f"Starting ImagePositionAssociation.dissociate_image_position with parameters: image_id={image_id}, position_id={position_id}",
            rid)

        try:
            # Use the timed operation context manager with request ID
            with log_timed_operation("ImagePositionAssociation.dissociate_image_position", rid):
                # Find the association
                association = session.query(cls).filter(
                    cls.image_id == image_id,
                    cls.position_id == position_id
                ).first()

                if not association:
                    debug_id(f"No association found between image {image_id} and position {position_id}", rid)
                    return False

                # Delete the association
                session.delete(association)

                # Commit if we created the session
                if not session_provided:
                    session.commit()
                    debug_id(f"Removed association between image {image_id} and position {position_id}", rid)

                return True

        except Exception as e:
            error_id(f"Error in ImagePositionAssociation.dissociate_image_position: {str(e)}", rid, exc_info=True)
            if not session_provided:
                session.rollback()
                debug_id(f"Rolled back transaction in ImagePositionAssociation.dissociate_image_position", rid)
            return False
        finally:
            # Close the session if we created it
            if not session_provided:
                session.close()
                debug_id(f"Closed database session for ImagePositionAssociation.dissociate_image_position", rid)

    @classmethod
    @with_request_id
    def get_positions_by_image(cls,
                               image_id: Optional[int] = None,
                               title: Optional[str] = None,
                               description: Optional[str] = None,
                               file_path: Optional[str] = None,
                               position_id: Optional[int] = None,
                               area_id: Optional[int] = None,
                               equipment_group_id: Optional[int] = None,
                               model_id: Optional[int] = None,
                               asset_number_id: Optional[int] = None,
                               location_id: Optional[int] = None,
                               subassembly_id: Optional[int] = None,
                               component_assembly_id: Optional[int] = None,
                               assembly_view_id: Optional[int] = None,
                               site_location_id: Optional[int] = None,
                               exact_match: bool = False,
                               limit: int = 100,
                               request_id: Optional[str] = None,
                               session: Optional[Session] = None) -> List['Position']:
        """
        Get positions associated with images based on flexible search criteria.

        Args:
            image_id: Optional image ID to filter by
            title: Optional image title to filter by
            description: Optional image description to filter by
            file_path: Optional file path to filter by
            position_id: Optional position ID to filter by
            area_id: Optional area ID to filter by
            equipment_group_id: Optional equipment group ID to filter by
            model_id: Optional model ID to filter by
            asset_number_id: Optional asset number ID to filter by
            location_id: Optional location ID to filter by
            subassembly_id: Optional subassembly ID to filter by
            component_assembly_id: Optional component assembly ID to filter by
            assembly_view_id: Optional assembly view ID to filter by
            site_location_id: Optional site location ID to filter by
            exact_match: If True, performs exact matching instead of partial matching for string fields
            limit: Maximum number of results to return (default 100)
            request_id: Optional request ID for tracking this operation in logs
            session: Optional SQLAlchemy session. If None, a new session will be created

        Returns:
            List of Position objects matching the search criteria
        """
        # Get or use the provided request_id
        rid = request_id or get_request_id()

        # Get a database session if one wasn't provided
        db_config = DatabaseConfig()
        session_provided = session is not None
        if not session_provided:
            session = db_config.get_main_session()
            debug_id(f"Created new database session for ImagePositionAssociation.get_positions_by_image", rid)

        # Log the search operation with request ID
        search_params = {
            'image_id': image_id,
            'title': title,
            'description': description,
            'file_path': file_path,
            'position_id': position_id,
            'area_id': area_id,
            'equipment_group_id': equipment_group_id,
            'model_id': model_id,
            'asset_number_id': asset_number_id,
            'location_id': location_id,
            'subassembly_id': subassembly_id,
            'component_assembly_id': component_assembly_id,
            'assembly_view_id': assembly_view_id,
            'site_location_id': site_location_id,
            'exact_match': exact_match,
            'limit': limit
        }
        # Filter out None values for cleaner logging
        logged_params = {k: v for k, v in search_params.items() if v is not None}
        debug_id(f"Starting ImagePositionAssociation.get_positions_by_image with parameters: {logged_params}", rid)

        try:
            # Use the timed operation context manager with request ID
            with log_timed_operation("ImagePositionAssociation.get_positions_by_image", rid):


                # Start with a query that joins Position and ImagePositionAssociation
                query = session.query(Position).join(cls, Position.id == cls.position_id).join(Image,
                                                                                               Image.id == cls.image_id)

                # Apply image filters
                if image_id is not None:
                    query = query.filter(Image.id == image_id)

                if title is not None:
                    if exact_match:
                        query = query.filter(Image.title == title)
                    else:
                        query = query.filter(Image.title.ilike(f"%{title}%"))

                if description is not None:
                    if exact_match:
                        query = query.filter(Image.description == description)
                    else:
                        query = query.filter(Image.description.ilike(f"%{description}%"))

                if file_path is not None:
                    if exact_match:
                        query = query.filter(Image.file_path == file_path)
                    else:
                        query = query.filter(Image.file_path.ilike(f"%{file_path}%"))

                # Apply position filters
                if position_id is not None:
                    query = query.filter(Position.id == position_id)

                if area_id is not None:
                    query = query.filter(Position.area_id == area_id)

                if equipment_group_id is not None:
                    query = query.filter(Position.equipment_group_id == equipment_group_id)

                if model_id is not None:
                    query = query.filter(Position.model_id == model_id)

                if asset_number_id is not None:
                    query = query.filter(Position.asset_number_id == asset_number_id)

                if location_id is not None:
                    query = query.filter(Position.location_id == location_id)

                if subassembly_id is not None:
                    query = query.filter(Position.subassembly_id == subassembly_id)

                if component_assembly_id is not None:
                    query = query.filter(Position.component_assembly_id == component_assembly_id)

                if assembly_view_id is not None:
                    query = query.filter(Position.assembly_view_id == assembly_view_id)

                if site_location_id is not None:
                    query = query.filter(Position.site_location_id == site_location_id)

                # Make results distinct to avoid duplicates
                query = query.distinct()

                # Apply limit
                query = query.limit(limit)

                # Execute query and log results
                results = query.all()
                debug_id(f"ImagePositionAssociation.get_positions_by_image completed, found {len(results)} positions",
                         rid)
                return results

        except Exception as e:
            error_id(f"Error in ImagePositionAssociation.get_positions_by_image: {str(e)}", rid, exc_info=True)
            return []
        finally:
            # Close the session if we created it
            if not session_provided:
                session.close()
                debug_id(f"Closed database session for ImagePositionAssociation.get_positions_by_image", rid)

    @classmethod
    @with_request_id
    def get_images_by_position(cls,
                               position_id: Optional[int] = None,
                               area_id: Optional[int] = None,
                               equipment_group_id: Optional[int] = None,
                               model_id: Optional[int] = None,
                               asset_number_id: Optional[int] = None,
                               location_id: Optional[int] = None,
                               subassembly_id: Optional[int] = None,
                               component_assembly_id: Optional[int] = None,
                               assembly_view_id: Optional[int] = None,
                               site_location_id: Optional[int] = None,
                               image_id: Optional[int] = None,
                               title: Optional[str] = None,
                               description: Optional[str] = None,
                               file_path: Optional[str] = None,
                               exact_match: bool = False,
                               limit: int = 100,
                               request_id: Optional[str] = None,
                               session: Optional[Session] = None) -> List['Image']:
        """
        Get images associated with positions based on flexible search criteria.

        Args:
            position_id: Optional position ID to filter by
            area_id: Optional area ID to filter by
            equipment_group_id: Optional equipment group ID to filter by
            model_id: Optional model ID to filter by
            asset_number_id: Optional asset number ID to filter by
            location_id: Optional location ID to filter by
            subassembly_id: Optional subassembly ID to filter by
            component_assembly_id: Optional component assembly ID to filter by
            assembly_view_id: Optional assembly view ID to filter by
            site_location_id: Optional site location ID to filter by
            image_id: Optional image ID to filter by
            title: Optional image title to filter by
            description: Optional image description to filter by
            file_path: Optional file path to filter by
            exact_match: If True, performs exact matching instead of partial matching for string fields
            limit: Maximum number of results to return (default 100)
            request_id: Optional request ID for tracking this operation in logs
            session: Optional SQLAlchemy session. If None, a new session will be created

        Returns:
            List of Image objects matching the search criteria
        """
        # Get or use the provided request_id
        rid = request_id or get_request_id()

        # Get a database session if one wasn't provided
        db_config = DatabaseConfig()
        session_provided = session is not None
        if not session_provided:
            session = db_config.get_main_session()
            debug_id(f"Created new database session for ImagePositionAssociation.get_images_by_position", rid)

        # Log the search operation with request ID
        search_params = {
            'position_id': position_id,
            'area_id': area_id,
            'equipment_group_id': equipment_group_id,
            'model_id': model_id,
            'asset_number_id': asset_number_id,
            'location_id': location_id,
            'subassembly_id': subassembly_id,
            'component_assembly_id': component_assembly_id,
            'assembly_view_id': assembly_view_id,
            'site_location_id': site_location_id,
            'image_id': image_id,
            'title': title,
            'description': description,
            'file_path': file_path,
            'exact_match': exact_match,
            'limit': limit
        }
        # Filter out None values for cleaner logging
        logged_params = {k: v for k, v in search_params.items() if v is not None}
        debug_id(f"Starting ImagePositionAssociation.get_images_by_position with parameters: {logged_params}", rid)

        try:
            # Use the timed operation context manager with request ID
            with log_timed_operation("ImagePositionAssociation.get_images_by_position", rid):


                # Start with a query that joins Image and ImagePositionAssociation
                query = session.query(Image).join(cls, Image.id == cls.image_id).join(Position,
                                                                                      Position.id == cls.position_id)

                # Apply position filters
                if position_id is not None:
                    query = query.filter(Position.id == position_id)

                if area_id is not None:
                    query = query.filter(Position.area_id == area_id)

                if equipment_group_id is not None:
                    query = query.filter(Position.equipment_group_id == equipment_group_id)

                if model_id is not None:
                    query = query.filter(Position.model_id == model_id)

                if asset_number_id is not None:
                    query = query.filter(Position.asset_number_id == asset_number_id)

                if location_id is not None:
                    query = query.filter(Position.location_id == location_id)

                if subassembly_id is not None:
                    query = query.filter(Position.subassembly_id == subassembly_id)

                if component_assembly_id is not None:
                    query = query.filter(Position.component_assembly_id == component_assembly_id)

                if assembly_view_id is not None:
                    query = query.filter(Position.assembly_view_id == assembly_view_id)

                if site_location_id is not None:
                    query = query.filter(Position.site_location_id == site_location_id)

                # Apply image filters
                if image_id is not None:
                    query = query.filter(Image.id == image_id)

                if title is not None:
                    if exact_match:
                        query = query.filter(Image.title == title)
                    else:
                        query = query.filter(Image.title.ilike(f"%{title}%"))

                if description is not None:
                    if exact_match:
                        query = query.filter(Image.description == description)
                    else:
                        query = query.filter(Image.description.ilike(f"%{description}%"))

                if file_path is not None:
                    if exact_match:
                        query = query.filter(Image.file_path == file_path)
                    else:
                        query = query.filter(Image.file_path.ilike(f"%{file_path}%"))

                # Make results distinct to avoid duplicates
                query = query.distinct()

                # Apply limit
                query = query.limit(limit)

                # Execute query and log results
                results = query.all()
                debug_id(f"ImagePositionAssociation.get_images_by_position completed, found {len(results)} images", rid)
                return results

        except Exception as e:
            error_id(f"Error in ImagePositionAssociation.get_images_by_position: {str(e)}", rid, exc_info=True)
            return []
        finally:
            # Close the session if we created it
            if not session_provided:
                session.close()
                debug_id(f"Closed database session for ImagePositionAssociation.get_images_by_position", rid)

class DrawingPositionAssociation(Base):
    __tablename__ = 'drawing_position'
    id = Column(Integer, primary_key=True)
    drawing_id = Column(Integer, ForeignKey('drawing.id'))
    position_id = Column(Integer, ForeignKey('position.id'))
    
    drawing = relationship("Drawing", back_populates="drawing_position")
    position = relationship("Position", back_populates="drawing_position")

    @classmethod
    @with_request_id
    def associate_drawing_position(cls,
                                   drawing_id: int,
                                   position_id: int,
                                   request_id: Optional[str] = None,
                                   session: Optional[Session] = None) -> Optional['DrawingPositionAssociation']:
        """
        Associate a drawing with a position.

        Args:
            drawing_id: ID of the drawing to associate
            position_id: ID of the position to associate
            request_id: Optional request ID for tracking this operation in logs
            session: Optional SQLAlchemy session. If None, a new session will be created

        Returns:
            The created DrawingPositionAssociation object if successful, None otherwise
        """
        # Get or use the provided request_id
        rid = request_id or get_request_id()

        # Get a database session if one wasn't provided
        db_config = DatabaseConfig()
        session_provided = session is not None
        if not session_provided:
            session = db_config.get_main_session()
            debug_id(f"Created new database session for DrawingPositionAssociation.associate_drawing_position", rid)

        # Log the operation with request ID
        debug_id(
            f"Starting DrawingPositionAssociation.associate_drawing_position with parameters: drawing_id={drawing_id}, position_id={position_id}",
            rid)

        try:
            # Use the timed operation context manager with request ID
            with log_timed_operation("DrawingPositionAssociation.associate_drawing_position", rid):


                # Check if drawing exists
                drawing = session.query(Drawing).filter(Drawing.id == drawing_id).first()
                if not drawing:
                    error_id(
                        f"Error in DrawingPositionAssociation.associate_drawing_position: Drawing with ID {drawing_id} not found",
                        rid)
                    return None

                # Check if position exists
                position = session.query(Position).filter(Position.id == position_id).first()
                if not position:
                    error_id(
                        f"Error in DrawingPositionAssociation.associate_drawing_position: Position with ID {position_id} not found",
                        rid)
                    return None

                # Check if association already exists
                existing = session.query(cls).filter(
                    cls.drawing_id == drawing_id,
                    cls.position_id == position_id
                ).first()

                if existing:
                    debug_id(f"Association between drawing {drawing_id} and position {position_id} already exists", rid)
                    return existing

                # Create new association
                association = cls(drawing_id=drawing_id, position_id=position_id)
                session.add(association)

                # Commit if we created the session
                if not session_provided:
                    session.commit()
                    debug_id(f"Committed new association between drawing {drawing_id} and position {position_id}", rid)

                return association

        except Exception as e:
            error_id(f"Error in DrawingPositionAssociation.associate_drawing_position: {str(e)}", rid, exc_info=True)
            if not session_provided:
                session.rollback()
                debug_id(f"Rolled back transaction in DrawingPositionAssociation.associate_drawing_position", rid)
            return None
        finally:
            # Close the session if we created it
            if not session_provided:
                session.close()
                debug_id(f"Closed database session for DrawingPositionAssociation.associate_drawing_position", rid)

    @classmethod
    @with_request_id
    def dissociate_drawing_position(cls,
                                    drawing_id: int,
                                    position_id: int,
                                    request_id: Optional[str] = None,
                                    session: Optional[Session] = None) -> bool:
        """
        Remove an association between a drawing and a position.

        Args:
            drawing_id: ID of the drawing to dissociate
            position_id: ID of the position to dissociate
            request_id: Optional request ID for tracking this operation in logs
            session: Optional SQLAlchemy session. If None, a new session will be created

        Returns:
            True if the association was removed, False otherwise
        """
        # Get or use the provided request_id
        rid = request_id or get_request_id()

        # Get a database session if one wasn't provided
        db_config = DatabaseConfig()
        session_provided = session is not None
        if not session_provided:
            session = db_config.get_main_session()
            debug_id(f"Created new database session for DrawingPositionAssociation.dissociate_drawing_position", rid)

        # Log the operation with request ID
        debug_id(
            f"Starting DrawingPositionAssociation.dissociate_drawing_position with parameters: drawing_id={drawing_id}, position_id={position_id}",
            rid)

        try:
            # Use the timed operation context manager with request ID
            with log_timed_operation("DrawingPositionAssociation.dissociate_drawing_position", rid):
                # Find the association
                association = session.query(cls).filter(
                    cls.drawing_id == drawing_id,
                    cls.position_id == position_id
                ).first()

                if not association:
                    debug_id(f"No association found between drawing {drawing_id} and position {position_id}", rid)
                    return False

                # Delete the association
                session.delete(association)

                # Commit if we created the session
                if not session_provided:
                    session.commit()
                    debug_id(f"Removed association between drawing {drawing_id} and position {position_id}", rid)

                return True

        except Exception as e:
            error_id(f"Error in DrawingPositionAssociation.dissociate_drawing_position: {str(e)}", rid, exc_info=True)
            if not session_provided:
                session.rollback()
                debug_id(f"Rolled back transaction in DrawingPositionAssociation.dissociate_drawing_position", rid)
            return False
        finally:
            # Close the session if we created it
            if not session_provided:
                session.close()
                debug_id(f"Closed database session for DrawingPositionAssociation.dissociate_drawing_position", rid)

    @classmethod
    @with_request_id
    def get_positions_by_drawing(cls,
                                 drawing_id: Optional[int] = None,
                                 drw_equipment_name: Optional[str] = None,
                                 drw_number: Optional[str] = None,
                                 drw_name: Optional[str] = None,
                                 drw_revision: Optional[str] = None,
                                 drw_spare_part_number: Optional[str] = None,
                                 file_path: Optional[str] = None,
                                 position_id: Optional[int] = None,
                                 area_id: Optional[int] = None,
                                 equipment_group_id: Optional[int] = None,
                                 model_id: Optional[int] = None,
                                 asset_number_id: Optional[int] = None,
                                 location_id: Optional[int] = None,
                                 subassembly_id: Optional[int] = None,
                                 component_assembly_id: Optional[int] = None,
                                 assembly_view_id: Optional[int] = None,
                                 site_location_id: Optional[int] = None,
                                 exact_match: bool = False,
                                 limit: int = 100,
                                 request_id: Optional[str] = None,
                                 session: Optional[Session] = None) -> List['Position']:
        """
        Get positions associated with drawings based on flexible search criteria.

        Args:
            drawing_id: Optional drawing ID to filter by
            drw_equipment_name: Optional equipment name to filter by
            drw_number: Optional drawing number to filter by
            drw_name: Optional drawing name to filter by
            drw_revision: Optional revision to filter by
            drw_spare_part_number: Optional spare part number to filter by
            file_path: Optional file path to filter by
            position_id: Optional position ID to filter by
            area_id: Optional area ID to filter by
            equipment_group_id: Optional equipment group ID to filter by
            model_id: Optional model ID to filter by
            asset_number_id: Optional asset number ID to filter by
            location_id: Optional location ID to filter by
            subassembly_id: Optional subassembly ID to filter by
            component_assembly_id: Optional component assembly ID to filter by
            assembly_view_id: Optional assembly view ID to filter by
            site_location_id: Optional site location ID to filter by
            exact_match: If True, performs exact matching instead of partial matching for string fields
            limit: Maximum number of results to return (default 100)
            request_id: Optional request ID for tracking this operation in logs
            session: Optional SQLAlchemy session. If None, a new session will be created

        Returns:
            List of Position objects matching the search criteria
        """
        # Get or use the provided request_id
        rid = request_id or get_request_id()

        # Get a database session if one wasn't provided
        db_config = DatabaseConfig()
        session_provided = session is not None
        if not session_provided:
            session = db_config.get_main_session()
            debug_id(f"Created new database session for DrawingPositionAssociation.get_positions_by_drawing", rid)

        # Log the search operation with request ID
        search_params = {
            'drawing_id': drawing_id,
            'drw_equipment_name': drw_equipment_name,
            'drw_number': drw_number,
            'drw_name': drw_name,
            'drw_revision': drw_revision,
            'drw_spare_part_number': drw_spare_part_number,
            'file_path': file_path,
            'position_id': position_id,
            'area_id': area_id,
            'equipment_group_id': equipment_group_id,
            'model_id': model_id,
            'asset_number_id': asset_number_id,
            'location_id': location_id,
            'subassembly_id': subassembly_id,
            'component_assembly_id': component_assembly_id,
            'assembly_view_id': assembly_view_id,
            'site_location_id': site_location_id,
            'exact_match': exact_match,
            'limit': limit
        }
        # Filter out None values for cleaner logging
        logged_params = {k: v for k, v in search_params.items() if v is not None}
        debug_id(f"Starting DrawingPositionAssociation.get_positions_by_drawing with parameters: {logged_params}", rid)

        try:
            # Use the timed operation context manager with request ID
            with log_timed_operation("DrawingPositionAssociation.get_positions_by_drawing", rid):


                # Start with a query that joins Position and DrawingPositionAssociation
                query = session.query(Position).join(cls, Position.id == cls.position_id).join(Drawing,
                                                                                               Drawing.id == cls.drawing_id)

                # Apply drawing filters
                if drawing_id is not None:
                    query = query.filter(Drawing.id == drawing_id)

                if drw_equipment_name is not None:
                    if exact_match:
                        query = query.filter(Drawing.drw_equipment_name == drw_equipment_name)
                    else:
                        query = query.filter(Drawing.drw_equipment_name.ilike(f"%{drw_equipment_name}%"))

                if drw_number is not None:
                    if exact_match:
                        query = query.filter(Drawing.drw_number == drw_number)
                    else:
                        query = query.filter(Drawing.drw_number.ilike(f"%{drw_number}%"))

                if drw_name is not None:
                    if exact_match:
                        query = query.filter(Drawing.drw_name == drw_name)
                    else:
                        query = query.filter(Drawing.drw_name.ilike(f"%{drw_name}%"))

                if drw_revision is not None:
                    if exact_match:
                        query = query.filter(Drawing.drw_revision == drw_revision)
                    else:
                        query = query.filter(Drawing.drw_revision.ilike(f"%{drw_revision}%"))

                if drw_spare_part_number is not None:
                    if exact_match:
                        query = query.filter(Drawing.drw_spare_part_number == drw_spare_part_number)
                    else:
                        query = query.filter(Drawing.drw_spare_part_number.ilike(f"%{drw_spare_part_number}%"))

                if file_path is not None:
                    if exact_match:
                        query = query.filter(Drawing.file_path == file_path)
                    else:
                        query = query.filter(Drawing.file_path.ilike(f"%{file_path}%"))

                # Apply position filters
                if position_id is not None:
                    query = query.filter(Position.id == position_id)

                if area_id is not None:
                    query = query.filter(Position.area_id == area_id)

                if equipment_group_id is not None:
                    query = query.filter(Position.equipment_group_id == equipment_group_id)

                if model_id is not None:
                    query = query.filter(Position.model_id == model_id)

                if asset_number_id is not None:
                    query = query.filter(Position.asset_number_id == asset_number_id)

                if location_id is not None:
                    query = query.filter(Position.location_id == location_id)

                if subassembly_id is not None:
                    query = query.filter(Position.subassembly_id == subassembly_id)

                if component_assembly_id is not None:
                    query = query.filter(Position.component_assembly_id == component_assembly_id)

                if assembly_view_id is not None:
                    query = query.filter(Position.assembly_view_id == assembly_view_id)

                if site_location_id is not None:
                    query = query.filter(Position.site_location_id == site_location_id)

                # Make results distinct to avoid duplicates
                query = query.distinct()

                # Apply limit
                query = query.limit(limit)

                # Execute query and log results
                results = query.all()
                debug_id(
                    f"DrawingPositionAssociation.get_positions_by_drawing completed, found {len(results)} positions",
                    rid)
                return results

        except Exception as e:
            error_id(f"Error in DrawingPositionAssociation.get_positions_by_drawing: {str(e)}", rid, exc_info=True)
            return []
        finally:
            # Close the session if we created it
            if not session_provided:
                session.close()
                debug_id(f"Closed database session for DrawingPositionAssociation.get_positions_by_drawing", rid)

    @classmethod
    @with_request_id
    def get_drawings_by_position(cls,
                                 position_id: Optional[int] = None,
                                 area_id: Optional[int] = None,
                                 equipment_group_id: Optional[int] = None,
                                 model_id: Optional[int] = None,
                                 asset_number_id: Optional[int] = None,
                                 location_id: Optional[int] = None,
                                 subassembly_id: Optional[int] = None,
                                 component_assembly_id: Optional[int] = None,
                                 assembly_view_id: Optional[int] = None,
                                 site_location_id: Optional[int] = None,
                                 drawing_id: Optional[int] = None,
                                 drw_equipment_name: Optional[str] = None,
                                 drw_number: Optional[str] = None,
                                 drw_name: Optional[str] = None,
                                 drw_revision: Optional[str] = None,
                                 drw_spare_part_number: Optional[str] = None,
                                 file_path: Optional[str] = None,
                                 exact_match: bool = False,
                                 limit: int = 100,
                                 request_id: Optional[str] = None,
                                 session: Optional[Session] = None) -> List['Drawing']:
        """
        Get drawings associated with positions based on flexible search criteria.

        Args:
            position_id: Optional position ID to filter by
            area_id: Optional area ID to filter by
            equipment_group_id: Optional equipment group ID to filter by
            model_id: Optional model ID to filter by
            asset_number_id: Optional asset number ID to filter by
            location_id: Optional location ID to filter by
            subassembly_id: Optional subassembly ID to filter by
            component_assembly_id: Optional component assembly ID to filter by
            assembly_view_id: Optional assembly view ID to filter by
            site_location_id: Optional site location ID to filter by
            drawing_id: Optional drawing ID to filter by
            drw_equipment_name: Optional equipment name to filter by
            drw_number: Optional drawing number to filter by
            drw_name: Optional drawing name to filter by
            drw_revision: Optional revision to filter by
            drw_spare_part_number: Optional spare part number to filter by
            file_path: Optional file path to filter by
            exact_match: If True, performs exact matching instead of partial matching for string fields
            limit: Maximum number of results to return (default 100)
            request_id: Optional request ID for tracking this operation in logs
            session: Optional SQLAlchemy session. If None, a new session will be created

        Returns:
            List of Drawing objects matching the search criteria
        """
        # Get or use the provided request_id
        rid = request_id or get_request_id()

        # Get a database session if one wasn't provided
        db_config = DatabaseConfig()
        session_provided = session is not None
        if not session_provided:
            session = db_config.get_main_session()
            debug_id(f"Created new database session for DrawingPositionAssociation.get_drawings_by_position", rid)

        # Log the search operation with request ID
        search_params = {
            'position_id': position_id,
            'area_id': area_id,
            'equipment_group_id': equipment_group_id,
            'model_id': model_id,
            'asset_number_id': asset_number_id,
            'location_id': location_id,
            'subassembly_id': subassembly_id,
            'component_assembly_id': component_assembly_id,
            'assembly_view_id': assembly_view_id,
            'site_location_id': site_location_id,
            'drawing_id': drawing_id,
            'drw_equipment_name': drw_equipment_name,
            'drw_number': drw_number,
            'drw_name': drw_name,
            'drw_revision': drw_revision,
            'drw_spare_part_number': drw_spare_part_number,
            'file_path': file_path,
            'exact_match': exact_match,
            'limit': limit
        }
        # Filter out None values for cleaner logging
        logged_params = {k: v for k, v in search_params.items() if v is not None}
        debug_id(f"Starting DrawingPositionAssociation.get_drawings_by_position with parameters: {logged_params}", rid)

        try:
            # Use the timed operation context manager with request ID
            with log_timed_operation("DrawingPositionAssociation.get_drawings_by_position", rid):


                # Start with a query that joins Drawing and DrawingPositionAssociation
                query = session.query(Drawing).join(cls, Drawing.id == cls.drawing_id).join(Position,
                                                                                            Position.id == cls.position_id)

                # Apply position filters
                if position_id is not None:
                    query = query.filter(Position.id == position_id)

                if area_id is not None:
                    query = query.filter(Position.area_id == area_id)

                if equipment_group_id is not None:
                    query = query.filter(Position.equipment_group_id == equipment_group_id)

                if model_id is not None:
                    query = query.filter(Position.model_id == model_id)

                if asset_number_id is not None:
                    query = query.filter(Position.asset_number_id == asset_number_id)

                if location_id is not None:
                    query = query.filter(Position.location_id == location_id)

                if subassembly_id is not None:
                    query = query.filter(Position.subassembly_id == subassembly_id)

                if component_assembly_id is not None:
                    query = query.filter(Position.component_assembly_id == component_assembly_id)

                if assembly_view_id is not None:
                    query = query.filter(Position.assembly_view_id == assembly_view_id)

                if site_location_id is not None:
                    query = query.filter(Position.site_location_id == site_location_id)

                # Apply drawing filters
                if drawing_id is not None:
                    query = query.filter(Drawing.id == drawing_id)

                if drw_equipment_name is not None:
                    if exact_match:
                        query = query.filter(Drawing.drw_equipment_name == drw_equipment_name)
                    else:
                        query = query.filter(Drawing.drw_equipment_name.ilike(f"%{drw_equipment_name}%"))

                if drw_number is not None:
                    if exact_match:
                        query = query.filter(Drawing.drw_number == drw_number)
                    else:
                        query = query.filter(Drawing.drw_number.ilike(f"%{drw_number}%"))

                if drw_name is not None:
                    if exact_match:
                        query = query.filter(Drawing.drw_name == drw_name)
                    else:
                        query = query.filter(Drawing.drw_name.ilike(f"%{drw_name}%"))

                if drw_revision is not None:
                    if exact_match:
                        query = query.filter(Drawing.drw_revision == drw_revision)
                    else:
                        query = query.filter(Drawing.drw_revision.ilike(f"%{drw_revision}%"))

                if drw_spare_part_number is not None:
                    if exact_match:
                        query = query.filter(Drawing.drw_spare_part_number == drw_spare_part_number)
                    else:
                        query = query.filter(Drawing.drw_spare_part_number.ilike(f"%{drw_spare_part_number}%"))

                if file_path is not None:
                    if exact_match:
                        query = query.filter(Drawing.file_path == file_path)
                    else:
                        query = query.filter(Drawing.file_path.ilike(f"%{file_path}%"))

                # Make results distinct to avoid duplicates
                query = query.distinct()

                # Apply limit
                query = query.limit(limit)

                # Execute query and log results
                results = query.all()
                debug_id(
                    f"DrawingPositionAssociation.get_drawings_by_position completed, found {len(results)} drawings",
                    rid)
                return results

        except Exception as e:
            error_id(f"Error in DrawingPositionAssociation.get_drawings_by_position: {str(e)}", rid, exc_info=True)
            return []
        finally:
            # Close the session if we created it
            if not session_provided:
                session.close()
                debug_id(f"Closed database session for DrawingPositionAssociation.get_drawings_by_position", rid)

class CompletedDocumentPositionAssociation(Base):
    __tablename__ = 'completed_document_position_association'
    id = Column(Integer, primary_key=True)
    complete_document_id = Column(Integer, ForeignKey('complete_document.id'))
    position_id = Column(Integer, ForeignKey('position.id'))

    complete_document = relationship("CompleteDocument", back_populates="completed_document_position_association")
    position = relationship("Position", back_populates="completed_document_position_association")

    @classmethod
    def associate(cls, session, position, complete_document):
        """
        Associate a given position with a complete document.

        Args:
            session: SQLAlchemy session object.
            position: Position instance to associate.
            complete_document: CompleteDocument instance to associate.

        Returns:
            The created association instance.
        """
        # Check if the association already exists
        association = session.query(cls).filter_by(
            position_id=position.id,
            complete_document_id=complete_document.id
        ).first()

        if not association:
            # Create a new association if it does not exist
            association = cls(
                position=position,
                complete_document=complete_document
            )
            session.add(association)
            session.commit()

        return association

class ImageCompletedDocumentAssociation(Base):
    __tablename__ = 'image_completed_document_association'

    id = Column(Integer, primary_key=True)
    complete_document_id = Column(Integer, ForeignKey('complete_document.id'))
    image_id = Column(Integer, ForeignKey('image.id'))
    document_id = Column(Integer, ForeignKey('document.id'), nullable=True)
    page_number = Column(Integer, nullable=True)
    chunk_index = Column(Integer, nullable=True)
    association_method = Column(String(50), default='sequential')
    confidence_score = Column(Float, nullable=True)
    context_metadata = Column(JSON, nullable=True)

    complete_document = relationship("CompleteDocument", back_populates="image_completed_document_association")
    image = relationship("Image", back_populates="image_completed_document_association")
    document_chunk = relationship("Document", back_populates="image_associations")

    # =====================================================
    # MODEL REFERENCES (ALIGNED)
    # =====================================================

    @classmethod
    def _get_image_class(cls):
        """Get Image class from same module."""
        return globals().get('Image')

    @classmethod
    def _get_document_class(cls):
        """Get Document class from same module."""
        return globals().get('Document')

    @classmethod
    def _get_complete_document_class(cls):
        """Get CompleteDocument class from same module."""
        return globals().get('CompleteDocument')

    # =====================================================
    # MAIN PUBLIC METHODS (ALIGNED)
    # =====================================================

    @classmethod
    @with_request_id
    def guided_extraction_with_mapping(cls, file_path, metadata, request_id=None):
        """
        FIXED: Enhanced guided extraction with proper chunk distribution and intelligent association.
        """
        try:
            complete_document_id = metadata.get('complete_document_id')
            position_id = metadata.get('position_id')

            if not complete_document_id:
                return False, {"error": "complete_document_id required"}, 400

            doc = fitz.open(file_path)
            file_name = os.path.splitext(os.path.basename(file_path))[0]
            associations_created = 0

            db_config = DatabaseConfig()
            with db_config.main_session() as session:
                # FIXED: Get all chunks and create a comprehensive mapping
                all_chunks = session.query(Document).filter_by(
                    complete_document_id=complete_document_id
                ).order_by(Document.id).all()

                if not all_chunks:
                    warning_id(f"No chunks found for complete_document_id {complete_document_id}", request_id)
                    return False, {"error": "No chunks found"}, 400

                info_id(f"Found {len(all_chunks)} total chunks for document {complete_document_id}", request_id)

                # FIXED: Create a better chunk mapping system
                chunk_page_map = cls._create_enhanced_chunk_page_mapping(all_chunks, request_id)

                for page_num in range(len(doc)):
                    page = doc[page_num]
                    img_list = page.get_images(full=True)

                    if not img_list:
                        debug_id(f"No images found on page {page_num + 1}", request_id)
                        continue

                    # FIXED: Get chunks for this specific page with better logic
                    page_chunks = cls._get_page_chunks_enhanced(
                        chunk_page_map, page_num, all_chunks, request_id
                    )

                    if not page_chunks:
                        warning_id(f"No chunks found for page {page_num + 1}, skipping image association", request_id)
                        continue

                    info_id(
                        f"Page {page_num + 1}: Processing {len(img_list)} images with {len(page_chunks)} available chunks",
                        request_id)

                    for img_index, img in enumerate(img_list):
                        try:
                            xref = img[0]
                            base_image = doc.extract_image(xref)
                            image_bytes = base_image["image"]
                            ext = base_image.get("ext", "jpg")

                            with tempfile.NamedTemporaryFile(suffix=f'.{ext}', delete=False) as tmp:
                                tmp.write(image_bytes)
                                temp_path = tmp.name

                            title = f"{file_name} - Page {page_num + 1} Image {img_index + 1}"

                            # FIXED: Intelligent chunk selection with multiple strategies
                            selected_chunk_info = cls._select_best_chunk_for_image(
                                page_chunks, img_index, page_num, img, request_id
                            )

                            if not selected_chunk_info:
                                warning_id(
                                    f"Could not select appropriate chunk for image {img_index + 1} on page {page_num + 1}",
                                    request_id)
                                continue

                            chunk_index = selected_chunk_info['chunk_index']
                            nearest_chunk = selected_chunk_info['chunk']
                            association_method = selected_chunk_info['method']
                            confidence = selected_chunk_info['confidence']

                            debug_id(
                                f"Page {page_num + 1}, Image {img_index + 1}: Selected chunk_index={chunk_index}, chunk_id={nearest_chunk.id}, method={association_method}",
                                request_id)

                            # Create image metadata
                            image_metadata = {
                                'page_number': page_num,
                                'image_index': img_index,
                                'extraction_method': 'structure_guided_enhanced',
                                'structure_guided': True,
                                'association_method': association_method,
                                'confidence_score': confidence
                            }

                            # Save the image
                            image_id = Image.add_to_db(
                                session=session,
                                title=title,
                                file_path=temp_path,
                                description=f"Enhanced guided extraction from {os.path.basename(file_path)}",
                                position_id=position_id,
                                complete_document_id=complete_document_id,
                                metadata=image_metadata,
                                request_id=request_id
                            )

                            if image_id is not None:
                                # FIXED: Create association with proper page and chunk indexing
                                association = cls(
                                    complete_document_id=complete_document_id,
                                    image_id=image_id,
                                    document_id=nearest_chunk.id,
                                    page_number=page_num,  # Correct page number (0-indexed)
                                    chunk_index=chunk_index,  # Proper chunk index within page
                                    association_method=association_method,
                                    confidence_score=confidence,
                                    context_metadata=json.dumps({
                                        'extraction_method': 'enhanced_guided',
                                        'selection_strategy': association_method,
                                        'page_total_images': len(img_list),
                                        'page_total_chunks': len(page_chunks),
                                        'created_at': datetime.now().isoformat()
                                    })
                                )
                                session.add(association)
                                associations_created += 1

                                info_id(
                                    f"Associated image {image_id} with chunk {nearest_chunk.id} (page {page_num + 1}, chunk_index {chunk_index})",
                                    request_id)
                            else:
                                warning_id(f"Failed to save image {title}", request_id)

                            # Cleanup temp file
                            try:
                                os.unlink(temp_path)
                            except:
                                pass

                        except Exception as e:
                            error_id(f"Error processing image {img_index + 1} on page {page_num + 1}: {e}", request_id)
                            continue

                session.commit()
                doc.close()

                info_id(f"Enhanced guided extraction completed: {associations_created} associations created",
                        request_id)
                return True, {"associations_created": associations_created}, 200

        except Exception as e:
            error_id(f"Enhanced guided extraction failed: {e}", request_id)
            return False, {"error": str(e)}, 500

    @classmethod
    @with_request_id
    def _process_images_with_structure_guidance(cls, file_path, complete_document_id, position_id,
                                                structure_map, session, request_id):
        """
        ALIGNED: Process images using structure guidance and Image.add_to_db method.
        """
        try:
            import fitz
            import tempfile
            import os

            ImageClass = cls._get_image_class()
            if not ImageClass:
                error_id("Image class not available", request_id)
                return 0

            associations_created = 0
            doc = fitz.open(file_path)

            for image_pos in structure_map.image_positions:
                try:
                    # Generate appropriate title and metadata
                    title = f"Page {image_pos.page_number + 1} - Image {image_pos.image_index + 1}"
                    if hasattr(image_pos, 'content_type'):
                        title += f" ({image_pos.content_type})"

                    # Enhanced metadata for structure-guided association
                    enhanced_metadata = {
                        'page_number': image_pos.page_number,
                        'image_index': image_pos.image_index,
                        'bbox': image_pos.bbox,
                        'estimated_size': image_pos.estimated_size,
                        'content_type': getattr(image_pos, 'content_type', 'image/png'),
                        'structure_guided': True,
                        'association_method': 'structure_guided',
                        'confidence_score': 0.9,
                        'extraction_source': 'DocumentStructureManager'
                    }

                    # Extract the actual image file
                    temp_image_path = cls._extract_image_from_position(
                        doc, image_pos, request_id
                    )

                    if temp_image_path and os.path.exists(temp_image_path):
                        try:
                            # ALIGNED: Use Image.add_to_db with proper metadata
                            image_record = ImageClass.add_to_db(
                                session=session,
                                title=title,
                                file_path=temp_image_path,
                                description=f"Structure-guided extraction from {os.path.basename(file_path)}",
                                position_id=position_id,
                                complete_document_id=complete_document_id,
                                metadata=enhanced_metadata,
                                request_id=request_id
                            )

                            if image_record:
                                associations_created += 1
                                debug_id(f"Successfully created structure-guided association for {title}", request_id)
                            else:
                                warning_id(f"Failed to create image record for {title}", request_id)

                        finally:
                            # Always clean up temp file
                            try:
                                os.unlink(temp_image_path)
                            except:
                                pass

                except Exception as e:
                    error_id(f"Error processing image {image_pos.image_index} on page {image_pos.page_number}: {e}",
                             request_id)
                    continue

            doc.close()
            info_id(f"Structure-guided image processing completed: {associations_created} associations created",
                    request_id)
            return associations_created

        except Exception as e:
            error_id(f"Error in structure-guided image processing: {e}", request_id)
            return 0

    @classmethod
    @with_request_id
    def _extract_image_from_position(cls, doc, image_pos, request_id):
        """
        ALIGNED: Extract a single image based on structure position data.
        """
        try:
            import tempfile
            import base64

            page = doc[image_pos.page_number]

            if getattr(image_pos, 'content_type', 'image/png') == 'image/svg+xml':
                # Handle SVG images
                if hasattr(image_pos, 'metadata') and isinstance(image_pos.metadata, dict):
                    svg_data = image_pos.metadata.get('svg_data')
                    if svg_data:
                        with tempfile.NamedTemporaryFile(suffix='.svg', delete=False) as tmp:
                            tmp.write(base64.b64decode(svg_data))
                            return tmp.name

                # Create placeholder SVG if no data
                with tempfile.NamedTemporaryFile(suffix='.svg', delete=False) as tmp:
                    svg_content = f'''<?xml version="1.0" encoding="UTF-8"?>
                    <svg width="100" height="100" xmlns="http://www.w3.org/2000/svg">
                        <rect width="100" height="100" fill="#f0f0f0" stroke="#ccc"/>
                        <text x="50" y="50" text-anchor="middle" dy=".3em">SVG Placeholder</text>
                    </svg>'''
                    tmp.write(svg_content.encode('utf-8'))
                    return tmp.name
            else:
                # Handle regular images
                img_list = page.get_images(full=True)
                if image_pos.image_index < len(img_list):
                    img = img_list[image_pos.image_index]
                    xref = img[0]
                    base_image = doc.extract_image(xref)
                    image_bytes = base_image["image"]
                    ext = base_image.get("ext", "png")

                    with tempfile.NamedTemporaryFile(suffix=f'.{ext}', delete=False) as tmp:
                        tmp.write(image_bytes)
                        return tmp.name

            return None

        except Exception as e:
            debug_id(f"Error extracting single image: {e}", request_id)
            return None

    @classmethod
    @with_request_id
    def _create_chunk_associations(cls, complete_document_id, structure_map, session, request_id):
        """
        ALIGNED: Create associations between images and document chunks.
        """
        try:
            DocumentClass = cls._get_document_class()
            ImageClass = cls._get_image_class()

            if not DocumentClass or not ImageClass:
                warning_id("Document or Image class not available for chunk associations", request_id)
                return 0

            chunk_associations = 0

            # Get existing chunks for this document
            chunks = session.query(DocumentClass).filter(
                DocumentClass.complete_document_id == complete_document_id
            ).all()

            # Get existing images for this document
            images = session.query(ImageClass).join(
                cls, ImageClass.id == cls.image_id
            ).filter(
                cls.complete_document_id == complete_document_id
            ).all()

            if not chunks or not images:
                debug_id("No chunks or images found for chunk association", request_id)
                return 0

            # Create associations based on structure analysis
            for chunk in chunks:
                try:
                    # Parse chunk metadata to find related images
                    chunk_metadata = {}
                    if hasattr(chunk, 'metadata') and chunk.metadata:
                        try:
                            chunk_metadata = json.loads(chunk.metadata) if isinstance(chunk.metadata,
                                                                                      str) else chunk.metadata
                        except:
                            pass

                    page_number = chunk_metadata.get('page_number')
                    if page_number is not None:
                        # Find images on the same page
                        related_images = [
                            img for img in images
                            if cls._get_image_page_number(img) == page_number
                        ]

                        # Create associations for related images
                        for image in related_images:
                            # Check if association already exists
                            existing = session.query(cls).filter(
                                cls.image_id == image.id,
                                cls.document_id == chunk.id,
                                cls.complete_document_id == complete_document_id
                            ).first()

                            if not existing:
                                association = cls(
                                    complete_document_id=complete_document_id,
                                    image_id=image.id,
                                    document_id=chunk.id,
                                    page_number=page_number,
                                    association_method='structure_guided_chunk',
                                    confidence_score=0.8,
                                    context_metadata=json.dumps({
                                        'created_by': 'aligned_chunk_association',
                                        'structure_guided': True,
                                        'page_number': page_number,
                                        'created_at': datetime.now().isoformat()
                                    })
                                )
                                session.add(association)
                                chunk_associations += 1

                except Exception as e:
                    debug_id(f"Error creating chunk association for chunk {chunk.id}: {e}", request_id)
                    continue

            info_id(f"Created {chunk_associations} chunk associations", request_id)
            return chunk_associations

        except Exception as e:
            error_id(f"Error creating chunk associations: {e}", request_id)
            return 0

    @classmethod
    def _get_image_page_number(cls, image):
        """Helper to extract page number from image metadata."""
        try:
            if hasattr(image, 'img_metadata') and image.img_metadata:
                metadata = image.img_metadata
                if isinstance(metadata, str):
                    metadata = json.loads(metadata)
                return metadata.get('page_number')
        except:
            pass
        return None

    @classmethod
    @with_request_id
    def _fallback_basic_extraction(cls, file_path, metadata, request_id):
        """
        ALIGNED: Fallback to basic extraction if structure analysis fails.
        """
        try:
            warning_id("Using fallback basic extraction method", request_id)

            complete_document_id = metadata.get('complete_document_id')
            position_id = metadata.get('position_id')

            if not complete_document_id:
                return False, {'error': 'Missing complete_document_id'}, 400

            # Use basic PDF image extraction
            associations_created = cls._basic_pdf_extraction(
                file_path, complete_document_id, position_id, request_id
            )

            result = {
                'success': True,
                'complete_document_id': complete_document_id,
                'associations_created': associations_created,
                'processing_method': 'fallback_basic_extraction'
            }

            return True, result, 200

        except Exception as e:
            error_id(f"Fallback extraction failed: {e}", request_id)
            return False, {'error': str(e)}, 500

    @classmethod
    @with_request_id
    def _basic_pdf_extraction(cls, file_path, complete_document_id, position_id, request_id):
        """Basic PDF image extraction using Image.add_to_db."""
        try:
            import fitz
            import tempfile
            from modules.configuration.config_env import DatabaseConfig

            ImageClass = cls._get_image_class()
            if not ImageClass:
                return 0

            db_config = DatabaseConfig()
            associations_created = 0

            with db_config.main_session() as session:
                doc = fitz.open(file_path)
                file_name = os.path.splitext(os.path.basename(file_path))[0]

                for page_num in range(len(doc)):
                    page = doc[page_num]
                    img_list = page.get_images(full=True)

                    for img_index, img in enumerate(img_list):
                        try:
                            xref = img[0]
                            base_image = doc.extract_image(xref)
                            image_bytes = base_image["image"]
                            ext = base_image.get("ext", "jpg")

                            with tempfile.NamedTemporaryFile(suffix=f'.{ext}', delete=False) as tmp:
                                tmp.write(image_bytes)
                                temp_path = tmp.name

                            title = f"{file_name} - Page {page_num + 1} Image {img_index + 1}"

                            metadata = {
                                'page_number': page_num,
                                'image_index': img_index,
                                'extraction_method': 'fallback_basic',
                                'structure_guided': False
                            }

                            image_record = ImageClass.add_to_db(
                                session=session,
                                title=title,
                                file_path=temp_path,
                                description=f"Fallback extraction from {os.path.basename(file_path)}",
                                position_id=position_id,
                                complete_document_id=complete_document_id,
                                metadata=metadata,
                                request_id=request_id
                            )

                            if image_record:
                                associations_created += 1

                            try:
                                os.unlink(temp_path)
                            except:
                                pass

                        except Exception as e:
                            debug_id(f"Error in basic extraction for image {img_index}: {e}", request_id)
                            continue

                doc.close()

            info_id(f"Basic extraction completed: {associations_created} associations", request_id)
            return associations_created

        except Exception as e:
            error_id(f"Basic extraction failed: {e}", request_id)
            return 0

    # =====================================================
    # STRUCTURE ANALYSIS METHODS (SIMPLIFIED AND ALIGNED)
    # =====================================================

    @classmethod
    @with_request_id
    def analyze_document_structure(cls, file_path: str, request_id=None, ocr_content=None):
        """
        ALIGNED: Use DocumentStructureManager from db_manager.py instead of duplicating logic.
        """
        try:
            from modules.database_manager.db_manager import DocumentStructureManager
            from modules.configuration.config_env import DatabaseConfig

            info_id(f"Using DocumentStructureManager for analysis: {file_path}", request_id)

            db_config = DatabaseConfig()
            with db_config.main_session() as session:
                structure_manager = DocumentStructureManager(session=session, request_id=request_id)
                return structure_manager.analyze_document_structure(file_path, request_id, ocr_content)

        except Exception as e:
            error_id(f"Error in aligned document structure analysis: {e}", request_id, exc_info=True)
            raise

    # =====================================================
    # QUERY AND UTILITY METHODS (ALIGNED)
    # =====================================================

    @classmethod
    @with_request_id
    def get_images_with_chunk_context(cls, complete_document_id, request_id=None):
        """
        ALIGNED: Get all images for a document with their associated chunk context.
        Delegates to Image class method.
        """
        try:
            ImageClass = cls._get_image_class()
            if not ImageClass:
                error_id("Image class not available", request_id)
                return []

            from modules.configuration.config_env import DatabaseConfig
            db_config = DatabaseConfig()

            with db_config.main_session() as session:
                return ImageClass.get_images_with_chunk_context(session, complete_document_id, request_id)

        except Exception as e:
            error_id(f"Failed to get images with chunk context: {e}", request_id)
            return []

    @classmethod
    @with_request_id
    def search_by_chunk_text(cls, search_text, complete_document_id=None, confidence_threshold=0.5, request_id=None):
        """
        ALIGNED: Search for images by their associated chunk text content.
        Delegates to Image class method.
        """
        try:
            ImageClass = cls._get_image_class()
            if not ImageClass:
                error_id("Image class not available", request_id)
                return []

            from modules.configuration.config_env import DatabaseConfig
            db_config = DatabaseConfig()

            with db_config.main_session() as session:
                return ImageClass.search_by_chunk_text(
                    session, search_text, complete_document_id, confidence_threshold, request_id
                )

        except Exception as e:
            error_id(f"Failed to search images by chunk text: {e}", request_id)
            return []

    @classmethod
    @with_request_id
    def get_association_statistics(cls, complete_document_id=None, request_id=None):
        """
        ALIGNED: Get statistics about image-chunk associations.
        Delegates to Image class method.
        """
        try:
            ImageClass = cls._get_image_class()
            if not ImageClass:
                error_id("Image class not available", request_id)
                return {}

            from modules.configuration.config_env import DatabaseConfig
            db_config = DatabaseConfig()

            with db_config.main_session() as session:
                return ImageClass.get_association_statistics(session, complete_document_id, request_id)

        except Exception as e:
            error_id(f"Failed to get association statistics: {e}", request_id)
            return {}

    @classmethod
    @with_request_id
    def update_association_confidence(cls, association_id, new_confidence, request_id=None):
        """Update the confidence score of an association."""
        try:
            from modules.configuration.config_env import DatabaseConfig
            db_config = DatabaseConfig()

            with db_config.main_session() as session:
                association = session.query(cls).filter(cls.id == association_id).first()
                if association:
                    association.confidence_score = new_confidence
                    session.commit()
                    info_id(f"Updated association {association_id} confidence to {new_confidence}", request_id)
                    return True
                else:
                    warning_id(f"Association {association_id} not found", request_id)
                    return False

        except Exception as e:
            error_id(f"Failed to update association confidence: {e}", request_id)
            return False

    @classmethod
    @with_request_id
    def bulk_update_associations(cls, document_id, association_method='bulk_update', confidence_score=0.7,
                                 request_id=None):
        """Bulk update associations for a document."""
        try:
            from modules.configuration.config_env import DatabaseConfig
            from sqlalchemy import text

            db_config = DatabaseConfig()

            with db_config.main_session() as session:
                # Use raw SQL for bulk update efficiency
                result = session.execute(text("""
                    UPDATE image_completed_document_association 
                    SET association_method = :method, confidence_score = :confidence 
                    WHERE complete_document_id = :doc_id
                """), {
                    'method': association_method,
                    'confidence': confidence_score,
                    'doc_id': document_id
                })

                updated_count = result.rowcount
                session.commit()

                info_id(f"Bulk updated {updated_count} associations for document {document_id}", request_id)
                return updated_count

        except Exception as e:
            error_id(f"Failed to bulk update associations: {e}", request_id)
            return 0

    # =====================================================
    # HELPER METHODS (CLEANED UP)
    # =====================================================

    @classmethod
    def _store_structure_analysis(cls, structure_map, request_id=None):
        """
        ALIGNED: Store structure analysis using the DocumentStructureManager approach.
        """
        try:
            from modules.database_manager.db_manager import DocumentStructureManager
            from modules.configuration.config_env import DatabaseConfig

            db_config = DatabaseConfig()
            with db_config.main_session() as session:
                structure_manager = DocumentStructureManager(session=session, request_id=request_id)
                # Use the manager's storage method if available
                if hasattr(structure_manager, '_store_structure_analysis'):
                    structure_manager._store_structure_analysis(structure_map, request_id)
                else:
                    debug_id("Structure analysis storage not available in DocumentStructureManager", request_id)

        except Exception as e:
            warning_id(f"Could not store structure analysis: {e}", request_id)

    def to_dict(self):
        """Convert association to dictionary for JSON serialization."""
        return {
            'id': self.id,
            'complete_document_id': self.complete_document_id,
            'image_id': self.image_id,
            'document_id': self.document_id,
            'page_number': self.page_number,
            'chunk_index': self.chunk_index,
            'association_method': self.association_method,
            'confidence_score': self.confidence_score,
            'context_metadata': self.context_metadata
        }

    def __repr__(self):
        return (f"<ImageCompletedDocumentAssociation(id={self.id}, "
                f"doc={self.complete_document_id}, img={self.image_id}, "
                f"method='{self.association_method}', confidence={self.confidence_score})>")

    @classmethod
    def _create_enhanced_chunk_page_mapping(cls, all_chunks, request_id=None):
        """
        FIXED: Create a comprehensive mapping of chunks to pages with better metadata handling.
        """
        chunk_page_map = {}
        chunks_without_page_info = []

        for chunk in all_chunks:
            page_number = None

            # Try multiple ways to get page number
            if hasattr(chunk, 'doc_metadata') and chunk.doc_metadata:
                if isinstance(chunk.doc_metadata, dict):
                    page_number = chunk.doc_metadata.get("page_number")
                elif isinstance(chunk.doc_metadata, str):
                    try:
                        metadata_dict = json.loads(chunk.doc_metadata)
                        page_number = metadata_dict.get("page_number")
                    except:
                        pass

            # Fallback: try metadata field
            if page_number is None and hasattr(chunk, 'metadata') and chunk.metadata:
                try:
                    if isinstance(chunk.metadata, dict):
                        page_number = chunk.metadata.get("page_number")
                    elif isinstance(chunk.metadata, str):
                        metadata_dict = json.loads(chunk.metadata)
                        page_number = metadata_dict.get("page_number")
                except:
                    pass

            if page_number is not None:
                if page_number not in chunk_page_map:
                    chunk_page_map[page_number] = []
                chunk_page_map[page_number].append(chunk)
            else:
                chunks_without_page_info.append(chunk)

        # Log mapping statistics
        info_id(
            f"Chunk page mapping: {len(chunk_page_map)} pages mapped, {len(chunks_without_page_info)} chunks without page info",
            request_id)
        for page_num, chunks in chunk_page_map.items():
            debug_id(f"Page {page_num}: {len(chunks)} chunks", request_id)

        # FIXED: Distribute unmapped chunks across pages if needed
        if chunks_without_page_info:
            warning_id(f"Found {len(chunks_without_page_info)} chunks without page information, distributing evenly",
                       request_id)
            cls._distribute_unmapped_chunks(chunk_page_map, chunks_without_page_info, request_id)

        return chunk_page_map

    @classmethod
    def _distribute_unmapped_chunks(cls, chunk_page_map, unmapped_chunks, request_id=None):
        """
        FIXED: Intelligently distribute chunks that don't have page information.
        """
        if not chunk_page_map:
            # If no page mapping exists, create a default page 0
            chunk_page_map[0] = unmapped_chunks
            warning_id(f"No page mapping found, assigning all {len(unmapped_chunks)} chunks to page 0", request_id)
            return

        # Get the pages that have chunks
        pages_with_chunks = sorted(chunk_page_map.keys())
        chunks_per_page = len(unmapped_chunks) // len(pages_with_chunks)
        remaining_chunks = len(unmapped_chunks) % len(pages_with_chunks)

        chunk_index = 0
        for i, page_num in enumerate(pages_with_chunks):
            # Calculate how many chunks to assign to this page
            chunks_to_assign = chunks_per_page
            if i < remaining_chunks:
                chunks_to_assign += 1

            # Assign chunks to this page
            for _ in range(chunks_to_assign):
                if chunk_index < len(unmapped_chunks):
                    chunk_page_map[page_num].append(unmapped_chunks[chunk_index])
                    chunk_index += 1

        debug_id(f"Distributed {len(unmapped_chunks)} unmapped chunks across {len(pages_with_chunks)} pages",
                 request_id)

    @classmethod
    def _get_page_chunks_enhanced(cls, chunk_page_map, page_num, all_chunks, request_id=None):
        """
        FIXED: Get chunks for a specific page with fallback strategies.
        """
        # Strategy 1: Direct page mapping
        if page_num in chunk_page_map:
            page_chunks = [(i, chunk) for i, chunk in enumerate(chunk_page_map[page_num])]
            debug_id(f"Strategy 1 - Direct mapping: Found {len(page_chunks)} chunks for page {page_num}", request_id)
            return page_chunks

        # Strategy 2: Check for off-by-one errors (0-indexed vs 1-indexed)
        alt_page_num = page_num + 1 if page_num + 1 in chunk_page_map else page_num - 1
        if alt_page_num in chunk_page_map and alt_page_num >= 0:
            page_chunks = [(i, chunk) for i, chunk in enumerate(chunk_page_map[alt_page_num])]
            debug_id(
                f"Strategy 2 - Off-by-one correction: Found {len(page_chunks)} chunks for page {alt_page_num} (original {page_num})",
                request_id)
            return page_chunks

        # Strategy 3: Use nearest page with chunks
        available_pages = sorted(chunk_page_map.keys())
        if available_pages:
            nearest_page = min(available_pages, key=lambda x: abs(x - page_num))
            page_chunks = [(i, chunk) for i, chunk in enumerate(chunk_page_map[nearest_page])]
            warning_id(f"Strategy 3 - Nearest page: Using page {nearest_page} chunks for page {page_num}", request_id)
            return page_chunks

        # Strategy 4: Fallback to distributing all chunks
        if all_chunks:
            chunk_count = min(5, len(all_chunks))  # Limit to 5 chunks per page as fallback
            start_idx = (page_num * chunk_count) % len(all_chunks)
            fallback_chunks = []
            for i in range(chunk_count):
                chunk_idx = (start_idx + i) % len(all_chunks)
                fallback_chunks.append((i, all_chunks[chunk_idx]))
            warning_id(f"Strategy 4 - Fallback distribution: Created {len(fallback_chunks)} chunks for page {page_num}",
                       request_id)
            return fallback_chunks

        warning_id(f"No chunks available for page {page_num} using any strategy", request_id)
        return []

    @classmethod
    def _select_best_chunk_for_image(cls, page_chunks, img_index, page_num, img_data, request_id=None):
        """
        FIXED: Intelligent chunk selection with multiple strategies for better distribution.
        """
        if not page_chunks:
            return None

        # Strategy 1: Round-robin distribution (most balanced)
        if len(page_chunks) > 1:
            chunk_index = img_index % len(page_chunks)
            selected_chunk_info = {
                'chunk_index': chunk_index,
                'chunk': page_chunks[chunk_index][1],
                'method': 'round_robin',
                'confidence': 0.8
            }
            debug_id(f"Round-robin selection: img {img_index} -> chunk {chunk_index}/{len(page_chunks)}", request_id)
            return selected_chunk_info

        # Strategy 2: Single chunk (assign all images to the one available chunk)
        elif len(page_chunks) == 1:
            selected_chunk_info = {
                'chunk_index': 0,
                'chunk': page_chunks[0][1],
                'method': 'single_chunk',
                'confidence': 0.7
            }
            debug_id(f"Single chunk selection: img {img_index} -> chunk 0 (only option)", request_id)
            return selected_chunk_info

        # Strategy 3: No chunks available
        else:
            warning_id(f"No chunks available for image {img_index} on page {page_num}", request_id)
            return None

    # ADDITIONAL DEBUGGING METHOD
    @classmethod
    @with_request_id
    def debug_chunk_distribution(cls, complete_document_id, request_id=None):
        """
        DEBUGGING: Analyze chunk distribution for a document.
        """
        try:
            db_config = DatabaseConfig()
            with db_config.main_session() as session:
                chunks = session.query(Document).filter_by(
                    complete_document_id=complete_document_id
                ).order_by(Document.id).all()

                info_id(f"=== CHUNK DISTRIBUTION ANALYSIS FOR DOCUMENT {complete_document_id} ===", request_id)
                info_id(f"Total chunks: {len(chunks)}", request_id)

                page_distribution = {}
                chunks_without_page = 0

                for i, chunk in enumerate(chunks):
                    page_num = None

                    # Check doc_metadata
                    if hasattr(chunk, 'doc_metadata') and chunk.doc_metadata:
                        if isinstance(chunk.doc_metadata, dict):
                            page_num = chunk.doc_metadata.get("page_number")
                        elif isinstance(chunk.doc_metadata, str):
                            try:
                                metadata_dict = json.loads(chunk.doc_metadata)
                                page_num = metadata_dict.get("page_number")
                            except:
                                pass

                    if page_num is not None:
                        if page_num not in page_distribution:
                            page_distribution[page_num] = []
                        page_distribution[page_num].append(chunk.id)
                    else:
                        chunks_without_page += 1
                        debug_id(f"Chunk {chunk.id} (#{i}): NO PAGE INFO - name: {chunk.name}", request_id)

                info_id(f"Chunks without page info: {chunks_without_page}", request_id)
                info_id(f"Page distribution:", request_id)
                for page_num in sorted(page_distribution.keys()):
                    chunk_ids = page_distribution[page_num]
                    info_id(
                        f"  Page {page_num}: {len(chunk_ids)} chunks - IDs: {chunk_ids[:5]}{'...' if len(chunk_ids) > 5 else ''}",
                        request_id)

                info_id("=== END CHUNK DISTRIBUTION ANALYSIS ===", request_id)
                return page_distribution

        except Exception as e:
            error_id(f"Error in chunk distribution analysis: {e}", request_id)
            return {}

# Process Classes
class FileLog(Base):
    __tablename__ = 'file_logs'
    log_id = Column(Integer, primary_key=True, autoincrement=True)
    session = Column(Integer, nullable=False)
    session_datetime = Column(DateTime, nullable=False)
    file_processed = Column(String)  # Added column for file processed
    total_time = Column(String)

class KeywordAction(Base):
    __tablename__ = 'keyword_actions'

    id = Column(Integer, primary_key=True)
    keyword = Column(String, unique=True)
    action = Column(String)

    # Manually define the query attribute
    query = scoped_session(session).query_property()

    @classmethod
    @with_request_id
    def find_best_match(cls, user_input, session):

        try:
            # Retrieve all keywords from the database
            all_keywords = [keyword.keyword for keyword in session.query(cls).all()]

            # Use fuzzy string matching to find the best matching keyword
            logger.debug("All keywords: %s", all_keywords)
            matched_keyword, similarity_score = process.extractOne(user_input, all_keywords)
            logger.debug("Matched keyword: %s", matched_keyword)
            logger.debug("Similarity score: %s", similarity_score)

            # Set a threshold for the minimum similarity score
            threshold = 50

            # If the similarity score exceeds the threshold, return the matched keyword and its associated action
            if similarity_score >= threshold:
                # Extract keyword and details using spaCy
                keyword, details = cls.extract_keyword_and_details(user_input)
                if keyword:
                    # Retrieve the associated action from the database using the matched keyword
                    keyword_entry = session.query(cls).filter_by(keyword=keyword).first()
                    if keyword_entry:
                        action = keyword_entry.action
                        logger.debug("Associated action: %s", action)
                        return keyword, action, None  # No need to extract details

            # If no matching keyword is found or similarity score is below threshold, return None
            logger.debug("No matching keyword found or similarity score is below threshold.")
            return None, None, None

        except SQLAlchemyError as e:
            # Handle SQLAlchemy errors
            logger.error("Database error: %s", e)
            return None, None, None

        except Exception as e:
            # Handle other unexpected errors
            logger.error("Unexpected error: %s", e)
            return None, None, None

    @classmethod
    @with_request_id
    def extract_keyword_and_details(cls, text: str, session=None, request_id=None):
        """
        Extract keywords and details from input text using spaCy NLP processing.

        Args:
            text (str): Input text to process
            session: SQLAlchemy session (optional, will create if None)
            request_id (str, optional): Unique identifier for the request

        Returns:
            tuple: (keyword, details) - extracted keyword and remaining details

        Raises:
            Exception: If processing fails
        """
        try:
            debug_id(f"Starting keyword extraction from text: {text[:100]}...", request_id)

            # Get database session if not provided
            session_provided = session is not None
            if not session_provided:
                from modules.configuration.config_env import DatabaseConfig
                db_config = DatabaseConfig()
                session = db_config.get_main_session()
                debug_id("Created new database session for keyword extraction", request_id)

            # Preprocess the input text
            preprocessed_text = cls._preprocess_text(text)
            debug_id(f"Preprocessed text: {preprocessed_text}", request_id)

            # Tokenize the preprocessed text using spaCy
            doc = nlp(preprocessed_text)
            debug_id(f"Tokenized text into {len(doc)} tokens", request_id)

            # Initialize variables to store keyword and details
            keyword = ""
            details = ""

            # Retrieve all keywords from the database
            all_keywords = [keyword_action.keyword for keyword_action in session.query(cls).all()]
            debug_id(f"Retrieved {len(all_keywords)} keywords from database", request_id)

            # Iterate through the tokens
            for token in doc:
                # Check if the token is a keyword
                if token.text in all_keywords:
                    keyword += token.text + " "
                else:
                    details += token.text + " "

            # Remove trailing whitespace
            keyword = keyword.strip()
            details = details.strip()

            info_id(f"Extracted keyword: '{keyword}', details: '{details[:50]}...'", request_id)
            return keyword, details

        except Exception as e:
            error_id(f"Error extracting keyword and details: {e}", request_id, exc_info=True)
            raise
        finally:
            # Close session if we created it
            if not session_provided and session:
                session.close()
                debug_id("Closed database session for keyword extraction", request_id)

    @staticmethod
    def _preprocess_text(text: str) -> str:
        """
        Preprocess text for keyword extraction.

        Args:
            text (str): Raw input text

        Returns:
            str: Preprocessed text
        """
        # Basic preprocessing - can be expanded as needed
        # Convert to lowercase, strip whitespace, etc.
        processed = text.lower().strip()

        # Remove extra whitespace
        import re
        processed = re.sub(r'\s+', ' ', processed)

        return processed

class ChatSession(Base):
    __tablename__ = 'chat_sessions'

    session_id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(String, nullable=False, index=True)
    start_time = Column(String, nullable=False)
    last_interaction = Column(String, nullable=False)
    session_data = Column(MutableList.as_mutable(JSON), default=[])
    conversation_summary = Column(MutableList.as_mutable(JSON), default=[])

    # Vector embeddings for semantic search
    summary_embedding = Column(Vector(1536))  # OpenAI embedding dimension
    topic_tags = Column(JSON, default=[])

    # PostgreSQL specific optimizations
    __table_args__ = (
        Index('idx_user_last_interaction', 'user_id', 'last_interaction'),
        Index('idx_summary_embedding_cosine', 'summary_embedding', postgresql_using='ivfflat',
              postgresql_with={'lists': 100}, postgresql_ops={'summary_embedding': 'vector_cosine_ops'}),
    )

    def __init__(self, user_id, start_time, last_interaction, session_data=None, conversation_summary=None):
        self.user_id = user_id
        self.start_time = start_time
        self.last_interaction = last_interaction
        self.session_data = session_data or []
        self.conversation_summary = conversation_summary or []

    @classmethod
    def find_similar_conversations(cls, query_embedding, user_id=None, limit=5, similarity_threshold=0.7,
                                   db_session=None):
        """
        Find conversations similar to the query using vector similarity.

        Args:
            query_embedding: Vector embedding of the query
            user_id: Optional user ID to filter by
            limit: Maximum number of results
            similarity_threshold: Minimum similarity score (0-1)
            db_session: SQLAlchemy session

        Returns:
            List of tuples (ChatSession, similarity_score)
        """
        query = db_session.query(
            cls,
            cls.summary_embedding.cosine_distance(query_embedding).label('distance')
        ).filter(
            cls.summary_embedding.is_not(None),
            cls.summary_embedding.cosine_distance(query_embedding) < (1 - similarity_threshold)
        )

        if user_id:
            query = query.filter(cls.user_id == user_id)

        results = query.order_by(text('distance')).limit(limit).all()

        # Convert distance to similarity score
        return [(session, 1 - distance) for session, distance in results]

    @classmethod
    def update_summary_embedding(cls, session_id, embedding, db_session):
        """
        Update the summary embedding for a session.

        Args:
            session_id: The ID of the session
            embedding: The embedding vector
            db_session: SQLAlchemy session

        Returns:
            Boolean indicating success
        """
        try:
            chat_session = db_session.query(cls).filter_by(session_id=session_id).first()
            if chat_session:
                chat_session.summary_embedding = embedding
                db_session.commit()
                logger.debug(f"Summary embedding updated for session ID {session_id}")
                return True
            else:
                logger.error(f"No chat session found for session ID {session_id}")
                return False
        except Exception as e:
            db_session.rollback()
            logger.error(f"Error updating summary embedding for session ID {session_id}: {e}")
            return False

    @classmethod
    def get_conversation_summary(cls, session_id, db_session):
        """Enhanced version with better error handling and logging."""
        logger.debug(f"Retrieving conversation summary for session ID: {session_id}")
        try:
            chat_session = db_session.query(cls).filter_by(session_id=session_id).first()
            if chat_session and chat_session.conversation_summary:
                logger.debug(f"Found conversation summary with {len(chat_session.conversation_summary)} entries")
                return chat_session.conversation_summary
            logger.debug(f"No conversation summary found for session ID {session_id}")
            return []
        except Exception as e:
            logger.error(f"Error retrieving conversation summary: {e}")
            return []

    @classmethod
    def update_conversation_summary(cls, session_id, summary_data, db_session):
        """Enhanced with better transaction handling."""
        logger.debug(f"Updating conversation summary for session ID: {session_id}")
        try:
            chat_session = db_session.query(cls).filter_by(session_id=session_id).first()
            if chat_session:
                chat_session.conversation_summary = summary_data
                chat_session.last_interaction = datetime.utcnow().isoformat()
                db_session.commit()
                logger.debug(f"Conversation summary updated for session ID {session_id}")
                return True
            else:
                logger.error(f"No chat session found for session ID {session_id}")
                return False
        except Exception as e:
            db_session.rollback()
            logger.error(f"Error updating conversation summary for session ID {session_id}: {e}")
            return False

    @classmethod
    def clear_conversation_summary(cls, session_id, db_session):
        """Enhanced with embedding cleanup."""
        logger.debug(f"Clearing conversation summary for session ID: {session_id}")
        try:
            chat_session = db_session.query(cls).filter_by(session_id=session_id).first()
            if chat_session:
                chat_session.conversation_summary = []
                chat_session.summary_embedding = None  # Clear embedding too
                db_session.commit()
                logger.info(f"Conversation summary cleared for session ID {session_id}")
                return True
            else:
                logger.error(f"No chat session found for session ID {session_id}")
                return False
        except Exception as e:
            db_session.rollback()
            logger.error(f"Error clearing conversation summary for session ID {session_id}: {e}")
            return False

class QandA(Base):
    __tablename__ = 'qanda'

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(String, index=True)
    question = Column(String)
    answer = Column(String)
    comment = Column(String)
    rating = Column(String)
    timestamp = Column(String, nullable=False, index=True)
    request_id = Column(String, index=True)
    raw_response = Column(JSON)

    # Vector embeddings for semantic search
    question_embedding = Column(Vector(1536))
    answer_embedding = Column(Vector(1536))

    # Additional metadata
    question_length = Column(Integer)
    answer_length = Column(Integer)
    processing_time_ms = Column(Integer)

    # PostgreSQL specific optimizations
    __table_args__ = (
        Index('idx_user_timestamp', 'user_id', 'timestamp'),
        Index('idx_question_embedding_cosine', 'question_embedding', postgresql_using='ivfflat',
              postgresql_with={'lists': 100}, postgresql_ops={'question_embedding': 'vector_cosine_ops'}),
        Index('idx_answer_embedding_cosine', 'answer_embedding', postgresql_using='ivfflat',
              postgresql_with={'lists': 100}, postgresql_ops={'answer_embedding': 'vector_cosine_ops'}),
    )

    def __init__(self, user_id, question, answer, timestamp, rating=None, comment=None):
        self.user_id = user_id
        self.question = question
        self.answer = answer
        self.timestamp = timestamp
        self.rating = rating
        self.comment = comment
        self.question_length = len(question) if question else 0
        self.answer_length = len(answer) if answer else 0

    @classmethod
    def find_similar_questions(cls, query_embedding, user_id=None, limit=5, similarity_threshold=0.8, db_session=None):
        """
        Find similar questions using vector similarity.

        Args:
            query_embedding: Vector embedding of the query
            user_id: Optional user ID to filter by
            limit: Maximum number of results
            similarity_threshold: Minimum similarity score (0-1)
            db_session: SQLAlchemy session

        Returns:
            List of tuples (QandA, similarity_score)
        """
        query = db_session.query(
            cls,
            cls.question_embedding.cosine_distance(query_embedding).label('distance')
        ).filter(
            cls.question_embedding.is_not(None),
            cls.question_embedding.cosine_distance(query_embedding) < (1 - similarity_threshold)
        )

        if user_id:
            query = query.filter(cls.user_id == user_id)

        results = query.order_by(text('distance')).limit(limit).all()

        return [(qa, 1 - distance) for qa, distance in results]

    @classmethod
    def get_user_analytics(cls, user_id, db_session):
        """
        Get analytics for a specific user using PostgreSQL aggregations.

        Args:
            user_id: The user ID
            db_session: SQLAlchemy session

        Returns:
            Dictionary with user analytics
        """
        result = db_session.execute(text("""
            SELECT 
                COUNT(*) as total_questions,
                AVG(question_length) as avg_question_length,
                AVG(answer_length) as avg_answer_length,
                AVG(processing_time_ms) as avg_processing_time,
                COUNT(CASE WHEN rating IS NOT NULL THEN 1 END) as rated_answers,
                AVG(CASE WHEN rating ~ '^[0-9]+$' THEN rating::int END) as avg_rating
            FROM qanda 
            WHERE user_id = :user_id
        """), {"user_id": user_id}).first()

        if result:
            return {
                'total_questions': result.total_questions,
                'avg_question_length': float(result.avg_question_length or 0),
                'avg_answer_length': float(result.avg_answer_length or 0),
                'avg_processing_time_ms': float(result.avg_processing_time or 0),
                'rated_answers': result.rated_answers,
                'avg_rating': float(result.avg_rating or 0)
            }
        return {}

    @classmethod
    def record_interaction(
            cls,
            user_id,
            question,
            answer,
            session,
            question_embedding=None,
            answer_embedding=None,
            processing_time_ms=None,
            request_id=None,  # NEW
            raw_response=None  # NEW
    ):
        """
        Enhanced interaction recording with embeddings, metadata, and request tracking.
        """
        try:
            timestamp = datetime.utcnow().isoformat()
            qa_record = cls(
                user_id=user_id,
                question=question,
                answer=answer,
                timestamp=timestamp
            )

            # Add embeddings if provided
            if question_embedding is not None:
                qa_record.question_embedding = question_embedding
            if answer_embedding is not None:
                qa_record.answer_embedding = answer_embedding
            if processing_time_ms is not None:
                qa_record.processing_time_ms = processing_time_ms

            # NEW: add request_id + raw_response
            if request_id:
                qa_record.request_id = request_id
            if raw_response:
                qa_record.raw_response = raw_response

            session.add(qa_record)
            session.commit()
            logger.debug(f"Recorded interaction {qa_record.id} for user {user_id} (req={request_id})")
            return qa_record

        except Exception as e:
            session.rollback()
            logger.error(f"Error recording QandA interaction: {e}", exc_info=True)
            return None

    @classmethod
    def update_embeddings(cls, qa_id, question_embedding=None, answer_embedding=None, db_session=None):
        """
        Update embeddings for an existing Q&A record.

        Args:
            qa_id: The Q&A record ID
            question_embedding: Optional question embedding
            answer_embedding: Optional answer embedding
            db_session: SQLAlchemy session

        Returns:
            Boolean indicating success
        """
        try:
            qa_record = db_session.query(cls).filter_by(id=qa_id).first()
            if qa_record:
                if question_embedding is not None:
                    qa_record.question_embedding = question_embedding
                if answer_embedding is not None:
                    qa_record.answer_embedding = answer_embedding
                db_session.commit()
                logger.debug(f"Embeddings updated for Q&A ID {qa_id}")
                return True
            else:
                logger.error(f"No Q&A record found for ID {qa_id}")
                return False
        except Exception as e:
            db_session.rollback()
            logger.error(f"Error updating embeddings for Q&A ID {qa_id}: {e}")
            return False

    def _record_basic_interaction(self, session, user_id, question, answer):
        """
        Fallback method for basic interaction recording when enhanced columns are missing.
        """
        try:
            from sqlalchemy import text
            import uuid
            from datetime import datetime

            # Use raw SQL for basic recording
            interaction_id = str(uuid.uuid4())
            current_time = datetime.now()

            # Try with basic columns only
            basic_insert = text("""
                INSERT INTO qanda (id, user_id, question, answer, timestamp)
                VALUES (:id, :user_id, :question, :answer, :timestamp)
            """)

            session.execute(basic_insert, {
                'id': interaction_id,
                'user_id': user_id,
                'question': question,
                'answer': answer,
                'timestamp': current_time
            })

            session.commit()
            logger.info("Successfully recorded basic interaction")

        except Exception as e:
            logger.error(f"Failed to record even basic interaction: {e}")
            session.rollback()

class UserLevel(PyEnum):
    ADMIN = 'ADMIN'
    LEVEL_III = 'LEVEL_III'
    LEVEL_II = 'LEVEL_II'
    LEVEL_I = 'LEVEL_I'
    STANDARD = 'STANDARD'

# region Todo: Create and Refactor class's to a new class called ModelsConfig

class ImageModelConfig(Base):
    __tablename__ = 'image_model_config'

    id = Column(Integer, primary_key=True)
    key = Column(String, unique=True, nullable=False)
    value = Column(String, nullable=False)

# endregion

# Models Configuration table


# Define the User model
class User(Base):
    __tablename__ = 'users'

    id = Column(Integer, primary_key=True)
    type = Column(String(50))  # This column is needed for SQLAlchemy inheritance
    employee_id = Column(String, unique=True, nullable=False)
    first_name = Column(String, nullable=False)
    last_name = Column(String, nullable=False)
    current_shift = Column(String, nullable=True)
    primary_area = Column(String, nullable=True)
    age = Column(Integer, nullable=True)
    education_level = Column(String, nullable=True)
    start_date = Column(DateTime, nullable=True)
    hashed_password = Column(String, nullable=False)

    # Store enum as string in the database
    user_level = Column(SqlEnum(UserLevel, values_callable=lambda obj: [e.value for e in obj]),
                        default=UserLevel.STANDARD, nullable=False)

    # Relationship to comments
    comments = relationship("UserComments", back_populates="user")
    logins = relationship('UserLogin', back_populates='user')

    # Add mapper arguments for inheritance
    __mapper_args__ = {
        'polymorphic_identity': 'user',
        'polymorphic_on': type
    }

    def set_password(self, password):
        self.hashed_password = generate_password_hash(password)

    def check_password_hash(self, password):
        return check_password_hash(self.hashed_password, password)

    @classmethod
    @with_request_id
    def create_new_user(cls, employee_id, first_name, last_name, password, current_shift=None,
                        primary_area=None, age=None, education_level=None, start_date=None,
                        text_to_voice="default", voice_to_text="default"):
        """
        Creates a new user with comprehensive error handling and proper session management.
        """
        logger = logging.getLogger('ematac_logger')
        logger.info(f"============ CREATE_NEW_USER STARTED for {employee_id} ============")

        from modules.configuration.config_env import DatabaseConfig
        from sqlalchemy.exc import IntegrityError, SQLAlchemyError
        import traceback

        # Get database session
        try:
            logger.info("Getting database session...")
            db_config = DatabaseConfig()
            session = db_config.get_main_session()
            logger.debug(f"Got database session: {session}")
        except Exception as e:
            logger.error(f"ERROR GETTING DATABASE SESSION: {e}")
            logger.error(traceback.format_exc())
            return False, f"Database connection error: {str(e)}"

        try:
            # Create new user object
            logger.info(f"Creating User object with: {employee_id}, {first_name}, {last_name}")
            new_user = User(
                employee_id=employee_id,
                first_name=first_name,
                last_name=last_name,
                current_shift=current_shift,
                primary_area=primary_area,
                age=age,
                education_level=education_level,
                start_date=start_date,
                user_level=UserLevel.STANDARD
            )
            logger.debug("Created User object successfully")

            # Set password
            logger.debug("Setting password...")
            new_user.set_password(password)
            logger.debug("Password set successfully")

            # Add to session
            logger.debug("Adding user to database session...")
            session.add(new_user)
            logger.debug("User added to session")

            # Commit changes
            logger.info("Committing session...")
            session.commit()
            logger.info("Session committed successfully")
            logger.info(f"User created successfully: {employee_id}")

            return True, "User created successfully"

        except IntegrityError as e:
            logger.error(f"INTEGRITY ERROR: {str(e)}")
            session.rollback()
            error_msg = str(e)
            logger.error(f"IntegrityError creating user: {error_msg}")

            if "UNIQUE constraint failed" in error_msg:
                return False, f"A user with employee ID {employee_id} already exists."
            else:
                return False, f"Database integrity error: {error_msg}"

        except SQLAlchemyError as e:
            logger.error(f"SQL ALCHEMY ERROR: {str(e)}")
            session.rollback()
            error_msg = str(e)
            logger.error(f"SQLAlchemy error creating user: {error_msg}")
            return False, f"Database error: {error_msg}"

        except Exception as e:
            logger.error(f"UNEXPECTED ERROR: {str(e)}")
            logger.error(traceback.format_exc())
            session.rollback()
            error_msg = str(e)
            logger.error(f"Unexpected error creating user: {error_msg}")
            return False, f"An unexpected error occurred: {error_msg}"

        finally:
            # Always close the session
            try:
                logger.debug("Closing database session")
                session.close()
                logger.debug("Database session closed")
            except Exception as e:
                logger.error(f"ERROR CLOSING SESSION: {e}")

            logger.info(f"============ CREATE_NEW_USER FINISHED for {employee_id} ============")

class UserLogin(Base):
    __tablename__ = 'user_logins'

    id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(Integer, ForeignKey('users.id'), nullable=False)
    session_id = Column(String, nullable=False)  # Flask session ID
    ip_address = Column(String, nullable=True)
    user_agent = Column(String, nullable=True)  # Browser/client info
    login_time = Column(DateTime, default=datetime.utcnow)
    last_activity = Column(DateTime, default=datetime.utcnow)
    logout_time = Column(DateTime, nullable=True)
    is_active = Column(Boolean, default=True)

    # Relationship to User
    user = relationship("User", back_populates="logins")

    def __init__(self, user_id, session_id, ip_address=None, user_agent=None):
        self.user_id = user_id
        self.session_id = session_id
        self.ip_address = ip_address
        self.user_agent = user_agent
        self.login_time = datetime.utcnow()
        self.last_activity = datetime.utcnow()
        self.is_active = True

class KivyUser(User):
    __tablename__ = 'kivy_users'

    id = Column(Integer, ForeignKey('users.id'), primary_key=True)

    # Relationship to layouts
    layouts = relationship("UserLayout", back_populates="user", cascade="all, delete-orphan")

    __mapper_args__ = {
        'polymorphic_identity': 'kivy_user',
    }

    def get_layout(self, layout_name):
        """Get a specific layout by name"""
        from sqlalchemy.orm.session import object_session

        session = object_session(self)
        if not session:
            raise ValueError("User is not attached to a session")

        layout = session.query(UserLayout).filter_by(
            user_id=self.id,
            layout_name=layout_name
        ).first()

        if layout:
            import json
            return json.loads(layout.layout_data)
        return None

    def save_layout(self, layout_name, layout_data):
        """Save a layout with a specific name"""
        from sqlalchemy.orm.session import object_session
        import json

        session = object_session(self)
        if not session:
            raise ValueError("User is not attached to a session")

        layout = session.query(UserLayout).filter_by(
            user_id=self.id,
            layout_name=layout_name
        ).first()

        if layout:
            # Update existing layout
            layout.layout_data = json.dumps(layout_data)
        else:
            # Create new layout
            layout = UserLayout(
                user_id=self.id,
                layout_name=layout_name,
                layout_data=json.dumps(layout_data)
            )
            session.add(layout)

        session.commit()

    def get_all_layouts(self):
        """Get all layouts for this user"""
        from sqlalchemy.orm.session import object_session
        import json

        session = object_session(self)
        if not session:
            raise ValueError("User is not attached to a session")

        layouts = session.query(UserLayout).filter_by(user_id=self.id).all()

        return {layout.layout_name: json.loads(layout.layout_data) for layout in layouts}

    def delete_layout(self, layout_name):
        """Delete a layout by name"""


        session = object_session(self)
        if not session:
            raise ValueError("User is not attached to a session")

        layout = session.query(UserLayout).filter_by(
            user_id=self.id,
            layout_name=layout_name
        ).first()

        if layout:
            session.delete(layout)
            session.commit()
            return True
        return False

    @classmethod
    @with_request_id
    def ensure_kivy_user(cls, session, user_or_id):
        """
        Ensures a KivyUser record exists for a given User or user ID.
        Args:
            session: SQLAlchemy session
            user_or_id: A User instance or user ID
        Returns:
            KivyUser instance if successful, None if failed
        """
        # Import logger and SQLAlchemy text

        logger = logging.getLogger(__name__)

        if user_or_id is None:
            logger.error("Cannot ensure KivyUser for None user")
            return None

        # Get user ID
        user_id = user_or_id.id if hasattr(user_or_id, 'id') else user_or_id

        # Try to get existing KivyUser
        kivy_user = session.query(cls).filter(cls.id == user_id).first()

        if kivy_user:
            logger.debug(f"Found existing KivyUser for ID {user_id}")
            return kivy_user

        # No KivyUser found, check if the User exists and has type='kivy_user'
        user = None
        if hasattr(user_or_id, 'id'):
            user = user_or_id
        else:
            user = session.query(User).filter(User.id == user_id).first()

        if not user:
            logger.error(f"No User found with ID {user_id}")
            return None

        # Check if the user is already marked as a KivyUser
        if user.type != 'kivy_user':
            # Update the user type to 'kivy_user'
            logger.info(f"Updating User {user.employee_id} (ID: {user.id}) type to 'kivy_user'")
            user.type = 'kivy_user'
            try:
                session.commit()
            except SQLAlchemyError as e:
                session.rollback()
                logger.error(f"Error updating User type: {e}")
                return None

        # Create the KivyUser record
        try:
            logger.info(f"Creating KivyUser record for User {user.employee_id} (ID: {user.id})")
            session.execute(
                text("INSERT INTO kivy_users (id) VALUES (:id)"),
                {"id": user.id}
            )
            session.commit()

            # Fetch the newly created KivyUser
            kivy_user = session.query(cls).filter(cls.id == user.id).first()

            if kivy_user:
                logger.info(f"Successfully created KivyUser for {user.employee_id} (ID: {user.id})")
                return kivy_user
            else:
                logger.error(f"Failed to retrieve created KivyUser for {user.employee_id} (ID: {user.id})")
                return None

        except SQLAlchemyError as e:
            session.rollback()
            logger.error(f"Error creating KivyUser record: {e}")
            return None

class UserLayout(Base):
    __tablename__ = 'user_layouts'

    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, ForeignKey('kivy_users.id'), nullable=False)
    layout_name = Column(String, nullable=False)
    layout_data = Column(Text, nullable=False)  # Store JSON layout data

    # Relationship to KivyUser
    user = relationship("KivyUser", back_populates="layouts")

    # Create a unique constraint to prevent duplicate layout names for a user
    __table_args__ = (
        UniqueConstraint('user_id', 'layout_name', name='uix_user_layout_name'),
    )

# Define the UserComments model
class UserComments(Base):
    __tablename__ = 'user_comments'

    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, ForeignKey('users.id'), nullable=True)
    comment = Column(Text, nullable=False)
    page_url = Column(String, nullable=False)
    screenshot_path = Column(String, nullable=True)
    timestamp = Column(DateTime, default=datetime.utcnow)

    # Relationship to User
    user = relationship("User", back_populates="comments")

class BOMResult(Base):
    __tablename__ = 'bom_result'
    id = Column(Integer, primary_key=True)
    part_id = Column(Integer, ForeignKey('part.id'), nullable=False)
    position_id = Column(Integer, ForeignKey('position.id'), nullable=False)
    image_id = Column(Integer, ForeignKey('image.id'), nullable=True)
    description = Column(String)

    part = relationship('Part', lazy='joined')
    image = relationship('Image', lazy='joined')

#class's dealing with tools
class ToolImageAssociation(Base):
    __tablename__ = 'tool_image_association'
    id = Column(Integer, primary_key=True)
    tool_id = Column(Integer, ForeignKey('tool.id'))
    image_id = Column(Integer, ForeignKey('image.id'))
    description = Column(Text, nullable=True)

    # Relationships
    tool = relationship('Tool', back_populates='tool_image_association')
    image = relationship('Image', back_populates='tool_image_association')

    @classmethod
    @with_request_id
    def associate_with_tool(cls, session, image_id, tool_id, description=None, request_id=None):
        """Associate an existing image with a tool in the database.

        Args:
            session: The database session
            image_id: ID of the existing image to associate
            tool_id: ID of the tool to associate with the image
            description: Optional description for this specific association
            request_id: Optional request ID for logging

        Returns:
            The created ToolImageAssociation object or existing one if found, or None on error
        """
        rid = request_id or get_request_id()

        info_id(f"Associating image ID {image_id} with tool ID {tool_id}", rid)

        try:
            # Check if association already exists
            existing_association = session.query(cls).filter(
                and_(
                    cls.image_id == image_id,
                    cls.tool_id == tool_id
                )
            ).first()

            if existing_association is not None:
                info_id(f"Association already exists between image ID {image_id} and tool ID {tool_id}", rid)

                # Update description if provided and different
                if description is not None and existing_association.description != description:
                    existing_association.description = description
                    info_id(f"Updated description for existing association", rid)

                return existing_association
            else:
                # Create new association
                info_id(f"Creating new association between image ID {image_id} and tool ID {tool_id}", rid)
                new_association = cls(
                    image_id=image_id,
                    tool_id=tool_id,
                    description=description
                )
                session.add(new_association)
                session.flush()  # Get ID without committing transaction

                info_id(f"Created ToolImageAssociation with ID {new_association.id}", rid)
                return new_association

        except Exception as e:
            error_id(f"Error in associate_with_tool: {e}", rid, exc_info=True)
            try:
                session.rollback()
            except:
                pass
            return None

    @classmethod
    @with_request_id
    def add_and_associate_with_tool(cls, session, title, file_path, tool_id, description="",
                                    association_description=None, request_id=None):
        """Add an image to the database and associate it with a tool in one operation.

        Args:
            session: The database session
            title: Title for the image
            file_path: Path to the image file
            tool_id: ID of the tool to associate with the image
            description: Description for the image itself
            association_description: Optional description for the tool-image association
            request_id: Optional request ID for logging

        Returns:
            Tuple of (Image object, ToolImageAssociation object) or (None, None) on error
        """
        rid = request_id or get_request_id()

        try:
            info_id(f"Starting add_and_associate_with_tool for '{title}' with tool ID {tool_id}", rid)

            # First add the image to the database - this returns just the ID (integer)
            created_image_id = Image.add_to_db(session, title, file_path, description, request_id=rid)

            if created_image_id is None:
                error_id(f"Failed to create image '{title}'", rid)
                return None, None

            info_id(f"Successfully created image with ID: {created_image_id}", rid)

            # Get the actual Image object from the database
            image_object = session.query(Image).filter(Image.id == created_image_id).first()
            if image_object is None:
                error_id(f"Could not retrieve created image with ID {created_image_id}", rid)
                return None, None

            debug_id(f"Successfully retrieved image object: '{image_object.title}'", rid)

            # Then create the association using the image ID (integer)
            association = cls.associate_with_tool(
                session,
                image_id=created_image_id,  # Use the integer ID directly
                tool_id=tool_id,
                description=association_description,
                request_id=rid
            )

            if association is None:
                error_id(f"Failed to create tool association for image ID {created_image_id}", rid)
                return image_object, None

            info_id(
                f"Successfully created image '{title}' (ID: {created_image_id}) and associated with tool ID {tool_id}",
                rid)
            return image_object, association

        except Exception as e:
            error_id(f"Error in add_and_associate_with_tool: {e}", rid, exc_info=True)
            try:
                session.rollback()
            except:
                pass
            return None, None

    # Additional helper methods for better tool-image management

    @classmethod
    @with_request_id
    def get_tools_for_image(cls, session, image_id, request_id=None):
        """Get all tools associated with a specific image.

        Args:
            session: Database session
            image_id: ID of the image
            request_id: Optional request ID for logging

        Returns:
            List of dictionaries containing tool information
        """
        rid = request_id or get_request_id()

        try:
            associations = session.query(cls).filter(cls.image_id == image_id).all()
            tools = []

            for assoc in associations:
                if assoc.tool:  # Use relationship if available
                    tools.append({
                        'tool_id': assoc.tool_id,
                        'tool_name': assoc.tool.name if hasattr(assoc.tool, 'name') else 'Unknown',
                        'association_description': assoc.description,
                        'association_id': assoc.id
                    })

            debug_id(f"Found {len(tools)} tools for image {image_id}", rid)
            return tools

        except Exception as e:
            error_id(f"Error getting tools for image {image_id}: {e}", rid)
            return []

    @classmethod
    @with_request_id
    def get_images_for_tool(cls, session, tool_id, request_id=None):
        """Get all images associated with a specific tool.

        Args:
            session: Database session
            tool_id: ID of the tool
            request_id: Optional request ID for logging

        Returns:
            List of dictionaries containing image information
        """
        rid = request_id or get_request_id()

        try:
            associations = session.query(cls).filter(cls.tool_id == tool_id).all()
            images = []

            for assoc in associations:
                if assoc.image:  # Use relationship if available
                    images.append({
                        'image_id': assoc.image_id,
                        'image_title': assoc.image.title,
                        'image_description': assoc.image.description,
                        'image_path': assoc.image.file_path,
                        'association_description': assoc.description,
                        'association_id': assoc.id,
                        'view_url': f'/add_document/image/{assoc.image_id}'
                    })

            debug_id(f"Found {len(images)} images for tool {tool_id}", rid)
            return images

        except Exception as e:
            error_id(f"Error getting images for tool {tool_id}: {e}", rid)
            return []

    @with_request_id
    def remove_association(self, session, request_id=None):
        """Remove this tool-image association.

        Args:
            session: Database session
            request_id: Optional request ID for logging

        Returns:
            Boolean indicating success
        """
        rid = request_id or get_request_id()

        try:
            info_id(f"Removing tool-image association ID {self.id} (tool: {self.tool_id}, image: {self.image_id})", rid)
            session.delete(self)
            session.flush()
            info_id(f"Successfully removed association ID {self.id}", rid)
            return True

        except Exception as e:
            error_id(f"Error removing association {self.id}: {e}", rid)
            try:
                session.rollback()
            except:
                pass
            return False

    @classmethod
    @with_request_id
    def bulk_associate_images_with_tool(cls, session, image_ids, tool_id, description=None, request_id=None):
        """Associate multiple images with a single tool.

        Args:
            session: Database session
            image_ids: List of image IDs to associate
            tool_id: ID of the tool
            description: Optional description for all associations
            request_id: Optional request ID for logging

        Returns:
            List of created ToolImageAssociation objects
        """
        rid = request_id or get_request_id()

        try:
            info_id(f"Bulk associating {len(image_ids)} images with tool ID {tool_id}", rid)

            associations = []
            for image_id in image_ids:
                association = cls.associate_with_tool(
                    session, image_id, tool_id, description, request_id=rid
                )
                if association:
                    associations.append(association)

            info_id(f"Successfully created {len(associations)} associations", rid)
            return associations

        except Exception as e:
            error_id(f"Error in bulk_associate_images_with_tool: {e}", rid)
            return []

class ToolPositionAssociation(Base):
    __tablename__ = 'tool_position_association'
    id = Column(Integer, primary_key=True)
    tool_id = Column(Integer, ForeignKey('tool.id'))
    position_id = Column(Integer, ForeignKey('position.id'))
    description = Column(Text, nullable=True)
    tool = relationship('Tool', back_populates='tool_position_association')
    position = relationship('Position', back_populates='tool_position_association')

class ToolCategory(Base):
    __tablename__ = 'tool_category'
    id = Column(Integer, primary_key=True)
    name = Column(String, nullable=False, unique=True)
    description = Column(Text)
    parent_id = Column(Integer, ForeignKey('tool_category.id'), nullable=True)

    # Self-referential relationships for hierarchy
    parent = relationship('ToolCategory', remote_side=[id], back_populates='subcategories')
    subcategories = relationship('ToolCategory', back_populates='parent', cascade="all, delete-orphan")
    tools = relationship('Tool', back_populates='tool_category', cascade="all, delete-orphan")

tool_package_association = Table(
    'tool_package_association',
    Base.metadata,
    Column('tool_id', Integer, ForeignKey('tool.id'), primary_key=True),
    Column('package_id', Integer, ForeignKey('tool_package.id'), primary_key=True),
    Column('quantity', Integer, nullable=False, default=1)
)

class ToolManufacturer(Base):
    __tablename__ = 'tool_manufacturer'
    id = Column(Integer, primary_key=True)
    name = Column(String, nullable=False, unique=True)
    description =Column(String, nullable=True)
    country = Column(String, nullable=True)
    website = Column(String, nullable=True)

    tools = relationship('Tool', back_populates='tool_manufacturer')

class Tool(Base):
    __tablename__ = 'tool'
    id = Column(Integer, primary_key=True)
    name = Column(String, nullable=False)
    size = Column(String, nullable=True)
    type = Column(String, nullable=True)
    material = Column(String, nullable=True)
    description = Column(Text)
    tool_category_id = Column(Integer, ForeignKey('tool_category.id'))
    tool_manufacturer_id = Column(Integer, ForeignKey('tool_manufacturer.id'))

    # Relationships
    tool_category = relationship('ToolCategory', back_populates='tools')
    tool_manufacturer = relationship('ToolManufacturer', back_populates='tools')
    tool_packages = relationship('ToolPackage', secondary=tool_package_association, back_populates='tools')
    tool_image_association = relationship('ToolImageAssociation', back_populates='tool')
    tool_position_association = relationship('ToolPositionAssociation',back_populates='tool',)
    tool_tasks = relationship('TaskToolAssociation', back_populates='tool', cascade="all, delete-orphan")


# ===========================================
# TOOL MANAGER CLASS (Business Logic)
# ===========================================

class ToolManager:
    """
    Comprehensive tool management class providing search, add, and delete operations.
    Integrates with existing database configuration and logging system.
    """

    def __init__(self, db_config: DatabaseConfig):
        """
        Initialize the ToolManager with database configuration.

        Args:
            db_config: DatabaseConfig instance for database operations
        """
        self.db_config = db_config
        self.request_id = get_request_id()
        logger.info(f"ToolManager initialized with request ID: {self.request_id}")

    # ===================
    # SEARCH OPERATIONS
    # ===================

    @with_request_id
    def search_tools(self,
                     name: Optional[str] = None,
                     category_id: Optional[int] = None,
                     category_name: Optional[str] = None,
                     manufacturer_id: Optional[int] = None,
                     manufacturer_name: Optional[str] = None,
                     tool_type: Optional[str] = None,
                     material: Optional[str] = None,
                     size: Optional[str] = None,
                     description_contains: Optional[str] = None,
                     include_relationships: bool = True,
                     limit: Optional[int] = None,
                     offset: Optional[int] = 0) -> List[Tool]:
        """
        Search for tools with various filter criteria.

        Args:
            name: Partial or exact tool name match
            category_id: Filter by specific category ID
            category_name: Filter by category name (partial match)
            manufacturer_id: Filter by specific manufacturer ID
            manufacturer_name: Filter by manufacturer name (partial match)
            tool_type: Filter by tool type
            material: Filter by material
            size: Filter by size
            description_contains: Search in description text
            include_relationships: Whether to eagerly load related data
            limit: Maximum number of results to return
            offset: Number of results to skip (for pagination)

        Returns:
            List of Tool objects matching the criteria
        """
        try:
            with self.db_config.main_session() as session:
                # Start with base query
                query = session.query(Tool)

                # Add eager loading for relationships if requested
                if include_relationships:
                    query = query.options(
                        joinedload(Tool.tool_category),
                        joinedload(Tool.tool_manufacturer),
                        selectinload(Tool.tool_packages),
                        selectinload(Tool.tool_image_association),
                        selectinload(Tool.tool_position_association),
                        selectinload(Tool.tool_tasks)
                    )

                # Build filter conditions
                conditions = []

                # Name filter (case-insensitive partial match)
                if name:
                    conditions.append(Tool.name.ilike(f'%{name}%'))

                # Category filters
                if category_id:
                    conditions.append(Tool.tool_category_id == category_id)
                elif category_name:
                    query = query.join(ToolCategory)
                    conditions.append(ToolCategory.name.ilike(f'%{category_name}%'))

                # Manufacturer filters
                if manufacturer_id:
                    conditions.append(Tool.tool_manufacturer_id == manufacturer_id)
                elif manufacturer_name:
                    query = query.join(ToolManufacturer)
                    conditions.append(ToolManufacturer.name.ilike(f'%{manufacturer_name}%'))

                # Type filter
                if tool_type:
                    conditions.append(Tool.type.ilike(f'%{tool_type}%'))

                # Material filter
                if material:
                    conditions.append(Tool.material.ilike(f'%{material}%'))

                # Size filter
                if size:
                    conditions.append(Tool.size.ilike(f'%{size}%'))

                # Description filter
                if description_contains:
                    conditions.append(Tool.description.ilike(f'%{description_contains}%'))

                # Apply all conditions
                if conditions:
                    query = query.filter(and_(*conditions))

                # Apply pagination
                if offset:
                    query = query.offset(offset)
                if limit:
                    query = query.limit(limit)

                # Execute query
                tools = query.all()

                logger.info(f"Search found {len(tools)} tools matching criteria")
                return tools

        except SQLAlchemyError as e:
            logger.error(f"Database error during tool search: {e}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error during tool search: {e}")
            raise

    @with_request_id
    def get_tool_by_id(self, tool_id: int, include_relationships: bool = True) -> Optional[Tool]:
        """
        Get a specific tool by its ID.

        Args:
            tool_id: The ID of the tool to retrieve
            include_relationships: Whether to eagerly load related data

        Returns:
            Tool object if found, None otherwise
        """
        try:
            with self.db_config.main_session() as session:
                query = session.query(Tool).filter(Tool.id == tool_id)

                if include_relationships:
                    query = query.options(
                        joinedload(Tool.tool_category),
                        joinedload(Tool.tool_manufacturer),
                        selectinload(Tool.tool_packages),
                        selectinload(Tool.tool_image_association),
                        selectinload(Tool.tool_position_association),
                        selectinload(Tool.tool_tasks)
                    )

                tool = query.first()

                if tool:
                    logger.info(f"Found tool: {tool.name} (ID: {tool_id})")
                else:
                    logger.warning(f"Tool with ID {tool_id} not found")

                return tool

        except SQLAlchemyError as e:
            logger.error(f"Database error getting tool by ID {tool_id}: {e}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error getting tool by ID {tool_id}: {e}")
            raise

    @with_request_id
    def search_tools_full_text(self, search_term: str, limit: Optional[int] = 50) -> List[Tool]:
        """
        Perform full-text search across tool name, type, material, and description.

        Args:
            search_term: Text to search for
            limit: Maximum number of results

        Returns:
            List of Tool objects matching the search term
        """
        try:
            with self.db_config.main_session() as session:
                # Create a comprehensive text search across multiple fields
                search_pattern = f'%{search_term}%'

                query = session.query(Tool).options(
                    joinedload(Tool.tool_category),
                    joinedload(Tool.tool_manufacturer)
                ).filter(
                    or_(
                        Tool.name.ilike(search_pattern),
                        Tool.type.ilike(search_pattern),
                        Tool.material.ilike(search_pattern),
                        Tool.description.ilike(search_pattern)
                    )
                )

                if limit:
                    query = query.limit(limit)

                tools = query.all()
                logger.info(f"Full-text search for '{search_term}' found {len(tools)} tools")
                return tools

        except SQLAlchemyError as e:
            logger.error(f"Database error during full-text search: {e}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error during full-text search: {e}")
            raise

    @with_request_id
    def get_tools_by_category(self, category_id: int, include_subcategories: bool = True) -> List[Tool]:
        """
        Get all tools in a specific category.

        Args:
            category_id: The category ID
            include_subcategories: Whether to include tools from subcategories

        Returns:
            List of tools in the category
        """
        try:
            with self.db_config.main_session() as session:
                if include_subcategories:
                    # Get the category and all its subcategories
                    category_ids = self._get_category_and_subcategory_ids(session, category_id)
                    tools = session.query(Tool).options(
                        joinedload(Tool.tool_category),
                        joinedload(Tool.tool_manufacturer)
                    ).filter(Tool.tool_category_id.in_(category_ids)).all()
                else:
                    # Get tools only from the specific category
                    tools = session.query(Tool).options(
                        joinedload(Tool.tool_category),
                        joinedload(Tool.tool_manufacturer)
                    ).filter(Tool.tool_category_id == category_id).all()

                logger.info(f"Found {len(tools)} tools in category {category_id}")
                return tools

        except SQLAlchemyError as e:
            logger.error(f"Database error getting tools by category: {e}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error getting tools by category: {e}")
            raise

    # ===================
    # ADD OPERATIONS
    # ===================

    @with_request_id
    def add_tool(self,
                 name: str,
                 tool_category_id: int,
                 tool_manufacturer_id: int,
                 size: Optional[str] = None,
                 tool_type: Optional[str] = None,
                 material: Optional[str] = None,
                 description: Optional[str] = None,
                 package_ids: Optional[List[int]] = None) -> Tool:
        """
        Add a new tool to the database.

        Args:
            name: Tool name
            tool_category_id: Category ID (must exist)
            tool_manufacturer_id: Manufacturer ID (must exist)
            size: Tool size specification
            tool_type: Type of tool
            material: Material composition
            description: Detailed description
            package_ids: List of package IDs to associate with this tool

        Returns:
            The created Tool object

        Raises:
            ValueError: If required references don't exist
            IntegrityError: If database constraints are violated
        """
        try:
            with self.db_config.main_session() as session:
                # Validate that category and manufacturer exist
                category = session.query(ToolCategory).filter(
                    ToolCategory.id == tool_category_id
                ).first()
                if not category:
                    raise ValueError(f"Tool category with ID {tool_category_id} does not exist")

                manufacturer = session.query(ToolManufacturer).filter(
                    ToolManufacturer.id == tool_manufacturer_id
                ).first()
                if not manufacturer:
                    raise ValueError(f"Tool manufacturer with ID {tool_manufacturer_id} does not exist")

                # Create the new tool
                new_tool = Tool(
                    name=name,
                    size=size,
                    type=tool_type,
                    material=material,
                    description=description,
                    tool_category_id=tool_category_id,
                    tool_manufacturer_id=tool_manufacturer_id
                )

                session.add(new_tool)
                session.flush()  # Get the ID without committing

                # Add package associations if provided
                if package_ids:
                    self._add_tool_package_associations(session, new_tool.id, package_ids)

                session.commit()

                # Reload with relationships
                created_tool = self.get_tool_by_id(new_tool.id)

                logger.info(f"Successfully created tool: {name} (ID: {new_tool.id})")
                return created_tool

        except ValueError as e:
            logger.error(f"Validation error creating tool: {e}")
            raise
        except IntegrityError as e:
            logger.error(f"Database integrity error creating tool: {e}")
            raise
        except SQLAlchemyError as e:
            logger.error(f"Database error creating tool: {e}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error creating tool: {e}")
            raise

    @with_request_id
    def add_tool_from_dict(self, tool_data: Dict[str, Any]) -> Tool:
        """
        Add a tool from a dictionary of data.

        Args:
            tool_data: Dictionary containing tool information

        Returns:
            The created Tool object
        """
        required_fields = ['name', 'tool_category_id', 'tool_manufacturer_id']

        # Validate required fields
        for field in required_fields:
            if field not in tool_data:
                raise ValueError(f"Required field '{field}' is missing from tool data")

        return self.add_tool(
            name=tool_data['name'],
            tool_category_id=tool_data['tool_category_id'],
            tool_manufacturer_id=tool_data['tool_manufacturer_id'],
            size=tool_data.get('size'),
            tool_type=tool_data.get('type'),
            material=tool_data.get('material'),
            description=tool_data.get('description'),
            package_ids=tool_data.get('package_ids')
        )

    # ===================
    # UPDATE OPERATIONS
    # ===================

    @with_request_id
    def update_tool(self,
                    tool_id: int,
                    name: Optional[str] = None,
                    size: Optional[str] = None,
                    tool_type: Optional[str] = None,
                    material: Optional[str] = None,
                    description: Optional[str] = None,
                    tool_category_id: Optional[int] = None,
                    tool_manufacturer_id: Optional[int] = None,
                    package_ids: Optional[List[int]] = None) -> Optional[Tool]:
        """
        Update an existing tool.

        Args:
            tool_id: ID of the tool to update
            name: New name (if provided)
            size: New size (if provided)
            tool_type: New type (if provided)
            material: New material (if provided)
            description: New description (if provided)
            tool_category_id: New category ID (if provided)
            tool_manufacturer_id: New manufacturer ID (if provided)
            package_ids: New list of package IDs (replaces existing associations)

        Returns:
            Updated Tool object if successful, None if tool not found
        """
        try:
            with self.db_config.main_session() as session:
                tool = session.query(Tool).filter(Tool.id == tool_id).first()

                if not tool:
                    logger.warning(f"Tool with ID {tool_id} not found for update")
                    return None

                # Validate references if provided
                if tool_category_id and not session.query(ToolCategory).filter(
                        ToolCategory.id == tool_category_id
                ).first():
                    raise ValueError(f"Tool category with ID {tool_category_id} does not exist")

                if tool_manufacturer_id and not session.query(ToolManufacturer).filter(
                        ToolManufacturer.id == tool_manufacturer_id
                ).first():
                    raise ValueError(f"Tool manufacturer with ID {tool_manufacturer_id} does not exist")

                # Update fields if provided
                if name is not None:
                    tool.name = name
                if size is not None:
                    tool.size = size
                if tool_type is not None:
                    tool.type = tool_type
                if material is not None:
                    tool.material = material
                if description is not None:
                    tool.description = description
                if tool_category_id is not None:
                    tool.tool_category_id = tool_category_id
                if tool_manufacturer_id is not None:
                    tool.tool_manufacturer_id = tool_manufacturer_id

                # Update package associations if provided
                if package_ids is not None:
                    # Remove existing associations
                    session.execute(
                        tool_package_association.delete().where(
                            tool_package_association.c.tool_id == tool_id
                        )
                    )
                    # Add new associations
                    if package_ids:
                        self._add_tool_package_associations(session, tool_id, package_ids)

                session.commit()

                # Reload with relationships
                updated_tool = self.get_tool_by_id(tool_id)

                logger.info(f"Successfully updated tool: {tool.name} (ID: {tool_id})")
                return updated_tool

        except ValueError as e:
            logger.error(f"Validation error updating tool: {e}")
            raise
        except SQLAlchemyError as e:
            logger.error(f"Database error updating tool: {e}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error updating tool: {e}")
            raise

    # ===================
    # DELETE OPERATIONS
    # ===================

    @with_request_id
    def delete_tool(self, tool_id: int, force: bool = False) -> bool:
        """
        Delete a tool from the database.

        Args:
            tool_id: ID of the tool to delete
            force: If True, will delete even if tool has dependencies

        Returns:
            True if deletion was successful, False if tool not found

        Raises:
            ValueError: If tool has dependencies and force=False
        """
        try:
            with self.db_config.main_session() as session:
                tool = session.query(Tool).filter(Tool.id == tool_id).first()

                if not tool:
                    logger.warning(f"Tool with ID {tool_id} not found for deletion")
                    return False

                tool_name = tool.name  # Store for logging

                # Check for dependencies if not forcing
                if not force:
                    dependencies = self._check_tool_dependencies(session, tool_id)
                    if dependencies:
                        raise ValueError(
                            f"Tool '{tool_name}' has dependencies: {', '.join(dependencies)}. "
                            f"Use force=True to delete anyway."
                        )

                # Delete the tool (cascading relationships will be handled automatically)
                session.delete(tool)
                session.commit()

                logger.info(f"Successfully deleted tool: {tool_name} (ID: {tool_id})")
                return True

        except ValueError as e:
            logger.error(f"Validation error deleting tool: {e}")
            raise
        except SQLAlchemyError as e:
            logger.error(f"Database error deleting tool: {e}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error deleting tool: {e}")
            raise

    @with_request_id
    def delete_tools_by_category(self, category_id: int, force: bool = False) -> int:
        """
        Delete all tools in a specific category.

        Args:
            category_id: Category ID
            force: If True, will delete even if tools have dependencies

        Returns:
            Number of tools deleted
        """
        try:
            tools = self.get_tools_by_category(category_id, include_subcategories=False)
            deleted_count = 0

            for tool in tools:
                if self.delete_tool(tool.id, force=force):
                    deleted_count += 1

            logger.info(f"Deleted {deleted_count} tools from category {category_id}")
            return deleted_count

        except Exception as e:
            logger.error(f"Error deleting tools by category: {e}")
            raise

    @with_request_id
    def delete_tools_by_manufacturer(self, manufacturer_id: int, force: bool = False) -> int:
        """
        Delete all tools from a specific manufacturer.

        Args:
            manufacturer_id: Manufacturer ID
            force: If True, will delete even if tools have dependencies

        Returns:
            Number of tools deleted
        """
        try:
            with self.db_config.main_session() as session:
                tools = session.query(Tool).filter(
                    Tool.tool_manufacturer_id == manufacturer_id
                ).all()

                deleted_count = 0
                for tool in tools:
                    if self.delete_tool(tool.id, force=force):
                        deleted_count += 1

                logger.info(f"Deleted {deleted_count} tools from manufacturer {manufacturer_id}")
                return deleted_count

        except Exception as e:
            logger.error(f"Error deleting tools by manufacturer: {e}")
            raise

    # ===================
    # UTILITY METHODS
    # ===================

    def _get_category_and_subcategory_ids(self, session, category_id: int) -> List[int]:
        """Get a category ID and all its subcategory IDs recursively."""
        category_ids = [category_id]

        # Get direct subcategories
        subcategories = session.query(ToolCategory).filter(
            ToolCategory.parent_id == category_id
        ).all()

        # Recursively get subcategories of subcategories
        for subcategory in subcategories:
            category_ids.extend(
                self._get_category_and_subcategory_ids(session, subcategory.id)
            )

        return category_ids

    def _add_tool_package_associations(self, session, tool_id: int, package_ids: List[int]):
        """Add tool-package associations."""
        for package_id in package_ids:
            # Validate package exists
            package = session.query(ToolPackage).filter(
                ToolPackage.id == package_id
            ).first()
            if not package:
                raise ValueError(f"Tool package with ID {package_id} does not exist")

            # Add association
            association = tool_package_association.insert().values(
                tool_id=tool_id,
                package_id=package_id,
                quantity=1  # Default quantity
            )
            session.execute(association)

    def _check_tool_dependencies(self, session, tool_id: int) -> List[str]:
        """Check if a tool has dependencies that would prevent deletion."""
        dependencies = []

        # Check tool packages
        package_count = session.query(func.count()).select_from(
            tool_package_association
        ).filter(tool_package_association.c.tool_id == tool_id).scalar()
        if package_count > 0:
            dependencies.append(f"{package_count} package associations")

        # Check tool images
        if hasattr(self, 'ToolImageAssociation'):
            image_count = session.query(ToolImageAssociation).filter(
                ToolImageAssociation.tool_id == tool_id
            ).count()
            if image_count > 0:
                dependencies.append(f"{image_count} image associations")

        # Check tool positions
        if hasattr(self, 'ToolPositionAssociation'):
            position_count = session.query(ToolPositionAssociation).filter(
                ToolPositionAssociation.tool_id == tool_id
            ).count()
            if position_count > 0:
                dependencies.append(f"{position_count} position associations")

        # Check tool tasks
        if hasattr(self, 'TaskToolAssociation'):
            task_count = session.query(TaskToolAssociation).filter(
                TaskToolAssociation.tool_id == tool_id
            ).count()
            if task_count > 0:
                dependencies.append(f"{task_count} task associations")

        return dependencies

    # ===================
    # STATISTICS AND REPORTING
    # ===================

    @with_request_id
    def get_tool_statistics(self) -> Dict[str, Any]:
        """Get comprehensive statistics about tools in the database."""
        try:
            with self.db_config.main_session() as session:
                stats = {}

                # Total tool count
                stats['total_tools'] = session.query(Tool).count()

                # Tools by category
                category_stats = session.query(
                    ToolCategory.name,
                    func.count(Tool.id).label('count')
                ).join(Tool).group_by(ToolCategory.name).all()
                stats['tools_by_category'] = {name: count for name, count in category_stats}

                # Tools by manufacturer
                manufacturer_stats = session.query(
                    ToolManufacturer.name,
                    func.count(Tool.id).label('count')
                ).join(Tool).group_by(ToolManufacturer.name).all()
                stats['tools_by_manufacturer'] = {name: count for name, count in manufacturer_stats}

                # Tools by type
                type_stats = session.query(
                    Tool.type,
                    func.count(Tool.id).label('count')
                ).filter(Tool.type.isnot(None)).group_by(Tool.type).all()
                stats['tools_by_type'] = {tool_type or 'Unknown': count for tool_type, count in type_stats}

                # Tools by material
                material_stats = session.query(
                    Tool.material,
                    func.count(Tool.id).label('count')
                ).filter(Tool.material.isnot(None)).group_by(Tool.material).all()
                stats['tools_by_material'] = {material or 'Unknown': count for material, count in material_stats}

                logger.info(f"Generated tool statistics: {stats['total_tools']} total tools")
                return stats

        except SQLAlchemyError as e:
            logger.error(f"Database error generating tool statistics: {e}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error generating tool statistics: {e}")
            raise

class ToolPackage(Base):
    __tablename__ = 'tool_package'
    id = Column(Integer, primary_key=True)
    name = Column(String, nullable=False)
    description = Column(Text)

    tools = relationship('Tool', secondary=tool_package_association, back_populates='tool_packages')

class KeywordSearch:
    """
    Comprehensive class for handling keyword-triggered searches in the chatbot.
    """

    # Cache for storing recent search results (limit size in a production environment)
    _search_cache = {}
    _cache_limit = 100  # Maximum number of cached searches

    def __init__(self, session=None):
        """
        Initialize the KeywordSearch class.

        Args:
            session: SQLAlchemy session (optional)
        """
        self._session = session
        self._db_config = DatabaseConfig()

    @property
    def session(self):
        """Get or create a database session if needed."""
        if self._session is None:
            self._session = self._db_config.get_main_session()
        return self._session

    def close_session(self):
        """Close the session if it was created by this class."""
        if self._session is not None and self._db_config is not None:
            self._session.close()
            self._session = None

    def __enter__(self):
        """Support for context manager usage."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Close session when exiting context."""
        self.close_session()

    # ======== Keyword Management Methods ========

    def register_keyword(self, keyword: str, action_type: str, search_pattern: str = None,
                         entity_type: str = None, description: str = None) -> Dict[str, Any]:
        """
        Register a new keyword with its associated action type and search pattern.

        Args:
            keyword: The keyword or phrase to match
            action_type: Type of action (e.g., 'image_search', 'part_search')
            search_pattern: Pattern to extract parameters (e.g., "show {equipment} in {area}")
            entity_type: Type of entity to search (e.g., 'image', 'part', 'drawing')
            description: Description of what this keyword does

        Returns:
            Dictionary with registration status
        """
        try:
            existing = self.session.query(KeywordAction).filter_by(keyword=keyword).first()

            action_data = {
                "type": action_type,
                "search_pattern": search_pattern,
                "entity_type": entity_type,
                "description": description,
                "created_at": datetime.utcnow().isoformat()
            }

            if existing:
                existing.action = json.dumps(action_data)
                self.session.commit()
                logger.info(f"Updated existing keyword: {keyword}")
                return {"status": "updated", "keyword": keyword}

            new_keyword = KeywordAction(
                keyword=keyword,
                action=json.dumps(action_data)
            )

            self.session.add(new_keyword)
            self.session.commit()
            logger.info(f"Registered new keyword: {keyword}")
            return {"status": "created", "keyword": keyword}

        except Exception as e:
            self.session.rollback()
            logger.error(f"Error registering keyword '{keyword}': {e}")
            return {"status": "error", "message": str(e)}

    def delete_keyword(self, keyword: str) -> Dict[str, Any]:
        """
        Delete a registered keyword.

        Args:
            keyword: The keyword to delete

        Returns:
            Dictionary with deletion status
        """
        try:
            keyword_entry = self.session.query(KeywordAction).filter_by(keyword=keyword).first()
            if not keyword_entry:
                return {"status": "not_found", "keyword": keyword}

            self.session.delete(keyword_entry)
            self.session.commit()
            logger.info(f"Deleted keyword: {keyword}")
            return {"status": "deleted", "keyword": keyword}

        except Exception as e:
            self.session.rollback()
            logger.error(f"Error deleting keyword '{keyword}': {e}")
            return {"status": "error", "message": str(e)}

    def get_all_keywords(self) -> List[Dict[str, Any]]:
        """
        Get all registered keywords and their actions.

        Returns:
            List of dictionaries with keyword information
        """
        try:
            keywords = self.session.query(KeywordAction).all()
            result = []

            for kw in keywords:
                try:
                    action_data = json.loads(kw.action)
                    result.append({
                        "id": kw.id,
                        "keyword": kw.keyword,
                        "action_type": action_data.get("type"),
                        "search_pattern": action_data.get("search_pattern"),
                        "entity_type": action_data.get("entity_type"),
                        "description": action_data.get("description")
                    })
                except json.JSONDecodeError:
                    # Handle legacy action format
                    result.append({
                        "id": kw.id,
                        "keyword": kw.keyword,
                        "action": kw.action
                    })

            return result

        except Exception as e:
            logger.error(f"Error retrieving keywords: {e}")
            return []

    # ======== Pattern Matching Methods ========

    def match_pattern(self, pattern: str, text: str) -> Optional[Dict[str, str]]:
        """
        Match a pattern against text and extract parameters.

        Args:
            pattern: The pattern with {param} placeholders
            text: The text to match against

        Returns:
            Dictionary of extracted parameters or None if no match
        """
        if not pattern:
            return None

        # Convert pattern like "show {equipment} in {area}" to regex
        pattern_regex = pattern.replace("{", "(?P<").replace("}", ">.*?)")
        match = re.match(pattern_regex, text, re.IGNORECASE)

        if match:
            params = match.groupdict()
            # Clean up parameters - remove extra spaces and make lowercase for matching
            return {k: v.strip().lower() if v else v for k, v in params.items()}

        return None

    def extract_search_parameters(self, user_input: str, action_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract search parameters from user input based on action data.

        Args:
            user_input: User's input text
            action_data: Action data containing search pattern

        Returns:
            Dictionary of extracted parameters and search metadata
        """
        search_pattern = action_data.get("search_pattern")
        entity_type = action_data.get("entity_type")

        params = {}

        # Try to match pattern if available
        if search_pattern:
            matched_params = self.match_pattern(search_pattern, user_input)
            if matched_params:
                params.update(matched_params)

        # Add basic search information
        params.update({
            "entity_type": entity_type,
            "action_type": action_data.get("type"),
            "raw_input": user_input,
            "timestamp": datetime.utcnow().isoformat()
        })

        # Extract any ID numbers that might be in the input
        id_matches = re.findall(r'\b(id|number|#)[\s:]*(\d+)\b', user_input, re.IGNORECASE)
        if id_matches:
            for match_type, match_id in id_matches:
                params["extracted_id"] = int(match_id)

        return params

    # ======== Search Execution Methods ========

    def execute_search(self, user_input: str) -> Dict[str, Any]:
        """
        Main entry point for executing a keyword search.

        Args:
            user_input: User's input text

        Returns:
            Dictionary with search results and metadata
        """
        # Check cache first
        cache_key = user_input.lower().strip()
        if cache_key in self._search_cache:
            cached_result = self._search_cache[cache_key]
            # Add cache info to result
            cached_result["from_cache"] = True
            return cached_result

        try:
            # Find matching keyword
            keyword, action, _ = KeywordAction.find_best_match(user_input, self.session)

            if not keyword:
                return {
                    "status": "error",
                    "message": "No matching keyword found",
                    "input": user_input
                }

            # Parse action data
            try:
                action_data = json.loads(action)
            except (json.JSONDecodeError, TypeError):
                # Fallback for legacy format
                action_data = {"type": action}

            action_type = action_data.get("type")

            # Extract search parameters
            params = self.extract_search_parameters(user_input, action_data)

            # Dispatch to appropriate search handler
            result = None

            if action_type == "image_search":
                result = self.search_images(params)
            elif action_type == "part_search":
                result = self.search_parts(params)
            elif action_type == "drawing_search":
                result = self.search_drawings(params)
            elif action_type == "tool_search":
                result = self.search_tools(params)
            elif action_type == "position_search":
                result = self.search_positions(params)
            elif action_type == "problem_search":
                result = self.search_problems(params)
            elif action_type == "task_search":
                result = self.search_tasks(params)
            else:
                return {
                    "status": "error",
                    "message": f"Unknown action type: {action_type}",
                    "input": user_input
                }

            # Add metadata to result
            result.update({
                "keyword": keyword,
                "action_type": action_type,
                "parameters": params,
                "timestamp": datetime.utcnow().isoformat()
            })

            # Cache result
            self._add_to_cache(cache_key, result)

            return result

        except Exception as e:
            logger.error(f"Error executing search: {e}")
            return {
                "status": "error",
                "message": f"Error executing search: {str(e)}",
                "input": user_input
            }

    def _add_to_cache(self, key: str, result: Dict[str, Any]) -> None:
        """Add a result to the search cache with LRU behavior."""
        # Implement simple LRU cache
        if len(self._search_cache) >= self._cache_limit:
            # Remove oldest entry
            oldest_key = next(iter(self._search_cache))
            del self._search_cache[oldest_key]

        # Add to cache
        self._search_cache[key] = result

    # ======== Entity-Specific Search Methods ========

    def search_images(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Search for images based on extracted parameters.

        Args:
            params: Dictionary of search parameters

        Returns:
            Dictionary with search results
        """
        try:
            # Start with base query
            query = self.session.query(Image)

            # Join with Position for hierarchy filters if needed
            if any(k in params for k in ['area', 'equipment', 'model', 'asset', 'location']):
                query = query.join(ImagePositionAssociation, Image.id == ImagePositionAssociation.image_id)
                query = query.join(Position, Position.id == ImagePositionAssociation.position_id)

                # Apply hierarchy filters
                if 'area' in params and params['area']:
                    area_name = params['area']
                    query = query.join(Area, Position.area_id == Area.id)
                    query = query.filter(Area.name.ilike(f"%{area_name}%"))

                if 'equipment' in params and params['equipment']:
                    equipment_name = params['equipment']
                    query = query.join(EquipmentGroup, Position.equipment_group_id == EquipmentGroup.id)
                    query = query.filter(EquipmentGroup.name.ilike(f"%{equipment_name}%"))

                if 'model' in params and params['model']:
                    model_name = params['model']
                    query = query.join(Model, Position.model_id == Model.id)
                    query = query.filter(Model.name.ilike(f"%{model_name}%"))

                if 'location' in params and params['location']:
                    location_name = params['location']
                    query = query.join(Location, Position.location_id == Location.id)
                    query = query.filter(Location.name.ilike(f"%{location_name}%"))

            # Apply image-specific filters
            if 'title' in params and params['title']:
                query = query.filter(Image.title.ilike(f"%{params['title']}%"))

            if 'description' in params and params['description']:
                query = query.filter(Image.description.ilike(f"%{params['description']}%"))

            # If raw input is available, use it as a fallback for general search
            if 'raw_input' in params:
                raw_terms = params['raw_input'].split()
                # Remove common words and the keyword itself
                search_terms = [term for term in raw_terms if len(term) > 3]

                if search_terms:
                    term_filters = []
                    for term in search_terms:
                        term_filters.append(
                            or_(
                                Image.title.ilike(f"%{term}%"),
                                Image.description.ilike(f"%{term}%")
                            )
                        )
                    query = query.filter(or_(*term_filters))

            # Check for a specific extracted ID
            if 'extracted_id' in params:
                query = query.filter(Image.id == params['extracted_id'])

            # Apply sorting and limiting
            query = query.order_by(desc(Image.id))
            query = query.distinct()

            # Get results
            limit = int(params.get('limit', 10))
            results = query.limit(limit).all()

            # Format results
            formatted_results = []
            for img in results:
                formatted_results.append({
                    'id': img.id,
                    'title': img.title,
                    'description': img.description,
                    'file_path': img.file_path,
                    'url': f"/serve_image/{img.id}"
                })

            return {
                'status': 'success',
                'count': len(formatted_results),
                'results': formatted_results,
                'entity_type': 'image'
            }

        except Exception as e:
            logger.error(f"Error searching images: {e}")
            return {
                'status': 'error',
                'message': f"Error searching images: {str(e)}",
                'entity_type': 'image'
            }

    def search_parts(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Search for parts based on extracted parameters.

        Args:
            params: Dictionary of search parameters

        Returns:
            Dictionary with search results
        """
        try:
            # Start with base query
            query = self.session.query(Part)

            # Apply part-specific filters
            if 'part_number' in params and params['part_number']:
                query = query.filter(Part.part_number.ilike(f"%{params['part_number']}%"))

            if 'name' in params and params['name']:
                query = query.filter(Part.name.ilike(f"%{params['name']}%"))

            if 'oem_mfg' in params and params['oem_mfg']:
                query = query.filter(Part.oem_mfg.ilike(f"%{params['oem_mfg']}%"))

            if 'model' in params and params['model']:
                query = query.filter(Part.model.ilike(f"%{params['model']}%"))

            # If raw input is available, use it as a fallback for general search
            if 'raw_input' in params:
                raw_terms = params['raw_input'].split()
                # Remove common words and the keyword itself
                search_terms = [term for term in raw_terms if len(term) > 3]

                if search_terms:
                    term_filters = []
                    for term in search_terms:
                        term_filters.append(
                            or_(
                                Part.part_number.ilike(f"%{term}%"),
                                Part.name.ilike(f"%{term}%"),
                                Part.oem_mfg.ilike(f"%{term}%"),
                                Part.model.ilike(f"%{term}%"),
                                Part.notes.ilike(f"%{term}%")
                            )
                        )
                    query = query.filter(or_(*term_filters))

            # Check for a specific extracted ID
            if 'extracted_id' in params:
                query = query.filter(Part.id == params['extracted_id'])

            # Apply sorting and limiting
            query = query.order_by(Part.part_number)

            # Get results
            limit = int(params.get('limit', 10))
            results = query.limit(limit).all()

            # Format results
            formatted_results = []
            for part in results:
                formatted_results.append({
                    'id': part.id,
                    'part_number': part.part_number,
                    'name': part.name,
                    'oem_mfg': part.oem_mfg,
                    'model': part.model,
                    'class_flag': part.class_flag,
                    'notes': part.notes
                })

            return {
                'status': 'success',
                'count': len(formatted_results),
                'results': formatted_results,
                'entity_type': 'part'
            }

        except Exception as e:
            logger.error(f"Error searching parts: {e}")
            return {
                'status': 'error',
                'message': f"Error searching parts: {str(e)}",
                'entity_type': 'part'
            }

    def search_drawings(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Search for drawings based on extracted parameters.

        Args:
            params: Dictionary of search parameters

        Returns:
            Dictionary with search results
        """
        try:
            # Start with base query
            query = self.session.query(Drawing)

            # Apply drawing-specific filters
            if 'equipment_name' in params and params['equipment_name']:
                query = query.filter(Drawing.drw_equipment_name.ilike(f"%{params['equipment_name']}%"))

            if 'number' in params and params['number']:
                query = query.filter(Drawing.drw_number.ilike(f"%{params['number']}%"))

            if 'name' in params and params['name']:
                query = query.filter(Drawing.drw_name.ilike(f"%{params['name']}%"))

            if 'revision' in params and params['revision']:
                query = query.filter(Drawing.drw_revision.ilike(f"%{params['revision']}%"))

            if 'spare_part_number' in params and params['spare_part_number']:
                query = query.filter(Drawing.drw_spare_part_number.ilike(f"%{params['spare_part_number']}%"))

            # If raw input is available, use it as a fallback for general search
            if 'raw_input' in params:
                raw_terms = params['raw_input'].split()
                # Remove common words and the keyword itself
                search_terms = [term for term in raw_terms if len(term) > 3]

                if search_terms:
                    term_filters = []
                    for term in search_terms:
                        term_filters.append(
                            or_(
                                Drawing.drw_equipment_name.ilike(f"%{term}%"),
                                Drawing.drw_number.ilike(f"%{term}%"),
                                Drawing.drw_name.ilike(f"%{term}%"),
                                Drawing.drw_spare_part_number.ilike(f"%{term}%")
                            )
                        )
                    query = query.filter(or_(*term_filters))

            # Check for a specific extracted ID
            if 'extracted_id' in params:
                query = query.filter(Drawing.id == params['extracted_id'])

            # Apply sorting and limiting
            query = query.order_by(Drawing.drw_number)

            # Get results
            limit = int(params.get('limit', 10))
            results = query.limit(limit).all()

            # Format results
            formatted_results = []
            for drawing in results:
                formatted_results.append({
                    'id': drawing.id,
                    'drw_equipment_name': drawing.drw_equipment_name,
                    'drw_number': drawing.drw_number,
                    'drw_name': drawing.drw_name,
                    'drw_revision': drawing.drw_revision,
                    'drw_spare_part_number': drawing.drw_spare_part_number,
                    'file_path': drawing.file_path
                })

            return {
                'status': 'success',
                'count': len(formatted_results),
                'results': formatted_results,
                'entity_type': 'drawing'
            }

        except Exception as e:
            logger.error(f"Error searching drawings: {e}")
            return {
                'status': 'error',
                'message': f"Error searching drawings: {str(e)}",
                'entity_type': 'drawing'
            }

    def search_tools(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Search for tools based on extracted parameters.

        Args:
            params: Dictionary of search parameters

        Returns:
            Dictionary with search results
        """
        try:
            # Start with base query
            query = self.session.query(Tool)

            # Apply tool-specific filters
            if 'name' in params and params['name']:
                query = query.filter(Tool.name.ilike(f"%{params['name']}%"))

            if 'type' in params and params['type']:
                query = query.filter(Tool.type.ilike(f"%{params['type']}%"))

            if 'material' in params and params['material']:
                query = query.filter(Tool.material.ilike(f"%{params['material']}%"))

            if 'size' in params and params['size']:
                query = query.filter(Tool.size.ilike(f"%{params['size']}%"))

            if 'category' in params and params['category']:
                category_name = params['category']
                query = query.join(ToolCategory, Tool.tool_category_id == ToolCategory.id)
                query = query.filter(ToolCategory.name.ilike(f"%{category_name}%"))

            # If raw input is available, use it as a fallback for general search
            if 'raw_input' in params:
                raw_terms = params['raw_input'].split()
                # Remove common words and the keyword itself
                search_terms = [term for term in raw_terms if len(term) > 3]

                if search_terms:
                    term_filters = []
                    for term in search_terms:
                        term_filters.append(
                            or_(
                                Tool.name.ilike(f"%{term}%"),
                                Tool.type.ilike(f"%{term}%"),
                                Tool.description.ilike(f"%{term}%")
                            )
                        )
                    query = query.filter(or_(*term_filters))

            # Check for a specific extracted ID
            if 'extracted_id' in params:
                query = query.filter(Tool.id == params['extracted_id'])

            # Apply sorting and limiting
            query = query.order_by(Tool.name)

            # Eager load related images for preview
            query = query.options(joinedload(Tool.tool_image_association).joinedload(ToolImageAssociation.image))

            # Get results
            limit = int(params.get('limit', 10))
            results = query.limit(limit).all()

            # Format results
            formatted_results = []
            for tool in results:
                # Get image if available
                image_info = None
                if tool.tool_image_association:
                    for assoc in tool.tool_image_association:
                        if assoc.image:
                            image_info = {
                                'id': assoc.image.id,
                                'title': assoc.image.title,
                                'file_path': assoc.image.file_path,
                                'url': f"/serve_image/{assoc.image.id}"
                            }
                            break

                formatted_results.append({
                    'id': tool.id,
                    'name': tool.name,
                    'type': tool.type,
                    'material': tool.material,
                    'size': tool.size,
                    'description': tool.description,
                    'category_id': tool.tool_category_id,
                    'image': image_info
                })

            return {
                'status': 'success',
                'count': len(formatted_results),
                'results': formatted_results,
                'entity_type': 'tool'
            }

        except Exception as e:
            logger.error(f"Error searching tools: {e}")
            return {
                'status': 'error',
                'message': f"Error searching tools: {str(e)}",
                'entity_type': 'tool'
            }

    def search_positions(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Search for positions based on extracted parameters.

        Args:
            params: Dictionary of search parameters

        Returns:
            Dictionary with search results
        """
        try:
            # Start with base query
            query = self.session.query(Position)

            # Apply hierarchy filters
            if 'area' in params and params['area']:
                area_name = params['area']
                query = query.join(Area, Position.area_id == Area.id)
                query = query.filter(Area.name.ilike(f"%{area_name}%"))

            if 'equipment_group' in params and params['equipment_group']:
                equipment_name = params['equipment_group']
                query = query.join(EquipmentGroup, Position.equipment_group_id == EquipmentGroup.id)
                query = query.filter(EquipmentGroup.name.ilike(f"%{equipment_name}%"))

            if 'model' in params and params['model']:
                model_name = params['model']
                query = query.join(Model, Position.model_id == Model.id)
                query = query.filter(Model.name.ilike(f"%{model_name}%"))

            if 'location' in params and params['location']:
                location_name = params['location']
                query = query.join(Location, Position.location_id == Location.id)
                query = query.filter(Location.name.ilike(f"%{location_name}%"))

            # Check for a specific extracted ID
            if 'extracted_id' in params:
                query = query.filter(Position.id == params['extracted_id'])

            # Eager load relationships for better performance
            query = query.options(
                joinedload(Position.area),
                joinedload(Position.equipment_group),
                joinedload(Position.model),
                joinedload(Position.location)
            )

            # Get results
            limit = int(params.get('limit', 10))
            results = query.limit(limit).all()

            # Format results
            formatted_results = []
            for pos in results:
                formatted_results.append({
                    'id': pos.id,
                    'area': pos.area.name if pos.area else None,
                    'equipment_group': pos.equipment_group.name if pos.equipment_group else None,
                    'model': pos.model.name if pos.model else None,
                    'location': pos.location.name if pos.location else None,
                    'area_id': pos.area_id,
                    'equipment_group_id': pos.equipment_group_id,
                    'model_id': pos.model_id,
                    'asset_number_id': pos.asset_number_id,
                    'location_id': pos.location_id
                })

            return {
                'status': 'success',
                'count': len(formatted_results),
                'results': formatted_results,
                'entity_type': 'position'
            }

        except Exception as e:
            logger.error(f"Error searching positions: {e}")
            return {
                'status': 'error',
                'message': f"Error searching positions: {str(e)}",
                'entity_type': 'position'
            }

    def search_problems(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Search for problems based on extracted parameters.

        Args:
            params: Dictionary of search parameters

        Returns:
            Dictionary with search results
        """
        try:
            # Start with base query
            query = self.session.query(Problem)

            # Apply problem-specific filters
            if 'name' in params and params['name']:
                query = query.filter(Problem.name.ilike(f"%{params['name']}%"))

            if 'description' in params and params['description']:
                query = query.filter(Problem.description.ilike(f"%{params['description']}%"))

            # If raw input is available, use it as a fallback for general search
            if 'raw_input' in params:
                raw_terms = params['raw_input'].split()
                # Remove common words and the keyword itself
                search_terms = [term for term in raw_terms if len(term) > 3]

                if search_terms:
                    term_filters = []
                    for term in search_terms:
                        term_filters.append(
                            or_(
                                Problem.name.ilike(f"%{term}%"),
                                Problem.description.ilike(f"%{term}%")
                            )
                        )
                    query = query.filter(or_(*term_filters))

            # Check for a specific extracted ID
            if 'extracted_id' in params:
                query = query.filter(Problem.id == params['extracted_id'])

            # Apply sorting
            query = query.order_by(Problem.name)

            # Eager load solutions for better performance
            query = query.options(joinedload(Problem.solutions))

            # Get results
            limit = int(params.get('limit', 10))
            results = query.limit(limit).all()

            # Format results
            formatted_results = []
            for problem in results:
                solutions = []
                for solution in problem.solutions:
                    solutions.append({
                        'id': solution.id,
                        'name': solution.name,
                        'description': solution.description
                    })

                formatted_results.append({
                    'id': problem.id,
                    'name': problem.name,
                    'description': problem.description,
                    'solutions': solutions,
                    'solution_count': len(solutions)
                })

            return {
                'status': 'success',
                'count': len(formatted_results),
                'results': formatted_results,
                'entity_type': 'problem'
            }

        except Exception as e:
            logger.error(f"Error searching problems: {e}")
            return {
                'status': 'error',
                'message': f"Error searching problems: {str(e)}",
                'entity_type': 'problem'
            }

    def search_tasks(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Search for tasks based on extracted parameters.

        Args:
            params: Dictionary of search parameters

        Returns:
            Dictionary with search results
        """
        try:
            # Start with base query
            query = self.session.query(Task)

            # Apply task-specific filters
            if 'name' in params and params['name']:
                query = query.filter(Task.name.ilike(f"%{params['name']}%"))

            if 'description' in params and params['description']:
                query = query.filter(Task.description.ilike(f"%{params['description']}%"))

            # If raw input is available, use it as a fallback for general search
            if 'raw_input' in params:
                raw_terms = params['raw_input'].split()
                # Remove common words and the keyword itself
                search_terms = [term for term in raw_terms if len(term) > 3]

                if search_terms:
                    term_filters = []
                    for term in search_terms:
                        term_filters.append(
                            or_(
                                Task.name.ilike(f"%{term}%"),
                                Task.description.ilike(f"%{term}%")
                            )
                        )
                    query = query.filter(or_(*term_filters))

            # Check for a specific extracted ID
            if 'extracted_id' in params:
                query = query.filter(Task.id == params['extracted_id'])

            # Apply sorting
            query = query.order_by(Task.name)

            # Get results
            limit = int(params.get('limit', 10))
            results = query.limit(limit).all()

            # Format results
            formatted_results = []
            for task in results:
                formatted_results.append({
                    'id': task.id,
                    'name': task.name,
                    'description': task.description
                })

            return {
                'status': 'success',
                'count': len(formatted_results),
                'results': formatted_results,
                'entity_type': 'task'
            }

        except Exception as e:
            logger.error(f"Error searching tasks: {e}")
            return {
                'status': 'error',
                'message': f"Error searching tasks: {str(e)}",
                'entity_type': 'task'
            }

    def load_keywords_from_excel(self, excel_path: str) -> Dict[str, Any]:
        """
        Load keywords and actions from an Excel file.

        Args:
            excel_path: Path to the Excel file

        Returns:
            Dictionary with summary of import results
        """
        try:
            import pandas as pd

            # Track results
            results = {
                "added": 0,
                "updated": 0,
                "skipped": 0,
                "failed": 0,
                "errors": []
            }

            # Read the Excel file
            df = pd.read_excel(excel_path)

            # Check required columns
            required_columns = ['keyword', 'action_type']
            if not all(col in df.columns for col in required_columns):
                missing = [col for col in required_columns if col not in df.columns]
                return {
                    "status": "error",
                    "message": f"Missing required columns in Excel: {', '.join(missing)}",
                    "required_columns": required_columns
                }

            # Identify optional columns
            optional_columns = ['search_pattern', 'entity_type', 'description']

            # Process each row
            for index, row in df.iterrows():
                try:
                    keyword = row['keyword']
                    action_type = row['action_type']

                    # Get optional parameters if available
                    params = {}
                    for col in optional_columns:
                        if col in df.columns and pd.notna(row[col]):
                            params[col] = row[col]

                    # Handle legacy 'action' column if present
                    if 'action' in df.columns and pd.notna(row['action']):
                        # Check if it's JSON
                        try:
                            action_data = json.loads(row['action'])
                            if 'type' in action_data and not action_type:
                                action_type = action_data['type']
                            if 'search_pattern' in action_data and 'search_pattern' not in params:
                                params['search_pattern'] = action_data['search_pattern']
                            if 'entity_type' in action_data and 'entity_type' not in params:
                                params['entity_type'] = action_data['entity_type']
                            if 'description' in action_data and 'description' not in params:
                                params['description'] = action_data['description']
                        except json.JSONDecodeError:
                            # Not JSON, use as-is if action_type is missing
                            if not action_type:
                                action_type = row['action']

                    # Skip if keyword or action_type is missing
                    if not keyword or not action_type:
                        logger.warning(f"Row {index + 1}: Missing keyword or action_type. Skipping.")
                        results["skipped"] += 1
                        continue

                    # Register the keyword
                    result = self.register_keyword(
                        keyword=keyword,
                        action_type=action_type,
                        **params
                    )

                    if result['status'] == 'created':
                        results["added"] += 1
                    elif result['status'] == 'updated':
                        results["updated"] += 1
                    else:
                        results["failed"] += 1
                        results["errors"].append(f"Row {index + 1}: {result.get('message', 'Unknown error')}")

                except Exception as e:
                    logger.error(f"Error processing row {index + 1}: {e}")
                    results["failed"] += 1
                    results["errors"].append(f"Row {index + 1}: {str(e)}")

            # Add summary
            results["status"] = "success"
            results["total_processed"] = len(df)
            results[
                "message"] = f"Processed {len(df)} keywords: {results['added']} added, {results['updated']} updated, {results['skipped']} skipped, {results['failed']} failed."

            return results

        except Exception as e:
            logger.error(f"Error loading keywords from Excel: {e}")
            return {
                "status": "error",
                "message": f"Error loading keywords from Excel: {str(e)}"
            }




# Base.metadata.create_all(engine, checkfirst=True)  # <-- COMMENTED OUT

_database_initialized = False


def initialize_database_tables():
    """Initialize database tables only when called, not at import time."""
    global _database_initialized

    if not _database_initialized:
        try:
            print("🗄  Initializing database tables...")
            Base.metadata.create_all(engine, checkfirst=True)
            _database_initialized = True
            print(" Database tables initialized successfully")
        except Exception as e:
            print(f" Failed to initialize database tables: {e}")
            raise
    else:
        print(" Database tables already initialized")

# Ask for API key if it's empty
if not OPENAI_API_KEY:
    OPENAI_API_KEY = input("Enter your OpenAI API key: ")
    with open('../configuration/config.py', 'w') as config_file:
        config_file.write(f'BASE_DIR = "{BASE_DIR}"\n')
        config_file.write(f'copy_files = {COPY_FILES}\n')
        config_file.write(f'OPENAI_API_KEY = "{OPENAI_API_KEY}"\n')


def load_image_model_config_from_db():
    return None