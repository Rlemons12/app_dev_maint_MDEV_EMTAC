import pandas as pd
import numpy as np
import logging
import os
import sys
import argparse
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.sql import text
import datetime
from typing import Dict, List, Tuple, Set, Optional
import hashlib

# Import database configuration
from modules.configuration.config_env import DatabaseConfig
# Import custom logging functions
from modules.configuration.log_config import (
    debug_id, info_id, warning_id, error_id, critical_id,
    with_request_id, get_request_id, set_request_id, log_timed_operation
)

# Import application configuration
from modules.configuration.config import (
    BASE_DIR, DATABASE_DIR, DB_LOADSHEET, DATABASE_URL
)

# Create the drawing_import_data directory path
DRAWING_IMPORT_DATA_DIR = os.path.join(DB_LOADSHEET, "drawing_import_data")

"""
Drawing to Location Upload Script
================================

This script processes Excel data containing drawing information and loads it into the database,
maintaining relationships between drawings, parts, positions, and other hierarchical data.

Key Features:
------------
1. Hierarchical Data Processing: Handles area → equipment group → model → location → position hierarchy
2. Relationship Management: Creates and maintains associations between drawings, positions, and parts
3. Batch Processing: Processes Excel data in configurable batches with transaction management
4. Change Detection: Uses hashing to detect changes and avoid redundant processing
5. Error Handling: Provides robust error handling with automatic retries
6. Validation & Repair: Includes tools to validate database entries and repair relationships
7. Reporting: Generates detailed reports of created, updated, and unchanged entities

Usage:
------
1. Normal import mode:
   python drawing_to_location_upload.py [--file FILE] [--batch-size SIZE]

2. Validation mode (check specific database entries):
   python drawing_to_location_upload.py --validate

3. Repair mode (fix missing relationships):
   python drawing_to_location_upload.py --repair

4. Force create mode (only create missing part-position entries):
   python drawing_to_location_upload.py --force-create

Arguments:
---------
  --file, -f       Path to Excel file to import (default: DB_LOADSHEET/Active Drawing List breakdown.xlsx)
  --batch-size, -b Batch size for processing (default: 50)
  --validate, -v   Run in validation mode to check specific entries
  --repair, -r     Repair missing relationships in the database
  --force-create, -c Force create missing part-position associations

Data Flow:
---------
1. Excel file is loaded and parsed row by row
2. For each row, the script processes hierarchical data:
   - Area → Equipment Group → Model → Location → Position
3. Drawing and part information is processed
4. Associations are created between drawings, positions, and parts
5. Missing part-position associations are automatically created
6. Reports are generated and saved to the drawing_import_data directory

File Structure:
--------------
- DB_LOADSHEET/drawing_import_data/ - Directory for import-related data
  - .row_hashes.csv - Stores hashes for change detection
  - import_report_*.txt - Generated import reports
  - validation_report_*.txt - Validation reports
  - missing_associations_*.csv - Lists of missing associations

Dependencies:
------------
- pandas, numpy - For data processing
- sqlalchemy - For database operations
- hashlib - For hash generation
- modules.emtacdb.emtacdb_fts - Database models
- modules.configuration.config - Application configuration
- modules.configuration.log_config - Logging utilities

Authors:
-------
Original script developed for EMTAC database system
Enhanced with validation, repair, and optimization features

Last Updated: May 2025
"""

import pandas as pd
import numpy as np
import logging
import os
import sys
import argparse
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.sql import text
import datetime
from typing import Dict, List, Tuple, Set, Optional
import hashlib

# Import database configuration
from modules.configuration.config_env import DatabaseConfig
# Import custom logging functions
from modules.configuration.log_config import (
    debug_id, info_id, warning_id, error_id, critical_id,
    with_request_id, get_request_id, set_request_id, log_timed_operation
)

# Import application configuration
from modules.configuration.config import (
    BASE_DIR, DATABASE_DIR, DB_LOADSHEET, DATABASE_URL
)

# Create the drawing_import_data directory path
DRAWING_IMPORT_DATA_DIR = os.path.join(DB_LOADSHEET, "drawing_import_data")


# Ensure the directory exists
def ensure_directories_exist():
    """Ensure all required directories exist."""
    directories = [
        DB_LOADSHEET,
        DRAWING_IMPORT_DATA_DIR
    ]

    for directory in directories:
        if not os.path.exists(directory):
            try:
                os.makedirs(directory)
                info_id(f"Created directory: {directory}")
            except Exception as e:
                error_id(f"Error creating directory {directory}: {e}")


# Call this at the beginning to make sure the directory exists
ensure_directories_exist()

# Import database models
from modules.emtacdb.emtacdb_fts import (
    Area, EquipmentGroup, Model, AssetNumber, Location, Position, Drawing, Part,
    DrawingPositionAssociation, DrawingPartAssociation, PartsPositionImageAssociation
)

# Use the module's logger
from modules.configuration.log_config import logger


class ChangeTracker:
    """
    Tracks changes made during database operations for reporting.
    """

    def __init__(self):
        self.created = {
            'area': [],
            'equipment_group': [],
            'model': [],
            'location': [],
            'position': [],
            'drawing': [],
            'part': [],
            'drawing_position_assoc': [],
            'drawing_part_assoc': [],
            'part_position_assoc': []
        }
        self.updated = {
            'drawing': [],
            'part': []
        }
        self.unchanged = {
            'drawing': 0,
            'position': 0,
            'part': 0,
            'associations': 0
        }
        self.errors = []
        self.processed_rows = 0
        self.skipped_rows = 0

    def add_created(self, entity_type, entity_id, name=None):
        """Record a created entity"""
        if name:
            self.created[entity_type].append((entity_id, name))
        else:
            self.created[entity_type].append(entity_id)

    def add_updated(self, entity_type, entity_id, name=None):
        """Record an updated entity"""
        if name:
            self.updated[entity_type].append((entity_id, name))
        else:
            self.updated[entity_type].append(entity_id)

    def add_unchanged(self, entity_type):
        """Record an unchanged entity"""
        self.unchanged[entity_type] += 1

    def add_error(self, row_idx, error_msg):
        """Record an error"""
        self.errors.append((row_idx, error_msg))

    def generate_report(self):
        """Generate a detailed report of changes"""
        report = []
        report.append(f"Change Report - {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("=" * 80)
        report.append(f"Processed {self.processed_rows} rows, skipped {self.skipped_rows} rows")
        report.append("-" * 80)

        report.append("\nCreated Entities:")
        for entity_type, entities in self.created.items():
            if entities:
                if isinstance(entities[0], tuple):
                    report.append(f"  {entity_type.replace('_', ' ').title()}: {len(entities)}")
                    for entity_id, name in entities[:10]:  # Limit to first 10 for readability
                        report.append(f"    - ID: {entity_id}, Name: {name}")
                    if len(entities) > 10:
                        report.append(f"    ... and {len(entities) - 10} more")
                else:
                    report.append(f"  {entity_type.replace('_', ' ').title()}: {len(entities)}")

        report.append("\nUpdated Entities:")
        for entity_type, entities in self.updated.items():
            if entities:
                if isinstance(entities[0], tuple):
                    report.append(f"  {entity_type.replace('_', ' ').title()}: {len(entities)}")
                    for entity_id, name in entities[:10]:
                        report.append(f"    - ID: {entity_id}, Name: {name}")
                    if len(entities) > 10:
                        report.append(f"    ... and {len(entities) - 10} more")
                else:
                    report.append(f"  {entity_type.replace('_', ' ').title()}: {len(entities)}")

        report.append("\nUnchanged Entities:")
        for entity_type, count in self.unchanged.items():
            if count > 0:
                report.append(f"  {entity_type.replace('_', ' ').title()}: {count}")

        if self.errors:
            report.append("\nErrors:")
            for row_idx, error_msg in self.errors:
                report.append(f"  Row {row_idx}: {error_msg}")

        return "\n".join(report)


class DataValidator:
    """
    Validates specific database entries and reports on missing or inconsistent data.
    """

    def __init__(self, session):
        """Initialize with database session"""
        self.session = session
        self.missing_entries = []
        self.inconsistent_entries = []

    def validate_ids(self, entity_type, id_pairs, relationship_type=None):
        """
        Validate specific ID pairs in the database

        Args:
            entity_type: Type of entity to check ('position', 'drawing', 'part', etc.)
            id_pairs: List of ID pairs to check [(id1, id2), ...]
            relationship_type: Type of relationship to check ('drawing_position', 'part_position', etc.)
        """
        if entity_type == 'drawing_position':
            self._validate_drawing_position_pairs(id_pairs)
        elif entity_type == 'drawing_part':
            self._validate_drawing_part_pairs(id_pairs)
        elif entity_type == 'part_position':
            self._validate_part_position_pairs(id_pairs)
        elif entity_type == 'hierarchy':
            self._validate_hierarchy_ids(id_pairs)
        else:
            info_id(f"Validation for {entity_type} not implemented")

    def _validate_drawing_position_pairs(self, drawing_position_pairs):
        """Check if drawing-position associations exist"""
        for drawing_id, position_id in drawing_position_pairs:
            try:
                assoc = (
                    self.session.query(DrawingPositionAssociation)
                    .filter(
                        DrawingPositionAssociation.drawing_id == drawing_id,
                        DrawingPositionAssociation.position_id == position_id
                    )
                    .first()
                )

                if not assoc:
                    self.missing_entries.append({
                        'type': 'drawing_position',
                        'ids': (drawing_id, position_id),
                        'message': f"Missing association between drawing ID {drawing_id} and position ID {position_id}"
                    })
                else:
                    debug_id(f"Found association between drawing {drawing_id} and position {position_id}")
            except Exception as e:
                error_id(f"Error validating drawing-position pair ({drawing_id}, {position_id}): {e}")

    def _validate_drawing_part_pairs(self, drawing_part_pairs):
        """Check if drawing-part associations exist"""
        for drawing_id, part_id in drawing_part_pairs:
            try:
                assoc = (
                    self.session.query(DrawingPartAssociation)
                    .filter(
                        DrawingPartAssociation.drawing_id == drawing_id,
                        DrawingPartAssociation.part_id == part_id
                    )
                    .first()
                )

                if not assoc:
                    self.missing_entries.append({
                        'type': 'drawing_part',
                        'ids': (drawing_id, part_id),
                        'message': f"Missing association between drawing ID {drawing_id} and part ID {part_id}"
                    })
                else:
                    debug_id(f"Found association between drawing {drawing_id} and part {part_id}")
            except Exception as e:
                error_id(f"Error validating drawing-part pair ({drawing_id}, {part_id}): {e}")

    def _validate_part_position_pairs(self, part_position_pairs):
        """Check if part-position associations exist"""
        for part_id, position_id in part_position_pairs:
            try:
                assoc = (
                    self.session.query(PartsPositionImageAssociation)
                    .filter(
                        PartsPositionImageAssociation.part_id == part_id,
                        PartsPositionImageAssociation.position_id == position_id
                    )
                    .first()
                )

                if not assoc:
                    self.missing_entries.append({
                        'type': 'part_position',
                        'ids': (part_id, position_id),
                        'message': f"Missing association between part ID {part_id} and position ID {position_id}"
                    })
                else:
                    debug_id(f"Found association between part {part_id} and position {position_id}")
            except Exception as e:
                error_id(f"Error validating part-position pair ({part_id}, {position_id}): {e}")

    def _validate_hierarchy_ids(self, hierarchy_ids):
        """Check if hierarchies (area, equipment_group, model, location) exist"""
        for area_id, equipment_group_id, model_id, location_id in hierarchy_ids:
            try:
                # Check if position with this hierarchy exists
                position = (
                    self.session.query(Position)
                    .filter(
                        Position.area_id == area_id,
                        Position.equipment_group_id == equipment_group_id,
                        Position.model_id == model_id,
                        Position.location_id == location_id
                    )
                    .first()
                )

                if not position:
                    self.missing_entries.append({
                        'type': 'hierarchy',
                        'ids': (area_id, equipment_group_id, model_id, location_id),
                        'message': f"Missing position for hierarchy: Area {area_id}, EG {equipment_group_id}, Model {model_id}, Location {location_id}"
                    })
                else:
                    debug_id(
                        f"Found position {position.id} for hierarchy {area_id}/{equipment_group_id}/{model_id}/{location_id}")
            except Exception as e:
                error_id(
                    f"Error validating hierarchy ({area_id}, {equipment_group_id}, {model_id}, {location_id}): {e}")

    def validate_entity_existence(self, entity_type, ids):
        """Check if specific entities exist by ID"""
        model_map = {
            'area': Area,
            'equipment_group': EquipmentGroup,
            'model': Model,
            'location': Location,
            'position': Position,
            'drawing': Drawing,
            'part': Part
        }

        if entity_type not in model_map:
            warning_id(f"Unknown entity type: {entity_type}")
            return

        Model = model_map[entity_type]

        for entity_id in ids:
            try:
                entity = self.session.query(Model).filter(Model.id == entity_id).first()
                if not entity:
                    self.missing_entries.append({
                        'type': entity_type,
                        'ids': (entity_id,),
                        'message': f"Missing {entity_type} with ID {entity_id}"
                    })
                else:
                    debug_id(f"Found {entity_type} with ID {entity_id}")
            except Exception as e:
                error_id(f"Error validating {entity_type} ID {entity_id}: {e}")

    def generate_validation_report(self):
        """Generate a report of validation findings"""
        if not self.missing_entries and not self.inconsistent_entries:
            return "All validated entries were found in the database."

        report = []
        report.append("Validation Report")
        report.append("=" * 80)

        if self.missing_entries:
            report.append("\nMissing Entries:")
            for entry in self.missing_entries:
                report.append(f"  • {entry['message']}")

        if self.inconsistent_entries:
            report.append("\nInconsistent Entries:")
            for entry in self.inconsistent_entries:
                report.append(f"  • {entry['message']}")

        return "\n".join(report)


class ExcelToDbMapper:
    """
    An optimized class to map Excel data to database models with change tracking and batch processing.
    """

    def __init__(self, db_url=None, batch_size=50):
        """
        Initialize with database connection using DatabaseConfig.

        Args:
            db_url (str, optional): SQLAlchemy database URL (kept for compatibility, not used)
            batch_size (int): Number of rows to process in a single transaction
        """
        # Set a request ID for this instance
        self.request_id = set_request_id()
        info_id(f"Initializing ExcelToDbMapper with DatabaseConfig", self.request_id)

        # For backward compatibility, log the provided db_url if present
        if db_url:
            debug_id(f"Note: Provided db_url: {db_url} will be ignored in favor of DatabaseConfig")

        # Create DB config and get a session
        self.db_config = DatabaseConfig()
        self.session = self.db_config.get_main_session()
        self.excel_data = None
        self.batch_size = batch_size
        self.change_tracker = ChangeTracker()

        # Caches to reduce database queries
        self.area_cache = {}
        self.equipment_group_cache = {}
        self.model_cache = {}
        self.location_cache = {}
        self.position_cache = {}
        self.drawing_cache = {}
        self.part_cache = {}
        self.assoc_cache = {}

        # Track row hashes for change detection
        self.previous_row_hashes = self._load_previous_hashes()

    def _load_previous_hashes(self) -> Dict[str, str]:
        """
        Load previously processed row hashes from database or file.
        Used to detect changes in the Excel file.
        """
        hash_dict = {}
        hash_file = os.path.join(DRAWING_IMPORT_DATA_DIR, ".row_hashes.csv")

        if os.path.exists(hash_file):
            try:
                hash_df = pd.read_csv(hash_file)
                for _, row in hash_df.iterrows():
                    hash_dict[row['row_key']] = row['hash']
                info_id(f"Loaded {len(hash_dict)} previous row hashes from {hash_file}")
            except Exception as e:
                warning_id(f"Could not load previous hashes: {e}")

        return hash_dict

    def _save_current_hashes(self, current_hashes: Dict[str, str]):
        """Save the current row hashes for future comparison"""
        # Use the drawing_import_data directory for storing hash files
        hash_file = os.path.join(DRAWING_IMPORT_DATA_DIR, ".row_hashes.csv")

        try:
            hash_df = pd.DataFrame([
                {'row_key': key, 'hash': value}
                for key, value in current_hashes.items()
            ])
            hash_df.to_csv(hash_file, index=False)
            info_id(f"Saved {len(current_hashes)} row hashes to {hash_file}")
        except Exception as e:
            warning_id(f"Could not save current hashes: {e}")

    def _generate_row_hash(self, row) -> Tuple[str, str]:
        """
        Generate a unique hash for a row to detect changes.
        Returns (row_key, hash_value)
        """
        # Create a unique key for this row based on hierarchical data
        area = str(row.get('area', '')).strip() if pd.notna(row.get('area', '')) else ''
        equip_group = str(row.get('equipment_group', '')).strip() if pd.notna(row.get('equipment_group', '')) else ''
        model = str(row.get('model', '')).strip() if pd.notna(row.get('model', '')) else ''
        stations = str(row.get('stations', '')).strip() if pd.notna(row.get('stations', '')) else ''
        drawing = str(row.get('DRAWING NUMBER', '')).strip() if pd.notna(row.get('DRAWING NUMBER', '')) else ''

        # Create a row key that uniquely identifies this hierarchy
        row_key = f"{area}|{equip_group}|{model}|{stations}|{drawing}"

        # Create a hash of all row values to detect any changes
        row_values_list = []
        for column, value in row.items():
            # Handle different types properly
            if pd.isna(value):
                row_values_list.append('')
            elif isinstance(value, (np.ndarray, list, tuple)):
                # Convert arrays/lists/tuples to string representation
                row_values_list.append(str(list(value)))
            else:
                # Regular values
                row_values_list.append(str(value).strip())

        row_values = '|'.join(row_values_list)
        hash_value = hashlib.md5(row_values.encode()).hexdigest()

        return row_key, hash_value

    def _preload_caches(self):
        """Preload frequently accessed data to reduce database queries"""
        with log_timed_operation("Preloading database caches"):
            # Load areas
            areas = self.session.query(Area).all()
            for area in areas:
                self.area_cache[area.name] = area
            info_id(f"Preloaded {len(areas)} areas into cache")

            # Load drawings
            drawings = self.session.query(Drawing).all()
            for drawing in drawings:
                self.drawing_cache[drawing.drw_number] = drawing
            info_id(f"Preloaded {len(drawings)} drawings into cache")

            # Load parts
            parts = self.session.query(Part).all()
            for part in parts:
                self.part_cache[part.part_number] = part
            info_id(f"Preloaded {len(parts)} parts into cache")

            # We don't preload all equipment groups, models and locations
            # as there could be too many, but we'll cache them as we go

    @with_request_id
    def load_excel(self, file_path):
        """
        Load data from Excel file.

        Args:
            file_path (str): Path to the Excel file

        Returns:
            bool: True if successful, False otherwise
        """
        try:
            info_id(f"Loading Excel file: {file_path}")

            # Verify file exists
            if not os.path.exists(file_path):
                error_id(f"Excel file not found: {file_path}")
                return False

            # Load the Excel file with correct data types
            self.excel_data = pd.read_excel(file_path, dtype=str)

            # Clean the data - remove rows with all NaN values
            self.excel_data = self.excel_data.dropna(how='all')

            # Fill NaN values with empty strings for all columns
            self.excel_data = self.excel_data.fillna('')

            info_id(f"Successfully loaded {len(self.excel_data)} rows")

            # Log column headers for verification
            headers = list(self.excel_data.columns)
            debug_id(f"Excel headers: {headers}")

            # Preload database caches
            self._preload_caches()

            return True
        except Exception as e:
            error_id(f"Error loading Excel file: {e}", exc_info=True)
            return False

    def _batch_safe_commit(self, batch_num, rows_processed):
        """
        Safely commit the current transaction with error handling.
        If commit fails, log the error and return False.
        """
        try:
            with log_timed_operation(f"Committing batch {batch_num}"):
                self.session.commit()
                info_id(f"Successfully committed batch {batch_num} with {rows_processed} rows")
            return True
        except SQLAlchemyError as e:
            self.session.rollback()
            error_id(f"Database error in batch {batch_num}: {e}", exc_info=True)
            return False
        except Exception as e:
            self.session.rollback()
            error_id(f"Unexpected error in batch {batch_num}: {e}", exc_info=True)
            return False

    def _process_batch(self, batch_df, batch_num):
        """Process a batch of Excel rows in a single transaction"""
        info_id(f"Processing batch {batch_num} with {len(batch_df)} rows")
        current_hashes = {}
        processed_count = 0
        max_attempts = 3  # Number of retry attempts for database issues

        for attempt in range(1, max_attempts + 1):
            try:
                # Clear out any uncommitted changes from previous failed attempts
                if attempt > 1:
                    self.session.rollback()
                    info_id(f"Retry attempt {attempt} for batch {batch_num}")

                # Process each row in the batch
                for index, row in batch_df.iterrows():
                    try:
                        # Get row hash to check for changes
                        row_key, row_hash = self._generate_row_hash(row)
                        current_hashes[row_key] = row_hash

                        # Check if row is unchanged from previous run
                        if row_key in self.previous_row_hashes and self.previous_row_hashes[row_key] == row_hash:
                            debug_id(f"Row {index + 1} is unchanged, skipping detailed processing")
                            # Still process it to ensure references exist, but don't need to update
                            self._process_row(row, index, skip_updates=True)
                        else:
                            # Process row normally
                            self._process_row(row, index)

                        processed_count += 1
                        self.change_tracker.processed_rows += 1

                    except Exception as e:
                        error_id(f"Error processing row {index + 1}: {e}", exc_info=True)
                        self.change_tracker.add_error(index + 1, str(e))
                        self.change_tracker.skipped_rows += 1

                # Commit the batch transaction
                if self._batch_safe_commit(batch_num, processed_count):
                    return current_hashes
                else:
                    # If commit failed but we have more attempts, try again
                    if attempt < max_attempts:
                        warning_id(
                            f"Commit failed for batch {batch_num}, will retry (attempt {attempt}/{max_attempts})")
                        # Reset processed count for next attempt
                        processed_count = 0
                        continue
                    else:
                        error_id(f"Failed to commit batch {batch_num} after {max_attempts} attempts")
                        self.change_tracker.skipped_rows += len(batch_df)
                        return {}

            except Exception as e:
                self.session.rollback()
                error_id(f"Error processing batch {batch_num} (attempt {attempt}): {e}", exc_info=True)

                # If we have more attempts, try again
                if attempt < max_attempts:
                    warning_id(
                        f"Processing failed for batch {batch_num}, will retry (attempt {attempt}/{max_attempts})")
                    # Reset processed count for next attempt
                    processed_count = 0
                    continue
                else:
                    # Add error for the entire batch
                    self.change_tracker.add_error(f"Batch {batch_num}", str(e))
                    self.change_tracker.skipped_rows += len(batch_df)
                    return {}

    def _process_row(self, row, index, skip_updates=False):
        """Process a single row of Excel data"""
        with log_timed_operation(f"Processing row {index + 1}"):
            debug_id(f"Processing row {index + 1}")

            # 1. Validate all required fields are present
            required_fields = ['area', 'equipment_group', 'model', 'stations']
            missing_fields = []

            for field in required_fields:
                value = row.get(field)
                if value is None or str(value).strip() == '':
                    missing_fields.append(field)

            if missing_fields:
                warning_id(f"Skipping row {index + 1}: Missing required fields: {', '.join(missing_fields)}")
                self.change_tracker.skipped_rows += 1
                return

            # Process area
            area = self._process_area(row, skip_updates)

            # Process equipment group
            equipment_group = self._process_equipment_group(row, area, skip_updates)

            # Process model
            model = self._process_model(row, equipment_group, skip_updates)

            # Process location
            location = self._process_location(row, model, skip_updates)

            # Process position
            position = self._process_position(area, equipment_group, model, location, skip_updates)

            # Process drawing if present
            drawing = None
            drawing_number = row.get('DRAWING NUMBER')
            if drawing_number and str(drawing_number).strip():
                drawing = self._process_drawing(row, skip_updates)

            # Associate drawing with position if both exist
            if drawing and position:
                self._associate_drawing_position(drawing, position, skip_updates)

            # Process spare part if present
            part = None
            spare_part_number = row.get('SPARE PART NUMBER')
            if drawing and spare_part_number and str(spare_part_number).strip():
                part = self._process_spare_part(row, skip_updates)

                # Associate drawing with part
                if part:
                    self._associate_drawing_part(drawing, part, skip_updates)

                    # Associate part with position
                    if position:
                        self._associate_part_position(part, position, skip_updates)

    def _process_area(self, row, skip_updates=False) -> Area:
        """Process area data from row"""
        area_name = str(row.get('area', '')).strip() if row.get('area') else ''

        # Check cache first
        if area_name in self.area_cache:
            self.change_tracker.add_unchanged('area')
            return self.area_cache[area_name]

        # Check database
        area = self.session.query(Area).filter(Area.name == area_name).first()

        # Create if not exists
        if not area:
            info_id(f"Creating new Area: {area_name}")
            area = Area(name=area_name)
            self.session.add(area)
            self.session.flush()
            self.change_tracker.add_created('area', area.id, area_name)
        else:
            self.change_tracker.add_unchanged('area')

        # Add to cache
        self.area_cache[area_name] = area
        return area

    def _process_equipment_group(self, row, area, skip_updates=False) -> EquipmentGroup:
        """Process equipment group data from row"""
        equipment_group_name = str(row.get('equipment_group', '')).strip() if row.get('equipment_group') else ''
        cache_key = f"{area.id}:{equipment_group_name}"

        # Check cache first
        if cache_key in self.equipment_group_cache:
            self.change_tracker.add_unchanged('equipment_group')
            return self.equipment_group_cache[cache_key]

        # Check database
        equipment_group = (
            self.session.query(EquipmentGroup)
            .filter(
                EquipmentGroup.name == equipment_group_name,
                EquipmentGroup.area_id == area.id
            )
            .first()
        )

        # Create if not exists
        if not equipment_group:
            info_id(f"Creating new EquipmentGroup: {equipment_group_name} in area {area.id}")
            equipment_group = EquipmentGroup(name=equipment_group_name, area_id=area.id)
            self.session.add(equipment_group)
            self.session.flush()
            self.change_tracker.add_created('equipment_group', equipment_group.id, equipment_group_name)
        else:
            self.change_tracker.add_unchanged('equipment_group')

        # Add to cache
        self.equipment_group_cache[cache_key] = equipment_group
        return equipment_group

    def _process_model(self, row, equipment_group, skip_updates=False) -> Model:
        """Process model data from row"""
        model_name = str(row.get('model', '')).strip() if row.get('model') else ''
        cache_key = f"{equipment_group.id}:{model_name}"

        # Check cache first
        if cache_key in self.model_cache:
            self.change_tracker.add_unchanged('model')
            return self.model_cache[cache_key]

        # Check database
        model = (
            self.session.query(Model)
            .filter(
                Model.name == model_name,
                Model.equipment_group_id == equipment_group.id
            )
            .first()
        )

        # Create if not exists
        if not model:
            info_id(f"Creating new Model: {model_name} in equipment group {equipment_group.id}")
            model = Model(name=model_name, equipment_group_id=equipment_group.id)
            self.session.add(model)
            self.session.flush()
            self.change_tracker.add_created('model', model.id, model_name)
        else:
            self.change_tracker.add_unchanged('model')

        # Add to cache
        self.model_cache[cache_key] = model
        return model

    def _process_location(self, row, model, skip_updates=False) -> Location:
        """Process location data from row"""
        stations_value = str(row.get('stations', '')).strip() if row.get('stations') else ''

        # Extract only the first part before any comma
        if ',' in stations_value:
            location_name = stations_value.split(',')[0].strip()
            debug_id(f"Using only first location from comma-separated list: {location_name}")
        else:
            location_name = stations_value

        cache_key = f"{model.id}:{location_name}"

        # Check cache first
        if cache_key in self.location_cache:
            self.change_tracker.add_unchanged('location')
            return self.location_cache[cache_key]

        # Check database
        location = (
            self.session.query(Location)
            .filter(
                Location.name == location_name,
                Location.model_id == model.id
            )
            .first()
        )

        # Create if not exists
        if not location:
            info_id(f"Creating new Location: {location_name} for model {model.id}")
            location = Location(name=location_name, model_id=model.id)
            self.session.add(location)
            self.session.flush()
            self.change_tracker.add_created('location', location.id, location_name)
        else:
            self.change_tracker.add_unchanged('location')

        # Add to cache
        self.location_cache[cache_key] = location
        return location

    def _process_position(self, area, equipment_group, model, location, skip_updates=False) -> Position:
        """Process position data"""
        cache_key = f"{area.id}:{equipment_group.id}:{model.id}:{location.id}"

        # Check cache first
        if cache_key in self.position_cache:
            self.change_tracker.add_unchanged('position')
            return self.position_cache[cache_key]

        # Check database
        position = (
            self.session.query(Position)
            .filter(
                Position.area_id == area.id,
                Position.equipment_group_id == equipment_group.id,
                Position.model_id == model.id,
                Position.location_id == location.id
            )
            .first()
        )

        # Create if not exists
        if not position:
            info_id(
                f"Creating new Position with hierarchy {area.name}/{equipment_group.name}/{model.name}/{location.name}")
            position = Position(
                area_id=area.id,
                equipment_group_id=equipment_group.id,
                model_id=model.id,
                location_id=location.id
            )
            self.session.add(position)
            self.session.flush()
            self.change_tracker.add_created('position', position.id)
        else:
            self.change_tracker.add_unchanged('position')

        # Add to cache
        self.position_cache[cache_key] = position
        return position

    def _process_drawing(self, row, skip_updates=False) -> Drawing:
        """Process drawing data from row"""
        drawing_number = str(row.get('DRAWING NUMBER', '')).strip() if row.get('DRAWING NUMBER') else ''

        # Check cache first
        if drawing_number in self.drawing_cache:
            drawing = self.drawing_cache[drawing_number]

            # Check if we need to update drawing information
            if not skip_updates:
                drawing_name = str(row.get('DRAWING NAME', '')) if row.get('DRAWING NAME') else ''
                revision = str(row.get('REVISION', '')) if row.get('REVISION') else ''
                equipment_name = str(row.get('EQUIPMENT NAME', '')) if row.get('EQUIPMENT NAME') else ''
                spare_part_number = str(row.get('SPARE PART NUMBER', '')) if row.get('SPARE PART NUMBER') else ''

                # Check if any fields have changed
                if (drawing.drw_name != drawing_name or
                        drawing.drw_revision != revision or
                        drawing.drw_equipment_name != equipment_name or
                        drawing.drw_spare_part_number != spare_part_number):

                    info_id(f"Updating Drawing: {drawing_number}")
                    drawing.drw_name = drawing_name
                    drawing.drw_revision = revision
                    drawing.drw_equipment_name = equipment_name
                    drawing.drw_spare_part_number = spare_part_number
                    self.change_tracker.add_updated('drawing', drawing.id, drawing_number)
                else:
                    self.change_tracker.add_unchanged('drawing')
            else:
                self.change_tracker.add_unchanged('drawing')

            return drawing

        # Not in cache, check database
        drawing = self.session.query(Drawing).filter(Drawing.drw_number == drawing_number).first()

        if drawing:
            # Drawing exists but not in cache
            if not skip_updates:
                # Check if we need to update
                drawing_name = str(row.get('DRAWING NAME', '')) if row.get('DRAWING NAME') else ''
                revision = str(row.get('REVISION', '')) if row.get('REVISION') else ''
                equipment_name = str(row.get('EQUIPMENT NAME', '')) if row.get('EQUIPMENT NAME') else ''
                spare_part_number = str(row.get('SPARE PART NUMBER', '')) if row.get('SPARE PART NUMBER') else ''

                # Check if any fields have changed
                if (drawing.drw_name != drawing_name or
                        drawing.drw_revision != revision or
                        drawing.drw_equipment_name != equipment_name or
                        drawing.drw_spare_part_number != spare_part_number):

                    info_id(f"Updating Drawing: {drawing_number}")
                    drawing.drw_name = drawing_name
                    drawing.drw_revision = revision
                    drawing.drw_equipment_name = equipment_name
                    drawing.drw_spare_part_number = spare_part_number
                    self.change_tracker.add_updated('drawing', drawing.id, drawing_number)
                else:
                    self.change_tracker.add_unchanged('drawing')
            else:
                self.change_tracker.add_unchanged('drawing')
        else:
            # Create new drawing
            drawing_name = str(row.get('DRAWING NAME', '')) if row.get('DRAWING NAME') else ''
            revision = str(row.get('REVISION', '')) if row.get('REVISION') else ''
            equipment_name = str(row.get('EQUIPMENT NAME', '')) if row.get('EQUIPMENT NAME') else ''
            spare_part_number = str(row.get('SPARE PART NUMBER', '')) if row.get('SPARE PART NUMBER') else ''

            info_id(f"Creating new Drawing: {drawing_number} - {drawing_name} (Rev: {revision})")
            drawing = Drawing(
                drw_number=drawing_number,
                drw_name=drawing_name,
                drw_revision=revision,
                drw_equipment_name=equipment_name,
                drw_spare_part_number=spare_part_number
            )
            self.session.add(drawing)
            self.session.flush()
            self.change_tracker.add_created('drawing', drawing.id, drawing_number)

        # Add to cache
        self.drawing_cache[drawing_number] = drawing
        return drawing

    def _process_spare_part(self, row, skip_updates=False) -> Part:
        """Process spare part data from row"""
        spare_part_number = str(row.get('SPARE PART NUMBER', '')).strip() if row.get('SPARE PART NUMBER') else ''

        # Check cache first
        if spare_part_number in self.part_cache:
            part = self.part_cache[spare_part_number]
            self.change_tracker.add_unchanged('part')
            return part

        # Check database
        part = self.session.query(Part).filter(Part.part_number == spare_part_number).first()

        if not part:
            # Create new part
            info_id(f"Creating new Part: {spare_part_number}")
            part = Part(
                part_number=spare_part_number,
                name=f"Part {spare_part_number}"
            )
            self.session.add(part)
            self.session.flush()
            self.change_tracker.add_created('part', part.id, spare_part_number)
        else:
            self.change_tracker.add_unchanged('part')

        # Add to cache
        self.part_cache[spare_part_number] = part
        return part

    def _associate_drawing_position(self, drawing, position, skip_updates=False):
        """Associate drawing with position"""
        assoc_key = f"dp:{drawing.id}:{position.id}"

        # Check cache first
        if assoc_key in self.assoc_cache:
            self.change_tracker.add_unchanged('associations')
            return self.assoc_cache[assoc_key]

        # Check if association already exists
        existing_assoc = (
            self.session.query(DrawingPositionAssociation)
            .filter(
                DrawingPositionAssociation.drawing_id == drawing.id,
                DrawingPositionAssociation.position_id == position.id
            )
            .first()
        )

        if not existing_assoc:
            # Create association
            try:
                drawing_position = DrawingPositionAssociation(
                    drawing_id=drawing.id,
                    position_id=position.id
                )
                self.session.add(drawing_position)
                self.session.flush()
                info_id(f"Associated drawing {drawing.id} with position {position.id}")
                self.change_tracker.add_created('drawing_position_assoc', drawing_position.id)
                self.assoc_cache[assoc_key] = drawing_position
                return drawing_position
            except Exception as e:
                error_id(f"Error associating drawing with position: {e}")
                raise
        else:
            self.change_tracker.add_unchanged('associations')
            self.assoc_cache[assoc_key] = existing_assoc
            return existing_assoc

    def _associate_drawing_part(self, drawing, part, skip_updates=False):
        """Associate drawing with part"""
        assoc_key = f"dp:{drawing.id}:{part.id}"

        # Check cache first
        if assoc_key in self.assoc_cache:
            self.change_tracker.add_unchanged('associations')
            return self.assoc_cache[assoc_key]

        # Check if association already exists
        existing_assoc = (
            self.session.query(DrawingPartAssociation)
            .filter(
                DrawingPartAssociation.drawing_id == drawing.id,
                DrawingPartAssociation.part_id == part.id
            )
            .first()
        )

        if not existing_assoc:
            # Create association
            try:
                drawing_part_assoc = DrawingPartAssociation(
                    drawing_id=drawing.id,
                    part_id=part.id
                )
                self.session.add(drawing_part_assoc)
                self.session.flush()
                info_id(f"Associated drawing {drawing.id} with part {part.id}")
                self.change_tracker.add_created('drawing_part_assoc', drawing_part_assoc.id)
                self.assoc_cache[assoc_key] = drawing_part_assoc
                return drawing_part_assoc
            except Exception as e:
                error_id(f"Error associating drawing with part: {e}")
                raise
        else:
            self.change_tracker.add_unchanged('associations')
            self.assoc_cache[assoc_key] = existing_assoc
            return existing_assoc

    def _associate_part_position(self, part, position, skip_updates=False):
        """Associate part with position"""
        assoc_key = f"pp:{part.id}:{position.id}"

        # Check cache first
        if assoc_key in self.assoc_cache:
            self.change_tracker.add_unchanged('associations')
            return self.assoc_cache[assoc_key]

        # Check if association already exists
        existing_assoc = (
            self.session.query(PartsPositionImageAssociation)
            .filter(
                PartsPositionImageAssociation.part_id == part.id,
                PartsPositionImageAssociation.position_id == position.id
            )
            .first()
        )

        if not existing_assoc:
            # Create association
            try:
                part_pos_assoc = PartsPositionImageAssociation(
                    part_id=part.id,
                    position_id=position.id,
                    image_id=None  # Can be updated later when an image is available
                )
                self.session.add(part_pos_assoc)
                self.session.flush()
                info_id(f"Associated part {part.id} with position {position.id}")
                self.change_tracker.add_created('part_position_assoc', part_pos_assoc.id)
                self.assoc_cache[assoc_key] = part_pos_assoc
                return part_pos_assoc
            except Exception as e:
                error_id(f"Error associating part with position: {e}")
                raise
        else:
            self.change_tracker.add_unchanged('associations')
            self.assoc_cache[assoc_key] = existing_assoc
            return existing_assoc

    def automate_part_position_associations(self):
        """
        Automatically create part-position associations based on drawing relationships.
        This fixes the issue where part_position_image entries are missing.

        When a drawing is associated with both a part and a position, there should be
        a part-position association as well.
        """
        info_id("Creating missing part-position associations based on drawing relationships")

        # Find all missing part-position associations using SQL for efficiency
        query = """
        SELECT DISTINCT dp.part_id, dpos.position_id
        FROM drawing_part dp
        JOIN drawing_position dpos ON dp.drawing_id = dpos.drawing_id
        LEFT JOIN part_position_image ppi ON dp.part_id = ppi.part_id AND dpos.position_id = ppi.position_id
        WHERE ppi.id IS NULL
        """

        try:
            result = self.session.execute(text(query))
            missing_associations = [(row[0], row[1]) for row in result]

            info_id(f"Found {len(missing_associations)} missing part-position associations")

            # Save the list of missing associations for reference
            if missing_associations:
                timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
                missing_file = os.path.join(DRAWING_IMPORT_DATA_DIR, f"missing_associations_{timestamp}.csv")
                pd.DataFrame(missing_associations, columns=['part_id', 'position_id']).to_csv(missing_file, index=False)
                info_id(f"Saved list of missing associations to {missing_file}")

            # Create the missing associations in batches
            batch_size = 100
            created_count = 0

            for i in range(0, len(missing_associations), batch_size):
                batch = missing_associations[i:i + batch_size]
                for part_id, position_id in batch:
                    try:
                        # Double check it's still missing (in case it was created in another session)
                        existing = (
                            self.session.query(PartsPositionImageAssociation)
                            .filter(
                                PartsPositionImageAssociation.part_id == part_id,
                                PartsPositionImageAssociation.position_id == position_id
                            )
                            .first()
                        )

                        if not existing:
                            new_assoc = PartsPositionImageAssociation(
                                part_id=part_id,
                                position_id=position_id,
                                image_id=None
                            )
                            self.session.add(new_assoc)
                            created_count += 1

                            # Add to change tracker
                            self.change_tracker.add_created('part_position_assoc', part_id)
                    except Exception as e:
                        error_id(f"Error creating part-position association ({part_id}, {position_id}): {e}")

                # Commit each batch
                self.session.flush()
                info_id(f"Created {created_count} part-position associations so far")

            if created_count > 0:
                info_id(f"Successfully created {created_count} missing part-position associations")
                self.session.commit()
            else:
                info_id("No missing part-position associations found")

            return created_count

        except Exception as e:
            error_id(f"Error finding or creating missing part-position associations: {e}", exc_info=True)
            self.session.rollback()
            return 0

    @with_request_id
    def process_data(self):
        """
        Process Excel data in batches with transaction management.
        Tracks changes for reporting.

        Returns:
            bool: True if successful, False otherwise
            str: Report of changes
        """
        if self.excel_data is None or len(self.excel_data) == 0:
            error_id("No Excel data loaded or data is empty")
            return False, "No data to process"

        try:
            info_id(f"Starting to process {len(self.excel_data)} rows of data in batches of {self.batch_size}")

            # Split data into batches
            total_rows = len(self.excel_data)
            num_batches = (total_rows + self.batch_size - 1) // self.batch_size
            current_hashes = {}

            for batch_num in range(num_batches):
                start_idx = batch_num * self.batch_size
                end_idx = min((batch_num + 1) * self.batch_size, total_rows)

                # Get current batch
                batch_df = self.excel_data.iloc[start_idx:end_idx]

                # Process batch
                batch_hashes = self._process_batch(batch_df, batch_num + 1)
                current_hashes.update(batch_hashes)

            # Create missing part-position associations
            created_associations = self.automate_part_position_associations()
            info_id(f"Created {created_associations} missing part-position associations")

            # Save current hashes for future comparison
            self._save_current_hashes(current_hashes)

            # Generate report
            report = self.change_tracker.generate_report()
            info_id("Completed processing all data")

            return True, report

        except Exception as e:
            error_id(f"Unexpected error while processing data: {e}", exc_info=True)
            self.session.rollback()
            return False, f"Error: {str(e)}"

    @with_request_id
    def close(self):
        """Close database session."""
        debug_id("Closing database session")
        if hasattr(self, 'session') and self.session:
            self.session.close()


def validate_specific_entries(session):
    """
    Validate and optionally repair specific database entries.
    Checks for the database entries shown in the screenshots.
    """
    info_id("Running validation on specific database entries")

    # Create validator
    validator = DataValidator(session)

    # Validate part_position_image table entries
    info_id("Validating part_position_image table...")
    expected_entries = [
        (1, 1), (2, 1), (3, 1), (4, 7), (5, 9), (3975, 9), (2299, 9)
    ]
    validator.validate_ids('part_position', expected_entries)

    # Validate drawing_part table entries
    info_id("Validating drawing_part table...")
    drawing_part_entries = [
        (6327, 5484), (6328, 5485), (6329, 5487), (17964, 2553),
        (25967, 18851), (25968, 18849), (64, 10317), (66, 3283),
        (67, 3284), (119, 3456), (120, 2071), (121, 2085),
        (121, 2086), (121, 2087), (122, 3484), (123, 3454),
        (152, 6520), (155, 4085), (161, 2229), (256, 11200)
    ]
    validator.validate_ids('drawing_part', drawing_part_entries)

    # Validate drawing_position table entries
    info_id("Validating drawing_position table...")
    drawing_position_entries = [
        (3707, 1), (3708, 1), (3709, 1), (3377, 9),
        (504, 9), (1, 9), (2, 9), (408, 18), (415, 18),
        (2244, 19)
    ]
    validator.validate_ids('drawing_position', drawing_position_entries)

    # Check for drawing-part-position relationships that should exist but don't have part_position entries
    info_id("Checking for missing part-position relationships...")
    missing_part_position_entries = find_missing_part_position_entries(session)

    if missing_part_position_entries:
        info_id(f"Found {len(missing_part_position_entries)} missing part-position relationships")

        # Ask if user wants to create these
        answer = input("\nDo you want to create these missing part-position relationships? (y/n): ")
        if answer.lower() in ('y', 'yes'):
            create_missing_part_position_entries(session, missing_part_position_entries)
            info_id("Created missing part-position relationships")
        else:
            info_id("Skipped creating missing relationships")
    else:
        info_id("No missing part-position relationships found")

    # Generate validation report
    report = validator.generate_validation_report()

    print("\n" + "=" * 80)
    print(report)
    print("=" * 80 + "\n")

    # Save report to file
    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    report_file = os.path.join(DRAWING_IMPORT_DATA_DIR, f"validation_report_{timestamp}.txt")
    with open(report_file, 'w') as f:
        f.write(report)
    info_id(f"Validation report saved to {report_file}")

    return True


def find_missing_part_position_entries(session):
    """
    Find drawing-part-position relationships that should have part_position entries but don't.
    Returns a list of (part_id, position_id) tuples that are missing.
    """
    info_id("Scanning for missing part-position relationships...")

    # This SQL query finds parts and positions that are linked through drawings
    # but don't have entries in the part_position_image table
    query = """
    SELECT DISTINCT dp.part_id, dpos.position_id
    FROM drawing_part dp
    JOIN drawing_position dpos ON dp.drawing_id = dpos.drawing_id
    LEFT JOIN part_position_image ppi ON dp.part_id = ppi.part_id AND dpos.position_id = ppi.position_id
    WHERE ppi.id IS NULL
    """

    try:
        # Execute the query
        result = session.execute(text(query))
        missing_entries = [(row[0], row[1]) for row in result]
        info_id(f"Found {len(missing_entries)} missing part-position relationships")

        # Get details about the missing entries
        if missing_entries:
            debug_id("Missing part-position relationships details:")
            for part_id, position_id in missing_entries[:20]:  # Limit to 20 for log readability
                debug_id(f"  Part ID: {part_id}, Position ID: {position_id}")

            if len(missing_entries) > 20:
                debug_id(f"  ... and {len(missing_entries) - 20} more")

        return missing_entries

    except Exception as e:
        error_id(f"Error finding missing part-position relationships: {e}", exc_info=True)
        return []


def create_missing_part_position_entries(session, missing_entries):
    """
    Create missing part_position_image entries based on the list of missing (part_id, position_id) pairs.

    Args:
        session: SQLAlchemy session
        missing_entries: List of (part_id, position_id) tuples
    """
    info_id(f"Creating {len(missing_entries)} missing part-position relationships...")

    batch_size = 100
    created_count = 0
    error_count = 0

    # Process in batches to avoid overloading the database
    for i in range(0, len(missing_entries), batch_size):
        batch = missing_entries[i:i + batch_size]
        try:
            for part_id, position_id in batch:
                try:
                    # Check if it already exists (might have been created in a previous batch)
                    existing = (
                        session.query(PartsPositionImageAssociation)
                        .filter(
                            PartsPositionImageAssociation.part_id == part_id,
                            PartsPositionImageAssociation.position_id == position_id
                        )
                        .first()
                    )

                    if not existing:
                        # Create the association
                        assoc = PartsPositionImageAssociation(
                            part_id=part_id,
                            position_id=position_id,
                            image_id=None  # No image initially
                        )
                        session.add(assoc)
                        created_count += 1
                except Exception as e:
                    error_id(f"Error creating part-position relationship ({part_id}, {position_id}): {e}")
                    error_count += 1

            # Commit this batch
            session.commit()
            info_id(
                f"Committed batch {i // batch_size + 1}/{(len(missing_entries) - 1) // batch_size + 1}, created {created_count} entries so far")

        except Exception as e:
            session.rollback()
            error_id(f"Error processing batch {i // batch_size + 1}: {e}", exc_info=True)
            error_count += len(batch)

    info_id(f"Created {created_count} part-position relationships, encountered {error_count} errors")


def repair_database_relationships(session):
    """
    Repair database relationships by creating missing part-position entries.
    This is a more comprehensive fix that scans the entire database.
    """
    info_id("Starting database relationship repair...")

    # Find missing part-position entries
    missing_entries = find_missing_part_position_entries(session)

    if missing_entries:
        info_id(f"Found {len(missing_entries)} missing part-position relationships")
        create_missing_part_position_entries(session, missing_entries)
    else:
        info_id("No missing relationships found, database is consistent")

    return True


@with_request_id
def main():
    """Main entry point for the script."""
    # Ensure all required directories exist
    ensure_directories_exist()

    # Use the database URL from the config module
    db_url = DATABASE_URL

    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Import drawing data from Excel to database')
    parser.add_argument('--file', '-f', help='Path to Excel file to import',
                        default=os.path.join(DB_LOADSHEET, "Active Drawing List breakdown.xlsx"))
    parser.add_argument('--batch-size', '-b', type=int, default=50, help='Batch size for processing')
    parser.add_argument('--validate', '-v', action='store_true', help='Run in validation mode to check entries')
    parser.add_argument('--repair', '-r', action='store_true', help='Repair missing relationships')
    parser.add_argument('--force-create', '-c', action='store_true',
                        help='Force create missing part-position entries')

    args = parser.parse_args()
    excel_file = args.file
    batch_size = args.batch_size
    validation_mode = args.validate
    repair_mode = args.repair
    force_create = args.force_create

    info_id(f"Excel file path: {excel_file}")
    info_id(f"Using batch size: {batch_size}")

    # Log information about execution environment
    info_id(f"Running from directory: {os.getcwd()}")
    info_id(f"Base directory: {BASE_DIR}")
    info_id(f"Database directory: {DATABASE_DIR}")
    info_id(f"DB_LOADSHEET directory: {DB_LOADSHEET}")
    info_id(f"Drawing import data directory: {DRAWING_IMPORT_DATA_DIR}")

    # Create mapper and process data
    mapper = ExcelToDbMapper(db_url, batch_size=batch_size)
    try:
        # If in validation mode, perform validations on specific IDs
        if validation_mode:
            info_id("Running in validation mode")
            validate_specific_entries(mapper.session)
            return

        # If in repair mode, fix missing relationships
        if repair_mode:
            info_id("Running in repair mode")
            repair_database_relationships(mapper.session)
            return

        # If force create mode, create part-position entries
        if force_create:
            info_id("Running in force create mode")
            missing_entries = find_missing_part_position_entries(mapper.session)
            if missing_entries:
                create_missing_part_position_entries(mapper.session, missing_entries)
            return

        # Normal import mode
        with log_timed_operation("Excel Import Process"):
            info_id(f"Starting Excel import process for file: {excel_file}")
            if mapper.load_excel(excel_file):
                success, report = mapper.process_data()
                if success:
                    info_id("Data mapping completed successfully")

                    # Print and save the report
                    print("\n" + "=" * 80)
                    print(report)
                    print("=" * 80 + "\n")

                    # Save report to file
                    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
                    report_file = os.path.join(DRAWING_IMPORT_DATA_DIR, f"import_report_{timestamp}.txt")
                    with open(report_file, 'w') as f:
                        f.write(report)
                    info_id(f"Report saved to {report_file}")
                else:
                    error_id("Data mapping failed")
                    print(f"Error: {report}")
            else:
                error_id("Failed to load Excel file")
    except Exception as e:
        error_id(f"Unhandled exception in main: {e}", exc_info=True)
    finally:
        mapper.close()


if __name__ == "__main__":
    main()

# Ensure the directory exists
def ensure_directories_exist():
    """Ensure all required directories exist."""
    directories = [
        DB_LOADSHEET,
        DRAWING_IMPORT_DATA_DIR
    ]

    for directory in directories:
        if not os.path.exists(directory):
            try:
                os.makedirs(directory)
                info_id(f"Created directory: {directory}")
            except Exception as e:
                error_id(f"Error creating directory {directory}: {e}")


# Call this at the beginning to make sure the directory exists
ensure_directories_exist()

# Import database models
from modules.emtacdb.emtacdb_fts import (
    Area, EquipmentGroup, Model, AssetNumber, Location, Position, Drawing, Part,
    DrawingPositionAssociation, DrawingPartAssociation, PartsPositionImageAssociation
)

# Use the module's logger
from modules.configuration.log_config import logger


class ChangeTracker:
    """
    Tracks changes made during database operations for reporting.
    """

    def __init__(self):
        self.created = {
            'area': [],
            'equipment_group': [],
            'model': [],
            'location': [],
            'position': [],
            'drawing': [],
            'part': [],
            'drawing_position_assoc': [],
            'drawing_part_assoc': [],
            'part_position_assoc': []
        }
        self.updated = {
            'drawing': [],
            'part': []
        }
        self.unchanged = {
            'drawing': 0,
            'position': 0,
            'part': 0,
            'associations': 0
        }
        self.errors = []
        self.processed_rows = 0
        self.skipped_rows = 0

    def add_created(self, entity_type, entity_id, name=None):
        """Record a created entity"""
        if name:
            self.created[entity_type].append((entity_id, name))
        else:
            self.created[entity_type].append(entity_id)

    def add_updated(self, entity_type, entity_id, name=None):
        """Record an updated entity"""
        if name:
            self.updated[entity_type].append((entity_id, name))
        else:
            self.updated[entity_type].append(entity_id)

    def add_unchanged(self, entity_type):
        """Record an unchanged entity"""
        self.unchanged[entity_type] += 1

    def add_error(self, row_idx, error_msg):
        """Record an error"""
        self.errors.append((row_idx, error_msg))

    def generate_report(self):
        """Generate a detailed report of changes"""
        report = []
        report.append(f"Change Report - {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("=" * 80)
        report.append(f"Processed {self.processed_rows} rows, skipped {self.skipped_rows} rows")
        report.append("-" * 80)

        report.append("\nCreated Entities:")
        for entity_type, entities in self.created.items():
            if entities:
                if isinstance(entities[0], tuple):
                    report.append(f"  {entity_type.replace('_', ' ').title()}: {len(entities)}")
                    for entity_id, name in entities[:10]:  # Limit to first 10 for readability
                        report.append(f"    - ID: {entity_id}, Name: {name}")
                    if len(entities) > 10:
                        report.append(f"    ... and {len(entities) - 10} more")
                else:
                    report.append(f"  {entity_type.replace('_', ' ').title()}: {len(entities)}")

        report.append("\nUpdated Entities:")
        for entity_type, entities in self.updated.items():
            if entities:
                if isinstance(entities[0], tuple):
                    report.append(f"  {entity_type.replace('_', ' ').title()}: {len(entities)}")
                    for entity_id, name in entities[:10]:
                        report.append(f"    - ID: {entity_id}, Name: {name}")
                    if len(entities) > 10:
                        report.append(f"    ... and {len(entities) - 10} more")
                else:
                    report.append(f"  {entity_type.replace('_', ' ').title()}: {len(entities)}")

        report.append("\nUnchanged Entities:")
        for entity_type, count in self.unchanged.items():
            if count > 0:
                report.append(f"  {entity_type.replace('_', ' ').title()}: {count}")

        if self.errors:
            report.append("\nErrors:")
            for row_idx, error_msg in self.errors:
                report.append(f"  Row {row_idx}: {error_msg}")

        return "\n".join(report)


class DataValidator:
    """
    Validates specific database entries and reports on missing or inconsistent data.
    """

    def __init__(self, session):
        """Initialize with database session"""
        self.session = session
        self.missing_entries = []
        self.inconsistent_entries = []

    def validate_ids(self, entity_type, id_pairs, relationship_type=None):
        """
        Validate specific ID pairs in the database

        Args:
            entity_type: Type of entity to check ('position', 'drawing', 'part', etc.)
            id_pairs: List of ID pairs to check [(id1, id2), ...]
            relationship_type: Type of relationship to check ('drawing_position', 'part_position', etc.)
        """
        if entity_type == 'drawing_position':
            self._validate_drawing_position_pairs(id_pairs)
        elif entity_type == 'drawing_part':
            self._validate_drawing_part_pairs(id_pairs)
        elif entity_type == 'part_position':
            self._validate_part_position_pairs(id_pairs)
        elif entity_type == 'hierarchy':
            self._validate_hierarchy_ids(id_pairs)
        else:
            info_id(f"Validation for {entity_type} not implemented")

    def _validate_drawing_position_pairs(self, drawing_position_pairs):
        """Check if drawing-position associations exist"""
        for drawing_id, position_id in drawing_position_pairs:
            try:
                assoc = (
                    self.session.query(DrawingPositionAssociation)
                    .filter(
                        DrawingPositionAssociation.drawing_id == drawing_id,
                        DrawingPositionAssociation.position_id == position_id
                    )
                    .first()
                )

                if not assoc:
                    self.missing_entries.append({
                        'type': 'drawing_position',
                        'ids': (drawing_id, position_id),
                        'message': f"Missing association between drawing ID {drawing_id} and position ID {position_id}"
                    })
                else:
                    debug_id(f"Found association between drawing {drawing_id} and position {position_id}")
            except Exception as e:
                error_id(f"Error validating drawing-position pair ({drawing_id}, {position_id}): {e}")

    def _validate_drawing_part_pairs(self, drawing_part_pairs):
        """Check if drawing-part associations exist"""
        for drawing_id, part_id in drawing_part_pairs:
            try:
                assoc = (
                    self.session.query(DrawingPartAssociation)
                    .filter(
                        DrawingPartAssociation.drawing_id == drawing_id,
                        DrawingPartAssociation.part_id == part_id
                    )
                    .first()
                )

                if not assoc:
                    self.missing_entries.append({
                        'type': 'drawing_part',
                        'ids': (drawing_id, part_id),
                        'message': f"Missing association between drawing ID {drawing_id} and part ID {part_id}"
                    })
                else:
                    debug_id(f"Found association between drawing {drawing_id} and part {part_id}")
            except Exception as e:
                error_id(f"Error validating drawing-part pair ({drawing_id}, {part_id}): {e}")

    def _validate_part_position_pairs(self, part_position_pairs):
        """Check if part-position associations exist"""
        for part_id, position_id in part_position_pairs:
            try:
                assoc = (
                    self.session.query(PartsPositionImageAssociation)
                    .filter(
                        PartsPositionImageAssociation.part_id == part_id,
                        PartsPositionImageAssociation.position_id == position_id
                    )
                    .first()
                )

                if not assoc:
                    self.missing_entries.append({
                        'type': 'part_position',
                        'ids': (part_id, position_id),
                        'message': f"Missing association between part ID {part_id} and position ID {position_id}"
                    })
                else:
                    debug_id(f"Found association between part {part_id} and position {position_id}")
            except Exception as e:
                error_id(f"Error validating part-position pair ({part_id}, {position_id}): {e}")

    def _validate_hierarchy_ids(self, hierarchy_ids):
        """Check if hierarchies (area, equipment_group, model, location) exist"""
        for area_id, equipment_group_id, model_id, location_id in hierarchy_ids:
            try:
                # Check if position with this hierarchy exists
                position = (
                    self.session.query(Position)
                    .filter(
                        Position.area_id == area_id,
                        Position.equipment_group_id == equipment_group_id,
                        Position.model_id == model_id,
                        Position.location_id == location_id
                    )
                    .first()
                )

                if not position:
                    self.missing_entries.append({
                        'type': 'hierarchy',
                        'ids': (area_id, equipment_group_id, model_id, location_id),
                        'message': f"Missing position for hierarchy: Area {area_id}, EG {equipment_group_id}, Model {model_id}, Location {location_id}"
                    })
                else:
                    debug_id(
                        f"Found position {position.id} for hierarchy {area_id}/{equipment_group_id}/{model_id}/{location_id}")
            except Exception as e:
                error_id(
                    f"Error validating hierarchy ({area_id}, {equipment_group_id}, {model_id}, {location_id}): {e}")

    def validate_entity_existence(self, entity_type, ids):
        """Check if specific entities exist by ID"""
        model_map = {
            'area': Area,
            'equipment_group': EquipmentGroup,
            'model': Model,
            'location': Location,
            'position': Position,
            'drawing': Drawing,
            'part': Part
        }

        if entity_type not in model_map:
            warning_id(f"Unknown entity type: {entity_type}")
            return

        Model = model_map[entity_type]

        for entity_id in ids:
            try:
                entity = self.session.query(Model).filter(Model.id == entity_id).first()
                if not entity:
                    self.missing_entries.append({
                        'type': entity_type,
                        'ids': (entity_id,),
                        'message': f"Missing {entity_type} with ID {entity_id}"
                    })
                else:
                    debug_id(f"Found {entity_type} with ID {entity_id}")
            except Exception as e:
                error_id(f"Error validating {entity_type} ID {entity_id}: {e}")

    def generate_validation_report(self):
        """Generate a report of validation findings"""
        if not self.missing_entries and not self.inconsistent_entries:
            return "All validated entries were found in the database."

        report = []
        report.append("Validation Report")
        report.append("=" * 80)

        if self.missing_entries:
            report.append("\nMissing Entries:")
            for entry in self.missing_entries:
                report.append(f"  • {entry['message']}")

        if self.inconsistent_entries:
            report.append("\nInconsistent Entries:")
            for entry in self.inconsistent_entries:
                report.append(f"  • {entry['message']}")

        return "\n".join(report)


class ExcelToDbMapper:
    """
    An optimized class to map Excel data to database models with change tracking and batch processing.
    """

    def __init__(self, db_url=None, batch_size=50):
        """
        Initialize with database connection using DatabaseConfig.

        Args:
            db_url (str, optional): SQLAlchemy database URL (kept for compatibility, not used)
            batch_size (int): Number of rows to process in a single transaction
        """
        # Set a request ID for this instance
        self.request_id = set_request_id()
        info_id(f"Initializing ExcelToDbMapper with DatabaseConfig", self.request_id)

        # For backward compatibility, log the provided db_url if present
        if db_url:
            debug_id(f"Note: Provided db_url: {db_url} will be ignored in favor of DatabaseConfig")

        # Create DB config and get a session
        self.db_config = DatabaseConfig()
        self.session = self.db_config.get_main_session()
        self.excel_data = None
        self.batch_size = batch_size
        self.change_tracker = ChangeTracker()

        # Caches to reduce database queries
        self.area_cache = {}
        self.equipment_group_cache = {}
        self.model_cache = {}
        self.location_cache = {}
        self.position_cache = {}
        self.drawing_cache = {}
        self.part_cache = {}
        self.assoc_cache = {}

        # Track row hashes for change detection
        self.previous_row_hashes = self._load_previous_hashes()

    def _load_previous_hashes(self) -> Dict[str, str]:
        """
        Load previously processed row hashes from database or file.
        Used to detect changes in the Excel file.
        """
        hash_dict = {}
        hash_file = os.path.join(DRAWING_IMPORT_DATA_DIR, ".row_hashes.csv")

        if os.path.exists(hash_file):
            try:
                hash_df = pd.read_csv(hash_file)
                for _, row in hash_df.iterrows():
                    hash_dict[row['row_key']] = row['hash']
                info_id(f"Loaded {len(hash_dict)} previous row hashes from {hash_file}")
            except Exception as e:
                warning_id(f"Could not load previous hashes: {e}")

        return hash_dict

    def _save_current_hashes(self, current_hashes: Dict[str, str]):
        """Save the current row hashes for future comparison"""
        # Use the drawing_import_data directory for storing hash files
        hash_file = os.path.join(DRAWING_IMPORT_DATA_DIR, ".row_hashes.csv")

        try:
            hash_df = pd.DataFrame([
                {'row_key': key, 'hash': value}
                for key, value in current_hashes.items()
            ])
            hash_df.to_csv(hash_file, index=False)
            info_id(f"Saved {len(current_hashes)} row hashes to {hash_file}")
        except Exception as e:
            warning_id(f"Could not save current hashes: {e}")

    def _generate_row_hash(self, row) -> Tuple[str, str]:
        """
        Generate a unique hash for a row to detect changes.
        Returns (row_key, hash_value)
        """
        # Create a unique key for this row based on hierarchical data
        area = str(row.get('area', '')).strip() if pd.notna(row.get('area', '')) else ''
        equip_group = str(row.get('equipment_group', '')).strip() if pd.notna(row.get('equipment_group', '')) else ''
        model = str(row.get('model', '')).strip() if pd.notna(row.get('model', '')) else ''
        stations = str(row.get('stations', '')).strip() if pd.notna(row.get('stations', '')) else ''
        drawing = str(row.get('DRAWING NUMBER', '')).strip() if pd.notna(row.get('DRAWING NUMBER', '')) else ''

        # Create a row key that uniquely identifies this hierarchy
        row_key = f"{area}|{equip_group}|{model}|{stations}|{drawing}"

        # Create a hash of all row values to detect any changes
        row_values_list = []
        for column, value in row.items():
            # Handle different types properly
            if pd.isna(value):
                row_values_list.append('')
            elif isinstance(value, (np.ndarray, list, tuple)):
                # Convert arrays/lists/tuples to string representation
                row_values_list.append(str(list(value)))
            else:
                # Regular values
                row_values_list.append(str(value).strip())

        row_values = '|'.join(row_values_list)
        hash_value = hashlib.md5(row_values.encode()).hexdigest()

        return row_key, hash_value

    def _preload_caches(self):
        """Preload frequently accessed data to reduce database queries"""
        with log_timed_operation("Preloading database caches"):
            # Load areas
            areas = self.session.query(Area).all()
            for area in areas:
                self.area_cache[area.name] = area
            info_id(f"Preloaded {len(areas)} areas into cache")

            # Load drawings
            drawings = self.session.query(Drawing).all()
            for drawing in drawings:
                self.drawing_cache[drawing.drw_number] = drawing
            info_id(f"Preloaded {len(drawings)} drawings into cache")

            # Load parts
            parts = self.session.query(Part).all()
            for part in parts:
                self.part_cache[part.part_number] = part
            info_id(f"Preloaded {len(parts)} parts into cache")

            # We don't preload all equipment groups, models and locations
            # as there could be too many, but we'll cache them as we go

    @with_request_id
    def load_excel(self, file_path):
        """
        Load data from Excel file.

        Args:
            file_path (str): Path to the Excel file

        Returns:
            bool: True if successful, False otherwise
        """
        try:
            info_id(f"Loading Excel file: {file_path}")

            # Verify file exists
            if not os.path.exists(file_path):
                error_id(f"Excel file not found: {file_path}")
                return False

            # Load the Excel file with correct data types
            self.excel_data = pd.read_excel(file_path, dtype=str)

            # Clean the data - remove rows with all NaN values
            self.excel_data = self.excel_data.dropna(how='all')

            # Fill NaN values with empty strings for all columns
            self.excel_data = self.excel_data.fillna('')

            info_id(f"Successfully loaded {len(self.excel_data)} rows")

            # Log column headers for verification
            headers = list(self.excel_data.columns)
            debug_id(f"Excel headers: {headers}")

            # Preload database caches
            self._preload_caches()

            return True
        except Exception as e:
            error_id(f"Error loading Excel file: {e}", exc_info=True)
            return False

    def _batch_safe_commit(self, batch_num, rows_processed):
        """
        Safely commit the current transaction with error handling.
        If commit fails, log the error and return False.
        """
        try:
            with log_timed_operation(f"Committing batch {batch_num}"):
                self.session.commit()
                info_id(f"Successfully committed batch {batch_num} with {rows_processed} rows")
            return True
        except SQLAlchemyError as e:
            self.session.rollback()
            error_id(f"Database error in batch {batch_num}: {e}", exc_info=True)
            return False
        except Exception as e:
            self.session.rollback()
            error_id(f"Unexpected error in batch {batch_num}: {e}", exc_info=True)
            return False

    def _process_batch(self, batch_df, batch_num):
        """Process a batch of Excel rows in a single transaction"""
        info_id(f"Processing batch {batch_num} with {len(batch_df)} rows")
        current_hashes = {}
        processed_count = 0
        max_attempts = 3  # Number of retry attempts for database issues

        for attempt in range(1, max_attempts + 1):
            try:
                # Clear out any uncommitted changes from previous failed attempts
                if attempt > 1:
                    self.session.rollback()
                    info_id(f"Retry attempt {attempt} for batch {batch_num}")

                # Process each row in the batch
                for index, row in batch_df.iterrows():
                    try:
                        # Get row hash to check for changes
                        row_key, row_hash = self._generate_row_hash(row)
                        current_hashes[row_key] = row_hash

                        # Check if row is unchanged from previous run
                        if row_key in self.previous_row_hashes and self.previous_row_hashes[row_key] == row_hash:
                            debug_id(f"Row {index + 1} is unchanged, skipping detailed processing")
                            # Still process it to ensure references exist, but don't need to update
                            self._process_row(row, index, skip_updates=True)
                        else:
                            # Process row normally
                            self._process_row(row, index)

                        processed_count += 1
                        self.change_tracker.processed_rows += 1

                    except Exception as e:
                        error_id(f"Error processing row {index + 1}: {e}", exc_info=True)
                        self.change_tracker.add_error(index + 1, str(e))
                        self.change_tracker.skipped_rows += 1

                # Commit the batch transaction
                if self._batch_safe_commit(batch_num, processed_count):
                    return current_hashes
                else:
                    # If commit failed but we have more attempts, try again
                    if attempt < max_attempts:
                        warning_id(
                            f"Commit failed for batch {batch_num}, will retry (attempt {attempt}/{max_attempts})")
                        # Reset processed count for next attempt
                        processed_count = 0
                        continue
                    else:
                        error_id(f"Failed to commit batch {batch_num} after {max_attempts} attempts")
                        self.change_tracker.skipped_rows += len(batch_df)
                        return {}

            except Exception as e:
                self.session.rollback()
                error_id(f"Error processing batch {batch_num} (attempt {attempt}): {e}", exc_info=True)

                # If we have more attempts, try again
                if attempt < max_attempts:
                    warning_id(
                        f"Processing failed for batch {batch_num}, will retry (attempt {attempt}/{max_attempts})")
                    # Reset processed count for next attempt
                    processed_count = 0
                    continue
                else:
                    # Add error for the entire batch
                    self.change_tracker.add_error(f"Batch {batch_num}", str(e))
                    self.change_tracker.skipped_rows += len(batch_df)
                    return {}

    def _process_row(self, row, index, skip_updates=False):
        """Process a single row of Excel data"""
        with log_timed_operation(f"Processing row {index + 1}"):
            debug_id(f"Processing row {index + 1}")

            # 1. Validate all required fields are present
            required_fields = ['area', 'equipment_group', 'model', 'stations']
            missing_fields = []

            for field in required_fields:
                value = row.get(field)
                if value is None or str(value).strip() == '':
                    missing_fields.append(field)

            if missing_fields:
                warning_id(f"Skipping row {index + 1}: Missing required fields: {', '.join(missing_fields)}")
                self.change_tracker.skipped_rows += 1
                return

            # Process area
            area = self._process_area(row, skip_updates)

            # Process equipment group
            equipment_group = self._process_equipment_group(row, area, skip_updates)

            # Process model
            model = self._process_model(row, equipment_group, skip_updates)

            # Process location
            location = self._process_location(row, model, skip_updates)

            # Process position
            position = self._process_position(area, equipment_group, model, location, skip_updates)

            # Process drawing if present
            drawing = None
            drawing_number = row.get('DRAWING NUMBER')
            if drawing_number and str(drawing_number).strip():
                drawing = self._process_drawing(row, skip_updates)

            # Associate drawing with position if both exist
            if drawing and position:
                self._associate_drawing_position(drawing, position, skip_updates)

            # Process spare part if present
            part = None
            spare_part_number = row.get('SPARE PART NUMBER')
            if drawing and spare_part_number and str(spare_part_number).strip():
                part = self._process_spare_part(row, skip_updates)

                # Associate drawing with part
                if part:
                    self._associate_drawing_part(drawing, part, skip_updates)

                    # Associate part with position
                    if position:
                        self._associate_part_position(part, position, skip_updates)

    def _process_area(self, row, skip_updates=False) -> Area:
        """Process area data from row"""
        area_name = str(row.get('area', '')).strip() if row.get('area') else ''

        # Check cache first
        if area_name in self.area_cache:
            self.change_tracker.add_unchanged('area')
            return self.area_cache[area_name]

        # Check database
        area = self.session.query(Area).filter(Area.name == area_name).first()

        # Create if not exists
        if not area:
            info_id(f"Creating new Area: {area_name}")
            area = Area(name=area_name)
            self.session.add(area)
            self.session.flush()
            self.change_tracker.add_created('area', area.id, area_name)
        else:
            self.change_tracker.add_unchanged('area')

        # Add to cache
        self.area_cache[area_name] = area
        return area

    def _process_equipment_group(self, row, area, skip_updates=False) -> EquipmentGroup:
        """Process equipment group data from row"""
        equipment_group_name = str(row.get('equipment_group', '')).strip() if row.get('equipment_group') else ''
        cache_key = f"{area.id}:{equipment_group_name}"

        # Check cache first
        if cache_key in self.equipment_group_cache:
            self.change_tracker.add_unchanged('equipment_group')
            return self.equipment_group_cache[cache_key]

        # Check database
        equipment_group = (
            self.session.query(EquipmentGroup)
            .filter(
                EquipmentGroup.name == equipment_group_name,
                EquipmentGroup.area_id == area.id
            )
            .first()
        )

        # Create if not exists
        if not equipment_group:
            info_id(f"Creating new EquipmentGroup: {equipment_group_name} in area {area.id}")
            equipment_group = EquipmentGroup(name=equipment_group_name, area_id=area.id)
            self.session.add(equipment_group)
            self.session.flush()
            self.change_tracker.add_created('equipment_group', equipment_group.id, equipment_group_name)
        else:
            self.change_tracker.add_unchanged('equipment_group')

        # Add to cache
        self.equipment_group_cache[cache_key] = equipment_group
        return equipment_group

    def _process_model(self, row, equipment_group, skip_updates=False) -> Model:
        """Process model data from row"""
        model_name = str(row.get('model', '')).strip() if row.get('model') else ''
        cache_key = f"{equipment_group.id}:{model_name}"

        # Check cache first
        if cache_key in self.model_cache:
            self.change_tracker.add_unchanged('model')
            return self.model_cache[cache_key]

        # Check database
        model = (
            self.session.query(Model)
            .filter(
                Model.name == model_name,
                Model.equipment_group_id == equipment_group.id
            )
            .first()
        )

        # Create if not exists
        if not model:
            info_id(f"Creating new Model: {model_name} in equipment group {equipment_group.id}")
            model = Model(name=model_name, equipment_group_id=equipment_group.id)
            self.session.add(model)
            self.session.flush()
            self.change_tracker.add_created('model', model.id, model_name)
        else:
            self.change_tracker.add_unchanged('model')

        # Add to cache
        self.model_cache[cache_key] = model
        return model

    def _process_location(self, row, model, skip_updates=False) -> Location:
        """Process location data from row"""
        stations_value = str(row.get('stations', '')).strip() if row.get('stations') else ''

        # Extract only the first part before any comma
        if ',' in stations_value:
            location_name = stations_value.split(',')[0].strip()
            debug_id(f"Using only first location from comma-separated list: {location_name}")
        else:
            location_name = stations_value

        cache_key = f"{model.id}:{location_name}"

        # Check cache first
        if cache_key in self.location_cache:
            self.change_tracker.add_unchanged('location')
            return self.location_cache[cache_key]

        # Check database
        location = (
            self.session.query(Location)
            .filter(
                Location.name == location_name,
                Location.model_id == model.id
            )
            .first()
        )

        # Create if not exists
        if not location:
            info_id(f"Creating new Location: {location_name} for model {model.id}")
            location = Location(name=location_name, model_id=model.id)
            self.session.add(location)
            self.session.flush()
            self.change_tracker.add_created('location', location.id, location_name)
        else:
            self.change_tracker.add_unchanged('location')

        # Add to cache
        self.location_cache[cache_key] = location
        return location

    def _process_position(self, area, equipment_group, model, location, skip_updates=False) -> Position:
        """Process position data"""
        cache_key = f"{area.id}:{equipment_group.id}:{model.id}:{location.id}"

        # Check cache first
        if cache_key in self.position_cache:
            self.change_tracker.add_unchanged('position')
            return self.position_cache[cache_key]

        # Check database
        position = (
            self.session.query(Position)
            .filter(
                Position.area_id == area.id,
                Position.equipment_group_id == equipment_group.id,
                Position.model_id == model.id,
                Position.location_id == location.id
            )
            .first()
        )

        # Create if not exists
        if not position:
            info_id(
                f"Creating new Position with hierarchy {area.name}/{equipment_group.name}/{model.name}/{location.name}")
            position = Position(
                area_id=area.id,
                equipment_group_id=equipment_group.id,
                model_id=model.id,
                location_id=location.id
            )
            self.session.add(position)
            self.session.flush()
            self.change_tracker.add_created('position', position.id)
        else:
            self.change_tracker.add_unchanged('position')

        # Add to cache
        self.position_cache[cache_key] = position
        return position

    def _process_drawing(self, row, skip_updates=False) -> Drawing:
        """Process drawing data from row"""
        drawing_number = str(row.get('DRAWING NUMBER', '')).strip() if row.get('DRAWING NUMBER') else ''

        # Check cache first
        if drawing_number in self.drawing_cache:
            drawing = self.drawing_cache[drawing_number]

            # Check if we need to update drawing information
            if not skip_updates:
                drawing_name = str(row.get('DRAWING NAME', '')) if row.get('DRAWING NAME') else ''
                revision = str(row.get('REVISION', '')) if row.get('REVISION') else ''
                equipment_name = str(row.get('EQUIPMENT NAME', '')) if row.get('EQUIPMENT NAME') else ''
                spare_part_number = str(row.get('SPARE PART NUMBER', '')) if row.get('SPARE PART NUMBER') else ''

                # Check if any fields have changed
                if (drawing.drw_name != drawing_name or
                        drawing.drw_revision != revision or
                        drawing.drw_equipment_name != equipment_name or
                        drawing.drw_spare_part_number != spare_part_number):

                    info_id(f"Updating Drawing: {drawing_number}")
                    drawing.drw_name = drawing_name
                    drawing.drw_revision = revision
                    drawing.drw_equipment_name = equipment_name
                    drawing.drw_spare_part_number = spare_part_number
                    self.change_tracker.add_updated('drawing', drawing.id, drawing_number)
                else:
                    self.change_tracker.add_unchanged('drawing')
            else:
                self.change_tracker.add_unchanged('drawing')

            return drawing

        # Not in cache, check database
        drawing = self.session.query(Drawing).filter(Drawing.drw_number == drawing_number).first()

        if drawing:
            # Drawing exists but not in cache
            if not skip_updates:
                # Check if we need to update
                drawing_name = str(row.get('DRAWING NAME', '')) if row.get('DRAWING NAME') else ''
                revision = str(row.get('REVISION', '')) if row.get('REVISION') else ''
                equipment_name = str(row.get('EQUIPMENT NAME', '')) if row.get('EQUIPMENT NAME') else ''
                spare_part_number = str(row.get('SPARE PART NUMBER', '')) if row.get('SPARE PART NUMBER') else ''

                # Check if any fields have changed
                if (drawing.drw_name != drawing_name or
                        drawing.drw_revision != revision or
                        drawing.drw_equipment_name != equipment_name or
                        drawing.drw_spare_part_number != spare_part_number):

                    info_id(f"Updating Drawing: {drawing_number}")
                    drawing.drw_name = drawing_name
                    drawing.drw_revision = revision
                    drawing.drw_equipment_name = equipment_name
                    drawing.drw_spare_part_number = spare_part_number
                    self.change_tracker.add_updated('drawing', drawing.id, drawing_number)
                else:
                    self.change_tracker.add_unchanged('drawing')
            else:
                self.change_tracker.add_unchanged('drawing')
        else:
            # Create new drawing
            drawing_name = str(row.get('DRAWING NAME', '')) if row.get('DRAWING NAME') else ''
            revision = str(row.get('REVISION', '')) if row.get('REVISION') else ''
            equipment_name = str(row.get('EQUIPMENT NAME', '')) if row.get('EQUIPMENT NAME') else ''
            spare_part_number = str(row.get('SPARE PART NUMBER', '')) if row.get('SPARE PART NUMBER') else ''

            info_id(f"Creating new Drawing: {drawing_number} - {drawing_name} (Rev: {revision})")
            drawing = Drawing(
                drw_number=drawing_number,
                drw_name=drawing_name,
                drw_revision=revision,
                drw_equipment_name=equipment_name,
                drw_spare_part_number=spare_part_number
            )
            self.session.add(drawing)
            self.session.flush()
            self.change_tracker.add_created('drawing', drawing.id, drawing_number)

        # Add to cache
        self.drawing_cache[drawing_number] = drawing
        return drawing

    def _process_spare_part(self, row, skip_updates=False) -> Part:
        """Process spare part data from row"""
        spare_part_number = str(row.get('SPARE PART NUMBER', '')).strip() if row.get('SPARE PART NUMBER') else ''

        # Check cache first
        if spare_part_number in self.part_cache:
            part = self.part_cache[spare_part_number]
            self.change_tracker.add_unchanged('part')
            return part

        # Check database
        part = self.session.query(Part).filter(Part.part_number == spare_part_number).first()

        if not part:
            # Create new part
            info_id(f"Creating new Part: {spare_part_number}")
            part = Part(
                part_number=spare_part_number,
                name=f"Part {spare_part_number}"
            )
            self.session.add(part)
            self.session.flush()
            self.change_tracker.add_created('part', part.id, spare_part_number)
        else:
            self.change_tracker.add_unchanged('part')

        # Add to cache
        self.part_cache[spare_part_number] = part
        return part

    def _associate_drawing_position(self, drawing, position, skip_updates=False):
        """Associate drawing with position"""
        assoc_key = f"dp:{drawing.id}:{position.id}"

        # Check cache first
        if assoc_key in self.assoc_cache:
            self.change_tracker.add_unchanged('associations')
            return self.assoc_cache[assoc_key]

        # Check if association already exists
        existing_assoc = (
            self.session.query(DrawingPositionAssociation)
            .filter(
                DrawingPositionAssociation.drawing_id == drawing.id,
                DrawingPositionAssociation.position_id == position.id
            )
            .first()
        )

        if not existing_assoc:
            # Create association
            try:
                drawing_position = DrawingPositionAssociation(
                    drawing_id=drawing.id,
                    position_id=position.id
                )
                self.session.add(drawing_position)
                self.session.flush()
                info_id(f"Associated drawing {drawing.id} with position {position.id}")
                self.change_tracker.add_created('drawing_position_assoc', drawing_position.id)
                self.assoc_cache[assoc_key] = drawing_position
                return drawing_position
            except Exception as e:
                error_id(f"Error associating drawing with position: {e}")
                raise
        else:
            self.change_tracker.add_unchanged('associations')
            self.assoc_cache[assoc_key] = existing_assoc
            return existing_assoc

    def _associate_drawing_part(self, drawing, part, skip_updates=False):
        """Associate drawing with part"""
        assoc_key = f"dp:{drawing.id}:{part.id}"

        # Check cache first
        if assoc_key in self.assoc_cache:
            self.change_tracker.add_unchanged('associations')
            return self.assoc_cache[assoc_key]

        # Check if association already exists
        existing_assoc = (
            self.session.query(DrawingPartAssociation)
            .filter(
                DrawingPartAssociation.drawing_id == drawing.id,
                DrawingPartAssociation.part_id == part.id
            )
            .first()
        )

        if not existing_assoc:
            # Create association
            try:
                drawing_part_assoc = DrawingPartAssociation(
                    drawing_id=drawing.id,
                    part_id=part.id
                )
                self.session.add(drawing_part_assoc)
                self.session.flush()
                info_id(f"Associated drawing {drawing.id} with part {part.id}")
                self.change_tracker.add_created('drawing_part_assoc', drawing_part_assoc.id)
                self.assoc_cache[assoc_key] = drawing_part_assoc
                return drawing_part_assoc
            except Exception as e:
                error_id(f"Error associating drawing with part: {e}")
                raise
        else:
            self.change_tracker.add_unchanged('associations')
            self.assoc_cache[assoc_key] = existing_assoc
            return existing_assoc

    def _associate_part_position(self, part, position, skip_updates=False):
        """Associate part with position"""
        assoc_key = f"pp:{part.id}:{position.id}"

        # Check cache first
        if assoc_key in self.assoc_cache:
            self.change_tracker.add_unchanged('associations')
            return self.assoc_cache[assoc_key]

        # Check if association already exists
        existing_assoc = (
            self.session.query(PartsPositionImageAssociation)
            .filter(
                PartsPositionImageAssociation.part_id == part.id,
                PartsPositionImageAssociation.position_id == position.id
            )
            .first()
        )

        if not existing_assoc:
            # Create association
            try:
                part_pos_assoc = PartsPositionImageAssociation(
                    part_id=part.id,
                    position_id=position.id,
                    image_id=None  # Can be updated later when an image is available
                )
                self.session.add(part_pos_assoc)
                self.session.flush()
                info_id(f"Associated part {part.id} with position {position.id}")
                self.change_tracker.add_created('part_position_assoc', part_pos_assoc.id)
                self.assoc_cache[assoc_key] = part_pos_assoc
                return part_pos_assoc
            except Exception as e:
                error_id(f"Error associating part with position: {e}")
                raise
        else:
            self.change_tracker.add_unchanged('associations')
            self.assoc_cache[assoc_key] = existing_assoc
            return existing_assoc

    def automate_part_position_associations(self):
        """
        Automatically create part-position associations based on drawing relationships.
        This fixes the issue where part_position_image entries are missing.

        When a drawing is associated with both a part and a position, there should be
        a part-position association as well.
        """
        info_id("Creating missing part-position associations based on drawing relationships")

        # Find all missing part-position associations using SQL for efficiency
        query = """
        SELECT DISTINCT dp.part_id, dpos.position_id
        FROM drawing_part dp
        JOIN drawing_position dpos ON dp.drawing_id = dpos.drawing_id
        LEFT JOIN part_position_image ppi ON dp.part_id = ppi.part_id AND dpos.position_id = ppi.position_id
        WHERE ppi.id IS NULL
        """

        try:
            result = self.session.execute(text(query))
            missing_associations = [(row[0], row[1]) for row in result]

            info_id(f"Found {len(missing_associations)} missing part-position associations")

            # Save the list of missing associations for reference
            if missing_associations:
                timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
                missing_file = os.path.join(DRAWING_IMPORT_DATA_DIR, f"missing_associations_{timestamp}.csv")
                pd.DataFrame(missing_associations, columns=['part_id', 'position_id']).to_csv(missing_file, index=False)
                info_id(f"Saved list of missing associations to {missing_file}")

            # Create the missing associations in batches
            batch_size = 100
            created_count = 0

            for i in range(0, len(missing_associations), batch_size):
                batch = missing_associations[i:i + batch_size]
                for part_id, position_id in batch:
                    try:
                        # Double check it's still missing (in case it was created in another session)
                        existing = (
                            self.session.query(PartsPositionImageAssociation)
                            .filter(
                                PartsPositionImageAssociation.part_id == part_id,
                                PartsPositionImageAssociation.position_id == position_id
                            )
                            .first()
                        )

                        if not existing:
                            new_assoc = PartsPositionImageAssociation(
                                part_id=part_id,
                                position_id=position_id,
                                image_id=None
                            )
                            self.session.add(new_assoc)
                            created_count += 1

                            # Add to change tracker
                            self.change_tracker.add_created('part_position_assoc', part_id)
                    except Exception as e:
                        error_id(f"Error creating part-position association ({part_id}, {position_id}): {e}")

                # Commit each batch
                self.session.flush()
                info_id(f"Created {created_count} part-position associations so far")

            if created_count > 0:
                info_id(f"Successfully created {created_count} missing part-position associations")
                self.session.commit()
            else:
                info_id("No missing part-position associations found")

            return created_count

        except Exception as e:
            error_id(f"Error finding or creating missing part-position associations: {e}", exc_info=True)
            self.session.rollback()
            return 0

    @with_request_id
    def process_data(self):
        """
        Process Excel data in batches with transaction management.
        Tracks changes for reporting.

        Returns:
            bool: True if successful, False otherwise
            str: Report of changes
        """
        if self.excel_data is None or len(self.excel_data) == 0:
            error_id("No Excel data loaded or data is empty")
            return False, "No data to process"

        try:
            info_id(f"Starting to process {len(self.excel_data)} rows of data in batches of {self.batch_size}")

            # Split data into batches
            total_rows = len(self.excel_data)
            num_batches = (total_rows + self.batch_size - 1) // self.batch_size
            current_hashes = {}

            for batch_num in range(num_batches):
                start_idx = batch_num * self.batch_size
                end_idx = min((batch_num + 1) * self.batch_size, total_rows)

                # Get current batch
                batch_df = self.excel_data.iloc[start_idx:end_idx]

                # Process batch
                batch_hashes = self._process_batch(batch_df, batch_num + 1)
                current_hashes.update(batch_hashes)

            # Create missing part-position associations
            created_associations = self.automate_part_position_associations()
            info_id(f"Created {created_associations} missing part-position associations")

            # Save current hashes for future comparison
            self._save_current_hashes(current_hashes)

            # Generate report
            report = self.change_tracker.generate_report()
            info_id("Completed processing all data")

            return True, report

        except Exception as e:
            error_id(f"Unexpected error while processing data: {e}", exc_info=True)
            self.session.rollback()
            return False, f"Error: {str(e)}"

    @with_request_id
    def close(self):
        """Close database session."""
        debug_id("Closing database session")
        if hasattr(self, 'session') and self.session:
            self.session.close()


def validate_specific_entries(session):
    """
    Validate and optionally repair specific database entries.
    Checks for the database entries shown in the screenshots.
    """
    info_id("Running validation on specific database entries")

    # Create validator
    validator = DataValidator(session)

    # Validate part_position_image table entries
    info_id("Validating part_position_image table...")
    expected_entries = [
        (1, 1), (2, 1), (3, 1), (4, 7), (5, 9), (3975, 9), (2299, 9)
    ]
    validator.validate_ids('part_position', expected_entries)

    # Validate drawing_part table entries
    info_id("Validating drawing_part table...")
    drawing_part_entries = [
        (6327, 5484), (6328, 5485), (6329, 5487), (17964, 2553),
        (25967, 18851), (25968, 18849), (64, 10317), (66, 3283),
        (67, 3284), (119, 3456), (120, 2071), (121, 2085),
        (121, 2086), (121, 2087), (122, 3484), (123, 3454),
        (152, 6520), (155, 4085), (161, 2229), (256, 11200)
    ]
    validator.validate_ids('drawing_part', drawing_part_entries)

    # Validate drawing_position table entries
    info_id("Validating drawing_position table...")
    drawing_position_entries = [
        (3707, 1), (3708, 1), (3709, 1), (3377, 9),
        (504, 9), (1, 9), (2, 9), (408, 18), (415, 18),
        (2244, 19)
    ]
    validator.validate_ids('drawing_position', drawing_position_entries)

    # Check for drawing-part-position relationships that should exist but don't have part_position entries
    info_id("Checking for missing part-position relationships...")
    missing_part_position_entries = find_missing_part_position_entries(session)

    if missing_part_position_entries:
        info_id(f"Found {len(missing_part_position_entries)} missing part-position relationships")

        # Ask if user wants to create these
        answer = input("\nDo you want to create these missing part-position relationships? (y/n): ")
        if answer.lower() in ('y', 'yes'):
            create_missing_part_position_entries(session, missing_part_position_entries)
            info_id("Created missing part-position relationships")
        else:
            info_id("Skipped creating missing relationships")
    else:
        info_id("No missing part-position relationships found")

    # Generate validation report
    report = validator.generate_validation_report()

    print("\n" + "=" * 80)
    print(report)
    print("=" * 80 + "\n")

    # Save report to file
    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    report_file = os.path.join(DRAWING_IMPORT_DATA_DIR, f"validation_report_{timestamp}.txt")
    with open(report_file, 'w') as f:
        f.write(report)
    info_id(f"Validation report saved to {report_file}")

    return True


def find_missing_part_position_entries(session):
    """
    Find drawing-part-position relationships that should have part_position entries but don't.
    Returns a list of (part_id, position_id) tuples that are missing.
    """
    info_id("Scanning for missing part-position relationships...")

    # This SQL query finds parts and positions that are linked through drawings
    # but don't have entries in the part_position_image table
    query = """
    SELECT DISTINCT dp.part_id, dpos.position_id
    FROM drawing_part dp
    JOIN drawing_position dpos ON dp.drawing_id = dpos.drawing_id
    LEFT JOIN part_position_image ppi ON dp.part_id = ppi.part_id AND dpos.position_id = ppi.position_id
    WHERE ppi.id IS NULL
    """

    try:
        # Execute the query
        result = session.execute(text(query))
        missing_entries = [(row[0], row[1]) for row in result]
        info_id(f"Found {len(missing_entries)} missing part-position relationships")

        # Get details about the missing entries
        if missing_entries:
            debug_id("Missing part-position relationships details:")
            for part_id, position_id in missing_entries[:20]:  # Limit to 20 for log readability
                debug_id(f"  Part ID: {part_id}, Position ID: {position_id}")

            if len(missing_entries) > 20:
                debug_id(f"  ... and {len(missing_entries) - 20} more")

        return missing_entries

    except Exception as e:
        error_id(f"Error finding missing part-position relationships: {e}", exc_info=True)
        return []


def create_missing_part_position_entries(session, missing_entries):
    """
    Create missing part_position_image entries based on the list of missing (part_id, position_id) pairs.

    Args:
        session: SQLAlchemy session
        missing_entries: List of (part_id, position_id) tuples
    """
    info_id(f"Creating {len(missing_entries)} missing part-position relationships...")

    batch_size = 100
    created_count = 0
    error_count = 0

    # Process in batches to avoid overloading the database
    for i in range(0, len(missing_entries), batch_size):
        batch = missing_entries[i:i + batch_size]
        try:
            for part_id, position_id in batch:
                try:
                    # Check if it already exists (might have been created in a previous batch)
                    existing = (
                        session.query(PartsPositionImageAssociation)
                        .filter(
                            PartsPositionImageAssociation.part_id == part_id,
                            PartsPositionImageAssociation.position_id == position_id
                        )
                        .first()
                    )

                    if not existing:
                        # Create the association
                        assoc = PartsPositionImageAssociation(
                            part_id=part_id,
                            position_id=position_id,
                            image_id=None  # No image initially
                        )
                        session.add(assoc)
                        created_count += 1
                except Exception as e:
                    error_id(f"Error creating part-position relationship ({part_id}, {position_id}): {e}")
                    error_count += 1

            # Commit this batch
            session.commit()
            info_id(
                f"Committed batch {i // batch_size + 1}/{(len(missing_entries) - 1) // batch_size + 1}, created {created_count} entries so far")

        except Exception as e:
            session.rollback()
            error_id(f"Error processing batch {i // batch_size + 1}: {e}", exc_info=True)
            error_count += len(batch)

    info_id(f"Created {created_count} part-position relationships, encountered {error_count} errors")


def repair_database_relationships(session):
    """
    Repair database relationships by creating missing part-position entries.
    This is a more comprehensive fix that scans the entire database.
    """
    info_id("Starting database relationship repair...")

    # Find missing part-position entries
    missing_entries = find_missing_part_position_entries(session)

    if missing_entries:
        info_id(f"Found {len(missing_entries)} missing part-position relationships")
        create_missing_part_position_entries(session, missing_entries)
    else:
        info_id("No missing relationships found, database is consistent")

    return True


@with_request_id
def main():
    """Main entry point for the script."""
    # Ensure all required directories exist
    ensure_directories_exist()

    # Use the database URL from the config module
    db_url = DATABASE_URL

    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Import drawing data from Excel to database')
    parser.add_argument('--file', '-f', help='Path to Excel file to import',
                        default=os.path.join(DB_LOADSHEET, "Active Drawing List breakdown.xlsx"))
    parser.add_argument('--batch-size', '-b', type=int, default=50, help='Batch size for processing')
    parser.add_argument('--validate', '-v', action='store_true', help='Run in validation mode to check entries')
    parser.add_argument('--repair', '-r', action='store_true', help='Repair missing relationships')
    parser.add_argument('--force-create', '-c', action='store_true',
                        help='Force create missing part-position entries')

    args = parser.parse_args()
    excel_file = args.file
    batch_size = args.batch_size
    validation_mode = args.validate
    repair_mode = args.repair
    force_create = args.force_create

    info_id(f"Excel file path: {excel_file}")
    info_id(f"Using batch size: {batch_size}")

    # Log information about execution environment
    info_id(f"Running from directory: {os.getcwd()}")
    info_id(f"Base directory: {BASE_DIR}")
    info_id(f"Database directory: {DATABASE_DIR}")
    info_id(f"DB_LOADSHEET directory: {DB_LOADSHEET}")
    info_id(f"Drawing import data directory: {DRAWING_IMPORT_DATA_DIR}")

    # Create mapper and process data
    mapper = ExcelToDbMapper(db_url, batch_size=batch_size)
    try:
        # If in validation mode, perform validations on specific IDs
        if validation_mode:
            info_id("Running in validation mode")
            validate_specific_entries(mapper.session)
            return

        # If in repair mode, fix missing relationships
        if repair_mode:
            info_id("Running in repair mode")
            repair_database_relationships(mapper.session)
            return

        # If force create mode, create part-position entries
        if force_create:
            info_id("Running in force create mode")
            missing_entries = find_missing_part_position_entries(mapper.session)
            if missing_entries:
                create_missing_part_position_entries(mapper.session, missing_entries)
            return

        # Normal import mode
        with log_timed_operation("Excel Import Process"):
            info_id(f"Starting Excel import process for file: {excel_file}")
            if mapper.load_excel(excel_file):
                success, report = mapper.process_data()
                if success:
                    info_id("Data mapping completed successfully")

                    # Print and save the report
                    print("\n" + "=" * 80)
                    print(report)
                    print("=" * 80 + "\n")

                    # Save report to file
                    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
                    report_file = os.path.join(DRAWING_IMPORT_DATA_DIR, f"import_report_{timestamp}.txt")
                    with open(report_file, 'w') as f:
                        f.write(report)
                    info_id(f"Report saved to {report_file}")
                else:
                    error_id("Data mapping failed")
                    print(f"Error: {report}")
            else:
                error_id("Failed to load Excel file")
    except Exception as e:
        error_id(f"Unhandled exception in main: {e}", exc_info=True)
    finally:
        mapper.close()


if __name__ == "__main__":
    main()