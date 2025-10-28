import os
import sys
import pandas as pd
import numpy as np
from sqlalchemy import func, inspect, text
from datetime import datetime

# Import the new PostgreSQL framework components
from modules.configuration.config import (
    BASE_DIR,
    DB_LOADSHEET,
    REVISION_CONTROL_DB_PATH
)
from modules.configuration.config_env import DatabaseConfig
from modules.configuration.log_config import (
    debug_id, info_id, warning_id, error_id,
    set_request_id, get_request_id, log_timed_operation
)
from modules.emtacdb.emtacdb_fts import (
    Area,
    EquipmentGroup,
    Model,
    AssetNumber,
    Location,
    SiteLocation,
    Base
)
from modules.emtacdb.emtac_revision_control_db import (
    VersionInfo,
    revision_control_engine,
    RevisionControlSession,
    AreaSnapshot,
    EquipmentGroupSnapshot,
    ModelSnapshot,
    AssetNumberSnapshot,
    LocationSnapshot
)
from modules.emtacdb.utlity.revision_database.snapshot_utils import (
    create_snapshot
)
from modules.initial_setup.initializer_logger import (
    initializer_logger,
    close_initializer_logger
)


class PostgreSQLEquipmentRelationshipsLoader:
    """PostgreSQL-enhanced equipment relationships loader with comprehensive data management."""

    def __init__(self):
        self.request_id = set_request_id()
        self.db_config = DatabaseConfig()
        info_id("Initialized PostgreSQL Equipment Relationships Loader", self.request_id)

        # Statistics tracking
        self.stats = {
            'areas_processed': 0,
            'equipment_groups_processed': 0,
            'models_processed': 0,
            'asset_numbers_processed': 0,
            'locations_processed': 0,
            'site_locations_processed': 0,
            'duplicates_removed': 0,
            'snapshots_created': 0,
            'errors_encountered': 0
        }

        # Table processing order (important for foreign key relationships)
        self.table_order = [
            ('Area', Area, ['name', 'description']),
            ('EquipmentGroup', EquipmentGroup, ['name', 'area_id']),
            ('Model', Model, ['name', 'description', 'equipment_group_id']),
            ('AssetNumber', AssetNumber, ['number', 'description', 'model_id']),
            ('Location', Location, ['name', 'model_id']),
            ('SiteLocation', SiteLocation, ['id', 'title', 'room_number', 'site_area'])
        ]

    def create_database_tables(self, session):
        """Create database tables if they don't exist - THIS IS THE KEY FIX!"""
        try:
            info_id("Creating database tables if they don't exist", self.request_id)
            info_id("Creating database tables...")

            # Get the engine from the session
            engine = session.bind

            # Create all tables defined in the Base metadata
            Base.metadata.create_all(engine)

            info_id("Database tables created/verified")
            info_id("Database tables created successfully", self.request_id)

        except Exception as e:
            error_id(f"Error creating database tables: {str(e)}", self.request_id)
            raise

    def validate_excel_file(self, file_path):
        """Validate the Excel file structure and required sheets."""
        info_id(f"Validating Excel file: {file_path}", self.request_id)

        if not os.path.exists(file_path):
            error_id(f"Excel file not found: {file_path}", self.request_id)
            return False, "File does not exist"

        try:
            # Check if all required sheets exist
            required_sheets = ['Area', 'EquipmentGroup', 'Model', 'AssetNumber', 'Location', 'SiteLocation']
            excel_file = pd.ExcelFile(file_path)

            missing_sheets = [sheet for sheet in required_sheets if sheet not in excel_file.sheet_names]
            if missing_sheets:
                error_id(f"Missing required sheets: {missing_sheets}", self.request_id)
                return False, f"Missing required sheets: {missing_sheets}"

            # Validate each sheet has data
            for sheet_name in required_sheets:
                df = pd.read_excel(file_path, sheet_name=sheet_name)
                if df.empty:
                    warning_id(f"Sheet '{sheet_name}' is empty", self.request_id)

            info_id("Excel file validation successful", self.request_id)
            return True, "Valid"

        except Exception as e:
            error_id(f"Error validating Excel file: {str(e)}", self.request_id)
            return False, f"Error reading file: {str(e)}"

    def clean_dataframe(self, df, required_columns, sheet_name):
        """Clean and validate DataFrame with enhanced error handling."""
        info_id(f"Cleaning DataFrame for sheet: {sheet_name}", self.request_id)

        try:
            original_rows = len(df)

            # Remove completely empty columns
            df = df.dropna(axis=1, how='all')

            # Remove columns that are just empty strings
            for col in df.columns:
                if df[col].astype(str).str.strip().eq('').all():
                    df = df.drop(columns=[col])

            # Check if all required columns exist
            missing_columns = [col for col in required_columns if col not in df.columns]
            if missing_columns:
                error_id(f"Missing required columns in {sheet_name}: {missing_columns}", self.request_id)
                raise ValueError(f"Missing required columns in {sheet_name}: {missing_columns}")

            # Replace NaN values with None for database compatibility
            df = df.replace({np.nan: None})

            # Return only required columns in the correct order
            cleaned_df = df[required_columns].copy()

            # Remove rows where critical fields are empty
            critical_fields = ['name'] if 'name' in required_columns else [
                'number'] if 'number' in required_columns else ['title'] if 'title' in required_columns else []

            for field in critical_fields:
                initial_count = len(cleaned_df)
                cleaned_df = cleaned_df[cleaned_df[field].notna() & (cleaned_df[field].astype(str).str.strip() != '')]
                removed_count = initial_count - len(cleaned_df)
                if removed_count > 0:
                    warning_id(f"Removed {removed_count} rows with empty {field} in {sheet_name}", self.request_id)

            final_rows = len(cleaned_df)
            info_id(f"Cleaned {sheet_name}: {original_rows} -> {final_rows} rows", self.request_id)

            return cleaned_df

        except Exception as e:
            error_id(f"Error cleaning DataFrame for {sheet_name}: {str(e)}", self.request_id)
            raise

    def check_existing_data(self, session):
        """Check for existing data in tables to prevent duplicates."""
        try:
            info_id("Checking existing data in database", self.request_id)

            existing_data = {}
            total_records = 0

            for sheet_name, model_class, _ in self.table_order:
                try:
                    count = session.query(model_class).count()
                    existing_data[sheet_name] = count
                    total_records += count
                except Exception as e:
                    # If table doesn't exist, count is 0
                    warning_id(f"Table {sheet_name} might not exist: {str(e)}", self.request_id)
                    existing_data[sheet_name] = 0

            if total_records > 0:
                warning_id("EXISTING EQUIPMENT DATA DETECTED")
                warning_id("=" * 45)
                for sheet_name, count in existing_data.items():
                    if count > 0:
                        info_id(f"{sheet_name}: {count} records")
                info_id(f"Total Records: {total_records}")
                warning_id("Loading new data may create duplicates!")
                info_id("The system will handle duplicates automatically")

                proceed = input("Continue with equipment relationships import? (y/n): ").strip().lower()
                if proceed not in ['y', 'yes']:
                    info_id("User chose to skip import due to existing data", self.request_id)
                    return False

            return True

        except Exception as e:
            error_id(f"Error checking existing data: {str(e)}", self.request_id)
            raise

    def create_database_backup(self, session):
        """Create a comprehensive database backup before making changes."""
        try:
            info_id("Creating database backup", self.request_id)
            info_id("Creating database backup...")

            with log_timed_operation("create_database_backup", self.request_id):
                # Define backup directory
                backup_directory = os.path.join(BASE_DIR, "Database", "DB_LOADSHEETS_BACKUP")
                os.makedirs(backup_directory, exist_ok=True)

                # Create timestamped backup file
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                excel_file_name = f"equipment_relationships_backup_{timestamp}.xlsx"
                excel_file_path = os.path.join(backup_directory, excel_file_name)

                # Extract data from each table
                backup_data = {}

                for sheet_name, model_class, columns in self.table_order:
                    try:
                        if sheet_name == 'Area':
                            data = [(area.name, area.description) for area in session.query(Area).all()]
                        elif sheet_name == 'EquipmentGroup':
                            data = [(group.name, group.area_id) for group in session.query(EquipmentGroup).all()]
                        elif sheet_name == 'Model':
                            data = [(model.name, model.description, model.equipment_group_id) for model in
                                    session.query(Model).all()]
                        elif sheet_name == 'AssetNumber':
                            data = [(asset.number, asset.model_id, asset.description) for asset in
                                    session.query(AssetNumber).all()]
                        elif sheet_name == 'Location':
                            data = [(location.name, location.model_id) for location in session.query(Location).all()]
                        elif sheet_name == 'SiteLocation':
                            data = [(site.id, site.title, site.room_number, site.site_area) for site in
                                    session.query(SiteLocation).all()]

                        backup_data[sheet_name] = pd.DataFrame(data, columns=columns)

                    except Exception as e:
                        warning_id(f"Error backing up {sheet_name}: {str(e)}", self.request_id)
                        backup_data[sheet_name] = pd.DataFrame()

                # Write to Excel file
                with pd.ExcelWriter(excel_file_path, engine='openpyxl') as writer:
                    for sheet_name, df in backup_data.items():
                        df.to_excel(writer, sheet_name=sheet_name, index=False)

                info_id(f"Database backup created: {excel_file_name}", self.request_id)
                info_id(f"Backup saved: {excel_file_name}")

        except Exception as e:
            error_id(f"Error creating database backup: {str(e)}", self.request_id)
            # Don't raise - backup failure shouldn't stop the import
            warning_id(f"Backup failed: {str(e)}")

    def delete_duplicates_enhanced(self, session, model, attribute, sheet_name):
        """Enhanced duplicate removal with detailed logging."""
        try:
            info_id(f"Removing duplicates from {sheet_name} based on '{attribute}'", self.request_id)

            # Find duplicates using PostgreSQL-optimized query
            if self.db_config.is_postgresql:
                # Use PostgreSQL-specific features for better performance
                duplicates = session.query(
                    getattr(model, attribute),
                    func.count().label('count')
                ).group_by(
                    getattr(model, attribute)
                ).having(
                    func.count() > 1
                ).all()
            else:
                # Fallback for SQLite
                duplicates = session.query(
                    getattr(model, attribute),
                    func.count()
                ).group_by(
                    getattr(model, attribute)
                ).having(
                    func.count() > 1
                ).all()

            duplicates_removed = 0

            for duplicate_data in duplicates:
                attr_value = duplicate_data[0]
                count = duplicate_data[1] if len(duplicate_data) > 1 else duplicate_data.count

                # Get all records with this attribute value
                records = session.query(model).filter(
                    getattr(model, attribute) == attr_value
                ).all()

                # Keep the first record, delete the rest
                for record in records[1:]:
                    session.delete(record)
                    duplicates_removed += 1

            if duplicates_removed > 0:
                info_id(f"Removed {duplicates_removed} duplicates from {sheet_name}", self.request_id)
                self.stats['duplicates_removed'] += duplicates_removed

        except Exception as e:
            error_id(f"Error removing duplicates from {sheet_name}: {str(e)}", self.request_id)
            raise

    def process_single_table(self, session, df, model_class, sheet_name):
        """Process a single table with enhanced error handling and progress tracking."""
        info_id(f"Processing {sheet_name} table", self.request_id)
        info_id(f"Processing {sheet_name}...")

        try:
            processed_count = 0

            for index, row in df.iterrows():
                try:
                    if sheet_name == 'Area':
                        area_name = str(row['name']).strip() if pd.notna(row['name']) else ''
                        if not area_name:
                            continue

                        area = session.query(Area).filter_by(name=area_name).first()
                        if area:
                            area.description = row['description'] if pd.notna(row['description']) else None
                        else:
                            area = Area(
                                name=area_name,
                                description=row['description'] if pd.notna(row['description']) else None
                            )
                            session.add(area)
                        processed_count += 1

                    elif sheet_name == 'EquipmentGroup':
                        equipment_group_name = str(row['name']).strip() if pd.notna(row['name']) else ''
                        if not equipment_group_name:
                            continue

                        area_id = int(row['area_id']) if pd.notna(row['area_id']) else None

                        equipment_group = session.query(EquipmentGroup).filter_by(name=equipment_group_name).first()
                        if equipment_group:
                            equipment_group.area_id = area_id
                        else:
                            equipment_group = EquipmentGroup(name=equipment_group_name, area_id=area_id)
                            session.add(equipment_group)
                        processed_count += 1

                    elif sheet_name == 'Model':
                        model_name = str(row['name']).strip() if pd.notna(row['name']) else ''
                        if not model_name:
                            continue

                        equipment_group_id = int(row['equipment_group_id']) if pd.notna(
                            row['equipment_group_id']) else None

                        model = session.query(Model).filter_by(name=model_name).first()
                        if model:
                            model.description = row['description'] if pd.notna(row['description']) else None
                            model.equipment_group_id = equipment_group_id
                        else:
                            equipment_group = None
                            if equipment_group_id:
                                equipment_group = session.query(EquipmentGroup).filter_by(id=equipment_group_id).first()

                            model = Model(
                                name=model_name,
                                description=row['description'] if pd.notna(row['description']) else None,
                                equipment_group=equipment_group
                            )
                            session.add(model)
                        processed_count += 1

                    elif sheet_name == 'AssetNumber':
                        asset_number_name = str(row['number']).strip() if pd.notna(row['number']) else ''
                        if not asset_number_name:
                            continue

                        model_id = int(row['model_id']) if pd.notna(row['model_id']) else None

                        asset_number = session.query(AssetNumber).filter_by(number=asset_number_name).first()
                        if asset_number:
                            asset_number.model_id = model_id
                            asset_number.description = row['description'] if pd.notna(row['description']) else None
                        else:
                            asset_number = AssetNumber(
                                number=asset_number_name,
                                model_id=model_id,
                                description=row['description'] if pd.notna(row['description']) else None
                            )
                            session.add(asset_number)
                        processed_count += 1

                    elif sheet_name == 'Location':
                        location_name = str(row['name']).strip() if pd.notna(row['name']) else ''
                        if not location_name:
                            continue

                        model_id = int(row['model_id']) if pd.notna(row['model_id']) else None

                        location = session.query(Location).filter_by(name=location_name).first()
                        if location:
                            location.model_id = model_id
                        else:
                            location = Location(name=location_name, model_id=model_id)
                            session.add(location)
                        processed_count += 1

                    elif sheet_name == 'SiteLocation':
                        title = str(row['title']).strip() if pd.notna(row['title']) else ''
                        if not title:
                            continue

                        site_location_id = int(row['id']) if pd.notna(row['id']) else None
                        room_number = str(row['room_number']).strip() if pd.notna(row['room_number']) else ''
                        site_area = str(row['site_area']).strip() if pd.notna(row['site_area']) else ''

                        # Check by ID first, then by title
                        site_location = None
                        if site_location_id:
                            site_location = session.query(SiteLocation).filter_by(id=site_location_id).first()

                        if not site_location:
                            site_location = session.query(SiteLocation).filter_by(title=title).first()

                        if site_location:
                            site_location.title = title
                            site_location.room_number = room_number
                            site_location.site_area = site_area
                        else:
                            site_location = SiteLocation(
                                id=site_location_id,
                                title=title,
                                room_number=room_number,
                                site_area=site_area
                            )
                            session.add(site_location)
                        processed_count += 1

                    # Progress reporting
                    if processed_count % 100 == 0:
                        debug_id(f"Processed {processed_count} {sheet_name} records", self.request_id)

                except Exception as e:
                    error_id(f"Error processing {sheet_name} row {index}: {str(e)}", self.request_id)
                    self.stats['errors_encountered'] += 1
                    continue

            # Update statistics
            stat_key = f"{sheet_name.lower()}s_processed"
            if stat_key.replace('s_processed', '_processed') in self.stats:
                stat_key = stat_key.replace('s_processed', '_processed')
            if stat_key in self.stats:
                self.stats[stat_key] = processed_count

            info_id(f"Processed {processed_count} {sheet_name} records", self.request_id)
            info_id(f"Processed {processed_count} records")

        except Exception as e:
            error_id(f"Error processing {sheet_name} table: {str(e)}", self.request_id)
            raise

    def create_revision_snapshots(self, main_session):
        """Create revision control snapshots with enhanced error handling."""
        try:
            info_id("Creating revision control snapshots", self.request_id)
            info_id("Creating revision snapshots...")

            with log_timed_operation("create_revision_snapshots", self.request_id):
                rev_session = RevisionControlSession()

                try:
                    # Create version info
                    new_version = VersionInfo(
                        version_number=1,
                        description="Equipment relationships data import with PostgreSQL enhancements"
                    )
                    rev_session.add(new_version)
                    rev_session.commit()

                    snapshots_created = 0

                    # Create snapshots for each entity type
                    snapshot_mapping = [
                        (Area, AreaSnapshot, "areas"),
                        (EquipmentGroup, EquipmentGroupSnapshot, "equipment groups"),
                        (Model, ModelSnapshot, "models"),
                        (AssetNumber, AssetNumberSnapshot, "asset numbers"),
                        (Location, LocationSnapshot, "locations")
                    ]

                    for entity_class, snapshot_class, entity_name in snapshot_mapping:
                        try:
                            entities = main_session.query(entity_class).all()
                            for entity in entities:
                                create_snapshot(entity, rev_session, snapshot_class)
                                snapshots_created += 1

                            info_id(f"Created {len(entities)} {entity_name} snapshots", self.request_id)

                        except Exception as e:
                            warning_id(f"Error creating {entity_name} snapshots: {str(e)}", self.request_id)

                    # Handle SiteLocation snapshots separately (may not exist)
                    try:
                        from modules.emtacdb.emtac_revision_control_db import SiteLocationSnapshot
                        site_locations = main_session.query(SiteLocation).all()
                        for site_location in site_locations:
                            create_snapshot(site_location, rev_session, SiteLocationSnapshot)
                            snapshots_created += 1

                        info_id(f"Created {len(site_locations)} site location snapshots", self.request_id)

                    except ImportError:
                        warning_id("SiteLocationSnapshot class not found - skipping site location snapshots",
                                   self.request_id)
                    except Exception as e:
                        warning_id(f"Error creating site location snapshots: {str(e)}", self.request_id)

                    rev_session.commit()
                    self.stats['snapshots_created'] = snapshots_created

                    info_id(f"Created {snapshots_created} total snapshots", self.request_id)
                    info_id(f"Created {snapshots_created} snapshots")

                except Exception as e:
                    rev_session.rollback()
                    error_id(f"Error in revision snapshot creation: {str(e)}", self.request_id)
                    raise
                finally:
                    rev_session.close()

        except Exception as e:
            error_id(f"Error creating revision snapshots: {str(e)}", self.request_id)
            # Don't raise - snapshot failure shouldn't stop the import
            warning_id(f"Snapshot creation failed: {str(e)}")

    def display_processing_summary(self):
        """Display comprehensive processing summary."""
        info_id("Processing Summary")
        info_id("=" * 40)

        for key, value in self.stats.items():
            if value > 0:
                formatted_key = key.replace('_', ' ').title()
                info_id(f"{formatted_key}: {value}")

        if self.stats['errors_encountered'] > 0:
            warning_id(f"Errors Encountered: {self.stats['errors_encountered']}")

        info_id(f"Processing summary: {self.stats}", self.request_id)

    def load_equipment_relationships(self, file_path=None):
        """Main method to load equipment relationships data."""
        try:
            info_id("Equipment Relationships Data Import")
            info_id("=" * 45)

            # Determine file path
            if not file_path:
                file_path = os.path.join(DB_LOADSHEET, "load_equipment_relationships_table_data.xlsx")

            # Validate Excel file
            is_valid, message = self.validate_excel_file(file_path)
            if not is_valid:
                raise ValueError(f"Invalid Excel file: {message}")

            info_id(f"Loading from: {os.path.basename(file_path)}")

            # Get database session
            with self.db_config.main_session() as session:
                # CREATE TABLES FIRST - This is the key fix for your original error!
                self.create_database_tables(session)

                # Check existing data
                if not self.check_existing_data(session):
                    return False

                # Create backup
                self.create_database_backup(session)

                # Process each table in correct order
                info_id("Processing tables in dependency order...")

                for sheet_name, model_class, required_columns in self.table_order:
                    try:
                        # Load and clean data
                        with log_timed_operation(f"load_{sheet_name}_sheet", self.request_id):
                            df_raw = pd.read_excel(file_path, sheet_name=sheet_name)
                            df_cleaned = self.clean_dataframe(df_raw, required_columns, sheet_name)

                        # Process table data
                        self.process_single_table(session, df_cleaned, model_class, sheet_name)

                    except Exception as e:
                        error_id(f"Error processing {sheet_name}: {str(e)}", self.request_id)
                        self.stats['errors_encountered'] += 1
                        error_id(f"Error processing {sheet_name}: {str(e)}")
                        continue

                # Remove duplicates
                info_id("Removing duplicates...")
                duplicate_mappings = [
                    (Area, 'name', 'Area'),
                    (EquipmentGroup, 'name', 'EquipmentGroup'),
                    (Model, 'name', 'Model'),
                    (AssetNumber, 'number', 'AssetNumber'),
                    (Location, 'name', 'Location'),
                    (SiteLocation, 'title', 'SiteLocation')
                ]

                for model_class, attribute, sheet_name in duplicate_mappings:
                    self.delete_duplicates_enhanced(session, model_class, attribute, sheet_name)

                # Commit all changes
                session.commit()
                info_id("All database changes committed successfully", self.request_id)
                info_id("All changes committed to database")

                # Create revision snapshots
                self.create_revision_snapshots(session)

            # Display summary
            self.display_processing_summary()

            info_id("Equipment Relationships Import Completed Successfully!")
            info_id("Equipment relationships import completed successfully", self.request_id)

            return True

        except Exception as e:
            error_id(f"Equipment relationships import failed: {str(e)}", self.request_id, exc_info=True)
            error_id(f"Import failed: {str(e)}")
            return False


def main():
    """
    Main function to load equipment relationships data.
    Uses the new PostgreSQL framework with enhanced error handling and features.
    """
    info_id("Starting Equipment Relationships Data Import")
    info_id("=" * 55)

    loader = None
    try:
        # Initialize the PostgreSQL loader
        loader = PostgreSQLEquipmentRelationshipsLoader()

        # Load the equipment relationships data
        success = loader.load_equipment_relationships()

        if success:
            info_id("Equipment Relationships Import Completed Successfully!")
            info_id("=" * 55)
        else:
            warning_id("Equipment Relationships Import Completed with Issues")
            info_id("=" * 55)

    except KeyboardInterrupt:
        warning_id("Import interrupted by user")
        if loader:
            error_id("Import interrupted by user", loader.request_id)
    except Exception as e:
        error_id(f"Import failed: {str(e)}")
        if loader:
            error_id(f"Import failed: {str(e)}", loader.request_id, exc_info=True)
    finally:
        # Close logger
        try:
            close_initializer_logger()
        except:
            pass


if __name__ == "__main__":
    main()