import os
import time
import pandas as pd
import numpy as np
from datetime import datetime
from sqlalchemy import func, text

# Import the new PostgreSQL framework components
from modules.emtacdb.emtacdb_fts import Part
from modules.configuration.config import DB_LOADSHEET
from modules.configuration.config_env import DatabaseConfig
from modules.configuration.log_config import (
    debug_id, info_id, warning_id, error_id,
    set_request_id, get_request_id, log_timed_operation
)
from modules.initial_setup.initializer_logger import initializer_logger, close_initializer_logger
from modules.database_manager.db_manager import RelationshipManager


class OptimizedPostgreSQLPartsSheetLoader:
    """High-performance PostgreSQL parts sheet loader with vectorized operations."""

    def __init__(self):
        self.request_id = set_request_id()
        self.db_config = DatabaseConfig()
        info_id("Initialized Optimized PostgreSQL Parts Sheet Loader", self.request_id)

        # Statistics tracking
        self.stats = {
            'total_rows_found': 0,
            'new_parts_added': 0,
            'duplicates_skipped': 0,
            'errors_encountered': 0,
            'associations_created': 0,
            'processing_time': 0
        }

        # Column mapping for cleaner code
        self.column_mapping = {
            'part_number': 'ITEMNUM',
            'name': 'DESCRIPTION',
            'oem_mfg': 'OEMMFG',
            'model': 'MODEL',
            'class_flag': 'Class Flag',
            'ud6': 'UD6',
            'type': 'TYPE',
            'notes': 'Notes',
            'documentation': 'Specifications'
        }

        # Expected columns for validation
        self.required_columns = list(self.column_mapping.values())

    def validate_excel_file(self, file_path):
        """Validate the Excel file structure and accessibility."""
        info_id(f"Validating Excel file: {file_path}", self.request_id)

        if not os.path.exists(file_path):
            error_id(f"Excel file not found: {file_path}", self.request_id)
            return False, "File does not exist"

        try:
            # Check if file is readable
            excel_file = pd.ExcelFile(file_path)

            # Check if required sheet exists
            sheet_name = "EQUIP_BOMS"
            if sheet_name not in excel_file.sheet_names:
                available_sheets = ", ".join(excel_file.sheet_names)
                error_id(f"Required sheet '{sheet_name}' not found. Available sheets: {available_sheets}",
                         self.request_id)
                return False, f"Sheet '{sheet_name}' not found. Available: {available_sheets}"

            # Try to read a few rows to validate structure
            df_sample = pd.read_excel(file_path, sheet_name=sheet_name, nrows=5)

            # Check for required columns
            missing_columns = [col for col in self.required_columns if col not in df_sample.columns]
            if missing_columns:
                error_id(f"Missing required columns: {missing_columns}", self.request_id)
                return False, f"Missing required columns: {missing_columns}"

            info_id("Excel file validation successful", self.request_id)
            return True, "Valid"

        except Exception as e:
            error_id(f"Error validating Excel file: {str(e)}", self.request_id)
            return False, f"Error reading file: {str(e)}"

    def check_existing_parts(self, session):
        """Check for existing parts to inform user about duplicates."""
        try:
            info_id("Checking existing parts in database", self.request_id)

            # Get count of existing parts
            part_count = session.query(Part).count()

            if part_count > 0:
                info_id(f"Found {part_count:,} existing parts in database", self.request_id)
                proceed = input("Continue with parts import? (y/n): ").strip().lower()
                if proceed not in ['y', 'yes']:
                    info_id("User chose to skip parts import due to existing data", self.request_id)
                    return False

            return True

        except Exception as e:
            error_id(f"Error checking existing parts: {str(e)}", self.request_id)
            raise

    def create_parts_backup(self, session):
        """Create a backup of existing parts data."""
        try:
            info_id("Creating parts backup", self.request_id)

            with log_timed_operation("create_parts_backup", self.request_id):
                # Define backup directory
                backup_directory = os.path.join(DB_LOADSHEET, "..", "Database", "DB_LOADSHEETS_BACKUP")
                os.makedirs(backup_directory, exist_ok=True)

                # Create timestamped backup file
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                backup_filename = f"parts_backup_{timestamp}.xlsx"
                backup_path = os.path.join(backup_directory, backup_filename)

                # Query existing parts using pandas for speed
                if self.db_config.is_postgresql:
                    query = text("""
                        SELECT part_number, name, oem_mfg, model, class_flag, 
                               ud6, type, notes, documentation 
                        FROM part ORDER BY part_number
                    """)
                    backup_df = pd.read_sql(query, session.bind)
                else:
                    # SQLite fallback
                    parts_query = session.query(
                        Part.part_number, Part.name, Part.oem_mfg, Part.model,
                        Part.class_flag, Part.ud6, Part.type, Part.notes, Part.documentation
                    ).all()
                    backup_df = pd.DataFrame(parts_query, columns=[
                        'part_number', 'name', 'oem_mfg', 'model', 'class_flag',
                        'ud6', 'type', 'notes', 'documentation'
                    ])

                if not backup_df.empty:
                    # Save to Excel
                    backup_df.to_excel(backup_path, index=False, sheet_name='Parts_Backup')
                    info_id(f"Parts backup created: {backup_filename} ({len(backup_df)} records)", self.request_id)
                else:
                    info_id("No existing parts to backup", self.request_id)

        except Exception as e:
            error_id(f"Error creating parts backup: {str(e)}", self.request_id)
            # Don't raise - backup failure shouldn't stop the import

    def clean_and_validate_data(self, df):
        """Clean and validate the parts data using vectorized operations."""
        info_id("Cleaning and validating parts data", self.request_id)

        try:
            original_rows = len(df)
            self.stats['total_rows_found'] = original_rows

            # Replace NaN values with None for database compatibility
            df = df.replace({np.nan: None, '': None})

            # Vectorized cleaning of string columns
            string_columns = list(self.column_mapping.values())
            for col in string_columns:
                if col in df.columns:
                    # Convert to string and strip whitespace
                    df[col] = df[col].astype(str).str.strip()
                    # Replace 'nan' strings with None
                    df[col] = df[col].replace(['nan', ''], None)

            # Validate critical fields using vectorized operations
            critical_fields = ['ITEMNUM', 'DESCRIPTION']
            for field in critical_fields:
                if field in df.columns:
                    initial_count = len(df)
                    # Vectorized filtering - much faster than row-by-row
                    df = df[
                        df[field].notna() &
                        (df[field] != '') &
                        (df[field] != 'nan')
                        ]
                    removed_count = initial_count - len(df)

                    if removed_count > 0:
                        warning_id(f"Removed {removed_count} rows with empty {field}", self.request_id)

            # Validate part numbers format using vectorized operations
            if 'ITEMNUM' in df.columns:
                initial_count = len(df)
                # Vectorized string length check
                df = df[df['ITEMNUM'].astype(str).str.len() >= 3]
                invalid_count = initial_count - len(df)

                if invalid_count > 0:
                    warning_id(f"Removed {invalid_count} rows with invalid part numbers", self.request_id)

            final_rows = len(df)
            info_id(f"Data cleaning complete: {original_rows} -> {final_rows} rows", self.request_id)

            return df

        except Exception as e:
            error_id(f"Error cleaning data: {str(e)}", self.request_id)
            raise

    def get_existing_part_numbers_fast(self, session):
        """Get existing part numbers using optimized PostgreSQL query."""
        try:
            info_id("Fetching existing part numbers from database", self.request_id)

            with log_timed_operation("fetch_existing_parts", self.request_id):
                if self.db_config.is_postgresql:
                    # Use pandas read_sql for maximum speed
                    query = text("SELECT part_number FROM part WHERE part_number IS NOT NULL")
                    existing_df = pd.read_sql(query, session.bind)
                    existing_set = set(existing_df['part_number'].tolist())
                else:
                    # SQLite fallback
                    existing_parts = session.query(Part.part_number).filter(Part.part_number.isnot(None)).all()
                    existing_set = {part[0] for part in existing_parts}

                info_id(f"Retrieved {len(existing_set)} existing part numbers", self.request_id)
                return existing_set

        except Exception as e:
            error_id(f"Error fetching existing part numbers: {str(e)}", self.request_id)
            raise

    def process_parts_data_vectorized(self, session, df):
        """Process parts data using vectorized operations."""
        info_id("Processing parts data using vectorized operations", self.request_id)

        try:
            # Get existing part numbers
            existing_part_numbers = self.get_existing_part_numbers_fast(session)
            info_id(f"Found {len(existing_part_numbers):,} existing parts in database", self.request_id)

            # Remove duplicates within the import data first (vectorized)
            initial_count = len(df)
            df_dedupe = df.drop_duplicates(subset=['ITEMNUM'], keep='last')
            internal_dupes = initial_count - len(df_dedupe)

            if internal_dupes > 0:
                info_id(f"Removed {internal_dupes:,} internal duplicates", self.request_id)

            # Filter out existing parts using vectorized operations
            if existing_part_numbers:
                new_parts_mask = ~df_dedupe['ITEMNUM'].isin(existing_part_numbers)
                df_new = df_dedupe[new_parts_mask]
                db_dupes = len(df_dedupe) - len(df_new)
            else:
                df_new = df_dedupe
                db_dupes = 0

            # Create dictionary mapping using vectorized operations
            parts_data = []
            if not df_new.empty:
                info_id(f"Processing {len(df_new):,} new parts", self.request_id)

                # Use vectorized operations to create the parts list
                for db_col, excel_col in self.column_mapping.items():
                    if excel_col not in df_new.columns:
                        df_new[excel_col] = None

                # Convert to dictionary format efficiently
                parts_data = df_new[list(self.column_mapping.values())].rename(
                    columns={v: k for k, v in self.column_mapping.items()}
                ).to_dict('records')

            # Update statistics
            self.stats['new_parts_added'] = len(parts_data)
            self.stats['duplicates_skipped'] = internal_dupes + db_dupes

            info_id(f"Vectorized processing complete: {len(parts_data)} new parts", self.request_id)
            info_id(f"Total duplicates skipped: {internal_dupes + db_dupes:,} (Internal: {internal_dupes:,}, Database: {db_dupes:,})", self.request_id)

            return parts_data

        except Exception as e:
            error_id(f"Error processing parts data: {str(e)}", self.request_id)
            raise

    def bulk_insert_parts_optimized(self, session, new_parts):
        """Optimized bulk insertion using PostgreSQL-specific features."""
        if not new_parts:
            info_id("No new parts to insert", self.request_id)
            return []

        try:
            info_id(f"Bulk inserting {len(new_parts)} parts", self.request_id)

            with log_timed_operation("bulk_insert_parts_optimized", self.request_id):
                if self.db_config.is_postgresql:
                    # Use the fastest bulk insert method for PostgreSQL
                    session.bulk_insert_mappings(Part, new_parts)
                    session.commit()

                    # Get IDs using a single optimized query
                    part_numbers = [p['part_number'] for p in new_parts]

                    # Use pandas for fast ID retrieval
                    query = text("""
                        SELECT id, part_number 
                        FROM part 
                        WHERE part_number = ANY(:part_numbers)
                    """)
                    id_df = pd.read_sql(query, session.bind, params={'part_numbers': part_numbers})
                    new_part_ids = id_df['id'].tolist()
                else:
                    # SQLite fallback
                    session.bulk_insert_mappings(Part, new_parts)
                    session.commit()

                    part_numbers = [p['part_number'] for p in new_parts]
                    newly_inserted_parts = session.query(Part.id).filter(
                        Part.part_number.in_(part_numbers)
                    ).all()
                    new_part_ids = [part.id for part in newly_inserted_parts]

                info_id(f"Successfully inserted {len(new_parts)} parts, retrieved {len(new_part_ids)} IDs",
                        self.request_id)
                return new_part_ids

        except Exception as e:
            session.rollback()
            error_id(f"Error in bulk insert: {str(e)}", self.request_id)
            raise

    def create_part_image_associations(self, session, new_part_ids):
        """Create automatic part-image associations with enhanced error handling."""
        if not new_part_ids:
            return

        try:
            info_id("Starting automatic part-image association process", self.request_id)

            with log_timed_operation("create_part_image_associations", self.request_id):
                # Use RelationshipManager for associations
                with RelationshipManager(session=session, request_id=self.request_id) as manager:
                    result = manager.associate_parts_with_images_by_title(part_ids=new_part_ids)
                    manager.commit()

                    # Count total associations created
                    total_associations = sum(len(assocs) for assocs in result.values())
                    self.stats['associations_created'] = total_associations

                    if total_associations > 0:
                        info_id(f"Created {total_associations} part-image associations", self.request_id)
                    else:
                        info_id("No part-image associations created", self.request_id)

        except Exception as e:
            error_id(f"Error during part-image association: {str(e)}", self.request_id)
            # Don't raise - association failure shouldn't stop the main import

    def display_final_summary(self):
        """Log comprehensive processing summary."""
        info_id("Parts import completed", self.request_id)
        info_id(f"Total rows processed: {self.stats['total_rows_found']:,}", self.request_id)
        info_id(f"New parts added: {self.stats['new_parts_added']:,}", self.request_id)
        info_id(f"Duplicates skipped: {self.stats['duplicates_skipped']:,}", self.request_id)
        info_id(f"Associations created: {self.stats['associations_created']:,}", self.request_id)

        if self.stats['errors_encountered'] > 0:
            error_id(f"Errors encountered: {self.stats['errors_encountered']:,}", self.request_id)

        if self.stats['processing_time'] > 0:
            info_id(f"Processing time: {self._format_time(self.stats['processing_time'])}", self.request_id)

    def _format_time(self, seconds):
        """Format seconds into readable time string."""
        if seconds < 60:
            return f"{int(seconds)}s"
        elif seconds < 3600:
            return f"{int(seconds // 60)}m {int(seconds % 60)}s"
        else:
            hours = int(seconds // 3600)
            minutes = int((seconds % 3600) // 60)
            return f"{hours}h {minutes}m"

    def load_parts_from_excel(self, file_path=None):
        """Main method to load parts from Excel file using optimized operations."""
        try:
            info_id("Starting optimized parts sheet data import", self.request_id)

            # Determine file path
            if not file_path:
                load_sheet_filename = "load_MP2_ITEMS_BOMS.xlsx"
                file_path = os.path.join(DB_LOADSHEET, load_sheet_filename)

            info_id(f"Source file: {os.path.basename(file_path)}", self.request_id)

            # Validate Excel file
            is_valid, message = self.validate_excel_file(file_path)
            if not is_valid:
                raise ValueError(f"Invalid Excel file: {message}")

            start_time = time.time()

            # Get database session
            with self.db_config.main_session() as session:
                # Check existing parts
                if not self.check_existing_parts(session):
                    return False

                # Create backup
                self.create_parts_backup(session)

                # Load data using optimized reading
                info_id(f"Loading Excel data", self.request_id)
                with log_timed_operation("load_excel_data", self.request_id):
                    # Read entire sheet at once - faster than chunks for most cases
                    df = pd.read_excel(file_path, sheet_name="EQUIP_BOMS")
                    info_id(f"Loaded Excel sheet with {len(df)} rows", self.request_id)

                # Clean and validate data using vectorized operations
                df_cleaned = self.clean_and_validate_data(df)

                # Process parts data using vectorized operations
                new_parts = self.process_parts_data_vectorized(session, df_cleaned)

                # Bulk insert parts using optimized method
                new_part_ids = self.bulk_insert_parts_optimized(session, new_parts)

                # Create associations
                self.create_part_image_associations(session, new_part_ids)

            # Update final statistics
            self.stats['processing_time'] = time.time() - start_time

            # Log summary
            self.display_final_summary()

            info_id("Optimized parts import completed successfully", self.request_id)
            return True

        except Exception as e:
            error_id(f"Optimized parts import failed: {str(e)}", self.request_id, exc_info=True)
            return False


def main():
    """
    Main function to load parts data using the optimized PostgreSQL framework.
    """
    info_id("Starting optimized parts sheet import", request_id=None)

    loader = None
    try:
        # Initialize the optimized PostgreSQL loader
        loader = OptimizedPostgreSQLPartsSheetLoader()

        # Load the parts data
        success = loader.load_parts_from_excel()

        if success:
            info_id("Optimized parts sheet import completed successfully", loader.request_id)
        else:
            warning_id("Optimized parts sheet import completed with issues", loader.request_id)

    except KeyboardInterrupt:
        error_id("Import interrupted by user", loader.request_id if loader else None)
    except Exception as e:
        error_id(f"Import failed: {str(e)}", loader.request_id if loader else None, exc_info=True)
    finally:
        # Close logger
        try:
            close_initializer_logger()
        except:
            pass


if __name__ == "__main__":
    main()