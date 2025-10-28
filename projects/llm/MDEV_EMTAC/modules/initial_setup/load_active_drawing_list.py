import os
import sys
import pandas as pd
from sqlalchemy import and_, text
from datetime import datetime

# Import the new PostgreSQL framework components
from modules.initial_setup.initializer_logger import initializer_logger, close_initializer_logger
from modules.configuration.config import DB_LOADSHEET
from modules.configuration.config_env import DatabaseConfig
from modules.configuration.log_config import (
    debug_id, info_id, warning_id, error_id,
    set_request_id, get_request_id, log_timed_operation
)
from modules.emtacdb.emtacdb_fts import Drawing


class PostgreSQLDrawingListLoader:
    """PostgreSQL-optimized drawing list loader with duplicate prevention."""

    # Define column name constants to avoid hardcoding and prevent errors
    EXCEL_COLUMNS = {
        'EQUIPMENT_NUMBER': 'EQUIPMENT NUMBER',
        'EQUIPMENT_NAME': 'EQUIPMENT NAME',
        'DRAWING_NUMBER': 'DRAWING NUMBER',
        'DRAWING_NAME': 'DRAWING NAME',
        'REVISION': 'REVISION',
        'CC_REQUIRED': 'CC REQUIRED',
        'SPARE_PART_NUMBER': 'SPARE PART NUMBER'
    }

    # Database column names (standardized with underscores)
    DB_COLUMNS = {
        'EQUIPMENT_NUMBER': 'EQUIPMENT_NUMBER',
        'EQUIPMENT_NAME': 'EQUIPMENT_NAME',
        'DRAWING_NUMBER': 'DRAWING_NUMBER',
        'DRAWING_NAME': 'DRAWING_NAME',
        'REVISION': 'REVISION',
        'CC_REQUIRED': 'CC_REQUIRED',
        'SPARE_PART_NUMBER': 'SPARE_PART_NUMBER'
    }

    def __init__(self):
        self.request_id = set_request_id()
        self.db_config = DatabaseConfig()
        info_id("Initialized PostgreSQL Drawing List Loader", self.request_id)

    def check_existing_drawings(self, session):
        """Check for existing drawings to prevent duplicates."""
        try:
            count = session.query(Drawing).count()
            info_id(f"Found {count} existing drawings in database", self.request_id)

            if count > 0:
                print(f"\n⚠️  EXISTING DRAWINGS DETECTED")
                print(f"📊 Current drawings in database: {count}")
                print(f"🔄 Loading new drawings may create duplicates!")
                print(f"💡 Consider backing up your database first.\n")

                proceed = input("⚠️  Continue with drawing import anyway? (y/n): ").strip().lower()
                if proceed not in ['y', 'yes']:
                    info_id("User chose to skip drawing import due to existing data", self.request_id)
                    return False

            return True
        except Exception as e:
            error_id(f"Error checking existing drawings: {str(e)}", self.request_id)
            raise

    def validate_drawing_data(self, df):
        """Validate the drawing data before processing."""
        info_id("Validating drawing data structure", self.request_id)

        # Use Excel column names for validation since we haven't renamed yet
        required_columns = [
            self.EXCEL_COLUMNS['EQUIPMENT_NAME'],
            self.EXCEL_COLUMNS['DRAWING_NUMBER'],
            self.EXCEL_COLUMNS['DRAWING_NAME'],
            self.EXCEL_COLUMNS['REVISION'],
            self.EXCEL_COLUMNS['SPARE_PART_NUMBER']
        ]

        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            error_id(f"Missing required columns: {missing_columns}", self.request_id)
            raise ValueError(f"Missing required columns: {missing_columns}")

        # Check for empty critical fields
        critical_fields = [
            self.EXCEL_COLUMNS['DRAWING_NUMBER'],
            self.EXCEL_COLUMNS['DRAWING_NAME']
        ]
        for field in critical_fields:
            empty_count = df[field].isna().sum() + (df[field] == '').sum()
            if empty_count > 0:
                warning_id(f"Found {empty_count} rows with empty {field}", self.request_id)

        info_id(f"Validation complete. Processing {len(df)} drawing records", self.request_id)
        return True

    def prepare_drawing_data(self, df):
        """Prepare and clean the drawing data for PostgreSQL insertion."""
        info_id("Preparing drawing data for database insertion", self.request_id)

        # Drop extra columns if they exist
        extra_columns = [col for col in df.columns if 'Unnamed:' in str(col)]
        if extra_columns:
            df = df.drop(columns=extra_columns)
            debug_id(f"Dropped extra columns: {extra_columns}", self.request_id)

        # Create column mapping from Excel names to DB names
        column_mapping = {
            excel_name: db_name for db_name, excel_name in self.EXCEL_COLUMNS.items()
        }

        # Rename columns to match our standardized naming
        df = df.rename(columns=column_mapping)
        debug_id(f"Standardized column names: {list(df.columns)}", self.request_id)

        # Clean string data - strip whitespace and handle nulls
        string_columns = df.select_dtypes(include=['object']).columns
        for col in string_columns:
            df[col] = df[col].astype(str).str.strip()
            df[col] = df[col].replace('nan', None)
            df[col] = df[col].replace('', None)

        info_id("Data preparation completed", self.request_id)
        return df

    def check_for_duplicates_in_data(self, df):
        """Check for duplicates within the data being imported."""
        # Use database column names since data has been standardized
        drawing_number_col = self.DB_COLUMNS['DRAWING_NUMBER']
        drawing_name_col = self.DB_COLUMNS['DRAWING_NAME']

        duplicate_drawings = df[df.duplicated(subset=[drawing_number_col], keep=False)]

        if not duplicate_drawings.empty:
            warning_id(f"Found {len(duplicate_drawings)} duplicate drawing numbers in import data", self.request_id)
            print("\n⚠️  DUPLICATE DRAWINGS IN IMPORT DATA:")
            print(duplicate_drawings[[drawing_number_col, drawing_name_col]].to_string(index=False))
            print()

            handle_duplicates = input("How to handle duplicates? (skip/first/last): ").strip().lower()

            if handle_duplicates == 'skip':
                df = df.drop_duplicates(subset=[drawing_number_col], keep=False)
                info_id("Removed all duplicate drawing numbers", self.request_id)
            elif handle_duplicates == 'first':
                df = df.drop_duplicates(subset=[drawing_number_col], keep='first')
                info_id("Kept first occurrence of duplicate drawing numbers", self.request_id)
            elif handle_duplicates == 'last':
                df = df.drop_duplicates(subset=[drawing_number_col], keep='last')
                info_id("Kept last occurrence of duplicate drawing numbers", self.request_id)
            else:
                warning_id("Invalid choice, keeping all duplicates", self.request_id)

        return df

    def check_existing_drawing_numbers(self, session, drawing_numbers):
        """Check which drawing numbers already exist in the database."""
        try:
            existing_numbers = session.query(Drawing.drw_number).filter(
                Drawing.drw_number.in_(drawing_numbers)
            ).all()
            existing_set = {num[0] for num in existing_numbers}

            if existing_set:
                info_id(f"Found {len(existing_set)} drawing numbers already in database", self.request_id)
                debug_id(f"Existing drawing numbers: {list(existing_set)[:10]}...", self.request_id)

            return existing_set
        except Exception as e:
            error_id(f"Error checking existing drawing numbers: {str(e)}", self.request_id)
            raise

    def bulk_insert_drawings(self, session, drawings_data):
        """Efficiently insert drawings using PostgreSQL bulk operations."""
        # Fix: Use .empty instead of 'not drawings_data' for DataFrame
        if drawings_data.empty:
            info_id("No drawings to insert", self.request_id)
            return 0

        try:
            with log_timed_operation("bulk_insert_drawings", self.request_id):
                # Create Drawing objects using database column names
                drawing_objects = []
                for _, row in drawings_data.iterrows():
                    drawing = Drawing(
                        drw_equipment_name=row.get(self.DB_COLUMNS['EQUIPMENT_NAME']),
                        drw_number=row.get(self.DB_COLUMNS['DRAWING_NUMBER']),
                        drw_name=row.get(self.DB_COLUMNS['DRAWING_NAME']),
                        drw_revision=row.get(self.DB_COLUMNS['REVISION']),
                        drw_spare_part_number=row.get(self.DB_COLUMNS['SPARE_PART_NUMBER']),
                        file_path="active_drawing_list_import"  # Default path indicating source
                    )
                    drawing_objects.append(drawing)

                # Use SQLAlchemy's bulk insert for better performance
                session.bulk_save_objects(drawing_objects)
                session.commit()

                info_id(f"Successfully bulk inserted {len(drawing_objects)} drawings", self.request_id)
                return len(drawing_objects)

        except Exception as e:
            session.rollback()
            error_id(f"Error in bulk insert: {str(e)}", self.request_id)
            raise

    def load_drawing_list(self, file_path=None):
        """Main method to load the active drawing list."""
        try:
            # Determine file path
            if not file_path:
                file_path = os.path.join(DB_LOADSHEET, "active drawing list.xlsx")

            if not os.path.exists(file_path):
                error_id(f"Drawing list file not found: {file_path}", self.request_id)
                print(f"❌ File not found: {file_path}")
                return False

            info_id(f"Loading active drawing list from: {file_path}", self.request_id)
            print(f"📂 Loading drawing list from: {file_path}")

            # Read Excel file
            with log_timed_operation("read_excel", self.request_id):
                df = pd.read_excel(file_path)
                info_id(f"Read {len(df)} rows from Excel file", self.request_id)
                print(f"📊 Found {len(df)} drawing records")

            # Validate data structure (uses Excel column names)
            self.validate_drawing_data(df)

            # Prepare and clean data (converts to database column names)
            df = self.prepare_drawing_data(df)

            # Check for duplicates in import data (uses database column names)
            df = self.check_for_duplicates_in_data(df)

            if df.empty:
                warning_id("No data to import after duplicate handling", self.request_id)
                print("⚠️  No data to import after processing")
                return False

            # Get database session
            with self.db_config.main_session() as session:
                # Check for existing drawings
                if not self.check_existing_drawings(session):
                    return False

                # Check which drawing numbers already exist (using database column names)
                drawing_numbers = df[self.DB_COLUMNS['DRAWING_NUMBER']].dropna().unique().tolist()
                existing_numbers = self.check_existing_drawing_numbers(session, drawing_numbers)

                # Filter out existing drawings
                if existing_numbers:
                    original_count = len(df)
                    df = df[~df[self.DB_COLUMNS['DRAWING_NUMBER']].isin(existing_numbers)]
                    skipped_count = original_count - len(df)

                    if skipped_count > 0:
                        warning_id(f"Skipping {skipped_count} drawings that already exist", self.request_id)
                        print(f"⏭️  Skipping {skipped_count} drawings that already exist in database")

                if df.empty:
                    info_id("All drawings already exist in database", self.request_id)
                    print("✅ All drawings already exist in database")
                    return True

                # Bulk insert new drawings
                inserted_count = self.bulk_insert_drawings(session, df)

                print(f"✅ Successfully imported {inserted_count} new drawings")
                info_id(f"Drawing list import completed. Inserted {inserted_count} records", self.request_id)

                return True

        except Exception as e:
            error_id(f"Error in drawing list loading: {str(e)}", self.request_id, exc_info=True)
            print(f"❌ Error loading drawing list: {str(e)}")
            return False


def main():
    """
    Main function to load active drawing list data.
    Uses the new PostgreSQL framework with enhanced error handling and duplicate prevention.
    """
    print("\n🎯 Starting Active Drawing List Import")
    print("=" * 50)

    loader = None
    try:
        # Initialize the PostgreSQL loader
        loader = PostgreSQLDrawingListLoader()

        # Load the drawing list
        success = loader.load_drawing_list()

        if success:
            print("\n✅ Active Drawing List Import Completed Successfully!")
            print("=" * 50)
        else:
            print("\n⚠️  Active Drawing List Import Completed with Issues")
            print("=" * 50)

    except KeyboardInterrupt:
        print("\n🛑 Import interrupted by user")
        if loader:
            error_id("Import interrupted by user", loader.request_id)
    except Exception as e:
        print(f"\n❌ Import failed: {str(e)}")
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