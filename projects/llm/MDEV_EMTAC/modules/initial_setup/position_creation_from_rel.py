import sys
import os
import argparse
import time
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from collections import defaultdict
from itertools import product
from sqlalchemy import text

# --- IMPORT YOUR APP CONTEXT ---
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import the new PostgreSQL framework components
from modules.emtacdb.emtacdb_fts import (
    Area, EquipmentGroup, Model, AssetNumber, Location, Subassembly,
    ComponentAssembly, AssemblyView, Position
)
from modules.configuration.config_env import DatabaseConfig
from modules.configuration.log_config import (
    debug_id, info_id, warning_id, error_id,
    set_request_id, get_request_id, log_timed_operation
)
from modules.initial_setup.initializer_logger import initializer_logger, close_initializer_logger


class OptimizedPositionCreator:
    """High-performance position creator using vectorized operations and bulk database operations."""

    def __init__(self):
        self.request_id = set_request_id()
        self.db_config = DatabaseConfig()
        info_id("Initialized Optimized Position Creator", self.request_id)

        # Statistics tracking
        self.stats = {
            'total_combinations_found': 0,
            'total_combinations_processed': 0,
            'positions_created': 0,
            'duplicates_skipped': 0,
            'errors_encountered': 0,
            'processing_time': 0,
            'data_loading_time': 0,
            'combination_generation_time': 0,
            'duplicate_filtering_time': 0,
            'bulk_insertion_time': 0
        }

        # Data storage
        self.hierarchy_data = {}
        self.existing_positions = set()

    def validate_database_structure(self, session):
        """Validate database structure using optimized queries."""
        info_id("Validating database structure for position creation", self.request_id)

        try:
            with log_timed_operation("validate_database_structure", self.request_id):
                validation_results = {}

                # Check each required table using optimized counts
                tables_to_check = [
                    ('area', Area, 'Area'),
                    ('equipment_group', EquipmentGroup, 'EquipmentGroup'),
                    ('model', Model, 'Model'),
                    ('asset_number', AssetNumber, 'AssetNumber'),
                    ('location', Location, 'Location'),
                    ('position', Position, 'Position')
                ]

                for table_name, model_class, display_name in tables_to_check:
                    try:
                        if self.db_config.is_postgresql:
                            # Use PostgreSQL optimized count
                            count = session.execute(text(f"SELECT COUNT(*) FROM {table_name}")).scalar()
                        else:
                            count = session.query(model_class).count()

                        validation_results[display_name] = count

                        if count == 0 and display_name in ['Area', 'EquipmentGroup', 'Model']:
                            warning_id(f"Critical table {display_name} is empty", self.request_id)
                        else:
                            info_id(f"{display_name}: {count:,} records", self.request_id)

                    except Exception as e:
                        error_id(f"Error checking {display_name} table: {str(e)}", self.request_id)
                        return False

                # Check for critical missing data
                critical_missing = []
                for table in ['Area', 'EquipmentGroup', 'Model']:
                    if validation_results.get(table, 0) == 0:
                        critical_missing.append(table)

                if critical_missing:
                    error_id(f"Critical tables are empty: {critical_missing}", self.request_id)
                    return False

                info_id("Database structure validation successful", self.request_id)
                return True

        except Exception as e:
            error_id(f"Error validating database structure: {str(e)}", self.request_id)
            return False

    def load_hierarchy_data_vectorized(self, session, area_limit=None):
        """Load all hierarchy data using optimized bulk queries."""
        info_id("Loading hierarchy data using vectorized operations", self.request_id)

        try:
            start_time = time.time()

            with log_timed_operation("load_hierarchy_data", self.request_id):
                # Load all hierarchy data at once using pandas for speed
                if self.db_config.is_postgresql:
                    # Use PostgreSQL optimized queries

                    # Areas
                    area_query = "SELECT id, name FROM area"
                    if area_limit:
                        area_query += f" LIMIT {area_limit}"
                    areas_df = pd.read_sql(text(area_query), session.bind)

                    # Equipment Groups with area relationships
                    eq_groups_query = text("""
                        SELECT eg.id, eg.name, eg.area_id 
                        FROM equipment_group eg
                        WHERE eg.area_id IS NOT NULL
                    """)
                    eq_groups_df = pd.read_sql(eq_groups_query, session.bind)

                    # Models with equipment group relationships
                    models_query = text("""
                        SELECT m.id, m.name, m.equipment_group_id
                        FROM model m
                        WHERE m.equipment_group_id IS NOT NULL
                    """)
                    models_df = pd.read_sql(models_query, session.bind)

                    # Asset Numbers with model relationships
                    assets_query = text("""
                        SELECT an.id, an.number, an.model_id
                        FROM asset_number an
                        WHERE an.model_id IS NOT NULL
                    """)
                    assets_df = pd.read_sql(assets_query, session.bind)

                    # Locations with model relationships
                    locations_query = text("""
                        SELECT l.id, l.name, l.model_id
                        FROM location l
                        WHERE l.model_id IS NOT NULL
                    """)
                    locations_df = pd.read_sql(locations_query, session.bind)

                    # Subassemblies
                    subassemblies_query = text("""
                        SELECT s.id, s.name, s.location_id
                        FROM subassembly s
                        WHERE s.location_id IS NOT NULL
                    """)
                    subassemblies_df = pd.read_sql(subassemblies_query, session.bind)

                    # Component Assemblies
                    comp_assemblies_query = text("""
                        SELECT ca.id, ca.name, ca.subassembly_id
                        FROM component_assembly ca
                        WHERE ca.subassembly_id IS NOT NULL
                    """)
                    comp_assemblies_df = pd.read_sql(comp_assemblies_query, session.bind)

                    # Assembly Views
                    assembly_views_query = text("""
                        SELECT av.id, av.name, av.component_assembly_id
                        FROM assembly_view av
                        WHERE av.component_assembly_id IS NOT NULL
                    """)
                    assembly_views_df = pd.read_sql(assembly_views_query, session.bind)

                else:
                    # SQLite fallback - still use pandas but with ORM queries
                    areas_query = session.query(Area.id, Area.name)
                    if area_limit:
                        areas_query = areas_query.limit(area_limit)
                    areas_df = pd.read_sql(areas_query.statement, session.bind)

                    # ... similar for other tables (simplified for brevity)

                # Store the data
                self.hierarchy_data = {
                    'areas': areas_df,
                    'equipment_groups': eq_groups_df,
                    'models': models_df,
                    'assets': assets_df,
                    'locations': locations_df,
                    'subassemblies': subassemblies_df,
                    'component_assemblies': comp_assemblies_df,
                    'assembly_views': assembly_views_df
                }

                # Apply area limit filtering to related data
                if area_limit and not areas_df.empty:
                    area_ids = set(areas_df['id'].tolist())

                    # Filter equipment groups
                    self.hierarchy_data['equipment_groups'] = eq_groups_df[
                        eq_groups_df['area_id'].isin(area_ids)
                    ]

                    # Filter models through equipment groups
                    eg_ids = set(self.hierarchy_data['equipment_groups']['id'].tolist())
                    self.hierarchy_data['models'] = models_df[
                        models_df['equipment_group_id'].isin(eg_ids)
                    ]

                    # Continue filtering down the hierarchy...

                self.stats['data_loading_time'] = time.time() - start_time

                # Log summary
                total_records = sum(len(df) for df in self.hierarchy_data.values())
                info_id(f"Loaded {total_records} hierarchy records", self.request_id)
                for name, df in self.hierarchy_data.items():
                    info_id(f"{name.title()}: {len(df):,}", self.request_id)

                return True

        except Exception as e:
            error_id(f"Error loading hierarchy data: {str(e)}", self.request_id)
            return False

    def load_existing_positions_vectorized(self, session):
        """Load existing positions using optimized bulk query."""
        try:
            info_id("Loading existing positions using vectorized operations", self.request_id)

            with log_timed_operation("load_existing_positions", self.request_id):
                if self.db_config.is_postgresql:
                    # Use pandas for fast bulk loading
                    existing_query = text("""
                        SELECT area_id, equipment_group_id, model_id, 
                               asset_number_id, location_id, subassembly_id,
                               component_assembly_id, assembly_view_id, site_location_id
                        FROM position
                    """)
                    existing_df = pd.read_sql(existing_query, session.bind)
                else:
                    # SQLite fallback
                    existing_query = session.query(
                        Position.area_id, Position.equipment_group_id, Position.model_id,
                        Position.asset_number_id, Position.location_id, Position.subassembly_id,
                        Position.component_assembly_id, Position.assembly_view_id, Position.site_location_id
                    )
                    existing_df = pd.read_sql(existing_query.statement, session.bind)

                # Convert to set of tuples for fast lookup
                if not existing_df.empty:
                    # Handle NaN values by converting to None
                    existing_df = existing_df.where(pd.notnull(existing_df), None)
                    self.existing_positions = set(
                        tuple(row) for row in existing_df.values
                    )

                info_id(f"Loaded {len(self.existing_positions)} existing positions", self.request_id)

        except Exception as e:
            error_id(f"Error loading existing positions: {str(e)}", self.request_id)
            self.existing_positions = set()

    def generate_position_combinations_vectorized(self):
        """Generate all possible position combinations using vectorized operations."""
        info_id("Generating position combinations using vectorized operations", self.request_id)

        try:
            start_time = time.time()

            with log_timed_operation("generate_combinations", self.request_id):
                all_combinations = []

                # Get base data
                areas_df = self.hierarchy_data['areas']
                eq_groups_df = self.hierarchy_data['equipment_groups']
                models_df = self.hierarchy_data['models']
                assets_df = self.hierarchy_data['assets']
                locations_df = self.hierarchy_data['locations']

                info_id(f"Processing {len(areas_df):,} areas", self.request_id)

                # For each area, generate combinations using vectorized operations
                for _, area in areas_df.iterrows():
                    area_id = area['id']

                    # Get equipment groups for this area
                    area_eq_groups = eq_groups_df[eq_groups_df['area_id'] == area_id]

                    for _, eq_group in area_eq_groups.iterrows():
                        eq_group_id = eq_group['id']

                        # Get models for this equipment group
                        eq_models = models_df[models_df['equipment_group_id'] == eq_group_id]

                        for _, model in eq_models.iterrows():
                            model_id = model['id']

                            # Asset number path - vectorized
                            model_assets = assets_df[assets_df['model_id'] == model_id]
                            for _, asset in model_assets.iterrows():
                                all_combinations.append({
                                    'area_id': area_id,
                                    'equipment_group_id': eq_group_id,
                                    'model_id': model_id,
                                    'asset_number_id': asset['id'],
                                    'location_id': None,
                                    'subassembly_id': None,
                                    'component_assembly_id': None,
                                    'assembly_view_id': None,
                                    'site_location_id': None
                                })

                            # Location hierarchy path - vectorized
                            model_locations = locations_df[locations_df['model_id'] == model_id]
                            location_combinations = self._generate_location_combinations_vectorized(
                                area_id, eq_group_id, model_id, model_locations
                            )
                            all_combinations.extend(location_combinations)

                # Convert to DataFrame for vectorized operations
                combinations_df = pd.DataFrame(all_combinations)

                self.stats['combination_generation_time'] = time.time() - start_time
                self.stats['total_combinations_found'] = len(combinations_df)

                info_id(f"Generated {len(combinations_df):,} position combinations", self.request_id)

                return combinations_df

        except Exception as e:
            error_id(f"Error generating combinations: {str(e)}", self.request_id)
            raise

    def _generate_location_combinations_vectorized(self, area_id, eq_group_id, model_id, locations_df):
        """Generate location hierarchy combinations using vectorized operations."""
        combinations = []

        subassemblies_df = self.hierarchy_data['subassemblies']
        comp_assemblies_df = self.hierarchy_data['component_assemblies']
        assembly_views_df = self.hierarchy_data['assembly_views']

        for _, location in locations_df.iterrows():
            location_id = location['id']

            # Get subassemblies for this location
            location_subassemblies = subassemblies_df[subassemblies_df['location_id'] == location_id]

            if location_subassemblies.empty:
                # Location with no subassemblies
                combinations.append({
                    'area_id': area_id,
                    'equipment_group_id': eq_group_id,
                    'model_id': model_id,
                    'asset_number_id': None,
                    'location_id': location_id,
                    'subassembly_id': None,
                    'component_assembly_id': None,
                    'assembly_view_id': None,
                    'site_location_id': None
                })
            else:
                # Process each subassembly
                for _, subassembly in location_subassemblies.iterrows():
                    subassembly_id = subassembly['id']

                    # Get component assemblies for this subassembly
                    sub_comp_assemblies = comp_assemblies_df[comp_assemblies_df['subassembly_id'] == subassembly_id]

                    if sub_comp_assemblies.empty:
                        # Subassembly with no component assemblies
                        combinations.append({
                            'area_id': area_id,
                            'equipment_group_id': eq_group_id,
                            'model_id': model_id,
                            'asset_number_id': None,
                            'location_id': location_id,
                            'subassembly_id': subassembly_id,
                            'component_assembly_id': None,
                            'assembly_view_id': None,
                            'site_location_id': None
                        })
                    else:
                        # Process each component assembly
                        for _, comp_assembly in sub_comp_assemblies.iterrows():
                            comp_assembly_id = comp_assembly['id']

                            # Get assembly views for this component assembly
                            comp_assembly_views = assembly_views_df[
                                assembly_views_df['component_assembly_id'] == comp_assembly_id]

                            if comp_assembly_views.empty:
                                # Component assembly with no assembly views
                                combinations.append({
                                    'area_id': area_id,
                                    'equipment_group_id': eq_group_id,
                                    'model_id': model_id,
                                    'asset_number_id': None,
                                    'location_id': location_id,
                                    'subassembly_id': subassembly_id,
                                    'component_assembly_id': comp_assembly_id,
                                    'assembly_view_id': None,
                                    'site_location_id': None
                                })
                            else:
                                # Process each assembly view
                                for _, assembly_view in comp_assembly_views.iterrows():
                                    combinations.append({
                                        'area_id': area_id,
                                        'equipment_group_id': eq_group_id,
                                        'model_id': model_id,
                                        'asset_number_id': None,
                                        'location_id': location_id,
                                        'subassembly_id': subassembly_id,
                                        'component_assembly_id': comp_assembly_id,
                                        'assembly_view_id': assembly_view['id'],
                                        'site_location_id': None
                                    })

        return combinations

    def filter_duplicates_vectorized(self, combinations_df):
        """Filter out duplicate combinations using vectorized operations."""
        info_id("Filtering duplicates using vectorized operations", self.request_id)

        try:
            start_time = time.time()

            with log_timed_operation("filter_duplicates", self.request_id):
                initial_count = len(combinations_df)

                if self.existing_positions:
                    # Create a column for tuple comparison
                    combinations_df['position_tuple'] = combinations_df.apply(
                        lambda row: (
                            row['area_id'], row['equipment_group_id'], row['model_id'],
                            row['asset_number_id'], row['location_id'], row['subassembly_id'],
                            row['component_assembly_id'], row['assembly_view_id'], row['site_location_id']
                        ), axis=1
                    )

                    # Filter out existing positions using vectorized operations
                    mask = ~combinations_df['position_tuple'].isin(self.existing_positions)
                    new_combinations_df = combinations_df[mask].drop(columns=['position_tuple'])

                    duplicates_filtered = initial_count - len(new_combinations_df)
                else:
                    new_combinations_df = combinations_df
                    duplicates_filtered = 0

                # Remove internal duplicates (shouldn't happen, but safety check)
                if not new_combinations_df.empty:
                    pre_dedup_count = len(new_combinations_df)
                    new_combinations_df = new_combinations_df.drop_duplicates()
                    internal_duplicates = pre_dedup_count - len(new_combinations_df)
                else:
                    internal_duplicates = 0

                self.stats['duplicate_filtering_time'] = time.time() - start_time
                self.stats['duplicates_skipped'] = duplicates_filtered + internal_duplicates
                self.stats['total_combinations_processed'] = len(new_combinations_df)

                info_id(
                    f"Filtered {duplicates_filtered + internal_duplicates} duplicates, {len(new_combinations_df)} new positions to create",
                    self.request_id)

                return new_combinations_df

        except Exception as e:
            error_id(f"Error filtering duplicates: {str(e)}", self.request_id)
            raise

    def bulk_insert_positions_optimized(self, session, positions_df):
        """Bulk insert positions using optimized database operations."""
        if positions_df.empty:
            info_id("No positions to insert", self.request_id)
            return

        try:
            info_id(f"Bulk inserting {len(positions_df)} positions", self.request_id)

            start_time = time.time()

            with log_timed_operation("bulk_insert_positions", self.request_id):
                # Clean DataFrame - convert NaN to None for PostgreSQL compatibility
                positions_df_clean = positions_df.copy()

                # Replace all NaN values with None (NULL for database)
                positions_df_clean = positions_df_clean.where(pd.notnull(positions_df_clean), None)

                # Ensure integer columns are properly typed (convert float64 to int where needed)
                integer_columns = ['area_id', 'equipment_group_id', 'model_id', 'asset_number_id',
                                   'location_id', 'subassembly_id', 'component_assembly_id',
                                   'assembly_view_id', 'site_location_id']

                for col in integer_columns:
                    if col in positions_df_clean.columns:
                        # Convert to nullable integer type, handling None values properly
                        positions_df_clean[col] = positions_df_clean[col].astype('Int64')
                        # Convert back to regular int/None for database compatibility
                        positions_df_clean[col] = positions_df_clean[col].apply(
                            lambda x: int(x) if pd.notnull(x) else None
                        )

                # Convert DataFrame to list of dictionaries for bulk insert
                positions_data = positions_df_clean.to_dict('records')

                # Double-check: ensure no NaN values remain
                cleaned_positions_data = []
                for record in positions_data:
                    cleaned_record = {}
                    for key, value in record.items():
                        if pd.isna(value) or (isinstance(value, float) and np.isnan(value)):
                            cleaned_record[key] = None
                        else:
                            cleaned_record[key] = value
                    cleaned_positions_data.append(cleaned_record)

                # Use SQLAlchemy's bulk insert for maximum performance
                session.bulk_insert_mappings(Position, cleaned_positions_data)
                session.commit()

                self.stats['bulk_insertion_time'] = time.time() - start_time
                self.stats['positions_created'] = len(cleaned_positions_data)

                info_id(f"Successfully bulk inserted {len(cleaned_positions_data)} positions", self.request_id)

        except Exception as e:
            session.rollback()
            error_id(f"Error in bulk insert: {str(e)}", self.request_id)
            raise

    def display_optimized_summary(self):
        """Log comprehensive processing summary."""
        info_id("Optimized position creation completed", self.request_id)
        info_id(f"Total combinations found: {self.stats['total_combinations_found']:,}", self.request_id)
        info_id(f"Combinations processed: {self.stats['total_combinations_processed']:,}", self.request_id)
        info_id(f"New positions created: {self.stats['positions_created']:,}", self.request_id)
        info_id(f"Duplicates skipped: {self.stats['duplicates_skipped']:,}", self.request_id)

        if self.stats['errors_encountered'] > 0:
            error_id(f"Errors encountered: {self.stats['errors_encountered']:,}", self.request_id)

        info_id(f"Total processing time: {self._format_time(self.stats['processing_time'])}", self.request_id)

        # Detailed timing breakdown
        info_id("Performance breakdown:", self.request_id)
        info_id(f"Data loading: {self._format_time(self.stats['data_loading_time'])}", self.request_id)
        info_id(f"Combination generation: {self._format_time(self.stats['combination_generation_time'])}", self.request_id)
        info_id(f"Duplicate filtering: {self._format_time(self.stats['duplicate_filtering_time'])}", self.request_id)
        info_id(f"Bulk insertion: {self._format_time(self.stats['bulk_insertion_time'])}", self.request_id)

        if self.stats['processing_time'] > 0:
            rate = self.stats['positions_created'] / self.stats['processing_time']
            info_id(f"Processing rate: {rate:.1f} positions/sec", self.request_id)

    def _format_time(self, seconds):
        """Format seconds into readable time string."""
        if seconds < 60:
            return f"{seconds:.2f}s"
        elif seconds < 3600:
            return f"{int(seconds // 60)}m {seconds % 60:.1f}s"
        else:
            hours = int(seconds // 3600)
            minutes = int((seconds % 3600) // 60)
            return f"{hours}h {minutes}m"

    def run_optimized_position_creation(self, area_limit=None, dry_run=False):
        """Main optimized method to run position creation."""
        try:
            info_id("Starting optimized equipment position creation", self.request_id)

            if dry_run:
                info_id("Running in optimized dry-run mode", self.request_id)

            total_start_time = time.time()

            # Get database session
            with self.db_config.main_session() as session:
                # Validate database structure
                if not self.validate_database_structure(session):
                    return False

                # Load all hierarchy data using vectorized operations
                if not self.load_hierarchy_data_vectorized(session, area_limit):
                    return False

                # Load existing positions using vectorized operations
                self.load_existing_positions_vectorized(session)

                # Generate all position combinations using vectorized operations
                combinations_df = self.generate_position_combinations_vectorized()

                if combinations_df.empty:
                    info_id("No position combinations found to process", self.request_id)
                    return True

                # Filter duplicates using vectorized operations
                new_positions_df = self.filter_duplicates_vectorized(combinations_df)

                if dry_run:
                    info_id(f"Dry run complete - would have created {len(new_positions_df):,} positions", self.request_id)
                    return True

                # Bulk insert positions using optimized operations
                self.bulk_insert_positions_optimized(session, new_positions_df)

            # Update final statistics
            self.stats['processing_time'] = time.time() - total_start_time

            # Log summary
            self.display_optimized_summary()

            return True

        except Exception as e:
            error_id(f"Optimized position creation failed: {str(e)}", self.request_id, exc_info=True)
            return False


def main():
    """Main function with enhanced argument parsing."""
    parser = argparse.ArgumentParser(
        description="Create equipment positions using optimized vectorized operations",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--area-limit", type=int,
                        help="Limit processing to first N areas (for testing)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Show what would be created without actually creating positions")

    args = parser.parse_args()

    info_id("Starting optimized equipment position creation", request_id=None)

    creator = None
    try:
        # Initialize the optimized creator
        creator = OptimizedPositionCreator()

        # Run optimized position creation
        success = creator.run_optimized_position_creation(
            area_limit=args.area_limit,
            dry_run=args.dry_run
        )

        if success:
            info_id("Optimized position creation completed successfully", creator.request_id)
        else:
            warning_id("Optimized position creation completed with issues", creator.request_id)

    except KeyboardInterrupt:
        error_id("Operation interrupted by user", creator.request_id if creator else None)
    except Exception as e:
        error_id(f"Operation failed: {str(e)}", creator.request_id if creator else None, exc_info=True)
    finally:
        # Close logger
        try:
            close_initializer_logger()
        except:
            pass


if __name__ == "__main__":
    main()