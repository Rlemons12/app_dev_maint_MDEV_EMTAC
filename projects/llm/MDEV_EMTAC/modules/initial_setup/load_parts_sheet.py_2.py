import os
import time
import pandas as pd
import numpy as np
from datetime import datetime
from sqlalchemy import func, text
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import multiprocessing as mp
from contextlib import contextmanager
import psycopg2
from psycopg2.extras import execute_values

# Ensure pandas imports for robust NA handling
import pandas._libs.missing

try:
    from pandas import NA
except ImportError:
    NA = None

# Your existing imports...
from modules.emtacdb.emtacdb_fts import Part
from modules.configuration.config import DB_LOADSHEET
from modules.configuration.config_env import DatabaseConfig
from modules.configuration.log_config import (
    debug_id, info_id, warning_id, error_id,
    set_request_id, get_request_id, log_timed_operation
)


class SuperOptimizedPostgreSQLPartsSheetLoader:
    """Ultra-high-performance PostgreSQL parts sheet loader with advanced optimizations and robust error handling."""

    def __init__(self):
        self.request_id = set_request_id()
        self.db_config = DatabaseConfig()
        info_id("Initialized Super Optimized PostgreSQL Parts Sheet Loader", self.request_id)

        # Enhanced statistics tracking with detailed timing
        self.stats = {
            'total_rows_found': 0,
            'new_parts_added': 0,
            'duplicates_skipped': 0,
            'errors_encountered': 0,
            'associations_created': 0,
            'processing_time': 0,
            'excel_read_time': 0,
            'data_cleaning_time': 0,
            'existing_parts_fetch_time': 0,
            'dedup_time': 0,
            'insert_time': 0,
            'association_time': 0,
            'memory_peak_mb': 0,
            'batch_times': [],
            'chunk_processing_times': [],
            'rows_per_second_overall': 0,
            'rows_per_second_insert': 0,
            'average_batch_time': 0,
            'insert_batch_sizes': [],
            'insert_batch_times': []
        }

        # Optimized column mapping
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

        # Performance settings
        self.chunk_size = 10000  # Process data in chunks
        self.use_multiprocessing = mp.cpu_count() > 2
        self.max_workers = min(mp.cpu_count() - 1, 4)

    @contextmanager
    def postgresql_performance_mode(self, session):
        """Context manager to set optimal PostgreSQL settings for bulk operations."""
        try:
            # Ensure we start with a clean transaction
            try:
                session.rollback()  # Clear any pending transaction
            except:
                pass

            # Store original settings to restore later
            original_settings = {}

            # Get current settings with error handling
            settings_to_check = ['work_mem', 'maintenance_work_mem', 'synchronous_commit']

            for setting in settings_to_check:
                try:
                    result = session.execute(text(f"SHOW {setting}")).fetchone()
                    if result:
                        original_settings[setting] = result[0]
                        debug_id(f"Current {setting}: {result[0]}", self.request_id)
                except Exception as e:
                    debug_id(f"Could not get {setting}: {e}", self.request_id)

            # Set optimized settings for bulk operations (only essential ones)
            performance_settings = {
                'work_mem': '256MB',  # Increased for sorting/hashing
                'maintenance_work_mem': '512MB',  # Conservative setting for bulk operations
                'synchronous_commit': 'off',  # Faster commits (less durable)
            }

            info_id("Setting PostgreSQL performance mode", self.request_id)
            settings_applied = []

            for setting, value in performance_settings.items():
                try:
                    session.execute(text(f"SET {setting} = '{value}'"))
                    settings_applied.append(setting)
                    debug_id(f"Successfully set {setting} = {value}", self.request_id)
                except Exception as e:
                    debug_id(f"Could not set {setting}: {e}", self.request_id)

            # Only commit if we successfully applied some settings
            if settings_applied:
                session.commit()
                info_id(f"Applied {len(settings_applied)} performance settings", self.request_id)
            else:
                info_id("No performance settings applied, using defaults", self.request_id)

            yield session

        except Exception as e:
            error_id(f"Error in performance mode setup: {e}", self.request_id)
            yield session  # Continue even if performance mode fails

        finally:
            # Restore original settings safely
            try:
                info_id("Restoring PostgreSQL settings", self.request_id)

                # Re-enable synchronous commit first for safety
                session.execute(text("SET synchronous_commit = 'on'"))

                # Restore other settings if we have them
                for setting, value in original_settings.items():
                    if setting != 'synchronous_commit':  # Already handled above
                        try:
                            session.execute(text(f"SET {setting} = '{value}'"))
                            debug_id(f"Restored {setting} = {value}", self.request_id)
                        except Exception as e:
                            debug_id(f"Could not restore {setting}: {e}", self.request_id)

                session.commit()
                debug_id("PostgreSQL settings restored", self.request_id)

            except Exception as e:
                debug_id(f"Error restoring settings: {e}", self.request_id)

    def read_excel_optimized(self, file_path):
        """Ultra-fast Excel reading with memory optimization."""
        info_id("Reading Excel file with optimizations", self.request_id)

        start_time = time.time()

        try:
            # Use engine='openpyxl' for better performance with .xlsx files
            # Read only required columns to save memory
            required_cols = list(self.column_mapping.values())

            # First, peek at the file to determine optimal reading strategy
            sample_df = pd.read_excel(file_path, sheet_name="EQUIP_BOMS", nrows=100)
            total_rows_estimate = len(sample_df) * 100  # Rough estimate

            info_id(f"Estimated {total_rows_estimate:,} rows, reading optimally", self.request_id)

            if total_rows_estimate > 50000:
                # For large files, use chunked reading
                df_chunks = []
                chunk_size = 10000

                for chunk in pd.read_excel(file_path, sheet_name="EQUIP_BOMS",
                                           usecols=required_cols, chunksize=chunk_size,
                                           engine='openpyxl'):
                    df_chunks.append(chunk)

                df = pd.concat(df_chunks, ignore_index=True)
                info_id(f"Read large file in {len(df_chunks)} chunks", self.request_id)
            else:
                # For smaller files, read all at once with column selection
                df = pd.read_excel(file_path, sheet_name="EQUIP_BOMS",
                                   usecols=required_cols, engine='openpyxl')

            self.stats['excel_read_time'] = time.time() - start_time
            info_id(f"Excel read completed in {self.stats['excel_read_time']:.2f}s", self.request_id)

            return df

        except Exception as e:
            error_id(f"Error reading Excel file: {str(e)}", self.request_id)
            raise

    def get_existing_parts_streaming(self, session):
        """Memory-efficient streaming approach to get existing part numbers with timing."""
        info_id("Fetching existing part numbers with streaming", self.request_id)

        start_time = time.time()

        try:
            # Instead of loading all into memory, use a more efficient approach
            if self.db_config.is_postgresql:
                # Use raw connection for better performance
                conn = session.connection().connection
                cursor = conn.cursor()

                # Use server-side cursor for streaming large result sets
                cursor.execute(
                    "DECLARE part_numbers_cursor CURSOR FOR SELECT part_number FROM part WHERE part_number IS NOT NULL")

                existing_parts = set()
                batch_count = 0
                while True:
                    batch_start = time.time()
                    cursor.execute("FETCH 10000 FROM part_numbers_cursor")
                    rows = cursor.fetchall()
                    if not rows:
                        break
                    existing_parts.update(row[0] for row in rows)
                    batch_count += 1
                    batch_time = time.time() - batch_start
                    debug_id(f"Fetched batch {batch_count}: {len(rows):,} parts in {batch_time:.3f}s", self.request_id)

                cursor.execute("CLOSE part_numbers_cursor")
                cursor.close()

                fetch_time = time.time() - start_time
                self.stats['existing_parts_fetch_time'] = fetch_time

                info_id(f"Streamed {len(existing_parts):,} existing part numbers in {fetch_time:.3f}s", self.request_id)
                info_id(f"Fetch rate: {len(existing_parts) / fetch_time:,.0f} parts/second", self.request_id)
                return existing_parts
            else:
                # Fallback for non-PostgreSQL
                existing_parts = session.query(Part.part_number).filter(Part.part_number.isnot(None)).all()
                fetch_time = time.time() - start_time
                self.stats['existing_parts_fetch_time'] = fetch_time
                return {part[0] for part in existing_parts}

        except Exception as e:
            error_id(f"Error fetching existing part numbers: {str(e)}", self.request_id)
            raise

    def clean_data_vectorized_advanced(self, df):
        """Advanced vectorized data cleaning with detailed timing and robust NA handling."""
        info_id("Advanced vectorized data cleaning", self.request_id)

        start_time = time.time()
        original_rows = len(df)
        self.stats['total_rows_found'] = original_rows

        try:
            # Track sub-operations timing
            step_start = time.time()

            # More aggressive NA handling - convert ALL possible NA variants to None
            # This handles pandas.NA, numpy.nan, pd.NaType, etc.
            df = df.replace({pd.NA: None, np.nan: None, 'NaN': None, 'nan': None, 'NA': None, '': None})

            # Use object dtype for string columns to avoid pandas NA issues
            string_columns = list(self.column_mapping.values())
            for col in string_columns:
                if col in df.columns:
                    # Convert to object type first, then handle nulls
                    df[col] = df[col].astype('object')
                    # Replace any remaining pandas NA types
                    df[col] = df[col].where(pd.notna(df[col]), None)

            step1_time = time.time() - step_start
            debug_id(f"Data type conversion and NA handling: {step1_time:.3f}s", self.request_id)

            # Vectorized string cleaning - all at once
            step_start = time.time()
            for col in string_columns:
                if col in df.columns:
                    # Only apply string operations to non-null values
                    mask = df[col].notna()
                    if mask.any():
                        # Convert to string and clean only non-null values
                        df.loc[mask, col] = df.loc[mask, col].astype(str).str.strip()
                        # Replace empty strings and 'nan' strings with None
                        df.loc[df[col].isin(['', 'nan', 'NaN', 'NA']), col] = None

            step2_time = time.time() - step_start
            debug_id(f"String cleaning: {step2_time:.3f}s", self.request_id)

            # Ultra-fast filtering using boolean indexing
            step_start = time.time()
            critical_mask = True
            for field in ['ITEMNUM', 'DESCRIPTION']:
                if field in df.columns:
                    # More robust field validation
                    field_mask = (
                            df[field].notna() &
                            (df[field] != '') &
                            (df[field] != 'nan') &
                            (df[field] != 'NaN') &
                            (df[field] != 'NA') &
                            (df[field].astype(str).str.len() >= 3)
                    )
                    critical_mask = critical_mask & field_mask

            df_cleaned = df[critical_mask].copy()
            step3_time = time.time() - step_start
            debug_id(f"Data filtering: {step3_time:.3f}s", self.request_id)

            # Final aggressive NA cleanup - ensure no pandas NA types remain
            step_start = time.time()
            for col in df_cleaned.columns:
                # Convert any remaining pandas-specific NA types to None
                df_cleaned[col] = df_cleaned[col].where(pd.notna(df_cleaned[col]), None)
                # For object columns, ensure no pandas NA objects
                if df_cleaned[col].dtype == 'object':
                    df_cleaned[col] = df_cleaned[col].apply(lambda x: None if pd.isna(x) else x)

            step4_time = time.time() - step_start
            debug_id(f"Final NA cleanup: {step4_time:.3f}s", self.request_id)

            removed_count = original_rows - len(df_cleaned)
            if removed_count > 0:
                info_id(f"Removed {removed_count:,} invalid rows", self.request_id)

            # Reset index for better performance in subsequent operations
            df_cleaned.reset_index(drop=True, inplace=True)

            total_cleaning_time = time.time() - start_time
            self.stats['data_cleaning_time'] = total_cleaning_time

            # Verify no pandas NA types remain
            na_count = 0
            for col in df_cleaned.columns:
                na_count += df_cleaned[col].isna().sum()
            debug_id(f"Remaining NA values after cleaning: {na_count}", self.request_id)

            info_id(f"Advanced cleaning: {original_rows:,} -> {len(df_cleaned):,} rows in {total_cleaning_time:.3f}s",
                    self.request_id)
            info_id(f"Cleaning rate: {original_rows / total_cleaning_time:,.0f} rows/second", self.request_id)

            return df_cleaned

        except Exception as e:
            error_id(f"Error in advanced data cleaning: {str(e)}", self.request_id)
            raise

    def deduplicate_ultra_fast(self, df, existing_parts_set):
        """Ultra-fast deduplication using optimized pandas operations."""
        info_id("Ultra-fast deduplication process", self.request_id)

        start_time = time.time()

        try:
            initial_count = len(df)

            # 1. Remove internal duplicates - keep last occurrence
            df_no_internal_dupes = df.drop_duplicates(subset=['ITEMNUM'], keep='last')
            internal_dupes = initial_count - len(df_no_internal_dupes)

            # 2. Filter out existing parts using vectorized operations
            if existing_parts_set:
                # Convert to pandas Index for faster lookup
                existing_parts_index = pd.Index(existing_parts_set)

                # Use isin with Index for maximum performance
                new_parts_mask = ~df_no_internal_dupes['ITEMNUM'].isin(existing_parts_index)
                df_new = df_no_internal_dupes[new_parts_mask].copy()
                db_dupes = len(df_no_internal_dupes) - len(df_new)
            else:
                df_new = df_no_internal_dupes
                db_dupes = 0

            # Reset index for optimal performance
            df_new.reset_index(drop=True, inplace=True)

            self.stats['duplicates_skipped'] = internal_dupes + db_dupes
            self.stats['dedup_time'] = time.time() - start_time

            info_id(f"Deduplication completed in {self.stats['dedup_time']:.2f}s", self.request_id)
            info_id(f"Internal dupes: {internal_dupes:,}, DB dupes: {db_dupes:,}", self.request_id)

            return df_new

        except Exception as e:
            error_id(f"Error in ultra-fast deduplication: {str(e)}", self.request_id)
            raise

    def bulk_insert_turbo(self, session, df_new):
        """Turbo-charged bulk insert with detailed batch timing and robust data handling."""
        if df_new.empty:
            info_id("No new parts to insert", self.request_id)
            return []

        info_id(f"Turbo bulk inserting {len(df_new):,} parts", self.request_id)

        start_time = time.time()

        try:
            with self.postgresql_performance_mode(session):
                # Prepare data for PostgreSQL COPY or bulk insert
                columns = list(self.column_mapping.keys())

                prep_start = time.time()
                # Create DataFrame with correct column names
                insert_df = pd.DataFrame()
                for db_col, excel_col in self.column_mapping.items():
                    if excel_col in df_new.columns:
                        insert_df[db_col] = df_new[excel_col]
                    else:
                        insert_df[db_col] = None

                # CRITICAL: Final data sanitization before PostgreSQL insert
                # This is the safety net to catch any remaining pandas NA types
                for col in insert_df.columns:
                    # Replace ANY type of pandas NA with None
                    insert_df[col] = insert_df[col].where(pd.notna(insert_df[col]), None)

                    # Additional check for pandas NA objects that might slip through
                    if insert_df[col].dtype == 'object':
                        insert_df[col] = insert_df[col].apply(
                            lambda x: None if (
                                        pd.isna(x) or str(type(x)) == "<class 'pandas._libs.missing.NAType'>") else x
                        )

                # Verify no problematic data types remain
                for col in insert_df.columns:
                    sample_values = insert_df[col].dropna().head(3).tolist()
                    debug_id(f"Column {col} sample values: {sample_values}", self.request_id)

                    # Check for any pandas NA types
                    na_types = insert_df[col].apply(lambda x: str(type(x)) if pd.isna(x) else None).dropna().unique()
                    if len(na_types) > 0:
                        warning_id(f"Found NA types in {col}: {na_types}", self.request_id)
                        # Force convert to None
                        insert_df[col] = insert_df[col].apply(lambda x: None if pd.isna(x) else x)

                prep_time = time.time() - prep_start
                debug_id(f"Data preparation and sanitization: {prep_time:.3f}s", self.request_id)

                # Use the fastest possible PostgreSQL insert method
                if self.db_config.is_postgresql and len(insert_df) > 1000:
                    try:
                        # Use raw COPY for maximum speed with enhanced error handling
                        new_part_ids = self._copy_insert_postgresql_with_timing(session, insert_df, columns)
                    except Exception as copy_error:
                        warning_id(f"PostgreSQL COPY failed: {copy_error}", self.request_id)
                        info_id("Falling back to SQLAlchemy bulk insert", self.request_id)

                        # Fallback to SQLAlchemy method
                        insert_start = time.time()
                        parts_data = insert_df.to_dict('records')

                        # Additional safety check on the dictionary data
                        for i, part_data in enumerate(parts_data):
                            for key, value in part_data.items():
                                if pd.isna(value) or str(type(value)) == "<class 'pandas._libs.missing.NAType'>":
                                    parts_data[i][key] = None

                        session.bulk_insert_mappings(Part, parts_data)
                        session.commit()
                        insert_time = time.time() - insert_start

                        # Track batch metrics
                        self.stats['insert_batch_sizes'].append(len(parts_data))
                        self.stats['insert_batch_times'].append(insert_time)

                        debug_id(f"Fallback SQLAlchemy bulk insert: {len(parts_data):,} parts in {insert_time:.3f}s",
                                 self.request_id)

                        # Get IDs efficiently
                        id_start = time.time()
                        part_numbers = insert_df['part_number'].tolist()
                        new_parts = session.query(Part.id).filter(
                            Part.part_number.in_(part_numbers)
                        ).all()
                        new_part_ids = [p.id for p in new_parts]
                        id_time = time.time() - id_start
                        debug_id(f"Fallback ID retrieval: {len(new_part_ids):,} IDs in {id_time:.3f}s", self.request_id)
                else:
                    # Direct SQLAlchemy bulk insert for smaller datasets
                    insert_start = time.time()
                    parts_data = insert_df.to_dict('records')

                    # Additional safety check on the dictionary data
                    for i, part_data in enumerate(parts_data):
                        for key, value in part_data.items():
                            if pd.isna(value) or str(type(value)) == "<class 'pandas._libs.missing.NAType'>":
                                parts_data[i][key] = None

                    session.bulk_insert_mappings(Part, parts_data)
                    session.commit()
                    insert_time = time.time() - insert_start

                    # Track batch metrics
                    self.stats['insert_batch_sizes'].append(len(parts_data))
                    self.stats['insert_batch_times'].append(insert_time)

                    debug_id(f"SQLAlchemy bulk insert: {len(parts_data):,} parts in {insert_time:.3f}s",
                             self.request_id)

                    # Get IDs efficiently
                    id_start = time.time()
                    part_numbers = insert_df['part_number'].tolist()
                    new_parts = session.query(Part.id).filter(
                        Part.part_number.in_(part_numbers)
                    ).all()
                    new_part_ids = [p.id for p in new_parts]
                    id_time = time.time() - id_start
                    debug_id(f"ID retrieval: {len(new_part_ids):,} IDs in {id_time:.3f}s", self.request_id)

            total_insert_time = time.time() - start_time
            self.stats['insert_time'] = total_insert_time
            self.stats['new_parts_added'] = len(new_part_ids)
            self.stats['rows_per_second_insert'] = len(new_part_ids) / total_insert_time if total_insert_time > 0 else 0

            info_id(f"Turbo insert completed in {total_insert_time:.3f}s", self.request_id)
            info_id(f"Insert rate: {self.stats['rows_per_second_insert']:,.0f} parts/second", self.request_id)

            # Calculate average batch metrics
            if self.stats['insert_batch_times']:
                self.stats['average_batch_time'] = sum(self.stats['insert_batch_times']) / len(
                    self.stats['insert_batch_times'])
                avg_batch_size = sum(self.stats['insert_batch_sizes']) / len(self.stats['insert_batch_sizes'])
                info_id(f"Average batch: {avg_batch_size:.0f} parts in {self.stats['average_batch_time']:.3f}s",
                        self.request_id)

            return new_part_ids

        except Exception as e:
            session.rollback()
            error_id(f"Error in turbo bulk insert: {str(e)}", self.request_id)
            raise

    def _copy_insert_postgresql_with_timing(self, session, df, columns):
        """Use PostgreSQL COPY for maximum insert performance with enhanced error handling."""
        try:
            # Get raw psycopg2 connection
            connection = session.connection().connection
            cursor = connection.cursor()

            # Data preparation timing with enhanced sanitization
            prep_start = time.time()

            # Final data sanitization before creating tuples
            sanitized_df = df.copy()
            for col in columns:
                if col in sanitized_df.columns:
                    # Convert any pandas NA types to None
                    sanitized_df[col] = sanitized_df[col].apply(
                        lambda x: None if (pd.isna(x) or str(type(x)) == "<class 'pandas._libs.missing.NAType'>") else x
                    )

                    # Additional check for string representations of NA
                    if sanitized_df[col].dtype == 'object':
                        sanitized_df[col] = sanitized_df[col].apply(
                            lambda x: None if (x in ['NaN', 'nan', 'NA', '', '<NA>']) else x
                        )

            # Create tuples with explicit None conversion
            data_tuples = []
            for _, row in sanitized_df[columns].iterrows():
                tuple_data = []
                for value in row:
                    # Final safety check for each value
                    if pd.isna(value) or str(type(value)) == "<class 'pandas._libs.missing.NAType'>":
                        tuple_data.append(None)
                    elif value in ['NaN', 'nan', 'NA', '<NA>']:
                        tuple_data.append(None)
                    else:
                        tuple_data.append(value)
                data_tuples.append(tuple(tuple_data))

            prep_time = time.time() - prep_start
            debug_id(f"Enhanced COPY data preparation: {len(data_tuples):,} tuples in {prep_time:.3f}s",
                     self.request_id)

            # Sample the first few tuples to verify data integrity
            if data_tuples:
                sample_tuple = data_tuples[0]
                debug_id(f"Sample tuple: {sample_tuple}", self.request_id)

                # Check for any remaining problematic types
                problematic_types = []
                for i, value in enumerate(sample_tuple):
                    if value is not None and str(type(value)) == "<class 'pandas._libs.missing.NAType'>":
                        problematic_types.append(f"Column {i}: {type(value)}")

                if problematic_types:
                    error_id(f"Found problematic types in sample: {problematic_types}", self.request_id)
                    raise ValueError(f"Data contains pandas NAType values: {problematic_types}")

            # Track batch metrics for COPY
            self.stats['insert_batch_sizes'].append(len(data_tuples))

            # Batch insert timing with error handling
            insert_start = time.time()
            cols_str = ', '.join(f'"{col}"' for col in columns)
            sql = f'INSERT INTO part ({cols_str}) VALUES %s'

            try:
                # Use execute_values with moderate page size for better error handling
                from psycopg2.extras import execute_values
                execute_values(cursor, sql, data_tuples, page_size=1000)

                insert_time = time.time() - insert_start
                self.stats['insert_batch_times'].append(insert_time)
                debug_id(f"COPY bulk insert: {len(data_tuples):,} parts in {insert_time:.3f}s", self.request_id)
                debug_id(f"COPY insert rate: {len(data_tuples) / insert_time:,.0f} parts/second", self.request_id)

            except Exception as e:
                error_id(f"PostgreSQL execute_values failed: {e}", self.request_id)
                # Try to provide more detailed error information
                if "can't adapt type" in str(e):
                    error_id("Data type adaptation error - checking data types in batch", self.request_id)
                    for i, tuple_data in enumerate(data_tuples[:5]):  # Check first 5 tuples
                        for j, value in enumerate(tuple_data):
                            if value is not None:
                                debug_id(f"Tuple {i}, Column {j}: {type(value)} = {repr(value)}", self.request_id)
                raise

            # ID retrieval timing
            id_start = time.time()
            part_numbers = [row[columns.index('part_number')] for row in data_tuples if
                            row[columns.index('part_number')] is not None]

            if not part_numbers:
                warning_id("No valid part numbers found for ID retrieval", self.request_id)
                return []

            cursor.execute(
                'SELECT id FROM part WHERE part_number = ANY(%s)',
                (part_numbers,)
            )
            new_part_ids = [row[0] for row in cursor.fetchall()]
            id_time = time.time() - id_start
            debug_id(f"COPY ID retrieval: {len(new_part_ids):,} IDs in {id_time:.3f}s", self.request_id)

            connection.commit()

            total_copy_time = prep_time + insert_time + id_time
            info_id(f"Enhanced PostgreSQL COPY completed: {len(new_part_ids)} parts in {total_copy_time:.3f}s",
                    self.request_id)
            return new_part_ids

        except Exception as e:
            try:
                connection.rollback()
            except:
                pass
            error_id(f"Error in enhanced PostgreSQL COPY insert: {str(e)}", self.request_id)

            # Provide fallback suggestion
            info_id("Falling back to SQLAlchemy bulk insert method", self.request_id)
            raise

    def parallel_association_creation(self, session, new_part_ids):
        """Create associations using parallel processing with detailed timing."""
        if not new_part_ids or not self.use_multiprocessing:
            # Fallback to original method
            return self.create_part_image_associations_original(session, new_part_ids)

        info_id(f"Creating associations for {len(new_part_ids)} parts using {self.max_workers} workers",
                self.request_id)

        association_start = time.time()

        try:
            # Split part IDs into chunks for parallel processing
            chunk_size = max(100, len(new_part_ids) // self.max_workers)
            part_id_chunks = [new_part_ids[i:i + chunk_size]
                              for i in range(0, len(new_part_ids), chunk_size)]

            info_id(f"Split into {len(part_id_chunks)} chunks of ~{chunk_size} parts each", self.request_id)

            total_associations = 0
            chunk_results = []

            # Use ThreadPoolExecutor for I/O bound database operations
            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                # Create separate database sessions for each worker
                futures = []
                for i, chunk in enumerate(part_id_chunks):
                    future = executor.submit(self._create_associations_chunk_with_timing, chunk, i + 1)
                    futures.append(future)

                # Collect results with timing
                for i, future in enumerate(futures):
                    try:
                        chunk_result = future.result(timeout=300)  # 5 minute timeout
                        chunk_associations = chunk_result['associations']
                        chunk_time = chunk_result['time']
                        chunk_size_actual = chunk_result['chunk_size']

                        total_associations += chunk_associations
                        chunk_results.append(chunk_result)

                        rate = chunk_associations / chunk_time if chunk_time > 0 else 0
                        info_id(
                            f"Chunk {i + 1}: {chunk_associations} associations for {chunk_size_actual} parts in {chunk_time:.3f}s ({rate:.1f} assoc/s)",
                            self.request_id)

                    except Exception as e:
                        error_id(f"Error in parallel association chunk {i + 1}: {e}", self.request_id)

            total_association_time = time.time() - association_start
            self.stats['association_time'] = total_association_time
            self.stats['associations_created'] = total_associations
            self.stats['chunk_processing_times'] = [r['time'] for r in chunk_results]

            # Calculate statistics
            if chunk_results:
                avg_chunk_time = sum(r['time'] for r in chunk_results) / len(chunk_results)
                max_chunk_time = max(r['time'] for r in chunk_results)
                min_chunk_time = min(r['time'] for r in chunk_results)

                info_id(f"Parallel association completed: {total_associations} total in {total_association_time:.3f}s",
                        self.request_id)
                info_id(
                    f"Chunk timing - Avg: {avg_chunk_time:.3f}s, Min: {min_chunk_time:.3f}s, Max: {max_chunk_time:.3f}s",
                    self.request_id)

                if total_association_time > 0:
                    overall_rate = total_associations / total_association_time
                    info_id(f"Overall association rate: {overall_rate:.1f} associations/second", self.request_id)

        except Exception as e:
            error_id(f"Error in parallel association setup: {e}", self.request_id)
            # Fallback to original method
            return self.create_part_image_associations_original(session, new_part_ids)

    def _create_associations_chunk_with_timing(self, part_ids_chunk, chunk_number):
        """Worker method for creating associations in parallel with timing."""
        from modules.database_manager.db_manager import RelationshipManager

        chunk_start = time.time()

        try:
            # Create new database session for this worker
            with self.db_config.main_session() as worker_session:
                with RelationshipManager(session=worker_session, request_id=self.request_id) as manager:
                    result = manager.associate_parts_with_images_by_title(part_ids=part_ids_chunk)
                    manager.commit()

                    # Count total associations created
                    total_associations = sum(len(assocs) for assocs in result.values())

                    chunk_time = time.time() - chunk_start

                    debug_id(
                        f"Worker {chunk_number}: {total_associations} associations for {len(part_ids_chunk)} parts in {chunk_time:.3f}s",
                        self.request_id)

                    return {
                        'associations': total_associations,
                        'time': chunk_time,
                        'chunk_size': len(part_ids_chunk),
                        'chunk_number': chunk_number
                    }

        except Exception as e:
            chunk_time = time.time() - chunk_start
            error_id(f"Error in association worker {chunk_number}: {e}", self.request_id)
            return {
                'associations': 0,
                'time': chunk_time,
                'chunk_size': len(part_ids_chunk),
                'chunk_number': chunk_number,
                'error': str(e)
            }

    def create_part_image_associations_original(self, session, new_part_ids):
        """Original association creation method as fallback with timing."""
        try:
            from modules.database_manager.db_manager import RelationshipManager

            association_start = time.time()

            with RelationshipManager(session=session, request_id=self.request_id) as manager:
                result = manager.associate_parts_with_images_by_title(part_ids=new_part_ids)
                manager.commit()

                total_associations = sum(len(assocs) for assocs in result.values())
                association_time = time.time() - association_start

                self.stats['associations_created'] = total_associations
                self.stats['association_time'] = association_time

                if total_associations > 0:
                    rate = total_associations / association_time if association_time > 0 else 0
                    info_id(
                        f"Created {total_associations} part-image associations in {association_time:.3f}s ({rate:.1f} assoc/s)",
                        self.request_id)

        except Exception as e:
            error_id(f"Error in original association creation: {e}", self.request_id)

    def monitor_memory_usage(self):
        """Monitor memory usage during processing."""
        try:
            import psutil
            process = psutil.Process(os.getpid())
            memory_mb = process.memory_info().rss / 1024 / 1024
            if memory_mb > self.stats['memory_peak_mb']:
                self.stats['memory_peak_mb'] = memory_mb
                debug_id(f"Peak memory usage: {memory_mb:.1f} MB", self.request_id)
        except:
            pass

    def load_parts_from_excel_turbo(self, file_path=None):
        """Main turbo-charged method with comprehensive timing metrics."""
        try:
            info_id("Starting TURBO-OPTIMIZED parts sheet data import", self.request_id)

            # Determine file path
            if not file_path:
                load_sheet_filename = "load_MP2_ITEMS_BOMS.xlsx"
                file_path = os.path.join(DB_LOADSHEET, load_sheet_filename)

            info_id(f"Source file: {os.path.basename(file_path)}", self.request_id)

            overall_start_time = time.time()

            # Monitor memory usage
            self.monitor_memory_usage()

            # Get database session
            with self.db_config.main_session() as session:
                # 1. Optimized Excel reading
                step_start = time.time()
                df = self.read_excel_optimized(file_path)
                self.monitor_memory_usage()
                info_id(f"Step 1 complete: Excel reading took {time.time() - step_start:.3f}s", self.request_id)

                # 2. Advanced data cleaning
                step_start = time.time()
                df_cleaned = self.clean_data_vectorized_advanced(df)
                self.monitor_memory_usage()
                info_id(f"Step 2 complete: Data cleaning took {time.time() - step_start:.3f}s", self.request_id)

                # 3. Streaming existing parts lookup
                step_start = time.time()
                existing_parts = self.get_existing_parts_streaming(session)
                self.monitor_memory_usage()
                info_id(f"Step 3 complete: Existing parts fetch took {time.time() - step_start:.3f}s", self.request_id)

                # 4. Ultra-fast deduplication
                step_start = time.time()
                df_new = self.deduplicate_ultra_fast(df_cleaned, existing_parts)
                self.monitor_memory_usage()
                info_id(f"Step 4 complete: Deduplication took {time.time() - step_start:.3f}s", self.request_id)

                # 5. Turbo bulk insert
                step_start = time.time()
                new_part_ids = self.bulk_insert_turbo(session, df_new)
                self.monitor_memory_usage()
                info_id(f"Step 5 complete: Bulk insert took {time.time() - step_start:.3f}s", self.request_id)

                # 6. Parallel association creation
                step_start = time.time()
                self.parallel_association_creation(session, new_part_ids)
                self.monitor_memory_usage()
                info_id(f"Step 6 complete: Association creation took {time.time() - step_start:.3f}s", self.request_id)

            # Calculate final statistics
            total_processing_time = time.time() - overall_start_time
            self.stats['processing_time'] = total_processing_time

            # Calculate per-row metrics
            if self.stats['total_rows_found'] > 0:
                self.stats['rows_per_second_overall'] = self.stats['total_rows_found'] / total_processing_time

            # Display comprehensive summary
            self.display_turbo_summary()

            info_id("TURBO-OPTIMIZED parts import completed successfully", self.request_id)
            return True

        except Exception as e:
            error_id(f"Turbo-optimized parts import failed: {str(e)}", self.request_id, exc_info=True)
            return False

    def display_turbo_summary(self):
        """Display comprehensive performance summary with detailed timing metrics."""
        info_id("=" * 60, self.request_id)
        info_id("🚀 TURBO PARTS IMPORT PERFORMANCE SUMMARY 🚀", self.request_id)
        info_id("=" * 60, self.request_id)

        # Main Results
        info_id(f"📊 MAIN RESULTS:", self.request_id)
        info_id(f"   Total rows processed: {self.stats['total_rows_found']:,}", self.request_id)
        info_id(f"   New parts added: {self.stats['new_parts_added']:,}", self.request_id)
        info_id(f"   Duplicates skipped: {self.stats['duplicates_skipped']:,}", self.request_id)
        info_id(f"   Associations created: {self.stats['associations_created']:,}", self.request_id)
        info_id(f"   Peak memory usage: {self.stats['memory_peak_mb']:.1f} MB", self.request_id)

        info_id("", self.request_id)
        info_id("⏱️  DETAILED TIMING BREAKDOWN:", self.request_id)
        info_id(f"   Excel reading: {self.stats['excel_read_time']:.3f}s", self.request_id)
        info_id(f"   Data cleaning: {self.stats['data_cleaning_time']:.3f}s", self.request_id)
        info_id(f"   Existing parts fetch: {self.stats['existing_parts_fetch_time']:.3f}s", self.request_id)
        info_id(f"   Deduplication: {self.stats['dedup_time']:.3f}s", self.request_id)
        info_id(f"   Database insert: {self.stats['insert_time']:.3f}s", self.request_id)
        info_id(f"   Association creation: {self.stats['association_time']:.3f}s", self.request_id)
        info_id(f"   TOTAL PROCESSING: {self.stats['processing_time']:.3f}s", self.request_id)

        info_id("", self.request_id)
        info_id("📈 PERFORMANCE RATES:", self.request_id)

        # Overall performance rates
        if self.stats['processing_time'] > 0:
            overall_rate = self.stats['total_rows_found'] / self.stats['processing_time']
            info_id(f"   Overall processing: {overall_rate:,.0f} rows/second", self.request_id)

        # Excel reading rate
        if self.stats['excel_read_time'] > 0:
            excel_rate = self.stats['total_rows_found'] / self.stats['excel_read_time']
            info_id(f"   Excel reading: {excel_rate:,.0f} rows/second", self.request_id)

        # Data cleaning rate
        if self.stats['data_cleaning_time'] > 0:
            cleaning_rate = self.stats['total_rows_found'] / self.stats['data_cleaning_time']
            info_id(f"   Data cleaning: {cleaning_rate:,.0f} rows/second", self.request_id)

        # Insert performance
        if self.stats['rows_per_second_insert'] > 0:
            info_id(f"   Database insert: {self.stats['rows_per_second_insert']:,.0f} parts/second", self.request_id)

        # Association performance
        if self.stats['association_time'] > 0 and self.stats['associations_created'] > 0:
            assoc_rate = self.stats['associations_created'] / self.stats['association_time']
            info_id(f"   Association creation: {assoc_rate:,.0f} associations/second", self.request_id)

        # Batch performance details
        if self.stats['insert_batch_times']:
            info_id("", self.request_id)
            info_id("🔄 BATCH PROCESSING DETAILS:", self.request_id)
            info_id(f"   Number of insert batches: {len(self.stats['insert_batch_times'])}", self.request_id)
            info_id(f"   Average batch time: {self.stats['average_batch_time']:.3f}s", self.request_id)

            if self.stats['insert_batch_sizes']:
                avg_batch_size = sum(self.stats['insert_batch_sizes']) / len(self.stats['insert_batch_sizes'])
                min_batch_size = min(self.stats['insert_batch_sizes'])
                max_batch_size = max(self.stats['insert_batch_sizes'])
                info_id(f"   Average batch size: {avg_batch_size:.0f} parts", self.request_id)
                info_id(f"   Batch size range: {min_batch_size} - {max_batch_size} parts", self.request_id)

                # Calculate per-row timing for batches
                total_batch_parts = sum(self.stats['insert_batch_sizes'])
                total_batch_time = sum(self.stats['insert_batch_times'])
                if total_batch_time > 0:
                    avg_time_per_part = total_batch_time / total_batch_parts
                    info_id(f"   Average time per part: {avg_time_per_part * 1000:.2f} milliseconds", self.request_id)

        # Parallel processing details
        if self.stats['chunk_processing_times']:
            info_id("", self.request_id)
            info_id("⚡ PARALLEL PROCESSING DETAILS:", self.request_id)
            info_id(f"   Number of parallel chunks: {len(self.stats['chunk_processing_times'])}", self.request_id)

            avg_chunk_time = sum(self.stats['chunk_processing_times']) / len(self.stats['chunk_processing_times'])
            min_chunk_time = min(self.stats['chunk_processing_times'])
            max_chunk_time = max(self.stats['chunk_processing_times'])

            info_id(f"   Average chunk time: {avg_chunk_time:.3f}s", self.request_id)
            info_id(f"   Chunk time range: {min_chunk_time:.3f}s - {max_chunk_time:.3f}s", self.request_id)

            # Parallel efficiency
            total_sequential_time = sum(self.stats['chunk_processing_times'])
            parallel_efficiency = (total_sequential_time / self.stats['association_time']) * 100 if self.stats[
                                                                                                        'association_time'] > 0 else 0
            info_id(f"   Parallel efficiency: {parallel_efficiency:.1f}%", self.request_id)

        # Time distribution analysis
        info_id("", self.request_id)
        info_id("📊 TIME DISTRIBUTION:", self.request_id)

        total_time = self.stats['processing_time']
        if total_time > 0:
            excel_pct = (self.stats['excel_read_time'] / total_time) * 100
            cleaning_pct = (self.stats['data_cleaning_time'] / total_time) * 100
            fetch_pct = (self.stats['existing_parts_fetch_time'] / total_time) * 100
            dedup_pct = (self.stats['dedup_time'] / total_time) * 100
            insert_pct = (self.stats['insert_time'] / total_time) * 100
            assoc_pct = (self.stats['association_time'] / total_time) * 100

            info_id(f"   Excel reading: {excel_pct:.1f}%", self.request_id)
            info_id(f"   Data cleaning: {cleaning_pct:.1f}%", self.request_id)
            info_id(f"   Existing parts fetch: {fetch_pct:.1f}%", self.request_id)
            info_id(f"   Deduplication: {dedup_pct:.1f}%", self.request_id)
            info_id(f"   Database insert: {insert_pct:.1f}%", self.request_id)
            info_id(f"   Association creation: {assoc_pct:.1f}%", self.request_id)

        # Performance recommendations
        info_id("", self.request_id)
        info_id("💡 PERFORMANCE INSIGHTS:", self.request_id)

        if total_time > 0:
            bottleneck_times = {
                'Excel reading': self.stats['excel_read_time'],
                'Data cleaning': self.stats['data_cleaning_time'],
                'Existing parts fetch': self.stats['existing_parts_fetch_time'],
                'Deduplication': self.stats['dedup_time'],
                'Database insert': self.stats['insert_time'],
                'Association creation': self.stats['association_time']
            }

            # Find the biggest bottleneck
            bottleneck = max(bottleneck_times.items(), key=lambda x: x[1])
            info_id(f"   Biggest bottleneck: {bottleneck[0]} ({bottleneck[1]:.3f}s)", self.request_id)

            # Provide specific recommendations
            if bottleneck[0] == 'Excel reading':
                info_id("   💡 Consider using CSV format or smaller Excel chunks", self.request_id)
            elif bottleneck[0] == 'Database insert':
                info_id("   💡 Consider tuning PostgreSQL bulk insert settings", self.request_id)
            elif bottleneck[0] == 'Association creation':
                info_id("   💡 Consider optimizing image association queries", self.request_id)
            elif bottleneck[0] == 'Existing parts fetch':
                info_id("   💡 Consider adding database indexes on part_number", self.request_id)

        info_id("=" * 60, self.request_id)


def main():
    """Main function using the turbo-optimized loader."""
    info_id("Starting TURBO-OPTIMIZED parts sheet import", request_id=None)

    loader = None
    try:
        # Initialize the turbo-optimized loader
        loader = SuperOptimizedPostgreSQLPartsSheetLoader()

        # Load the parts data with all optimizations
        success = loader.load_parts_from_excel_turbo()

        if success:
            info_id("TURBO-OPTIMIZED parts sheet import completed successfully", loader.request_id)
        else:
            warning_id("TURBO-OPTIMIZED parts sheet import completed with issues", loader.request_id)

    except KeyboardInterrupt:
        error_id("Import interrupted by user", loader.request_id if loader else None)
    except Exception as e:
        error_id(f"Import failed: {str(e)}", loader.request_id if loader else None, exc_info=True)


if __name__ == "__main__":
    main()