# modules/maintenance/optimized_db_maintenance.py
import sys
import os
import csv
import click
import time
import pandas as pd
import numpy as np
from datetime import datetime
from sqlalchemy import and_, text, func
from collections import defaultdict
# Add to imports at the top of optimized_db_maintenance.py
from modules.emtacdb.emtacdb_fts import ImageEmbedding

# Add the project root directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..')))

# Define default log directory (relative to this script)
DEFAULT_LOG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "db_maint_logs")
os.makedirs(DEFAULT_LOG_DIR, exist_ok=True)

from modules.configuration.config_env import DatabaseConfig
from modules.database_manager.db_manager import RelationshipManager, DuplicateManager
from modules.initial_setup.initializer_logger import initializer_logger
from modules.configuration.log_config import debug_id, info_id, error_id, get_request_id, log_timed_operation, \
    warning_id
from modules.emtacdb.emtacdb_fts import Part, Image, Drawing, PartsPositionImageAssociation, DrawingPartAssociation


def get_app_root():
    """Get the application root directory."""
    return os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))


class OptimizedDatabaseMaintenance:
    """High-performance database maintenance operations using vectorized processing."""

    def __init__(self):
        self.request_id = get_request_id()
        self.db_config = DatabaseConfig()
        info_id("Initialized Optimized Database Maintenance", self.request_id)

    def validate_image_embeddings_vectorized(self, export_report=True, report_dir=None):
        """
        Validate that all images have embeddings and all embeddings map to images.
        Uses vectorized operations with pandas for performance.

        Returns:
            Dictionary with validation results and statistics
        """
        info_id("Starting optimized image-embedding validation", self.request_id)
        start_time = time.time()

        result = {
            'stats': {
                'total_images': 0,
                'total_embeddings': 0,
                'images_missing_embeddings': 0,
                'orphan_embeddings': 0,
                'duration_seconds': 0
            },
            'images_missing_embeddings': [],
            'orphan_embeddings': []
        }

        try:
            with self.db_config.main_session() as session:
                # Load images
                images_df = pd.read_sql(
                    text("SELECT id, title, file_path FROM image ORDER BY id"),
                    session.bind
                )
                # Load embeddings
                embeddings_df = pd.read_sql(
                    text("SELECT id, image_id FROM image_embedding ORDER BY id"),
                    session.bind
                )

                result['stats']['total_images'] = len(images_df)
                result['stats']['total_embeddings'] = len(embeddings_df)

                # Find images missing embeddings
                if not images_df.empty:
                    emb_img_ids = set(embeddings_df['image_id'].dropna().tolist())
                    missing = images_df[~images_df['id'].isin(emb_img_ids)]
                    result['stats']['images_missing_embeddings'] = len(missing)
                    result['images_missing_embeddings'] = missing.to_dict('records')

                # Find orphan embeddings (embedding without valid image)
                if not embeddings_df.empty:
                    img_ids = set(images_df['id'].tolist())
                    orphan = embeddings_df[~embeddings_df['image_id'].isin(img_ids)]
                    result['stats']['orphan_embeddings'] = len(orphan)
                    result['orphan_embeddings'] = orphan.to_dict('records')

            # Duration
            result['stats']['duration_seconds'] = time.time() - start_time
            info_id(f"Image-embedding validation complete in {result['stats']['duration_seconds']:.2f}s", self.request_id)

            # Export report
            if export_report:
                report_files = self.export_image_embedding_report_optimized(result, report_dir)
                result['report_files'] = report_files

            return result

        except Exception as e:
            error_id(f"Error in optimized image-embedding validation: {str(e)}", self.request_id, exc_info=True)
            result['error'] = str(e)
            return result


    def export_image_embedding_report_optimized(self, result, report_dir=None):
        """Export validation results for image embeddings."""
        if not report_dir:
            report_dir = DEFAULT_LOG_DIR
        os.makedirs(report_dir, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_files = {}

        try:
            # Summary
            summary_file = os.path.join(report_dir, f"image_embedding_summary_{timestamp}.csv")
            pd.DataFrame([result['stats']]).to_csv(summary_file, index=False)
            report_files['summary'] = summary_file

            # Missing embeddings
            if result['images_missing_embeddings']:
                missing_file = os.path.join(report_dir, f"images_missing_embeddings_{timestamp}.csv")
                pd.DataFrame(result['images_missing_embeddings']).to_csv(missing_file, index=False)
                report_files['missing'] = missing_file

            # Orphan embeddings
            if result['orphan_embeddings']:
                orphan_file = os.path.join(report_dir, f"orphan_embeddings_{timestamp}.csv")
                pd.DataFrame(result['orphan_embeddings']).to_csv(orphan_file, index=False)
                report_files['orphan'] = orphan_file

            info_id(f"Exported image embedding validation reports to {report_dir}", self.request_id)
            return report_files

        except Exception as e:
            error_id(f"Error exporting image embedding report: {str(e)}", self.request_id)
            return {}


    def associate_all_parts_with_images_vectorized(self, export_report=True, report_dir=None):
        """
        Associate all parts with matching images using vectorized operations - MUCH faster!

        Returns:
            Dictionary containing association results and statistics
        """
        info_id("Starting optimized part-image association", self.request_id)
        start_time = time.time()

        result = {
            'stats': {
                'total_parts_processed': 0,
                'parts_with_matching_images': 0,
                'total_associations_created': 0,
                'parts_with_no_matches': 0,
                'duration_seconds': 0
            },
            'associations': {},
            'unmatched_parts': []
        }

        try:
            with self.db_config.main_session() as session:
                # Load ALL data at once using pandas - much faster than individual queries
                with log_timed_operation("load_parts_and_images_data", self.request_id):
                    # Get all parts data
                    parts_query = text("""
                        SELECT id, part_number, name 
                        FROM part 
                        WHERE part_number IS NOT NULL
                        ORDER BY id
                    """)
                    parts_df = pd.read_sql(parts_query, session.bind)

                    # Get all images data
                    images_query = text("""
                        SELECT id, title, file_path 
                        FROM image 
                        WHERE title IS NOT NULL
                        ORDER BY id
                    """)
                    images_df = pd.read_sql(images_query, session.bind)

                    # Get existing associations to avoid duplicates (with error handling)
                    try:
                        existing_assocs_query = text("""
                            SELECT part_id, image_id
                            FROM part_position_image
                        """)
                        existing_assocs_df = pd.read_sql(existing_assocs_query, session.bind)
                    except Exception as e:
                        warning_id(
                            f"Association table not found or empty, proceeding without existing associations: {str(e)}",
                            self.request_id)
                        existing_assocs_df = pd.DataFrame(columns=['part_id', 'image_id'])

                    info_id(
                        f"Loaded {len(parts_df)} parts, {len(images_df)} images, {len(existing_assocs_df)} existing associations",
                        self.request_id)

                # Create vectorized matching using pandas operations
                with log_timed_operation("vectorized_matching", self.request_id):
                    new_associations = []
                    matched_parts = set()

                    # Check if we have images to work with
                    if images_df.empty:
                        info_id("No images found in database - skipping part-image associations", self.request_id)
                        result['stats']['parts_with_no_matches'] = len(parts_df)
                    else:
                        # Create existing associations set for fast lookup
                        existing_pairs = set()
                        if not existing_assocs_df.empty:
                            existing_pairs = set(zip(existing_assocs_df['part_id'], existing_assocs_df['image_id']))

                        # Vectorized string matching - much faster than nested loops
                        for _, part in parts_df.iterrows():
                            part_number = str(part['part_number']).strip().lower()

                            # Use vectorized string operations to find matches
                            image_matches = images_df[
                                images_df['title'].str.lower().str.contains(part_number, na=False, regex=False)
                            ]

                            if not image_matches.empty:
                                matched_parts.add(part['id'])

                                # Create associations for all matching images
                                for _, image in image_matches.iterrows():
                                    # Skip if association already exists
                                    if (part['id'], image['id']) not in existing_pairs:
                                        new_associations.append({
                                            'part_id': part['id'],
                                            'image_id': image['id'],
                                            'position_id': None,  # Optional field
                                            'created_by': 'optimized_maintenance',
                                            'association_type': 'automatic_by_title'
                                        })

                # Bulk insert new associations - single database operation!
                associations_created = 0
                if new_associations:
                    with log_timed_operation("bulk_insert_associations", self.request_id):
                        session.bulk_insert_mappings(PartsPositionImageAssociation, new_associations)
                        session.commit()
                        associations_created = len(new_associations)
                        info_id(f"Bulk inserted {associations_created} new associations", self.request_id)

                # Build results using vectorized operations
                with log_timed_operation("build_results", self.request_id):
                    result['stats']['total_parts_processed'] = len(parts_df)
                    result['stats']['parts_with_matching_images'] = len(matched_parts)
                    result['stats']['total_associations_created'] = associations_created
                    result['stats']['parts_with_no_matches'] = len(parts_df) - len(matched_parts)

                    # Build associations dictionary efficiently
                    if export_report:
                        # Get updated associations data for reporting
                        all_assocs_query = text("""
                            SELECT p.part_number, p.name as part_name, p.id as part_id,
                                   i.id as image_id, i.title, i.file_path,
                                   a.id as association_id
                            FROM part p
                            JOIN parts_position_image_association a ON p.id = a.part_id
                            JOIN image i ON a.image_id = i.id
                            WHERE p.id = ANY(:part_ids)
                        """)

                        if matched_parts:
                            assocs_df = pd.read_sql(all_assocs_query, session.bind,
                                                    params={'part_ids': list(matched_parts)})

                            # Group by part_number efficiently
                            for part_number, group in assocs_df.groupby('part_number'):
                                images = []
                                for _, row in group.iterrows():
                                    images.append({
                                        'image_id': row['image_id'],
                                        'image_title': row['title'],
                                        'image_file_path': row['file_path'],
                                        'association_id': row['association_id']
                                    })

                                result['associations'][part_number] = {
                                    'part_id': group.iloc[0]['part_id'],
                                    'part_name': group.iloc[0]['part_name'],
                                    'images': images
                                }

                        # Get unmatched parts efficiently
                        unmatched_mask = ~parts_df['id'].isin(matched_parts)
                        unmatched_parts_df = parts_df[unmatched_mask]

                        result['unmatched_parts'] = unmatched_parts_df[
                            ['id', 'part_number', 'name']
                        ].rename(columns={
                            'id': 'part_id',
                            'name': 'part_name'
                        }).to_dict('records')

                # Calculate duration
                result['stats']['duration_seconds'] = time.time() - start_time

                # Log results
                info_id(f"Optimized part-image association completed in {result['stats']['duration_seconds']:.2f}s",
                        self.request_id)
                info_id(f"Created {associations_created} new associations from {len(parts_df)} parts", self.request_id)

                # Export report if requested
                if export_report:
                    report_files = self.export_association_report_optimized(result, report_dir)
                    result['report_files'] = report_files

                return result

        except Exception as e:
            error_id(f"Error in optimized part-image association: {str(e)}", self.request_id, exc_info=True)
            result['error'] = str(e)
            return result

    def associate_all_drawings_with_parts_vectorized(self, export_report=True, report_dir=None):
        """
        Associate all drawings with parts using vectorized operations - MUCH faster!

        Returns:
            Dictionary containing association results and statistics
        """
        info_id("Starting optimized drawing-part association", self.request_id)
        start_time = time.time()

        result = {
            'stats': {
                'total_drawings_processed': 0,
                'drawings_with_matching_parts': 0,
                'total_associations_created': 0,
                'drawings_with_no_matches': 0,
                'drawings_without_spare_part_number': 0,
                'multiple_part_numbers_count': 0,
                'duration_seconds': 0
            },
            'associations': {},
            'unmatched_drawings': []
        }

        try:
            with self.db_config.main_session() as session:
                # Load ALL data at once - much faster
                with log_timed_operation("load_drawings_and_parts_data", self.request_id):
                    # Get all drawings with spare part numbers
                    drawings_query = text("""
                        SELECT id, drw_number, drw_name, drw_spare_part_number
                        FROM drawing 
                        WHERE drw_spare_part_number IS NOT NULL 
                        AND drw_spare_part_number != ''
                        ORDER BY id
                    """)
                    drawings_df = pd.read_sql(drawings_query, session.bind)

                    # Get all parts data
                    parts_query = text("""
                        SELECT id, part_number, name
                        FROM part
                        WHERE part_number IS NOT NULL
                        ORDER BY part_number
                    """)
                    parts_df = pd.read_sql(parts_query, session.bind)

                    # Get existing associations (with error handling)
                    try:
                        existing_assocs_query = text("""
                            SELECT drawing_id, part_id
                            FROM drawing_part
                        """)
                        existing_assocs_df = pd.read_sql(existing_assocs_query, session.bind)
                    except Exception as e:
                        warning_id(
                            f"Drawing-part association table not found or empty, proceeding without existing associations: {str(e)}",
                            self.request_id)
                        existing_assocs_df = pd.DataFrame(columns=['drawing_id', 'part_id'])

                    info_id(
                        f"Loaded {len(drawings_df)} drawings, {len(parts_df)} parts, {len(existing_assocs_df)} existing associations",
                        self.request_id)

                # Create part lookup dictionary for fast matching
                part_lookup = parts_df.set_index('part_number')['id'].to_dict()
                part_names_lookup = parts_df.set_index('part_number')['name'].to_dict()
                existing_pairs = set()
                if not existing_assocs_df.empty:
                    existing_pairs = set(zip(existing_assocs_df['drawing_id'], existing_assocs_df['part_id']))

                # Vectorized processing of drawings
                with log_timed_operation("vectorized_drawing_processing", self.request_id):
                    new_associations = []
                    matched_drawings = set()

                    # Process all drawings with vectorized operations
                    for _, drawing in drawings_df.iterrows():
                        spare_part_str = str(drawing['drw_spare_part_number']).strip()

                        if not spare_part_str:
                            result['stats']['drawings_without_spare_part_number'] += 1
                            continue

                        # Split and clean part numbers
                        part_numbers = [pn.strip() for pn in spare_part_str.split(',') if pn.strip()]

                        if len(part_numbers) > 1:
                            result['stats']['multiple_part_numbers_count'] += 1

                        drawing_matched = False

                        # Match each part number
                        for part_number in part_numbers:
                            if part_number in part_lookup:
                                part_id = part_lookup[part_number]

                                # Skip if association already exists
                                if (drawing['id'], part_id) not in existing_pairs:
                                    new_associations.append({
                                        'drawing_id': drawing['id'],
                                        'part_id': part_id,
                                        'created_by': 'optimized_maintenance',
                                        'association_type': 'automatic_by_number'
                                    })
                                    drawing_matched = True

                        if drawing_matched:
                            matched_drawings.add(drawing['id'])

                # Bulk insert new associations
                associations_created = 0
                if new_associations:
                    with log_timed_operation("bulk_insert_drawing_associations", self.request_id):
                        session.bulk_insert_mappings(DrawingPartAssociation, new_associations)
                        session.commit()
                        associations_created = len(new_associations)
                        info_id(f"Bulk inserted {associations_created} new drawing-part associations", self.request_id)

                # Build results efficiently
                with log_timed_operation("build_drawing_results", self.request_id):
                    result['stats']['total_drawings_processed'] = len(drawings_df)
                    result['stats']['drawings_with_matching_parts'] = len(matched_drawings)
                    result['stats']['total_associations_created'] = associations_created
                    result['stats']['drawings_with_no_matches'] = len(drawings_df) - len(matched_drawings)

                    # Build detailed results for reporting
                    if export_report:
                        # Get all associations for matched drawings
                        if matched_drawings:
                            assocs_query = text("""
                                SELECT d.drw_number, d.drw_name, d.drw_spare_part_number, d.id as drawing_id,
                                       p.id as part_id, p.part_number, p.name as part_name,
                                       a.id as association_id
                                FROM drawing d
                                JOIN drawing_part a ON d.id = a.drawing_id
                                JOIN part p ON a.part_id = p.id
                                WHERE d.id = ANY(:drawing_ids)
                            """)

                            assocs_df = pd.read_sql(assocs_query, session.bind,
                                                    params={'drawing_ids': list(matched_drawings)})

                            # Group by drawing number efficiently
                            for drw_number, group in assocs_df.groupby('drw_number'):
                                parts = []
                                for _, row in group.iterrows():
                                    parts.append({
                                        'part_id': row['part_id'],
                                        'part_number': row['part_number'],
                                        'part_name': row['part_name'],
                                        'association_id': row['association_id']
                                    })

                                result['associations'][drw_number] = {
                                    'drawing_id': group.iloc[0]['drawing_id'],
                                    'drawing_name': group.iloc[0]['drw_name'],
                                    'spare_part_number': group.iloc[0]['drw_spare_part_number'],
                                    'parts': parts
                                }

                        # Get unmatched drawings
                        unmatched_mask = ~drawings_df['id'].isin(matched_drawings)
                        unmatched_drawings_df = drawings_df[unmatched_mask]

                        for _, drawing in unmatched_drawings_df.iterrows():
                            spare_part_str = str(drawing['drw_spare_part_number']).strip()
                            part_numbers = [pn.strip() for pn in spare_part_str.split(',') if pn.strip()]

                            result['unmatched_drawings'].append({
                                'drawing_id': drawing['id'],
                                'drawing_number': drawing['drw_number'],
                                'drawing_name': drawing['drw_name'],
                                'spare_part_number': drawing['drw_spare_part_number'],
                                'parsed_part_numbers': part_numbers
                            })

                # Calculate duration
                result['stats']['duration_seconds'] = time.time() - start_time

                # Log results
                info_id(f"Optimized drawing-part association completed in {result['stats']['duration_seconds']:.2f}s",
                        self.request_id)
                info_id(f"Created {associations_created} new associations from {len(drawings_df)} drawings",
                        self.request_id)

                # Export report if requested
                if export_report:
                    report_files = self.export_drawing_part_association_report_optimized(result, report_dir)
                    result['report_files'] = report_files

                return result

        except Exception as e:
            error_id(f"Error in optimized drawing-part association: {str(e)}", self.request_id, exc_info=True)
            result['error'] = str(e)
            return result

    def export_association_report_optimized(self, result, report_dir=None):
        """Export association results using optimized pandas operations."""
        if not report_dir:
            report_dir = DEFAULT_LOG_DIR
        os.makedirs(report_dir, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_files = {}

        try:
            # Create summary report
            summary_file = os.path.join(report_dir, f"optimized_part_image_summary_{timestamp}.csv")
            summary_data = pd.DataFrame([{
                'timestamp': datetime.now().isoformat(),
                'total_parts_processed': result['stats']['total_parts_processed'],
                'parts_with_matches': result['stats']['parts_with_matching_images'],
                'parts_without_matches': result['stats']['parts_with_no_matches'],
                'total_associations': result['stats']['total_associations_created'],
                'duration_seconds': result['stats']['duration_seconds']
            }])
            summary_data.to_csv(summary_file, index=False)
            report_files['summary'] = summary_file

            # Create detailed associations report
            if result['associations']:
                details_file = os.path.join(report_dir, f"optimized_part_image_details_{timestamp}.csv")
                details_rows = []

                for part_number, part_info in result['associations'].items():
                    for image in part_info['images']:
                        details_rows.append({
                            'part_id': part_info['part_id'],
                            'part_number': part_number,
                            'part_name': part_info['part_name'],
                            'image_id': image['image_id'],
                            'image_title': image['image_title'],
                            'image_file_path': image['image_file_path'],
                            'association_id': image['association_id']
                        })

                details_df = pd.DataFrame(details_rows)
                details_df.to_csv(details_file, index=False)
                report_files['details'] = details_file

            # Create unmatched parts report
            if result['unmatched_parts']:
                unmatched_file = os.path.join(report_dir, f"optimized_unmatched_parts_{timestamp}.csv")
                unmatched_df = pd.DataFrame(result['unmatched_parts'])
                unmatched_df.to_csv(unmatched_file, index=False)
                report_files['unmatched'] = unmatched_file

            info_id(f"Exported optimized association reports to {report_dir}", self.request_id)
            return report_files

        except Exception as e:
            error_id(f"Error exporting optimized association report: {str(e)}", self.request_id)
            return {}

    def export_drawing_part_association_report_optimized(self, result, report_dir=None):
        """Export drawing-part association results using optimized operations."""
        if not report_dir:
            report_dir = DEFAULT_LOG_DIR
        os.makedirs(report_dir, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_files = {}

        try:
            # Create matched drawings report using pandas
            if result['associations']:
                matched_file = os.path.join(report_dir, f"optimized_drawing_part_matches_{timestamp}.csv")
                matched_rows = []

                for drawing_number, data in result['associations'].items():
                    for part in data['parts']:
                        matched_rows.append({
                            'drawing_number': drawing_number,
                            'drawing_name': data['drawing_name'],
                            'drawing_spare_part_numbers': data['spare_part_number'],
                            'matched_part_number': part['part_number'],
                            'part_name': part['part_name'],
                            'association_id': part['association_id']
                        })

                matched_df = pd.DataFrame(matched_rows)
                matched_df.to_csv(matched_file, index=False)
                report_files['matched'] = matched_file

            # Create unmatched drawings report
            if result['unmatched_drawings']:
                unmatched_file = os.path.join(report_dir, f"optimized_drawing_part_unmatched_{timestamp}.csv")
                unmatched_df = pd.DataFrame(result['unmatched_drawings'])
                unmatched_df.to_csv(unmatched_file, index=False)
                report_files['unmatched'] = unmatched_file

            # Create summary report
            summary_file = os.path.join(report_dir, f"optimized_drawing_part_summary_{timestamp}.txt")
            with open(summary_file, 'w') as f:
                f.write("Optimized Drawing-Part Association Summary\n")
                f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
                for key, value in result['stats'].items():
                    f.write(f"{key.replace('_', ' ').title()}: {value}\n")

            report_files['summary'] = summary_file
            info_id(f"Exported optimized drawing-part reports to {report_dir}", self.request_id)
            return report_files

        except Exception as e:
            error_id(f"Error exporting optimized drawing-part report: {str(e)}", self.request_id)
            return {}


# Initialize the optimized maintenance instance
maintenance = OptimizedDatabaseMaintenance()


# CLI interface with optimized functions
@click.group()
def cli():
    """Optimized database maintenance utilities."""
    pass

@cli.command()
@click.option('--export-report/--no-export-report', default=False, hidden=True)
@click.option('--report-dir', default=None, hidden=True)
@click.option('--batch-size', default=100, help='Commit every N images')
def add_image_embeddings(export_report, report_dir, batch_size):
    """
    Generate embeddings for all images that do not yet have them.
    """
    from modules.emtacdb.emtacdb_fts import Image, ImageEmbedding
    from modules.configuration.config_env import DatabaseConfig
    from plugins.ai_modules.ai_models import ModelsConfig
    from modules.configuration.log_config import info_id, warning_id, error_id, get_request_id

    rid = get_request_id()
    db_config = DatabaseConfig()

    try:
        with db_config.get_main_session() as session:
            model_handler = ModelsConfig.load_image_model()
            if not model_handler:
                error_id("No image embedding model configured!", rid)
                return

            # Find images with no embeddings
            images = session.query(Image).outerjoin(ImageEmbedding).filter(ImageEmbedding.id == None).all()
            total = len(images)
            info_id(f"Found {total} images missing embeddings", rid)

            processed, failed = 0, 0
            for i, image in enumerate(images, start=1):
                try:
                    ok = image.generate_embedding(session, model_handler, request_id=rid)
                    if ok:
                        processed += 1
                    else:
                        failed += 1
                except Exception as e:
                    error_id(f"Failed to embed image {image.id}: {e}", rid)
                    failed += 1

                if i % batch_size == 0:
                    session.commit()
                    info_id(f"Committed {i}/{total} images", rid)

            session.commit()
            info_id(f"Embedding run finished: {processed} added, {failed} failed, out of {total}", rid)

    except Exception as e:
        error_id(f"Critical failure in add_image_embeddings: {e}", rid, exc_info=True)


@cli.command()
@click.option('--export-report/--no-export-report', default=True, help='Export detailed report to CSV')
@click.option('--report-dir', default=None, help='Directory to save the report')
def associate_images_fast(export_report, report_dir):
    """Associate all parts with matching images using optimized vectorized operations."""
    result = maintenance.associate_all_parts_with_images_vectorized(export_report, report_dir)

    click.echo("\nOPTIMIZED Part-Image Association Summary:")
    click.echo(f"- Parts processed: {result['stats']['total_parts_processed']:,}")
    click.echo(f"- Parts with matching images: {result['stats']['parts_with_matching_images']:,}")
    click.echo(f"- Parts with no matches: {result['stats']['parts_with_no_matches']:,}")
    click.echo(f"- Total associations created: {result['stats']['total_associations_created']:,}")
    click.echo(f"- Duration: {result['stats']['duration_seconds']:.2f} seconds")

    # Add helpful messages
    if result['stats']['total_associations_created'] == 0:
        if result['stats']['total_parts_processed'] > 0:
            click.echo("💡 No associations created - this usually means:")
            click.echo("   • No images in database, or")
            click.echo("   • No matching image titles found")
            click.echo("   • Consider importing images first!")
        else:
            click.echo("💡 No parts found in database")

    if export_report and 'error' not in result:
        click.echo("\nReports exported to:")
        for report_type, file_path in result.get('report_files', {}).items():
            if file_path and os.path.exists(file_path):
                click.echo(f"- {report_type.capitalize()}: {file_path}")


@cli.command()
@click.option('--export-report/--no-export-report', default=True, help='Export detailed report to CSV')
@click.option('--report-dir', default=None, help='Directory to save the report')
def associate_drawings_fast(export_report, report_dir):
    """Associate all drawings with matching parts using optimized vectorized operations."""
    result = maintenance.associate_all_drawings_with_parts_vectorized(export_report, report_dir)

    click.echo("\nOPTIMIZED Drawing-Part Association Summary:")
    click.echo(f"- Drawings processed: {result['stats']['total_drawings_processed']:,}")
    click.echo(f"- Drawings with matching parts: {result['stats']['drawings_with_matching_parts']:,}")
    click.echo(f"- Drawings with no matches: {result['stats']['drawings_with_no_matches']:,}")
    click.echo(f"- Drawings without spare part number: {result['stats']['drawings_without_spare_part_number']:,}")
    click.echo(f"- Total associations created: {result['stats']['total_associations_created']:,}")
    click.echo(f"- Duration: {result['stats']['duration_seconds']:.2f} seconds")

    # Add helpful messages
    if result['stats']['total_associations_created'] == 0:
        if result['stats']['total_drawings_processed'] > 0:
            click.echo("💡 No associations created - this usually means:")
            click.echo("   • No matching part numbers found, or")
            click.echo("   • All associations already exist, or")
            click.echo("   • Drawing spare part numbers don't match part database")
        else:
            click.echo("💡 No drawings with spare part numbers found")

    if export_report and 'error' not in result:
        click.echo("\nReports exported to:")
        for report_type, file_path in result.get('report_files', {}).items():
            if file_path and os.path.exists(file_path):
                click.echo(f"- {report_type.capitalize()}: {file_path}")


@cli.command()
@click.option('--export-report/--no-export-report', default=True, help='Export detailed reports to CSV')
@click.option('--report-dir', default=None, help='Directory to save the reports')
def run_all_fast(export_report, report_dir):
    """Run all optimized maintenance operations."""
    click.echo("Starting ALL optimized database maintenance operations...\n")

    total_start_time = time.time()

    # Part-Image associations
    click.echo("1️⃣  Running optimized part-image associations...")
    img_start = time.time()
    img_result = maintenance.associate_all_parts_with_images_vectorized(export_report, report_dir)
    img_time = time.time() - img_start

    click.echo(
        f"   Created {img_result['stats']['total_associations_created']:,} part-image associations in {img_time:.2f}s")

    if img_result['stats']['total_associations_created'] == 0:
        if img_result['stats']['total_parts_processed'] > 0:
            click.echo("   💡 No part-image associations: No images in database or no matches found")
        else:
            click.echo("   💡 No parts found in database")

    # Drawing-Part associations
    click.echo("\n2️⃣  Running optimized drawing-part associations...")
    drw_start = time.time()
    drw_result = maintenance.associate_all_drawings_with_parts_vectorized(export_report, report_dir)
    drw_time = time.time() - drw_start

    click.echo(
        f"   Created {drw_result['stats']['total_associations_created']:,} drawing-part associations in {drw_time:.2f}s")

    if drw_result['stats']['total_associations_created'] == 0:
        if drw_result['stats']['total_drawings_processed'] > 0:
            click.echo("   💡 No drawing-part associations: No matching part numbers or all exist already")
        else:
            click.echo("   💡 No drawings with spare part numbers found")

    # Summary
    total_associations = img_result['stats']['total_associations_created'] + drw_result['stats'][
        'total_associations_created']
    total_time = time.time() - total_start_time

    click.echo(f"\n🎉 ALL OPTIMIZED MAINTENANCE COMPLETE!")
    click.echo(f"Total associations created: {total_associations:,}")
    click.echo(f"⏱ Total time: {total_time:.2f} seconds")

    # Overall guidance
    if total_associations == 0:
        click.echo(f"\n💡 GUIDANCE:")
        click.echo(f"   • To create part-image associations: Import images first")
        click.echo(f"   • To create drawing-part associations: Ensure drawings have matching spare part numbers")
        click.echo(f"   • Check that your data is properly loaded in the database")

    if export_report and 'error' not in img_result and 'error' not in drw_result:
        click.echo("\nReports exported to:")
        for result in [img_result, drw_result]:
            for report_type, file_path in result.get('report_files', {}).items():
                if file_path and os.path.exists(file_path):
                    click.echo(f"- {report_type.capitalize()}: {file_path}")

@cli.command()
@click.option('--export-report/--no-export-report', default=True, help='Export embedding validation report to CSV')
@click.option('--report-dir', default=None, help='Directory to save the report')
def validate_embeddings(export_report, report_dir):
    """Validate that all images have embeddings and embeddings map correctly."""
    result = maintenance.validate_image_embeddings_vectorized(export_report, report_dir)

    click.echo("\nOPTIMIZED Image-Embedding Validation Summary:")
    click.echo(f"- Total images: {result['stats']['total_images']:,}")
    click.echo(f"- Total embeddings: {result['stats']['total_embeddings']:,}")
    click.echo(f"- Images missing embeddings: {result['stats']['images_missing_embeddings']:,}")
    click.echo(f"- Orphan embeddings: {result['stats']['orphan_embeddings']:,}")
    click.echo(f"- Duration: {result['stats']['duration_seconds']:.2f} seconds")

    if export_report and 'error' not in result:
        click.echo("\nReports exported to:")
        for report_type, file_path in result.get('report_files', {}).items():
            if file_path and os.path.exists(file_path):
                click.echo(f"- {report_type.capitalize()}: {file_path}")


if __name__ == '__main__':
    cli()