# modules/maintenance/db_maintenance.py
import sys
import os
import csv  # Import missing csv module

# Add the project root directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..')))

import click
import time
import pandas as pd
from datetime import datetime
from sqlalchemy import and_

# Define default log directory (relative to this script)
DEFAULT_LOG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "db_maint_logs")
# Create the default log directory if it doesn't exist
os.makedirs(DEFAULT_LOG_DIR, exist_ok=True)

from modules.configuration.config_env import DatabaseConfig
from modules.database_manager.db_manager import RelationshipManager, DuplicateManager
from modules.initial_setup.initializer_logger import initializer_logger
from modules.configuration.log_config import debug_id, info_id, error_id, get_request_id
from modules.emtacdb.emtacdb_fts import Part, Image, Drawing, PartsPositionImageAssociation, DrawingPartAssociation

# Ensure the project root is in the Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

def get_app_root():
    """Get the application root directory."""
    return os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))

def associate_all_parts_with_images(batch_size=1000, export_report=True, report_dir=None):
    """
    Associate all parts in the database with matching images.

    Args:
        batch_size: Number of parts to process in each batch
        export_report: Whether to export a detailed report to CSV
        report_dir: Directory to save the report (default: db_maint_logs directory)

    Returns:
        Dictionary containing association results and statistics
    """
    request_id = get_request_id()
    initializer_logger.info("Starting database maintenance: Associating all parts with matching images")
    info_id("Starting database maintenance: Associating all parts with matching images", request_id)
    start_time = time.time()

    # Initialize result structure
    result = {
        'stats': {
            'total_parts_processed': 0,
            'parts_with_matching_images': 0,
            'total_associations_created': 0,
            'parts_with_no_matches': 0,
            'duration_seconds': 0
        },
        'associations': {},  # Will hold part_number -> list of image details
        'unmatched_parts': []  # Will hold parts with no matching images
    }

    try:
        # Initialize database connection
        db_config = DatabaseConfig()
        session = db_config.get_main_session()
        debug_id("Database session initialized", request_id)

        # Use RelationshipManager to create associations
        with RelationshipManager(session=session, request_id=request_id) as manager:
            debug_id("RelationshipManager initialized", request_id)

            # Get association results
            assoc_result = manager.associate_parts_with_images_by_title()
            manager.commit()

            # Process results to build detailed report
            for part_id, associations in assoc_result.items():
                # Get part details
                part = session.query(Part).filter(Part.id == part_id).first()
                if not part:
                    initializer_logger.warning(f"Part with ID {part_id} not found in database")
                    continue

                # Track part in our results
                result['stats']['total_parts_processed'] += 1

                if associations:
                    # Part had matching images
                    result['stats']['parts_with_matching_images'] += 1
                    result['stats']['total_associations_created'] += len(associations)

                    # Get image details for each association
                    part_associations = []
                    for assoc in associations:
                        image = session.query(Image).filter(Image.id == assoc.image_id).first()
                        if image:
                            part_associations.append({
                                'image_id': image.id,
                                'image_title': image.title,
                                'image_file_path': image.file_path,
                                'association_id': assoc.id
                            })

                    # Store associations for this part
                    result['associations'][part.part_number] = {
                        'part_id': part.id,
                        'part_name': part.name,
                        'images': part_associations
                    }
                else:
                    # Part had no matching images
                    result['stats']['parts_with_no_matches'] += 1
                    result['unmatched_parts'].append({
                        'part_id': part.id,
                        'part_number': part.part_number,
                        'part_name': part.name
                    })

        # Calculate duration and add to stats
        duration = time.time() - start_time
        result['stats']['duration_seconds'] = duration

        # Log comprehensive results
        parts_processed = result['stats']['total_parts_processed']
        total_associations = result['stats']['total_associations_created']

        initializer_logger.info(f"Part-image association completed:")
        initializer_logger.info(f"- Parts processed: {parts_processed}")
        initializer_logger.info(f"- Parts with matching images: {result['stats']['parts_with_matching_images']}")
        initializer_logger.info(f"- Parts with no matches: {result['stats']['parts_with_no_matches']}")
        initializer_logger.info(f"- Associations created: {total_associations}")
        initializer_logger.info(f"- Duration: {duration:.2f} seconds")

        # Export report if requested
        if export_report and total_associations > 0:
            report_files = export_association_report(result, report_dir, request_id)
            result['report_files'] = report_files

        return result

    except Exception as e:
        initializer_logger.error(f"Error during part-image association: {e}", exc_info=True)
        error_id(f"Error during part-image association: {e}", request_id, exc_info=True)
        result['error'] = str(e)
        return result
    finally:
        # Ensure the session is properly closed
        db_config.MainSession.remove()
        debug_id("Database session closed", request_id)

def export_association_report(result, report_dir=None, request_id=None):
    """
    Export the association results to CSV files.

    Args:
        result: The association result dictionary
        report_dir: Directory to save the report (default: db_maint_logs directory)
        request_id: Optional request ID for logging

    Returns:
        Dictionary with paths to the generated report files
    """
    rid = request_id or get_request_id()
    initializer_logger.info("Exporting association report to CSV")
    info_id("Exporting association report to CSV", rid)

    # Determine report directory
    if not report_dir:
        report_dir = DEFAULT_LOG_DIR
    os.makedirs(report_dir, exist_ok=True)

    # Generate timestamp for filenames
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Export file paths
    report_files = {
        'summary': os.path.join(report_dir, f"part_image_association_summary_{timestamp}.csv"),
        'details': os.path.join(report_dir, f"part_image_association_details_{timestamp}.csv"),
        'unmatched': os.path.join(report_dir, f"unmatched_parts_{timestamp}.csv")
    }

    try:
        # 1. Create summary DataFrame
        summary_data = [{
            'timestamp': datetime.now().isoformat(),
            'total_parts_processed': result['stats']['total_parts_processed'],
            'parts_with_matches': result['stats']['parts_with_matching_images'],
            'parts_without_matches': result['stats']['parts_with_no_matches'],
            'total_associations': result['stats']['total_associations_created'],
            'duration_seconds': result['stats']['duration_seconds']
        }]

        summary_df = pd.DataFrame(summary_data)
        summary_df.to_csv(report_files['summary'], index=False)
        initializer_logger.info(f"Exported summary report to {report_files['summary']}")

        # 2. Create detailed associations DataFrame
        details_data = []

        for part_number, part_info in result['associations'].items():
            for image in part_info['images']:
                details_data.append({
                    'part_id': part_info['part_id'],
                    'part_number': part_number,
                    'part_name': part_info['part_name'],
                    'image_id': image['image_id'],
                    'image_title': image['image_title'],
                    'image_file_path': image['image_file_path'],
                    'association_id': image['association_id']
                })

        if details_data:
            details_df = pd.DataFrame(details_data)
            details_df.to_csv(report_files['details'], index=False)
            initializer_logger.info(f"Exported association details to {report_files['details']}")

        # 3. Create unmatched parts DataFrame
        if result['unmatched_parts']:
            unmatched_df = pd.DataFrame(result['unmatched_parts'])
            unmatched_df.to_csv(report_files['unmatched'], index=False)
            initializer_logger.info(f"Exported unmatched parts to {report_files['unmatched']}")

        return report_files

    except Exception as e:
        initializer_logger.error(f"Error exporting association report: {e}", exc_info=True)
        error_id(f"Error exporting association report: {e}", rid, exc_info=True)
        return None

def export_drawing_part_association_report(result, report_dir=None, request_id=None, session=None):
    """
    Export a detailed report of drawing-part associations.
    Handles multiple spare part numbers per drawing.

    Args:
        result: Result dictionary from associate_all_drawings_with_parts
        report_dir: Directory to save reports (default: db_maint_logs directory)
        request_id: Request ID for logging
        session: Database session

    Returns:
        Dictionary with paths to created report files
    """
    if not report_dir:
        report_dir = os.path.join(get_app_root(), 'db_maint_logs')

    # Ensure directory exists
    os.makedirs(report_dir, exist_ok=True)

    # Create timestamp for filenames
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    # Initialize files dictionary
    report_files = {}

    # 1. Create matched drawings report
    if result['associations']:
        matched_file = os.path.join(report_dir, f'drawing_part_matches_{timestamp}.csv')
        report_files['matched'] = matched_file

        with open(matched_file, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            # Write header
            writer.writerow([
                'Drawing Number',
                'Drawing Name',
                'Drawing Spare Part Numbers',
                'Matched Part Number',
                'Part Name',
                'Association ID'
            ])

            # Write data rows
            for drawing_number, data in result['associations'].items():
                # Get list of all parts for this drawing
                parts = data['parts']

                for part in parts:
                    writer.writerow([
                        drawing_number,
                        data['drawing_name'],
                        data['spare_part_number'],  # This now may contain multiple comma-separated values
                        part['part_number'],
                        part['part_name'],
                        part['association_id']
                    ])

        initializer_logger.info(f"Exported {len(result['associations'])} drawings with matches to {matched_file}")
        if request_id:
            info_id(f"Exported {len(result['associations'])} drawings with matches to {matched_file}", request_id)

    # 2. Create unmatched drawings report
    if result['unmatched_drawings']:
        unmatched_file = os.path.join(report_dir, f'drawing_part_unmatched_{timestamp}.csv')
        report_files['unmatched'] = unmatched_file

        with open(unmatched_file, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            # Write header
            writer.writerow([
                'Drawing Number',
                'Drawing Name',
                'Spare Part Numbers',
                'Parsed Part Numbers'  # New column showing the individual parts we tried to match
            ])

            # Write data rows
            for drawing in result['unmatched_drawings']:
                writer.writerow([
                    drawing['drawing_number'],
                    drawing['drawing_name'],
                    drawing['spare_part_number'],
                    ', '.join(drawing.get('parsed_part_numbers', []))
                ])

        initializer_logger.info(f"Exported {len(result['unmatched_drawings'])} unmatched drawings to {unmatched_file}")
        if request_id:
            info_id(f"Exported {len(result['unmatched_drawings'])} unmatched drawings to {unmatched_file}", request_id)

    # 3. Create summary report
    summary_file = os.path.join(report_dir, f'drawing_part_association_summary_{timestamp}.txt')
    report_files['summary'] = summary_file

    with open(summary_file, 'w') as f:
        f.write("Drawing-Part Association Summary\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write(f"Total drawings processed: {result['stats']['total_drawings_processed']}\n")
        f.write(f"Drawings with matching parts: {result['stats']['drawings_with_matching_parts']}\n")
        f.write(f"Drawings with no matches: {result['stats']['drawings_with_no_matches']}\n")
        f.write(f"Drawings without spare part number: {result['stats']['drawings_without_spare_part_number']}\n")
        f.write(f"Drawings with multiple part numbers: {result['stats']['multiple_part_numbers_count']}\n")
        f.write(f"Total associations created: {result['stats']['total_associations_created']}\n")
        f.write(f"Duration: {result['stats']['duration_seconds']:.2f} seconds\n")

    initializer_logger.info(f"Exported summary report to {summary_file}")
    if request_id:
        info_id(f"Exported summary report to {summary_file}", request_id)

    return report_files

def find_and_report_duplicate_parts(threshold=0.9, export_to_csv=True, report_dir=None):
    """
    Find potential duplicate parts and generate a report.

    Args:
        threshold: Similarity threshold (0.0-1.0)
        export_to_csv: Whether to export results to CSV
        report_dir: Directory to save the report (default: db_maint_logs directory)

    Returns:
        List of potential duplicate pairs
    """
    request_id = get_request_id()
    initializer_logger.info(f"Starting database maintenance: Finding duplicate parts (threshold: {threshold})")
    start_time = time.time()

    try:
        # Initialize database connection
        db_config = DatabaseConfig()
        session = db_config.get_main_session()

        # Use DuplicateManager to find duplicates
        with DuplicateManager(session=session, request_id=request_id) as manager:
            duplicates = manager.find_duplicate_parts(threshold=threshold)

        # Log results
        duration = time.time() - start_time
        initializer_logger.info(f"Duplicate part detection completed:")
        initializer_logger.info(f"- Potential duplicates found: {len(duplicates)}")
        initializer_logger.info(f"- Duration: {duration:.2f} seconds")

        # Export to CSV if requested
        if export_to_csv and duplicates:
            # Determine report directory
            if not report_dir:
                report_dir = DEFAULT_LOG_DIR
            os.makedirs(report_dir, exist_ok=True)

            # Convert to DataFrame
            duplicate_data = []
            for source_id, target_id, similarity in duplicates:
                duplicate_data.append({
                    'source_id': source_id,
                    'target_id': target_id,
                    'similarity': similarity
                })

            df = pd.DataFrame(duplicate_data)

            # Generate filename with timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = os.path.join(report_dir, f"duplicate_parts_{timestamp}.csv")

            # Save to file
            df.to_csv(filename, index=False)
            initializer_logger.info(f"Exported duplicate report to {filename}")

        return duplicates

    except Exception as e:
        initializer_logger.error(f"Error during duplicate part detection: {e}", exc_info=True)
        return []
    finally:
        # Ensure the session is properly closed
        db_config.MainSession.remove()

def validate_data_integrity():
    """
    Validate data integrity across the database.

    Checks:
    - Parts without images
    - Images without associations
    - Missing required fields
    - Etc.

    Returns:
        Dictionary with validation results
    """
    # Implementation for data integrity validation
    pass

def associate_drawings_with_parts_by_number(self):
    """
    Associates drawings with parts based on spare part numbers.
    Handles multiple comma-separated part numbers per drawing.

    Returns:
        Dict mapping drawing_id to list of created associations
    """
    # Import the logging functions with correct module path
    from modules.configuration.log_config import debug_id, info_id

    # Use info_id with correct parameter order (message, request_id)
    info_id("Associating drawings with parts based on spare part numbers", self.request_id)

    # Get all drawings with spare part numbers
    drawings = self.session.query(Drawing).filter(Drawing.drw_spare_part_number.isnot(None))

    # Track associations for each drawing
    associations_by_drawing = {}

    for drawing in drawings:
        # Skip if drawing has no spare part number
        if not drawing.drw_spare_part_number or not drawing.drw_spare_part_number.strip():
            continue

        # Split by comma and clean up each part number
        part_numbers = [pn.strip() for pn in drawing.drw_spare_part_number.split(',') if pn.strip()]

        drawing_associations = []

        # Process each part number for this drawing
        for part_number in part_numbers:
            # Find parts matching this part number
            matching_parts = self.session.query(Part).filter(
                Part.part_number == part_number
            ).all()

            # Create association for each matching part
            for part in matching_parts:
                # Check if association already exists
                existing = self.session.query(DrawingPartAssociation).filter(
                    DrawingPartAssociation.drawing_id == drawing.id,
                    DrawingPartAssociation.part_id == part.id
                ).first()

                if not existing:
                    # Create new association
                    association = DrawingPartAssociation(
                        drawing_id=drawing.id,
                        part_id=part.id,
                        created_by="system_maintenance",
                        association_type="automatic_by_number"
                    )
                    self.session.add(association)

                    # We need to flush to get the ID
                    self.session.flush()

                    drawing_associations.append(association)
                    debug_id(f"Created association between drawing {drawing.drw_number} "
                             f"and part {part.part_number}", self.request_id)
                else:
                    debug_id(f"Association already exists between drawing {drawing.drw_number} "
                             f"and part {part.part_number}", self.request_id)

        # Store associations for this drawing
        if drawing_associations:
            associations_by_drawing[drawing.id] = drawing_associations

    return associations_by_drawing

def associate_all_drawings_with_parts(export_report=True, report_dir=None):
    """
    Associate all drawings in the database with matching parts based on spare part numbers.
    Handles multiple spare part numbers per drawing (comma-separated).

    Args:
        export_report: Whether to export a detailed report to CSV
        report_dir: Directory to save the report (default: db_maint_logs directory)

    Returns:
        Dictionary containing association results and statistics
    """
    request_id = get_request_id()
    initializer_logger.info("Starting database maintenance: Associating drawings with parts")
    info_id("Starting database maintenance: Associating drawings with parts", request_id)
    start_time = time.time()

    # Initialize result structure
    result = {
        'stats': {
            'total_drawings_processed': 0,
            'drawings_with_matching_parts': 0,
            'total_associations_created': 0,
            'drawings_with_no_matches': 0,
            'drawings_without_spare_part_number': 0,
            'multiple_part_numbers_count': 0,  # New stat for drawings with multiple part numbers
            'duration_seconds': 0
        },
        'associations': {},  # Will hold drawing_id -> list of part details
        'unmatched_drawings': []  # Will hold drawings with no matching parts
    }

    try:
        # Initialize database connection
        db_config = DatabaseConfig()
        session = db_config.get_main_session()
        debug_id("Database session initialized", request_id)

        # Use RelationshipManager with our specific request_id
        with RelationshipManager(session=session, request_id=request_id) as manager:
            debug_id("RelationshipManager initialized", request_id)

            # Get association results
            assoc_result = manager.associate_drawings_with_parts_by_number()
            manager.commit()

            # Debug info about the result
            initializer_logger.info(f"Raw association result has {len(assoc_result)} items")
            debug_id(f"Raw association result has {len(assoc_result)} items", request_id)

            # Process results to build detailed report
            # First, get all drawings to ensure we count them even if no new associations were created
            drawings = session.query(Drawing).filter(Drawing.drw_spare_part_number.isnot(None)).all()
            initializer_logger.info(f"Total drawings with spare part numbers in database: {len(drawings)}")
            debug_id(f"Total drawings with spare part numbers in database: {len(drawings)}", request_id)

            # Track all drawings with spare part numbers
            for drawing in drawings:
                result['stats']['total_drawings_processed'] += 1

                # Check if this drawing has multiple part numbers
                if drawing.drw_spare_part_number:
                    part_numbers = [pn.strip() for pn in drawing.drw_spare_part_number.split(',') if pn.strip()]
                    if len(part_numbers) > 1:
                        result['stats']['multiple_part_numbers_count'] += 1
                        initializer_logger.debug(
                            f"Drawing {drawing.drw_number} has multiple part numbers: {part_numbers}")
                else:
                    result['stats']['drawings_without_spare_part_number'] += 1

            # Now process newly created associations
            for drawing_id, associations in assoc_result.items():
                # Get drawing details
                drawing = session.query(Drawing).filter(Drawing.id == drawing_id).first()
                if not drawing:
                    initializer_logger.warning(f"Drawing with ID {drawing_id} not found in database")
                    continue

                if associations:
                    # Drawing had matching parts
                    result['stats']['drawings_with_matching_parts'] += 1
                    result['stats']['total_associations_created'] += len(associations)

                    # Get part details for each association
                    drawing_associations = []
                    for assoc in associations:
                        part = session.query(Part).filter(Part.id == assoc.part_id).first()
                        if part:
                            drawing_associations.append({
                                'part_id': part.id,
                                'part_number': part.part_number,
                                'part_name': part.name,
                                'association_id': assoc.id
                            })

                    # Store associations for this drawing
                    result['associations'][drawing.drw_number] = {
                        'drawing_id': drawing.id,
                        'drawing_name': drawing.drw_name,
                        'spare_part_number': drawing.drw_spare_part_number,
                        'matched_part_numbers': [part.get('part_number', '') for part in drawing_associations],
                        'parts': drawing_associations
                    }

            # Check for unmatched drawings
            for drawing in drawings:
                # Check if we have any associations for this drawing
                has_associations = any(
                    assoc.drawing_id == drawing.id
                    for assocs in assoc_result.values()
                    for assoc in assocs
                )

                # Also check existing associations
                existing_count = session.query(DrawingPartAssociation).filter(
                    DrawingPartAssociation.drawing_id == drawing.id
                ).count()

                if not has_associations and existing_count == 0:
                    # Drawing had no matching parts
                    result['stats']['drawings_with_no_matches'] += 1

                    # Split part numbers for reporting
                    part_numbers = []
                    if drawing.drw_spare_part_number:
                        part_numbers = [pn.strip() for pn in drawing.drw_spare_part_number.split(',') if pn.strip()]

                    result['unmatched_drawings'].append({
                        'drawing_id': drawing.id,
                        'drawing_number': drawing.drw_number,
                        'drawing_name': drawing.drw_name,
                        'spare_part_number': drawing.drw_spare_part_number,
                        'parsed_part_numbers': part_numbers
                    })

        # Calculate duration and add to stats
        duration = time.time() - start_time
        result['stats']['duration_seconds'] = duration

        # Log comprehensive results
        initializer_logger.info(f"Drawing-part association completed:")
        initializer_logger.info(f"- Drawings processed: {result['stats']['total_drawings_processed']}")
        initializer_logger.info(f"- Drawings with matching parts: {result['stats']['drawings_with_matching_parts']}")
        initializer_logger.info(f"- Drawings with no matches: {result['stats']['drawings_with_no_matches']}")
        initializer_logger.info(
            f"- Drawings without spare part number: {result['stats']['drawings_without_spare_part_number']}")
        initializer_logger.info(
            f"- Drawings with multiple part numbers: {result['stats']['multiple_part_numbers_count']}")
        initializer_logger.info(f"- Associations created: {result['stats']['total_associations_created']}")
        initializer_logger.info(f"- Duration: {duration:.2f} seconds")

        # Export report if requested
        if export_report and result['stats']['total_drawings_processed'] > 0:
            report_files = export_drawing_part_association_report(result, report_dir, request_id, session)
            result['report_files'] = report_files

        return result

    except Exception as e:
        initializer_logger.error(f"Error during drawing-part association: {e}", exc_info=True)
        error_id(f"Error during drawing-part association: {e}", request_id, exc_info=True)
        result['error'] = str(e)
        return result
    finally:
        # Ensure the session is properly closed
        db_config.MainSession.remove()
        debug_id("Database session closed", request_id)

# CLI interface
@click.group()
def cli():
    """Database maintenance utilities."""
    pass


@cli.command()
@click.option('--export-report/--no-export-report', default=True, help='Export detailed report to CSV')
@click.option('--report-dir', default=None, help='Directory to save the report (default: db_maint_logs directory)')
def associate_drawings_with_parts(export_report, report_dir):
    """Associate all drawings with matching parts based on spare part numbers."""
    result = associate_all_drawings_with_parts(export_report, report_dir)

    # Print a summary to the console
    click.echo("\nDrawing-Part Association Summary:")
    click.echo(f"- Drawings processed: {result['stats']['total_drawings_processed']}")
    click.echo(f"- Drawings with matching parts: {result['stats']['drawings_with_matching_parts']}")
    click.echo(f"- Drawings with no matches: {result['stats']['drawings_with_no_matches']}")
    click.echo(f"- Drawings without spare part number: {result['stats']['drawings_without_spare_part_number']}")
    click.echo(f"- Total associations created: {result['stats']['total_associations_created']}")
    click.echo(f"- Duration: {result['stats']['duration_seconds']:.2f} seconds")

    if export_report and 'error' not in result:
        click.echo("\nReports have been exported to:")
        for report_type, file_path in result.get('report_files', {}).items():
            if file_path and os.path.exists(file_path):
                click.echo(f"- {report_type.capitalize()}: {file_path}")


@cli.command()
@click.option('--batch-size', default=1000, help='Number of parts to process in each batch')
@click.option('--export-report/--no-export-report', default=True, help='Export detailed report to CSV')
@click.option('--report-dir', default=None, help='Directory to save the report (default: db_maint_logs directory)')
def associate_images(batch_size, export_report, report_dir):
    """Associate all parts with matching images and generate a report."""
    result = associate_all_parts_with_images(batch_size, export_report, report_dir)

    # Print a summary to the console
    click.echo("\nPart-Image Association Summary:")
    click.echo(f"- Parts processed: {result['stats']['total_parts_processed']}")
    click.echo(f"- Parts with matching images: {result['stats']['parts_with_matching_images']}")
    click.echo(f"- Parts with no matches: {result['stats']['parts_with_no_matches']}")
    click.echo(f"- Total associations created: {result['stats']['total_associations_created']}")
    click.echo(f"- Duration: {result['stats']['duration_seconds']:.2f} seconds")

    if export_report and 'error' not in result:
        click.echo("\nReports have been exported to:")
        for report_type, file_path in result.get('report_files', {}).items():
            if file_path and os.path.exists(file_path):
                click.echo(f"- {report_type.capitalize()}: {file_path}")


@cli.command()
@click.option('--threshold', default=0.9, help='Similarity threshold (0.0-1.0)')
@click.option('--export/--no-export', default=True, help='Export results to CSV')
@click.option('--report-dir', default=None, help='Directory to save the report (default: db_maint_logs directory)')
def find_duplicates(threshold, export, report_dir):
    """Find potential duplicate parts."""
    duplicates = find_and_report_duplicate_parts(threshold, export, report_dir)
    click.echo(f"Found {len(duplicates)} potential duplicate parts")


@cli.command()
def validate():
    """Validate data integrity."""
    results = validate_data_integrity()
    # Display results
    pass


if __name__ == '__main__':
    cli()