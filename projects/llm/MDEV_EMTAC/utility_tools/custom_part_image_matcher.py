#!/usr/bin/env python3
"""
Custom Part-Image Matcher for PART PICTURE LISTING.xls

This script is customized specifically for your Excel file at:
C:\Users\10169062\Desktop\AU_IndusMaintdb\Database\DB_LOADSHEETS\PART PICTURE LISTING.xls

Expected columns: ITEMNUM, FILE, DESCRIPTION, UPDATE, USER, NEWFILENAME, NAMEONLY
"""

import pandas as pd
import os
import sys
from datetime import datetime

# Add your project paths
# Adjust these imports based on your actual project structure
sys.path.append(r'C:\Users\10169062\Desktop\AU_IndusMaintdb')
from modules.configuration.config_env import DatabaseConfig
from modules.configuration.log_config import info_id, error_id, debug_id, warning_id, get_request_id

# Import your models - adjust these imports to match your actual model locations
# from models import Part, Image, PartsPositionImageAssociation, Position

# File paths
EXCEL_FILE_PATH = r"C:\Users\10169062\Desktop\AU_IndusMaintdb\Database\DB_LOADSHEETS\PART PICTURE LISTING.xls"
RESULTS_FILE_PATH = r"C:\Users\10169062\Desktop\AU_IndusMaintdb\Database\DB_LOADSHEETS\matching_results.csv"
LOG_FILE_PATH = r"C:\Users\10169062\Desktop\AU_IndusMaintdb\Database\DB_LOADSHEETS\matching_log.txt"


def log_to_file(message: str, log_file_path: str = LOG_FILE_PATH):
    """Log messages to both console and file."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_message = f"[{timestamp}] {message}"
    print(log_message)

    try:
        with open(log_file_path, 'a', encoding='utf-8') as f:
            f.write(log_message + '\n')
    except Exception as e:
        print(f"Warning: Could not write to log file: {e}")


def analyze_excel_file():
    """
    First, let's analyze the Excel file to understand its structure.
    """
    log_to_file("=" * 60)
    log_to_file("ANALYZING EXCEL FILE")
    log_to_file("=" * 60)

    try:
        # Check if file exists
        if not os.path.exists(EXCEL_FILE_PATH):
            log_to_file(f"❌ ERROR: File not found at {EXCEL_FILE_PATH}")
            return None

        log_to_file(f"✅ File found: {EXCEL_FILE_PATH}")
        file_size = os.path.getsize(EXCEL_FILE_PATH)
        log_to_file(f"📁 File size: {file_size:,} bytes")

        # Read Excel file
        log_to_file("📖 Reading Excel file...")
        df = pd.read_excel(EXCEL_FILE_PATH, engine='xlrd')  # xlrd for .xls files

        log_to_file(f"✅ Successfully loaded {len(df)} rows")
        log_to_file(f"📊 Columns found: {list(df.columns)}")

        # Check for expected columns
        expected_columns = ['ITEMNUM', 'NAMEONLY']
        available_columns = [col for col in expected_columns if col in df.columns]
        missing_columns = [col for col in expected_columns if col not in df.columns]

        if available_columns:
            log_to_file(f"✅ Available columns: {available_columns}")

        if missing_columns:
            log_to_file(f"❌ Missing columns: {missing_columns}")
            log_to_file("Available columns in file:")
            for i, col in enumerate(df.columns):
                log_to_file(f"  {i + 1}. {col}")
            return None

        # Show sample data
        log_to_file("\n📋 Sample data (first 5 rows):")
        for idx in range(min(5, len(df))):
            row = df.iloc[idx]
            itemnum = str(row.get('ITEMNUM', 'N/A')).strip()
            nameonly = str(row.get('NAMEONLY', 'N/A')).strip()
            description = str(row.get('DESCRIPTION', 'N/A')).strip()[:50]  # Truncate long descriptions
            log_to_file(f"  Row {idx + 1}: ITEMNUM='{itemnum}' | NAMEONLY='{nameonly}' | DESC='{description}...'")

        # Data quality analysis
        log_to_file("\n🔍 Data Quality Analysis:")

        # Clean the data
        df['ITEMNUM_clean'] = df['ITEMNUM'].astype(str).str.strip()
        df['NAMEONLY_clean'] = df['NAMEONLY'].astype(str).str.strip()

        # Count empty/null values
        itemnum_empty = df['ITEMNUM_clean'].isin(['', 'nan', 'None', 'NaN']).sum()
        nameonly_empty = df['NAMEONLY_clean'].isin(['', 'nan', 'None', 'NaN']).sum()

        log_to_file(f"  Empty ITEMNUM values: {itemnum_empty}")
        log_to_file(f"  Empty NAMEONLY values: {nameonly_empty}")

        # Count unique values
        itemnum_unique = df['ITEMNUM_clean'].nunique()
        nameonly_unique = df['NAMEONLY_clean'].nunique()

        log_to_file(f"  Unique ITEMNUM values: {itemnum_unique}")
        log_to_file(f"  Unique NAMEONLY values: {nameonly_unique}")

        # Check for duplicates
        itemnum_duplicates = len(df) - itemnum_unique
        nameonly_duplicates = len(df) - nameonly_unique

        if itemnum_duplicates > 0:
            log_to_file(f"  ⚠️ Duplicate ITEMNUM entries: {itemnum_duplicates}")

        if nameonly_duplicates > 0:
            log_to_file(f"  ⚠️ Duplicate NAMEONLY entries: {nameonly_duplicates}")

        # Clean data for processing
        clean_df = df.dropna(subset=['ITEMNUM', 'NAMEONLY'])
        clean_df = clean_df[
            (~clean_df['ITEMNUM_clean'].isin(['', 'nan', 'None', 'NaN'])) &
            (~clean_df['NAMEONLY_clean'].isin(['', 'nan', 'None', 'NaN']))
            ]

        log_to_file(f"✅ Clean data: {len(clean_df)} rows ready for processing")

        return clean_df

    except Exception as e:
        log_to_file(f"❌ ERROR analyzing Excel file: {e}")
        return None


def analyze_database():
    """
    Analyze the current database content.
    """
    log_to_file("\n" + "=" * 60)
    log_to_file("ANALYZING DATABASE")
    log_to_file("=" * 60)

    try:
        db_config = DatabaseConfig()

        with db_config.main_session() as session:
            # Count parts
            part_count = session.query(Part).count()
            log_to_file(f"📦 Total parts in database: {part_count:,}")

            # Sample part numbers
            sample_parts = session.query(Part.part_number, Part.name).limit(5).all()
            log_to_file("Sample parts:")
            for i, (part_num, name) in enumerate(sample_parts, 1):
                log_to_file(f"  {i}. {part_num} - {name}")

            # Count images
            image_count = session.query(Image).count()
            log_to_file(f"\n🖼️ Total images in database: {image_count:,}")

            # Sample image titles
            sample_images = session.query(Image.title, Image.description).limit(5).all()
            log_to_file("Sample images:")
            for i, (title, desc) in enumerate(sample_images, 1):
                desc_short = (desc[:50] + '...') if desc and len(desc) > 50 else (desc or 'No description')
                log_to_file(f"  {i}. {title} - {desc_short}")

            # Count existing associations
            association_count = session.query(PartsPositionImageAssociation).count()
            log_to_file(f"\n🔗 Existing part-image associations: {association_count:,}")

            return {
                'parts': part_count,
                'images': image_count,
                'associations': association_count
            }

    except Exception as e:
        log_to_file(f"❌ ERROR analyzing database: {e}")
        return None


def match_parts_and_images(df):
    """
    Main function to match parts and images.
    """
    log_to_file("\n" + "=" * 60)
    log_to_file("STARTING PART-IMAGE MATCHING")
    log_to_file("=" * 60)

    if df is None or len(df) == 0:
        log_to_file("❌ No data to process")
        return

    # Statistics tracking
    stats = {
        'total': len(df),
        'processed': 0,
        'success': 0,
        'part_not_found': 0,
        'image_not_found': 0,
        'both_not_found': 0,
        'already_exists': 0,
        'errors': 0
    }

    results = []

    try:
        db_config = DatabaseConfig()
        request_id = get_request_id()

        with db_config.main_session() as session:
            log_to_file(f"🔄 Processing {len(df)} rows...")

            for index, row in df.iterrows():
                try:
                    itemnum = str(row['ITEMNUM']).strip()
                    nameonly = str(row['NAMEONLY']).strip()

                    stats['processed'] += 1

                    if stats['processed'] % 10 == 0:  # Progress update every 10 rows
                        log_to_file(
                            f"Progress: {stats['processed']}/{stats['total']} ({stats['processed'] / stats['total'] * 100:.1f}%)")

                    # Find the part
                    parts = Part.search(
                        part_number=itemnum,
                        exact_match=True,
                        limit=1,
                        session=session,
                        request_id=request_id
                    )
                    part = parts[0] if parts else None

                    # Find the image
                    image = session.query(Image).filter(Image.title == nameonly).first()

                    # Determine result
                    result = {
                        'row': stats['processed'],
                        'itemnum': itemnum,
                        'nameonly': nameonly,
                        'part_found': part is not None,
                        'image_found': image is not None,
                        'part_id': part.id if part else None,
                        'image_id': image.id if image else None,
                        'status': '',
                        'action': ''
                    }

                    if not part and not image:
                        result['status'] = 'BOTH_NOT_FOUND'
                        result['action'] = 'No action - neither part nor image found'
                        stats['both_not_found'] += 1

                    elif not part:
                        result['status'] = 'PART_NOT_FOUND'
                        result['action'] = f'Image found (ID: {image.id}) but part not found'
                        stats['part_not_found'] += 1

                    elif not image:
                        result['status'] = 'IMAGE_NOT_FOUND'
                        result['action'] = f'Part found (ID: {part.id}) but image not found'
                        stats['image_not_found'] += 1

                    else:
                        # Both found - check if association exists
                        existing = session.query(PartsPositionImageAssociation).filter(
                            PartsPositionImageAssociation.part_id == part.id,
                            PartsPositionImageAssociation.image_id == image.id
                        ).first()

                        if existing:
                            result['status'] = 'ASSOCIATION_EXISTS'
                            result['action'] = f'Association already exists (ID: {existing.id})'
                            stats['already_exists'] += 1
                        else:
                            # SUCCESS - both found and no existing association
                            result['status'] = 'SUCCESS'
                            result['action'] = f'Ready to create association: Part {part.id} <-> Image {image.id}'
                            stats['success'] += 1

                            # Here you would create the association
                            # Note: You'll need to handle the position_id requirement
                            # Option 1: Find a default position
                            # Option 2: Create association with NULL position (if schema allows)
                            # Option 3: Skip creating and just report the match

                            # For now, just logging the potential match
                            debug_id(f"Match found: {itemnum} ({part.id}) <-> {nameonly} ({image.id})", request_id)

                    results.append(result)

                except Exception as row_error:
                    error_msg = f"Error processing row {stats['processed']}: {row_error}"
                    log_to_file(f"❌ {error_msg}")
                    stats['errors'] += 1

                    results.append({
                        'row': stats['processed'],
                        'itemnum': itemnum if 'itemnum' in locals() else 'Unknown',
                        'nameonly': nameonly if 'nameonly' in locals() else 'Unknown',
                        'part_found': False,
                        'image_found': False,
                        'part_id': None,
                        'image_id': None,
                        'status': 'ERROR',
                        'action': error_msg
                    })

            # Don't commit yet - just analyze
            # session.commit()  # Uncomment when ready to save associations

    except Exception as e:
        log_to_file(f"❌ CRITICAL ERROR during matching: {e}")
        return None

    # Save results
    save_results(results, stats)

    # Print final statistics
    print_final_statistics(stats)

    return results, stats


def save_results(results, stats):
    """Save results to CSV file."""
    try:
        results_df = pd.DataFrame(results)
        results_df.to_csv(RESULTS_FILE_PATH, index=False)
        log_to_file(f"✅ Results saved to: {RESULTS_FILE_PATH}")
    except Exception as e:
        log_to_file(f"❌ Error saving results: {e}")


def print_final_statistics(stats):
    """Print final matching statistics."""
    log_to_file("\n" + "=" * 60)
    log_to_file("FINAL MATCHING STATISTICS")
    log_to_file("=" * 60)

    log_to_file(f"📊 Total rows: {stats['total']}")
    log_to_file(f"✅ Successfully processed: {stats['processed']}")
    log_to_file(f"🎯 Perfect matches (both found): {stats['success']}")
    log_to_file(f"📦 Part not found: {stats['part_not_found']}")
    log_to_file(f"🖼️ Image not found: {stats['image_not_found']}")
    log_to_file(f"❌ Both not found: {stats['both_not_found']}")
    log_to_file(f"🔗 Association already exists: {stats['already_exists']}")
    log_to_file(f"⚠️ Errors: {stats['errors']}")

    if stats['total'] > 0:
        success_rate = (stats['success'] / stats['total'] * 100)
        log_to_file(f"📈 Success rate: {success_rate:.1f}%")

        potential_matches = stats['success'] + stats['already_exists']
        match_rate = (potential_matches / stats['total'] * 100)
        log_to_file(f"🎯 Total match rate: {match_rate:.1f}%")


def main():
    """Main execution function."""
    log_to_file("PART-IMAGE MATCHING TOOL")
    log_to_file(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    log_to_file(f"Excel file: {EXCEL_FILE_PATH}")

    try:
        # Step 1: Analyze Excel file
        df = analyze_excel_file()
        if df is None:
            log_to_file("❌ Cannot proceed without valid Excel data")
            return

        # Step 2: Analyze database
        db_stats = analyze_database()
        if db_stats is None:
            log_to_file("❌ Cannot proceed without database access")
            return

        # Step 3: Perform matching
        results, stats = match_parts_and_images(df)

        log_to_file(f"\n✅ Process completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        log_to_file(f"📄 Results saved to: {RESULTS_FILE_PATH}")
        log_to_file(f"📄 Log saved to: {LOG_FILE_PATH}")

    except Exception as e:
        log_to_file(f"❌ CRITICAL ERROR in main process: {e}")
        import traceback
        log_to_file(f"Traceback: {traceback.format_exc()}")


if __name__ == "__main__":
    main()
