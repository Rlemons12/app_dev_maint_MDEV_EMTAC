#!/usr/bin/env python3
"""
Migrate existing embeddings from legacy to pgvector format

This script migrates existing DocumentEmbedding records from the legacy LargeBinary
format to the new pgvector format for better performance and native vector operations.
"""

import sys
import os
import json
from datetime import datetime

# Add the project root to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from modules.configuration.config_env import DatabaseConfig
from modules.configuration.log_config import (
    debug_id, info_id, warning_id, error_id,
    set_request_id, get_request_id, with_request_id, log_timed_operation
)
from modules.emtacdb.emtacdb_fts import DocumentEmbedding


@with_request_id
def migrate_existing_embeddings(request_id=None):
    """
    Migrate existing embeddings from legacy LargeBinary to pgvector format.

    Args:
        request_id: Optional request ID for logging tracking

    Returns:
        dict: Migration results with counts and statistics
    """
    info_id("Starting migration of existing embeddings to pgvector format", request_id)

    db_config = DatabaseConfig()
    migration_stats = {
        'total_found': 0,
        'successfully_migrated': 0,
        'errors': 0,
        'already_migrated': 0,
        'start_time': datetime.now(),
        'end_time': None
    }

    try:
        with log_timed_operation("embedding_migration", request_id):
            with db_config.main_session() as session:

                # Get count of all embeddings for statistics
                total_embeddings = session.query(DocumentEmbedding).count()
                info_id(f"Total embeddings in database: {total_embeddings}", request_id)

                # Get embeddings that only have legacy data (need migration)
                legacy_embeddings = session.query(DocumentEmbedding).filter(
                    DocumentEmbedding.model_embedding.isnot(None),
                    DocumentEmbedding.embedding_vector.is_(None)
                ).all()

                migration_stats['total_found'] = len(legacy_embeddings)
                info_id(f"Found {len(legacy_embeddings)} legacy embeddings to migrate", request_id)

                # Check how many are already migrated
                already_migrated = session.query(DocumentEmbedding).filter(
                    DocumentEmbedding.embedding_vector.isnot(None)
                ).count()

                migration_stats['already_migrated'] = already_migrated
                info_id(f"Found {already_migrated} embeddings already in pgvector format", request_id)

                if len(legacy_embeddings) == 0:
                    info_id("No embeddings need migration. All are already in pgvector format or missing.", request_id)
                    migration_stats['end_time'] = datetime.now()
                    return migration_stats

                # Process embeddings in batches for better performance
                batch_size = 100
                migrated_count = 0
                error_count = 0

                info_id(f"Starting migration in batches of {batch_size}", request_id)

                for i, embedding in enumerate(legacy_embeddings):
                    try:
                        # Use the built-in migration method
                        if embedding.migrate_to_pgvector():
                            migrated_count += 1
                            debug_id(f"Migrated embedding {embedding.id} for document {embedding.document_id}",
                                     request_id)

                            # Commit periodically to avoid large transactions
                            if migrated_count % batch_size == 0:
                                session.commit()
                                info_id(
                                    f"Committed batch: {migrated_count}/{len(legacy_embeddings)} embeddings migrated",
                                    request_id)
                        else:
                            error_count += 1
                            warning_id(
                                f"Failed to migrate embedding {embedding.id} - migrate_to_pgvector returned False",
                                request_id)

                    except Exception as e:
                        error_count += 1
                        error_id(f"Error migrating embedding {embedding.id}: {str(e)}", request_id, exc_info=True)
                        continue

                # Final commit for any remaining changes
                try:
                    session.commit()
                    info_id("Final commit completed", request_id)
                except Exception as e:
                    session.rollback()
                    error_id(f"Final commit failed, rolling back: {str(e)}", request_id, exc_info=True)
                    raise

                migration_stats['successfully_migrated'] = migrated_count
                migration_stats['errors'] = error_count
                migration_stats['end_time'] = datetime.now()

                # Log final statistics
                duration = migration_stats['end_time'] - migration_stats['start_time']
                info_id(f"Migration completed in {duration.total_seconds():.2f} seconds", request_id)
                info_id(f"Successfully migrated: {migrated_count}", request_id)
                info_id(f"Errors encountered: {error_count}", request_id)

                if error_count > 0:
                    warning_id(f"Migration completed with {error_count} errors. Check logs for details.", request_id)
                else:
                    info_id("Migration completed successfully with no errors!", request_id)

                return migration_stats

    except Exception as e:
        migration_stats['end_time'] = datetime.now()
        error_id(f"Critical error during migration: {str(e)}", request_id, exc_info=True)
        raise


def verify_migration(request_id=None):
    """
    Verify that the migration was successful by checking the data.

    Args:
        request_id: Optional request ID for logging tracking

    Returns:
        dict: Verification results
    """
    info_id("Starting migration verification", request_id)

    db_config = DatabaseConfig()
    verification_results = {
        'total_embeddings': 0,
        'legacy_only': 0,
        'pgvector_only': 0,
        'both_formats': 0,
        'no_data': 0,
        'verification_passed': False
    }

    try:
        with db_config.main_session() as session:
            all_embeddings = session.query(DocumentEmbedding).all()
            verification_results['total_embeddings'] = len(all_embeddings)

            for embedding in all_embeddings:
                storage_type = embedding.get_storage_type()

                if storage_type == 'legacy':
                    verification_results['legacy_only'] += 1
                elif storage_type == 'pgvector':
                    verification_results['pgvector_only'] += 1
                elif storage_type == 'both':
                    verification_results['both_formats'] += 1
                else:  # 'none'
                    verification_results['no_data'] += 1

            # Check if migration was successful (no legacy-only records)
            verification_results['verification_passed'] = verification_results['legacy_only'] == 0

            info_id(f"Verification results:", request_id)
            info_id(f"  Total embeddings: {verification_results['total_embeddings']}", request_id)
            info_id(f"  Legacy only: {verification_results['legacy_only']}", request_id)
            info_id(f"  pgvector only: {verification_results['pgvector_only']}", request_id)
            info_id(f"  Both formats: {verification_results['both_formats']}", request_id)
            info_id(f"  No data: {verification_results['no_data']}", request_id)

            if verification_results['verification_passed']:
                info_id("Verification PASSED: All embeddings have pgvector data", request_id)
            else:
                warning_id(
                    f"Verification FAILED: {verification_results['legacy_only']} embeddings still only have legacy data",
                    request_id)

            return verification_results

    except Exception as e:
        error_id(f"Error during verification: {str(e)}", request_id, exc_info=True)
        return verification_results


def create_pgvector_indexes(request_id=None):
    """
    Create pgvector indexes for better performance.

    Args:
        request_id: Optional request ID for logging tracking

    Returns:
        bool: True if indexes created successfully
    """
    info_id("Creating pgvector indexes for optimal performance", request_id)

    db_config = DatabaseConfig()

    try:
        with db_config.main_session() as session:
            success = DocumentEmbedding.create_pgvector_indexes(session)

            if success:
                info_id("pgvector indexes created successfully", request_id)
            else:
                warning_id("Some issues occurred while creating pgvector indexes", request_id)

            return success

    except Exception as e:
        error_id(f"Error creating pgvector indexes: {str(e)}", request_id, exc_info=True)
        return False


def print_migration_summary(migration_stats, verification_results):
    """
    Print a comprehensive summary of the migration process.

    Args:
        migration_stats: Results from the migration process
        verification_results: Results from the verification process
    """
    print("\n" + "=" * 60)
    print("PGVECTOR MIGRATION SUMMARY")
    print("=" * 60)

    duration = migration_stats['end_time'] - migration_stats['start_time']

    print(f"Migration Duration: {duration.total_seconds():.2f} seconds")
    print(f"Start Time: {migration_stats['start_time'].strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"End Time: {migration_stats['end_time'].strftime('%Y-%m-%d %H:%M:%S')}")
    print()

    print("MIGRATION RESULTS:")
    print(f"  Embeddings found for migration: {migration_stats['total_found']}")
    print(f"  Successfully migrated: {migration_stats['successfully_migrated']}")
    print(f"  Errors encountered: {migration_stats['errors']}")
    print(f"  Already migrated: {migration_stats['already_migrated']}")

    if migration_stats['total_found'] > 0:
        success_rate = (migration_stats['successfully_migrated'] / migration_stats['total_found']) * 100
        print(f"  Success rate: {success_rate:.1f}%")

    print()
    print("VERIFICATION RESULTS:")
    print(f"  Total embeddings: {verification_results['total_embeddings']}")
    print(f"  pgvector format: {verification_results['pgvector_only'] + verification_results['both_formats']}")
    print(f"  Legacy format only: {verification_results['legacy_only']}")
    print(f"  Verification status: {'PASSED' if verification_results['verification_passed'] else 'FAILED'}")

    print("\n" + "=" * 60)

    if verification_results['verification_passed'] and migration_stats['errors'] == 0:
        print("MIGRATION COMPLETED SUCCESSFULLY!")
        print("Your embeddings are now using pgvector for better performance.")
    elif verification_results['verification_passed']:
        print("MIGRATION COMPLETED WITH WARNINGS")
        print("Most embeddings migrated successfully, but some errors occurred.")
    else:
        print("MIGRATION NEEDS ATTENTION")
        print("Some embeddings still need to be migrated.")

    print("=" * 60)


def main():
    """Main function to run the migration process."""
    # Set up request tracking for this migration run
    request_id = set_request_id()

    info_id("Starting pgvector migration process", request_id)

    try:
        # Step 1: Perform the migration
        migration_stats = migrate_existing_embeddings(request_id)

        # Step 2: Verify the migration
        verification_results = verify_migration(request_id)

        # Step 3: Create pgvector indexes for performance
        indexes_created = create_pgvector_indexes(request_id)

        # Step 4: Print comprehensive summary
        print_migration_summary(migration_stats, verification_results)

        # Step 5: Final recommendations
        if verification_results['verification_passed']:
            info_id("Migration process completed successfully!", request_id)

            if indexes_created:
                info_id("Recommendation: You can now use pgvector similarity search functions for better performance",
                        request_id)
            else:
                warning_id("Recommendation: Manually create pgvector indexes for optimal performance", request_id)

            # Optional: Suggest cleaning up legacy data
            if verification_results['both_formats'] > 0:
                info_id(f"Optional: {verification_results['both_formats']} embeddings have both formats. " +
                        "You can clean up legacy data after confirming everything works correctly.", request_id)
        else:
            error_id("Migration incomplete. Please check the logs and re-run the migration.", request_id)
            return 1

        return 0

    except Exception as e:
        error_id(f"Migration process failed: {str(e)}", request_id, exc_info=True)
        print(f"\nERROR: Migration failed - {str(e)}")
        print("Check the logs for detailed error information.")
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)