#!/usr/bin/env python3
"""
Migrate existing image embeddings from legacy to pgvector format

This script migrates existing ImageEmbedding records from the legacy LargeBinary
format to the new pgvector format for better performance and native vector operations.
"""

import sys
import os
import numpy as np
from datetime import datetime

# Add the project root to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from modules.configuration.config_env import DatabaseConfig
from modules.configuration.log_config import (
    debug_id, info_id, warning_id, error_id,
    set_request_id, get_request_id, with_request_id, log_timed_operation
)
from modules.emtacdb.emtacdb_fts import ImageEmbedding


@with_request_id
def migrate_image_embeddings(request_id=None):
    """
    Migrate existing image embeddings from legacy LargeBinary to pgvector format.

    Args:
        request_id: Optional request ID for logging tracking

    Returns:
        dict: Migration results with counts and statistics
    """
    info_id("Starting migration of existing image embeddings to pgvector format", request_id)

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
        with log_timed_operation("image_embedding_migration", request_id):
            with db_config.main_session() as session:

                # Get count of all image embeddings for statistics
                total_embeddings = session.query(ImageEmbedding).count()
                info_id(f"Total image embeddings in database: {total_embeddings}", request_id)

                # Get embeddings that only have legacy data (need migration)
                legacy_embeddings = session.query(ImageEmbedding).filter(
                    ImageEmbedding.model_embedding.isnot(None),
                    ImageEmbedding.embedding_vector.is_(None)
                ).all()

                migration_stats['total_found'] = len(legacy_embeddings)
                info_id(f"Found {len(legacy_embeddings)} legacy image embeddings to migrate", request_id)

                # Check how many are already migrated
                already_migrated = session.query(ImageEmbedding).filter(
                    ImageEmbedding.embedding_vector.isnot(None)
                ).count()

                migration_stats['already_migrated'] = already_migrated
                info_id(f"Found {already_migrated} image embeddings already in pgvector format", request_id)

                if len(legacy_embeddings) == 0:
                    info_id("No image embeddings need migration. All are already in pgvector format or missing.",
                            request_id)
                    migration_stats['end_time'] = datetime.now()
                    return migration_stats

                # Process embeddings in batches for better performance
                batch_size = 100
                migrated_count = 0
                error_count = 0

                info_id(f"Starting image embedding migration in batches of {batch_size}", request_id)

                for i, embedding in enumerate(legacy_embeddings):
                    try:
                        # Use the built-in migration method
                        if embedding.migrate_to_pgvector():
                            migrated_count += 1
                            debug_id(f"Migrated image embedding {embedding.id} for image {embedding.image_id}",
                                     request_id)

                            # Commit periodically to avoid large transactions
                            if migrated_count % batch_size == 0:
                                session.commit()
                                info_id(
                                    f"Committed batch: {migrated_count}/{len(legacy_embeddings)} image embeddings migrated",
                                    request_id)
                        else:
                            error_count += 1
                            warning_id(
                                f"Failed to migrate image embedding {embedding.id} - migrate_to_pgvector returned False",
                                request_id)

                    except Exception as e:
                        error_count += 1
                        error_id(f"Error migrating image embedding {embedding.id}: {str(e)}", request_id, exc_info=True)
                        continue

                # Final commit for any remaining changes
                try:
                    session.commit()
                    info_id("Final commit completed for image embeddings", request_id)
                except Exception as e:
                    session.rollback()
                    error_id(f"Final commit failed for image embeddings, rolling back: {str(e)}", request_id,
                             exc_info=True)
                    raise

                migration_stats['successfully_migrated'] = migrated_count
                migration_stats['errors'] = error_count
                migration_stats['end_time'] = datetime.now()

                # Log final statistics
                duration = migration_stats['end_time'] - migration_stats['start_time']
                info_id(f"Image embedding migration completed in {duration.total_seconds():.2f} seconds", request_id)
                info_id(f"Successfully migrated: {migrated_count}", request_id)
                info_id(f"Errors encountered: {error_count}", request_id)

                if error_count > 0:
                    warning_id(
                        f"Image embedding migration completed with {error_count} errors. Check logs for details.",
                        request_id)
                else:
                    info_id("Image embedding migration completed successfully with no errors!", request_id)

                return migration_stats

    except Exception as e:
        migration_stats['end_time'] = datetime.now()
        error_id(f"Critical error during image embedding migration: {str(e)}", request_id, exc_info=True)
        raise


def verify_image_migration(request_id=None):
    """
    Verify that the image embedding migration was successful.

    Args:
        request_id: Optional request ID for logging tracking

    Returns:
        dict: Verification results
    """
    info_id("Starting image embedding migration verification", request_id)

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
            all_embeddings = session.query(ImageEmbedding).all()
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

            info_id(f"Image embedding verification results:", request_id)
            info_id(f"  Total embeddings: {verification_results['total_embeddings']}", request_id)
            info_id(f"  Legacy only: {verification_results['legacy_only']}", request_id)
            info_id(f"  pgvector only: {verification_results['pgvector_only']}", request_id)
            info_id(f"  Both formats: {verification_results['both_formats']}", request_id)
            info_id(f"  No data: {verification_results['no_data']}", request_id)

            if verification_results['verification_passed']:
                info_id("Image embedding verification PASSED: All embeddings have pgvector data", request_id)
            else:
                warning_id(
                    f"Image embedding verification FAILED: {verification_results['legacy_only']} embeddings still only have legacy data",
                    request_id)

            return verification_results

    except Exception as e:
        error_id(f"Error during image embedding verification: {str(e)}", request_id, exc_info=True)
        return verification_results


def create_image_pgvector_indexes(request_id=None):
    """
    Create pgvector indexes for better image similarity search performance.

    Args:
        request_id: Optional request ID for logging tracking

    Returns:
        bool: True if indexes created successfully
    """
    info_id("Creating pgvector indexes for optimal image similarity search performance", request_id)

    db_config = DatabaseConfig()

    try:
        with db_config.main_session() as session:
            success = ImageEmbedding.create_pgvector_indexes(session)

            if success:
                info_id("pgvector indexes created successfully for image embeddings", request_id)
            else:
                warning_id("Some issues occurred while creating pgvector indexes for image embeddings", request_id)

            return success

    except Exception as e:
        error_id(f"Error creating pgvector indexes for image embeddings: {str(e)}", request_id, exc_info=True)
        return False


def verify_image_similarity_search(request_id=None):
    """
    Test the image similarity search functionality.

    Args:
        request_id: Optional request ID for logging tracking
    """
    info_id("Testing image similarity search functionality", request_id)

    db_config = DatabaseConfig()

    try:
        with db_config.main_session() as session:
            # Get a random image embedding for testing
            sample_embedding = session.query(ImageEmbedding).filter(
                ImageEmbedding.embedding_vector.isnot(None)
            ).first()

            if not sample_embedding:
                warning_id("No pgvector image embeddings found for testing", request_id)
                return

            # Test similarity search
            query_embedding = sample_embedding.embedding_as_list

            results = ImageEmbedding.search_similar_images(
                session,
                query_embedding,
                limit=5
            )

            info_id(f"Similarity search test completed. Found {len(results)} similar images", request_id)

            for i, result in enumerate(results[:3]):
                info_id(
                    f"  {i + 1}. Image {result['image_id']}: {result['image_title']} (similarity: {result['similarity']:.4f})",
                    request_id)

    except Exception as e:
        error_id(f"Error testing image similarity search: {str(e)}", request_id, exc_info=True)


def print_image_migration_summary(migration_stats, verification_results):
    """
    Print a comprehensive summary of the image embedding migration process.
    """
    print("\n" + "=" * 60)
    print("IMAGE EMBEDDING PGVECTOR MIGRATION SUMMARY")
    print("=" * 60)

    duration = migration_stats['end_time'] - migration_stats['start_time']

    print(f"Migration Duration: {duration.total_seconds():.2f} seconds")
    print(f"Start Time: {migration_stats['start_time'].strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"End Time: {migration_stats['end_time'].strftime('%Y-%m-%d %H:%M:%S')}")
    print()

    print("MIGRATION RESULTS:")
    print(f"  Image embeddings found for migration: {migration_stats['total_found']}")
    print(f"  Successfully migrated: {migration_stats['successfully_migrated']}")
    print(f"  Errors encountered: {migration_stats['errors']}")
    print(f"  Already migrated: {migration_stats['already_migrated']}")

    if migration_stats['total_found'] > 0:
        success_rate = (migration_stats['successfully_migrated'] / migration_stats['total_found']) * 100
        print(f"  Success rate: {success_rate:.1f}%")

    print()
    print("VERIFICATION RESULTS:")
    print(f"  Total image embeddings: {verification_results['total_embeddings']}")
    print(f"  pgvector format: {verification_results['pgvector_only'] + verification_results['both_formats']}")
    print(f"  Legacy format only: {verification_results['legacy_only']}")
    print(f"  Verification status: {'PASSED' if verification_results['verification_passed'] else 'FAILED'}")

    print("\n" + "=" * 60)

    if verification_results['verification_passed'] and migration_stats['errors'] == 0:
        print("IMAGE EMBEDDING MIGRATION COMPLETED SUCCESSFULLY!")
        print("Your image embeddings are now using pgvector for better performance.")
        print("You can now perform fast image similarity searches!")
    elif verification_results['verification_passed']:
        print("IMAGE EMBEDDING MIGRATION COMPLETED WITH WARNINGS")
        print("Most embeddings migrated successfully, but some errors occurred.")
    else:
        print("IMAGE EMBEDDING MIGRATION NEEDS ATTENTION")
        print("Some embeddings still need to be migrated.")

    print("=" * 60)


def main():
    """Main function to run the image embedding migration process."""
    # Set up request tracking for this migration run
    request_id = set_request_id()

    info_id("Starting pgvector image embedding migration process", request_id)

    try:
        # Step 1: Perform the migration
        migration_stats = migrate_image_embeddings(request_id)

        # Step 2: Verify the migration
        verification_results = verify_image_migration(request_id)

        # Step 3: Create pgvector indexes for performance
        indexes_created = create_image_pgvector_indexes(request_id)

        # Step 4: Test similarity search
        verify_image_similarity_search(request_id)

        # Step 5: Print comprehensive summary
        print_image_migration_summary(migration_stats, verification_results)

        # Step 6: Final recommendations
        if verification_results['verification_passed']:
            info_id("Image embedding migration process completed successfully!", request_id)

            if indexes_created:
                info_id("You can now search for similar images using ImageEmbedding.search_similar_images()",
                        request_id)
            else:
                warning_id("Recommendation: Manually create pgvector indexes for optimal performance", request_id)

            # Optional: Suggest cleaning up legacy data
            if verification_results['both_formats'] > 0:
                info_id(f"Optional: {verification_results['both_formats']} image embeddings have both formats. " +
                        "You can clean up legacy data after confirming everything works correctly.", request_id)
        else:
            error_id("Image embedding migration incomplete. Please check the logs and re-run the migration.",
                     request_id)
            return 1

        return 0

    except Exception as e:
        error_id(f"Image embedding migration process failed: {str(e)}", request_id, exc_info=True)
        print(f"\nERROR: Image embedding migration failed - {str(e)}")
        print("Check the logs for detailed error information.")
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)