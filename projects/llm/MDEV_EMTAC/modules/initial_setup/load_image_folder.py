import os
import sys
import time
import random
import shutil
import pandas as pd
import numpy as np
from PIL import Image as PILImage, ImageFile
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta
from sqlalchemy import func, text
from collections import defaultdict

# Import the new PostgreSQL framework components
from modules.configuration.log_config import (
    logger, with_request_id, get_request_id, set_request_id,
    info_id, debug_id, warning_id, error_id, log_timed_operation
)
from modules.configuration.config import DATABASE_PATH_IMAGES_FOLDER, ALLOWED_EXTENSIONS, DATABASE_DIR
from modules.initial_setup.initializer_logger import (
    LOG_DIRECTORY, initializer_logger, close_initializer_logger
)
from modules.configuration.config_env import DatabaseConfig
from modules.emtacdb.emtacdb_fts import Image, ImageEmbedding
from plugins.ai_modules.ai_models import ModelsConfig

# Make sure truncated images won't crash PIL
ImageFile.LOAD_TRUNCATED_IMAGES = True


class OptimizedImageFolderProcessor:
    """High-performance image folder processor using vectorized operations and bulk database inserts."""

    def __init__(self):
        self.request_id = set_request_id()
        self.db_config = DatabaseConfig()
        info_id("Initialized Optimized Image Folder Processor", self.request_id)

        # Statistics tracking
        self.stats = {
            'total_files_found': 0,
            'files_processed': 0,
            'files_skipped': 0,
            'files_errored': 0,
            'duplicates_found': 0,
            'new_images_added': 0,
            'processing_time': 0,
            'average_time_per_file': 0
        }

        # Initialize model configuration
        self._initialize_models()

    def _initialize_models(self):
        """Initialize AI models configuration."""
        try:
            from plugins.ai_modules.ai_models import initialize_models_config
            if initialize_models_config():
                info_id("AI models configuration initialized successfully", self.request_id)
            else:
                warning_id("AI models configuration initialization had issues", self.request_id)
        except Exception as e:
            error_id(f"Failed to initialize AI models configuration: {str(e)}", self.request_id)

    def check_existing_images(self, session):
        """Check for existing images in database using optimized query."""
        try:
            info_id("Checking existing images in database", self.request_id)

            # Use optimized count query
            if self.db_config.is_postgresql:
                # PostgreSQL optimized count
                result = session.execute(text("SELECT COUNT(*) FROM image")).scalar()
                image_count = result if result else 0
            else:
                # SQLite fallback
                image_count = session.query(Image).count()

            if image_count > 0:
                info_id(f"Found {image_count:,} existing images in database", self.request_id)
                proceed = input("Continue with image import? (y/n): ").strip().lower()
                if proceed not in ['y', 'yes']:
                    info_id("User chose to skip image import due to existing data", self.request_id)
                    return False

            return True

        except Exception as e:
            error_id(f"Error checking existing images: {str(e)}", self.request_id)
            raise

    def scan_for_images_vectorized(self, folder_path, recursive=True):
        """Optimized image scanning using vectorized operations."""
        info_id(f"Scanning for images in: {folder_path} (recursive={recursive})", self.request_id)

        try:
            with log_timed_operation("scan_for_images", self.request_id):
                # Get all files at once
                all_files = []

                if recursive:
                    # Use os.walk for recursive scanning
                    for root, dirs, files in os.walk(folder_path):
                        for filename in files:
                            full_path = os.path.join(root, filename)
                            all_files.append((root, filename, full_path))
                else:
                    # Non-recursive scanning
                    if os.path.exists(folder_path):
                        for filename in os.listdir(folder_path):
                            full_path = os.path.join(folder_path, filename)
                            if os.path.isfile(full_path):
                                all_files.append((folder_path, filename, full_path))

                # Convert to DataFrame for vectorized filtering
                if all_files:
                    df = pd.DataFrame(all_files, columns=['root', 'filename', 'full_path'])

                    # Vectorized extension filtering
                    extensions = [ext.lower() for ext in ALLOWED_EXTENSIONS]
                    df['extension'] = df['filename'].str.lower().str.extract(r'\.([^.]+)$')[0]

                    # Filter for valid image files
                    valid_mask = df['extension'].isin(extensions) & df['full_path'].apply(os.path.isfile)
                    image_files_df = df[valid_mask]

                    # Convert back to list format
                    image_files = image_files_df[['root', 'filename', 'full_path']].values.tolist()
                else:
                    image_files = []

                info_id(f"Found {len(image_files)} image files", self.request_id)
                return image_files

        except Exception as e:
            error_id(f"Error scanning for images: {str(e)}", self.request_id)
            raise

    def get_existing_image_titles_fast(self, session):
        """Get all existing image titles using optimized bulk query."""
        try:
            with log_timed_operation("get_existing_titles", self.request_id):
                if self.db_config.is_postgresql:
                    # Use pandas for fast bulk retrieval
                    query = text("SELECT title FROM image WHERE title IS NOT NULL")
                    existing_df = pd.read_sql(query, session.bind)
                    existing_set = set(existing_df['title'].tolist()) if not existing_df.empty else set()
                else:
                    # SQLite fallback
                    existing_titles = session.query(Image.title).filter(Image.title.isnot(None)).all()
                    existing_set = {title[0] for title in existing_titles}

                info_id(f"Retrieved {len(existing_set)} existing image titles", self.request_id)
                return existing_set

        except Exception as e:
            error_id(f"Error getting existing titles: {str(e)}", self.request_id)
            return set()

    def prepare_image_data_vectorized(self, image_files, existing_titles):
        """Prepare image data using vectorized operations."""
        info_id("Preparing image data using vectorized operations", self.request_id)

        try:
            with log_timed_operation("prepare_image_data", self.request_id):
                if not image_files:
                    return [], []

                # Convert to DataFrame for vectorized processing
                df = pd.DataFrame(image_files, columns=['root', 'filename', 'full_path'])

                # Extract titles (base names without extensions)
                df['title'] = df['filename'].apply(lambda x: os.path.splitext(x)[0])

                # Clean titles using vectorized operations
                df['clean_title'] = df['title'].str.replace(r'[^\w\s-]', '', regex=True).str.strip()

                # Filter out duplicates using vectorized operations
                initial_count = len(df)

                # Remove internal duplicates (keep last)
                df_dedupe = df.drop_duplicates(subset=['clean_title'], keep='last')
                internal_dupes = initial_count - len(df_dedupe)

                # Filter out existing titles
                if existing_titles:
                    new_images_mask = ~df_dedupe['clean_title'].isin(existing_titles)
                    df_new = df_dedupe[new_images_mask]
                    db_dupes = len(df_dedupe) - len(df_new)
                else:
                    df_new = df_dedupe
                    db_dupes = 0

                # Prepare data for bulk insert
                new_image_data = []
                failed_copies = []

                if not df_new.empty:
                    info_id(f"Processing {len(df_new):,} new images", self.request_id)

                    # Create destination directory
                    dest_dir = os.path.join(DATABASE_DIR, "DB_IMAGES")
                    os.makedirs(dest_dir, exist_ok=True)

                    # Process files in batches for copying
                    batch_size = 100
                    for i in range(0, len(df_new), batch_size):
                        batch = df_new.iloc[i:i + batch_size]

                        for _, row in batch.iterrows():
                            try:
                                # Prepare destination path
                                _, ext = os.path.splitext(row['filename'])
                                dest_name = f"{row['clean_title']}{ext}"
                                dest_rel = os.path.join("DB_IMAGES", dest_name)
                                dest_abs = os.path.join(DATABASE_DIR, dest_rel)

                                # Copy file
                                shutil.copy2(row['full_path'], dest_abs)

                                # Prepare data for bulk insert
                                new_image_data.append({
                                    'title': row['clean_title'],
                                    'description': 'Auto-imported image',
                                    'file_path': dest_rel
                                })

                            except Exception as e:
                                error_id(f"Failed to copy {row['filename']}: {str(e)}", self.request_id)
                                failed_copies.append((row['filename'], str(e)))

                        # Progress update
                        if i + batch_size < len(df_new):
                            info_id(f"Copied {i + batch_size:,}/{len(df_new):,} files", self.request_id)

                # Update statistics
                self.stats['duplicates_found'] = internal_dupes + db_dupes
                self.stats['new_images_added'] = len(new_image_data)

                info_id(f"Prepared {len(new_image_data)} new images for database insertion", self.request_id)
                info_id(f"Internal duplicates: {internal_dupes:,}", self.request_id)
                info_id(f"Database duplicates: {db_dupes:,}", self.request_id)
                if failed_copies:
                    info_id(f"Copy failures: {len(failed_copies):,}", self.request_id)

                return new_image_data, failed_copies

        except Exception as e:
            error_id(f"Error preparing image data: {str(e)}", self.request_id)
            raise

    def bulk_insert_images_optimized(self, session, new_image_data):
        """Perform optimized bulk insertion of images."""
        if not new_image_data:
            info_id("No new images to insert", self.request_id)
            return []

        try:
            info_id(f"Bulk inserting {len(new_image_data)} images", self.request_id)

            with log_timed_operation("bulk_insert_images", self.request_id):
                # Use optimized bulk insert
                session.bulk_insert_mappings(Image, new_image_data)
                session.commit()

                # Get IDs of newly inserted images for embedding processing
                titles = [img['title'] for img in new_image_data]

                if self.db_config.is_postgresql:
                    # PostgreSQL optimized ID retrieval
                    query = text("""
                        SELECT id, title, file_path 
                        FROM image 
                        WHERE title = ANY(:titles)
                        ORDER BY id DESC
                    """)
                    ids_df = pd.read_sql(query, session.bind, params={'titles': titles})
                    new_image_ids = ids_df[['id', 'title', 'file_path']].to_dict('records')
                else:
                    # SQLite fallback
                    newly_inserted = session.query(Image.id, Image.title, Image.file_path).filter(
                        Image.title.in_(titles)
                    ).all()
                    new_image_ids = [{'id': img.id, 'title': img.title, 'file_path': img.file_path}
                                     for img in newly_inserted]

                info_id(f"Successfully inserted {len(new_image_data)} images", self.request_id)
                return new_image_ids

        except Exception as e:
            session.rollback()
            error_id(f"Error in bulk insert: {str(e)}", self.request_id)
            raise

    def bulk_generate_embeddings_optimized(self, session, new_image_ids):
        """Generate embeddings in optimized batches."""
        if not new_image_ids:
            return

        try:
            info_id("Starting optimized embedding generation", self.request_id)

            # Load the model once
            model_handler = ModelsConfig.load_image_model()
            model_name = type(model_handler).__name__

            # Skip embedding generation for NoImageModel
            if model_name == 'NoImageModel':
                info_id("Skipping embedding generation for NoImageModel", self.request_id)
                return

            with log_timed_operation("bulk_generate_embeddings", self.request_id):
                embeddings_data = []
                processed_count = 0

                # Process in batches to manage memory
                batch_size = 50
                for i in range(0, len(new_image_ids), batch_size):
                    batch = new_image_ids[i:i + batch_size]

                    for img_data in batch:
                        try:
                            # Construct absolute path
                            if os.path.isabs(img_data['file_path']):
                                abs_path = img_data['file_path']
                            else:
                                abs_path = os.path.join(DATABASE_DIR, img_data['file_path'])

                            # Generate embedding
                            image = PILImage.open(abs_path).convert("RGB")

                            if model_handler.is_valid_image(image):
                                embedding = model_handler.get_image_embedding(image)

                                if embedding is not None:
                                    embeddings_data.append({
                                        'image_id': img_data['id'],
                                        'model_name': model_name,
                                        'model_embedding': embedding.tobytes()
                                    })

                            processed_count += 1

                            # Progress update
                            if processed_count % 25 == 0:
                                info_id(f"Generated {processed_count:,}/{len(new_image_ids):,} embeddings", self.request_id)

                        except Exception as e:
                            error_id(f"Error generating embedding for {img_data['title']}: {str(e)}", self.request_id)
                            continue

                # Bulk insert embeddings
                if embeddings_data:
                    session.bulk_insert_mappings(ImageEmbedding, embeddings_data)
                    session.commit()

                    info_id(f"Generated {len(embeddings_data)} embeddings", self.request_id)
                else:
                    info_id("No embeddings generated", self.request_id)

        except Exception as e:
            error_id(f"Error generating embeddings: {str(e)}", self.request_id)

    def process_folder_optimized(self, folder_path, recursive=True):
        """Main optimized method to process an image folder."""
        try:
            info_id(f"Starting optimized image folder processing for: {folder_path} (recursive={'Yes' if recursive else 'No'})", self.request_id)

            # Validate folder
            if not os.path.exists(folder_path):
                raise ValueError(f"Folder does not exist: {folder_path}")

            if not os.path.isdir(folder_path):
                raise ValueError(f"Path is not a directory: {folder_path}")

            start_time = time.time()

            # Get database session
            with self.db_config.main_session() as session:
                # Check existing images
                if not self.check_existing_images(session):
                    return False

                # Scan for images using optimized method
                image_files = self.scan_for_images_vectorized(folder_path, recursive)

                if not image_files:
                    info_id("No image files found to process", self.request_id)
                    return True

                self.stats['total_files_found'] = len(image_files)

                # Get existing titles in one optimized query
                existing_titles = self.get_existing_image_titles_fast(session)

                # Prepare image data using vectorized operations
                new_image_data, failed_copies = self.prepare_image_data_vectorized(image_files, existing_titles)

                # Bulk insert images
                new_image_ids = self.bulk_insert_images_optimized(session, new_image_data)

                # Generate embeddings in batches
                self.bulk_generate_embeddings_optimized(session, new_image_ids)

            # Update final statistics
            self.stats['processing_time'] = time.time() - start_time
            self.stats['files_processed'] = len(new_image_data)
            self.stats['files_errored'] = len(failed_copies)
            self.stats['average_time_per_file'] = (
                self.stats['processing_time'] / self.stats['total_files_found']
                if self.stats['total_files_found'] > 0 else 0
            )

            # Log summary
            self.display_optimized_summary()

            return True

        except Exception as e:
            error_id(f"Error in optimized folder processing: {str(e)}", self.request_id, exc_info=True)
            return False

    def display_optimized_summary(self):
        """Log comprehensive processing summary."""
        info_id("Optimized image processing completed", self.request_id)
        info_id(f"Total files found: {self.stats['total_files_found']:,}", self.request_id)
        info_id(f"Successfully processed: {self.stats['new_images_added']:,}", self.request_id)
        info_id(f"Duplicates skipped: {self.stats['duplicates_found']:,}", self.request_id)
        info_id(f"Files with errors: {self.stats['files_errored']:,}", self.request_id)
        info_id(f"Total processing time: {self._format_time(self.stats['processing_time'])}", self.request_id)

        if self.stats['processing_time'] > 0:
            rate = self.stats['total_files_found'] / self.stats['processing_time']
            info_id(f"Processing rate: {rate:.1f} files/sec", self.request_id)

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

    def setup_ai_model(self):
        """Setup AI model configuration."""
        try:
            info_id("Setting up AI model configuration", self.request_id)

            while True:
                choice = input("Select option (1-3): ").strip()

                if choice == "1":
                    selected_model = "CLIPModelHandler"
                    break
                elif choice == "2":
                    selected_model = "NoImageModel"
                    break
                elif choice == "3":
                    model_name = input("Enter custom model name: ").strip()
                    if model_name:
                        selected_model = model_name
                        break
                    else:
                        warning_id("Invalid model name provided", self.request_id)
                else:
                    warning_id("Invalid choice. Please select 1, 2, or 3.", self.request_id)

            # Set the model
            success = ModelsConfig.set_current_image_model(selected_model)

            if success:
                info_id(f"Successfully set image model to: {selected_model}", self.request_id)
            else:
                error_id(f"Failed to set image model to: {selected_model}", self.request_id)
                warning_id("Falling back to NoImageModel", self.request_id)
                ModelsConfig.set_current_image_model("NoImageModel")

        except Exception as e:
            error_id(f"Error setting up AI model: {str(e)}", self.request_id)
            raise


def main():
    """
    Main function for optimized image processing.
    """
    info_id("Starting optimized image folder processing", request_id=None)

    processor = None
    try:
        # Initialize optimized processor
        processor = OptimizedImageFolderProcessor()

        # Setup AI model
        processor.setup_ai_model()

        # Get folder paths
        folders = []
        if len(sys.argv) > 1:
            folders = sys.argv[1:]
            info_id(f"Using command line folders: {folders}", processor.request_id)
        else:
            while True:
                folder_path = input("Folder path: ").strip().strip('"').strip("'")
                if not folder_path:
                    break

                if not os.path.isdir(folder_path):
                    warning_id(f"Invalid directory: {folder_path}", processor.request_id)
                    continue

                folders.append(folder_path)
                info_id(f"Added folder: {folder_path}", processor.request_id)

        if not folders:
            warning_id("No valid folders provided", processor.request_id)
            return

        # Processing options
        recursive = input("Process subfolders recursively? (y/n, default: y): ").strip().lower() != 'n'
        info_id(f"Recursive processing: {'Yes' if recursive else 'No'}", processor.request_id)

        # Process each folder with optimized method
        success_count = 0
        total_start_time = time.time()

        for i, folder in enumerate(folders, 1):
            info_id(f"Processing folder {i}/{len(folders)}: {os.path.basename(folder)}", processor.request_id)

            try:
                if processor.process_folder_optimized(folder, recursive=recursive):
                    success_count += 1
                    info_id(f"Folder {i} completed successfully", processor.request_id)
                else:
                    warning_id(f"Folder {i} completed with issues", processor.request_id)
            except Exception as e:
                error_id(f"Error processing folder {folder}: {str(e)}", processor.request_id)

        # Final summary
        total_time = time.time() - total_start_time
        info_id("All folders processed", processor.request_id)
        info_id(f"Successful: {success_count}/{len(folders)}", processor.request_id)
        info_id(f"Total time: {processor._format_time(total_time)}", processor.request_id)
        if success_count < len(folders):
            warning_id(f"Issues: {len(folders) - success_count}/{len(folders)}", processor.request_id)

        info_id("Optimized image folder processing completed", processor.request_id)

    except KeyboardInterrupt:
        error_id("Processing interrupted by user", processor.request_id if processor else None)
    except Exception as e:
        error_id(f"Processing failed: {str(e)}", processor.request_id if processor else None, exc_info=True)
    finally:
        # Cleanup
        try:
            close_initializer_logger()
        except:
            pass


if __name__ == "__main__":
    main()