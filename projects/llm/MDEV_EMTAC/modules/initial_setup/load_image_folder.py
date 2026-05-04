import os
import sys
import mimetypes
from dataclasses import dataclass
from typing import List, Dict, Any

from modules.configuration.log_config import (
    set_request_id,
    info_id,
    warning_id,
    error_id,
)
from modules.initial_setup.initializer_logger import close_initializer_logger
from modules.coordinators.image_processing_coordinator import ImageProcessingCoordinator


@dataclass
class LocalImageUploadFile:
    """
    Adapter that makes a local disk file look enough like an uploaded file
    for the coordinator/orchestrator pipeline.
    """
    path: str

    @property
    def filename(self) -> str:
        return os.path.basename(self.path)

    @property
    def content_type(self) -> str:
        guessed, _ = mimetypes.guess_type(self.path)
        return guessed or "application/octet-stream"

    def read(self) -> bytes:
        with open(self.path, "rb") as f:
            return f.read()

    def seek(self, offset: int, whence: int = 0) -> None:
        """
        Compatibility stub.
        Some upload workflows expect a file object with seek().
        Since this adapter re-opens on read(), seek is not needed here.
        """
        return None

    def save(self, destination: str) -> None:
        """
        Compatibility helper if downstream code expects a save() method.
        """
        with open(self.path, "rb") as src, open(destination, "wb") as dst:
            dst.write(src.read())


class FolderToCoordinatorImageLoader:
    """
    Initializer script that scans local folders and routes discovered images
    into the coordinator/orchestrator image ingestion workflow.
    """

    IMAGE_EXTENSIONS = {
        ".jpg",
        ".jpeg",
        ".png",
        ".gif",
        ".bmp",
        ".tif",
        ".tiff",
        ".webp",
    }

    def __init__(self) -> None:
        self.request_id = set_request_id()
        self.coordinator = ImageProcessingCoordinator()
        self.stats = {
            "folders_seen": 0,
            "files_found": 0,
            "files_valid": 0,
            "files_invalid": 0,
            "successful_files": 0,
            "failed_files": 0,
        }

        info_id("Initialized FolderToCoordinatorImageLoader", self.request_id)

    def scan_for_images(self, folder_path: str, recursive: bool = True) -> List[str]:
        """
        Return a list of absolute local image file paths.
        """
        image_paths: List[str] = []

        if not os.path.exists(folder_path):
            raise ValueError(f"Folder does not exist: {folder_path}")

        if not os.path.isdir(folder_path):
            raise ValueError(f"Path is not a directory: {folder_path}")

        if recursive:
            for root, _, files in os.walk(folder_path):
                for filename in files:
                    ext = os.path.splitext(filename)[1].lower()
                    full_path = os.path.join(root, filename)
                    if ext in self.IMAGE_EXTENSIONS and os.path.isfile(full_path):
                        image_paths.append(full_path)
        else:
            for filename in os.listdir(folder_path):
                full_path = os.path.join(folder_path, filename)
                ext = os.path.splitext(filename)[1].lower()
                if ext in self.IMAGE_EXTENSIONS and os.path.isfile(full_path):
                    image_paths.append(full_path)

        return image_paths

    def build_file_objects(self, image_paths: List[str]) -> List[LocalImageUploadFile]:
        return [LocalImageUploadFile(path=p) for p in image_paths]

    def build_metadata_for_file(self, file_path: str) -> Dict[str, Any]:
        """
        Build minimal metadata expected by the coordinator.
        You can expand this later if you want folder names parsed into area/model/etc.
        """
        title = os.path.splitext(os.path.basename(file_path))[0]

        return {
            "title": title,
            "description": "Auto-imported image from local folder initializer",
            "area": "",
            "equipment_group": "",
            "model": "",
            "asset_number": "",
            "location": "",
            "site_location": "",
            "room_number": "Unknown",
            "department": "",
            "tags": "initializer_import",
            "priority": "normal",
        }

    def process_folder(
        self,
        folder_path: str,
        recursive: bool = True,
        concurrent: bool = False,
        max_workers: int = 4,
    ) -> bool:
        info_id(
            f"Starting folder-to-coordinator image load for: {folder_path} "
            f"(recursive={'Yes' if recursive else 'No'})",
            self.request_id,
        )

        try:
            self.stats["folders_seen"] += 1

            image_paths = self.scan_for_images(folder_path, recursive=recursive)
            self.stats["files_found"] += len(image_paths)

            if not image_paths:
                warning_id(f"No valid image files found in folder: {folder_path}", self.request_id)
                return True

            info_id(f"Discovered {len(image_paths)} image files", self.request_id)

            # Option A: process one-by-one so each file gets its own metadata/title
            # This is safest unless your orchestrator supports per-file metadata arrays.
            folder_success = True

            for image_path in image_paths:
                try:
                    file_obj = LocalImageUploadFile(path=image_path)
                    metadata = self.build_metadata_for_file(image_path)

                    success, response, status_code = self.coordinator.process_upload(
                        files=[file_obj],
                        metadata=metadata,
                        concurrent=False,
                        max_workers=max_workers,
                        request_id=self.request_id,
                    )

                    if success:
                        self.stats["files_valid"] += 1
                        self.stats["successful_files"] += 1
                        info_id(
                            f"Imported image successfully | file={file_obj.filename} "
                            f"| status_code={status_code} | status={response.get('status')}",
                            self.request_id,
                        )
                    else:
                        self.stats["files_valid"] += 1
                        self.stats["failed_files"] += 1
                        folder_success = False
                        warning_id(
                            f"Image import failed | file={file_obj.filename} "
                            f"| status_code={status_code} | response={response}",
                            self.request_id,
                        )

                except Exception as e:
                    self.stats["failed_files"] += 1
                    folder_success = False
                    error_id(
                        f"Unhandled error importing local image file '{image_path}': {e}",
                        self.request_id,
                        exc_info=True,
                    )

            return folder_success

        except Exception as e:
            error_id(
                f"Error processing folder '{folder_path}': {e}",
                self.request_id,
                exc_info=True,
            )
            return False

    def display_summary(self) -> None:
        info_id("Folder-to-coordinator image load complete", self.request_id)
        info_id(f"Folders processed: {self.stats['folders_seen']}", self.request_id)
        info_id(f"Files found: {self.stats['files_found']}", self.request_id)
        info_id(f"Files attempted: {self.stats['files_valid']}", self.request_id)
        info_id(f"Successful files: {self.stats['successful_files']}", self.request_id)
        info_id(f"Failed files: {self.stats['failed_files']}", self.request_id)


def prompt_for_folders(request_id: str) -> List[str]:
    folders: List[str] = []

    while True:
        folder_path = input("Folder path: ").strip().strip('"').strip("'")
        if not folder_path:
            break

        if not os.path.isdir(folder_path):
            warning_id(f"Invalid directory: {folder_path}", request_id)
            continue

        folders.append(folder_path)
        info_id(f"Added folder: {folder_path}", request_id)

    return folders


def main() -> None:
    loader = None

    try:
        loader = FolderToCoordinatorImageLoader()

        if len(sys.argv) > 1:
            folders = sys.argv[1:]
            info_id(f"Using command line folders: {folders}", loader.request_id)
        else:
            folders = prompt_for_folders(loader.request_id)

        if not folders:
            warning_id("No valid folders provided", loader.request_id)
            return

        recursive = input("Process subfolders recursively? (y/n, default: y): ").strip().lower() != "n"

        success_count = 0
        for i, folder in enumerate(folders, 1):
            info_id(f"Processing folder {i}/{len(folders)}: {folder}", loader.request_id)

            if loader.process_folder(folder_path=folder, recursive=recursive):
                success_count += 1
                info_id(f"Folder completed successfully: {folder}", loader.request_id)
            else:
                warning_id(f"Folder completed with issues: {folder}", loader.request_id)

        loader.display_summary()
        info_id(f"Successful folders: {success_count}/{len(folders)}", loader.request_id)

    except KeyboardInterrupt:
        error_id("Processing interrupted by user", loader.request_id if loader else None)

    except Exception as e:
        error_id(
            f"Processing failed: {e}",
            loader.request_id if loader else None,
            exc_info=True,
        )

    finally:
        try:
            close_initializer_logger()
        except Exception:
            pass


if __name__ == "__main__":
    main()