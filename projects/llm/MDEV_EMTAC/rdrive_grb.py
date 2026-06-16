from __future__ import annotations

import os
import shutil
import logging
from pathlib import Path
from datetime import datetime

# -----------------------------------------------------------------------------
# CONFIG
# -----------------------------------------------------------------------------
SOURCE_DIR = r"\\tsclient\R\Public-AUS\Maintenance\Technical Training\TECHNICAL_TRAINING_COURSES"
DEST_DIR = r"E:\emtac\data\TECHNICAL_TRAINING_COURSES"

LOG_DIR = r"E:\emtac\logs"
os.makedirs(LOG_DIR, exist_ok=True)

LOG_FILE = os.path.join(
    LOG_DIR,
    f"copy_training_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
)

# -----------------------------------------------------------------------------
# LOGGING SETUP
# -----------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE, encoding="utf-8"),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)


# -----------------------------------------------------------------------------
# HELPERS
# -----------------------------------------------------------------------------
def ensure_directory(path: Path):
    if not path.exists():
        path.mkdir(parents=True, exist_ok=True)


def files_are_different(src: Path, dst: Path) -> bool:
    """
    Compare file size and modified time to decide if copy is needed.
    """
    if not dst.exists():
        return True

    return not (
        src.stat().st_size == dst.stat().st_size and
        int(src.stat().st_mtime) == int(dst.stat().st_mtime)
    )


# -----------------------------------------------------------------------------
# MAIN COPY FUNCTION
# -----------------------------------------------------------------------------
def copy_directory(src_root: Path, dst_root: Path):
    total_files = 0
    copied_files = 0
    skipped_files = 0
    errors = 0

    logger.info(f"Starting copy...")
    logger.info(f"Source: {src_root}")
    logger.info(f"Destination: {dst_root}")

    for root, dirs, files in os.walk(src_root):
        root_path = Path(root)

        # Build destination path
        relative_path = root_path.relative_to(src_root)
        target_root = dst_root / relative_path

        ensure_directory(target_root)

        # Copy files
        for file_name in files:
            total_files += 1

            src_file = root_path / file_name
            dst_file = target_root / file_name

            try:
                if files_are_different(src_file, dst_file):
                    shutil.copy2(src_file, dst_file)
                    copied_files += 1

                    if copied_files % 100 == 0:
                        logger.info(f"Copied {copied_files} files so far...")

                else:
                    skipped_files += 1

            except Exception as e:
                errors += 1
                logger.error(f"ERROR copying {src_file} -> {dst_file}: {e}")

    logger.info("=" * 60)
    logger.info("COPY COMPLETE")
    logger.info(f"Total files scanned : {total_files}")
    logger.info(f"Files copied        : {copied_files}")
    logger.info(f"Files skipped       : {skipped_files}")
    logger.info(f"Errors              : {errors}")
    logger.info("=" * 60)


# -----------------------------------------------------------------------------
# ENTRY POINT
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    src = Path(SOURCE_DIR)
    dst = Path(DEST_DIR)

    if not src.exists():
        logger.error(f"Source path does not exist: {src}")
        exit(1)

    ensure_directory(dst)

    copy_directory(src, dst)