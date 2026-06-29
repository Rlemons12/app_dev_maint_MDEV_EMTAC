from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

# -------------------------------------------------------------------
# Base path (adjust ONLY if needed)
# -------------------------------------------------------------------
BASE_PACKAGE_PATH = Path(r'C:\Users\10169062\PycharmProjects\MDEV_EMTAC\accelerator')

# -------------------------------------------------------------------
# Directory structure
# -------------------------------------------------------------------
DIRECTORIES = [
    BASE_PACKAGE_PATH,
    BASE_PACKAGE_PATH / "backends",
]

# -------------------------------------------------------------------
# Files to create (empty)
# -------------------------------------------------------------------
FILES = [
    BASE_PACKAGE_PATH / "__init__.py",
    BASE_PACKAGE_PATH / "manager.py",
    BASE_PACKAGE_PATH / "detection.py",
    BASE_PACKAGE_PATH / "config.py",
    BASE_PACKAGE_PATH / "diagnostics.py",
    BASE_PACKAGE_PATH / "policies.py",
    BASE_PACKAGE_PATH / "base.py",
    BASE_PACKAGE_PATH / "backends" / "__init__.py",
    BASE_PACKAGE_PATH / "backends" / "cpu.py",
    BASE_PACKAGE_PATH / "backends" / "cuda.py",
    BASE_PACKAGE_PATH / "backends" / "mps.py",
    BASE_PACKAGE_PATH / "backends" / "rocm.py",
    BASE_PACKAGE_PATH / "backends" / "xpu.py",
]

# -------------------------------------------------------------------
# Create directories
# -------------------------------------------------------------------
for directory in DIRECTORIES:
    if not directory.exists():
        directory.mkdir(parents=True, exist_ok=True)
        logger.info(f"Created directory: {directory}")
    else:
        logger.info(f"Directory exists: {directory}")

# -------------------------------------------------------------------
# Create files (no overwrite)
# -------------------------------------------------------------------
for file_path in FILES:
    if not file_path.exists():
        file_path.touch()
        logger.info(f"Created file: {file_path}")
    else:
        logger.info(f"File exists: {file_path}")

logger.info("Accelerator package scaffold complete.")
