"""
Creates directory structure for the GPU FSDP training service.

Safe to run multiple times.
No external dependencies.
"""

from pathlib import Path
import logging

# --------------------------------------------------
# CONFIG
# --------------------------------------------------

# CHANGE THIS if you want a different root
BASE_DIR = Path(r"E:\emtac\services\gpu_training_service")

# --------------------------------------------------
# LOGGING
# --------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)

log = logging.getLogger("dir_creator")

# --------------------------------------------------
# DIRECTORY STRUCTURE
# --------------------------------------------------

DIRS = [
    # Root
    BASE_DIR,

    # App core
    BASE_DIR / "app",
    BASE_DIR / "app" / "api",
    BASE_DIR / "app" / "training",
    BASE_DIR / "app" / "schemas",
    BASE_DIR / "app" / "config",

    # Runtime
    BASE_DIR / "logs",
    BASE_DIR / "training_runs",
    BASE_DIR / "checkpoints",

    # Optional extras
    BASE_DIR / "scripts",
    BASE_DIR / "tests",
]

# --------------------------------------------------
# FILE PLACEHOLDERS (OPTIONAL)
# --------------------------------------------------

PLACEHOLDER_FILES = [
    BASE_DIR / "app" / "__init__.py",
    BASE_DIR / "app" / "api" / "__init__.py",
    BASE_DIR / "app" / "training" / "__init__.py",
    BASE_DIR / "app" / "schemas" / "__init__.py",
    BASE_DIR / "app" / "config" / "__init__.py",
]

# --------------------------------------------------
# MAIN
# --------------------------------------------------

def create_dirs():
    log.info(f"Creating GPU training service structure at: {BASE_DIR}")

    for d in DIRS:
        d.mkdir(parents=True, exist_ok=True)
        log.info(f"✔ Directory: {d}")

    for f in PLACEHOLDER_FILES:
        if not f.exists():
            f.touch()
            log.info(f"✔ File: {f}")
        else:
            log.info(f"• Exists: {f}")

    log.info("Directory creation complete.")


if __name__ == "__main__":
    create_dirs()
