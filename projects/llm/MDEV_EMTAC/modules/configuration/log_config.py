# modules/configuration/log_config.py

from __future__ import annotations

import functools
import gzip
import logging
import os
import shutil
import sys
import threading
import time
import unicodedata
import uuid
from contextlib import contextmanager
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional


# ---------------------------------------------------------------------
# Optional Flask imports
# ---------------------------------------------------------------------
# This logging module is used by both Flask routes and standalone scripts.
# Keeping Flask optional makes the logger safer for maintenance scripts,
# database runners, unit tests, and CLI tools.

try:
    from flask import request, g, has_request_context
except Exception:  # pragma: no cover - fallback for non-Flask script contexts
    request = None
    g = None

    def has_request_context() -> bool:
        return False


# ---------------------------------------------------------------------
# Windows-safe rotating file handler
# ---------------------------------------------------------------------
# Preferred package:
#   pip install concurrent-log-handler
#
# Python package name:
#   concurrent-log-handler
#
# Python import name:
#   concurrent_log_handler
#
# Why:
#   ConcurrentRotatingFileHandler is safer on Windows when more than one
#   process/thread may touch the same log file, such as Flask debug reloaders,
#   services, scripts, or multiple workers.
#
# If the package is not installed, this module falls back to Python's standard
# RotatingFileHandler.
# ---------------------------------------------------------------------

try:
    from concurrent_log_handler import (
        ConcurrentRotatingFileHandler as _ActiveRotatingFileHandler,
    )

    CONCURRENT_LOG_HANDLER_AVAILABLE = True
    ACTIVE_ROTATING_HANDLER_NAME = "ConcurrentRotatingFileHandler"

except ImportError:
    from logging.handlers import RotatingFileHandler as _ActiveRotatingFileHandler

    CONCURRENT_LOG_HANDLER_AVAILABLE = False
    ACTIVE_ROTATING_HANDLER_NAME = "RotatingFileHandler"


# ---------------------------------------------------------------------
# Base paths
# ---------------------------------------------------------------------

if getattr(sys, "frozen", False):  # Running as a PyInstaller executable
    BASE_DIR = os.path.dirname(sys.executable)
else:
    BASE_DIR = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "..")
    )

if BASE_DIR not in sys.path:
    sys.path.append(BASE_DIR)


# ---------------------------------------------------------------------
# Log directories
# ---------------------------------------------------------------------

log_directory = os.path.join(BASE_DIR, "logs")
log_backup_directory = os.path.join(BASE_DIR, "log_backup")

os.makedirs(log_directory, exist_ok=True)
os.makedirs(log_backup_directory, exist_ok=True)

TRAINING_LOG_DIR = os.path.join(log_directory, "training")
TRAINING_LOG_BACKUP_DIR = os.path.join(log_backup_directory, "training")

os.makedirs(TRAINING_LOG_DIR, exist_ok=True)
os.makedirs(TRAINING_LOG_BACKUP_DIR, exist_ok=True)


# ---------------------------------------------------------------------
# Console UTF-8 support on Windows
# ---------------------------------------------------------------------

def _configure_windows_console_utf8() -> None:
    """
    Best-effort UTF-8 console setup for Windows.

    This reduces logging errors when messages include special characters.
    """

    if not sys.platform.startswith("win"):
        return

    try:
        os.system("chcp 65001 >nul")
    except Exception:
        pass

    for stream in (sys.stdout, sys.stderr):
        try:
            if hasattr(stream, "reconfigure"):
                stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass


_configure_windows_console_utf8()


# ---------------------------------------------------------------------
# Shared formatting
# ---------------------------------------------------------------------

DEFAULT_FORMATTER = logging.Formatter(
    "%(asctime)s - %(name)s - %(levelname)s - "
    "%(filename)s:%(lineno)d - %(funcName)s() - %(message)s"
)


# ---------------------------------------------------------------------
# Handler helpers
# ---------------------------------------------------------------------

def using_concurrent_log_handler() -> bool:
    """
    Returns True when concurrent-log-handler is available and active.
    """

    return CONCURRENT_LOG_HANDLER_AVAILABLE


def get_active_log_handler_name() -> str:
    """
    Returns the rotating file handler class name currently selected.
    """

    return ACTIVE_ROTATING_HANDLER_NAME


def _handler_targets_file(handler: logging.Handler, target_path: str) -> bool:
    """
    Returns True when a handler writes to the same absolute file path.
    """

    handler_path = getattr(handler, "baseFilename", None)

    if not handler_path:
        return False

    return os.path.abspath(handler_path) == os.path.abspath(target_path)


def _handler_is_active_rotating_type(handler: logging.Handler) -> bool:
    """
    Returns True when the handler is using the currently selected rotating
    handler implementation.

    This matters because the app may have previously attached a standard
    RotatingFileHandler before concurrent-log-handler was installed.
    """

    return isinstance(handler, _ActiveRotatingFileHandler)


def _has_console_handler(target_logger: logging.Logger) -> bool:
    """
    Returns True when a non-file StreamHandler is already attached.

    Note:
        FileHandler/RotatingFileHandler inherit from StreamHandler, so we
        must exclude handlers that have baseFilename.
    """

    return any(
        isinstance(handler, logging.StreamHandler)
        and not getattr(handler, "baseFilename", None)
        for handler in target_logger.handlers
    )


def _remove_handler_safely(
    target_logger: logging.Logger,
    handler: logging.Handler,
) -> None:
    """
    Remove, flush, and close a handler safely.
    """

    try:
        target_logger.removeHandler(handler)
    except Exception:
        pass

    try:
        handler.flush()
    except Exception:
        pass

    try:
        handler.close()
    except Exception:
        pass


def _create_rotating_file_handler(
    *,
    filename: str,
    max_bytes: int,
    backup_count: int,
    level: int,
    formatter: logging.Formatter,
) -> logging.Handler:
    """
    Creates a rotating file handler.

    Uses ConcurrentRotatingFileHandler when installed. This is safer on
    Windows and safer when Flask debug/reloader, threaded requests, CLI tools,
    or service processes may touch the same log file.

    Falls back to Python's standard RotatingFileHandler when the package is not
    installed.
    """

    os.makedirs(os.path.dirname(os.path.abspath(filename)), exist_ok=True)

    handler = _ActiveRotatingFileHandler(
        filename=filename,
        maxBytes=max_bytes,
        backupCount=backup_count,
        encoding="utf-8",
        delay=True,
    )
    handler.setLevel(level)
    handler.setFormatter(formatter)

    return handler


def _create_console_handler(
    *,
    level: int,
    formatter: logging.Formatter,
) -> logging.Handler:
    """
    Creates a stdout console handler.
    """

    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(level)
    handler.setFormatter(formatter)

    return handler


def _ensure_rotating_file_handler(
    *,
    target_logger: logging.Logger,
    filename: str,
    max_bytes: int,
    backup_count: int,
    level: int,
    formatter: logging.Formatter,
    replace_wrong_handler_type: bool = True,
) -> logging.Handler:
    """
    Ensure a logger has exactly one suitable rotating file handler for a file.

    If a same-file handler already exists but was created from the fallback
    handler before concurrent-log-handler was installed, it can be replaced.
    """

    absolute_filename = os.path.abspath(filename)

    matching_handlers = [
        handler
        for handler in target_logger.handlers
        if _handler_targets_file(handler, absolute_filename)
    ]

    for handler in matching_handlers:
        if _handler_is_active_rotating_type(handler):
            handler.setLevel(level)
            handler.setFormatter(formatter)
            return handler

        if replace_wrong_handler_type:
            _remove_handler_safely(target_logger, handler)

    new_handler = _create_rotating_file_handler(
        filename=absolute_filename,
        max_bytes=max_bytes,
        backup_count=backup_count,
        level=level,
        formatter=formatter,
    )
    target_logger.addHandler(new_handler)

    return new_handler


def _ensure_console_handler(
    *,
    target_logger: logging.Logger,
    level: int,
    formatter: logging.Formatter,
) -> Optional[logging.Handler]:
    """
    Ensure a logger has one non-file console handler.
    """

    if _has_console_handler(target_logger):
        return None

    console_handler = _create_console_handler(
        level=level,
        formatter=formatter,
    )
    target_logger.addHandler(console_handler)

    return console_handler


def _safe_console_text(message: object) -> str:
    """
    Converts a message to text and makes it safe for the active console.
    """

    text = str(message)

    console_encoding = getattr(sys.stdout, "encoding", None) or "cp1252"

    try:
        text.encode(console_encoding, errors="strict")
        return text
    except (UnicodeEncodeError, AttributeError):
        normalized = unicodedata.normalize("NFD", text)
        without_combining = "".join(
            char for char in normalized
            if unicodedata.category(char) != "Mn"
        )

        safe_chars = []

        for char in without_combining:
            try:
                char.encode(console_encoding)
                safe_chars.append(char)
            except UnicodeEncodeError:
                safe_chars.append(f"U+{ord(char):04X}")

        return "".join(safe_chars)


# ---------------------------------------------------------------------
# Main app logger
# ---------------------------------------------------------------------

logger = logging.getLogger("ematac_logger")
logger.setLevel(logging.DEBUG)
logger.propagate = False

APP_LOG_PATH = os.path.join(log_directory, "app.log")

_ensure_rotating_file_handler(
    target_logger=logger,
    filename=APP_LOG_PATH,
    max_bytes=5 * 1024 * 1024,
    backup_count=5,
    level=logging.DEBUG,
    formatter=DEFAULT_FORMATTER,
)

_ensure_console_handler(
    target_logger=logger,
    level=logging.DEBUG,
    formatter=DEFAULT_FORMATTER,
)

if not getattr(logger, "_emtac_logging_config_initialized", False):
    logger.info(
        "Logging initialized. rotating_handler=%s concurrent_available=%s "
        "log_file=%s",
        ACTIVE_ROTATING_HANDLER_NAME,
        CONCURRENT_LOG_HANDLER_AVAILABLE,
        APP_LOG_PATH,
    )
    logger._emtac_logging_config_initialized = True


# ---------------------------------------------------------------------
# UUID-based request tracking
# ---------------------------------------------------------------------

_local = threading.local()


def get_request_id():
    """
    Get the current request ID from Flask context if available,
    or from thread-local storage as a fallback.

    If neither exists, generate a new one.
    """

    if has_request_context() and g is not None and hasattr(g, "request_id"):
        return g.request_id

    if hasattr(_local, "request_id"):
        return _local.request_id

    request_id = str(uuid.uuid4())[:8]
    _local.request_id = request_id

    return request_id


def set_request_id(request_id=None):
    """
    Set a request ID in thread-local storage.

    If no request_id is provided, generate a new one.
    Returns the request ID that was set.
    """

    if request_id is None:
        request_id = str(uuid.uuid4())[:8]

    _local.request_id = request_id

    return request_id


def clear_request_id():
    """
    Clear the request ID from thread-local storage.
    """

    if hasattr(_local, "request_id"):
        delattr(_local, "request_id")


def log_with_id(level, message, request_id=None, *args, **kwargs):
    """
    Log message with request ID.

    This is defensive against Unicode/console encoding issues on Windows.
    """

    try:
        if request_id:
            final = f"[REQ-{request_id}] {message}"
        else:
            final = str(message)

        final = _safe_console_text(final)

        if level == logging.DEBUG:
            logger.debug(final, *args, **kwargs)
        elif level == logging.INFO:
            logger.info(final, *args, **kwargs)
        elif level == logging.WARNING:
            logger.warning(final, *args, **kwargs)
        elif level == logging.ERROR:
            logger.error(final, *args, **kwargs)
        elif level == logging.CRITICAL:
            logger.critical(final, *args, **kwargs)
        else:
            logger.log(level, final, *args, **kwargs)

    except Exception as exc:
        fallback_msg = (
            f"[REQ-{request_id or 'unknown'}] "
            f"LOGGING_ERROR: {str(exc)[:200]}"
        )
        ascii_only = fallback_msg.encode("ascii", "replace").decode("ascii")

        try:
            logger.error(ascii_only)
        except Exception:
            try:
                print(f"CRITICAL LOGGING FAILURE - REQUEST: {request_id or 'unknown'}")
            except Exception:
                pass


def debug_id(message, request_id=None, *args, **kwargs):
    log_with_id(logging.DEBUG, message, request_id, *args, **kwargs)


def info_id(message, request_id=None, *args, **kwargs):
    log_with_id(logging.INFO, message, request_id, *args, **kwargs)


def warning_id(message, request_id=None, *args, **kwargs):
    log_with_id(logging.WARNING, message, request_id, *args, **kwargs)


def error_id(message, request_id=None, *args, **kwargs):
    log_with_id(logging.ERROR, message, request_id, *args, **kwargs)


def critical_id(message, request_id=None, *args, **kwargs):
    log_with_id(logging.CRITICAL, message, request_id, *args, **kwargs)


def with_request_id(func):
    """
    Decorator that adds request ID tracking to a function.

    Creates a new request ID if one does not exist.
    """

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        request_id = get_request_id()
        start_time = time.time()

        debug_id(f"Starting {func.__name__}", request_id)

        try:
            result = func(*args, **kwargs)
            end_time = time.time()
            debug_id(
                f"Completed {func.__name__} in {end_time - start_time:.3f}s",
                request_id,
            )
            return result

        except Exception as exc:
            end_time = time.time()
            error_id(
                f"Error in {func.__name__} after "
                f"{end_time - start_time:.3f}s: {exc}",
                request_id,
                exc_info=True,
            )
            raise

    return wrapper


# ---------------------------------------------------------------------
# Flask middleware
# ---------------------------------------------------------------------

def request_id_middleware(app):
    """
    Add request ID middleware to a Flask app.

    This sets a unique request ID for each HTTP request.
    """

    @app.before_request
    def before_request():
        g.request_id = str(uuid.uuid4())[:8]
        g.request_start_time = time.time()

        _local.request_id = g.request_id

        request_method = getattr(request, "method", "UNKNOWN")
        request_path = getattr(request, "path", "UNKNOWN")

        info_id(
            f"Processing request: {request_method} {request_path}",
            g.request_id,
        )

    @app.after_request
    def after_request(response):
        request_id = getattr(g, "request_id", None)

        if hasattr(g, "request_start_time"):
            duration = time.time() - g.request_start_time
            info_id(
                f"Request completed in {duration:.3f}s "
                f"with status {response.status_code}",
                request_id,
            )

        clear_request_id()

        return response

    @app.teardown_request
    def teardown_request(exception=None):
        request_id = getattr(g, "request_id", None)

        if exception:
            error_id(
                f"Request failed with exception: {str(exception)}",
                request_id,
                exc_info=True,
            )

        clear_request_id()

    return app


# ---------------------------------------------------------------------
# Timed operation helper
# ---------------------------------------------------------------------

def log_timed_operation(operation_name, request_id=None):
    """
    Context manager for timing and logging operations.
    """

    class TimedOperationContext:
        def __init__(self, operation_name, request_id):
            self.operation_name = operation_name
            self.request_id = request_id if request_id else get_request_id()
            self.start_time = None

        def __enter__(self):
            self.start_time = time.time()
            debug_id(
                f"Starting operation: {self.operation_name}",
                self.request_id,
            )
            return self

        def __exit__(self, exc_type, exc_val, exc_tb):
            duration = time.time() - self.start_time

            if exc_type:
                error_id(
                    f"Operation {self.operation_name} failed after "
                    f"{duration:.3f}s: {str(exc_val)}",
                    self.request_id,
                    exc_info=True,
                )
            else:
                debug_id(
                    f"Operation {self.operation_name} completed in "
                    f"{duration:.3f}s",
                    self.request_id,
                )

            return False

    return TimedOperationContext(operation_name, request_id)


# ---------------------------------------------------------------------
# Log cleanup / backup helpers
# ---------------------------------------------------------------------

def compress_and_backup_logs(log_directory, backup_directory):
    """
    Consolidate and compress log files older than 14 days into biweekly
    backup files.

    Notes:
        - Skips directories.
        - Skips .gz files.
        - Skips currently active app/training/maintenance logs.
        - Handles Windows file locks safely.
    """

    now = datetime.now()
    biweekly_logs = {}

    if not os.path.isdir(log_directory):
        return

    for file_name in os.listdir(log_directory):
        file_path = os.path.join(log_directory, file_name)

        if os.path.isdir(file_path):
            continue

        if file_name.endswith(".gz"):
            continue

        # Avoid touching active log files.
        if file_name in {
            "app.log",
            "training.log",
            "database_maintenance.log",
        }:
            continue

        try:
            file_modified_time = datetime.fromtimestamp(os.path.getmtime(file_path))
        except OSError as exc:
            warning_id(f"Could not stat log file for cleanup: {file_path} - {exc}")
            continue

        if now - file_modified_time <= timedelta(days=14):
            continue

        year = file_modified_time.year
        day_of_year = file_modified_time.timetuple().tm_yday
        biweek = (day_of_year - 1) // 14 + 1
        biweek_key = f"{year}-BW{biweek:02d}"

        biweekly_logs.setdefault(biweek_key, []).append(file_path)
        logger.debug(
            f"Grouping file {file_path} under biweekly period {biweek_key}"
        )

    os.makedirs(backup_directory, exist_ok=True)

    for biweek_key, files in biweekly_logs.items():
        backup_file_path = os.path.join(
            backup_directory,
            f"backup_{biweek_key}.gz",
        )

        logger.info(
            f"Creating biweekly backup: {backup_file_path} "
            f"with {len(files)} file(s)"
        )

        try:
            with gzip.open(backup_file_path, "ab") as f_out:
                for file_path in files:
                    try:
                        with open(file_path, "rb") as f_in:
                            shutil.copyfileobj(f_in, f_out)

                        os.remove(file_path)
                        logger.info(f"Compressed and removed: {file_path}")

                    except PermissionError as exc:
                        warning_id(
                            f"Skipping locked log file during compression: "
                            f"{file_path} - {exc}"
                        )

                    except OSError as exc:
                        warning_id(
                            f"Skipping log file during compression due to OS error: "
                            f"{file_path} - {exc}"
                        )

        except OSError as exc:
            warning_id(
                f"Could not create compressed backup: "
                f"{backup_file_path} - {exc}"
            )


def delete_old_backups(backup_directory, retention_weeks=2):
    """
    Delete backup files older than the specified retention period.
    """

    if not os.path.isdir(backup_directory):
        return

    now = datetime.now()

    for file_name in os.listdir(backup_directory):
        file_path = os.path.join(backup_directory, file_name)

        if os.path.isdir(file_path):
            continue

        if not file_name.endswith(".gz"):
            continue

        try:
            file_modified_time = datetime.fromtimestamp(os.path.getmtime(file_path))

            if now - file_modified_time > timedelta(weeks=retention_weeks):
                os.remove(file_path)
                logger.info(f"Deleted old backup: {file_path}")

        except PermissionError as exc:
            warning_id(f"Skipping locked backup file: {file_path} - {exc}")

        except OSError as exc:
            warning_id(f"Could not process backup file: {file_path} - {exc}")


def initial_log_cleanup():
    """
    Perform initial cleanup by consolidating old logs into biweekly backups
    and deleting old backups.
    """

    logger.info("Starting initial log cleanup...")

    compress_and_backup_logs(log_directory, log_backup_directory)
    delete_old_backups(log_backup_directory, retention_weeks=2)

    logger.info("Initial log cleanup completed.")


def maintain_training_logs(retention_weeks: int = 2):
    """
    Mirror app log maintenance for training logs.
    """

    try:
        logger.info("Starting training log cleanup...")

        compress_and_backup_logs(TRAINING_LOG_DIR, TRAINING_LOG_BACKUP_DIR)
        delete_old_backups(
            TRAINING_LOG_BACKUP_DIR,
            retention_weeks=retention_weeks,
        )

        logger.info("Training log cleanup complete.")

    except Exception as exc:
        logger.exception(f"Training log maintenance failed: {exc}")


# ---------------------------------------------------------------------
# TrainingLogManager
# ---------------------------------------------------------------------

class TrainingLogManager:
    """
    Dedicated training logger.

    Features:
        - Non-propagating logger.
        - Writes to <run_dir>/training.log when run_dir is provided.
        - Otherwise writes to BASE_DIR/logs/training/training.log.
        - Optional console mirror.
        - Context-manager closes handlers cleanly.
        - make_trainer_callback() bridges Hugging Face Trainer logs.
    """

    def __init__(
        self,
        run_dir: Optional[os.PathLike] = None,
        run_name: Optional[str] = None,
        to_console: bool = False,
        level: int = logging.INFO,
        rotate_mb: int = 10,
        backups: int = 5,
    ):
        self.run_dir = Path(run_dir) if run_dir else None
        self.run_name = run_name or (
            self.run_dir.name if self.run_dir else "global"
        )
        self.logger_name = f"ematac_train.{self.run_name}"
        self.level = level
        self.to_console = to_console
        self.rotate_mb = rotate_mb
        self.backups = backups

        self._logger = logging.getLogger(self.logger_name)
        self._logger.setLevel(level)
        self._logger.propagate = False
        self._adapter = None

        self._attach_handlers()

    @property
    def logger(self) -> logging.LoggerAdapter:
        return self._adapter

    def _attach_handlers(self):
        os.makedirs(TRAINING_LOG_DIR, exist_ok=True)
        os.makedirs(TRAINING_LOG_BACKUP_DIR, exist_ok=True)

        if self.run_dir:
            log_path = self.run_dir / "training.log"
            log_path.parent.mkdir(parents=True, exist_ok=True)
        else:
            log_path = Path(TRAINING_LOG_DIR) / "training.log"

        formatter = logging.Formatter(
            "%(asctime)s - ematac_train - [%(run)s] "
            "%(levelname)s - %(filename)s:%(lineno)d - %(message)s"
        )

        _ensure_rotating_file_handler(
            target_logger=self._logger,
            filename=str(log_path),
            max_bytes=self.rotate_mb * 1024 * 1024,
            backup_count=self.backups,
            level=self.level,
            formatter=formatter,
        )

        if self.to_console:
            console_formatter = logging.Formatter(
                "%(asctime)s - ematac_train - [%(run)s] "
                "%(levelname)s - %(message)s"
            )

            _ensure_console_handler(
                target_logger=self._logger,
                level=self.level,
                formatter=console_formatter,
            )

        self._adapter = logging.LoggerAdapter(
            self._logger,
            extra={"run": self.run_name},
        )

        if not getattr(self._logger, "_emtac_training_logger_initialized", False):
            self._adapter.info(
                "Training logger initialized. rotating_handler=%s "
                "concurrent_available=%s",
                ACTIVE_ROTATING_HANDLER_NAME,
                CONCURRENT_LOG_HANDLER_AVAILABLE,
            )
            self._logger._emtac_training_logger_initialized = True

    def make_trainer_callback(self):
        """
        Return a Hugging Face TrainerCallback that mirrors logs/eval metrics
        into this training logger.
        """

        try:
            from transformers import TrainerCallback
        except Exception:
            class _Noop:
                pass

            return _Noop()

        adapter = self._adapter

        class _MetricsToLogger(TrainerCallback):
            def on_log(self, args, state, control, logs=None, **kwargs):
                if logs:
                    adapter.info(f"HF_LOG: {logs}")

            def on_evaluate(self, args, state, control, metrics=None, **kwargs):
                if metrics:
                    adapter.info(f"EVAL: {metrics}")

            def on_train_end(self, args, state, control, **kwargs):
                adapter.info("Training finished.")

        return _MetricsToLogger()

    def close(self):
        """
        Flush and close handlers.
        """

        for handler in list(self._logger.handlers):
            _remove_handler_safely(self._logger, handler)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()




# ---------------------------------------------------------------------
# DatabaseMaintLogManager
# ---------------------------------------------------------------------

class DatabaseMaintLogManager:
    """
    Dedicated database maintenance logger.

    Features:
        - Keeps database maintenance logs separate from general app logs.
        - Supports per-run log files inside a run/output directory.
        - Uses a non-propagating named logger.
        - Optionally mirrors logs to console.
        - Cleans up handlers safely for repeated runs/tests.
    """

    def __init__(
        self,
        run_dir: Optional[os.PathLike] = None,
        run_name: Optional[str] = None,
        to_console: bool = False,
        level: int = logging.INFO,
        rotate_mb: int = 10,
        backups: int = 5,
    ):
        self.run_dir = Path(run_dir) if run_dir else None
        self.run_name = run_name or (
            self.run_dir.name if self.run_dir else "global"
        )
        self.logger_name = f"ematac_db_maint.{self.run_name}"
        self.level = level
        self.to_console = to_console
        self.rotate_mb = rotate_mb
        self.backups = backups

        self._logger = logging.getLogger(self.logger_name)
        self._logger.setLevel(level)
        self._logger.propagate = False
        self._adapter = None

        self._attach_handlers()

    @property
    def logger(self) -> logging.LoggerAdapter:
        return self._adapter

    def _attach_handlers(self):
        try:
            from modules.configuration.config import LOGS_DIR
        except Exception:
            LOGS_DIR = None

        base_logs_dir = Path(LOGS_DIR) if LOGS_DIR else Path(log_directory)
        db_maint_log_dir = base_logs_dir / "database_maintenance"
        db_maint_log_dir.mkdir(parents=True, exist_ok=True)

        if self.run_dir:
            self.run_dir.mkdir(parents=True, exist_ok=True)
            log_path = self.run_dir / "database_maintenance.log"
        else:
            log_path = db_maint_log_dir / "database_maintenance.log"

        formatter = logging.Formatter(
            "%(asctime)s - ematac_db_maint - [%(run)s] "
            "%(levelname)s - %(filename)s:%(lineno)d - "
            "%(funcName)s() - %(message)s"
        )

        _ensure_rotating_file_handler(
            target_logger=self._logger,
            filename=str(log_path),
            max_bytes=self.rotate_mb * 1024 * 1024,
            backup_count=self.backups,
            level=self.level,
            formatter=formatter,
        )

        if self.to_console:
            console_formatter = logging.Formatter(
                "%(asctime)s - ematac_db_maint - [%(run)s] "
                "%(levelname)s - %(message)s"
            )

            _ensure_console_handler(
                target_logger=self._logger,
                level=self.level,
                formatter=console_formatter,
            )

        self._adapter = logging.LoggerAdapter(
            self._logger,
            extra={"run": self.run_name},
        )

        if not getattr(self._logger, "_emtac_db_maint_logger_initialized", False):
            self._adapter.info(
                "Database maintenance logger initialized. rotating_handler=%s "
                "concurrent_available=%s",
                ACTIVE_ROTATING_HANDLER_NAME,
                CONCURRENT_LOG_HANDLER_AVAILABLE,
            )
            self._logger._emtac_db_maint_logger_initialized = True

    def log_start(self, operation_name: str):
        self.logger.info(
            f"Starting database maintenance operation: {operation_name}"
        )

    def log_success(
        self,
        operation_name: str,
        duration_seconds: Optional[float] = None,
    ):
        if duration_seconds is not None:
            self.logger.info(
                f"Completed database maintenance operation: "
                f"{operation_name} in {duration_seconds:.3f}s"
            )
        else:
            self.logger.info(
                f"Completed database maintenance operation: {operation_name}"
            )

    def log_failure(
        self,
        operation_name: str,
        error: Exception,
        duration_seconds: Optional[float] = None,
    ):
        if duration_seconds is not None:
            self.logger.error(
                f"Failed database maintenance operation: {operation_name} "
                f"after {duration_seconds:.3f}s - {error}",
                exc_info=True,
            )
        else:
            self.logger.error(
                f"Failed database maintenance operation: "
                f"{operation_name} - {error}",
                exc_info=True,
            )

    @contextmanager
    def timed_operation(self, operation_name: str):
        """
        Context manager for timing a maintenance operation.
        """

        start_time = time.time()
        self.log_start(operation_name)

        try:
            yield self
            duration = time.time() - start_time
            self.log_success(operation_name, duration)

        except Exception as exc:
            duration = time.time() - start_time
            self.log_failure(operation_name, exc, duration)
            raise

    def close(self):
        """
        Flush and close handlers safely.

        Useful for scripts, tests, and repeated maintenance runs.
        """

        for handler in list(self._logger.handlers):
            _remove_handler_safely(self._logger, handler)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()


# ---------------------------------------------------------------------
# SearchAuditLogManager
# ---------------------------------------------------------------------

class SearchAuditLogManager:
    """
    Dedicated search pathway audit logger.

    Purpose:
        - Keep search/RAG/payload audit logs separate from app.log.
        - Support per-run logs when needed.
        - Track request_id, pathway, stage, and audit events clearly.
        - Avoid polluting the main application logger.

    This logger is for human-readable troubleshooting.

    The PostgreSQL audit schema should remain the source of truth for
    structured, searchable audit records.
    """

    def __init__(
        self,
        run_dir: Optional[os.PathLike] = None,
        run_name: Optional[str] = None,
        to_console: bool = False,
        level: int = logging.INFO,
        rotate_mb: int = 10,
        backups: int = 5,
    ):
        self.run_dir = Path(run_dir) if run_dir else None
        self.run_name = run_name or (
            self.run_dir.name if self.run_dir else "global"
        )
        self.logger_name = f"ematac_search_audit.{self.run_name}"
        self.level = level
        self.to_console = to_console
        self.rotate_mb = rotate_mb
        self.backups = backups

        self._logger = logging.getLogger(self.logger_name)
        self._logger.setLevel(level)
        self._logger.propagate = False
        self._adapter = None

        self._attach_handlers()

    @property
    def logger(self) -> logging.LoggerAdapter:
        return self._adapter

    def _attach_handlers(self):
        try:
            from modules.configuration.config import LOGS_DIR
        except Exception:
            LOGS_DIR = None

        base_logs_dir = Path(LOGS_DIR) if LOGS_DIR else Path(log_directory)
        search_audit_log_dir = base_logs_dir / "search_audit"
        search_audit_log_dir.mkdir(parents=True, exist_ok=True)

        if self.run_dir:
            self.run_dir.mkdir(parents=True, exist_ok=True)
            log_path = self.run_dir / "search_audit.log"
        else:
            log_path = search_audit_log_dir / "search_audit.log"

        formatter = logging.Formatter(
            "%(asctime)s - ematac_search_audit - [%(run)s] "
            "%(levelname)s - %(filename)s:%(lineno)d - "
            "%(funcName)s() - %(message)s"
        )

        _ensure_rotating_file_handler(
            target_logger=self._logger,
            filename=str(log_path),
            max_bytes=self.rotate_mb * 1024 * 1024,
            backup_count=self.backups,
            level=self.level,
            formatter=formatter,
        )

        if self.to_console:
            console_formatter = logging.Formatter(
                "%(asctime)s - ematac_search_audit - [%(run)s] "
                "%(levelname)s - %(message)s"
            )

            _ensure_console_handler(
                target_logger=self._logger,
                level=self.level,
                formatter=console_formatter,
            )

        self._adapter = logging.LoggerAdapter(
            self._logger,
            extra={"run": self.run_name},
        )

        if not getattr(self._logger, "_emtac_search_audit_logger_initialized", False):
            self._adapter.info(
                "Search audit logger initialized. rotating_handler=%s "
                "concurrent_available=%s",
                ACTIVE_ROTATING_HANDLER_NAME,
                CONCURRENT_LOG_HANDLER_AVAILABLE,
            )
            self._logger._emtac_search_audit_logger_initialized = True

    def log_run_start(
        self,
        *,
        request_id: str,
        pathway_name: str,
        question: str | None = None,
    ):
        self.logger.info(
            "AUDIT_RUN_START request_id=%s pathway=%s question=%r",
            request_id,
            pathway_name,
            question,
        )

    def log_run_success(
        self,
        *,
        request_id: str,
        pathway_name: str,
        duration_ms: int | None = None,
        counts: dict | None = None,
    ):
        self.logger.info(
            "AUDIT_RUN_SUCCESS request_id=%s pathway=%s duration_ms=%s counts=%s",
            request_id,
            pathway_name,
            duration_ms,
            counts or {},
        )

    def log_run_failure(
        self,
        *,
        request_id: str,
        pathway_name: str,
        error: Exception,
        duration_ms: int | None = None,
    ):
        self.logger.error(
            "AUDIT_RUN_FAILURE request_id=%s pathway=%s duration_ms=%s error=%s",
            request_id,
            pathway_name,
            duration_ms,
            error,
            exc_info=True,
        )

    def log_stage_start(
        self,
        *,
        request_id: str,
        pathway_name: str,
        stage_name: str,
    ):
        self.logger.debug(
            "AUDIT_STAGE_START request_id=%s pathway=%s stage=%s",
            request_id,
            pathway_name,
            stage_name,
        )

    def log_stage_success(
        self,
        *,
        request_id: str,
        pathway_name: str,
        stage_name: str,
        duration_ms: int | None = None,
        output_count: int | None = None,
    ):
        self.logger.debug(
            "AUDIT_STAGE_SUCCESS request_id=%s pathway=%s stage=%s "
            "duration_ms=%s output_count=%s",
            request_id,
            pathway_name,
            stage_name,
            duration_ms,
            output_count,
        )

    def log_stage_failure(
        self,
        *,
        request_id: str,
        pathway_name: str,
        stage_name: str,
        error: Exception,
        duration_ms: int | None = None,
    ):
        self.logger.error(
            "AUDIT_STAGE_FAILURE request_id=%s pathway=%s stage=%s "
            "duration_ms=%s error=%s",
            request_id,
            pathway_name,
            stage_name,
            duration_ms,
            error,
            exc_info=True,
        )

    def log_validation_result(
        self,
        *,
        request_id: str,
        pathway_name: str,
        check_name: str,
        check_status: str,
        details: dict | None = None,
    ):
        log_method = self.logger.info

        if check_status in {"failed", "error"}:
            log_method = self.logger.error
        elif check_status == "warning":
            log_method = self.logger.warning

        log_method(
            "AUDIT_VALIDATION request_id=%s pathway=%s check=%s status=%s details=%s",
            request_id,
            pathway_name,
            check_name,
            check_status,
            details or {},
        )

    def log_payload_counts(
        self,
        *,
        request_id: str,
        pathway_name: str,
        counts: dict,
    ):
        self.logger.info(
            "AUDIT_PAYLOAD_COUNTS request_id=%s pathway=%s counts=%s",
            request_id,
            pathway_name,
            counts,
        )

    def close(self):
        """
        Flush and close handlers safely.

        Useful for scripts, tests, and repeated maintenance runs.
        """

        for handler in list(self._logger.handlers):
            _remove_handler_safely(self._logger, handler)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()