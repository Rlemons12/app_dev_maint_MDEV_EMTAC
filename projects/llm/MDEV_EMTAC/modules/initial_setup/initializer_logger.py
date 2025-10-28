import logging
import os
import sys
import gzip
import shutil
from logging.handlers import RotatingFileHandler

#
# 1. Figure out the directory where THIS FILE (initializer_logger.py) resides
#    That will be: .../modules/initial_setup/
#
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

#
# 2. Create a logs directory INSIDE initial_setup/ if it doesn’t exist yet
#    so logs go to: .../modules/initial_setup/logs/
#
LOG_DIRECTORY = os.path.join(BASE_DIR, 'logs')
os.makedirs(LOG_DIRECTORY, exist_ok=True)

#
# 3. Initialize the "initializer_logger"
#
initializer_logger = logging.getLogger('initializer_logger')
initializer_logger.setLevel(logging.DEBUG)

# Path to the initializer log file
initializer_log_path = os.path.join(LOG_DIRECTORY, 'initializer.log')

# RotatingFileHandler -> splits log after it grows beyond 5 MB
# 'delay=True' prevents the file from being opened until the first call to emit()
# This reduces the chance of a file lock error on Windows.
initializer_file_handler = RotatingFileHandler(
    initializer_log_path,
    maxBytes=5 * 1024 * 1024,  # 5 MB
    backupCount=5,
    delay=True
)
initializer_file_handler.setLevel(logging.DEBUG)

# (OPTIONAL) Also log to console
console_handler = logging.StreamHandler(sys.stdout)
console_handler.setLevel(logging.INFO)  # Adjust level if needed

#
# 4. Set up a consistent log format
#
log_formatter = logging.Formatter(
    '%(asctime)s - %(name)s - %(levelname)s - '
    '%(filename)s:%(lineno)d - %(funcName)s() - %(message)s'
)
initializer_file_handler.setFormatter(log_formatter)
console_handler.setFormatter(log_formatter)

#
# 5. Add handlers to the logger (if not already added)
#
if not initializer_logger.handlers:
    initializer_logger.addHandler(initializer_file_handler)
    initializer_logger.addHandler(console_handler)

# Avoid propagating to the root logger
initializer_logger.propagate = False


def close_initializer_logger():
    """
    Call this at the end of a script or once you’re done logging.
    It closes all handlers and fully shuts down logging,
    preventing open-file locks on Windows.
    """
    initializer_logger.info("Closing initializer_logger and its handlers.")
    handlers = initializer_logger.handlers[:]
    for h in handlers:
        h.flush()
        h.close()
        initializer_logger.removeHandler(h)

    # Also shut down the logging system in general
    logging.shutdown()


#
# 6. Optional convenience function to log initialization steps
#
def log_initialization_step(step_description: str):
    """
    Log a step in the initialization process.
    """
    initializer_logger.info(f"[INIT STEP] {step_description}")


#
# 7. Helper function: Compress older logs except the most recent
#
def compress_logs_except_most_recent(log_directory: str):
    """
    Compress all .log files in `log_directory` except for the most recently
    modified one. The compressed files will end with .gz, and the original
    .log files are removed after compression.
    """
    # Gather all *.log files
    log_files = [f for f in os.listdir(log_directory) if f.endswith(".log")]
    if not log_files:
        initializer_logger.info("No .log files found to compress.")
        return

    # Sort logs by last modification time (descending = newest first)
    files_with_times = [
        (fname, os.path.getmtime(os.path.join(log_directory, fname)))
        for fname in log_files
    ]
    files_sorted_desc = sorted(files_with_times, key=lambda x: x[1], reverse=True)

    # The first entry is the newest log file; skip compressing it
    newest_log = files_sorted_desc[0][0]
    older_logs = files_sorted_desc[1:]  # everything else

    initializer_logger.info(f"Skipping compression for newest log: {newest_log}")

    if not older_logs:
        initializer_logger.info("No older logs to compress.")
        return

    # Compress the older logs
    for log_file, _mtime in older_logs:
        full_path = os.path.join(log_directory, log_file)
        gz_path = full_path + ".gz"
        try:
            initializer_logger.info(f"Compressing old log: {log_file} -> {log_file}.gz")
            with open(full_path, 'rb') as f_in, gzip.open(gz_path, 'wb') as f_out:
                shutil.copyfileobj(f_in, f_out)
            os.remove(full_path)
        except Exception as e:
            initializer_logger.error(f"Error compressing {log_file}: {e}")
