import logging
import os
import sys
import time
from typing import Optional, Any

from .directory import LOG_DIR
from .opsgenie import raise_alert, HIGH
from .time_util import next_n_minute, dt_to_str

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

LOG_FORMAT = '%(asctime)s %(levelname)s %(name)-8s: %(message)s'


class ScriptNameFormatter(logging.Formatter):
    """Custom logging formatter that includes script name in log messages.

    Extracts the script name from sys.argv[0] and adds it to the log record
    for more informative logging output.
    """

    def format(self, record: logging.LogRecord) -> str:
        """Format log record with script name included.

        Args:
            record: LogRecord to format.

        Returns:
            Formatted log message string with script name.
        """
        record.script_name = sys.argv[0].split('/')[-1]
        return super().format(record)


def get_logging_config(filename: str, log_dir: Optional[str] = LOG_DIR) -> dict:
    """Generate logging configuration dictionary for file and console output.

    Creates a logging configuration that outputs to both console and a timestamped
    file in the specified directory. Uses custom ScriptNameFormatter for better
    log context.

    Args:
        filename: Base name for the log file (without extension).
        log_dir: Directory to write log files. Defaults to LOG_DIR.

    Returns:
        Dictionary containing logging configuration for use with logging.config.dictConfig.

    Notes:
        - Log files are named as: {filename}.{timestamp}.log
        - Both console and file handlers are set to INFO level
        - Uses ScriptNameFormatter to include script name in logs
    """
    filename = f"{log_dir}/{filename}.{dt_to_str()}.log"
    print(f"Logging to {filename}")

    logging_config = {
        'version': 1,
        'disable_existing_loggers': False,
        'formatters': {
            'standard': {
                '()': ScriptNameFormatter,
                'format': '%(asctime)s %(levelname)s %(script_name)s %(name)-8s: %(message)s',
            },
        },
        'handlers': {
            'console': {
                'level': 'INFO',
                'class': 'logging.StreamHandler',
                'formatter': 'standard',
            },
            'file': {
                'level': 'INFO',
                'class': 'logging.FileHandler',
                'formatter': 'standard',
                'filename': filename,
                'mode': 'a',
            },
        },
        'root': {
            'handlers': ['console', 'file'],
            'level': 'INFO',
        },
        'loggers': {
            '': {
                'handlers': ['console', 'file'],
                'level': 'INFO',
                'propagate': True,
            },
        },
    }
    return logging_config


class SyncLogReader:
    def __init__(self, filename: str):
        self.filename = filename
        self.file = None
        self.st_size = None
        self.next_heartbeat_ts = next_n_minute(time.time(), 2)

    def _open_file(self, init: bool = True):
        """Open the file and store its st_size."""
        try:
            self.file = open(self.filename, 'r')
            self.st_size = os.fstat(self.file.fileno()).st_size
            if init:
                self.file.seek(0, 2)  # Move pointer to the end of the file
            else:
                self.file.seek(0, 0)
            logger.info(f"Open file {self.file} st_size: {self.st_size}")
        except Exception as e:
            logger.error(f"Failed to open file {self.filename}: {e}")
            self.file = None
            self.st_size = None

    def _check_for_replacement(self) -> bool:
        """Check if the file has been replaced by comparing st_size."""
        try:
            if self.st_size is None:
                return True
            current_st_size = os.stat(self.filename).st_size
            return current_st_size < self.st_size
        except FileNotFoundError:
            return True  # File is missing

    def read_file(self):
        """Continuously read new lines from the file, reopening if replaced."""
        self._open_file(init=True)
        if not self.file:
            return

        while True:
            if self._check_for_replacement():
                logger.warning(f"File {self.filename} has been replaced. Reopening...")
                if self.file is not None:
                    self.file.close()
                self._open_file(init=False)
                if not self.file:
                    time.sleep(60)
                    continue

            line = self.file.readline()
            if time.time() > self.next_heartbeat_ts:
                logger.info(f"Running log reader with current size {self.st_size} heartbeat {self.next_heartbeat_ts} latest logging {line}")
                self.next_heartbeat_ts = next_n_minute(self.next_heartbeat_ts, 2)

            if line:
                self.st_size = os.fstat(self.file.fileno()).st_size
                yield line.strip()
            else:
                time.sleep(60)  # Avoid busy waiting


def process_error_log(error_line: str, lower_line: str, log_file: str = ""):
    try:
        # New format of error msg should contain key inside two @
        if len(error_line.split('@')) > 1:
            key = f"{error_line.split('@')[1]}".strip()
        else:
            error_index = lower_line.find('error')
            if error_index != -1:
                key = error_line[error_index + 5:].split(':')[0].strip()
            else:
                key = "Unknown Error"
            if not key:
                key = "Unknown Error"
        # Include log file source in description
        log_source = os.path.basename(log_file) if log_file else "unknown"
        description = f"[Source: {log_source}] {error_line}"
        raise_alert(key=key, priority=HIGH, description=description)

    except Exception as e:
        logger.exception(f"Error in process_error_log: {str(e)}")


def process_log(filename: str):
    reader = SyncLogReader(filename)

    logger.info(f"Reading file {filename} and waiting for new entries:")
    for line in reader.read_file():
        lower_line = line.lower()
        if ('ERROR' in line or 'Error' in line) and 'warning' not in lower_line:
            logger.info(f"Found error line: {line}")
            try:
                process_error_log(line, lower_line, log_file=filename)
            except Exception as e:
                logger.exception(f"Error when processing error log: {str(e)}")

    logger.info("Finished reading log.")


class KeyLogger:
    """Logger wrapper that adds key-based tagging to log messages.

    Wraps a standard logger to allow adding searchable keys to log messages
    using the format '@ key @: message'. Useful for filtering logs by topic.

    Attributes:
        logger: The underlying logging.Logger instance.
    """

    def __init__(self, logger: logging.Logger):
        """Initialize KeyLogger with an existing logger.

        Args:
            logger: Logger instance to wrap.
        """
        self.logger = logger

    def _log_with_key(self, level: str, msg: str, *args: Any, key: Optional[str] = None, **kwargs: Any):
        """Internal method to log with optional key tagging.

        Args:
            level: Log level name ('debug', 'info', etc.).
            msg: Message to log.
            *args: Positional arguments for message formatting.
            key: Optional key to tag the message with.
            **kwargs: Keyword arguments for the logger.
        """
        if key is not None:
            msg = f"@ {key} @: {msg}"
        getattr(self.logger, level)(msg, *args, **kwargs)

    def debug(self, msg: str, *args: Any, key: Optional[str] = None, **kwargs: Any):
        """Log debug message with optional key.

        Args:
            msg: Message to log.
            *args: Positional arguments for message formatting.
            key: Optional key to tag the message with.
            **kwargs: Keyword arguments for the logger.
        """
        self._log_with_key('debug', msg, *args, key=key, **kwargs)

    def info(self, msg: str, *args: Any, key: Optional[str] = None, **kwargs: Any):
        """Log info message with optional key.

        Args:
            msg: Message to log.
            *args: Positional arguments for message formatting.
            key: Optional key to tag the message with.
            **kwargs: Keyword arguments for the logger.
        """
        self._log_with_key('info', msg, *args, key=key, **kwargs)

    def warning(self, msg: str, *args: Any, key: Optional[str] = None, **kwargs: Any):
        """Log warning message with optional key.

        Args:
            msg: Message to log.
            *args: Positional arguments for message formatting.
            key: Optional key to tag the message with.
            **kwargs: Keyword arguments for the logger.
        """
        self._log_with_key('warning', msg, *args, key=key, **kwargs)

    def error(self, msg: str, *args: Any, key: Optional[str] = None, **kwargs: Any):
        """Log error message with optional key.

        Args:
            msg: Message to log.
            *args: Positional arguments for message formatting.
            key: Optional key to tag the message with.
            **kwargs: Keyword arguments for the logger.
        """
        self._log_with_key('error', msg, *args, key=key, **kwargs)

    def critical(self, msg: str, *args: Any, key: Optional[str] = None, **kwargs: Any):
        """Log critical message with optional key.

        Args:
            msg: Message to log.
            *args: Positional arguments for message formatting.
            key: Optional key to tag the message with.
            **kwargs: Keyword arguments for the logger.
        """
        self._log_with_key('critical', msg, *args, key=key, **kwargs)

    def __getattr__(self, name: str):
        """Delegate attribute access to underlying logger.

        Args:
            name: Attribute name to access.

        Returns:
            Attribute from the underlying logger.
        """
        return getattr(self.logger, name)
