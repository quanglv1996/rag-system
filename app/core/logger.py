"""Structured logging module.

Configures application-wide logging using structlog with support for
JSON output (production), console output (development), file logging,
request ID tracking, and execution time measurement.

Example:
    >>> from app.core.logger import get_logger
    >>> logger = get_logger(__name__)
    >>> logger.info("Processing request", request_id="abc123")
"""

import logging
import logging.handlers
import sys
import time
import uuid
from contextvars import ContextVar
from pathlib import Path
from typing import Any

import structlog
from structlog.types import EventDict, WrappedLogger

# Context variable to carry request_id across async boundaries
request_id_ctx: ContextVar[str] = ContextVar("request_id", default="")

# =============================================================================
# Custom Processors
# =============================================================================


def add_request_id(
    logger: WrappedLogger,
    method_name: str,
    event_dict: EventDict,
) -> EventDict:
    """Inject the current request ID into every log record.

    Args:
        logger: The wrapped logger instance.
        method_name: The logging method name (info, error, etc.).
        event_dict: The current log event dictionary.

    Returns:
        EventDict with request_id field added.
    """
    request_id = request_id_ctx.get("")
    if request_id:
        event_dict["request_id"] = request_id
    return event_dict


def add_app_context(
    logger: WrappedLogger,
    method_name: str,
    event_dict: EventDict,
) -> EventDict:
    """Add application-level context to every log record.

    Args:
        logger: The wrapped logger instance.
        method_name: The logging method name.
        event_dict: The current log event dictionary.

    Returns:
        EventDict with app context fields added.
    """
    # Import here to avoid circular imports
    from app.core.config import get_settings

    settings = get_settings()
    event_dict["app"] = settings.app_name
    event_dict["env"] = settings.app_env
    return event_dict


# =============================================================================
# Logger Setup
# =============================================================================


def setup_logging(
    log_level: str = "INFO",
    log_format: str = "json",
    log_file: str | None = None,
) -> None:
    """Configure structlog and standard logging.

    Sets up both structlog (for structured logging) and the standard
    logging module (for third-party libraries). Supports JSON and
    console output formats.

    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL).
        log_format: Output format - 'json' for production, 'console' for dev.
        log_file: Optional path to a rotating log file.
    """
    # Shared processors applied to every log event
    shared_processors: list[Any] = [
        structlog.contextvars.merge_contextvars,
        add_request_id,
        add_app_context,
        structlog.stdlib.add_log_level,
        structlog.stdlib.add_logger_name,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
    ]

    if log_format == "json":
        # JSON format for production - machine-parseable
        renderer: Any = structlog.processors.JSONRenderer()
    else:
        # Console format for development - human-readable with colors
        renderer = structlog.dev.ConsoleRenderer(colors=True)

    structlog.configure(
        processors=shared_processors + [renderer],
        wrapper_class=structlog.make_filtering_bound_logger(
            logging.getLevelName(log_level)
        ),
        context_class=dict,
        logger_factory=structlog.PrintLoggerFactory(),
        cache_logger_on_first_use=True,
    )

    # Configure stdlib logging to route through structlog
    logging.basicConfig(
        format="%(message)s",
        stream=sys.stdout,
        level=log_level,
    )

    # Set third-party library log levels to WARNING to reduce noise
    for noisy_lib in ("httpx", "httpcore", "asyncio", "sqlalchemy.engine"):
        logging.getLogger(noisy_lib).setLevel(logging.WARNING)

    # File handler with rotation (if configured)
    if log_file:
        _setup_file_handler(log_file, log_level)


def _setup_file_handler(log_file: str, log_level: str) -> None:
    """Set up a rotating file handler for persistent log storage.

    Args:
        log_file: Path to the log file.
        log_level: Logging level for the file handler.
    """
    log_path = Path(log_file)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    file_handler = logging.handlers.RotatingFileHandler(
        filename=log_path,
        maxBytes=10 * 1024 * 1024,  # 10 MB
        backupCount=5,
        encoding="utf-8",
    )
    file_handler.setLevel(log_level)

    # Use JSON format for file output regardless of console format
    formatter = logging.Formatter(
        fmt='{"timestamp": "%(asctime)s", "level": "%(levelname)s",'
        ' "logger": "%(name)s", "message": "%(message)s"}',
        datefmt="%Y-%m-%dT%H:%M:%S",
    )
    file_handler.setFormatter(formatter)

    root_logger = logging.getLogger()
    root_logger.addHandler(file_handler)


def get_logger(name: str) -> structlog.BoundLogger:
    """Get a named structlog logger.

    Args:
        name: Logger name, typically __name__ of the calling module.

    Returns:
        structlog.BoundLogger: Configured bound logger instance.

    Example:
        >>> logger = get_logger(__name__)
        >>> logger.info("Service started", port=8000)
    """
    return structlog.get_logger(name)


def generate_request_id() -> str:
    """Generate a unique request ID.

    Returns:
        str: UUID4 string for request tracking.
    """
    return str(uuid.uuid4())


def set_request_id(request_id: str) -> None:
    """Set the request ID in the current context.

    Args:
        request_id: The request ID to set.
    """
    request_id_ctx.set(request_id)


class TimingLogger:
    """Context manager for logging execution time.

    Example:
        >>> async with TimingLogger("embedding_generation", logger):
        ...     embeddings = await embed(texts)
    """

    def __init__(self, operation: str, logger: structlog.BoundLogger) -> None:
        """Initialize the timing logger.

        Args:
            operation: Name of the operation being timed.
            logger: Logger instance to use for output.
        """
        self.operation = operation
        self.logger = logger
        self._start: float = 0.0

    def __enter__(self) -> "TimingLogger":
        """Start timing.

        Returns:
            Self for context manager usage.
        """
        self._start = time.perf_counter()
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Log elapsed time on exit.

        Args:
            exc_type: Exception type if an exception occurred.
            exc_val: Exception value if an exception occurred.
            exc_tb: Exception traceback if an exception occurred.
        """
        elapsed_ms = (time.perf_counter() - self._start) * 1000
        if exc_type is None:
            self.logger.info(
                "Operation completed",
                operation=self.operation,
                elapsed_ms=round(elapsed_ms, 2),
            )
        else:
            self.logger.error(
                "Operation failed",
                operation=self.operation,
                elapsed_ms=round(elapsed_ms, 2),
                error=str(exc_val),
            )
