"""
Logging configuration for semantic search engine.

Integrates with KCS logging infrastructure using structlog for consistent
structured logging across the entire KCS platform.
"""

import os
import sys
from typing import Any, cast

import structlog
from structlog.stdlib import LoggerFactory


def configure_logging(
    level: str = "info",
    format_type: str = "json",
    enable_metrics: bool = True,
    file_path: str | None = None,
) -> None:
    """Configure structured logging for semantic search engine.

    Args:
        level: Log level (trace, debug, info, warn, error)
        format_type: Output format (json, pretty, compact)
        enable_metrics: Whether to include performance metrics
        file_path: Optional log file path
    """
    # Map string levels to structlog levels
    level_mapping = {
        "trace": "DEBUG",  # structlog doesn't have TRACE
        "debug": "DEBUG",
        "info": "INFO",
        "warn": "WARNING",
        "error": "ERROR",
    }

    log_level = level_mapping.get(level.lower(), "INFO")

    # Base processors for all log formats
    processors = [
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.processors.UnicodeDecoder(),
    ]

    # Add metrics processor if enabled
    if enable_metrics:
        processors.append(_add_performance_context)

    # Choose final renderer based on format
    if format_type == "json":
        processors.append(structlog.processors.JSONRenderer())
    elif format_type == "pretty":
        processors.append(
            structlog.dev.ConsoleRenderer(
                colors=sys.stderr.isatty(),
                exception_formatter=structlog.dev.plain_traceback,
            )
        )
    elif format_type == "compact":
        processors.append(
            structlog.processors.KeyValueRenderer(
                key_order=["timestamp", "level", "logger", "event"],
                drop_missing=True,
            )
        )
    else:
        processors.append(structlog.processors.JSONRenderer())

    # Configure structlog
    structlog.configure(
        processors=cast(Any, processors),
        context_class=dict,
        logger_factory=LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )

    # Configure stdlib logging
    import logging

    logging.basicConfig(
        format="%(message)s",
        stream=sys.stderr if not file_path else None,
        level=getattr(logging, log_level),
        filename=file_path,
    )

    # Set specific log levels for third-party libraries
    logging.getLogger("sentence_transformers").setLevel(logging.WARNING)
    logging.getLogger("transformers").setLevel(logging.WARNING)
    logging.getLogger("torch").setLevel(logging.WARNING)
    logging.getLogger("asyncpg").setLevel(logging.INFO)

    logger = structlog.get_logger("semantic_search.logging")
    logger.info(
        "Semantic search logging configured",
        level=level,
        format=format_type,
        metrics_enabled=enable_metrics,
        file_path=file_path,
    )


def _add_performance_context(
    logger: Any, method_name: str, event_dict: dict[str, Any]
) -> dict[str, Any]:
    """Add performance context to log events."""
    import time

    import psutil

    # Add timestamp for performance tracking
    event_dict["timestamp_ns"] = time.time_ns()

    # Add memory usage if available
    try:
        process = psutil.Process()
        memory_info = process.memory_info()
        event_dict["memory_mb"] = round(memory_info.rss / 1024 / 1024, 2)
        event_dict["cpu_percent"] = process.cpu_percent()
    except (ImportError, psutil.NoSuchProcess):
        # psutil might not be available or process might be gone
        pass

    return event_dict


def get_logger(name: str) -> structlog.stdlib.BoundLogger:
    """Get a configured logger instance.

    Args:
        name: Logger name (typically __name__)

    Returns:
        Configured structured logger
    """
    return cast(structlog.stdlib.BoundLogger, structlog.get_logger(name))


def setup_from_config(config: dict[str, Any]) -> None:
    """Setup logging from configuration dictionary.

    Args:
        config: Logging configuration with keys:
            - level: Log level string
            - format: Output format
            - enable_metrics: Boolean for metrics
            - file_path: Optional file path
    """
    configure_logging(
        level=config.get("level", "info"),
        format_type=config.get("format", "json"),
        enable_metrics=config.get("enable_metrics", True),
        file_path=config.get("file_path"),
    )


def setup_from_env(prefix: str = "SEMANTIC_SEARCH_") -> None:
    """Setup logging from environment variables.

    Args:
        prefix: Environment variable prefix
    """
    level = os.getenv(f"{prefix}LOG_LEVEL", "info")
    format_type = os.getenv(f"{prefix}LOG_FORMAT", "json")
    enable_metrics = os.getenv(f"{prefix}LOG_METRICS", "true").lower() in (
        "true",
        "1",
        "yes",
    )
    file_path = os.getenv(f"{prefix}LOG_FILE")

    configure_logging(
        level=level,
        format_type=format_type,
        enable_metrics=enable_metrics,
        file_path=file_path,
    )


# Default configuration for semantic search
def setup_default() -> None:
    """Setup default logging configuration for semantic search."""
    configure_logging(
        level="info",
        format_type="json",
        enable_metrics=True,
    )


def integrate_with_kcs_logging() -> None:
    """
    Integrate semantic search logging with KCS infrastructure.

    This function ensures semantic search components use the same
    logging configuration as the rest of the KCS platform.
    """
    # Check if KCS logging is already configured
    current_config = structlog.get_config()

    if current_config and current_config.get("processors"):
        # KCS logging is already configured, just add semantic search specific settings
        logger = get_logger("semantic_search.integration")
        logger.info("Integrating with existing KCS logging configuration")

        # Set log levels for semantic search specific libraries
        import logging

        logging.getLogger("sentence_transformers").setLevel(logging.WARNING)
        logging.getLogger("transformers").setLevel(logging.WARNING)
        logging.getLogger("torch").setLevel(logging.WARNING)
        logging.getLogger("asyncpg").setLevel(logging.INFO)

        logger.info("Semantic search logging integration complete")
    else:
        # No existing KCS logging, set up default configuration
        logger_before = structlog.get_logger("semantic_search.integration")
        logger_before.info(
            "No existing KCS logging found, setting up default configuration"
        )
        setup_default()


class LoggingContext:
    """Context manager for temporary logging configuration."""

    def __init__(
        self,
        level: str | None = None,
        format_type: str | None = None,
        enable_metrics: bool | None = None,
    ):
        """Initialize logging context.

        Args:
            level: Temporary log level
            format_type: Temporary format type
            enable_metrics: Temporary metrics setting
        """
        self.level = level
        self.format_type = format_type
        self.enable_metrics = enable_metrics
        self._original_config: dict[str, Any] | None = None

    def __enter__(self) -> "LoggingContext":
        """Enter context and apply temporary configuration."""
        # Store original configuration (simplified)
        self._original_config = {
            "processors": structlog.get_config()["processors"].copy()
        }

        # Apply temporary configuration
        if any([self.level, self.format_type, self.enable_metrics is not None]):
            configure_logging(
                level=self.level or "info",
                format_type=self.format_type or "json",
                enable_metrics=self.enable_metrics
                if self.enable_metrics is not None
                else True,
            )

        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Exit context and restore original configuration."""
        if self._original_config:
            # Restore original processors
            current_config = structlog.get_config()
            structlog.configure(
                processors=cast(Any, self._original_config["processors"]),
                context_class=current_config["context_class"],
                logger_factory=current_config["logger_factory"],
                wrapper_class=current_config["wrapper_class"],
                cache_logger_on_first_use=current_config["cache_logger_on_first_use"],
            )
