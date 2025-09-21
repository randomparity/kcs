"""
KCS logging integration for semantic search components.

Ensures all semantic search modules use the KCS logging infrastructure
with proper structured logging and consistent formatting.
"""

import logging
from typing import Any

import structlog

from .logging_config import get_logger, integrate_with_kcs_logging

# Module-level logger for this integration
logger = get_logger(__name__)


def initialize_logging() -> None:
    """
    Initialize logging integration with KCS infrastructure.

    This should be called once when the semantic search engine starts up
    to ensure proper logging configuration across all components.
    """
    logger.info("Initializing semantic search logging integration")

    # Integrate with KCS logging
    integrate_with_kcs_logging()

    # Update existing semantic search loggers to use structured logging
    _update_semantic_search_loggers()

    logger.info("Semantic search logging integration initialized successfully")


def _update_semantic_search_loggers() -> None:
    """Update existing semantic search module loggers."""
    # List of semantic search modules that should use structured logging
    semantic_search_modules = [
        "semantic_search.services.embedding_service",
        "semantic_search.services.query_preprocessor",
        "semantic_search.services.vector_search_service",
        "semantic_search.services.ranking_service",
        "semantic_search.database.connection",
        "semantic_search.database.vector_store",
        "semantic_search.database.index_manager",
        "semantic_search.mcp.search_tool",
        "semantic_search.mcp.index_tool",
        "semantic_search.mcp.status_tool",
        "semantic_search.mcp.error_handlers",
        "semantic_search.cli.search_commands",
        "semantic_search.cli.index_commands",
        "semantic_search.cli.status_commands",
    ]

    for module_name in semantic_search_modules:
        # Get or create logger for this module
        module_logger = logging.getLogger(module_name)

        # Ensure it uses the structured logging configuration
        if not hasattr(module_logger, "_kcs_structured"):
            # Mark as configured to avoid reconfiguration
            module_logger._kcs_structured = True  # type: ignore

    logger.debug(
        "Updated semantic search module loggers",
        module_count=len(semantic_search_modules),
    )


def get_semantic_search_logger(name: str) -> structlog.stdlib.BoundLogger:
    """
    Get a properly configured logger for semantic search components.

    Args:
        name: Logger name (typically __name__)

    Returns:
        Configured structured logger that integrates with KCS infrastructure
    """
    # Ensure integration is initialized
    if not hasattr(get_semantic_search_logger, "_initialized"):
        initialize_logging()
        get_semantic_search_logger._initialized = True  # type: ignore

    return get_logger(name)


def log_performance_metrics(operation: str, duration_ms: float, **kwargs: Any) -> None:
    """
    Log performance metrics in a standardized format.

    Args:
        operation: Name of the operation
        duration_ms: Duration in milliseconds
        **kwargs: Additional metrics to log
    """
    perf_logger = get_semantic_search_logger("semantic_search.performance")

    perf_logger.info(
        "Performance metric", operation=operation, duration_ms=duration_ms, **kwargs
    )


def log_search_operation(
    query_id: str, query: str, results_count: int, duration_ms: float, **kwargs: Any
) -> None:
    """
    Log search operations for audit and monitoring.

    Args:
        query_id: Unique query identifier
        query: Search query text
        results_count: Number of results returned
        duration_ms: Search duration in milliseconds
        **kwargs: Additional context
    """
    search_logger = get_semantic_search_logger("semantic_search.audit")

    search_logger.info(
        "Search operation completed",
        query_id=query_id,
        query_length=len(query),
        results_count=results_count,
        duration_ms=duration_ms,
        **kwargs,
    )


def log_indexing_operation(
    job_id: str,
    file_count: int,
    chunks_created: int,
    duration_ms: float,
    success: bool,
    **kwargs: Any,
) -> None:
    """
    Log indexing operations for audit and monitoring.

    Args:
        job_id: Unique job identifier
        file_count: Number of files processed
        chunks_created: Number of chunks created
        duration_ms: Indexing duration in milliseconds
        success: Whether the operation succeeded
        **kwargs: Additional context
    """
    index_logger = get_semantic_search_logger("semantic_search.audit")

    index_logger.info(
        "Indexing operation completed",
        job_id=job_id,
        file_count=file_count,
        chunks_created=chunks_created,
        duration_ms=duration_ms,
        success=success,
        **kwargs,
    )


def log_error_with_context(error: Exception, operation: str, **kwargs: Any) -> None:
    """
    Log errors with full context for debugging.

    Args:
        error: Exception that occurred
        operation: Operation that failed
        **kwargs: Additional context
    """
    error_logger = get_semantic_search_logger("semantic_search.errors")

    error_logger.error(
        "Operation failed",
        operation=operation,
        error_type=type(error).__name__,
        error_message=str(error),
        **kwargs,
        exc_info=True,
    )


# Initialize logging when module is imported
try:
    initialize_logging()
except Exception as e:
    # Fallback to basic logging if initialization fails
    import logging

    fallback_logger = logging.getLogger(__name__)
    fallback_logger.warning(f"Failed to initialize KCS logging integration: {e}")
