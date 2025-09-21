"""
MCP error handling and validation for semantic search tools.

Provides standardized error handling, validation, and exception management
for all MCP tool implementations in the semantic search engine.
"""

import functools
import logging
import re
import time
from collections.abc import Callable
from typing import Any, TypeVar

from pydantic import ValidationError

logger = logging.getLogger(__name__)

# Type variable for decorated functions
F = TypeVar("F", bound=Callable[..., Any])


class MCPError(Exception):
    """Base exception for MCP tool errors."""

    def __init__(self, code: str, message: str, retryable: bool = False) -> None:
        """
        Initialize MCP error.

        Args:
            code: Error code matching MCP contract
            message: Human-readable error message
            retryable: Whether the operation can be retried
        """
        super().__init__(message)
        self.code = code
        self.message = message
        self.retryable = retryable

    def to_dict(self) -> dict[str, Any]:
        """Convert error to dictionary for MCP response."""
        return {
            "error": {
                "code": self.code,
                "message": self.message,
                "retryable": self.retryable,
            }
        }


class QueryTooLongError(MCPError):
    """Query exceeds maximum length of 1000 characters."""

    def __init__(self, query_length: int) -> None:
        super().__init__(
            code="QUERY_TOO_LONG",
            message=f"Query exceeds maximum length of 1000 characters (got {query_length})",
            retryable=False,
        )


class EmbeddingFailedError(MCPError):
    """Failed to generate embedding for query."""

    def __init__(self, reason: str | None = None) -> None:
        message = "Failed to generate embedding for query"
        if reason:
            message += f": {reason}"
        super().__init__(code="EMBEDDING_FAILED", message=message, retryable=True)


class SearchTimeoutError(MCPError):
    """Search exceeded maximum time limit of 600ms."""

    def __init__(self, actual_time_ms: int) -> None:
        super().__init__(
            code="SEARCH_TIMEOUT",
            message=f"Search exceeded maximum time limit of 600ms (took {actual_time_ms}ms)",
            retryable=True,
        )


class IndexUnavailableError(MCPError):
    """Vector index is currently unavailable."""

    def __init__(self, reason: str | None = None) -> None:
        message = "Vector index is currently unavailable"
        if reason:
            message += f": {reason}"
        super().__init__(code="INDEX_UNAVAILABLE", message=message, retryable=True)


class InvalidConfigContextError(MCPError):
    """Invalid kernel configuration context provided."""

    def __init__(self, invalid_configs: list[str]) -> None:
        config_list = ", ".join(invalid_configs)
        super().__init__(
            code="INVALID_CONFIG_CONTEXT",
            message=f"Invalid kernel configuration context provided: {config_list}",
            retryable=False,
        )


class IndexingInProgressError(MCPError):
    """Content indexing is currently in progress."""

    def __init__(self, job_id: str | None = None) -> None:
        message = "Content indexing is currently in progress"
        if job_id:
            message += f" (job: {job_id})"
        super().__init__(code="INDEXING_IN_PROGRESS", message=message, retryable=True)


class ValidationHelper:
    """Helper class for MCP input validation."""

    @staticmethod
    def validate_query(query: str) -> None:
        """
        Validate search query.

        Args:
            query: Query string to validate

        Raises:
            QueryTooLongError: If query is too long
            ValueError: If query is invalid
        """
        if not isinstance(query, str):
            raise ValueError("Query must be a string")

        if not query.strip():
            raise ValueError("Query cannot be empty")

        if len(query) > 1000:
            raise QueryTooLongError(len(query))

    @staticmethod
    def validate_config_context(config_context: list[str]) -> None:
        """
        Validate kernel configuration context.

        Args:
            config_context: List of config patterns to validate

        Raises:
            InvalidConfigContextError: If any config is invalid
        """
        if not isinstance(config_context, list):
            raise ValueError("Config context must be a list")

        invalid_configs = []
        config_pattern = re.compile(r"^!?CONFIG_[A-Z0-9_]+$")

        for config in config_context:
            if not isinstance(config, str):
                invalid_configs.append(str(config))  # type: ignore[unreachable]
            elif not config_pattern.match(config):
                invalid_configs.append(config)

        if invalid_configs:
            raise InvalidConfigContextError(invalid_configs)

    @staticmethod
    def validate_file_paths(file_paths: list[str]) -> None:
        """
        Validate file paths for indexing.

        Args:
            file_paths: List of file paths to validate

        Raises:
            ValueError: If file paths are invalid
        """
        if not isinstance(file_paths, list):
            raise ValueError("File paths must be a list")

        if not file_paths:
            raise ValueError("At least one file path is required")

        if len(file_paths) > 1000:
            raise ValueError("Cannot index more than 1000 files at once")

        for i, path in enumerate(file_paths):
            if not isinstance(path, str):
                raise ValueError(f"File path {i} must be a string")

            if not path.strip():
                raise ValueError(f"File path {i} cannot be empty")

    @staticmethod
    def validate_content_types(content_types: list[str]) -> None:
        """
        Validate content types filter.

        Args:
            content_types: List of content types to validate

        Raises:
            ValueError: If content types are invalid
        """
        if not isinstance(content_types, list):
            raise ValueError("Content types must be a list")

        valid_types = {"SOURCE_CODE", "DOCUMENTATION", "HEADER", "COMMENT"}
        invalid_types = [ct for ct in content_types if ct not in valid_types]

        if invalid_types:
            raise ValueError(
                f"Invalid content types: {invalid_types}. "
                f"Valid types: {sorted(valid_types)}"
            )


def handle_mcp_errors(func: F) -> F:
    """
    Decorator to handle MCP tool errors consistently.

    Converts various exceptions to standardized MCP error responses
    and adds performance monitoring.

    Args:
        func: Function to decorate

    Returns:
        Decorated function with error handling
    """

    @functools.wraps(func)
    async def wrapper(*args: Any, **kwargs: Any) -> dict[str, Any]:
        start_time = time.time()

        try:
            result: dict[str, Any] = await func(*args, **kwargs)

            # Add performance metrics if not already present
            if isinstance(result, dict) and "search_stats" in result:
                total_time_ms = int((time.time() - start_time) * 1000)
                if "total_time_ms" not in result["search_stats"]:
                    result["search_stats"]["total_time_ms"] = total_time_ms

            return result

        except MCPError as e:
            # MCP errors are already properly formatted
            logger.error(f"MCP error in {func.__name__}: {e.code} - {e.message}")
            return e.to_dict()

        except ValidationError as e:
            # Pydantic validation errors
            logger.error(f"Validation error in {func.__name__}: {e}")
            error = MCPError(
                code="VALIDATION_ERROR",
                message=f"Invalid request parameters: {e!s}",
                retryable=False,
            )
            return error.to_dict()

        except ValueError as e:
            # General validation errors
            logger.error(f"Value error in {func.__name__}: {e}")
            error = MCPError(code="INVALID_INPUT", message=str(e), retryable=False)
            return error.to_dict()

        except TimeoutError as e:
            # Timeout errors
            total_time_ms = int((time.time() - start_time) * 1000)
            logger.error(f"Timeout in {func.__name__}: {e}")
            error = SearchTimeoutError(total_time_ms)
            return error.to_dict()

        except ConnectionError as e:
            # Database connection errors
            logger.error(f"Connection error in {func.__name__}: {e}")
            error = IndexUnavailableError(str(e))
            return error.to_dict()

        except Exception as e:
            # Unexpected errors
            logger.exception(f"Unexpected error in {func.__name__}: {e}")
            error = MCPError(
                code="INTERNAL_ERROR",
                message="An unexpected error occurred",
                retryable=True,
            )
            return error.to_dict()

    return wrapper  # type: ignore[return-value]


def validate_search_request(request_data: dict[str, Any]) -> None:
    """
    Validate semantic search request data.

    Args:
        request_data: Request data to validate

    Raises:
        MCPError: For various validation failures
    """
    # Validate query
    if "query" in request_data:
        ValidationHelper.validate_query(request_data["query"])

    # Validate config context
    if "config_context" in request_data:
        ValidationHelper.validate_config_context(request_data["config_context"])

    # Validate content types
    if "content_types" in request_data:
        ValidationHelper.validate_content_types(request_data["content_types"])


def validate_index_request(request_data: dict[str, Any]) -> None:
    """
    Validate index content request data.

    Args:
        request_data: Request data to validate

    Raises:
        MCPError: For various validation failures
    """
    # Validate file paths
    if "file_paths" in request_data:
        ValidationHelper.validate_file_paths(request_data["file_paths"])


def check_performance_limits(start_time: float, operation: str) -> None:
    """
    Check if operation exceeded performance limits.

    Args:
        start_time: Operation start time
        operation: Operation name for logging

    Raises:
        SearchTimeoutError: If operation took too long
    """
    elapsed_ms = int((time.time() - start_time) * 1000)

    # Performance limits based on requirements
    if operation == "search" and elapsed_ms > 600:
        raise SearchTimeoutError(elapsed_ms)
    elif operation in ["embedding", "indexing"] and elapsed_ms > 30000:  # 30s limit
        raise SearchTimeoutError(elapsed_ms)


class PerformanceMonitor:
    """Monitor and enforce performance limits for MCP operations."""

    def __init__(self, operation: str, timeout_ms: int = 600) -> None:
        """
        Initialize performance monitor.

        Args:
            operation: Operation name
            timeout_ms: Timeout in milliseconds
        """
        self.operation = operation
        self.timeout_ms = timeout_ms
        self.start_time = time.time()

    def check_timeout(self) -> None:
        """
        Check if operation has timed out.

        Raises:
            SearchTimeoutError: If operation timed out
        """
        elapsed_ms = int((time.time() - self.start_time) * 1000)
        if elapsed_ms > self.timeout_ms:
            raise SearchTimeoutError(elapsed_ms)

    def get_elapsed_ms(self) -> int:
        """Get elapsed time in milliseconds."""
        return int((time.time() - self.start_time) * 1000)


# Export commonly used functions and classes
__all__ = [
    "EmbeddingFailedError",
    "IndexUnavailableError",
    "IndexingInProgressError",
    "InvalidConfigContextError",
    "MCPError",
    "PerformanceMonitor",
    "QueryTooLongError",
    "SearchTimeoutError",
    "ValidationHelper",
    "check_performance_limits",
    "handle_mcp_errors",
    "validate_index_request",
    "validate_search_request",
]
