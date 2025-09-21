"""
Enhanced error handling and logging for the call graph extraction pipeline.

This module provides comprehensive error handling, recovery strategies, and
structured logging for the Python side of the call graph extraction pipeline.
It integrates with the Rust error handling system and provides MCP-specific
error management.
"""

import time
from collections import defaultdict
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import structlog
from fastapi import HTTPException, status

logger = structlog.get_logger(__name__)


class ErrorCategory(Enum):
    """Categories of errors for classification and handling."""

    VALIDATION = "validation"
    DATABASE = "database"
    RUST_BRIDGE = "rust_bridge"
    FILE_SYSTEM = "file_system"
    RESOURCE = "resource"
    CONFIGURATION = "configuration"
    NETWORK = "network"
    TIMEOUT = "timeout"
    CRITICAL = "critical"


class ErrorSeverity(Enum):
    """Severity levels for error classification."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class ErrorContext:
    """Context information for enhanced error reporting."""

    operation: str
    file_path: str | None = None
    function_name: str | None = None
    line_number: int | None = None
    user_id: str | None = None
    session_id: str | None = None
    request_id: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert context to dictionary for logging."""
        return {
            "operation": self.operation,
            "file_path": self.file_path,
            "function_name": self.function_name,
            "line_number": self.line_number,
            "user_id": self.user_id,
            "session_id": self.session_id,
            "request_id": self.request_id,
            "metadata": self.metadata,
        }


@dataclass
class ErrorStats:
    """Statistics for error tracking and monitoring."""

    total_errors: int = 0
    errors_by_category: dict[str, int] = field(default_factory=lambda: defaultdict(int))
    errors_by_severity: dict[str, int] = field(default_factory=lambda: defaultdict(int))
    recoverable_errors: int = 0
    retries_attempted: int = 0
    timeouts: int = 0
    critical_errors: int = 0
    error_rate_window: list[float] = field(
        default_factory=list
    )  # Rolling window of error rates
    last_reset: float = field(default_factory=time.time)

    def record_error(
        self,
        category: ErrorCategory,
        severity: ErrorSeverity,
        is_recoverable: bool = False,
        is_retry: bool = False,
    ) -> None:
        """Record an error occurrence with classification."""
        self.total_errors += 1
        self.errors_by_category[category.value] += 1
        self.errors_by_severity[severity.value] += 1

        if is_recoverable:
            self.recoverable_errors += 1

        if is_retry:
            self.retries_attempted += 1

        if severity == ErrorSeverity.CRITICAL:
            self.critical_errors += 1

    def record_timeout(self) -> None:
        """Record a timeout occurrence."""
        self.timeouts += 1
        self.record_error(ErrorCategory.TIMEOUT, ErrorSeverity.MEDIUM)

    def get_error_rate(self, window_size: int = 60) -> float:
        """Get current error rate (errors per minute)."""
        current_time = time.time()
        window_start = current_time - window_size

        # Clean old entries
        self.error_rate_window = [t for t in self.error_rate_window if t > window_start]

        return len(self.error_rate_window)

    def update_error_rate(self) -> None:
        """Update the error rate window with current time."""
        self.error_rate_window.append(time.time())

    def reset(self) -> None:
        """Reset all statistics."""
        self.total_errors = 0
        self.errors_by_category.clear()
        self.errors_by_severity.clear()
        self.recoverable_errors = 0
        self.retries_attempted = 0
        self.timeouts = 0
        self.critical_errors = 0
        self.error_rate_window.clear()
        self.last_reset = time.time()

    def to_dict(self) -> dict[str, Any]:
        """Convert stats to dictionary for reporting."""
        return {
            "total_errors": self.total_errors,
            "errors_by_category": dict(self.errors_by_category),
            "errors_by_severity": dict(self.errors_by_severity),
            "recoverable_errors": self.recoverable_errors,
            "retries_attempted": self.retries_attempted,
            "timeouts": self.timeouts,
            "critical_errors": self.critical_errors,
            "current_error_rate": self.get_error_rate(),
            "uptime_seconds": time.time() - self.last_reset,
        }


class CallGraphError(Exception):
    """Base exception for call graph operations with enhanced context."""

    def __init__(
        self,
        message: str,
        category: ErrorCategory,
        severity: ErrorSeverity = ErrorSeverity.MEDIUM,
        context: ErrorContext | None = None,
        original_error: Exception | None = None,
        is_recoverable: bool = True,
        is_retryable: bool = False,
    ):
        super().__init__(message)
        self.message = message
        self.category = category
        self.severity = severity
        self.context = context or ErrorContext(operation="unknown")
        self.original_error = original_error
        self.is_recoverable = is_recoverable
        self.is_retryable = is_retryable
        self.timestamp = time.time()

    def to_dict(self) -> dict[str, Any]:
        """Convert error to dictionary for logging and responses."""
        return {
            "message": self.message,
            "category": self.category.value,
            "severity": self.severity.value,
            "is_recoverable": self.is_recoverable,
            "is_retryable": self.is_retryable,
            "timestamp": self.timestamp,
            "context": self.context.to_dict(),
            "original_error": str(self.original_error) if self.original_error else None,
        }

    def to_http_exception(self) -> HTTPException:
        """Convert to FastAPI HTTPException with appropriate status code."""
        status_map = {
            ErrorCategory.VALIDATION: status.HTTP_400_BAD_REQUEST,
            ErrorCategory.DATABASE: status.HTTP_503_SERVICE_UNAVAILABLE,
            ErrorCategory.RUST_BRIDGE: status.HTTP_502_BAD_GATEWAY,
            ErrorCategory.FILE_SYSTEM: status.HTTP_404_NOT_FOUND,
            ErrorCategory.RESOURCE: status.HTTP_507_INSUFFICIENT_STORAGE,
            ErrorCategory.CONFIGURATION: status.HTTP_500_INTERNAL_SERVER_ERROR,
            ErrorCategory.NETWORK: status.HTTP_502_BAD_GATEWAY,
            ErrorCategory.TIMEOUT: status.HTTP_408_REQUEST_TIMEOUT,
            ErrorCategory.CRITICAL: status.HTTP_500_INTERNAL_SERVER_ERROR,
        }

        status_code = status_map.get(
            self.category, status.HTTP_500_INTERNAL_SERVER_ERROR
        )

        # Adjust for severity
        if self.severity == ErrorSeverity.CRITICAL:
            status_code = status.HTTP_500_INTERNAL_SERVER_ERROR
        elif self.severity == ErrorSeverity.LOW and status_code >= 500:
            status_code = status.HTTP_400_BAD_REQUEST

        return HTTPException(
            status_code=status_code,
            detail={
                "error": self.category.value,
                "message": self.message,
                "severity": self.severity.value,
                "context": self.context.to_dict(),
            },
        )


class ErrorHandler:
    """Enhanced error handler with recovery strategies and monitoring."""

    def __init__(self, enable_detailed_logging: bool = True):
        self.stats = ErrorStats()
        self.enable_detailed_logging = enable_detailed_logging
        self._recovery_strategies: dict[
            ErrorCategory, Callable[[CallGraphError], bool]
        ] = {
            ErrorCategory.DATABASE: self._handle_database_error,
            ErrorCategory.RUST_BRIDGE: self._handle_rust_bridge_error,
            ErrorCategory.FILE_SYSTEM: self._handle_file_system_error,
            ErrorCategory.TIMEOUT: self._handle_timeout_error,
        }

    def handle_error(
        self,
        error: Exception | CallGraphError,
        context: ErrorContext | None = None,
        auto_recover: bool = True,
    ) -> CallGraphError | None:
        """
        Handle an error with appropriate logging, recovery, and monitoring.

        Args:
            error: The error that occurred
            context: Additional context for the error
            auto_recover: Whether to attempt automatic recovery

        Returns:
            CallGraphError if the error should be propagated, None if recovered
        """
        # Convert to CallGraphError if needed
        if not isinstance(error, CallGraphError):
            call_graph_error = self._classify_error(error, context)
        else:
            call_graph_error = error

        # Record statistics
        self.stats.record_error(
            call_graph_error.category,
            call_graph_error.severity,
            call_graph_error.is_recoverable,
            False,  # This is the initial error, not a retry
        )
        self.stats.update_error_rate()

        # Log the error
        self._log_error(call_graph_error)

        # Attempt recovery if enabled
        if auto_recover and call_graph_error.is_recoverable:
            recovery_strategy = self._recovery_strategies.get(call_graph_error.category)
            if recovery_strategy:
                try:
                    recovery_result = recovery_strategy(call_graph_error)
                    if recovery_result:
                        logger.info(
                            "Error recovered successfully",
                            category=call_graph_error.category.value,
                            recovery_strategy=recovery_strategy.__name__,
                        )
                        return None  # Error was recovered
                except Exception as recovery_error:
                    logger.error(
                        "Error recovery failed",
                        category=call_graph_error.category.value,
                        recovery_error=str(recovery_error),
                    )

        # Error could not be recovered
        return call_graph_error

    def _classify_error(
        self, error: Exception, context: ErrorContext | None
    ) -> CallGraphError:
        """Classify a generic exception into a CallGraphError."""
        error_str = str(error).lower()

        # Database errors
        if any(
            keyword in error_str
            for keyword in ["database", "connection", "sql", "postgres"]
        ):
            return CallGraphError(
                message=str(error),
                category=ErrorCategory.DATABASE,
                severity=ErrorSeverity.HIGH,
                context=context,
                original_error=error,
                is_recoverable=True,
                is_retryable=True,
            )

        # File system errors
        elif any(
            keyword in error_str
            for keyword in ["file", "directory", "permission", "not found"]
        ):
            return CallGraphError(
                message=str(error),
                category=ErrorCategory.FILE_SYSTEM,
                severity=ErrorSeverity.MEDIUM,
                context=context,
                original_error=error,
                is_recoverable=True,
                is_retryable=False,
            )

        # Validation errors
        elif any(
            keyword in error_str for keyword in ["validation", "invalid", "required"]
        ):
            return CallGraphError(
                message=str(error),
                category=ErrorCategory.VALIDATION,
                severity=ErrorSeverity.LOW,
                context=context,
                original_error=error,
                is_recoverable=False,
                is_retryable=False,
            )

        # Resource errors
        elif any(
            keyword in error_str for keyword in ["memory", "space", "limit", "capacity"]
        ):
            return CallGraphError(
                message=str(error),
                category=ErrorCategory.RESOURCE,
                severity=ErrorSeverity.HIGH,
                context=context,
                original_error=error,
                is_recoverable=False,
                is_retryable=False,
            )

        # Timeout errors
        elif any(
            keyword in error_str for keyword in ["timeout", "timed out", "deadline"]
        ):
            return CallGraphError(
                message=str(error),
                category=ErrorCategory.TIMEOUT,
                severity=ErrorSeverity.MEDIUM,
                context=context,
                original_error=error,
                is_recoverable=True,
                is_retryable=True,
            )

        # Default to critical error
        else:
            return CallGraphError(
                message=str(error),
                category=ErrorCategory.CRITICAL,
                severity=ErrorSeverity.CRITICAL,
                context=context,
                original_error=error,
                is_recoverable=False,
                is_retryable=False,
            )

    def _log_error(self, error: CallGraphError) -> None:
        """Log an error with appropriate level and detail."""
        log_data = error.to_dict()

        if self.enable_detailed_logging:
            log_method = {
                ErrorSeverity.LOW: logger.info,
                ErrorSeverity.MEDIUM: logger.warning,
                ErrorSeverity.HIGH: logger.error,
                ErrorSeverity.CRITICAL: logger.critical,
            }.get(error.severity, logger.error)
        else:
            log_method = logger.error

        log_method("Call graph error occurred", **log_data)

    def _handle_database_error(self, error: CallGraphError) -> bool:
        """Handle database-related errors."""
        logger.info("Attempting database error recovery", error_message=error.message)
        # Placeholder for database recovery logic
        # Could include connection pool reset, retry with backoff, etc.
        return False

    def _handle_rust_bridge_error(self, error: CallGraphError) -> bool:
        """Handle Rust bridge communication errors."""
        logger.info(
            "Attempting Rust bridge error recovery", error_message=error.message
        )
        # Placeholder for bridge recovery logic
        # Could include bridge reinitialization, process restart, etc.
        return False

    def _handle_file_system_error(self, error: CallGraphError) -> bool:
        """Handle file system errors."""
        logger.info(
            "Attempting file system error recovery", error_message=error.message
        )
        # Placeholder for file system recovery logic
        # Could include path validation, permission checks, etc.
        return False

    def _handle_timeout_error(self, error: CallGraphError) -> bool:
        """Handle timeout errors."""
        logger.info("Attempting timeout error recovery", error_message=error.message)
        # Placeholder for timeout recovery logic
        # Could include operation retry with increased timeout, etc.
        return False

    async def with_retry(
        self,
        operation: Callable[[], Any],
        context: ErrorContext,
        max_retries: int = 3,
        backoff_factor: float = 1.0,
        timeout: float | None = None,
    ) -> Any:
        """
        Execute an operation with retry logic and error handling.

        Args:
            operation: Async function to execute
            context: Error context for logging
            max_retries: Maximum number of retry attempts
            backoff_factor: Multiplier for retry delay
            timeout: Timeout for each attempt

        Returns:
            Result of the operation

        Raises:
            CallGraphError: If all retries are exhausted
        """
        last_error = None

        for attempt in range(max_retries + 1):
            try:
                if timeout:
                    import asyncio

                    return await asyncio.wait_for(operation(), timeout=timeout)
                else:
                    return await operation()

            except Exception as e:
                last_error = e

                # Classify and handle the error
                call_graph_error = self._classify_error(e, context)

                if attempt < max_retries and call_graph_error.is_retryable:
                    # Record retry attempt
                    self.stats.record_error(
                        call_graph_error.category,
                        call_graph_error.severity,
                        is_retry=True,
                    )

                    # Calculate backoff delay
                    delay = backoff_factor * (2**attempt)

                    logger.warning(
                        "Operation failed, retrying",
                        attempt=attempt + 1,
                        max_retries=max_retries,
                        delay_seconds=delay,
                        error=str(e),
                        context=context.to_dict(),
                    )

                    # Wait before retry
                    import asyncio

                    await asyncio.sleep(delay)
                    continue
                else:
                    # No more retries or error is not retryable
                    final_error = self.handle_error(e, context, auto_recover=False)
                    if final_error:
                        raise final_error.to_http_exception() from e
                    break

        # Should not reach here, but handle gracefully
        if last_error:
            error = CallGraphError(
                message=f"Operation failed after {max_retries} retries: {last_error}",
                category=ErrorCategory.CRITICAL,
                severity=ErrorSeverity.CRITICAL,
                context=context,
                original_error=last_error,
            )
            raise error.to_http_exception() from last_error

        # This should never be reached
        raise RuntimeError("Unexpected state in retry loop")

    def get_stats(self) -> dict[str, Any]:
        """Get current error statistics."""
        return self.stats.to_dict()

    def reset_stats(self) -> None:
        """Reset error statistics."""
        self.stats.reset()

    @asynccontextmanager
    async def error_context(
        self, operation: str, **context_kwargs: Any
    ) -> AsyncIterator[ErrorContext]:
        """
        Context manager for automatic error handling within an operation.

        Usage:
            async with error_handler.error_context("extract_calls", file_path="test.c") as ctx:
                # Perform operation
                result = await some_operation()
        """
        context = ErrorContext(operation=operation, **context_kwargs)

        try:
            yield context
        except Exception as e:
            handled_error = self.handle_error(e, context)
            if handled_error:
                raise handled_error.to_http_exception() from e


# Global error handler instance
global_error_handler = ErrorHandler()


def create_error_context(
    operation: str,
    file_path: str | None = None,
    function_name: str | None = None,
    **kwargs: Any,
) -> ErrorContext:
    """Convenience function to create an error context."""
    return ErrorContext(
        operation=operation, file_path=file_path, function_name=function_name, **kwargs
    )


def handle_extraction_error(
    error: Exception,
    file_path: str | None = None,
    operation: str = "call_graph_extraction",
) -> HTTPException:
    """
    Convenience function to handle call graph extraction errors.

    Args:
        error: The exception that occurred
        file_path: Path of the file being processed
        operation: Name of the operation that failed

    Returns:
        HTTPException to be raised
    """
    context = create_error_context(operation, file_path=file_path)
    handled_error = global_error_handler.handle_error(error, context)

    if handled_error:
        return handled_error.to_http_exception()
    else:
        # Error was recovered, return a generic error
        return HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={"error": "unknown", "message": "An unexpected error occurred"},
        )
