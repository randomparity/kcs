//! Error handling and logging for call graph extraction pipeline.
//!
//! This module provides comprehensive error handling, recovery strategies, and
//! structured logging for the call graph extraction pipeline. It defines
//! standardized error types, logging utilities, and recovery mechanisms.

use anyhow;
use serde::{Deserialize, Serialize};
use std::fmt;
use std::path::Path;
use thiserror::Error;
use tracing::{debug, info, warn};

/// Configuration for error handling and logging behavior.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ErrorHandlingConfig {
    /// Continue processing other files if one file fails
    pub continue_on_file_error: bool,
    /// Maximum number of retries for transient errors
    pub max_retries: usize,
    /// Enable detailed error context in logs
    pub detailed_error_context: bool,
    /// Log level for extraction progress
    pub progress_log_level: LogLevel,
    /// Log level for performance metrics
    pub performance_log_level: LogLevel,
}

impl Default for ErrorHandlingConfig {
    fn default() -> Self {
        Self {
            continue_on_file_error: true,
            max_retries: 3,
            detailed_error_context: true,
            progress_log_level: LogLevel::Info,
            performance_log_level: LogLevel::Debug,
        }
    }
}

/// Log levels for configurable logging.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum LogLevel {
    Error,
    Warn,
    Info,
    Debug,
    Trace,
}

/// Comprehensive error types for call graph extraction.
#[derive(Error, Debug)]
pub enum CallGraphError {
    /// File system related errors
    #[error("File error for '{path}': {source}")]
    FileError {
        path: String,
        #[source]
        source: std::io::Error,
    },

    /// Tree-sitter parsing errors
    #[error("Parse error in '{file}' at line {line}: {message}")]
    ParseError {
        file: String,
        line: usize,
        message: String,
    },

    /// AST traversal errors
    #[error("AST traversal error in '{file}': {message}")]
    AstError { file: String, message: String },

    /// Call detection errors
    #[error("Call detection error in '{file}' at line {line}: {message}")]
    DetectionError {
        file: String,
        line: usize,
        message: String,
    },

    /// Configuration errors
    #[error("Configuration error: {message}")]
    ConfigError { message: String },

    /// Resource exhaustion errors
    #[error("Resource limit exceeded: {message}")]
    ResourceError { message: String },

    /// Database operation errors
    #[error("Database operation failed: {message}")]
    DatabaseError { message: String },

    /// Validation errors
    #[error("Validation error: {message}")]
    ValidationError { message: String },

    /// Transient errors that can be retried
    #[error("Transient error (attempt {attempt}/{max_attempts}): {message}")]
    TransientError {
        attempt: usize,
        max_attempts: usize,
        message: String,
    },

    /// Critical errors that should halt processing
    #[error("Critical error: {message}")]
    CriticalError { message: String },
}

impl CallGraphError {
    /// Check if this error is recoverable and processing should continue.
    pub fn is_recoverable(&self) -> bool {
        match self {
            CallGraphError::FileError { .. } => true,
            CallGraphError::ParseError { .. } => true,
            CallGraphError::AstError { .. } => true,
            CallGraphError::DetectionError { .. } => true,
            CallGraphError::TransientError { .. } => true,
            CallGraphError::ConfigError { .. } => false,
            CallGraphError::ResourceError { .. } => false,
            CallGraphError::DatabaseError { .. } => false,
            CallGraphError::ValidationError { .. } => false,
            CallGraphError::CriticalError { .. } => false,
        }
    }

    /// Check if this error should be retried.
    pub fn is_retryable(&self) -> bool {
        matches!(self, CallGraphError::TransientError { .. })
    }

    /// Get the error category for metrics and logging.
    pub fn category(&self) -> &'static str {
        match self {
            CallGraphError::FileError { .. } => "file",
            CallGraphError::ParseError { .. } => "parse",
            CallGraphError::AstError { .. } => "ast",
            CallGraphError::DetectionError { .. } => "detection",
            CallGraphError::ConfigError { .. } => "config",
            CallGraphError::ResourceError { .. } => "resource",
            CallGraphError::DatabaseError { .. } => "database",
            CallGraphError::ValidationError { .. } => "validation",
            CallGraphError::TransientError { .. } => "transient",
            CallGraphError::CriticalError { .. } => "critical",
        }
    }
}

impl From<anyhow::Error> for CallGraphError {
    fn from(err: anyhow::Error) -> Self {
        CallGraphError::CriticalError {
            message: err.to_string(),
        }
    }
}

/// Statistics for error tracking and reporting.
#[derive(Debug, Clone, Default)]
pub struct ErrorStats {
    /// Total number of errors encountered
    pub total_errors: usize,
    /// Errors by category
    pub errors_by_category: std::collections::HashMap<String, usize>,
    /// Number of files that failed processing
    pub failed_files: usize,
    /// Number of retries attempted
    pub retries_attempted: usize,
    /// Number of recoverable errors
    pub recoverable_errors: usize,
    /// Number of critical errors
    pub critical_errors: usize,
}

impl ErrorStats {
    /// Record an error occurrence.
    pub fn record_error(&mut self, error: &CallGraphError) {
        self.total_errors += 1;

        let category = error.category().to_string();
        *self.errors_by_category.entry(category).or_insert(0) += 1;

        if error.is_recoverable() {
            self.recoverable_errors += 1;
        } else {
            self.critical_errors += 1;
        }

        if error.is_retryable() {
            self.retries_attempted += 1;
        }
    }

    /// Record a file processing failure.
    pub fn record_file_failure(&mut self) {
        self.failed_files += 1;
    }

    /// Get error rate as a percentage.
    pub fn error_rate(&self, total_operations: usize) -> f64 {
        if total_operations == 0 {
            0.0
        } else {
            (self.total_errors as f64 / total_operations as f64) * 100.0
        }
    }

    /// Get critical error rate as a percentage.
    pub fn critical_error_rate(&self, total_operations: usize) -> f64 {
        if total_operations == 0 {
            0.0
        } else {
            (self.critical_errors as f64 / total_operations as f64) * 100.0
        }
    }
}

/// Context information for enhanced error reporting.
#[derive(Debug, Clone)]
pub struct ErrorContext {
    /// File being processed
    pub file_path: Option<String>,
    /// Function being analyzed
    pub function_name: Option<String>,
    /// Line number in source code
    pub line_number: Option<usize>,
    /// Column number in source code
    pub column_number: Option<usize>,
    /// Additional context data
    pub metadata: std::collections::HashMap<String, String>,
}

impl Default for ErrorContext {
    fn default() -> Self {
        Self::new()
    }
}

impl ErrorContext {
    /// Create a new error context.
    pub fn new() -> Self {
        Self {
            file_path: None,
            function_name: None,
            line_number: None,
            column_number: None,
            metadata: std::collections::HashMap::new(),
        }
    }

    /// Set the file path context.
    pub fn with_file<P: AsRef<Path>>(mut self, file_path: P) -> Self {
        self.file_path = Some(file_path.as_ref().to_string_lossy().to_string());
        self
    }

    /// Set the function name context.
    pub fn with_function(mut self, function_name: String) -> Self {
        self.function_name = Some(function_name);
        self
    }

    /// Set the line number context.
    pub fn with_line(mut self, line_number: usize) -> Self {
        self.line_number = Some(line_number);
        self
    }

    /// Set the column number context.
    pub fn with_column(mut self, column_number: usize) -> Self {
        self.column_number = Some(column_number);
        self
    }

    /// Add metadata to the context.
    pub fn with_metadata(mut self, key: String, value: String) -> Self {
        self.metadata.insert(key, value);
        self
    }
}

impl fmt::Display for ErrorContext {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if let Some(file) = &self.file_path {
            write!(f, "file: {}", file)?;

            if let Some(line) = self.line_number {
                write!(f, ":{}", line)?;

                if let Some(col) = self.column_number {
                    write!(f, ":{}", col)?;
                }
            }

            if let Some(func) = &self.function_name {
                write!(f, " in function '{}'", func)?;
            }
        }

        if !self.metadata.is_empty() {
            let metadata_str: Vec<String> =
                self.metadata.iter().map(|(k, v)| format!("{}={}", k, v)).collect();
            write!(f, " [{}]", metadata_str.join(", "))?;
        }

        Ok(())
    }
}

/// Enhanced error handler with recovery strategies and logging.
pub struct ErrorHandler {
    config: ErrorHandlingConfig,
    stats: ErrorStats,
}

impl ErrorHandler {
    /// Create a new error handler with the given configuration.
    pub fn new(config: ErrorHandlingConfig) -> Self {
        Self {
            config,
            stats: ErrorStats::default(),
        }
    }

    /// Create a new error handler with default configuration.
    pub fn new_default() -> Self {
        Self::new(ErrorHandlingConfig::default())
    }

    /// Handle an error with appropriate logging and recovery strategy.
    pub fn handle_error(
        &mut self,
        error: CallGraphError,
        context: ErrorContext,
    ) -> Result<(), CallGraphError> {
        // Record error in statistics
        self.stats.record_error(&error);

        // Log the error with appropriate level and context
        self.log_error(&error, &context);

        // Determine recovery strategy
        if error.is_recoverable() && self.config.continue_on_file_error {
            warn!(
                target = "call_graph_extraction",
                error = ?error,
                context = %context,
                "Recoverable error encountered, continuing processing"
            );
            return Ok(());
        }

        // Non-recoverable error - propagate it
        Err(error)
    }

    /// Handle file processing error with retry logic.
    pub async fn handle_file_error<F, T>(
        &mut self,
        file_path: &Path,
        mut operation: F,
    ) -> Result<T, CallGraphError>
    where
        F: FnMut() -> Result<T, CallGraphError>,
    {
        let mut attempts = 0;
        let max_attempts = self.config.max_retries + 1;

        loop {
            attempts += 1;

            match operation() {
                Ok(result) => return Ok(result),
                Err(error) => {
                    if error.is_retryable() && attempts < max_attempts {
                        warn!(
                            target = "call_graph_extraction",
                            file = %file_path.display(),
                            attempt = attempts,
                            max_attempts = max_attempts,
                            error = ?error,
                            "Retrying operation after transient error"
                        );

                        // Simple exponential backoff
                        let delay = std::time::Duration::from_millis(100 * (1 << (attempts - 1)));
                        tokio::time::sleep(delay).await;
                        continue;
                    }

                    // Max retries reached or non-retryable error
                    self.stats.record_file_failure();
                    return Err(error);
                },
            }
        }
    }

    /// Log an error with appropriate level and detail.
    fn log_error(&self, error: &CallGraphError, context: &ErrorContext) {
        let category = error.category();
        let is_recoverable = error.is_recoverable();
        let is_retryable = error.is_retryable();

        match error {
            CallGraphError::CriticalError { .. } => {
                tracing::error!(
                    target = "call_graph_extraction",
                    error = ?error,
                    context = %context,
                    category = category,
                    is_recoverable = is_recoverable,
                    is_retryable = is_retryable,
                    "Critical error in call graph extraction"
                );
            },
            CallGraphError::ConfigError { .. } | CallGraphError::ValidationError { .. } => {
                tracing::error!(
                    target = "call_graph_extraction",
                    error = ?error,
                    context = %context,
                    category = category,
                    is_recoverable = is_recoverable,
                    is_retryable = is_retryable,
                    "Configuration or validation error"
                );
            },
            CallGraphError::ResourceError { .. } | CallGraphError::DatabaseError { .. } => {
                tracing::error!(
                    target = "call_graph_extraction",
                    error = ?error,
                    context = %context,
                    category = category,
                    is_recoverable = is_recoverable,
                    is_retryable = is_retryable,
                    "Resource or database error"
                );
            },
            _ => {
                if self.config.detailed_error_context {
                    warn!(
                        target = "call_graph_extraction",
                        error = ?error,
                        context = %context,
                        category = category,
                        is_recoverable = is_recoverable,
                        is_retryable = is_retryable,
                        "Processing error encountered"
                    );
                } else {
                    warn!(
                        target = "call_graph_extraction",
                        error_message = %error,
                        category = category,
                        is_recoverable = is_recoverable,
                        is_retryable = is_retryable,
                        "Processing error encountered"
                    );
                }
            },
        }
    }

    /// Log progress information.
    pub fn log_progress(&self, message: &str, metadata: &[(&str, &dyn fmt::Display)]) {
        if self.config.progress_log_level == LogLevel::Info {
            let mut fields = Vec::new();
            for (key, value) in metadata {
                fields.push(format!("{}={}", key, value));
            }

            info!(target = "call_graph_extraction", metadata = fields.join(", "), "{}", message);
        } else if self.config.progress_log_level == LogLevel::Debug {
            debug!(target = "call_graph_extraction", "{}", message);
        }
    }

    /// Log performance metrics.
    pub fn log_performance(
        &self,
        operation: &str,
        duration_ms: u64,
        _metadata: &[(&str, &dyn fmt::Display)],
    ) {
        if self.config.performance_log_level == LogLevel::Debug {
            debug!(
                target = "call_graph_performance",
                operation = operation,
                duration_ms = duration_ms,
                "Performance metric"
            );
        } else if self.config.performance_log_level == LogLevel::Info {
            info!(
                target = "call_graph_performance",
                operation = operation,
                duration_ms = duration_ms,
                "Performance metric"
            );
        }
    }

    /// Get current error statistics.
    pub fn get_stats(&self) -> &ErrorStats {
        &self.stats
    }

    /// Reset error statistics.
    pub fn reset_stats(&mut self) {
        self.stats = ErrorStats::default();
    }

    /// Create a contextual error for file operations.
    pub fn file_error<P: AsRef<Path>>(file_path: P, source: std::io::Error) -> CallGraphError {
        CallGraphError::FileError {
            path: file_path.as_ref().to_string_lossy().to_string(),
            source,
        }
    }

    /// Create a contextual error for parsing operations.
    pub fn parse_error<P: AsRef<Path>>(
        file_path: P,
        line: usize,
        message: String,
    ) -> CallGraphError {
        CallGraphError::ParseError {
            file: file_path.as_ref().to_string_lossy().to_string(),
            line,
            message,
        }
    }

    /// Create a contextual error for AST operations.
    pub fn ast_error<P: AsRef<Path>>(file_path: P, message: String) -> CallGraphError {
        CallGraphError::AstError {
            file: file_path.as_ref().to_string_lossy().to_string(),
            message,
        }
    }

    /// Create a contextual error for call detection.
    pub fn detection_error<P: AsRef<Path>>(
        file_path: P,
        line: usize,
        message: String,
    ) -> CallGraphError {
        CallGraphError::DetectionError {
            file: file_path.as_ref().to_string_lossy().to_string(),
            line,
            message,
        }
    }
}

/// Convenience macro for creating error contexts.
#[macro_export]
macro_rules! error_context {
    ($file:expr) => {
        ErrorContext::new().with_file($file)
    };
    ($file:expr, $func:expr) => {
        ErrorContext::new().with_file($file).with_function($func.to_string())
    };
    ($file:expr, $func:expr, $line:expr) => {
        ErrorContext::new()
            .with_file($file)
            .with_function($func.to_string())
            .with_line($line)
    };
    ($file:expr, $func:expr, $line:expr, $col:expr) => {
        ErrorContext::new()
            .with_file($file)
            .with_function($func.to_string())
            .with_line($line)
            .with_column($col)
    };
}

/// Result type alias for call graph operations.
pub type CallGraphResult<T> = Result<T, CallGraphError>;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_error_categorization() {
        let file_error = CallGraphError::FileError {
            path: "test.c".to_string(),
            source: std::io::Error::new(std::io::ErrorKind::NotFound, "File not found"),
        };

        assert!(file_error.is_recoverable());
        assert!(!file_error.is_retryable());
        assert_eq!(file_error.category(), "file");

        let critical_error = CallGraphError::CriticalError {
            message: "System failure".to_string(),
        };

        assert!(!critical_error.is_recoverable());
        assert!(!critical_error.is_retryable());
        assert_eq!(critical_error.category(), "critical");
    }

    #[test]
    fn test_error_stats() {
        let mut stats = ErrorStats::default();

        let file_error = CallGraphError::FileError {
            path: "test.c".to_string(),
            source: std::io::Error::new(std::io::ErrorKind::NotFound, "File not found"),
        };

        stats.record_error(&file_error);
        assert_eq!(stats.total_errors, 1);
        assert_eq!(stats.recoverable_errors, 1);
        assert_eq!(stats.critical_errors, 0);
        assert_eq!(stats.errors_by_category.get("file"), Some(&1));
    }

    #[test]
    fn test_error_context() {
        let context = ErrorContext::new()
            .with_file("test.c")
            .with_function("main".to_string())
            .with_line(42)
            .with_column(10)
            .with_metadata("parser".to_string(), "tree-sitter".to_string());

        let context_str = format!("{}", context);
        assert!(context_str.contains("test.c"));
        assert!(context_str.contains("42"));
        assert!(context_str.contains("10"));
        assert!(context_str.contains("main"));
        assert!(context_str.contains("parser=tree-sitter"));
    }

    #[tokio::test]
    async fn test_error_handler_retry() {
        let mut handler = ErrorHandler::new_default();
        let test_path = Path::new("test.c");

        let mut attempt_count = 0;
        let result = handler
            .handle_file_error(test_path, || {
                attempt_count += 1;
                if attempt_count < 3 {
                    Err(CallGraphError::TransientError {
                        attempt: attempt_count,
                        max_attempts: 3,
                        message: "Temporary failure".to_string(),
                    })
                } else {
                    Ok("success")
                }
            })
            .await;

        assert!(result.is_ok());
        assert_eq!(attempt_count, 3);
    }
}
