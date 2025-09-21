//! CallEdge data model for representing function call relationships.
//!
//! This module defines the core CallEdge struct that represents a single call
//! relationship between two functions in the call graph. Each CallEdge captures
//! the complete context of a function call including the caller, callee, call site
//! information, call type classification, and confidence level.

use serde::{Deserialize, Serialize};
use std::fmt;

use crate::types::{CallType, ConfidenceLevel};

/// Represents a single call relationship between two functions.
///
/// A CallEdge captures all relevant information about a function call,
/// including the calling and called functions, the exact location where
/// the call occurs, the type of call (direct, indirect, etc.), and the
/// confidence level of the call detection.
///
/// # Examples
///
/// ```rust
/// use kcs_graph::{CallEdgeModel, CallTypeEnum, ConfidenceLevel};
///
/// let edge = CallEdgeModel::new(
///     "main_function".to_string(),
///     "helper_function".to_string(),
///     "/path/to/file.c".to_string(),
///     42,
///     CallTypeEnum::Direct,
///     ConfidenceLevel::High,
///     false,
/// );
///
/// assert_eq!(edge.caller_name(), "main_function");
/// assert_eq!(edge.callee_name(), "helper_function");
/// assert_eq!(edge.line_number(), 42);
/// ```
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CallEdge {
    /// The name of the calling function
    caller_name: String,

    /// The name of the called function
    callee_name: String,

    /// The file path where the call occurs
    file_path: String,

    /// The line number where the call occurs
    line_number: u32,

    /// The type of function call (direct, indirect, callback, etc.)
    call_type: CallType,

    /// The confidence level of call detection
    confidence: ConfidenceLevel,

    /// Whether the call is conditional (inside if/switch/loop)
    conditional: bool,

    /// Optional configuration guard (e.g., "CONFIG_DEBUG")
    config_guard: Option<String>,

    /// Additional metadata as key-value pairs
    metadata: std::collections::HashMap<String, String>,
}

impl CallEdge {
    /// Creates a new CallEdge with the specified parameters.
    ///
    /// # Arguments
    ///
    /// * `caller_name` - Name of the calling function
    /// * `callee_name` - Name of the called function
    /// * `file_path` - File path where the call occurs
    /// * `line_number` - Line number where the call occurs
    /// * `call_type` - Type of the function call
    /// * `confidence` - Confidence level of call detection
    /// * `conditional` - Whether the call is conditional
    ///
    /// # Examples
    ///
    /// ```rust
    /// use kcs_graph::{CallEdgeModel, CallTypeEnum, ConfidenceLevel};
    ///
    /// let edge = CallEdgeModel::new(
    ///     "sys_open".to_string(),
    ///     "generic_file_open".to_string(),
    ///     "fs/open.c".to_string(),
    ///     123,
    ///     CallTypeEnum::Direct,
    ///     ConfidenceLevel::High,
    ///     false,
    ///     );
    /// ```
    pub fn new(
        caller_name: String,
        callee_name: String,
        file_path: String,
        line_number: u32,
        call_type: CallType,
        confidence: ConfidenceLevel,
        conditional: bool,
    ) -> Self {
        Self {
            caller_name,
            callee_name,
            file_path,
            line_number,
            call_type,
            confidence,
            conditional,
            config_guard: None,
            metadata: std::collections::HashMap::new(),
        }
    }

    /// Creates a new CallEdge builder for fluent construction.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use kcs_graph::{CallEdgeModel, CallTypeEnum, ConfidenceLevel};
    ///
    /// let edge = CallEdgeModel::builder()
    ///     .caller("main")
    ///     .callee("helper")
    ///     .file_path("main.c")
    ///     .line_number(42)
    ///     .call_type(CallTypeEnum::Direct)
    ///     .confidence(ConfidenceLevel::High)
    ///     .conditional(false)
    ///     .config_guard("CONFIG_DEBUG")
    ///     .build();
    /// ```
    pub fn builder() -> CallEdgeBuilder {
        CallEdgeBuilder::default()
    }

    // Accessor methods

    /// Returns the name of the calling function.
    pub fn caller_name(&self) -> &str {
        &self.caller_name
    }

    /// Returns the name of the called function.
    pub fn callee_name(&self) -> &str {
        &self.callee_name
    }

    /// Returns the file path where the call occurs.
    pub fn file_path(&self) -> &str {
        &self.file_path
    }

    /// Returns the line number where the call occurs.
    pub fn line_number(&self) -> u32 {
        self.line_number
    }

    /// Returns the type of function call.
    pub fn call_type(&self) -> CallType {
        self.call_type
    }

    /// Returns the confidence level of call detection.
    pub fn confidence(&self) -> ConfidenceLevel {
        self.confidence
    }

    /// Returns whether the call is conditional.
    pub fn is_conditional(&self) -> bool {
        self.conditional
    }

    /// Returns the configuration guard, if any.
    pub fn config_guard(&self) -> Option<&str> {
        self.config_guard.as_deref()
    }

    /// Returns a reference to the metadata map.
    pub fn metadata(&self) -> &std::collections::HashMap<String, String> {
        &self.metadata
    }

    // Mutator methods

    /// Sets the configuration guard for this call edge.
    pub fn set_config_guard(&mut self, guard: Option<String>) {
        self.config_guard = guard;
    }

    /// Adds a metadata key-value pair.
    pub fn add_metadata(&mut self, key: String, value: String) {
        self.metadata.insert(key, value);
    }

    /// Removes a metadata key.
    pub fn remove_metadata(&mut self, key: &str) -> Option<String> {
        self.metadata.remove(key)
    }

    /// Clears all metadata.
    pub fn clear_metadata(&mut self) {
        self.metadata.clear();
    }

    // Utility methods

    /// Returns true if this call edge represents a direct function call.
    pub fn is_direct_call(&self) -> bool {
        matches!(self.call_type, CallType::Direct)
    }

    /// Returns true if this call edge represents an indirect function call.
    pub fn is_indirect_call(&self) -> bool {
        matches!(self.call_type, CallType::Indirect)
    }

    /// Returns true if this call edge represents a macro call.
    pub fn is_macro_call(&self) -> bool {
        matches!(self.call_type, CallType::Macro)
    }

    /// Returns true if this call edge represents a callback.
    pub fn is_callback(&self) -> bool {
        matches!(self.call_type, CallType::Callback)
    }

    /// Returns true if this call edge has high confidence.
    pub fn is_high_confidence(&self) -> bool {
        matches!(self.confidence, ConfidenceLevel::High)
    }

    /// Returns true if this call edge has medium confidence.
    pub fn is_medium_confidence(&self) -> bool {
        matches!(self.confidence, ConfidenceLevel::Medium)
    }

    /// Returns true if this call edge has low confidence.
    pub fn is_low_confidence(&self) -> bool {
        matches!(self.confidence, ConfidenceLevel::Low)
    }

    /// Creates a unique identifier for this call edge.
    ///
    /// The identifier is based on the caller name, callee name, file path,
    /// and line number, making it suitable for deduplication and indexing.
    pub fn edge_id(&self) -> String {
        format!(
            "{}->{}:{}:{}",
            self.caller_name, self.callee_name, self.file_path, self.line_number
        )
    }

    /// Returns a display-friendly representation of the call edge.
    pub fn display_name(&self) -> String {
        format!(
            "{} -> {} ({:?}, {:?})",
            self.caller_name, self.callee_name, self.call_type, self.confidence
        )
    }
}

impl fmt::Display for CallEdge {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{} -> {} at {}:{} ({:?}, {:?}{})",
            self.caller_name,
            self.callee_name,
            self.file_path,
            self.line_number,
            self.call_type,
            self.confidence,
            if self.conditional {
                ", conditional"
            } else {
                ""
            }
        )
    }
}

/// Builder for constructing CallEdge instances.
///
/// Provides a fluent interface for creating CallEdge instances with
/// optional parameters and validation.
#[derive(Debug, Default)]
pub struct CallEdgeBuilder {
    caller_name: Option<String>,
    callee_name: Option<String>,
    file_path: Option<String>,
    line_number: Option<u32>,
    call_type: Option<CallType>,
    confidence: Option<ConfidenceLevel>,
    conditional: bool,
    config_guard: Option<String>,
    metadata: std::collections::HashMap<String, String>,
}

impl CallEdgeBuilder {
    /// Sets the caller function name.
    pub fn caller<S: Into<String>>(mut self, name: S) -> Self {
        self.caller_name = Some(name.into());
        self
    }

    /// Sets the callee function name.
    pub fn callee<S: Into<String>>(mut self, name: S) -> Self {
        self.callee_name = Some(name.into());
        self
    }

    /// Sets the file path.
    pub fn file_path<S: Into<String>>(mut self, path: S) -> Self {
        self.file_path = Some(path.into());
        self
    }

    /// Sets the line number.
    pub fn line_number(mut self, line: u32) -> Self {
        self.line_number = Some(line);
        self
    }

    /// Sets the call type.
    pub fn call_type(mut self, call_type: CallType) -> Self {
        self.call_type = Some(call_type);
        self
    }

    /// Sets the confidence level.
    pub fn confidence(mut self, confidence: ConfidenceLevel) -> Self {
        self.confidence = Some(confidence);
        self
    }

    /// Sets whether the call is conditional.
    pub fn conditional(mut self, conditional: bool) -> Self {
        self.conditional = conditional;
        self
    }

    /// Sets the configuration guard.
    pub fn config_guard<S: Into<String>>(mut self, guard: S) -> Self {
        self.config_guard = Some(guard.into());
        self
    }

    /// Adds metadata.
    pub fn metadata<K: Into<String>, V: Into<String>>(mut self, key: K, value: V) -> Self {
        self.metadata.insert(key.into(), value.into());
        self
    }

    /// Builds the CallEdge instance.
    ///
    /// # Panics
    ///
    /// Panics if any required field is missing.
    pub fn build(self) -> CallEdge {
        CallEdge {
            caller_name: self.caller_name.expect("caller_name is required"),
            callee_name: self.callee_name.expect("callee_name is required"),
            file_path: self.file_path.expect("file_path is required"),
            line_number: self.line_number.expect("line_number is required"),
            call_type: self.call_type.expect("call_type is required"),
            confidence: self.confidence.expect("confidence is required"),
            conditional: self.conditional,
            config_guard: self.config_guard,
            metadata: self.metadata,
        }
    }

    /// Attempts to build the CallEdge instance, returning an error if validation fails.
    pub fn try_build(self) -> Result<CallEdge, String> {
        let caller_name = self.caller_name.ok_or("caller_name is required")?;
        let callee_name = self.callee_name.ok_or("callee_name is required")?;
        let file_path = self.file_path.ok_or("file_path is required")?;
        let line_number = self.line_number.ok_or("line_number is required")?;
        let call_type = self.call_type.ok_or("call_type is required")?;
        let confidence = self.confidence.ok_or("confidence is required")?;

        // Validation
        if caller_name.is_empty() {
            return Err("caller_name cannot be empty".to_string());
        }
        if callee_name.is_empty() {
            return Err("callee_name cannot be empty".to_string());
        }
        if file_path.is_empty() {
            return Err("file_path cannot be empty".to_string());
        }
        if line_number == 0 {
            return Err("line_number must be greater than 0".to_string());
        }

        Ok(CallEdge {
            caller_name,
            callee_name,
            file_path,
            line_number,
            call_type,
            confidence,
            conditional: self.conditional,
            config_guard: self.config_guard,
            metadata: self.metadata,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_call_edge_new() {
        let edge = CallEdge::new(
            "main".to_string(),
            "helper".to_string(),
            "main.c".to_string(),
            42,
            CallType::Direct,
            ConfidenceLevel::High,
            false,
        );

        assert_eq!(edge.caller_name(), "main");
        assert_eq!(edge.callee_name(), "helper");
        assert_eq!(edge.file_path(), "main.c");
        assert_eq!(edge.line_number(), 42);
        assert_eq!(edge.call_type(), CallType::Direct);
        assert_eq!(edge.confidence(), ConfidenceLevel::High);
        assert!(!edge.is_conditional());
        assert!(edge.config_guard().is_none());
    }

    #[test]
    fn test_call_edge_builder() {
        let edge = CallEdge::builder()
            .caller("sys_open")
            .callee("generic_file_open")
            .file_path("fs/open.c")
            .line_number(123)
            .call_type(CallType::Direct)
            .confidence(ConfidenceLevel::High)
            .conditional(true)
            .config_guard("CONFIG_DEBUG")
            .metadata("context", "file_operation")
            .build();

        assert_eq!(edge.caller_name(), "sys_open");
        assert_eq!(edge.callee_name(), "generic_file_open");
        assert_eq!(edge.file_path(), "fs/open.c");
        assert_eq!(edge.line_number(), 123);
        assert!(edge.is_conditional());
        assert_eq!(edge.config_guard(), Some("CONFIG_DEBUG"));
        assert_eq!(edge.metadata().get("context"), Some(&"file_operation".to_string()));
    }

    #[test]
    fn test_call_edge_utility_methods() {
        let edge = CallEdge::new(
            "caller".to_string(),
            "callee".to_string(),
            "file.c".to_string(),
            1,
            CallType::Direct,
            ConfidenceLevel::High,
            false,
        );

        assert!(edge.is_direct_call());
        assert!(!edge.is_indirect_call());
        assert!(!edge.is_macro_call());
        assert!(!edge.is_callback());
        assert!(edge.is_high_confidence());
        assert!(!edge.is_medium_confidence());
        assert!(!edge.is_low_confidence());
    }

    #[test]
    fn test_call_edge_id() {
        let edge = CallEdge::new(
            "main".to_string(),
            "helper".to_string(),
            "main.c".to_string(),
            42,
            CallType::Direct,
            ConfidenceLevel::High,
            false,
        );

        assert_eq!(edge.edge_id(), "main->helper:main.c:42");
    }

    #[test]
    fn test_call_edge_metadata() {
        let mut edge = CallEdge::new(
            "caller".to_string(),
            "callee".to_string(),
            "file.c".to_string(),
            1,
            CallType::Direct,
            ConfidenceLevel::High,
            false,
        );

        edge.add_metadata("key1".to_string(), "value1".to_string());
        edge.add_metadata("key2".to_string(), "value2".to_string());

        assert_eq!(edge.metadata().get("key1"), Some(&"value1".to_string()));
        assert_eq!(edge.metadata().get("key2"), Some(&"value2".to_string()));

        let removed = edge.remove_metadata("key1");
        assert_eq!(removed, Some("value1".to_string()));
        assert!(edge.metadata().get("key1").is_none());

        edge.clear_metadata();
        assert!(edge.metadata().is_empty());
    }

    #[test]
    fn test_builder_validation() {
        let result = CallEdge::builder()
            .caller("")
            .callee("helper")
            .file_path("file.c")
            .line_number(1)
            .call_type(CallType::Direct)
            .confidence(ConfidenceLevel::High)
            .try_build();

        assert!(result.is_err());
        assert!(result.unwrap_err().contains("caller_name cannot be empty"));
    }

    #[test]
    fn test_display() {
        let edge = CallEdge::new(
            "main".to_string(),
            "helper".to_string(),
            "main.c".to_string(),
            42,
            CallType::Direct,
            ConfidenceLevel::High,
            true,
        );

        let display = format!("{}", edge);
        assert!(display.contains("main -> helper"));
        assert!(display.contains("main.c:42"));
        assert!(display.contains("Direct"));
        assert!(display.contains("High"));
        assert!(display.contains("conditional"));
    }
}
