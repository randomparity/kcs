//! CallPath data model for representing sequences of function calls in a call graph.
//!
//! This module defines the CallPath struct that represents a sequence of function
//! calls from a starting function to an ending function, capturing the complete
//! path through the call graph including all intermediate calls, timing information,
//! and path analysis metrics.

use serde::{Deserialize, Serialize};
use std::fmt;

use crate::call_edge::CallEdge;
use crate::types::{CallType, ConfidenceLevel};

/// Represents a sequence of function calls through the call graph.
///
/// A CallPath captures a complete execution path from a starting function
/// to an ending function, including all intermediate calls, timing information,
/// and analysis metrics for the path traversal.
///
/// # Examples
///
/// ```rust
/// use kcs_graph::{CallPath, CallEdgeModel, CallTypeEnum, ConfidenceLevel};
///
/// let edge1 = CallEdgeModel::new(
///     "main".to_string(),
///     "init_system".to_string(),
///     "main.c".to_string(),
///     10,
///     CallTypeEnum::Direct,
///     ConfidenceLevel::High,
///     false,
/// );
///
/// let edge2 = CallEdgeModel::new(
///     "init_system".to_string(),
///     "setup_hardware".to_string(),
///     "init.c".to_string(),
///     25,
///     CallTypeEnum::Direct,
///     ConfidenceLevel::High,
///     false,
/// );
///
/// let path = CallPath::new(
///     "main".to_string(),
///     "setup_hardware".to_string(),
///     vec![edge1, edge2],
/// );
///
/// assert_eq!(path.start_function(), "main");
/// assert_eq!(path.end_function(), "setup_hardware");
/// assert_eq!(path.depth(), 2);
/// ```
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CallPath {
    /// The starting function name
    start_function: String,

    /// The ending function name
    end_function: String,

    /// Sequence of call edges forming the path
    edges: Vec<CallEdge>,

    /// Total path depth (number of calls)
    depth: usize,

    /// Minimum confidence level across all edges
    min_confidence: ConfidenceLevel,

    /// Maximum confidence level across all edges
    max_confidence: ConfidenceLevel,

    /// Whether any call in the path is conditional
    has_conditional_calls: bool,

    /// Whether the path contains any indirect calls
    has_indirect_calls: bool,

    /// Total execution cost estimate (relative units)
    execution_cost: Option<f64>,

    /// Path discovery timestamp
    discovered_at: Option<chrono::DateTime<chrono::Utc>>,

    /// Additional metadata as key-value pairs
    metadata: std::collections::HashMap<String, String>,
}

impl CallPath {
    /// Creates a new CallPath with the specified parameters.
    ///
    /// # Arguments
    ///
    /// * `start_function` - Name of the starting function
    /// * `end_function` - Name of the ending function
    /// * `edges` - Sequence of call edges forming the path
    ///
    /// # Examples
    ///
    /// ```rust
    /// use kcs_graph::{CallPath, CallEdgeModel, CallTypeEnum, ConfidenceLevel};
    ///
    /// let edge = CallEdgeModel::new(
    ///     "caller".to_string(),
    ///     "callee".to_string(),
    ///     "file.c".to_string(),
    ///     42,
    ///     CallTypeEnum::Direct,
    ///     ConfidenceLevel::High,
    ///     false,
    /// );
    ///
    /// let path = CallPath::new(
    ///     "caller".to_string(),
    ///     "callee".to_string(),
    ///     vec![edge],
    /// );
    /// ```
    pub fn new(start_function: String, end_function: String, edges: Vec<CallEdge>) -> Self {
        let depth = edges.len();

        let (min_confidence, max_confidence) = if edges.is_empty() {
            (ConfidenceLevel::High, ConfidenceLevel::High)
        } else {
            let confidences: Vec<_> = edges.iter().map(|e| e.confidence()).collect();
            let min_conf = confidences.iter().min().copied().unwrap_or(ConfidenceLevel::Low);
            let max_conf = confidences.iter().max().copied().unwrap_or(ConfidenceLevel::High);
            (min_conf, max_conf)
        };

        let has_conditional_calls = edges.iter().any(|e| e.is_conditional());
        let has_indirect_calls = edges.iter().any(|e| e.is_indirect_call());

        Self {
            start_function,
            end_function,
            edges,
            depth,
            min_confidence,
            max_confidence,
            has_conditional_calls,
            has_indirect_calls,
            execution_cost: None,
            discovered_at: Some(chrono::Utc::now()),
            metadata: std::collections::HashMap::new(),
        }
    }

    /// Creates a new CallPath builder for fluent construction.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use kcs_graph::{CallPath, CallEdgeModel, CallTypeEnum, ConfidenceLevel};
    ///
    /// let edge = CallEdgeModel::new(
    ///     "main".to_string(),
    ///     "helper".to_string(),
    ///     "main.c".to_string(),
    ///     42,
    ///     CallTypeEnum::Direct,
    ///     ConfidenceLevel::High,
    ///     false,
    /// );
    ///
    /// let path = CallPath::builder()
    ///     .start_function("main")
    ///     .end_function("helper")
    ///     .add_edge(edge)
    ///     .execution_cost(1.5)
    ///     .metadata("context", "initialization")
    ///     .build();
    /// ```
    pub fn builder() -> CallPathBuilder {
        CallPathBuilder::default()
    }

    // Accessor methods

    /// Returns the name of the starting function.
    pub fn start_function(&self) -> &str {
        &self.start_function
    }

    /// Returns the name of the ending function.
    pub fn end_function(&self) -> &str {
        &self.end_function
    }

    /// Returns a reference to the sequence of call edges.
    pub fn edges(&self) -> &[CallEdge] {
        &self.edges
    }

    /// Returns the path depth (number of calls).
    pub fn depth(&self) -> usize {
        self.depth
    }

    /// Returns the minimum confidence level across all edges.
    pub fn min_confidence(&self) -> ConfidenceLevel {
        self.min_confidence
    }

    /// Returns the maximum confidence level across all edges.
    pub fn max_confidence(&self) -> ConfidenceLevel {
        self.max_confidence
    }

    /// Returns whether any call in the path is conditional.
    pub fn has_conditional_calls(&self) -> bool {
        self.has_conditional_calls
    }

    /// Returns whether the path contains any indirect calls.
    pub fn has_indirect_calls(&self) -> bool {
        self.has_indirect_calls
    }

    /// Returns the execution cost estimate, if available.
    pub fn execution_cost(&self) -> Option<f64> {
        self.execution_cost
    }

    /// Returns the path discovery timestamp.
    pub fn discovered_at(&self) -> Option<chrono::DateTime<chrono::Utc>> {
        self.discovered_at
    }

    /// Returns a reference to the metadata map.
    pub fn metadata(&self) -> &std::collections::HashMap<String, String> {
        &self.metadata
    }

    // Mutator methods

    /// Sets the execution cost estimate.
    pub fn set_execution_cost(&mut self, cost: Option<f64>) {
        self.execution_cost = cost;
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

    // Analysis methods

    /// Returns true if this is a direct path (all edges are direct calls).
    pub fn is_direct_path(&self) -> bool {
        self.edges.iter().all(|e| e.is_direct_call())
    }

    /// Returns true if this path has high confidence throughout.
    pub fn is_high_confidence_path(&self) -> bool {
        matches!(self.min_confidence, ConfidenceLevel::High)
    }

    /// Returns true if this is a simple path (single call).
    pub fn is_simple_path(&self) -> bool {
        self.depth == 1
    }

    /// Returns true if this is a complex path (multiple calls).
    pub fn is_complex_path(&self) -> bool {
        self.depth > 1
    }

    /// Returns all intermediate functions in the path.
    pub fn intermediate_functions(&self) -> Vec<String> {
        if self.edges.is_empty() {
            return Vec::new();
        }

        let mut functions = Vec::new();

        // Add all callee functions except the last one (which is the end function)
        for (i, edge) in self.edges.iter().enumerate() {
            if i < self.edges.len() - 1 {
                functions.push(edge.callee_name().to_string());
            }
        }

        functions
    }

    /// Returns all unique files involved in this path.
    pub fn involved_files(&self) -> Vec<String> {
        let files: std::collections::HashSet<String> =
            self.edges.iter().map(|e| e.file_path().to_string()).collect();

        let mut result: Vec<String> = files.into_iter().collect();
        result.sort();
        result
    }

    /// Returns the call types present in this path.
    pub fn call_types(&self) -> Vec<CallType> {
        let types: std::collections::HashSet<CallType> =
            self.edges.iter().map(|e| e.call_type()).collect();

        let mut result: Vec<CallType> = types.into_iter().collect();
        result.sort_by_key(|t| format!("{:?}", t));
        result
    }

    /// Creates a unique identifier for this call path.
    pub fn path_id(&self) -> String {
        if self.edges.is_empty() {
            return format!("{}->{}:empty", self.start_function, self.end_function);
        }

        let edge_ids: Vec<String> = self.edges.iter().map(|e| e.edge_id()).collect();
        let edges_hash = format!("{:x}", md5::compute(edge_ids.join("|")));

        format!(
            "{}->{}:{}:{}",
            self.start_function,
            self.end_function,
            self.depth,
            &edges_hash[..8]
        )
    }

    /// Returns a display-friendly representation of the call path.
    pub fn display_name(&self) -> String {
        if self.edges.is_empty() {
            return format!("{} -> {} (empty path)", self.start_function, self.end_function);
        }

        let functions: Vec<String> = std::iter::once(self.start_function.clone())
            .chain(self.edges.iter().map(|e| e.callee_name().to_string()))
            .collect();

        format!(
            "{} (depth: {}, confidence: {:?}-{:?})",
            functions.join(" -> "),
            self.depth,
            self.min_confidence,
            self.max_confidence
        )
    }

    /// Validates the path consistency.
    pub fn validate(&self) -> Result<(), String> {
        if self.start_function.is_empty() {
            return Err("start_function cannot be empty".to_string());
        }

        if self.end_function.is_empty() {
            return Err("end_function cannot be empty".to_string());
        }

        if self.edges.is_empty() {
            return Ok(());
        }

        // Check first edge starts with start_function
        if self.edges[0].caller_name() != self.start_function {
            return Err(format!(
                "First edge caller '{}' does not match start_function '{}'",
                self.edges[0].caller_name(),
                self.start_function
            ));
        }

        // Check last edge ends with end_function
        let last_edge = &self.edges[self.edges.len() - 1];
        if last_edge.callee_name() != self.end_function {
            return Err(format!(
                "Last edge callee '{}' does not match end_function '{}'",
                last_edge.callee_name(),
                self.end_function
            ));
        }

        // Check edge continuity
        for i in 0..self.edges.len() - 1 {
            let current_callee = self.edges[i].callee_name();
            let next_caller = self.edges[i + 1].caller_name();

            if current_callee != next_caller {
                return Err(format!(
                    "Edge continuity broken at position {}: '{}' -> '{}'",
                    i, current_callee, next_caller
                ));
            }
        }

        // Check depth consistency
        if self.edges.len() != self.depth {
            return Err(format!(
                "Depth mismatch: expected {}, got {}",
                self.depth,
                self.edges.len()
            ));
        }

        Ok(())
    }
}

impl fmt::Display for CallPath {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.display_name())
    }
}

/// Builder for constructing CallPath instances.
///
/// Provides a fluent interface for creating CallPath instances with
/// optional parameters and validation.
#[derive(Debug, Default)]
pub struct CallPathBuilder {
    start_function: Option<String>,
    end_function: Option<String>,
    edges: Vec<CallEdge>,
    execution_cost: Option<f64>,
    metadata: std::collections::HashMap<String, String>,
}

impl CallPathBuilder {
    /// Sets the start function name.
    pub fn start_function<S: Into<String>>(mut self, name: S) -> Self {
        self.start_function = Some(name.into());
        self
    }

    /// Sets the end function name.
    pub fn end_function<S: Into<String>>(mut self, name: S) -> Self {
        self.end_function = Some(name.into());
        self
    }

    /// Adds a call edge to the path.
    pub fn add_edge(mut self, edge: CallEdge) -> Self {
        self.edges.push(edge);
        self
    }

    /// Sets the execution cost estimate.
    pub fn execution_cost(mut self, cost: f64) -> Self {
        self.execution_cost = Some(cost);
        self
    }

    /// Adds metadata.
    pub fn metadata<K: Into<String>, V: Into<String>>(mut self, key: K, value: V) -> Self {
        self.metadata.insert(key.into(), value.into());
        self
    }

    /// Builds the CallPath instance.
    ///
    /// # Panics
    ///
    /// Panics if any required field is missing.
    pub fn build(self) -> CallPath {
        let start_function = self.start_function.expect("start_function is required");
        let end_function = self.end_function.expect("end_function is required");

        let mut path = CallPath::new(start_function, end_function, self.edges);

        if let Some(cost) = self.execution_cost {
            path.set_execution_cost(Some(cost));
        }

        for (key, value) in self.metadata {
            path.add_metadata(key, value);
        }

        path
    }

    /// Attempts to build the CallPath instance, returning an error if validation fails.
    pub fn try_build(self) -> Result<CallPath, String> {
        let start_function = self.start_function.ok_or("start_function is required")?;
        let end_function = self.end_function.ok_or("end_function is required")?;

        if start_function.is_empty() {
            return Err("start_function cannot be empty".to_string());
        }

        if end_function.is_empty() {
            return Err("end_function cannot be empty".to_string());
        }

        let mut path = CallPath::new(start_function, end_function, self.edges);

        // Validate the constructed path
        path.validate()?;

        if let Some(cost) = self.execution_cost {
            if cost < 0.0 {
                return Err("execution_cost cannot be negative".to_string());
            }
            path.set_execution_cost(Some(cost));
        }

        for (key, value) in self.metadata {
            path.add_metadata(key, value);
        }

        Ok(path)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{CallType, ConfidenceLevel};

    fn create_test_edge(caller: &str, callee: &str, line: u32) -> CallEdge {
        CallEdge::new(
            caller.to_string(),
            callee.to_string(),
            "test.c".to_string(),
            line,
            CallType::Direct,
            ConfidenceLevel::High,
            false,
        )
    }

    #[test]
    fn test_call_path_new() {
        let edge = create_test_edge("main", "helper", 42);
        let path = CallPath::new("main".to_string(), "helper".to_string(), vec![edge]);

        assert_eq!(path.start_function(), "main");
        assert_eq!(path.end_function(), "helper");
        assert_eq!(path.depth(), 1);
        assert!(path.is_simple_path());
        assert!(path.is_direct_path());
        assert!(path.is_high_confidence_path());
    }

    #[test]
    fn test_call_path_empty() {
        let path = CallPath::new("start".to_string(), "end".to_string(), vec![]);

        assert_eq!(path.depth(), 0);
        assert!(!path.is_simple_path());
        assert!(!path.is_complex_path());
        assert!(path.intermediate_functions().is_empty());
    }

    #[test]
    fn test_call_path_complex() {
        let edge1 = create_test_edge("main", "init", 10);
        let edge2 = create_test_edge("init", "setup", 20);
        let edge3 = create_test_edge("setup", "configure", 30);

        let path =
            CallPath::new("main".to_string(), "configure".to_string(), vec![edge1, edge2, edge3]);

        assert_eq!(path.depth(), 3);
        assert!(path.is_complex_path());
        assert_eq!(path.intermediate_functions(), vec!["init", "setup"]);

        let involved_files = path.involved_files();
        assert_eq!(involved_files, vec!["test.c"]);
    }

    #[test]
    fn test_call_path_builder() {
        let edge = create_test_edge("caller", "callee", 42);

        let path = CallPath::builder()
            .start_function("caller")
            .end_function("callee")
            .add_edge(edge)
            .execution_cost(2.5)
            .metadata("context", "test")
            .build();

        assert_eq!(path.start_function(), "caller");
        assert_eq!(path.end_function(), "callee");
        assert_eq!(path.execution_cost(), Some(2.5));
        assert_eq!(path.metadata().get("context"), Some(&"test".to_string()));
    }

    #[test]
    fn test_call_path_validation() {
        let edge1 = create_test_edge("main", "helper", 10);
        let edge2 = create_test_edge("other", "final", 20); // Broken continuity

        let path = CallPath::new("main".to_string(), "final".to_string(), vec![edge1, edge2]);

        let result = path.validate();
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("Edge continuity broken"));
    }

    #[test]
    fn test_call_path_utility_methods() {
        let mut edge = create_test_edge("caller", "callee", 42);
        edge.set_config_guard(Some("CONFIG_DEBUG".to_string()));

        let path = CallPath::new("caller".to_string(), "callee".to_string(), vec![edge]);

        assert!(path.is_direct_path());
        assert!(path.is_high_confidence_path());
        assert!(!path.has_conditional_calls());
        assert!(!path.has_indirect_calls());

        let path_id = path.path_id();
        assert!(path_id.contains("caller->callee:1:"));
    }

    #[test]
    fn test_call_path_confidence_analysis() {
        let edge1 = create_test_edge("main", "mid", 10);
        let edge2 = CallEdge::new(
            "mid".to_string(),
            "end".to_string(),
            "test.c".to_string(),
            20,
            CallType::Indirect,
            ConfidenceLevel::Medium,
            true,
        );

        let path = CallPath::new("main".to_string(), "end".to_string(), vec![edge1, edge2]);

        assert_eq!(path.min_confidence(), ConfidenceLevel::Medium);
        assert_eq!(path.max_confidence(), ConfidenceLevel::High);
        assert!(!path.is_high_confidence_path());
        assert!(path.has_conditional_calls());
        assert!(path.has_indirect_calls());
        assert!(!path.is_direct_path());
    }

    #[test]
    fn test_builder_validation() {
        let result = CallPath::builder().start_function("").end_function("end").try_build();

        assert!(result.is_err());
        assert!(result.unwrap_err().contains("start_function cannot be empty"));
    }

    #[test]
    fn test_call_path_metadata() {
        let mut path = CallPath::new("start".to_string(), "end".to_string(), vec![]);

        path.add_metadata("key1".to_string(), "value1".to_string());
        path.add_metadata("key2".to_string(), "value2".to_string());

        assert_eq!(path.metadata().get("key1"), Some(&"value1".to_string()));
        assert_eq!(path.metadata().get("key2"), Some(&"value2".to_string()));

        let removed = path.remove_metadata("key1");
        assert_eq!(removed, Some("value1".to_string()));
        assert!(path.metadata().get("key1").is_none());

        path.clear_metadata();
        assert!(path.metadata().is_empty());
    }

    #[test]
    fn test_call_path_display() {
        let edge = create_test_edge("main", "helper", 42);
        let path = CallPath::new("main".to_string(), "helper".to_string(), vec![edge]);

        let display = format!("{}", path);
        assert!(display.contains("main -> helper"));
        assert!(display.contains("depth: 1"));
        assert!(display.contains("High"));
    }
}
