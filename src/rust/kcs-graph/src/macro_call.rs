//! MacroCall data model for representing function calls through macro expansion.
//!
//! This module defines the MacroCall struct that represents function calls that
//! occur through macro expansion. MacroCall captures the macro definition,
//! expansion site, resulting function calls, and preprocessor context.

use serde::{Deserialize, Serialize};
use std::fmt;

use crate::call_edge::CallEdge;
use crate::call_site::CallSite;

/// Represents function calls that occur through macro expansion.
///
/// A MacroCall captures information about macros that expand to function calls,
/// including the macro definition, where it's expanded, what function calls
/// result from the expansion, and the surrounding preprocessor context.
///
/// # Examples
///
/// ```rust
/// use kcs_graph::{MacroCall, CallSite};
///
/// let expansion_site = CallSite::new(
///     "/path/to/file.c".to_string(),
///     42,
///     10,
///     "main_function".to_string(),
/// );
///
/// let macro_call = MacroCall::new(
///     "DEBUG_PRINT".to_string(),
///     expansion_site,
/// );
///
/// assert_eq!(macro_call.macro_name(), "DEBUG_PRINT");
/// assert_eq!(macro_call.expansion_site().line_number(), 42);
/// ```
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct MacroCall {
    /// The name of the macro being expanded
    macro_name: String,

    /// Optional macro definition if available
    macro_definition: Option<String>,

    /// Location where the macro is used/expanded
    expansion_site: CallSite,

    /// Function calls that result from macro expansion
    expanded_calls: Vec<CallEdge>,

    /// Surrounding preprocessor directives and context
    preprocessor_context: Option<String>,

    /// Additional metadata as key-value pairs
    metadata: std::collections::HashMap<String, String>,
}

impl MacroCall {
    /// Creates a new MacroCall with the specified parameters.
    ///
    /// # Arguments
    ///
    /// * `macro_name` - Name of the macro being expanded
    /// * `expansion_site` - Location where the macro is used
    ///
    /// # Examples
    ///
    /// ```rust
    /// use kcs_graph::{MacroCall, CallSite};
    ///
    /// let expansion_site = CallSite::new(
    ///     "kernel/debug.c".to_string(),
    ///     123,
    ///     15,
    ///     "debug_function".to_string(),
    /// );
    ///
    /// let macro_call = MacroCall::new(
    ///     "WARN_ON".to_string(),
    ///     expansion_site,
    /// );
    /// ```
    pub fn new(macro_name: String, expansion_site: CallSite) -> Self {
        Self {
            macro_name,
            macro_definition: None,
            expansion_site,
            expanded_calls: Vec::new(),
            preprocessor_context: None,
            metadata: std::collections::HashMap::new(),
        }
    }

    /// Creates a new MacroCall builder for fluent construction.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use kcs_graph::{MacroCall, CallSite};
    ///
    /// let expansion_site = CallSite::new(
    ///     "main.c".to_string(),
    ///     42,
    ///     10,
    ///     "main".to_string(),
    /// );
    ///
    /// let macro_call = MacroCall::builder()
    ///     .macro_name("DEBUG_PRINT")
    ///     .expansion_site(expansion_site)
    ///     .macro_definition("#define DEBUG_PRINT(x) printf(x)")
    ///     .preprocessor_context("#ifdef DEBUG")
    ///     .build();
    /// ```
    pub fn builder() -> MacroCallBuilder {
        MacroCallBuilder::default()
    }

    // Accessor methods

    /// Returns the name of the macro being expanded.
    pub fn macro_name(&self) -> &str {
        &self.macro_name
    }

    /// Returns the macro definition, if available.
    pub fn macro_definition(&self) -> Option<&str> {
        self.macro_definition.as_deref()
    }

    /// Returns the location where the macro is expanded.
    pub fn expansion_site(&self) -> &CallSite {
        &self.expansion_site
    }

    /// Returns the function calls that result from macro expansion.
    pub fn expanded_calls(&self) -> &[CallEdge] {
        &self.expanded_calls
    }

    /// Returns the preprocessor context, if available.
    pub fn preprocessor_context(&self) -> Option<&str> {
        self.preprocessor_context.as_deref()
    }

    /// Returns a reference to the metadata map.
    pub fn metadata(&self) -> &std::collections::HashMap<String, String> {
        &self.metadata
    }

    // Mutator methods

    /// Sets the macro definition.
    pub fn set_macro_definition(&mut self, definition: Option<String>) {
        self.macro_definition = definition;
    }

    /// Sets the preprocessor context.
    pub fn set_preprocessor_context(&mut self, context: Option<String>) {
        self.preprocessor_context = context;
    }

    /// Adds a function call that results from macro expansion.
    pub fn add_expanded_call(&mut self, call_edge: CallEdge) {
        self.expanded_calls.push(call_edge);
    }

    /// Removes all expanded calls.
    pub fn clear_expanded_calls(&mut self) {
        self.expanded_calls.clear();
    }

    /// Sets all expanded calls at once.
    pub fn set_expanded_calls(&mut self, calls: Vec<CallEdge>) {
        self.expanded_calls = calls;
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

    /// Returns the number of function calls generated by this macro expansion.
    pub fn call_count(&self) -> usize {
        self.expanded_calls.len()
    }

    /// Returns true if this macro expansion generates any function calls.
    pub fn has_calls(&self) -> bool {
        !self.expanded_calls.is_empty()
    }

    /// Returns true if this macro has a definition available.
    pub fn has_definition(&self) -> bool {
        self.macro_definition.is_some()
    }

    /// Returns true if this macro has preprocessor context.
    pub fn has_preprocessor_context(&self) -> bool {
        self.preprocessor_context.is_some()
    }

    /// Returns all unique caller function names from expanded calls.
    pub fn caller_functions(&self) -> std::collections::HashSet<&str> {
        self.expanded_calls.iter().map(|call| call.caller_name()).collect()
    }

    /// Returns all unique callee function names from expanded calls.
    pub fn callee_functions(&self) -> std::collections::HashSet<&str> {
        self.expanded_calls.iter().map(|call| call.callee_name()).collect()
    }

    /// Returns expanded calls that match a specific call type.
    pub fn calls_by_type(&self, call_type: crate::types::CallType) -> Vec<&CallEdge> {
        self.expanded_calls
            .iter()
            .filter(|call| call.call_type() == call_type)
            .collect()
    }

    /// Returns expanded calls with a specific confidence level.
    pub fn calls_by_confidence(&self, confidence: crate::types::ConfidenceLevel) -> Vec<&CallEdge> {
        self.expanded_calls
            .iter()
            .filter(|call| call.confidence() == confidence)
            .collect()
    }

    /// Creates a unique identifier for this macro call.
    ///
    /// The identifier is based on the macro name and expansion site,
    /// making it suitable for deduplication and indexing.
    pub fn macro_id(&self) -> String {
        format!("{}:{}", self.macro_name, self.expansion_site.site_id())
    }

    /// Returns a display-friendly representation of the macro call.
    pub fn display_name(&self) -> String {
        format!(
            "{} -> {} calls at {}",
            self.macro_name,
            self.expanded_calls.len(),
            self.expansion_site.position_string()
        )
    }
}

impl fmt::Display for MacroCall {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "Macro {} expands to {} calls at {}",
            self.macro_name,
            self.expanded_calls.len(),
            self.expansion_site
        )
    }
}

/// Builder for constructing MacroCall instances.
///
/// Provides a fluent interface for creating MacroCall instances with
/// optional parameters and validation.
#[derive(Debug, Default)]
pub struct MacroCallBuilder {
    macro_name: Option<String>,
    macro_definition: Option<String>,
    expansion_site: Option<CallSite>,
    expanded_calls: Vec<CallEdge>,
    preprocessor_context: Option<String>,
    metadata: std::collections::HashMap<String, String>,
}

impl MacroCallBuilder {
    /// Sets the macro name.
    pub fn macro_name<S: Into<String>>(mut self, name: S) -> Self {
        self.macro_name = Some(name.into());
        self
    }

    /// Sets the macro definition.
    pub fn macro_definition<S: Into<String>>(mut self, definition: S) -> Self {
        self.macro_definition = Some(definition.into());
        self
    }

    /// Sets the expansion site.
    pub fn expansion_site(mut self, site: CallSite) -> Self {
        self.expansion_site = Some(site);
        self
    }

    /// Adds an expanded call.
    pub fn expanded_call(mut self, call: CallEdge) -> Self {
        self.expanded_calls.push(call);
        self
    }

    /// Sets all expanded calls at once.
    pub fn expanded_calls(mut self, calls: Vec<CallEdge>) -> Self {
        self.expanded_calls = calls;
        self
    }

    /// Sets the preprocessor context.
    pub fn preprocessor_context<S: Into<String>>(mut self, context: S) -> Self {
        self.preprocessor_context = Some(context.into());
        self
    }

    /// Adds metadata.
    pub fn metadata<K: Into<String>, V: Into<String>>(mut self, key: K, value: V) -> Self {
        self.metadata.insert(key.into(), value.into());
        self
    }

    /// Builds the MacroCall instance.
    ///
    /// # Panics
    ///
    /// Panics if any required field is missing.
    pub fn build(self) -> MacroCall {
        MacroCall {
            macro_name: self.macro_name.expect("macro_name is required"),
            macro_definition: self.macro_definition,
            expansion_site: self.expansion_site.expect("expansion_site is required"),
            expanded_calls: self.expanded_calls,
            preprocessor_context: self.preprocessor_context,
            metadata: self.metadata,
        }
    }

    /// Attempts to build the MacroCall instance, returning an error if validation fails.
    pub fn try_build(self) -> Result<MacroCall, String> {
        let macro_name = self.macro_name.ok_or("macro_name is required")?;
        let expansion_site = self.expansion_site.ok_or("expansion_site is required")?;

        // Validation
        if macro_name.is_empty() {
            return Err("macro_name cannot be empty".to_string());
        }

        // Validate macro name is a valid C identifier
        if !is_valid_c_identifier(&macro_name) {
            return Err("macro_name must be a valid C identifier".to_string());
        }

        Ok(MacroCall {
            macro_name,
            macro_definition: self.macro_definition,
            expansion_site,
            expanded_calls: self.expanded_calls,
            preprocessor_context: self.preprocessor_context,
            metadata: self.metadata,
        })
    }
}

/// Validates if a string is a valid C identifier.
fn is_valid_c_identifier(name: &str) -> bool {
    if name.is_empty() {
        return false;
    }

    // First character must be letter or underscore
    let mut chars = name.chars();
    let first = chars.next().unwrap();
    if !first.is_ascii_alphabetic() && first != '_' {
        return false;
    }

    // Remaining characters must be letters, digits, or underscores
    chars.all(|c| c.is_ascii_alphanumeric() || c == '_')
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{CallType, ConfidenceLevel};

    fn create_test_call_site() -> CallSite {
        CallSite::new("test.c".to_string(), 42, 10, "test_function".to_string())
    }

    fn create_test_call_edge() -> CallEdge {
        CallEdge::new(
            "caller".to_string(),
            "callee".to_string(),
            "test.c".to_string(),
            43,
            CallType::Direct,
            ConfidenceLevel::High,
            false,
        )
    }

    #[test]
    fn test_macro_call_new() {
        let expansion_site = create_test_call_site();
        let macro_call = MacroCall::new("DEBUG_PRINT".to_string(), expansion_site);

        assert_eq!(macro_call.macro_name(), "DEBUG_PRINT");
        assert_eq!(macro_call.expansion_site().line_number(), 42);
        assert!(macro_call.macro_definition().is_none());
        assert!(macro_call.expanded_calls().is_empty());
        assert!(macro_call.preprocessor_context().is_none());
    }

    #[test]
    fn test_macro_call_builder() {
        let expansion_site = create_test_call_site();
        let call_edge = create_test_call_edge();

        let macro_call = MacroCall::builder()
            .macro_name("WARN_ON")
            .expansion_site(expansion_site)
            .macro_definition("#define WARN_ON(x) if (x) warn()")
            .expanded_call(call_edge)
            .preprocessor_context("#ifdef DEBUG")
            .metadata("context", "error_handling")
            .build();

        assert_eq!(macro_call.macro_name(), "WARN_ON");
        assert_eq!(macro_call.macro_definition(), Some("#define WARN_ON(x) if (x) warn()"));
        assert_eq!(macro_call.expanded_calls().len(), 1);
        assert_eq!(macro_call.preprocessor_context(), Some("#ifdef DEBUG"));
        assert_eq!(macro_call.metadata().get("context"), Some(&"error_handling".to_string()));
    }

    #[test]
    fn test_macro_call_utility_methods() {
        let expansion_site = create_test_call_site();
        let mut macro_call = MacroCall::new("TEST_MACRO".to_string(), expansion_site);

        assert_eq!(macro_call.call_count(), 0);
        assert!(!macro_call.has_calls());
        assert!(!macro_call.has_definition());
        assert!(!macro_call.has_preprocessor_context());

        let call_edge = create_test_call_edge();
        macro_call.add_expanded_call(call_edge);

        assert_eq!(macro_call.call_count(), 1);
        assert!(macro_call.has_calls());

        macro_call.set_macro_definition(Some("test definition".to_string()));
        assert!(macro_call.has_definition());

        macro_call.set_preprocessor_context(Some("#ifdef TEST".to_string()));
        assert!(macro_call.has_preprocessor_context());
    }

    #[test]
    fn test_macro_call_functions() {
        let expansion_site = create_test_call_site();
        let mut macro_call = MacroCall::new("TEST_MACRO".to_string(), expansion_site);

        let call1 = CallEdge::new(
            "func1".to_string(),
            "target1".to_string(),
            "test.c".to_string(),
            1,
            CallType::Direct,
            ConfidenceLevel::High,
            false,
        );

        let call2 = CallEdge::new(
            "func2".to_string(),
            "target2".to_string(),
            "test.c".to_string(),
            2,
            CallType::Indirect,
            ConfidenceLevel::Medium,
            false,
        );

        macro_call.add_expanded_call(call1);
        macro_call.add_expanded_call(call2);

        let callers = macro_call.caller_functions();
        assert!(callers.contains("func1"));
        assert!(callers.contains("func2"));

        let callees = macro_call.callee_functions();
        assert!(callees.contains("target1"));
        assert!(callees.contains("target2"));

        let direct_calls = macro_call.calls_by_type(CallType::Direct);
        assert_eq!(direct_calls.len(), 1);

        let high_confidence_calls = macro_call.calls_by_confidence(ConfidenceLevel::High);
        assert_eq!(high_confidence_calls.len(), 1);
    }

    #[test]
    fn test_macro_call_id() {
        let expansion_site = create_test_call_site();
        let macro_call = MacroCall::new("TEST_MACRO".to_string(), expansion_site);

        let expected_id = format!("TEST_MACRO:{}", macro_call.expansion_site().site_id());
        assert_eq!(macro_call.macro_id(), expected_id);
    }

    #[test]
    fn test_macro_call_metadata() {
        let expansion_site = create_test_call_site();
        let mut macro_call = MacroCall::new("TEST_MACRO".to_string(), expansion_site);

        macro_call.add_metadata("key1".to_string(), "value1".to_string());
        macro_call.add_metadata("key2".to_string(), "value2".to_string());

        assert_eq!(macro_call.metadata().get("key1"), Some(&"value1".to_string()));
        assert_eq!(macro_call.metadata().get("key2"), Some(&"value2".to_string()));

        let removed = macro_call.remove_metadata("key1");
        assert_eq!(removed, Some("value1".to_string()));
        assert!(macro_call.metadata().get("key1").is_none());

        macro_call.clear_metadata();
        assert!(macro_call.metadata().is_empty());
    }

    #[test]
    fn test_builder_validation() {
        let expansion_site = create_test_call_site();

        let result = MacroCall::builder().macro_name("").expansion_site(expansion_site).try_build();

        assert!(result.is_err());
        assert!(result.unwrap_err().contains("macro_name cannot be empty"));

        let result = MacroCall::builder()
            .macro_name("123invalid")
            .expansion_site(create_test_call_site())
            .try_build();

        assert!(result.is_err());
        assert!(result.unwrap_err().contains("must be a valid C identifier"));
    }

    #[test]
    fn test_valid_c_identifier() {
        assert!(is_valid_c_identifier("valid_name"));
        assert!(is_valid_c_identifier("_private"));
        assert!(is_valid_c_identifier("NAME123"));
        assert!(is_valid_c_identifier("__builtin"));

        assert!(!is_valid_c_identifier(""));
        assert!(!is_valid_c_identifier("123invalid"));
        assert!(!is_valid_c_identifier("invalid-name"));
        assert!(!is_valid_c_identifier("invalid.name"));
        assert!(!is_valid_c_identifier("invalid name"));
    }

    #[test]
    fn test_display() {
        let expansion_site = create_test_call_site();
        let macro_call = MacroCall::new("DEBUG_PRINT".to_string(), expansion_site);

        let display = format!("{}", macro_call);
        assert!(display.contains("DEBUG_PRINT"));
        assert!(display.contains("0 calls"));
        assert!(display.contains("test_function"));
    }

    #[test]
    fn test_clear_and_set_calls() {
        let expansion_site = create_test_call_site();
        let mut macro_call = MacroCall::new("TEST_MACRO".to_string(), expansion_site);

        let call1 = create_test_call_edge();
        let call2 = create_test_call_edge();

        macro_call.add_expanded_call(call1.clone());
        macro_call.add_expanded_call(call2.clone());
        assert_eq!(macro_call.call_count(), 2);

        macro_call.clear_expanded_calls();
        assert_eq!(macro_call.call_count(), 0);

        macro_call.set_expanded_calls(vec![call1, call2]);
        assert_eq!(macro_call.call_count(), 2);
    }
}
