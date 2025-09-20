//! FunctionPointer data model for representing function pointer assignments and usage.
//!
//! This module defines the FunctionPointer struct that represents function pointer
//! assignments and their usage patterns in C code. It tracks where function pointers
//! are assigned, what functions they point to, and where they are used.

use serde::{Deserialize, Serialize};
use std::fmt;

use crate::call_site::CallSite;
use crate::types::PointerType;

/// Represents function pointer assignments and their usage patterns.
///
/// A FunctionPointer captures information about function pointer variables,
/// including where they are assigned, what function they point to, where
/// they are used, and their context (e.g., struct member, local variable).
///
/// # Examples
///
/// ```rust
/// use kcs_graph::{FunctionPointer, CallSite};
///
/// let assignment_site = CallSite::new(
///     "/path/to/file.c".to_string(),
///     42,
///     10,
///     "setup_function".to_string(),
/// );
///
/// let function_pointer = FunctionPointer::new(
///     "callback_handler".to_string(),
///     assignment_site,
///     "actual_handler_function".to_string(),
/// );
///
/// assert_eq!(function_pointer.pointer_name(), "callback_handler");
/// assert_eq!(function_pointer.assigned_function(), "actual_handler_function");
/// ```
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct FunctionPointer {
    /// Name of the function pointer variable
    pointer_name: String,

    /// Location where the function pointer is assigned
    assignment_site: CallSite,

    /// Name of the function being assigned to the pointer
    assigned_function: String,

    /// Locations where the function pointer is called/used
    usage_sites: Vec<CallSite>,

    /// Optional struct name if the pointer is a struct member
    struct_context: Option<String>,

    /// Type of function pointer (struct field, variable, parameter, etc.)
    pointer_type: PointerType,

    /// Additional metadata as key-value pairs
    metadata: std::collections::HashMap<String, String>,
}

impl FunctionPointer {
    /// Creates a new FunctionPointer with the specified parameters.
    ///
    /// # Arguments
    ///
    /// * `pointer_name` - Name of the function pointer variable
    /// * `assignment_site` - Location where the pointer is assigned
    /// * `assigned_function` - Name of the function being assigned
    ///
    /// # Examples
    ///
    /// ```rust
    /// use kcs_graph::{FunctionPointer, CallSite};
    ///
    /// let assignment_site = CallSite::new(
    ///     "driver/init.c".to_string(),
    ///     123,
    ///     15,
    ///     "driver_init".to_string(),
    /// );
    ///
    /// let function_pointer = FunctionPointer::new(
    ///     "ops->open".to_string(),
    ///     assignment_site,
    ///     "device_open".to_string(),
    /// );
    /// ```
    pub fn new(pointer_name: String, assignment_site: CallSite, assigned_function: String) -> Self {
        Self {
            pointer_name,
            assignment_site,
            assigned_function,
            usage_sites: Vec::new(),
            struct_context: None,
            pointer_type: PointerType::Variable,
            metadata: std::collections::HashMap::new(),
        }
    }

    /// Creates a new FunctionPointer builder for fluent construction.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use kcs_graph::{FunctionPointer, CallSite, PointerType};
    ///
    /// let assignment_site = CallSite::new(
    ///     "main.c".to_string(),
    ///     42,
    ///     10,
    ///     "main".to_string(),
    /// );
    ///
    /// let function_pointer = FunctionPointer::builder()
    ///     .pointer_name("handler")
    ///     .assignment_site(assignment_site)
    ///     .assigned_function("signal_handler")
    ///     .pointer_type(PointerType::Callback)
    ///     .struct_context("device_ops")
    ///     .build();
    /// ```
    pub fn builder() -> FunctionPointerBuilder {
        FunctionPointerBuilder::default()
    }

    // Accessor methods

    /// Returns the name of the function pointer variable.
    pub fn pointer_name(&self) -> &str {
        &self.pointer_name
    }

    /// Returns the location where the function pointer is assigned.
    pub fn assignment_site(&self) -> &CallSite {
        &self.assignment_site
    }

    /// Returns the name of the function being assigned to the pointer.
    pub fn assigned_function(&self) -> &str {
        &self.assigned_function
    }

    /// Returns the locations where the function pointer is used.
    pub fn usage_sites(&self) -> &[CallSite] {
        &self.usage_sites
    }

    /// Returns the struct context, if any.
    pub fn struct_context(&self) -> Option<&str> {
        self.struct_context.as_deref()
    }

    /// Returns the type of function pointer.
    pub fn pointer_type(&self) -> PointerType {
        self.pointer_type
    }

    /// Returns a reference to the metadata map.
    pub fn metadata(&self) -> &std::collections::HashMap<String, String> {
        &self.metadata
    }

    // Mutator methods

    /// Sets the struct context.
    pub fn set_struct_context(&mut self, context: Option<String>) {
        self.struct_context = context;
    }

    /// Sets the pointer type.
    pub fn set_pointer_type(&mut self, pointer_type: PointerType) {
        self.pointer_type = pointer_type;
    }

    /// Adds a usage site where the function pointer is called.
    pub fn add_usage_site(&mut self, usage_site: CallSite) {
        self.usage_sites.push(usage_site);
    }

    /// Removes all usage sites.
    pub fn clear_usage_sites(&mut self) {
        self.usage_sites.clear();
    }

    /// Sets all usage sites at once.
    pub fn set_usage_sites(&mut self, sites: Vec<CallSite>) {
        self.usage_sites = sites;
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

    /// Returns the number of usage sites for this function pointer.
    pub fn usage_count(&self) -> usize {
        self.usage_sites.len()
    }

    /// Returns true if this function pointer has any usage sites.
    pub fn has_usage(&self) -> bool {
        !self.usage_sites.is_empty()
    }

    /// Returns true if this function pointer is part of a struct.
    pub fn is_struct_member(&self) -> bool {
        self.struct_context.is_some()
    }

    /// Returns true if this function pointer is a callback.
    pub fn is_callback(&self) -> bool {
        matches!(self.pointer_type, PointerType::Callback)
    }

    /// Returns true if this function pointer is a struct field.
    pub fn is_struct_field(&self) -> bool {
        matches!(self.pointer_type, PointerType::StructureField)
    }

    /// Returns true if this function pointer is a variable.
    pub fn is_variable(&self) -> bool {
        matches!(self.pointer_type, PointerType::Variable)
    }

    /// Returns true if this function pointer is a parameter.
    pub fn is_parameter(&self) -> bool {
        matches!(self.pointer_type, PointerType::Parameter)
    }

    /// Returns true if this function pointer is a return value.
    pub fn is_return_value(&self) -> bool {
        matches!(self.pointer_type, PointerType::ReturnValue)
    }

    /// Returns true if this function pointer is in an array.
    pub fn is_array_element(&self) -> bool {
        matches!(self.pointer_type, PointerType::Array)
    }

    /// Returns all unique files where this function pointer is used.
    pub fn usage_files(&self) -> std::collections::HashSet<&str> {
        self.usage_sites.iter().map(|site| site.file_path()).collect()
    }

    /// Returns all unique functions where this function pointer is used.
    pub fn usage_functions(&self) -> std::collections::HashSet<&str> {
        self.usage_sites.iter().map(|site| site.function_context()).collect()
    }

    /// Returns usage sites in a specific file.
    pub fn usage_sites_in_file(&self, file_path: &str) -> Vec<&CallSite> {
        self.usage_sites.iter().filter(|site| site.file_path() == file_path).collect()
    }

    /// Returns usage sites in a specific function.
    pub fn usage_sites_in_function(&self, function_name: &str) -> Vec<&CallSite> {
        self.usage_sites
            .iter()
            .filter(|site| site.function_context() == function_name)
            .collect()
    }

    /// Creates a unique identifier for this function pointer.
    ///
    /// The identifier is based on the pointer name and assignment site,
    /// making it suitable for deduplication and indexing.
    pub fn pointer_id(&self) -> String {
        format!("{}:{}", self.pointer_name, self.assignment_site.site_id())
    }

    /// Returns a display-friendly representation of the function pointer.
    pub fn display_name(&self) -> String {
        match &self.struct_context {
            Some(struct_name) => format!(
                "{}.{} -> {} ({} usages)",
                struct_name,
                self.pointer_name,
                self.assigned_function,
                self.usage_sites.len()
            ),
            None => format!(
                "{} -> {} ({} usages)",
                self.pointer_name,
                self.assigned_function,
                self.usage_sites.len()
            ),
        }
    }

    /// Checks if the pointer name is a valid C identifier.
    pub fn is_valid_pointer_name(&self) -> bool {
        is_valid_c_identifier(&self.pointer_name)
    }

    /// Checks if the assigned function name is a valid C identifier.
    pub fn is_valid_function_name(&self) -> bool {
        is_valid_c_identifier(&self.assigned_function)
    }
}

impl fmt::Display for FunctionPointer {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "Function pointer {} assigned to {} at {} ({} usages)",
            self.pointer_name,
            self.assigned_function,
            self.assignment_site,
            self.usage_sites.len()
        )
    }
}

/// Builder for constructing FunctionPointer instances.
///
/// Provides a fluent interface for creating FunctionPointer instances with
/// optional parameters and validation.
#[derive(Debug, Default)]
pub struct FunctionPointerBuilder {
    pointer_name: Option<String>,
    assignment_site: Option<CallSite>,
    assigned_function: Option<String>,
    usage_sites: Vec<CallSite>,
    struct_context: Option<String>,
    pointer_type: Option<PointerType>,
    metadata: std::collections::HashMap<String, String>,
}

impl FunctionPointerBuilder {
    /// Sets the pointer name.
    pub fn pointer_name<S: Into<String>>(mut self, name: S) -> Self {
        self.pointer_name = Some(name.into());
        self
    }

    /// Sets the assignment site.
    pub fn assignment_site(mut self, site: CallSite) -> Self {
        self.assignment_site = Some(site);
        self
    }

    /// Sets the assigned function name.
    pub fn assigned_function<S: Into<String>>(mut self, function: S) -> Self {
        self.assigned_function = Some(function.into());
        self
    }

    /// Adds a usage site.
    pub fn usage_site(mut self, site: CallSite) -> Self {
        self.usage_sites.push(site);
        self
    }

    /// Sets all usage sites at once.
    pub fn usage_sites(mut self, sites: Vec<CallSite>) -> Self {
        self.usage_sites = sites;
        self
    }

    /// Sets the struct context.
    pub fn struct_context<S: Into<String>>(mut self, context: S) -> Self {
        self.struct_context = Some(context.into());
        self
    }

    /// Sets the pointer type.
    pub fn pointer_type(mut self, pointer_type: PointerType) -> Self {
        self.pointer_type = Some(pointer_type);
        self
    }

    /// Adds metadata.
    pub fn metadata<K: Into<String>, V: Into<String>>(mut self, key: K, value: V) -> Self {
        self.metadata.insert(key.into(), value.into());
        self
    }

    /// Builds the FunctionPointer instance.
    ///
    /// # Panics
    ///
    /// Panics if any required field is missing.
    pub fn build(self) -> FunctionPointer {
        FunctionPointer {
            pointer_name: self.pointer_name.expect("pointer_name is required"),
            assignment_site: self.assignment_site.expect("assignment_site is required"),
            assigned_function: self.assigned_function.expect("assigned_function is required"),
            usage_sites: self.usage_sites,
            struct_context: self.struct_context,
            pointer_type: self.pointer_type.unwrap_or(PointerType::Variable),
            metadata: self.metadata,
        }
    }

    /// Attempts to build the FunctionPointer instance, returning an error if validation fails.
    pub fn try_build(self) -> Result<FunctionPointer, String> {
        let pointer_name = self.pointer_name.ok_or("pointer_name is required")?;
        let assignment_site = self.assignment_site.ok_or("assignment_site is required")?;
        let assigned_function = self.assigned_function.ok_or("assigned_function is required")?;

        // Validation
        if pointer_name.is_empty() {
            return Err("pointer_name cannot be empty".to_string());
        }
        if assigned_function.is_empty() {
            return Err("assigned_function cannot be empty".to_string());
        }

        // Validate names are valid C identifiers
        if !is_valid_c_identifier(&pointer_name) {
            return Err("pointer_name must be a valid C identifier".to_string());
        }
        if !is_valid_c_identifier(&assigned_function) {
            return Err("assigned_function must be a valid C identifier".to_string());
        }

        Ok(FunctionPointer {
            pointer_name,
            assignment_site,
            assigned_function,
            usage_sites: self.usage_sites,
            struct_context: self.struct_context,
            pointer_type: self.pointer_type.unwrap_or(PointerType::Variable),
            metadata: self.metadata,
        })
    }
}

/// Validates if a string is a valid C identifier.
///
/// A valid C identifier:
/// - Starts with a letter or underscore
/// - Contains only letters, digits, and underscores
/// - Is not empty
/// - May contain dots for struct member access (e.g., "ops.open")
fn is_valid_c_identifier(name: &str) -> bool {
    if name.is_empty() {
        return false;
    }

    // Handle struct member access notation (e.g., "ops->open", "ops.open")
    if name.contains("->") || name.contains('.') {
        let parts: Vec<&str> = if name.contains("->") {
            name.split("->").collect()
        } else {
            name.split('.').collect()
        };

        // All parts must be valid identifiers
        return parts.iter().all(|part| is_simple_c_identifier(part));
    }

    is_simple_c_identifier(name)
}

/// Validates if a string is a simple C identifier (no dots or arrows).
fn is_simple_c_identifier(name: &str) -> bool {
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

    fn create_test_call_site() -> CallSite {
        CallSite::new("test.c".to_string(), 42, 10, "test_function".to_string())
    }

    fn create_usage_site() -> CallSite {
        CallSite::new("test.c".to_string(), 50, 15, "caller_function".to_string())
    }

    #[test]
    fn test_function_pointer_new() {
        let assignment_site = create_test_call_site();
        let function_pointer = FunctionPointer::new(
            "callback".to_string(),
            assignment_site,
            "handler_function".to_string(),
        );

        assert_eq!(function_pointer.pointer_name(), "callback");
        assert_eq!(function_pointer.assigned_function(), "handler_function");
        assert_eq!(function_pointer.assignment_site().line_number(), 42);
        assert!(function_pointer.usage_sites().is_empty());
        assert!(function_pointer.struct_context().is_none());
        assert_eq!(function_pointer.pointer_type(), PointerType::Variable);
    }

    #[test]
    fn test_function_pointer_builder() {
        let assignment_site = create_test_call_site();
        let usage_site = create_usage_site();

        let function_pointer = FunctionPointer::builder()
            .pointer_name("ops->open")
            .assignment_site(assignment_site)
            .assigned_function("device_open")
            .usage_site(usage_site)
            .struct_context("device_operations")
            .pointer_type(PointerType::StructureField)
            .metadata("context", "device_driver")
            .build();

        assert_eq!(function_pointer.pointer_name(), "ops->open");
        assert_eq!(function_pointer.assigned_function(), "device_open");
        assert_eq!(function_pointer.usage_sites().len(), 1);
        assert_eq!(function_pointer.struct_context(), Some("device_operations"));
        assert_eq!(function_pointer.pointer_type(), PointerType::StructureField);
        assert_eq!(function_pointer.metadata().get("context"), Some(&"device_driver".to_string()));
    }

    #[test]
    fn test_function_pointer_utility_methods() {
        let assignment_site = create_test_call_site();
        let mut function_pointer =
            FunctionPointer::new("callback".to_string(), assignment_site, "handler".to_string());

        assert_eq!(function_pointer.usage_count(), 0);
        assert!(!function_pointer.has_usage());
        assert!(!function_pointer.is_struct_member());
        assert!(!function_pointer.is_callback());
        assert!(function_pointer.is_variable());

        let usage_site = create_usage_site();
        function_pointer.add_usage_site(usage_site);

        assert_eq!(function_pointer.usage_count(), 1);
        assert!(function_pointer.has_usage());

        function_pointer.set_struct_context(Some("ops_struct".to_string()));
        assert!(function_pointer.is_struct_member());

        function_pointer.set_pointer_type(PointerType::Callback);
        assert!(function_pointer.is_callback());
        assert!(!function_pointer.is_variable());
    }

    #[test]
    fn test_function_pointer_type_checks() {
        let assignment_site = create_test_call_site();
        let mut function_pointer =
            FunctionPointer::new("ptr".to_string(), assignment_site, "func".to_string());

        function_pointer.set_pointer_type(PointerType::StructureField);
        assert!(function_pointer.is_struct_field());
        assert!(!function_pointer.is_variable());

        function_pointer.set_pointer_type(PointerType::Parameter);
        assert!(function_pointer.is_parameter());

        function_pointer.set_pointer_type(PointerType::ReturnValue);
        assert!(function_pointer.is_return_value());

        function_pointer.set_pointer_type(PointerType::Array);
        assert!(function_pointer.is_array_element());
    }

    #[test]
    fn test_function_pointer_usage_analysis() {
        let assignment_site = create_test_call_site();
        let mut function_pointer =
            FunctionPointer::new("callback".to_string(), assignment_site, "handler".to_string());

        let usage1 = CallSite::new("file1.c".to_string(), 10, 5, "func1".to_string());
        let usage2 = CallSite::new("file2.c".to_string(), 20, 10, "func2".to_string());
        let usage3 = CallSite::new("file1.c".to_string(), 30, 15, "func3".to_string());

        function_pointer.add_usage_site(usage1);
        function_pointer.add_usage_site(usage2);
        function_pointer.add_usage_site(usage3);

        let files = function_pointer.usage_files();
        assert_eq!(files.len(), 2);
        assert!(files.contains("file1.c"));
        assert!(files.contains("file2.c"));

        let functions = function_pointer.usage_functions();
        assert_eq!(functions.len(), 3);
        assert!(functions.contains("func1"));
        assert!(functions.contains("func2"));
        assert!(functions.contains("func3"));

        let file1_usages = function_pointer.usage_sites_in_file("file1.c");
        assert_eq!(file1_usages.len(), 2);

        let func1_usages = function_pointer.usage_sites_in_function("func1");
        assert_eq!(func1_usages.len(), 1);
    }

    #[test]
    fn test_function_pointer_id() {
        let assignment_site = create_test_call_site();
        let function_pointer =
            FunctionPointer::new("callback".to_string(), assignment_site, "handler".to_string());

        let expected_id = format!("callback:{}", function_pointer.assignment_site().site_id());
        assert_eq!(function_pointer.pointer_id(), expected_id);
    }

    #[test]
    fn test_function_pointer_metadata() {
        let assignment_site = create_test_call_site();
        let mut function_pointer =
            FunctionPointer::new("callback".to_string(), assignment_site, "handler".to_string());

        function_pointer.add_metadata("key1".to_string(), "value1".to_string());
        function_pointer.add_metadata("key2".to_string(), "value2".to_string());

        assert_eq!(function_pointer.metadata().get("key1"), Some(&"value1".to_string()));
        assert_eq!(function_pointer.metadata().get("key2"), Some(&"value2".to_string()));

        let removed = function_pointer.remove_metadata("key1");
        assert_eq!(removed, Some("value1".to_string()));
        assert!(function_pointer.metadata().get("key1").is_none());

        function_pointer.clear_metadata();
        assert!(function_pointer.metadata().is_empty());
    }

    #[test]
    fn test_builder_validation() {
        let assignment_site = create_test_call_site();

        let result = FunctionPointer::builder()
            .pointer_name("")
            .assignment_site(assignment_site)
            .assigned_function("handler")
            .try_build();

        assert!(result.is_err());
        assert!(result.unwrap_err().contains("pointer_name cannot be empty"));

        let result = FunctionPointer::builder()
            .pointer_name("123invalid")
            .assignment_site(create_test_call_site())
            .assigned_function("handler")
            .try_build();

        assert!(result.is_err());
        assert!(result.unwrap_err().contains("pointer_name must be a valid C identifier"));

        let result = FunctionPointer::builder()
            .pointer_name("valid_ptr")
            .assignment_site(create_test_call_site())
            .assigned_function("123invalid")
            .try_build();

        assert!(result.is_err());
        assert!(result.unwrap_err().contains("assigned_function must be a valid C identifier"));
    }

    #[test]
    fn test_valid_c_identifier() {
        // Simple identifiers
        assert!(is_valid_c_identifier("valid_name"));
        assert!(is_valid_c_identifier("_private"));
        assert!(is_valid_c_identifier("NAME123"));
        assert!(is_valid_c_identifier("__builtin"));

        // Struct member access
        assert!(is_valid_c_identifier("ops.open"));
        assert!(is_valid_c_identifier("ops->close"));
        assert!(is_valid_c_identifier("device.driver.ops"));
        assert!(is_valid_c_identifier("struct_ptr->field"));

        // Invalid identifiers
        assert!(!is_valid_c_identifier(""));
        assert!(!is_valid_c_identifier("123invalid"));
        assert!(!is_valid_c_identifier("invalid-name"));
        assert!(!is_valid_c_identifier("invalid name"));
        assert!(!is_valid_c_identifier("ops.123invalid"));
        assert!(!is_valid_c_identifier("ops->"));
        assert!(!is_valid_c_identifier(".invalid"));
    }

    #[test]
    fn test_display() {
        let assignment_site = create_test_call_site();
        let function_pointer =
            FunctionPointer::new("callback".to_string(), assignment_site, "handler".to_string());

        let display = format!("{}", function_pointer);
        assert!(display.contains("callback"));
        assert!(display.contains("handler"));
        assert!(display.contains("0 usages"));
    }

    #[test]
    fn test_display_name_with_struct() {
        let assignment_site = create_test_call_site();
        let mut function_pointer =
            FunctionPointer::new("open".to_string(), assignment_site, "device_open".to_string());

        function_pointer.set_struct_context(Some("file_operations".to_string()));

        let display_name = function_pointer.display_name();
        assert!(display_name.contains("file_operations.open"));
        assert!(display_name.contains("device_open"));
        assert!(display_name.contains("0 usages"));
    }

    #[test]
    fn test_clear_and_set_usage_sites() {
        let assignment_site = create_test_call_site();
        let mut function_pointer =
            FunctionPointer::new("callback".to_string(), assignment_site, "handler".to_string());

        let usage1 = create_usage_site();
        let usage2 = create_usage_site();

        function_pointer.add_usage_site(usage1.clone());
        function_pointer.add_usage_site(usage2.clone());
        assert_eq!(function_pointer.usage_count(), 2);

        function_pointer.clear_usage_sites();
        assert_eq!(function_pointer.usage_count(), 0);

        function_pointer.set_usage_sites(vec![usage1, usage2]);
        assert_eq!(function_pointer.usage_count(), 2);
    }

    #[test]
    fn test_validation_methods() {
        let assignment_site = create_test_call_site();
        let valid_pointer = FunctionPointer::new(
            "valid_callback".to_string(),
            assignment_site.clone(),
            "valid_handler".to_string(),
        );

        assert!(valid_pointer.is_valid_pointer_name());
        assert!(valid_pointer.is_valid_function_name());

        let invalid_pointer = FunctionPointer::new(
            "123invalid".to_string(),
            assignment_site,
            "456invalid".to_string(),
        );

        assert!(!invalid_pointer.is_valid_pointer_name());
        assert!(!invalid_pointer.is_valid_function_name());
    }
}
