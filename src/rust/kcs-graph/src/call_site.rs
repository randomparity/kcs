//! CallSite data model for representing function call locations.
//!
//! This module defines the CallSite struct that represents the specific location
//! where a function call occurs in source code. CallSite provides detailed
//! context about the call location including file path, line/column numbers,
//! function context, and surrounding code context.

use serde::{Deserialize, Serialize};
use std::fmt;

/// Represents a specific location where a function call occurs.
///
/// A CallSite captures detailed information about where a function call
/// is made in the source code, including the exact position, surrounding
/// context, and the function containing the call.
///
/// # Examples
///
/// ```rust
/// use kcs_graph::CallSite;
///
/// let call_site = CallSite::new(
///     "/path/to/file.c".to_string(),
///     42,
///     10,
///     "main_function".to_string(),
/// );
///
/// assert_eq!(call_site.file_path(), "/path/to/file.c");
/// assert_eq!(call_site.line_number(), 42);
/// assert_eq!(call_site.column_number(), 10);
/// assert_eq!(call_site.function_context(), "main_function");
/// ```
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CallSite {
    /// File path where the call occurs
    file_path: String,

    /// Line number where the call occurs (1-based)
    line_number: u32,

    /// Column number where the call occurs (1-based)
    column_number: u32,

    /// Name of the function containing this call
    function_context: String,

    /// Optional source code snippet around the call
    source_snippet: Option<String>,

    /// Optional preprocessor context (macros, includes)
    preprocessor_context: Option<String>,

    /// Additional metadata as key-value pairs
    metadata: std::collections::HashMap<String, String>,
}

impl CallSite {
    /// Creates a new CallSite with the specified parameters.
    ///
    /// # Arguments
    ///
    /// * `file_path` - File path where the call occurs
    /// * `line_number` - Line number where the call occurs (1-based)
    /// * `column_number` - Column number where the call occurs (1-based)
    /// * `function_context` - Name of the function containing this call
    ///
    /// # Examples
    ///
    /// ```rust
    /// use kcs_graph::CallSite;
    ///
    /// let call_site = CallSite::new(
    ///     "fs/open.c".to_string(),
    ///     123,
    ///     15,
    ///     "sys_open".to_string(),
    /// );
    /// ```
    pub fn new(
        file_path: String,
        line_number: u32,
        column_number: u32,
        function_context: String,
    ) -> Self {
        Self {
            file_path,
            line_number,
            column_number,
            function_context,
            source_snippet: None,
            preprocessor_context: None,
            metadata: std::collections::HashMap::new(),
        }
    }

    /// Creates a new CallSite builder for fluent construction.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use kcs_graph::CallSite;
    ///
    /// let call_site = CallSite::builder()
    ///     .file_path("main.c")
    ///     .line_number(42)
    ///     .column_number(10)
    ///     .function_context("main")
    ///     .source_snippet("result = helper_function(42);")
    ///     .build();
    /// ```
    pub fn builder() -> CallSiteBuilder {
        CallSiteBuilder::default()
    }

    // Accessor methods

    /// Returns the file path where the call occurs.
    pub fn file_path(&self) -> &str {
        &self.file_path
    }

    /// Returns the line number where the call occurs.
    pub fn line_number(&self) -> u32 {
        self.line_number
    }

    /// Returns the column number where the call occurs.
    pub fn column_number(&self) -> u32 {
        self.column_number
    }

    /// Returns the name of the function containing this call.
    pub fn function_context(&self) -> &str {
        &self.function_context
    }

    /// Returns the source code snippet around the call, if available.
    pub fn source_snippet(&self) -> Option<&str> {
        self.source_snippet.as_deref()
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

    /// Sets the source code snippet around the call.
    pub fn set_source_snippet(&mut self, snippet: Option<String>) {
        self.source_snippet = snippet;
    }

    /// Sets the preprocessor context.
    pub fn set_preprocessor_context(&mut self, context: Option<String>) {
        self.preprocessor_context = context;
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

    /// Returns a position string in the format "file:line:column".
    ///
    /// # Examples
    ///
    /// ```rust
    /// use kcs_graph::CallSite;
    ///
    /// let call_site = CallSite::new(
    ///     "main.c".to_string(),
    ///     42,
    ///     10,
    ///     "main".to_string(),
    /// );
    ///
    /// assert_eq!(call_site.position_string(), "main.c:42:10");
    /// ```
    pub fn position_string(&self) -> String {
        format!("{}:{}:{}", self.file_path, self.line_number, self.column_number)
    }

    /// Returns a context string with function and position.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use kcs_graph::CallSite;
    ///
    /// let call_site = CallSite::new(
    ///     "main.c".to_string(),
    ///     42,
    ///     10,
    ///     "main".to_string(),
    /// );
    ///
    /// assert_eq!(call_site.context_string(), "main() at main.c:42:10");
    /// ```
    pub fn context_string(&self) -> String {
        format!(
            "{}() at {}:{}:{}",
            self.function_context, self.file_path, self.line_number, self.column_number
        )
    }

    /// Creates a unique identifier for this call site.
    ///
    /// The identifier includes the file path, line number, column number,
    /// and function context to ensure uniqueness.
    pub fn site_id(&self) -> String {
        format!(
            "{}:{}:{}:{}",
            self.file_path, self.line_number, self.column_number, self.function_context
        )
    }

    /// Returns true if this call site is in the same file as another.
    pub fn same_file(&self, other: &CallSite) -> bool {
        self.file_path == other.file_path
    }

    /// Returns true if this call site is in the same function as another.
    pub fn same_function(&self, other: &CallSite) -> bool {
        self.function_context == other.function_context
    }

    /// Returns the distance in lines between this call site and another.
    ///
    /// Returns None if the call sites are in different files.
    pub fn line_distance(&self, other: &CallSite) -> Option<u32> {
        if self.same_file(other) {
            Some(self.line_number.abs_diff(other.line_number))
        } else {
            None
        }
    }

    /// Returns true if this call site is before another in the same file.
    pub fn is_before(&self, other: &CallSite) -> Option<bool> {
        if self.same_file(other) {
            Some(
                self.line_number < other.line_number
                    || (self.line_number == other.line_number
                        && self.column_number < other.column_number),
            )
        } else {
            None
        }
    }

    /// Returns true if this call site is after another in the same file.
    pub fn is_after(&self, other: &CallSite) -> Option<bool> {
        if self.same_file(other) {
            Some(
                self.line_number > other.line_number
                    || (self.line_number == other.line_number
                        && self.column_number > other.column_number),
            )
        } else {
            None
        }
    }
}

impl fmt::Display for CallSite {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{}() at {}:{}:{}",
            self.function_context, self.file_path, self.line_number, self.column_number
        )
    }
}

/// Builder for constructing CallSite instances.
///
/// Provides a fluent interface for creating CallSite instances with
/// optional parameters and validation.
#[derive(Debug, Default)]
pub struct CallSiteBuilder {
    file_path: Option<String>,
    line_number: Option<u32>,
    column_number: Option<u32>,
    function_context: Option<String>,
    source_snippet: Option<String>,
    preprocessor_context: Option<String>,
    metadata: std::collections::HashMap<String, String>,
}

impl CallSiteBuilder {
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

    /// Sets the column number.
    pub fn column_number(mut self, column: u32) -> Self {
        self.column_number = Some(column);
        self
    }

    /// Sets the function context.
    pub fn function_context<S: Into<String>>(mut self, context: S) -> Self {
        self.function_context = Some(context.into());
        self
    }

    /// Sets the source snippet.
    pub fn source_snippet<S: Into<String>>(mut self, snippet: S) -> Self {
        self.source_snippet = Some(snippet.into());
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

    /// Builds the CallSite instance.
    ///
    /// # Panics
    ///
    /// Panics if any required field is missing.
    pub fn build(self) -> CallSite {
        CallSite {
            file_path: self.file_path.expect("file_path is required"),
            line_number: self.line_number.expect("line_number is required"),
            column_number: self.column_number.expect("column_number is required"),
            function_context: self.function_context.expect("function_context is required"),
            source_snippet: self.source_snippet,
            preprocessor_context: self.preprocessor_context,
            metadata: self.metadata,
        }
    }

    /// Attempts to build the CallSite instance, returning an error if validation fails.
    pub fn try_build(self) -> Result<CallSite, String> {
        let file_path = self.file_path.ok_or("file_path is required")?;
        let line_number = self.line_number.ok_or("line_number is required")?;
        let column_number = self.column_number.ok_or("column_number is required")?;
        let function_context = self.function_context.ok_or("function_context is required")?;

        // Validation
        if file_path.is_empty() {
            return Err("file_path cannot be empty".to_string());
        }
        if function_context.is_empty() {
            return Err("function_context cannot be empty".to_string());
        }
        if line_number == 0 {
            return Err("line_number must be greater than 0".to_string());
        }
        if column_number == 0 {
            return Err("column_number must be greater than 0".to_string());
        }

        Ok(CallSite {
            file_path,
            line_number,
            column_number,
            function_context,
            source_snippet: self.source_snippet,
            preprocessor_context: self.preprocessor_context,
            metadata: self.metadata,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_call_site_new() {
        let call_site = CallSite::new("main.c".to_string(), 42, 10, "main".to_string());

        assert_eq!(call_site.file_path(), "main.c");
        assert_eq!(call_site.line_number(), 42);
        assert_eq!(call_site.column_number(), 10);
        assert_eq!(call_site.function_context(), "main");
        assert!(call_site.source_snippet().is_none());
        assert!(call_site.preprocessor_context().is_none());
    }

    #[test]
    fn test_call_site_builder() {
        let call_site = CallSite::builder()
            .file_path("fs/open.c")
            .line_number(123)
            .column_number(15)
            .function_context("sys_open")
            .source_snippet("result = generic_file_open(file, flags);")
            .preprocessor_context("#include <linux/fs.h>")
            .metadata("context", "file_operation")
            .build();

        assert_eq!(call_site.file_path(), "fs/open.c");
        assert_eq!(call_site.line_number(), 123);
        assert_eq!(call_site.column_number(), 15);
        assert_eq!(call_site.function_context(), "sys_open");
        assert_eq!(call_site.source_snippet(), Some("result = generic_file_open(file, flags);"));
        assert_eq!(call_site.preprocessor_context(), Some("#include <linux/fs.h>"));
        assert_eq!(call_site.metadata().get("context"), Some(&"file_operation".to_string()));
    }

    #[test]
    fn test_call_site_utility_methods() {
        let call_site = CallSite::new("main.c".to_string(), 42, 10, "main".to_string());

        assert_eq!(call_site.position_string(), "main.c:42:10");
        assert_eq!(call_site.context_string(), "main() at main.c:42:10");
        assert_eq!(call_site.site_id(), "main.c:42:10:main");
    }

    #[test]
    fn test_call_site_comparisons() {
        let site1 = CallSite::new("main.c".to_string(), 10, 5, "main".to_string());
        let site2 = CallSite::new("main.c".to_string(), 20, 10, "main".to_string());
        let site3 = CallSite::new("other.c".to_string(), 15, 8, "other".to_string());

        assert!(site1.same_file(&site2));
        assert!(!site1.same_file(&site3));

        assert!(site1.same_function(&site2));
        assert!(!site1.same_function(&site3));

        assert_eq!(site1.line_distance(&site2), Some(10));
        assert_eq!(site1.line_distance(&site3), None);

        assert_eq!(site1.is_before(&site2), Some(true));
        assert_eq!(site2.is_before(&site1), Some(false));
        assert_eq!(site1.is_before(&site3), None);

        assert_eq!(site1.is_after(&site2), Some(false));
        assert_eq!(site2.is_after(&site1), Some(true));
        assert_eq!(site1.is_after(&site3), None);
    }

    #[test]
    fn test_call_site_metadata() {
        let mut call_site = CallSite::new("file.c".to_string(), 1, 1, "func".to_string());

        call_site.add_metadata("key1".to_string(), "value1".to_string());
        call_site.add_metadata("key2".to_string(), "value2".to_string());

        assert_eq!(call_site.metadata().get("key1"), Some(&"value1".to_string()));
        assert_eq!(call_site.metadata().get("key2"), Some(&"value2".to_string()));

        let removed = call_site.remove_metadata("key1");
        assert_eq!(removed, Some("value1".to_string()));
        assert!(call_site.metadata().get("key1").is_none());

        call_site.clear_metadata();
        assert!(call_site.metadata().is_empty());
    }

    #[test]
    fn test_builder_validation() {
        let result = CallSite::builder()
            .file_path("")
            .line_number(1)
            .column_number(1)
            .function_context("func")
            .try_build();

        assert!(result.is_err());
        assert!(result.unwrap_err().contains("file_path cannot be empty"));

        let result = CallSite::builder()
            .file_path("file.c")
            .line_number(0)
            .column_number(1)
            .function_context("func")
            .try_build();

        assert!(result.is_err());
        assert!(result.unwrap_err().contains("line_number must be greater than 0"));
    }

    #[test]
    fn test_display() {
        let call_site = CallSite::new("main.c".to_string(), 42, 10, "main".to_string());

        let display = format!("{}", call_site);
        assert_eq!(display, "main() at main.c:42:10");
    }

    #[test]
    fn test_same_line_column_ordering() {
        let site1 = CallSite::new("file.c".to_string(), 10, 5, "func".to_string());
        let site2 = CallSite::new("file.c".to_string(), 10, 15, "func".to_string());

        assert_eq!(site1.is_before(&site2), Some(true));
        assert_eq!(site2.is_before(&site1), Some(false));
        assert_eq!(site1.is_after(&site2), Some(false));
        assert_eq!(site2.is_after(&site1), Some(true));
    }
}
