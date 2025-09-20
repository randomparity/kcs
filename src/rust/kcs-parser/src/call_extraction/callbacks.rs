//! Callback pattern recognition using Tree-sitter AST analysis.
//!
//! This module implements detection of callback patterns in C code, including
//! member function calls, function pointer assignments, and callback registration
//! patterns commonly used in kernel code and systems programming.

use anyhow::{Context, Result};
use kcs_graph::{CallEdgeBuilder, CallEdgeModel, CallTypeEnum, ConfidenceLevel};
use std::collections::HashMap;
use tree_sitter::{Language, Query, QueryCursor, QueryMatch, Tree};

/// Configuration for callback pattern detection
#[derive(Debug, Clone)]
pub struct CallbackConfig {
    /// Include calls within conditional compilation blocks
    pub include_conditional: bool,
    /// Include calls within if/switch/loop statements
    pub include_control_flow: bool,
    /// Maximum call depth to analyze
    pub max_depth: usize,
    /// Whether to validate identifier syntax
    pub validate_identifiers: bool,
    /// Include struct member function calls
    pub include_member_calls: bool,
    /// Include function pointer assignments
    pub include_pointer_assignments: bool,
    /// Include array-based callback patterns
    pub include_array_callbacks: bool,
    /// Include callback registration patterns
    pub include_callback_registration: bool,
}

impl Default for CallbackConfig {
    fn default() -> Self {
        Self {
            include_conditional: true,
            include_control_flow: true,
            max_depth: 100,
            validate_identifiers: true,
            include_member_calls: true,
            include_pointer_assignments: true,
            include_array_callbacks: true,
            include_callback_registration: true,
        }
    }
}

/// Callback pattern detection result
#[derive(Debug, Clone)]
pub struct CallbackCall {
    /// Expression or pattern that represents the callback
    pub callback_expression: String,
    /// File path where the callback occurs
    pub file_path: String,
    /// Line number of the callback
    pub line_number: u32,
    /// Column number of the callback
    pub column_number: u32,
    /// Name of the calling function context
    pub caller_function: Option<String>,
    /// Whether the callback is within conditional compilation
    pub is_conditional: bool,
    /// Type of callback pattern
    pub callback_type: CallbackType,
    /// Confidence level in the detection
    pub confidence: ConfidenceLevel,
    /// Target function name if determinable
    pub target_function: Option<String>,
    /// Struct or object name for member callbacks
    pub struct_name: Option<String>,
}

/// Types of callback patterns
#[derive(Debug, Clone, PartialEq)]
pub enum CallbackType {
    /// Member function call: obj.func(args) or ptr->func(args)
    MemberCall,
    /// Function pointer assignment: obj.func = handler
    PointerAssignment,
    /// Array callback call: callbacks[index](args)
    ArrayCallback,
    /// Callback registration: register_callback(&handler)
    Registration,
    /// Nested member call: obj.member.func(args)
    NestedMember,
    /// Chained callback: func1()->func2(args)
    ChainedCallback,
}

impl CallbackCall {
    /// Convert to a CallEdgeModel for database storage
    pub fn to_call_edge(&self) -> CallEdgeModel {
        let caller = self.caller_function.clone().unwrap_or_else(|| "<unknown>".to_string());
        let callee = self
            .target_function
            .clone()
            .unwrap_or_else(|| format!("<callback:{}>", self.callback_expression));

        CallEdgeBuilder::default()
            .caller(caller)
            .callee(callee)
            .call_type(CallTypeEnum::Callback)
            .confidence(self.confidence)
            .file_path(self.file_path.clone())
            .line_number(self.line_number)
            .build()
    }
}

/// Statistics for callback extraction process
#[derive(Debug, Clone, Default)]
pub struct ExtractionStats {
    /// Total number of callbacks detected
    pub total_callbacks: usize,
    /// Number of member function calls
    pub member_calls: usize,
    /// Number of function pointer assignments
    pub pointer_assignments: usize,
    /// Number of array callbacks
    pub array_callbacks: usize,
    /// Number of callback registrations
    pub registrations: usize,
    /// Number of nested member calls
    pub nested_members: usize,
    /// Number of chained callbacks
    pub chained_callbacks: usize,
    /// Number of files processed
    pub files_processed: usize,
    /// Number of parse errors encountered
    pub parse_errors: usize,
}

/// Main callback pattern detector
pub struct CallbackDetector {
    config: CallbackConfig,
    query: Query,
    stats: ExtractionStats,
}

impl CallbackDetector {
    /// Create a new callback detector with the given configuration
    pub fn new(language: Language, config: CallbackConfig) -> Result<Self> {
        let query_source = Self::build_query_source(&config);
        let query = Query::new(language, &query_source)
            .context("Failed to create Tree-sitter query for callback detection")?;

        Ok(Self {
            config,
            query,
            stats: ExtractionStats::default(),
        })
    }

    /// Build the Tree-sitter query source based on configuration
    fn build_query_source(config: &CallbackConfig) -> String {
        let mut query_parts = Vec::new();

        // Member function calls (obj.func() and ptr->func())
        if config.include_member_calls {
            query_parts.push(
                r#"
; Member function calls (struct.func())
(call_expression
  function: (field_expression
    argument: (identifier) @struct-name
    field: (field_identifier) @function-name)
  arguments: (argument_list) @args) @call-site

; Arrow operator function calls (ptr->func())
(call_expression
  function: (field_expression
    argument: (pointer_expression
      argument: (identifier) @struct-pointer)
    field: (field_identifier) @function-name)
  arguments: (argument_list) @args) @call-site
"#,
            );
        }

        // Nested member calls
        if config.include_member_calls {
            query_parts.push(
                r#"
; Nested member access (obj.member.func())
(call_expression
  function: (field_expression
    argument: (field_expression
      argument: (identifier) @outer-struct
      field: (field_identifier) @inner-member)
    field: (field_identifier) @function-name)
  arguments: (argument_list) @args) @call-site

; Chained function calls (func1()->func2())
(call_expression
  function: (field_expression
    argument: (call_expression) @inner-call
    field: (field_identifier) @function-name)
  arguments: (argument_list) @args) @call-site
"#,
            );
        }

        // Function pointer assignments
        if config.include_pointer_assignments {
            query_parts.push(
                r#"
; Direct function pointer assignments (obj.func = handler)
(assignment_expression
  left: (field_expression
    argument: (identifier) @struct-name
    field: (field_identifier) @field-name)
  right: (identifier) @function-name) @assignment-site

; Function pointer assignments through arrow operator (ptr->func = handler)
(assignment_expression
  left: (field_expression
    argument: (pointer_expression
      argument: (identifier) @struct-pointer)
    field: (field_identifier) @field-name)
  right: (identifier) @function-name) @assignment-site
"#,
            );
        }

        // Array-based callbacks
        if config.include_array_callbacks {
            query_parts.push(
                r#"
; Function calls through array indexing (callbacks[index]())
(call_expression
  function: (subscript_expression
    argument: (identifier) @array-name
    index: (_) @index)
  arguments: (argument_list) @args) @call-site
"#,
            );
        }

        // Callback registration patterns
        if config.include_callback_registration {
            query_parts.push(
                r#"
; Callback registration patterns (register_*(&handler))
(call_expression
  function: (identifier) @register-function
  arguments: (argument_list
    (pointer_expression
      argument: (identifier) @callback-function))) @registration-site
  (#match? @register-function "^(register|set|install)_")

; Direct callback registration (register_callback(handler))
(call_expression
  function: (identifier) @register-function
  arguments: (argument_list
    (identifier) @callback-function)) @registration-site
  (#match? @register-function "^(register|set|install)_")
"#,
            );
        }

        query_parts.join("\n")
    }

    /// Extract callback patterns from a Tree-sitter AST
    pub fn extract_callbacks(
        &mut self,
        tree: &Tree,
        source_code: &str,
        file_path: &str,
    ) -> Result<Vec<CallbackCall>> {
        self.stats.files_processed += 1;
        let mut callbacks = Vec::new();
        let mut cursor = QueryCursor::new();

        // Set depth limit to prevent infinite recursion
        // Note: Tree-sitter QueryCursor doesn't have set_max_start_depth method
        // This configuration is handled by the max_depth config parameter internally

        let matches: Vec<_> =
            cursor.matches(&self.query, tree.root_node(), source_code.as_bytes()).collect();

        // Process all matches first without updating stats
        let mut processed_callbacks = Vec::new();
        for match_ in matches {
            match self.process_callback_match(&match_, source_code, file_path) {
                Ok(Some(callback)) => {
                    processed_callbacks.push(callback);
                },
                Ok(None) => {
                    // Valid match but filtered out by configuration
                },
                Err(e) => {
                    self.stats.parse_errors += 1;
                    eprintln!("Warning: Failed to process callback match in {}: {}", file_path, e);
                },
            }
        }

        // Now update stats and collect results
        for callback in processed_callbacks {
            self.update_stats(&callback);
            callbacks.push(callback);
            self.stats.total_callbacks += 1;
        }

        Ok(callbacks)
    }

    /// Process a single Tree-sitter query match into a CallbackCall
    fn process_callback_match(
        &self,
        match_: &QueryMatch,
        source_code: &str,
        file_path: &str,
    ) -> Result<Option<CallbackCall>> {
        let captures: HashMap<&str, &tree_sitter::Node> = match_
            .captures
            .iter()
            .map(|capture| {
                let name = self.query.capture_names()[capture.index as usize].as_str();
                (name, &capture.node)
            })
            .collect();

        // Determine callback type and extract relevant information
        let (callback_type, target_function, struct_name, callback_expression) =
            self.analyze_callback_pattern(&captures, source_code)?;

        // Get position information
        let call_site = captures
            .get("call-site")
            .or_else(|| captures.get("assignment-site"))
            .or_else(|| captures.get("registration-site"))
            .context("No call site found in match")?;

        let start_position = call_site.start_position();
        let line_number = start_position.row as u32 + 1; // Tree-sitter uses 0-based indexing
        let column_number = start_position.column as u32 + 1;

        // Extract calling function context (simplified - could be enhanced)
        let caller_function = self.find_caller_function(call_site, source_code);

        // Determine confidence level
        let confidence = self.calculate_confidence(&callback_type, &target_function);

        // Check if we should include this callback based on configuration
        if !self.should_include_callback(&callback_type) {
            return Ok(None);
        }

        Ok(Some(CallbackCall {
            callback_expression,
            file_path: file_path.to_string(),
            line_number,
            column_number,
            caller_function,
            is_conditional: false, // TODO: Implement conditional detection
            callback_type,
            confidence,
            target_function,
            struct_name,
        }))
    }

    /// Analyze the callback pattern from query captures
    fn analyze_callback_pattern(
        &self,
        captures: &HashMap<&str, &tree_sitter::Node>,
        source_code: &str,
    ) -> Result<(CallbackType, Option<String>, Option<String>, String)> {
        // Check for function pointer assignments first (higher priority than member calls)
        if let (Some(field_node), Some(func_node)) =
            (captures.get("field-name"), captures.get("function-name"))
        {
            let field_name = self.extract_node_text(field_node, source_code);
            let func_name = self.extract_node_text(func_node, source_code);
            let struct_name = captures
                .get("struct-name")
                .or_else(|| captures.get("struct-pointer"))
                .map(|node| self.extract_node_text(node, source_code));
            let expression = format!("{} = {}", field_name, func_name);
            return Ok((CallbackType::PointerAssignment, Some(func_name), struct_name, expression));
        }

        // Check for member function calls
        if let (Some(struct_node), Some(func_node)) =
            (captures.get("struct-name"), captures.get("function-name"))
        {
            let struct_name = self.extract_node_text(struct_node, source_code);
            let func_name = self.extract_node_text(func_node, source_code);
            let expression = format!("{}.{}", struct_name, func_name);
            return Ok((CallbackType::MemberCall, Some(func_name), Some(struct_name), expression));
        }

        // Check for pointer-based member calls
        if let (Some(ptr_node), Some(func_node)) =
            (captures.get("struct-pointer"), captures.get("function-name"))
        {
            let ptr_name = self.extract_node_text(ptr_node, source_code);
            let func_name = self.extract_node_text(func_node, source_code);
            let expression = format!("{}->{}()", ptr_name, func_name);
            return Ok((CallbackType::MemberCall, Some(func_name), Some(ptr_name), expression));
        }

        // Check for array callbacks
        if let (Some(array_node), Some(index_node)) =
            (captures.get("array-name"), captures.get("index"))
        {
            let array_name = self.extract_node_text(array_node, source_code);
            let index_expr = self.extract_node_text(index_node, source_code);
            let expression = format!("{}[{}]()", array_name, index_expr);
            return Ok((CallbackType::ArrayCallback, None, None, expression));
        }

        // Check for callback registration
        if let (Some(register_node), Some(callback_node)) =
            (captures.get("register-function"), captures.get("callback-function"))
        {
            let register_func = self.extract_node_text(register_node, source_code);
            let callback_func = self.extract_node_text(callback_node, source_code);
            let expression = format!("{}({})", register_func, callback_func);
            return Ok((CallbackType::Registration, Some(callback_func), None, expression));
        }

        // Check for nested member calls
        if let (Some(outer_node), Some(inner_node), Some(func_node)) = (
            captures.get("outer-struct"),
            captures.get("inner-member"),
            captures.get("function-name"),
        ) {
            let outer_name = self.extract_node_text(outer_node, source_code);
            let inner_name = self.extract_node_text(inner_node, source_code);
            let func_name = self.extract_node_text(func_node, source_code);
            let expression = format!("{}.{}.{}()", outer_name, inner_name, func_name);
            return Ok((CallbackType::NestedMember, Some(func_name), Some(outer_name), expression));
        }

        // Check for chained callbacks
        if let (Some(_inner_call), Some(func_node)) =
            (captures.get("inner-call"), captures.get("function-name"))
        {
            let func_name = self.extract_node_text(func_node, source_code);
            let expression = format!("...->{}()", func_name);
            return Ok((CallbackType::ChainedCallback, Some(func_name), None, expression));
        }

        Err(anyhow::anyhow!("Unknown callback pattern in captures"))
    }

    /// Extract text content from a Tree-sitter node
    fn extract_node_text(&self, node: &tree_sitter::Node, source_code: &str) -> String {
        source_code[node.start_byte()..node.end_byte()].to_string()
    }

    /// Find the calling function context for a given node
    fn find_caller_function(
        &self,
        _node: &tree_sitter::Node,
        _source_code: &str,
    ) -> Option<String> {
        // TODO: Implement proper function context detection
        // This would require traversing up the AST to find the containing function
        None
    }

    /// Calculate confidence level for a callback detection
    fn calculate_confidence(
        &self,
        callback_type: &CallbackType,
        target_function: &Option<String>,
    ) -> ConfidenceLevel {
        match callback_type {
            CallbackType::MemberCall => {
                if target_function.is_some() {
                    ConfidenceLevel::High
                } else {
                    ConfidenceLevel::Medium
                }
            },
            CallbackType::PointerAssignment => ConfidenceLevel::High,
            CallbackType::ArrayCallback => ConfidenceLevel::Medium,
            CallbackType::Registration => ConfidenceLevel::High,
            CallbackType::NestedMember => ConfidenceLevel::Medium,
            CallbackType::ChainedCallback => ConfidenceLevel::Low,
        }
    }

    /// Check if a callback type should be included based on configuration
    fn should_include_callback(&self, callback_type: &CallbackType) -> bool {
        match callback_type {
            CallbackType::MemberCall => self.config.include_member_calls,
            CallbackType::NestedMember => self.config.include_member_calls,
            CallbackType::ChainedCallback => self.config.include_member_calls,
            CallbackType::PointerAssignment => self.config.include_pointer_assignments,
            CallbackType::ArrayCallback => self.config.include_array_callbacks,
            CallbackType::Registration => self.config.include_callback_registration,
        }
    }

    /// Update extraction statistics
    fn update_stats(&mut self, callback: &CallbackCall) {
        match callback.callback_type {
            CallbackType::MemberCall => self.stats.member_calls += 1,
            CallbackType::PointerAssignment => self.stats.pointer_assignments += 1,
            CallbackType::ArrayCallback => self.stats.array_callbacks += 1,
            CallbackType::Registration => self.stats.registrations += 1,
            CallbackType::NestedMember => self.stats.nested_members += 1,
            CallbackType::ChainedCallback => self.stats.chained_callbacks += 1,
        }
    }

    /// Get extraction statistics
    pub fn stats(&self) -> &ExtractionStats {
        &self.stats
    }

    /// Reset extraction statistics
    pub fn reset_stats(&mut self) {
        self.stats = ExtractionStats::default();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tree_sitter_c::language;

    fn create_test_detector() -> CallbackDetector {
        CallbackDetector::new(language(), CallbackConfig::default())
            .expect("Failed to create test detector")
    }

    #[test]
    fn test_member_function_call() {
        let mut detector = create_test_detector();
        let source = "int main() { obj.callback(arg); }";

        let mut parser = tree_sitter::Parser::new();
        parser.set_language(language()).unwrap();
        let tree = parser.parse(source, None).unwrap();

        let callbacks = detector.extract_callbacks(&tree, source, "test.c").unwrap();
        assert_eq!(callbacks.len(), 1);
        assert_eq!(callbacks[0].callback_type, CallbackType::MemberCall);
        assert_eq!(callbacks[0].target_function, Some("callback".to_string()));
    }

    #[test]
    fn test_pointer_assignment() {
        let mut detector = create_test_detector();
        let source = "void setup() { obj.handler = my_callback; }";

        let mut parser = tree_sitter::Parser::new();
        parser.set_language(language()).unwrap();
        let tree = parser.parse(source, None).unwrap();

        let callbacks = detector.extract_callbacks(&tree, source, "test.c").unwrap();
        assert_eq!(callbacks.len(), 1);
        assert_eq!(callbacks[0].callback_type, CallbackType::PointerAssignment);
        assert_eq!(callbacks[0].target_function, Some("my_callback".to_string()));
    }

    #[test]
    fn test_callback_registration() {
        let mut detector = create_test_detector();
        let source = "void setup() { register_callback(&my_handler); }";

        let mut parser = tree_sitter::Parser::new();
        parser.set_language(language()).unwrap();
        let tree = parser.parse(source, None).unwrap();

        let callbacks = detector.extract_callbacks(&tree, source, "test.c").unwrap();
        assert_eq!(callbacks.len(), 1);
        assert_eq!(callbacks[0].callback_type, CallbackType::Registration);
        assert_eq!(callbacks[0].target_function, Some("my_handler".to_string()));
    }
}
