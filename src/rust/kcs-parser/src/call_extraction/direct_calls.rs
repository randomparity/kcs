//! Direct function call detection using Tree-sitter AST analysis.
//!
//! This module implements detection of direct function calls in C code,
//! where the function name is explicitly written at the call site.
//! These are the most common and highest-confidence call patterns.

use anyhow::{Context, Result};
use kcs_graph::{CallEdgeModel, CallTypeEnum, ConfidenceLevel};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use tree_sitter::{Language, Node, Query, QueryCursor, QueryMatch, Tree};

/// Configuration for direct call detection
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct DirectCallConfig {
    /// Include calls within conditional compilation blocks
    pub include_conditional: bool,
    /// Include calls within if/switch/loop statements
    pub include_control_flow: bool,
    /// Maximum call depth to analyze
    pub max_depth: usize,
    /// Whether to validate function name syntax
    pub validate_identifiers: bool,
}

impl Default for DirectCallConfig {
    fn default() -> Self {
        Self {
            include_conditional: true,
            include_control_flow: true,
            max_depth: 100,
            validate_identifiers: true,
        }
    }
}

/// Direct call detection result
#[derive(Debug, Clone)]
pub struct DirectCall {
    /// Name of the called function
    pub function_name: String,
    /// File path where the call occurs
    pub file_path: String,
    /// Line number of the call
    pub line_number: u32,
    /// Column number of the call
    pub column_number: u32,
    /// Name of the calling function context
    pub caller_function: Option<String>,
    /// Whether the call is within conditional compilation
    pub is_conditional: bool,
    /// Configuration guard if applicable
    pub config_guard: Option<String>,
    /// Call arguments as string representation
    pub arguments: String,
    /// Additional metadata
    pub metadata: HashMap<String, String>,
}

impl DirectCall {
    /// Convert to CallEdge model for graph storage
    pub fn to_call_edge(&self, caller_name: String) -> Result<CallEdgeModel> {
        let edge = CallEdgeModel::new(
            caller_name,
            self.function_name.clone(),
            self.file_path.clone(),
            self.line_number,
            CallTypeEnum::Direct,
            ConfidenceLevel::High,
            self.is_conditional,
        );

        Ok(edge)
    }

    /// Check if this is a kernel-specific function call
    pub fn is_kernel_function(&self) -> bool {
        // Common kernel function prefixes
        let kernel_prefixes = [
            "sys_",
            "do_",
            "vfs_",
            "generic_",
            "kernel_",
            "kmalloc",
            "kfree",
            "vmalloc",
            "vfree",
            "get_user",
            "put_user",
            "copy_from_user",
            "copy_to_user",
            "spin_lock",
            "spin_unlock",
            "mutex_lock",
            "mutex_unlock",
            "wake_up",
            "schedule",
            "msleep",
            "ssleep",
            "jiffies_to",
            "time_",
            "init_",
            "exit_",
            "__init",
            "__exit",
            "INIT_",
            "EXPORT_SYMBOL",
        ];

        kernel_prefixes.iter().any(|prefix| self.function_name.starts_with(prefix))
    }

    /// Check if this is likely a macro call
    pub fn is_likely_macro(&self) -> bool {
        // Heuristic: all uppercase with underscores
        self.function_name
            .chars()
            .all(|c| c.is_uppercase() || c == '_' || c.is_numeric())
            && self.function_name.contains('_')
    }

    /// Get call signature for display
    pub fn call_signature(&self) -> String {
        format!("{}({})", self.function_name, self.arguments)
    }
}

/// Direct function call detector
pub struct DirectCallDetector {
    /// Tree-sitter language for C
    #[allow(dead_code)]
    language: Language,
    /// Query for direct function calls
    call_query: Query,
    /// Query for function definitions to find caller context
    function_query: Query,
    /// Configuration
    config: DirectCallConfig,
    /// Query cursor for reuse
    #[allow(dead_code)]
    cursor: QueryCursor,
}

impl DirectCallDetector {
    /// Create a new direct call detector
    pub fn new(language: Language, config: DirectCallConfig) -> Result<Self> {
        // Query for direct function calls
        let call_query_source = r#"
            ; Basic function calls with identifier names
            (call_expression
              function: (identifier) @function-name
              arguments: (argument_list) @args) @call-site
        "#;

        let call_query = Query::new(language, call_query_source)
            .context("Failed to create direct call query")?;

        // Query for function definitions to determine caller context
        let function_query_source = r#"
            (function_definition
              declarator: (function_declarator
                declarator: (identifier) @function-name)
              body: (compound_statement) @function-body) @function-def

            (function_definition
              declarator: (pointer_declarator
                declarator: (function_declarator
                  declarator: (identifier) @function-name))
              body: (compound_statement) @function-body) @function-def
        "#;

        let function_query = Query::new(language, function_query_source)
            .context("Failed to create function definition query")?;

        Ok(Self {
            language,
            call_query,
            function_query,
            config,
            cursor: QueryCursor::new(),
        })
    }

    /// Extract direct function calls from source tree
    pub fn extract_calls(
        &self,
        tree: &Tree,
        source: &str,
        file_path: &str,
    ) -> Result<Vec<DirectCall>> {
        let mut calls = Vec::new();
        let root_node = tree.root_node();

        // First, find all function definitions to establish caller context
        let function_map = self.build_function_map(&root_node, source)?;

        // Use a separate cursor for call extraction
        let mut call_cursor = QueryCursor::new();
        let matches: Vec<_> =
            call_cursor.matches(&self.call_query, root_node, source.as_bytes()).collect();

        for query_match in matches {
            if let Some(call) = Self::process_call_match(
                &self.call_query,
                &self.config,
                query_match,
                source,
                file_path,
                &function_map,
            )? {
                // Apply filtering based on configuration
                if Self::should_include_call(&self.config, &call) {
                    calls.push(call);
                }
            }
        }

        Ok(calls)
    }

    /// Build a map of function definitions for caller context
    fn build_function_map(
        &self,
        root_node: &Node,
        source: &str,
    ) -> Result<HashMap<(u32, u32), String>> {
        let mut function_map = HashMap::new();

        // Use a separate cursor for function extraction
        let mut func_cursor = QueryCursor::new();
        let matches: Vec<_> = func_cursor
            .matches(&self.function_query, *root_node, source.as_bytes())
            .collect();

        for query_match in matches {
            if let (Some(name_node), Some(body_node)) = (
                Self::get_capture_node(&self.function_query, &query_match, "function-name"),
                Self::get_capture_node(&self.function_query, &query_match, "function-body"),
            ) {
                let function_name = Self::get_node_text(name_node, source);
                let start_pos = body_node.start_position();
                let end_pos = body_node.end_position();

                // Map line ranges to function names - use line number only for key
                for line in start_pos.row..=end_pos.row {
                    function_map.insert((line as u32 + 1, 0), function_name.clone());
                    // +1 for tree-sitter 0-based to 1-based conversion
                }
            }
        }

        Ok(function_map)
    }

    /// Process a single call match
    fn process_call_match(
        call_query: &Query,
        config: &DirectCallConfig,
        query_match: QueryMatch,
        source: &str,
        file_path: &str,
        function_map: &HashMap<(u32, u32), String>,
    ) -> Result<Option<DirectCall>> {
        let function_name_node =
            match Self::get_capture_node(call_query, &query_match, "function-name") {
                Some(node) => node,
                None => return Ok(None),
            };

        let call_site_node = match Self::get_capture_node(call_query, &query_match, "call-site") {
            Some(node) => node,
            None => return Ok(None),
        };

        let args_node = Self::get_capture_node(call_query, &query_match, "args");

        let function_name = Self::get_node_text(function_name_node, source);

        // Validate function name if configured
        if config.validate_identifiers && !Self::is_valid_c_identifier(&function_name) {
            return Ok(None);
        }

        let start_pos = call_site_node.start_position();
        let line_number = start_pos.row as u32 + 1; // Tree-sitter uses 0-based lines
        let column_number = start_pos.column as u32;

        // Get arguments text
        let arguments = if let Some(args) = args_node {
            Self::get_node_text(args, source)
        } else {
            String::new()
        };

        // Find caller function context - search by line number only

        let caller_function = function_map
            .get(&(line_number, 0))
            .or_else(|| {
                // Search nearby lines if exact match not found
                for offset in 1..=5 {
                    if let Some(func) = function_map.get(&(line_number.saturating_sub(offset), 0)) {
                        return Some(func);
                    }
                    if let Some(func) = function_map.get(&(line_number + offset, 0)) {
                        return Some(func);
                    }
                }
                None
            })
            .cloned();

        // Check for conditional compilation context
        let (is_conditional, config_guard) =
            Self::check_conditional_context(&call_site_node, source);

        let mut metadata = HashMap::new();

        // Add metadata for kernel-specific calls
        if function_name.starts_with("EXPORT_SYMBOL") {
            metadata.insert("category".to_string(), "export".to_string());
        } else if function_name.starts_with("module_") {
            metadata.insert("category".to_string(), "module".to_string());
        } else if function_name.starts_with("SYSCALL_DEFINE") {
            metadata.insert("category".to_string(), "syscall".to_string());
        }

        Ok(Some(DirectCall {
            function_name,
            file_path: file_path.to_string(),
            line_number,
            column_number,
            caller_function,
            is_conditional,
            config_guard,
            arguments,
            metadata,
        }))
    }

    /// Check if a call should be included based on configuration
    fn should_include_call(config: &DirectCallConfig, call: &DirectCall) -> bool {
        if !config.include_conditional && call.is_conditional {
            return false;
        }

        if !config.include_control_flow {
            // Check if call is within control flow statements
            // This would require more sophisticated AST traversal
            // For now, we include all calls
        }

        true
    }

    /// Check if a call is within conditional compilation context
    fn check_conditional_context(node: &Node, source: &str) -> (bool, Option<String>) {
        // Walk up the AST to find preprocessor directives
        let mut current = node.parent();
        while let Some(parent) = current {
            match parent.kind() {
                "preproc_if" | "preproc_ifdef" | "preproc_ifndef" => {
                    // Extract the condition
                    if let Some(condition_node) = parent.child_by_field_name("condition") {
                        let condition = Self::get_node_text(condition_node, source);
                        return (true, Some(condition));
                    }
                    return (true, None);
                },
                "preproc_else" | "preproc_elif" => {
                    return (true, Some("else_branch".to_string()));
                },
                _ => {},
            }
            current = parent.parent();
        }
        (false, None)
    }

    /// Get capture node by name from query match
    fn get_capture_node<'a>(
        query: &Query,
        query_match: &'a QueryMatch,
        capture_name: &str,
    ) -> Option<Node<'a>> {
        for capture in query_match.captures.iter() {
            let capture_name_str = &query.capture_names()[capture.index as usize];
            if capture_name_str == capture_name {
                return Some(capture.node);
            }
        }
        None
    }

    /// Get text content of a node
    fn get_node_text(node: Node, source: &str) -> String {
        let start_byte = node.start_byte();
        let end_byte = node.end_byte();
        source[start_byte..end_byte].to_string()
    }

    /// Validate C identifier
    fn is_valid_c_identifier(name: &str) -> bool {
        if name.is_empty() {
            return false;
        }

        let mut chars = name.chars();

        // First character must be letter or underscore
        match chars.next() {
            Some(c) if c.is_alphabetic() || c == '_' => {},
            _ => return false,
        }

        // Remaining characters must be alphanumeric or underscore
        chars.all(|c| c.is_alphanumeric() || c == '_')
    }

    /// Update configuration
    pub fn set_config(&mut self, config: DirectCallConfig) {
        self.config = config;
    }

    /// Get current configuration
    pub fn config(&self) -> &DirectCallConfig {
        &self.config
    }

    /// Get statistics about the last extraction
    pub fn get_stats(&self) -> ExtractionStats {
        ExtractionStats {
            total_calls_found: 0, // Would be populated during extraction
            kernel_calls: 0,
            macro_calls: 0,
            conditional_calls: 0,
        }
    }
}

/// Statistics from call extraction
#[derive(Debug, Clone, Default)]
pub struct ExtractionStats {
    pub total_calls_found: usize,
    pub kernel_calls: usize,
    pub macro_calls: usize,
    pub conditional_calls: usize,
}

#[cfg(test)]
mod tests {
    use super::*;
    use tree_sitter_c::language;

    fn create_test_detector() -> DirectCallDetector {
        let config = DirectCallConfig::default();
        DirectCallDetector::new(language(), config).unwrap()
    }

    fn parse_test_code(code: &str) -> tree_sitter::Tree {
        let mut parser = tree_sitter::Parser::new();
        parser.set_language(language()).unwrap();
        parser.parse(code, None).unwrap()
    }

    #[test]
    fn test_simple_direct_call() {
        let detector = create_test_detector();
        let code = r#"
            void test_function() {
                printf("Hello, world!");
                return;
            }
        "#;

        let tree = parse_test_code(code);
        let calls = detector.extract_calls(&tree, code, "test.c").unwrap();

        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].function_name, "printf");
        assert_eq!(calls[0].caller_function, Some("test_function".to_string()));
        assert!(!calls[0].is_conditional);
    }

    #[test]
    fn test_conditional_call() {
        let detector = create_test_detector();
        let code = r#"
            void test_function() {
                if (condition) {
                    debug_print("Debug message");
                }
            }
        "#;

        let tree = parse_test_code(code);
        let calls = detector.extract_calls(&tree, code, "test.c").unwrap();

        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].function_name, "debug_print");
    }

    #[test]
    fn test_kernel_function_detection() {
        let call = DirectCall {
            function_name: "kmalloc".to_string(),
            file_path: "test.c".to_string(),
            line_number: 1,
            column_number: 1,
            caller_function: None,
            is_conditional: false,
            config_guard: None,
            arguments: "sizeof(int), GFP_KERNEL".to_string(),
            metadata: HashMap::new(),
        };

        assert!(call.is_kernel_function());
    }

    #[test]
    fn test_macro_call_detection() {
        let call = DirectCall {
            function_name: "EXPORT_SYMBOL".to_string(),
            file_path: "test.c".to_string(),
            line_number: 1,
            column_number: 1,
            caller_function: None,
            is_conditional: false,
            config_guard: None,
            arguments: "my_function".to_string(),
            metadata: HashMap::new(),
        };

        assert!(call.is_likely_macro());
    }

    #[test]
    fn test_identifier_validation() {
        assert!(DirectCallDetector::is_valid_c_identifier("function_name"));
        assert!(DirectCallDetector::is_valid_c_identifier("_private_func"));
        assert!(DirectCallDetector::is_valid_c_identifier("func123"));
        assert!(!DirectCallDetector::is_valid_c_identifier("123func"));
        assert!(!DirectCallDetector::is_valid_c_identifier("func-name"));
        assert!(!DirectCallDetector::is_valid_c_identifier(""));
    }

    #[test]
    fn test_call_signature() {
        let call = DirectCall {
            function_name: "strlen".to_string(),
            file_path: "test.c".to_string(),
            line_number: 1,
            column_number: 1,
            caller_function: None,
            is_conditional: false,
            config_guard: None,
            arguments: "\"hello world\"".to_string(),
            metadata: HashMap::new(),
        };

        assert_eq!(call.call_signature(), "strlen(\"hello world\")");
    }

    #[test]
    fn test_call_edge_conversion() {
        let call = DirectCall {
            function_name: "target_func".to_string(),
            file_path: "test.c".to_string(),
            line_number: 42,
            column_number: 8,
            caller_function: Some("caller_func".to_string()),
            is_conditional: false,
            config_guard: None,
            arguments: "arg1, arg2".to_string(),
            metadata: HashMap::new(),
        };

        let edge = call.to_call_edge("caller_func".to_string()).unwrap();

        assert_eq!(edge.caller_name(), "caller_func");
        assert_eq!(edge.callee_name(), "target_func");
        assert_eq!(edge.file_path(), "test.c");
        assert_eq!(edge.line_number(), 42);
        assert!(edge.is_direct_call());
        assert!(edge.is_high_confidence());
    }
}
