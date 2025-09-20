//! Function pointer call detection using Tree-sitter AST analysis.
//!
//! This module implements detection of function pointer calls in C code,
//! including explicit pointer dereferences, implicit calls, member function
//! pointers, and array-based callback mechanisms commonly used in kernel code.

use anyhow::{Context, Result};
use kcs_graph::{CallEdgeBuilder, CallEdgeModel, CallTypeEnum, ConfidenceLevel};
use std::collections::HashMap;
use tree_sitter::{Language, Query, QueryCursor, QueryMatch, Tree};

/// Configuration for function pointer call detection
#[derive(Debug, Clone)]
pub struct PointerCallConfig {
    /// Include calls within conditional compilation blocks
    pub include_conditional: bool,
    /// Include calls within if/switch/loop statements
    pub include_control_flow: bool,
    /// Maximum call depth to analyze
    pub max_depth: usize,
    /// Whether to validate pointer name syntax
    pub validate_identifiers: bool,
    /// Include array-based callback patterns
    pub include_callback_arrays: bool,
    /// Include member function pointer calls
    pub include_member_pointers: bool,
}

impl Default for PointerCallConfig {
    fn default() -> Self {
        Self {
            include_conditional: true,
            include_control_flow: true,
            max_depth: 100,
            validate_identifiers: true,
            include_callback_arrays: true,
            include_member_pointers: true,
        }
    }
}

/// Function pointer call detection result
#[derive(Debug, Clone)]
pub struct PointerCall {
    /// Name or expression of the function pointer being called
    pub pointer_expression: String,
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
    /// Type of pointer call (explicit, implicit, member, array)
    pub call_type: PointerCallType,
    /// Confidence level in the detection
    pub confidence: ConfidenceLevel,
}

/// Types of function pointer calls
#[derive(Debug, Clone, PartialEq)]
pub enum PointerCallType {
    /// Explicit pointer dereference: (*func_ptr)(args)
    ExplicitDereference,
    /// Implicit call: func_ptr(args) where func_ptr is a pointer
    Implicit,
    /// Member function pointer: obj->func_ptr(args)
    MemberPointer,
    /// Array callback: callbacks[index](args)
    ArrayCallback,
}

impl PointerCall {
    /// Convert to a CallEdgeModel for database storage
    pub fn to_call_edge(&self) -> CallEdgeModel {
        CallEdgeBuilder::default()
            .caller(self.caller_function.clone().unwrap_or_else(|| "<unknown>".to_string()))
            .callee(self.pointer_expression.clone())
            .call_type(match self.call_type {
                PointerCallType::ExplicitDereference => CallTypeEnum::Indirect,
                PointerCallType::Implicit => CallTypeEnum::Indirect,
                PointerCallType::MemberPointer => CallTypeEnum::Indirect,
                PointerCallType::ArrayCallback => CallTypeEnum::Callback,
            })
            .confidence(self.confidence)
            .file_path(self.file_path.clone())
            .line_number(self.line_number)
            .build()
    }

    /// Check if this is a kernel callback pattern
    pub fn is_kernel_callback(&self) -> bool {
        // Common kernel callback patterns
        let callback_patterns = [
            "ops->",
            "callback",
            "handler",
            "->read",
            "->write",
            "->open",
            "->close",
            "->ioctl",
            "->probe",
            "->remove",
            "->suspend",
            "->resume",
            "->init",
            "->exit",
            "->show",
            "->store",
            "->get",
            "->set",
            "workqueue",
            "tasklet",
            "timer",
            "interrupt",
            "irq_handler",
            "bottom_half",
        ];

        callback_patterns
            .iter()
            .any(|&pattern| self.pointer_expression.contains(pattern))
    }

    /// Get the potential target function names for this pointer call
    pub fn get_potential_targets(&self) -> Vec<String> {
        // For now, we can't statically determine function pointer targets
        // This would require dataflow analysis or runtime information
        vec![]
    }
}

/// Extraction statistics
#[derive(Debug, Default)]
pub struct ExtractionStats {
    pub total_calls: usize,
    pub explicit_dereference_calls: usize,
    pub implicit_calls: usize,
    pub member_pointer_calls: usize,
    pub array_callback_calls: usize,
    pub kernel_callbacks: usize,
    pub high_confidence: usize,
    pub medium_confidence: usize,
    pub low_confidence: usize,
}

/// Function pointer call detector
pub struct PointerCallDetector {
    #[allow(dead_code)]
    language: Language,
    call_query: Query,
    function_query: Query,
    config: PointerCallConfig,
    #[allow(dead_code)]
    cursor: QueryCursor,
}

impl PointerCallDetector {
    /// Create a new function pointer call detector
    pub fn new(language: Language, config: PointerCallConfig) -> Result<Self> {
        // Query for function pointer calls
        let call_query_source = r#"
            ; Explicit pointer dereference calls: (*func_ptr)(args)
            (call_expression
              function: (parenthesized_expression
                (pointer_expression
                  argument: (identifier) @pointer-name)) @function-expr
              arguments: (argument_list) @args) @call-site

            ; Array callback calls: callbacks[index](args)
            (call_expression
              function: (subscript_expression
                argument: (identifier) @array-name
                index: (_) @index) @function-expr
              arguments: (argument_list) @args) @call-site

            ; Member function pointer calls: obj->func_ptr(args) or obj.func_ptr(args)
            (call_expression
              function: (field_expression
                argument: (_) @object
                field: (field_identifier) @field-name) @function-expr
              arguments: (argument_list) @args) @call-site
        "#;

        let call_query = Query::new(language, call_query_source)
            .context("Failed to create pointer call query")?;

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

    /// Extract function pointer calls from source tree
    pub fn extract_calls(
        &self,
        tree: &Tree,
        source: &str,
        file_path: &str,
    ) -> Result<Vec<PointerCall>> {
        let mut calls = Vec::new();
        let function_map = Self::build_function_map(&self.function_query, tree, source)?;

        // Create a new cursor for the call query
        let mut call_cursor = QueryCursor::new();

        // Execute the query and collect matches
        let matches: Vec<QueryMatch> = call_cursor
            .matches(&self.call_query, tree.root_node(), source.as_bytes())
            .collect();

        for query_match in matches {
            if let Some(call) = Self::process_call_match(
                &self.call_query,
                &query_match,
                source,
                file_path,
                &function_map,
                &self.config,
            )? {
                calls.push(call);
            }
        }

        Ok(calls)
    }

    /// Build a map of line numbers to function names for context resolution
    fn build_function_map(
        function_query: &Query,
        tree: &Tree,
        source: &str,
    ) -> Result<HashMap<(u32, u32), String>> {
        let mut function_map = HashMap::new();
        let mut function_cursor = QueryCursor::new();

        // Execute the function query
        let matches: Vec<QueryMatch> = function_cursor
            .matches(function_query, tree.root_node(), source.as_bytes())
            .collect();

        for query_match in matches {
            let mut function_name = None;
            let mut function_body = None;

            for capture in query_match.captures {
                let capture_name = &function_query.capture_names()[capture.index as usize];
                match capture_name.as_str() {
                    "function-name" => {
                        function_name = Some(capture.node.utf8_text(source.as_bytes())?);
                    },
                    "function-body" => {
                        function_body = Some(capture.node);
                    },
                    _ => {},
                }
            }

            if let (Some(name), Some(body)) = (function_name, function_body) {
                let start_pos = body.start_position();
                let end_pos = body.end_position();

                // Map line ranges to function names - use line number only for key
                for line in start_pos.row..=end_pos.row {
                    function_map.insert((line as u32 + 1, 0), name.to_string());
                }
            }
        }

        Ok(function_map)
    }

    /// Process a single call match and create a PointerCall
    fn process_call_match(
        call_query: &Query,
        query_match: &QueryMatch,
        source: &str,
        file_path: &str,
        function_map: &HashMap<(u32, u32), String>,
        config: &PointerCallConfig,
    ) -> Result<Option<PointerCall>> {
        let mut function_expr_text = None;
        let mut call_site_node = None;
        let mut pointer_name = None;
        let mut field_name = None;
        let mut array_name = None;

        // Extract captures from the match
        for capture in query_match.captures {
            let capture_name = &call_query.capture_names()[capture.index as usize];
            let text = capture.node.utf8_text(source.as_bytes())?;

            match capture_name.as_str() {
                "function-expr" => {
                    function_expr_text = Some(text.to_string());
                },
                "call-site" => {
                    call_site_node = Some(capture.node);
                },
                "pointer-name" => {
                    pointer_name = Some(text.to_string());
                },
                "field-name" => {
                    field_name = Some(text.to_string());
                },
                "array-name" => {
                    array_name = Some(text.to_string());
                },
                _ => {},
            }
        }

        let Some(call_node) = call_site_node else {
            return Ok(None);
        };

        let Some(expr_text) = function_expr_text else {
            return Ok(None);
        };

        // Determine call type and confidence based on pattern
        let (call_type, confidence) = Self::classify_pointer_call(
            &expr_text,
            pointer_name.as_deref(),
            field_name.as_deref(),
            array_name.as_deref(),
        );

        // Skip if this call type is disabled in config
        if !Self::should_include_call_type(&call_type, config) {
            return Ok(None);
        }

        // Validate identifier if enabled
        if config.validate_identifiers && !Self::is_valid_identifier(&expr_text) {
            return Ok(None);
        }

        let start_pos = call_node.start_position();
        let line_number = start_pos.row as u32 + 1;
        let column_number = start_pos.column as u32;

        // Find the containing function
        let caller_function = function_map.get(&(line_number, 0)).cloned();

        Ok(Some(PointerCall {
            pointer_expression: expr_text,
            file_path: file_path.to_string(),
            line_number,
            column_number,
            caller_function,
            is_conditional: false, // TODO: Implement conditional detection
            call_type,
            confidence,
        }))
    }

    /// Classify the type of pointer call and assign confidence
    fn classify_pointer_call(
        expr_text: &str,
        pointer_name: Option<&str>,
        field_name: Option<&str>,
        array_name: Option<&str>,
    ) -> (PointerCallType, ConfidenceLevel) {
        // Explicit dereference pattern: (*ptr)
        if expr_text.starts_with('(') && expr_text.contains("*") && pointer_name.is_some() {
            return (PointerCallType::ExplicitDereference, ConfidenceLevel::High);
        }

        // Array callback pattern: array[index]
        if array_name.is_some() && expr_text.contains('[') {
            return (PointerCallType::ArrayCallback, ConfidenceLevel::High);
        }

        // Member pointer pattern: obj->member or obj.member
        if field_name.is_some() && (expr_text.contains("->") || expr_text.contains('.')) {
            return (PointerCallType::MemberPointer, ConfidenceLevel::Medium);
        }

        // Default to implicit call with low confidence
        (PointerCallType::Implicit, ConfidenceLevel::Low)
    }

    /// Check if a call type should be included based on configuration
    fn should_include_call_type(call_type: &PointerCallType, config: &PointerCallConfig) -> bool {
        match call_type {
            PointerCallType::ArrayCallback => config.include_callback_arrays,
            PointerCallType::MemberPointer => config.include_member_pointers,
            PointerCallType::ExplicitDereference | PointerCallType::Implicit => true,
        }
    }

    /// Validate that a string is a valid C identifier
    fn is_valid_identifier(text: &str) -> bool {
        if text.is_empty() {
            return false;
        }

        // Allow expressions that contain valid identifier patterns
        // This is more permissive than strict identifier validation
        text.chars().any(|c| c.is_ascii_alphanumeric() || c == '_')
    }

    /// Get extraction statistics
    pub fn get_stats(&self, calls: &[PointerCall]) -> ExtractionStats {
        let mut stats = ExtractionStats {
            total_calls: calls.len(),
            ..Default::default()
        };

        for call in calls {
            match call.call_type {
                PointerCallType::ExplicitDereference => stats.explicit_dereference_calls += 1,
                PointerCallType::Implicit => stats.implicit_calls += 1,
                PointerCallType::MemberPointer => stats.member_pointer_calls += 1,
                PointerCallType::ArrayCallback => stats.array_callback_calls += 1,
            }

            match call.confidence {
                ConfidenceLevel::High => stats.high_confidence += 1,
                ConfidenceLevel::Medium => stats.medium_confidence += 1,
                ConfidenceLevel::Low => stats.low_confidence += 1,
            }

            if call.is_kernel_callback() {
                stats.kernel_callbacks += 1;
            }
        }

        stats
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tree_sitter_c::language;

    fn create_test_detector() -> PointerCallDetector {
        PointerCallDetector::new(language(), PointerCallConfig::default()).unwrap()
    }

    #[test]
    fn test_explicit_pointer_dereference() {
        let detector = create_test_detector();
        let source = r#"
            void test_function() {
                int (*func_ptr)(int);
                int result = (*func_ptr)(42);
            }
        "#;

        let mut parser = tree_sitter::Parser::new();
        parser.set_language(language()).unwrap();
        let tree = parser.parse(source, None).unwrap();

        let calls = detector.extract_calls(&tree, source, "test.c").unwrap();
        assert_eq!(calls.len(), 1);

        let call = &calls[0];
        assert_eq!(call.call_type, PointerCallType::ExplicitDereference);
        assert_eq!(call.confidence, ConfidenceLevel::High);
        assert_eq!(call.caller_function, Some("test_function".to_string()));
    }

    #[test]
    fn test_array_callback() {
        let detector = create_test_detector();
        let source = r#"
            void test_function() {
                int (*callbacks[10])(int);
                int result = callbacks[5](42);
            }
        "#;

        let mut parser = tree_sitter::Parser::new();
        parser.set_language(language()).unwrap();
        let tree = parser.parse(source, None).unwrap();

        let calls = detector.extract_calls(&tree, source, "test.c").unwrap();
        assert_eq!(calls.len(), 1);

        let call = &calls[0];
        assert_eq!(call.call_type, PointerCallType::ArrayCallback);
        assert_eq!(call.confidence, ConfidenceLevel::High);
        assert!(call.pointer_expression.contains("callbacks[5]"));
    }

    #[test]
    fn test_member_pointer() {
        let detector = create_test_detector();
        let source = r#"
            struct ops {
                int (*read)(void);
                int (*write)(void);
            };

            void test_function() {
                struct ops *operations;
                int result = operations->read();
            }
        "#;

        let mut parser = tree_sitter::Parser::new();
        parser.set_language(language()).unwrap();
        let tree = parser.parse(source, None).unwrap();

        let calls = detector.extract_calls(&tree, source, "test.c").unwrap();
        assert_eq!(calls.len(), 1);

        let call = &calls[0];
        assert_eq!(call.call_type, PointerCallType::MemberPointer);
        assert_eq!(call.confidence, ConfidenceLevel::Medium);
        assert!(call.pointer_expression.contains("operations->read"));
    }

    #[test]
    fn test_kernel_callback_detection() {
        let call = PointerCall {
            pointer_expression: "file_ops->read".to_string(),
            file_path: "test.c".to_string(),
            line_number: 10,
            column_number: 5,
            caller_function: Some("sys_read".to_string()),
            is_conditional: false,
            call_type: PointerCallType::MemberPointer,
            confidence: ConfidenceLevel::Medium,
        };

        assert!(call.is_kernel_callback());
    }

    #[test]
    fn test_call_edge_conversion() {
        let call = PointerCall {
            pointer_expression: "(*func_ptr)".to_string(),
            file_path: "test.c".to_string(),
            line_number: 42,
            column_number: 10,
            caller_function: Some("caller_func".to_string()),
            is_conditional: false,
            call_type: PointerCallType::ExplicitDereference,
            confidence: ConfidenceLevel::High,
        };

        let edge = call.to_call_edge();
        assert_eq!(edge.caller_name(), "caller_func");
        assert_eq!(edge.callee_name(), "(*func_ptr)");
        assert_eq!(edge.call_type(), CallTypeEnum::Indirect);
        assert_eq!(edge.confidence(), ConfidenceLevel::High);
        assert_eq!(edge.file_path(), "test.c");
        assert_eq!(edge.line_number(), 42);
    }

    #[test]
    fn test_extraction_stats() {
        let calls = vec![
            PointerCall {
                pointer_expression: "(*func1)".to_string(),
                file_path: "test.c".to_string(),
                line_number: 1,
                column_number: 1,
                caller_function: None,
                is_conditional: false,
                call_type: PointerCallType::ExplicitDereference,
                confidence: ConfidenceLevel::High,
            },
            PointerCall {
                pointer_expression: "callbacks[0]".to_string(),
                file_path: "test.c".to_string(),
                line_number: 2,
                column_number: 1,
                caller_function: None,
                is_conditional: false,
                call_type: PointerCallType::ArrayCallback,
                confidence: ConfidenceLevel::High,
            },
        ];

        let detector = create_test_detector();
        let stats = detector.get_stats(&calls);

        assert_eq!(stats.total_calls, 2);
        assert_eq!(stats.explicit_dereference_calls, 1);
        assert_eq!(stats.array_callback_calls, 1);
        assert_eq!(stats.high_confidence, 2);
    }
}
