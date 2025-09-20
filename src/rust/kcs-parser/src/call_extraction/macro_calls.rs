//! Macro call expansion detection using Tree-sitter AST analysis.
//!
//! This module implements detection of macro expansions that result in function calls
//! in C code. Kernel code heavily uses macros that expand to function calls, making
//! this a critical component for complete call graph analysis.

use anyhow::{Context, Result};
use kcs_graph::{CallEdgeBuilder, CallEdgeModel, CallTypeEnum, ConfidenceLevel};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use tree_sitter::{Language, Query, QueryCursor, QueryMatch, Tree};

/// Configuration for macro call detection
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct MacroCallConfig {
    /// Include calls within conditional compilation blocks
    pub include_conditional: bool,
    /// Include calls within if/switch/loop statements
    pub include_control_flow: bool,
    /// Maximum call depth to analyze
    pub max_depth: usize,
    /// Whether to validate macro name syntax
    pub validate_identifiers: bool,
    /// Include kernel-specific macro patterns
    pub include_kernel_macros: bool,
    /// Include custom user-defined macros
    pub include_user_macros: bool,
}

impl Default for MacroCallConfig {
    fn default() -> Self {
        Self {
            include_conditional: true,
            include_control_flow: true,
            max_depth: 100,
            validate_identifiers: true,
            include_kernel_macros: true,
            include_user_macros: true,
        }
    }
}

/// Macro call detection result
#[derive(Debug, Clone)]
pub struct MacroCall {
    /// Name of the macro being called
    pub macro_name: String,
    /// File path where the macro call occurs
    pub file_path: String,
    /// Line number of the macro call
    pub line_number: u32,
    /// Column number of the macro call
    pub column_number: u32,
    /// Name of the calling function context
    pub caller_function: Option<String>,
    /// Whether the call is within conditional compilation
    pub is_conditional: bool,
    /// Type of macro call
    pub call_type: MacroCallType,
    /// Confidence level in the detection
    pub confidence: ConfidenceLevel,
    /// Arguments passed to the macro
    pub arguments: Vec<String>,
    /// Potential function names that this macro might expand to
    pub potential_expansions: Vec<String>,
}

/// Types of macro calls
#[derive(Debug, Clone, PartialEq)]
pub enum MacroCallType {
    /// Function-like macro: MACRO(args)
    FunctionLike,
    /// Object-like macro: MACRO
    ObjectLike,
    /// Kernel API macro (e.g., EXPORT_SYMBOL, MODULE_INIT)
    KernelApi,
    /// Debug/logging macro (e.g., printk, pr_debug)
    Debug,
    /// Memory management macro (e.g., kmalloc, kfree wrappers)
    Memory,
    /// Locking macro (e.g., spin_lock, mutex_lock wrappers)
    Locking,
}

impl MacroCall {
    /// Convert to a CallEdgeModel for database storage
    pub fn to_call_edge(&self) -> CallEdgeModel {
        CallEdgeBuilder::default()
            .caller(self.caller_function.clone().unwrap_or_else(|| "<unknown>".to_string()))
            .callee(self.macro_name.clone())
            .call_type(CallTypeEnum::Macro)
            .confidence(self.confidence)
            .file_path(self.file_path.clone())
            .line_number(self.line_number)
            .build()
    }

    /// Check if this is a kernel-specific macro
    pub fn is_kernel_macro(&self) -> bool {
        // Common kernel macro patterns
        let kernel_patterns = [
            "EXPORT_SYMBOL",
            "MODULE_",
            "DEVICE_ATTR",
            "DRIVER_ATTR",
            "CLASS_ATTR",
            "BUS_ATTR",
            "KERN_",
            "pr_",
            "dev_",
            "netdev_",
            "printk",
            "__init",
            "__exit",
            "__user",
            "__kernel",
            "__iomem",
            "__force",
            "DECLARE_",
            "DEFINE_",
            "INIT_",
            "SETUP_",
            "early_param",
            "module_param",
            "core_param",
            "device_initcall",
            "subsys_initcall",
            "fs_initcall",
            "late_initcall",
            "ACPI_",
            "PCI_",
            "USB_",
            "DMA_",
        ];

        kernel_patterns.iter().any(|&pattern| self.macro_name.starts_with(pattern))
    }

    /// Check if this is a debug/logging macro
    pub fn is_debug_macro(&self) -> bool {
        let debug_patterns = [
            "printk",
            "pr_debug",
            "pr_info",
            "pr_warn",
            "pr_err",
            "pr_crit",
            "pr_alert",
            "pr_emerg",
            "dev_dbg",
            "dev_info",
            "dev_warn",
            "dev_err",
            "netdev_dbg",
            "netdev_info",
            "netdev_warn",
            "netdev_err",
            "WARN_ON",
            "BUG_ON",
            "ASSERT",
        ];

        debug_patterns.iter().any(|&pattern| self.macro_name.starts_with(pattern))
    }

    /// Get the potential target function names for this macro call
    pub fn get_potential_targets(&self) -> Vec<String> {
        match self.call_type {
            MacroCallType::Debug => {
                // Debug macros often expand to printk or similar
                if self.macro_name.starts_with("pr_") {
                    vec!["printk".to_string()]
                } else if self.macro_name.starts_with("dev_") {
                    vec!["dev_printk".to_string()]
                } else {
                    self.potential_expansions.clone()
                }
            },
            MacroCallType::Memory => {
                // Memory macros often expand to kmalloc/kfree family
                vec!["__kmalloc".to_string(), "__kfree".to_string()]
            },
            MacroCallType::Locking => {
                // Locking macros expand to various locking primitives
                vec![
                    "_raw_spin_lock".to_string(),
                    "_raw_spin_unlock".to_string(),
                    "mutex_lock".to_string(),
                    "mutex_unlock".to_string(),
                ]
            },
            _ => self.potential_expansions.clone(),
        }
    }
}

/// Extraction statistics
#[derive(Debug, Clone, Default)]
pub struct ExtractionStats {
    pub total_calls: usize,
    pub function_like_calls: usize,
    pub object_like_calls: usize,
    pub kernel_api_calls: usize,
    pub debug_calls: usize,
    pub memory_calls: usize,
    pub locking_calls: usize,
    pub kernel_macros: usize,
    pub user_macros: usize,
    pub high_confidence: usize,
    pub medium_confidence: usize,
    pub low_confidence: usize,
}

/// Macro call detector
pub struct MacroCallDetector {
    #[allow(dead_code)]
    language: Language,
    call_query: Query,
    function_query: Query,
    config: MacroCallConfig,
    #[allow(dead_code)]
    cursor: QueryCursor,
}

impl MacroCallDetector {
    /// Create a new macro call detector
    pub fn new(language: Language, config: MacroCallConfig) -> Result<Self> {
        // Query for macro calls - simplified to focus on call expressions
        let call_query_source = r#"
            ; Function-like macro calls: MACRO(args)
            (call_expression
              function: (identifier) @macro-name
              arguments: (argument_list) @args) @call-site
        "#;

        let call_query =
            Query::new(language, call_query_source).context("Failed to create macro call query")?;

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

    /// Extract macro calls from source tree
    pub fn extract_calls(
        &self,
        tree: &Tree,
        source: &str,
        file_path: &str,
    ) -> Result<Vec<MacroCall>> {
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

    /// Process a single call match and create a MacroCall
    fn process_call_match(
        call_query: &Query,
        query_match: &QueryMatch,
        source: &str,
        file_path: &str,
        function_map: &HashMap<(u32, u32), String>,
        config: &MacroCallConfig,
    ) -> Result<Option<MacroCall>> {
        let mut macro_name = None;
        let mut call_site_node = None;
        let mut arguments = Vec::new();

        // Extract captures from the match
        for capture in query_match.captures {
            let capture_name = &call_query.capture_names()[capture.index as usize];
            let text = capture.node.utf8_text(source.as_bytes())?;

            match capture_name.as_str() {
                "macro-name" => {
                    macro_name = Some(text.to_string());
                },
                "call-site" => {
                    call_site_node = Some(capture.node);
                },
                "args" => {
                    // Parse arguments - simplified for now
                    arguments.push(text.to_string());
                },
                _ => {},
            }
        }

        let Some(call_node) = call_site_node else {
            return Ok(None);
        };

        let Some(name) = macro_name else {
            return Ok(None);
        };

        // Filter out obvious function calls (this is heuristic-based)
        if Self::is_likely_function_call(&name) {
            return Ok(None);
        }

        // Classify macro type and determine confidence
        let (call_type, confidence) = Self::classify_macro_call(&name, &arguments);

        // Skip if this call type is disabled in config
        if !Self::should_include_call_type(&call_type, config) {
            return Ok(None);
        }

        // Validate identifier if enabled
        if config.validate_identifiers && !Self::is_valid_identifier(&name) {
            return Ok(None);
        }

        let start_pos = call_node.start_position();
        let line_number = start_pos.row as u32 + 1;
        let column_number = start_pos.column as u32;

        // Find the containing function
        let caller_function = function_map.get(&(line_number, 0)).cloned();

        // Get potential expansions
        let potential_expansions = Self::get_macro_expansions(&name, &call_type);

        Ok(Some(MacroCall {
            macro_name: name,
            file_path: file_path.to_string(),
            line_number,
            column_number,
            caller_function,
            is_conditional: false, // TODO: Implement conditional detection
            call_type,
            confidence,
            arguments,
            potential_expansions,
        }))
    }

    /// Check if a name is likely a regular function call rather than a macro
    fn is_likely_function_call(name: &str) -> bool {
        // Be more conservative - only filter out very obvious function calls
        // Most macros contain uppercase letters or special patterns
        name.chars().all(|c| c.is_ascii_lowercase() || c == '_') && // all lowercase
        !name.starts_with("__") && // not system prefix
        !name.contains("alloc") && // could be memory macro
        !name.contains("lock") && // could be locking macro
        !name.starts_with("pr_") && // kernel debug macros
        !name.starts_with("dev_") && // device debug macros
        name != "printk" && // kernel logging function/macro
        name.len() > 3 // reasonable length
    }

    /// Classify the type of macro call and assign confidence
    fn classify_macro_call(name: &str, arguments: &[String]) -> (MacroCallType, ConfidenceLevel) {
        // Check for kernel API macros
        if name.starts_with("EXPORT_SYMBOL")
            || name.starts_with("MODULE_")
            || name.starts_with("DEVICE_ATTR")
            || name.starts_with("DECLARE_")
            || name.starts_with("DEFINE_")
        {
            return (MacroCallType::KernelApi, ConfidenceLevel::High);
        }

        // Check for debug macros
        if name.starts_with("pr_")
            || name.starts_with("dev_")
            || name.starts_with("netdev_")
            || name == "printk"
            || name.starts_with("WARN_")
            || name.starts_with("BUG_")
        {
            return (MacroCallType::Debug, ConfidenceLevel::High);
        }

        // Check for memory macros
        if name.contains("alloc") || name.contains("free") || name.contains("mem") {
            return (MacroCallType::Memory, ConfidenceLevel::Medium);
        }

        // Check for locking macros
        if name.contains("lock") || name.contains("mutex") || name.contains("spin") {
            return (MacroCallType::Locking, ConfidenceLevel::Medium);
        }

        // Function-like vs object-like based on arguments
        if !arguments.is_empty() {
            (MacroCallType::FunctionLike, ConfidenceLevel::Medium)
        } else {
            (MacroCallType::ObjectLike, ConfidenceLevel::Low)
        }
    }

    /// Check if a call type should be included based on configuration
    fn should_include_call_type(call_type: &MacroCallType, config: &MacroCallConfig) -> bool {
        match call_type {
            MacroCallType::KernelApi
            | MacroCallType::Debug
            | MacroCallType::Memory
            | MacroCallType::Locking => config.include_kernel_macros,
            MacroCallType::FunctionLike | MacroCallType::ObjectLike => config.include_user_macros,
        }
    }

    /// Validate that a string is a valid C identifier
    fn is_valid_identifier(text: &str) -> bool {
        if text.is_empty() {
            return false;
        }

        // First character must be letter or underscore
        let mut chars = text.chars();
        if let Some(first) = chars.next() {
            if !(first.is_ascii_alphabetic() || first == '_') {
                return false;
            }
        }

        // Remaining characters must be alphanumeric or underscore
        chars.all(|c| c.is_ascii_alphanumeric() || c == '_')
    }

    /// Get potential macro expansions based on name and type
    fn get_macro_expansions(name: &str, call_type: &MacroCallType) -> Vec<String> {
        match call_type {
            MacroCallType::Debug => {
                if name.starts_with("pr_") {
                    vec!["printk".to_string()]
                } else if name.starts_with("dev_") {
                    vec!["dev_printk".to_string()]
                } else {
                    vec![]
                }
            },
            MacroCallType::Memory => {
                vec!["__kmalloc".to_string(), "__kfree".to_string(), "vmalloc".to_string()]
            },
            MacroCallType::Locking => {
                vec!["_raw_spin_lock".to_string(), "mutex_lock".to_string()]
            },
            _ => vec![],
        }
    }

    /// Get extraction statistics
    pub fn get_stats(&self, calls: &[MacroCall]) -> ExtractionStats {
        let mut stats = ExtractionStats {
            total_calls: calls.len(),
            ..Default::default()
        };

        for call in calls {
            match call.call_type {
                MacroCallType::FunctionLike => stats.function_like_calls += 1,
                MacroCallType::ObjectLike => stats.object_like_calls += 1,
                MacroCallType::KernelApi => stats.kernel_api_calls += 1,
                MacroCallType::Debug => stats.debug_calls += 1,
                MacroCallType::Memory => stats.memory_calls += 1,
                MacroCallType::Locking => stats.locking_calls += 1,
            }

            match call.confidence {
                ConfidenceLevel::High => stats.high_confidence += 1,
                ConfidenceLevel::Medium => stats.medium_confidence += 1,
                ConfidenceLevel::Low => stats.low_confidence += 1,
            }

            if call.is_kernel_macro() {
                stats.kernel_macros += 1;
            } else {
                stats.user_macros += 1;
            }
        }

        stats
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tree_sitter_c::language;

    fn create_test_detector() -> MacroCallDetector {
        MacroCallDetector::new(language(), MacroCallConfig::default()).unwrap()
    }

    #[test]
    fn test_export_symbol_macro() {
        let detector = create_test_detector();
        let source = "EXPORT_SYMBOL(my_function);";

        let mut parser = tree_sitter::Parser::new();
        parser.set_language(language()).unwrap();
        let tree = parser.parse(source, None).unwrap();

        let calls = detector.extract_calls(&tree, source, "test.c").unwrap();
        assert_eq!(calls.len(), 1);

        let call = &calls[0];
        assert_eq!(call.macro_name, "EXPORT_SYMBOL");
        assert_eq!(call.call_type, MacroCallType::KernelApi);
        assert_eq!(call.confidence, ConfidenceLevel::High);
        assert!(call.is_kernel_macro());
    }

    #[test]
    fn test_module_author_macro() {
        let detector = create_test_detector();
        let source = r#"MODULE_AUTHOR("Test Author");"#;

        let mut parser = tree_sitter::Parser::new();
        parser.set_language(language()).unwrap();
        let tree = parser.parse(source, None).unwrap();

        let calls = detector.extract_calls(&tree, source, "test.c").unwrap();
        assert_eq!(calls.len(), 1);

        let call = &calls[0];
        assert_eq!(call.macro_name, "MODULE_AUTHOR");
        assert_eq!(call.call_type, MacroCallType::KernelApi);
        assert_eq!(call.confidence, ConfidenceLevel::High);
        assert!(call.is_kernel_macro());
    }

    #[test]
    fn test_kernel_api_macro() {
        let detector = create_test_detector();
        let source = r#"
            void test_function() {
                MODULE_AUTHOR("Test Author");
            }
        "#;

        let mut parser = tree_sitter::Parser::new();
        parser.set_language(language()).unwrap();
        let tree = parser.parse(source, None).unwrap();

        let calls = detector.extract_calls(&tree, source, "test.c").unwrap();

        // Should find MODULE_AUTHOR call
        let module_calls: Vec<_> =
            calls.iter().filter(|c| c.macro_name.starts_with("MODULE_")).collect();

        assert!(!module_calls.is_empty(), "MODULE_AUTHOR call not found");

        if let Some(call) = module_calls.first() {
            assert_eq!(call.call_type, MacroCallType::KernelApi);
            assert_eq!(call.confidence, ConfidenceLevel::High);
            assert_eq!(call.caller_function, Some("test_function".to_string()));
            assert!(call.is_kernel_macro());
        }
    }

    #[test]
    fn test_debug_macro() {
        let detector = create_test_detector();
        let source = r#"
            void test_function() {
                pr_info("Test message %d", value);
                printk(KERN_DEBUG "Debug message");
            }
        "#;

        let mut parser = tree_sitter::Parser::new();
        parser.set_language(language()).unwrap();
        let tree = parser.parse(source, None).unwrap();

        let calls = detector.extract_calls(&tree, source, "test.c").unwrap();

        let debug_calls: Vec<_> = calls.iter().filter(|c| c.is_debug_macro()).collect();

        assert!(!debug_calls.is_empty());

        for call in debug_calls {
            assert_eq!(call.call_type, MacroCallType::Debug);
            assert_eq!(call.confidence, ConfidenceLevel::High);
            assert!(call.is_kernel_macro());
        }
    }

    #[test]
    fn test_function_like_macro() {
        let detector = create_test_detector();
        let source = r#"
            #define MAX(a, b) ((a) > (b) ? (a) : (b))

            void test_function() {
                int result = MAX(x, y);
            }
        "#;

        let mut parser = tree_sitter::Parser::new();
        parser.set_language(language()).unwrap();
        let tree = parser.parse(source, None).unwrap();

        let calls = detector.extract_calls(&tree, source, "test.c").unwrap();

        let max_calls: Vec<_> = calls.iter().filter(|c| c.macro_name == "MAX").collect();

        assert!(!max_calls.is_empty());

        if let Some(call) = max_calls.first() {
            assert_eq!(call.call_type, MacroCallType::FunctionLike);
            assert!(!call.arguments.is_empty());
        }
    }

    #[test]
    fn test_kernel_macro_detection() {
        let call = MacroCall {
            macro_name: "EXPORT_SYMBOL_GPL".to_string(),
            file_path: "test.c".to_string(),
            line_number: 10,
            column_number: 5,
            caller_function: Some("init_module".to_string()),
            is_conditional: false,
            call_type: MacroCallType::KernelApi,
            confidence: ConfidenceLevel::High,
            arguments: vec!["my_function".to_string()],
            potential_expansions: vec![],
        };

        assert!(call.is_kernel_macro());
        assert!(!call.is_debug_macro());
    }

    #[test]
    fn test_call_edge_conversion() {
        let call = MacroCall {
            macro_name: "pr_info".to_string(),
            file_path: "test.c".to_string(),
            line_number: 42,
            column_number: 10,
            caller_function: Some("caller_func".to_string()),
            is_conditional: false,
            call_type: MacroCallType::Debug,
            confidence: ConfidenceLevel::High,
            arguments: vec!["\"message\"".to_string()],
            potential_expansions: vec!["printk".to_string()],
        };

        let edge = call.to_call_edge();
        assert_eq!(edge.caller_name(), "caller_func");
        assert_eq!(edge.callee_name(), "pr_info");
        assert_eq!(edge.call_type(), CallTypeEnum::Macro);
        assert_eq!(edge.confidence(), ConfidenceLevel::High);
        assert_eq!(edge.file_path(), "test.c");
        assert_eq!(edge.line_number(), 42);
    }

    #[test]
    fn test_extraction_stats() {
        let calls = vec![
            MacroCall {
                macro_name: "EXPORT_SYMBOL".to_string(),
                file_path: "test.c".to_string(),
                line_number: 1,
                column_number: 1,
                caller_function: None,
                is_conditional: false,
                call_type: MacroCallType::KernelApi,
                confidence: ConfidenceLevel::High,
                arguments: vec![],
                potential_expansions: vec![],
            },
            MacroCall {
                macro_name: "pr_info".to_string(),
                file_path: "test.c".to_string(),
                line_number: 2,
                column_number: 1,
                caller_function: None,
                is_conditional: false,
                call_type: MacroCallType::Debug,
                confidence: ConfidenceLevel::High,
                arguments: vec![],
                potential_expansions: vec![],
            },
        ];

        let detector = create_test_detector();
        let stats = detector.get_stats(&calls);

        assert_eq!(stats.total_calls, 2);
        assert_eq!(stats.kernel_api_calls, 1);
        assert_eq!(stats.debug_calls, 1);
        assert_eq!(stats.high_confidence, 2);
        assert_eq!(stats.kernel_macros, 2);
    }

    #[test]
    fn test_macro_classification() {
        let (call_type, confidence) = MacroCallDetector::classify_macro_call("EXPORT_SYMBOL", &[]);
        assert_eq!(call_type, MacroCallType::KernelApi);
        assert_eq!(confidence, ConfidenceLevel::High);

        let (call_type, confidence) = MacroCallDetector::classify_macro_call("pr_debug", &[]);
        assert_eq!(call_type, MacroCallType::Debug);
        assert_eq!(confidence, ConfidenceLevel::High);

        let (call_type, confidence) =
            MacroCallDetector::classify_macro_call("MAX", &["a".to_string(), "b".to_string()]);
        assert_eq!(call_type, MacroCallType::FunctionLike);
        assert_eq!(confidence, ConfidenceLevel::Medium);
    }
}
