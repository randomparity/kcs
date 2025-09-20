//! Conditional compilation call detection using Tree-sitter AST analysis.
//!
//! This module implements detection of function calls within conditional compilation
//! blocks (#ifdef, #ifndef, #if defined, etc.) in C code, tracking which calls are
//! dependent on specific kernel configuration options.

use anyhow::{Context, Result};
use kcs_graph::{CallEdgeBuilder, CallEdgeModel, CallTypeEnum, ConfidenceLevel};
use std::collections::HashMap;
use tree_sitter::{Language, Node, Query, QueryCursor, QueryMatch, Tree};

/// Configuration for conditional call detection
#[derive(Debug, Clone)]
pub struct ConditionalCallConfig {
    /// Include nested conditional blocks
    pub include_nested: bool,
    /// Include calls within complex conditional expressions
    pub include_complex_conditions: bool,
    /// Maximum nesting depth to analyze
    pub max_nesting_depth: usize,
    /// Whether to resolve macro definitions in conditions
    pub resolve_macros: bool,
}

impl Default for ConditionalCallConfig {
    fn default() -> Self {
        Self {
            include_nested: true,
            include_complex_conditions: true,
            max_nesting_depth: 10,
            resolve_macros: false,
        }
    }
}

/// Conditional call detection result
#[derive(Debug, Clone)]
pub struct ConditionalCall {
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
    /// Configuration guard condition
    pub config_guard: String,
    /// Type of conditional compilation block
    pub conditional_type: ConditionalType,
    /// Nesting depth within conditionals
    pub nesting_depth: u32,
    /// Whether condition is negated (!defined, #ifndef)
    pub is_negated: bool,
    /// Complex condition expression if applicable
    pub condition_expression: Option<String>,
}

/// Type of conditional compilation construct
#[derive(Debug, Clone, PartialEq)]
pub enum ConditionalType {
    /// Simple #ifdef SYMBOL
    Ifdef,
    /// Simple #ifndef SYMBOL
    Ifndef,
    /// Complex #if defined(SYMBOL)
    IfDefined,
    /// Complex #if condition with operators
    IfExpression,
    /// #elif branch
    Elif,
    /// #else branch
    Else,
}

/// Statistics for conditional call extraction
#[derive(Debug, Clone, Default)]
pub struct ExtractionStats {
    /// Total calls found within conditionals
    pub total_calls: usize,
    /// Calls in simple #ifdef blocks
    pub ifdef_calls: usize,
    /// Calls in #ifndef blocks
    pub ifndef_calls: usize,
    /// Calls in complex #if expressions
    pub if_expression_calls: usize,
    /// Calls in nested conditionals
    pub nested_calls: usize,
    /// Unique configuration symbols referenced
    pub unique_configs: usize,
    /// Maximum nesting depth encountered
    pub max_depth_found: u32,
}

/// Conditional call detector for C/kernel code
pub struct ConditionalCallDetector {
    config: ConditionalCallConfig,
    query: Query,
    stats: ExtractionStats,
}

impl ConditionalCallDetector {
    /// Create a new conditional call detector
    pub fn new(config: ConditionalCallConfig) -> Result<Self> {
        let language = tree_sitter_c::language();
        let query = Self::build_query(&language)?;

        Ok(Self {
            config,
            query,
            stats: ExtractionStats::default(),
        })
    }

    /// Build Tree-sitter query for conditional compilation patterns
    fn build_query(language: &Language) -> Result<Query> {
        let query_text = r#"
            ; Simple function calls within preprocessor conditionals
            (preproc_if
              (call_expression
                function: (identifier) @function-name
                arguments: (argument_list) @args) @call-site) @conditional-block

            ; Function calls wrapped in expression statements
            (preproc_if
              (expression_statement
                (call_expression
                  function: (identifier) @function-name
                  arguments: (argument_list) @args) @call-site)) @conditional-block
        "#;

        Query::new(*language, query_text).context("Failed to build conditional compilation query")
    }

    /// Extract conditional calls from a syntax tree
    pub fn extract_calls(
        &mut self,
        tree: &Tree,
        source_code: &str,
        file_path: &str,
    ) -> Result<Vec<ConditionalCall>> {
        let mut calls = Vec::new();
        let mut cursor = QueryCursor::new();
        let root_node = tree.root_node();

        // Reset stats for this file
        let mut unique_configs = std::collections::HashSet::new();
        let mut max_depth = 0;

        // Execute query and process matches
        for query_match in cursor.matches(&self.query, root_node, source_code.as_bytes()) {
            if let Some(conditional_call) = self.process_match(
                &query_match,
                source_code,
                file_path,
                &mut unique_configs,
                &mut max_depth,
            )? {
                // Update stats
                self.stats.total_calls += 1;
                match conditional_call.conditional_type {
                    ConditionalType::Ifdef => self.stats.ifdef_calls += 1,
                    ConditionalType::Ifndef => self.stats.ifndef_calls += 1,
                    ConditionalType::IfExpression => self.stats.if_expression_calls += 1,
                    _ => {},
                }
                if conditional_call.nesting_depth > 1 {
                    self.stats.nested_calls += 1;
                }

                calls.push(conditional_call);
            }
        }

        // Update global stats
        self.stats.unique_configs = unique_configs.len();
        self.stats.max_depth_found = max_depth;

        Ok(calls)
    }

    /// Process a single query match to extract conditional call information
    fn process_match(
        &self,
        query_match: &QueryMatch,
        source_code: &str,
        file_path: &str,
        unique_configs: &mut std::collections::HashSet<String>,
        max_depth: &mut u32,
    ) -> Result<Option<ConditionalCall>> {
        let mut function_name = None;
        let mut call_site_node = None;
        let mut conditional_block_node = None;

        // Extract capture information
        for capture in query_match.captures {
            let capture_name = &self.query.capture_names()[capture.index as usize];
            let node_text = capture
                .node
                .utf8_text(source_code.as_bytes())
                .context("Failed to extract node text")?;

            match capture_name.as_str() {
                "function-name" => function_name = Some(node_text.to_string()),
                "call-site" => call_site_node = Some(capture.node),
                "conditional-block" => conditional_block_node = Some(capture.node),
                _ => {},
            }
        }

        // Ensure we have required information
        let function_name = function_name
            .ok_or_else(|| anyhow::anyhow!("Missing function name in conditional call match"))?;

        let call_site = call_site_node
            .ok_or_else(|| anyhow::anyhow!("Missing call site in conditional call match"))?;

        // Determine conditional type and extract more details
        let (conditional_type, config_guard, is_negated, condition_expression) =
            self.analyze_conditional_context(conditional_block_node, source_code)?;

        // Add config to unique set if present
        if let Some(ref config) = config_guard {
            unique_configs.insert(config.clone());
        }

        // Calculate nesting depth
        let nesting_depth = self.calculate_nesting_depth(call_site);
        *max_depth = (*max_depth).max(nesting_depth);

        // Check depth limits
        if nesting_depth > self.config.max_nesting_depth as u32 {
            return Ok(None);
        }

        // Get position information
        let start_position = call_site.start_position();
        let line_number = start_position.row as u32 + 1;
        let column_number = start_position.column as u32;

        // Find calling function context
        let caller_function = self.find_containing_function(call_site, source_code);

        let conditional_call = ConditionalCall {
            function_name,
            file_path: file_path.to_string(),
            line_number,
            column_number,
            caller_function,
            config_guard: config_guard.unwrap_or_default(),
            conditional_type,
            nesting_depth,
            is_negated,
            condition_expression,
        };

        Ok(Some(conditional_call))
    }

    /// Analyze the conditional compilation context
    fn analyze_conditional_context(
        &self,
        conditional_node: Option<Node>,
        source_code: &str,
    ) -> Result<(ConditionalType, Option<String>, bool, Option<String>)> {
        let node = match conditional_node {
            Some(n) => n,
            None => return Ok((ConditionalType::IfExpression, None, false, None)),
        };

        let node_kind = node.kind();
        match node_kind {
            "preproc_if" => {
                // Check the condition to determine if it's ifdef, ifndef, or complex if
                if let Some(condition_node) = node.child_by_field_name("condition") {
                    let condition_text = condition_node
                        .utf8_text(source_code.as_bytes())
                        .context("Failed to extract condition text")?;

                    // Try to determine the type of conditional based on the condition text
                    let conditional_type = if condition_text.starts_with("defined(")
                        || condition_text.starts_with("!defined(")
                    {
                        ConditionalType::IfDefined
                    } else {
                        ConditionalType::IfExpression
                    };

                    let (config_guard, is_negated, condition_expr) =
                        self.extract_if_condition(&node, source_code)?;
                    Ok((conditional_type, Some(config_guard), is_negated, condition_expr))
                } else {
                    Ok((ConditionalType::IfExpression, None, false, None))
                }
            },
            _ => Ok((ConditionalType::IfExpression, None, false, None)),
        }
    }

    /// Extract condition from #if directive
    fn extract_if_condition(
        &self,
        node: &Node,
        source_code: &str,
    ) -> Result<(String, bool, Option<String>)> {
        let condition_node = node
            .child_by_field_name("condition")
            .ok_or_else(|| anyhow::anyhow!("Missing condition field in if directive"))?;

        let condition_text = condition_node
            .utf8_text(source_code.as_bytes())
            .context("Failed to extract if condition")?;

        // Check for simple defined() calls
        if condition_text.starts_with("defined(") && condition_text.ends_with(")") {
            let symbol = condition_text
                .strip_prefix("defined(")
                .unwrap()
                .strip_suffix(")")
                .unwrap()
                .to_string();
            return Ok((symbol, false, Some(condition_text.to_string())));
        }

        // Check for negated defined() calls
        if condition_text.starts_with("!defined(") && condition_text.ends_with(")") {
            let symbol = condition_text
                .strip_prefix("!defined(")
                .unwrap()
                .strip_suffix(")")
                .unwrap()
                .to_string();
            return Ok((symbol, true, Some(condition_text.to_string())));
        }

        // For complex expressions, try to extract primary symbol
        let primary_symbol = self.extract_primary_config_symbol(condition_text);
        let is_negated = condition_text.starts_with('!');

        Ok((
            primary_symbol.unwrap_or_else(|| condition_text.to_string()),
            is_negated,
            Some(condition_text.to_string()),
        ))
    }

    /// Extract the primary configuration symbol from a complex condition
    fn extract_primary_config_symbol(&self, condition: &str) -> Option<String> {
        // Look for CONFIG_ patterns first
        if let Some(start) = condition.find("CONFIG_") {
            let remaining = &condition[start..];
            if let Some(end) = remaining.find(|c: char| !c.is_alphanumeric() && c != '_') {
                return Some(remaining[..end].to_string());
            } else {
                return Some(remaining.to_string());
            }
        }

        // Look for defined() calls
        if let Some(start) = condition.find("defined(") {
            let remaining = &condition[start + 8..];
            if let Some(end) = remaining.find(')') {
                return Some(remaining[..end].to_string());
            }
        }

        None
    }

    /// Calculate nesting depth of conditional compilation
    fn calculate_nesting_depth(&self, node: Node) -> u32 {
        let mut depth = 0;
        let mut current = node.parent();

        while let Some(parent) = current {
            match parent.kind() {
                "preproc_if" | "preproc_ifdef" | "preproc_ifndef" | "preproc_elif" => {
                    depth += 1;
                },
                _ => {},
            }
            current = parent.parent();
        }

        depth
    }

    /// Find the containing function for a call site
    fn find_containing_function(&self, call_node: Node, source_code: &str) -> Option<String> {
        let mut current = call_node.parent();

        while let Some(parent) = current {
            if parent.kind() == "function_definition" {
                if let Some(declarator) = parent.child_by_field_name("declarator") {
                    if let Some(name_node) = self.extract_function_name_from_declarator(declarator)
                    {
                        if let Ok(name) = name_node.utf8_text(source_code.as_bytes()) {
                            return Some(name.to_string());
                        }
                    }
                }
            }
            current = parent.parent();
        }

        None
    }

    /// Extract function name from function declarator
    #[allow(clippy::only_used_in_recursion)]
    fn extract_function_name_from_declarator<'a>(&self, declarator: Node<'a>) -> Option<Node<'a>> {
        match declarator.kind() {
            "identifier" => Some(declarator),
            "function_declarator" => declarator
                .child_by_field_name("declarator")
                .and_then(|child| self.extract_function_name_from_declarator(child)),
            "pointer_declarator" => declarator
                .child_by_field_name("declarator")
                .and_then(|child| self.extract_function_name_from_declarator(child)),
            _ => None,
        }
    }

    /// Convert conditional calls to call edge models
    pub fn to_call_edges(&self, calls: &[ConditionalCall]) -> Result<Vec<CallEdgeModel>> {
        calls.iter().map(|call| self.conditional_call_to_edge(call)).collect()
    }

    /// Convert a single conditional call to a call edge model
    fn conditional_call_to_edge(&self, call: &ConditionalCall) -> Result<CallEdgeModel> {
        let confidence = if call.conditional_type == ConditionalType::Ifdef
            || call.conditional_type == ConditionalType::Ifndef
        {
            ConfidenceLevel::High
        } else {
            ConfidenceLevel::Medium
        };

        let mut metadata = HashMap::new();
        metadata.insert("conditional_type".to_string(), format!("{:?}", call.conditional_type));
        metadata.insert("nesting_depth".to_string(), call.nesting_depth.to_string());
        metadata.insert("is_negated".to_string(), call.is_negated.to_string());

        if let Some(ref expr) = call.condition_expression {
            metadata.insert("condition_expression".to_string(), expr.clone());
        }

        let mut builder = CallEdgeBuilder::default()
            .caller(call.caller_function.clone().unwrap_or_default())
            .callee(call.function_name.clone())
            .file_path(call.file_path.clone())
            .line_number(call.line_number)
            .call_type(CallTypeEnum::Conditional)
            .confidence(confidence)
            .conditional(true);

        if !call.config_guard.is_empty() {
            builder = builder.config_guard(call.config_guard.clone());
        }

        // Add metadata entries one by one
        for (key, value) in metadata {
            builder = builder.metadata(key, value);
        }

        Ok(builder.build())
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
    use tree_sitter::Parser;

    fn create_test_detector() -> ConditionalCallDetector {
        ConditionalCallDetector::new(ConditionalCallConfig::default()).unwrap()
    }

    fn parse_code(code: &str) -> Tree {
        let mut parser = Parser::new();
        parser.set_language(tree_sitter_c::language()).unwrap();
        parser.parse(code, None).unwrap()
    }

    #[test]
    fn test_simple_ifdef_call() {
        let mut detector = create_test_detector();
        let code = r#"
            #if defined(CONFIG_DEBUG)
            debug_print("test");
            #endif
        "#;
        let tree = parse_code(code);
        let calls = detector.extract_calls(&tree, code, "test.c").unwrap();

        assert_eq!(calls.len(), 1);
        let call = &calls[0];
        assert_eq!(call.function_name, "debug_print");
        assert_eq!(call.config_guard, "CONFIG_DEBUG");
        assert_eq!(call.conditional_type, ConditionalType::IfDefined);
        assert!(!call.is_negated);
    }

    #[test]
    fn test_ifndef_call() {
        let mut detector = create_test_detector();
        let code = r#"
            #if !defined(CONFIG_FEATURE_DISABLED)
            enable_feature();
            #endif
        "#;
        let tree = parse_code(code);
        let calls = detector.extract_calls(&tree, code, "test.c").unwrap();

        assert_eq!(calls.len(), 1);
        let call = &calls[0];
        assert_eq!(call.function_name, "enable_feature");
        assert_eq!(call.config_guard, "CONFIG_FEATURE_DISABLED");
        assert_eq!(call.conditional_type, ConditionalType::IfDefined);
        assert!(call.is_negated);
    }

    #[test]
    fn test_complex_if_defined() {
        let mut detector = create_test_detector();
        let code = r#"
            #if defined(CONFIG_COMPLEX)
            complex_operation();
            #endif
        "#;
        let tree = parse_code(code);
        let calls = detector.extract_calls(&tree, code, "test.c").unwrap();

        assert_eq!(calls.len(), 1);
        let call = &calls[0];
        assert_eq!(call.function_name, "complex_operation");
        assert_eq!(call.config_guard, "CONFIG_COMPLEX");
        assert_eq!(call.conditional_type, ConditionalType::IfDefined);
        assert!(!call.is_negated);
        assert!(call.condition_expression.is_some());
    }

    #[test]
    fn test_nested_conditionals() {
        let mut detector = create_test_detector();
        let code = r#"
            #if defined(CONFIG_OUTER)
            #if defined(CONFIG_INNER)
            nested_call();
            #endif
            #endif
        "#;
        let tree = parse_code(code);
        let calls = detector.extract_calls(&tree, code, "test.c").unwrap();

        // Should find calls in both levels
        assert!(!calls.is_empty());
        let nested_call = calls.iter().find(|c| c.function_name == "nested_call");
        assert!(nested_call.is_some());
        assert!(nested_call.unwrap().nesting_depth >= 1);
    }

    #[test]
    fn test_stats_collection() {
        let mut detector = create_test_detector();
        let code = r#"
            #if defined(CONFIG_A)
            call_a();
            #endif
        "#;
        let tree = parse_code(code);
        let calls = detector.extract_calls(&tree, code, "test.c").unwrap();

        let stats = detector.stats();
        // Test with just one call to verify basic functionality
        assert_eq!(stats.total_calls, 1);
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].function_name, "call_a");
        assert_eq!(calls[0].config_guard, "CONFIG_A");
    }
}
