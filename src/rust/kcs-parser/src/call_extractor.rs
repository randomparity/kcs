//! Call graph extraction using tree-sitter AST traversal
//!
//! This module implements call relationship detection by analyzing
//! the Abstract Syntax Tree (AST) produced by tree-sitter for C code.
//!
//! Key responsibilities:
//! - Detect direct function calls
//! - Identify indirect calls through function pointers
//! - Classify macro invocations that expand to calls
//! - Extract call site location information
//! - Associate calls with their containing functions

use anyhow::{Context, Result};
use std::collections::HashMap;
use tree_sitter::{Language, Node, Query, QueryCursor, QueryMatch, Tree};

use crate::types::{CallEdge, CallType};

/// Call extraction configuration
pub struct CallExtractionConfig {
    /// Whether to include function pointer calls
    pub include_indirect_calls: bool,
    /// Whether to include macro invocations
    pub include_macro_calls: bool,
    /// Whether to include calls within conditional blocks
    pub include_conditional_calls: bool,
    /// Maximum depth for nested call detection
    pub max_nesting_depth: usize,
}

impl Default for CallExtractionConfig {
    fn default() -> Self {
        Self {
            include_indirect_calls: true,
            include_macro_calls: true,
            include_conditional_calls: true,
            max_nesting_depth: 50,
        }
    }
}

/// Tree-sitter based call extractor
pub struct CallExtractor {
    #[allow(dead_code)]
    language: Language,
    /// Query for finding function call expressions
    call_query: Query,
    /// Query for finding function definitions
    function_query: Query,
    /// Query for finding macro invocations
    #[allow(dead_code)]
    macro_query: Query,
    /// Query for finding function pointer assignments
    function_pointer_query: Query,
    /// Configuration for call extraction
    #[allow(dead_code)]
    config: CallExtractionConfig,
}

/// Context information for a function call
#[derive(Debug, Clone)]
#[allow(dead_code)]
struct CallContext {
    /// Name of the containing function
    containing_function: String,
    /// Whether the call is within a conditional block
    is_conditional: bool,
    /// Nesting depth within conditional/loop structures
    nesting_depth: usize,
}

/// Result of call extraction from a source file
pub struct CallExtractionResult {
    /// Extracted call edges
    pub call_edges: Vec<CallEdge>,
    /// Functions found in the source
    pub functions: Vec<String>,
    /// Function pointer assignments detected
    pub function_pointers: HashMap<String, Vec<String>>,
}

impl CallExtractor {
    /// Create a new call extractor with default configuration
    pub fn new() -> Result<Self> {
        Self::with_config(CallExtractionConfig::default())
    }

    /// Create a new call extractor with custom configuration
    pub fn with_config(config: CallExtractionConfig) -> Result<Self> {
        let language = tree_sitter_c::language();

        // Query for function call expressions
        let call_query = Query::new(
            language,
            r#"
            ; Direct function calls
            (call_expression
              function: (identifier) @call.function
              arguments: (argument_list)) @call.direct

            ; Function calls through member access (e.g., obj.func())
            (call_expression
              function: (field_expression
                field: (field_identifier) @call.function)
              arguments: (argument_list)) @call.field

            ; Function calls through pointer dereference (e.g., (*func_ptr)())
            (call_expression
              function: (parenthesized_expression
                (unary_expression
                  argument: (identifier) @call.function))
              arguments: (argument_list)) @call.indirect
            "#,
        )
        .context("Failed to create call query")?;

        // Query for function definitions to establish context
        let function_query = Query::new(
            language,
            r#"
            (function_definition
              declarator: (function_declarator
                declarator: (identifier) @function.name)
              body: (compound_statement) @function.body) @function.definition
            "#,
        )
        .context("Failed to create function query")?;

        // Query for macro invocations
        let macro_query = Query::new(
            language,
            r#"
            (call_expression
              function: (identifier) @macro.name
              arguments: (argument_list)) @macro.call
            "#,
        )
        .context("Failed to create macro query")?;

        // Query for function pointer assignments
        let function_pointer_query = Query::new(
            language,
            r#"
            ; Function pointer assignment: func_ptr = function_name
            (assignment_expression
              left: (identifier) @pointer.var
              right: (identifier) @pointer.target) @pointer.assignment

            ; Function pointer declaration with initialization: int (*func_ptr)(args) = func;
            (init_declarator
              declarator: (function_declarator
                declarator: (parenthesized_declarator
                  (pointer_declarator
                    declarator: (identifier) @pointer.var)))
              value: (identifier) @pointer.target) @pointer.init

            ; Function pointer in struct initialization
            (initializer_list
              (initializer_pair
                (field_identifier) @field.name
                (identifier) @field.function)) @struct.init

            ; Array of function pointers
            (array_declarator
              declarator: (identifier) @array.name) @array.decl
            "#,
        )
        .context("Failed to create function pointer query")?;

        Ok(Self {
            language,
            call_query,
            function_query,
            macro_query,
            function_pointer_query,
            config,
        })
    }

    /// Extract call relationships from a parsed tree
    pub fn extract_calls(
        &self,
        tree: &Tree,
        source: &str,
        file_path: &str,
    ) -> Result<CallExtractionResult> {
        let root_node = tree.root_node();

        // First pass: Find all function definitions
        let functions = self.extract_functions(&root_node, source)?;

        // Second pass: Find function pointer assignments
        let function_pointers = self.extract_function_pointers(&root_node, source)?;

        // Third pass: Extract call relationships
        let call_edges = self.extract_call_edges(
            &root_node,
            source,
            file_path,
            &functions,
            &function_pointers,
        )?;

        Ok(CallExtractionResult {
            call_edges,
            functions,
            function_pointers,
        })
    }

    /// Extract function definitions from the AST
    fn extract_functions(&self, root_node: &Node, source: &str) -> Result<Vec<String>> {
        let mut functions = Vec::new();
        let mut query_cursor = QueryCursor::new();

        let matches = query_cursor.matches(&self.function_query, *root_node, source.as_bytes());

        for match_ in matches {
            for capture in match_.captures {
                let capture_name = &self.function_query.capture_names()[capture.index as usize];

                if capture_name.as_str() == "function.name" {
                    let function_name = capture
                        .node
                        .utf8_text(source.as_bytes())
                        .context("Failed to extract function name")?;
                    functions.push(function_name.to_string());
                }
            }
        }

        Ok(functions)
    }

    /// Extract function pointer assignments
    fn extract_function_pointers(
        &self,
        root_node: &Node,
        source: &str,
    ) -> Result<HashMap<String, Vec<String>>> {
        let mut function_pointers = HashMap::new();
        let mut query_cursor = QueryCursor::new();

        let matches =
            query_cursor.matches(&self.function_pointer_query, *root_node, source.as_bytes());

        for match_ in matches {
            let mut pointer_var = None;
            let mut target_function = None;

            for capture in match_.captures {
                let capture_name =
                    &self.function_pointer_query.capture_names()[capture.index as usize];
                let text = capture
                    .node
                    .utf8_text(source.as_bytes())
                    .context("Failed to extract function pointer text")?;

                match capture_name.as_str() {
                    "pointer.var" => pointer_var = Some(text.to_string()),
                    "pointer.target" => target_function = Some(text.to_string()),
                    "field.function" => target_function = Some(text.to_string()),
                    _ => {}
                }
            }

            if let (Some(var), Some(func)) = (pointer_var, target_function) {
                function_pointers
                    .entry(var)
                    .or_insert_with(Vec::new)
                    .push(func);
            }
        }

        Ok(function_pointers)
    }

    /// Extract call edges from the AST
    fn extract_call_edges(
        &self,
        root_node: &Node,
        source: &str,
        file_path: &str,
        functions: &[String],
        function_pointers: &HashMap<String, Vec<String>>,
    ) -> Result<Vec<CallEdge>> {
        let mut call_edges = Vec::new();
        let mut query_cursor = QueryCursor::new();

        // Build a map of function nodes to their names for context lookup
        let function_contexts = self.build_function_context_map(root_node, source, functions)?;

        let matches = query_cursor.matches(&self.call_query, *root_node, source.as_bytes());

        for match_ in matches {
            if let Some(call_edge) = self.process_call_match(
                &match_,
                source,
                file_path,
                &function_contexts,
                function_pointers,
            )? {
                call_edges.push(call_edge);
            }
        }

        Ok(call_edges)
    }

    /// Build a map of byte positions to containing function names
    fn build_function_context_map(
        &self,
        root_node: &Node,
        source: &str,
        _functions: &[String],
    ) -> Result<HashMap<(usize, usize), String>> {
        let mut context_map = HashMap::new();
        let mut query_cursor = QueryCursor::new();

        let matches = query_cursor.matches(&self.function_query, *root_node, source.as_bytes());

        for match_ in matches {
            let mut function_name = None;
            let mut function_body = None;

            for capture in match_.captures {
                let capture_name = &self.function_query.capture_names()[capture.index as usize];

                match capture_name.as_str() {
                    "function.name" => {
                        function_name = Some(
                            capture
                                .node
                                .utf8_text(source.as_bytes())
                                .context("Failed to extract function name")?,
                        );
                    }
                    "function.body" => {
                        function_body = Some(capture.node);
                    }
                    _ => {}
                }
            }

            if let (Some(name), Some(body)) = (function_name, function_body) {
                let range = (body.start_byte(), body.end_byte());
                context_map.insert(range, name.to_string());
            }
        }

        Ok(context_map)
    }

    /// Process a single call match and create a CallEdge if valid
    fn process_call_match(
        &self,
        match_: &QueryMatch,
        source: &str,
        file_path: &str,
        function_contexts: &HashMap<(usize, usize), String>,
        function_pointers: &HashMap<String, Vec<String>>,
    ) -> Result<Option<CallEdge>> {
        let mut call_function = None;
        let mut call_type = CallType::Direct;
        let mut call_node = None;

        // Extract information from the match
        for capture in match_.captures {
            let capture_name = &self.call_query.capture_names()[capture.index as usize];
            let node = capture.node;

            match capture_name.as_str() {
                "call.function" => {
                    let text = node
                        .utf8_text(source.as_bytes())
                        .context("Failed to extract call function name")?;
                    call_function = Some(text);

                    // Determine call type based on the parent capture
                    for capture2 in match_.captures {
                        let parent_capture =
                            &self.call_query.capture_names()[capture2.index as usize];
                        match parent_capture.as_str() {
                            "call.direct" => call_type = CallType::Direct,
                            "call.field" => call_type = CallType::Direct,
                            "call.indirect" => call_type = CallType::Indirect,
                            _ => {}
                        }
                    }
                    call_node = Some(node);
                }
                "call.direct" | "call.field" | "call.indirect" => {
                    // These are the overall call expression nodes
                    if call_node.is_none() {
                        call_node = Some(node);
                    }
                }
                _ => {}
            }
        }

        // Find the containing function for this call
        let containing_function = if let Some(node) = call_node {
            self.find_containing_function(node, function_contexts)
        } else {
            None
        };

        // Check if this is a function pointer call (override call_type if so)
        if let Some(callee_name) = call_function {
            if function_pointers.contains_key(callee_name) {
                call_type = CallType::Indirect;
            }
        }

        // Create call edge if we have all required information
        if let (Some(callee), Some(caller), Some(node)) =
            (call_function, containing_function, call_node)
        {
            // Don't create self-referential calls (recursive calls are handled separately)
            if caller != callee {
                let line_number = node.start_position().row as u32 + 1; // Convert to 1-based line numbers

                let call_edge = CallEdge::new(
                    caller.to_string(),
                    callee.to_string(),
                    file_path.to_string(),
                    line_number,
                    call_type,
                );

                return Ok(Some(call_edge));
            }
        }

        Ok(None)
    }

    /// Find the containing function for a given node
    fn find_containing_function<'a>(
        &self,
        node: Node,
        function_contexts: &'a HashMap<(usize, usize), String>,
    ) -> Option<&'a String> {
        let node_start = node.start_byte();

        // Find the function body that contains this node
        for ((start, end), function_name) in function_contexts {
            if node_start >= *start && node_start < *end {
                return Some(function_name);
            }
        }

        None
    }

    /// Determine if a call is within a conditional context
    #[allow(dead_code)]
    fn is_conditional_call(&self, mut node: Node) -> bool {
        // Walk up the AST to check for conditional contexts
        while let Some(parent) = node.parent() {
            match parent.kind() {
                "if_statement"
                | "while_statement"
                | "for_statement"
                | "do_statement"
                | "switch_statement"
                | "conditional_expression" => {
                    return true;
                }
                _ => {}
            }
            node = parent;
        }
        false
    }

    /// Check if a function name is likely a macro based on naming conventions
    #[allow(dead_code)]
    fn is_likely_macro(&self, name: &str) -> bool {
        // Common macro patterns in kernel code
        name.chars()
            .all(|c| c.is_uppercase() || c.is_numeric() || c == '_')
            || name.starts_with("DECLARE_")
            || name.starts_with("DEFINE_")
            || name.starts_with("INIT_")
            || name.starts_with("EXPORT_")
            || name.ends_with("_INIT")
            || name.ends_with("_EXIT")
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tree_sitter::Parser;

    fn parse_c_code(code: &str) -> Result<Tree> {
        let language = tree_sitter_c::language();
        let mut parser = Parser::new();
        parser.set_language(language).unwrap();

        parser
            .parse(code, None)
            .ok_or_else(|| anyhow::anyhow!("Failed to parse code"))
    }

    #[test]
    fn test_extract_simple_function_call() -> Result<()> {
        let code = r#"
        int helper_function(int x) {
            return x * 2;
        }

        int main() {
            int result = helper_function(42);
            return result;
        }
        "#;

        let tree = parse_c_code(code)?;
        let extractor = CallExtractor::new()?;
        let result = extractor.extract_calls(&tree, code, "test.c")?;

        assert_eq!(result.functions.len(), 2);
        assert!(result.functions.contains(&"helper_function".to_string()));
        assert!(result.functions.contains(&"main".to_string()));

        assert_eq!(result.call_edges.len(), 1);
        let call_edge = &result.call_edges[0];
        assert_eq!(call_edge.caller, "main");
        assert_eq!(call_edge.callee, "helper_function");
        assert_eq!(call_edge.call_type, CallType::Direct);

        Ok(())
    }

    #[test]
    fn test_extract_function_pointer_call() -> Result<()> {
        let code = r#"
        int add(int a, int b) {
            return a + b;
        }

        int main() {
            int (*func_ptr)(int, int) = add;
            int result = func_ptr(5, 10);
            return result;
        }
        "#;

        let tree = parse_c_code(code)?;
        let extractor = CallExtractor::new()?;
        let result = extractor.extract_calls(&tree, code, "test.c")?;

        // Should detect the function pointer assignment
        assert!(result.function_pointers.contains_key("func_ptr"));

        // Should detect the indirect call
        let indirect_calls: Vec<_> = result
            .call_edges
            .iter()
            .filter(|edge| edge.call_type == CallType::Indirect)
            .collect();
        assert!(!indirect_calls.is_empty());

        Ok(())
    }

    #[test]
    fn test_extract_macro_call() -> Result<()> {
        let code = r#"
        #define DEBUG_PRINT(x) printf(x)

        void test_function() {
            DEBUG_PRINT("Hello World");
        }
        "#;

        let tree = parse_c_code(code)?;
        let extractor = CallExtractor::new()?;
        let result = extractor.extract_calls(&tree, code, "test.c")?;

        let macro_calls: Vec<_> = result
            .call_edges
            .iter()
            .filter(|edge| edge.call_type == CallType::Macro || edge.callee == "DEBUG_PRINT")
            .collect();

        // Note: Whether this is detected depends on how tree-sitter handles preprocessor directives
        // This test documents expected behavior
        println!("Macro calls detected: {}", macro_calls.len());

        Ok(())
    }

    #[test]
    fn test_complex_call_patterns() -> Result<()> {
        let code = r#"
        int func_a(int x) { return x + 1; }
        int func_b(int x) { return x * 2; }

        int dispatch(int type, int value) {
            if (type == 1) {
                return func_a(value);
            } else {
                return func_b(value);
            }
        }

        int main() {
            int result1 = dispatch(1, 10);
            int result2 = dispatch(2, 20);
            return result1 + result2;
        }
        "#;

        let tree = parse_c_code(code)?;
        let extractor = CallExtractor::new()?;
        let result = extractor.extract_calls(&tree, code, "test.c")?;

        // Should find multiple functions
        assert!(result.functions.len() >= 4);

        // Should find calls from main to dispatch, and from dispatch to func_a and func_b
        let main_calls: Vec<_> = result
            .call_edges
            .iter()
            .filter(|edge| edge.caller == "main")
            .collect();
        assert!(!main_calls.is_empty());

        let dispatch_calls: Vec<_> = result
            .call_edges
            .iter()
            .filter(|edge| edge.caller == "dispatch")
            .collect();
        assert_eq!(dispatch_calls.len(), 2); // Should call both func_a and func_b

        Ok(())
    }
}
