//! AST traversal coordinator for call graph extraction.
//!
//! This module provides utilities for traversing Tree-sitter ASTs efficiently
//! and coordinating the extraction of call information from different parts
//! of the syntax tree. It manages the interface between Tree-sitter parser
//! output and the specialized call extraction modules.

use anyhow::{Context, Result};
use std::collections::{HashMap, VecDeque};
use tree_sitter::{Node, Tree};

/// Configuration for AST traversal behavior.
#[derive(Debug, Clone)]
pub struct TraversalConfig {
    /// Maximum depth to traverse in the AST
    pub max_depth: usize,
    /// Whether to include preprocessor directives in traversal
    pub include_preprocessor: bool,
    /// Whether to traverse into function bodies
    pub traverse_function_bodies: bool,
    /// Whether to traverse macro definitions
    pub traverse_macro_definitions: bool,
    /// Maximum number of nodes to visit (safety limit)
    pub max_nodes: usize,
    /// Node types to skip during traversal
    pub skip_node_types: Vec<String>,
}

impl Default for TraversalConfig {
    fn default() -> Self {
        Self {
            max_depth: 1000,
            include_preprocessor: true,
            traverse_function_bodies: true,
            traverse_macro_definitions: false,
            max_nodes: 1_000_000,
            skip_node_types: vec![
                "comment".to_string(),
                "line_comment".to_string(),
                "block_comment".to_string(),
            ],
        }
    }
}

/// Statistics collected during AST traversal.
#[derive(Debug, Clone, Default)]
pub struct TraversalStats {
    /// Total number of nodes visited
    pub nodes_visited: usize,
    /// Maximum depth reached
    pub max_depth_reached: usize,
    /// Number of function definitions found
    pub function_definitions: usize,
    /// Number of call expressions found
    pub call_expressions: usize,
    /// Number of preprocessor directives found
    pub preprocessor_directives: usize,
    /// Number of macro definitions found
    pub macro_definitions: usize,
    /// Time taken for traversal in milliseconds
    pub traversal_time_ms: u64,
    /// Nodes skipped due to configuration
    pub nodes_skipped: usize,
}

/// Context information during AST traversal.
#[derive(Debug, Clone)]
pub struct TraversalContext {
    /// Current depth in the AST
    pub depth: usize,
    /// Path of node types from root to current node
    pub node_path: Vec<String>,
    /// Current function scope (if any)
    pub current_function: Option<String>,
    /// Current file being processed
    pub file_path: String,
    /// Whether we're inside a conditional compilation block
    pub in_conditional: bool,
    /// Current conditional compilation context
    pub conditional_context: Vec<String>,
}

impl TraversalContext {
    /// Create a new traversal context for a file.
    pub fn new(file_path: String) -> Self {
        Self {
            depth: 0,
            node_path: Vec::new(),
            current_function: None,
            file_path,
            in_conditional: false,
            conditional_context: Vec::new(),
        }
    }

    /// Push a new node type onto the path.
    pub fn push_node(&mut self, node_type: &str) {
        self.node_path.push(node_type.to_string());
        self.depth += 1;
    }

    /// Pop the last node type from the path.
    pub fn pop_node(&mut self) {
        self.node_path.pop();
        if self.depth > 0 {
            self.depth -= 1;
        }
    }

    /// Enter a function scope.
    pub fn enter_function(&mut self, function_name: String) {
        self.current_function = Some(function_name);
    }

    /// Exit the current function scope.
    pub fn exit_function(&mut self) {
        self.current_function = None;
    }

    /// Enter a conditional compilation block.
    pub fn enter_conditional(&mut self, condition: String) {
        self.in_conditional = true;
        self.conditional_context.push(condition);
    }

    /// Exit the current conditional compilation block.
    pub fn exit_conditional(&mut self) {
        self.conditional_context.pop();
        self.in_conditional = !self.conditional_context.is_empty();
    }
}

/// Visitor trait for processing AST nodes during traversal.
pub trait AstVisitor {
    /// Called when entering a node.
    fn visit_node(&mut self, node: &Node, context: &TraversalContext, source: &str) -> Result<()>;

    /// Called when exiting a node (optional).
    fn exit_node(&mut self, node: &Node, context: &TraversalContext) -> Result<()> {
        let _ = (node, context);
        Ok(())
    }

    /// Check if this node type should be processed by this visitor.
    fn should_visit(&self, node_type: &str) -> bool;

    /// Get the name of this visitor for debugging.
    fn name(&self) -> &'static str;
}

/// Main AST traversal coordinator.
pub struct AstTraversal {
    /// Configuration for traversal behavior
    config: TraversalConfig,
    /// Statistics collected during traversal
    stats: TraversalStats,
    /// Registered visitors
    visitors: Vec<Box<dyn AstVisitor>>,
}

impl AstTraversal {
    /// Create a new AST traversal coordinator.
    pub fn new(config: TraversalConfig) -> Self {
        Self {
            config,
            stats: TraversalStats::default(),
            visitors: Vec::new(),
        }
    }

    /// Create a new AST traversal coordinator with default configuration.
    pub fn new_default() -> Self {
        Self::new(TraversalConfig::default())
    }

    /// Register a visitor to be called during traversal.
    pub fn add_visitor(&mut self, visitor: Box<dyn AstVisitor>) {
        self.visitors.push(visitor);
    }

    /// Get the current traversal statistics.
    pub fn stats(&self) -> &TraversalStats {
        &self.stats
    }

    /// Reset the traversal statistics.
    pub fn reset_stats(&mut self) {
        self.stats = TraversalStats::default();
    }

    /// Traverse the entire AST tree using depth-first traversal.
    pub fn traverse_tree(&mut self, tree: &Tree, source: &str, file_path: &str) -> Result<()> {
        let start_time = std::time::Instant::now();
        self.reset_stats();

        let root_node = tree.root_node();
        let mut context = TraversalContext::new(file_path.to_string());

        self.traverse_node_recursive(&root_node, &mut context, source)
            .with_context(|| format!("Failed to traverse AST for file: {}", file_path))?;

        self.stats.traversal_time_ms = start_time.elapsed().as_millis() as u64;
        Ok(())
    }

    /// Traverse using an iterative approach with a cursor for better performance.
    pub fn traverse_tree_iterative(
        &mut self,
        tree: &Tree,
        source: &str,
        file_path: &str,
    ) -> Result<()> {
        let start_time = std::time::Instant::now();
        self.reset_stats();

        let cursor = tree.walk();
        let mut context = TraversalContext::new(file_path.to_string());
        let mut node_stack = VecDeque::new();

        // Start with the root node
        node_stack.push_back((cursor.node(), 0));

        while let Some((node, depth)) = node_stack.pop_front() {
            if self.stats.nodes_visited >= self.config.max_nodes {
                break;
            }

            if depth > self.config.max_depth {
                continue;
            }

            // Update context
            context.depth = depth;
            if depth > self.stats.max_depth_reached {
                self.stats.max_depth_reached = depth;
            }

            // Check if we should skip this node
            if self.should_skip_node(&node) {
                self.stats.nodes_skipped += 1;
                continue;
            }

            // Update context based on node type
            self.update_context_for_node(&node, &mut context, source)?;

            // Visit the node with all applicable visitors
            self.visit_node_with_visitors(&node, &context, source)?;

            self.stats.nodes_visited += 1;

            // Add children to the stack (in reverse order for left-to-right processing)
            let mut children = Vec::new();
            let mut child_cursor = node.walk();
            if child_cursor.goto_first_child() {
                loop {
                    children.push((child_cursor.node(), depth + 1));
                    if !child_cursor.goto_next_sibling() {
                        break;
                    }
                }
            }

            // Add children in reverse order so they're processed left-to-right
            for child in children.into_iter().rev() {
                node_stack.push_front(child);
            }
        }

        self.stats.traversal_time_ms = start_time.elapsed().as_millis() as u64;
        Ok(())
    }

    /// Find all nodes of a specific type in the tree.
    pub fn find_nodes_by_type<'a>(&self, tree: &'a Tree, node_type: &str) -> Vec<Node<'a>> {
        let mut nodes = Vec::new();
        let cursor = tree.walk();
        let mut stack = VecDeque::new();
        stack.push_back(cursor.node());

        while let Some(node) = stack.pop_front() {
            if node.kind() == node_type {
                nodes.push(node);
            }

            let mut child_cursor = node.walk();
            if child_cursor.goto_first_child() {
                loop {
                    stack.push_back(child_cursor.node());
                    if !child_cursor.goto_next_sibling() {
                        break;
                    }
                }
            }
        }

        nodes
    }

    /// Get all function definitions in the tree.
    pub fn get_function_definitions<'a>(
        &self,
        tree: &'a Tree,
        source: &str,
    ) -> Result<HashMap<String, Node<'a>>> {
        let mut functions = HashMap::new();
        let function_nodes = self.find_nodes_by_type(tree, "function_definition");

        for node in function_nodes {
            if let Some(name) = self.extract_function_name(&node, source) {
                functions.insert(name, node);
            }
        }

        Ok(functions)
    }

    /// Recursive traversal implementation.
    fn traverse_node_recursive(
        &mut self,
        node: &Node,
        context: &mut TraversalContext,
        source: &str,
    ) -> Result<()> {
        if self.stats.nodes_visited >= self.config.max_nodes {
            return Ok(());
        }

        if context.depth > self.config.max_depth {
            return Ok(());
        }

        if context.depth > self.stats.max_depth_reached {
            self.stats.max_depth_reached = context.depth;
        }

        // Check if we should skip this node
        if self.should_skip_node(node) {
            self.stats.nodes_skipped += 1;
            return Ok(());
        }

        // Update context based on node type
        context.push_node(node.kind());
        self.update_context_for_node(node, context, source)?;

        // Visit the node with all applicable visitors
        self.visit_node_with_visitors(node, context, source)?;

        self.stats.nodes_visited += 1;

        // Traverse children
        let mut cursor = node.walk();
        if cursor.goto_first_child() {
            loop {
                let child = cursor.node();
                self.traverse_node_recursive(&child, context, source)?;

                if !cursor.goto_next_sibling() {
                    break;
                }
            }
        }

        // Exit the node
        for visitor in &mut self.visitors {
            visitor.exit_node(node, context)?;
        }

        context.pop_node();
        Ok(())
    }

    /// Check if a node should be skipped based on configuration.
    fn should_skip_node(&self, node: &Node) -> bool {
        let node_type = node.kind();

        // Skip nodes in the skip list
        if self.config.skip_node_types.contains(&node_type.to_string()) {
            return true;
        }

        // Skip preprocessor nodes if disabled
        if !self.config.include_preprocessor && node_type.starts_with("preproc") {
            return true;
        }

        // Skip macro definitions if disabled
        if !self.config.traverse_macro_definitions
            && (node_type == "preproc_def" || node_type == "preproc_function_def")
        {
            return true;
        }

        false
    }

    /// Update context based on the current node.
    fn update_context_for_node(
        &mut self,
        node: &Node,
        context: &mut TraversalContext,
        source: &str,
    ) -> Result<()> {
        let node_type = node.kind();

        match node_type {
            "function_definition" => {
                if let Some(name) = self.extract_function_name(node, source) {
                    context.enter_function(name);
                    self.stats.function_definitions += 1;
                }
            },
            "call_expression" => {
                self.stats.call_expressions += 1;
            },
            "preproc_ifdef" | "preproc_ifndef" | "preproc_if" => {
                if let Some(condition) = self.extract_preprocessor_condition(node, source) {
                    context.enter_conditional(condition);
                }
                self.stats.preprocessor_directives += 1;
            },
            "preproc_def" | "preproc_function_def" => {
                self.stats.macro_definitions += 1;
            },
            _ => {},
        }

        Ok(())
    }

    /// Visit a node with all applicable visitors.
    fn visit_node_with_visitors(
        &mut self,
        node: &Node,
        context: &TraversalContext,
        source: &str,
    ) -> Result<()> {
        for visitor in &mut self.visitors {
            if visitor.should_visit(node.kind()) {
                visitor.visit_node(node, context, source).with_context(|| {
                    format!("Visitor '{}' failed on node type '{}'", visitor.name(), node.kind())
                })?;
            }
        }
        Ok(())
    }

    /// Extract function name from a function definition node.
    fn extract_function_name(&self, node: &Node, source: &str) -> Option<String> {
        let mut cursor = node.walk();

        // Look for function_declarator node
        if cursor.goto_first_child() {
            loop {
                let child = cursor.node();
                if child.kind() == "function_declarator" {
                    // Look for identifier within the declarator
                    let mut decl_cursor = child.walk();
                    if decl_cursor.goto_first_child() {
                        loop {
                            let decl_child = decl_cursor.node();
                            if decl_child.kind() == "identifier" {
                                return Some(
                                    decl_child.utf8_text(source.as_bytes()).ok()?.to_string(),
                                );
                            }
                            if !decl_cursor.goto_next_sibling() {
                                break;
                            }
                        }
                    }
                }
                if !cursor.goto_next_sibling() {
                    break;
                }
            }
        }

        None
    }

    /// Extract preprocessor condition from a preprocessor directive.
    fn extract_preprocessor_condition(&self, node: &Node, source: &str) -> Option<String> {
        // For now, just return the node text as the condition
        node.utf8_text(source.as_bytes()).ok().map(|s| s.to_string())
    }
}

/// Simple visitor that collects basic statistics.
#[derive(Default)]
pub struct StatsCollector {
    pub function_calls: Vec<String>,
    pub function_definitions: Vec<String>,
    pub macro_usages: Vec<String>,
}

impl StatsCollector {
    pub fn new() -> Self {
        Self::default()
    }
}

impl AstVisitor for StatsCollector {
    fn visit_node(&mut self, node: &Node, _context: &TraversalContext, source: &str) -> Result<()> {
        match node.kind() {
            "call_expression" => {
                if let Ok(text) = node.utf8_text(source.as_bytes()) {
                    self.function_calls.push(text.to_string());
                }
            },
            "function_definition" => {
                if let Ok(text) = node.utf8_text(source.as_bytes()) {
                    self.function_definitions.push(text.to_string());
                }
            },
            "preproc_def" | "preproc_function_def" => {
                if let Ok(text) = node.utf8_text(source.as_bytes()) {
                    self.macro_usages.push(text.to_string());
                }
            },
            _ => {},
        }
        Ok(())
    }

    fn should_visit(&self, node_type: &str) -> bool {
        matches!(
            node_type,
            "call_expression" | "function_definition" | "preproc_def" | "preproc_function_def"
        )
    }

    fn name(&self) -> &'static str {
        "StatsCollector"
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tree_sitter::Parser;

    fn create_test_parser() -> Parser {
        let mut parser = Parser::new();
        parser.set_language(tree_sitter_c::language()).unwrap();
        parser
    }

    fn parse_test_code(code: &str) -> tree_sitter::Tree {
        let mut parser = create_test_parser();
        parser.parse(code, None).unwrap()
    }

    #[test]
    fn test_basic_traversal() {
        let code = r#"
            int main() {
                printf("Hello, world!");
                return 0;
            }
        "#;

        let tree = parse_test_code(code);
        let mut traversal = AstTraversal::new_default();

        let result = traversal.traverse_tree(&tree, code, "test.c");
        assert!(result.is_ok());

        let stats = traversal.stats();
        assert!(stats.nodes_visited > 0);
        assert!(stats.function_definitions >= 1);
        assert!(stats.call_expressions >= 1);
    }

    #[test]
    fn test_iterative_traversal() {
        let code = r#"
            void foo() { bar(); }
            void bar() { baz(); }
        "#;

        let tree = parse_test_code(code);
        let mut traversal = AstTraversal::new_default();

        let result = traversal.traverse_tree_iterative(&tree, code, "test.c");
        assert!(result.is_ok());

        let stats = traversal.stats();
        assert!(stats.nodes_visited > 0);
        assert!(stats.function_definitions >= 2);
        assert!(stats.call_expressions >= 2);
    }

    #[test]
    fn test_stats_collector_visitor() {
        let code = r#"
            int main() {
                printf("Hello");
                fprintf(stderr, "Error");
                return 0;
            }
        "#;

        let tree = parse_test_code(code);
        let mut traversal = AstTraversal::new_default();
        let collector = Box::new(StatsCollector::new());
        traversal.add_visitor(collector);

        let result = traversal.traverse_tree(&tree, code, "test.c");
        assert!(result.is_ok());
    }

    #[test]
    fn test_function_definition_extraction() {
        let code = r#"
            int add(int a, int b) {
                return a + b;
            }

            void print_hello() {
                printf("Hello");
            }
        "#;

        let tree = parse_test_code(code);
        let traversal = AstTraversal::new_default();

        let functions = traversal.get_function_definitions(&tree, code).unwrap();
        assert!(functions.contains_key("add"));
        assert!(functions.contains_key("print_hello"));
    }

    #[test]
    fn test_find_nodes_by_type() {
        let code = r#"
            int main() {
                foo();
                bar();
                return 0;
            }
        "#;

        let tree = parse_test_code(code);
        let traversal = AstTraversal::new_default();

        let call_nodes = traversal.find_nodes_by_type(&tree, "call_expression");
        assert!(call_nodes.len() >= 2); // foo() and bar()
    }

    #[test]
    fn test_max_depth_limit() {
        let code = r#"
            int main() {
                if (1) {
                    if (2) {
                        if (3) {
                            printf("deep");
                        }
                    }
                }
                return 0;
            }
        "#;

        let config = TraversalConfig {
            max_depth: 5, // Very shallow limit
            ..Default::default()
        };

        let tree = parse_test_code(code);
        let mut traversal = AstTraversal::new(config);

        let result = traversal.traverse_tree(&tree, code, "test.c");
        assert!(result.is_ok());

        let stats = traversal.stats();
        assert!(stats.max_depth_reached <= 5);
    }

    #[test]
    fn test_traversal_context() {
        let mut context = TraversalContext::new("test.c".to_string());

        context.push_node("function_definition");
        assert_eq!(context.depth, 1);
        assert_eq!(context.node_path.len(), 1);

        context.enter_function("main".to_string());
        assert_eq!(context.current_function, Some("main".to_string()));

        context.enter_conditional("DEBUG".to_string());
        assert!(context.in_conditional);

        context.exit_conditional();
        assert!(!context.in_conditional);

        context.exit_function();
        assert_eq!(context.current_function, None);

        context.pop_node();
        assert_eq!(context.depth, 0);
    }
}
