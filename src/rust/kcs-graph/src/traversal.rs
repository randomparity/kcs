//! Graph traversal algorithms for kernel call graphs
//!
//! This module provides various traversal algorithms optimized for kernel
//! call graph analysis, including BFS, DFS, and specialized traversals.

use crate::{CallType, KernelGraph, Symbol};
use anyhow::Result;
use petgraph::graph::NodeIndex;
use petgraph::visit::EdgeRef;
use petgraph::Direction;
use serde::{Deserialize, Serialize};
use std::collections::{HashSet, VecDeque};

/// Result of a traversal operation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TraversalResult {
    /// Symbols visited during traversal
    pub visited: Vec<Symbol>,
    /// Order in which symbols were visited
    pub visit_order: Vec<String>,
    /// Edges traversed (as symbol name pairs)
    pub edges: Vec<(String, String)>,
    /// Maximum depth reached
    pub max_depth: usize,
    /// Number of symbols visited
    pub count: usize,
}

/// Options for controlling traversal behavior
#[derive(Debug, Clone)]
pub struct TraversalOptions {
    /// Maximum depth to traverse (None for unlimited)
    pub max_depth: Option<usize>,
    /// Filter by call type
    pub call_type_filter: Option<CallType>,
    /// Only traverse edges with specific config guards
    pub config_filter: Option<String>,
    /// Visit each node only once
    pub unique_visits: bool,
    /// Follow only conditional edges
    pub conditional_only: bool,
}

impl Default for TraversalOptions {
    fn default() -> Self {
        Self {
            max_depth: None,
            call_type_filter: None,
            config_filter: None,
            unique_visits: true,
            conditional_only: false,
        }
    }
}

/// Graph traversal algorithms
pub struct GraphTraversal<'a> {
    graph: &'a KernelGraph,
}

impl<'a> GraphTraversal<'a> {
    /// Create a new traversal for the given graph
    pub fn new(graph: &'a KernelGraph) -> Self {
        Self { graph }
    }

    /// Perform breadth-first traversal from a starting symbol
    pub fn bfs(&self, start: &str, options: TraversalOptions) -> Result<TraversalResult> {
        let Some(&start_idx) = self.graph.symbol_index().get(start) else {
            return Ok(TraversalResult {
                visited: Vec::new(),
                visit_order: Vec::new(),
                edges: Vec::new(),
                max_depth: 0,
                count: 0,
            });
        };

        let mut visited = HashSet::new();
        let mut visit_order = Vec::new();
        let mut visited_symbols = Vec::new();
        let mut edges = Vec::new();
        let mut queue = VecDeque::new();
        let mut max_depth_reached = 0;

        queue.push_back((start_idx, 0));

        while let Some((node_idx, depth)) = queue.pop_front() {
            // Check depth limit
            if let Some(max_depth) = options.max_depth {
                if depth > max_depth {
                    continue;
                }
            }

            // Check if already visited (for unique visits)
            if options.unique_visits && visited.contains(&node_idx) {
                continue;
            }

            visited.insert(node_idx);
            max_depth_reached = max_depth_reached.max(depth);

            // Get the symbol and add to results
            if let Some(symbol) = self.graph.graph().node_weight(node_idx) {
                visited_symbols.push(symbol.clone());
                visit_order.push(symbol.name.clone());
            }

            // Process edges
            for edge_ref in self.graph.graph().edges(node_idx) {
                let edge_data = edge_ref.weight();
                let target_idx = edge_ref.target();

                // Apply filters
                if let Some(ref filter_type) = options.call_type_filter {
                    if edge_data.call_type != *filter_type {
                        continue;
                    }
                }

                if options.conditional_only && !edge_data.conditional {
                    continue;
                }

                if let Some(ref config) = options.config_filter {
                    if edge_data.config_guard.as_ref() != Some(config) {
                        continue;
                    }
                }

                // Edge passed filters, now record it and queue target

                // Record edge
                if let (Some(from), Some(to)) = (
                    self.graph.graph().node_weight(node_idx),
                    self.graph.graph().node_weight(target_idx),
                ) {
                    edges.push((from.name.clone(), to.name.clone()));
                }

                // Add to queue if not visited or not using unique visits
                if !options.unique_visits || !visited.contains(&target_idx) {
                    queue.push_back((target_idx, depth + 1));
                }
            }
        }

        Ok(TraversalResult {
            visited: visited_symbols.clone(),
            visit_order,
            edges,
            max_depth: max_depth_reached,
            count: visited_symbols.len(),
        })
    }

    /// Perform depth-first traversal from a starting symbol
    pub fn dfs(&self, start: &str, options: TraversalOptions) -> Result<TraversalResult> {
        let Some(&start_idx) = self.graph.symbol_index().get(start) else {
            return Ok(TraversalResult {
                visited: Vec::new(),
                visit_order: Vec::new(),
                edges: Vec::new(),
                max_depth: 0,
                count: 0,
            });
        };

        let mut visited = HashSet::new();
        let mut visit_order = Vec::new();
        let mut visited_symbols = Vec::new();
        let mut edges = Vec::new();
        let mut max_depth_reached = 0;

        self.dfs_recursive(
            start_idx,
            0,
            &mut visited,
            &mut visit_order,
            &mut visited_symbols,
            &mut edges,
            &mut max_depth_reached,
            &options,
        );

        Ok(TraversalResult {
            visited: visited_symbols.clone(),
            visit_order,
            edges,
            max_depth: max_depth_reached,
            count: visited_symbols.len(),
        })
    }

    /// Recursive helper for DFS
    #[allow(clippy::too_many_arguments)]
    fn dfs_recursive(
        &self,
        node_idx: NodeIndex,
        depth: usize,
        visited: &mut HashSet<NodeIndex>,
        visit_order: &mut Vec<String>,
        visited_symbols: &mut Vec<Symbol>,
        edges: &mut Vec<(String, String)>,
        max_depth_reached: &mut usize,
        options: &TraversalOptions,
    ) {
        // Check depth limit
        if let Some(max_depth) = options.max_depth {
            if depth > max_depth {
                return;
            }
        }

        // Check if already visited (for unique visits)
        if options.unique_visits && visited.contains(&node_idx) {
            return;
        }

        visited.insert(node_idx);
        *max_depth_reached = (*max_depth_reached).max(depth);

        // Get the symbol and add to results
        if let Some(symbol) = self.graph.graph().node_weight(node_idx) {
            visited_symbols.push(symbol.clone());
            visit_order.push(symbol.name.clone());
        }

        // Process edges
        for edge_ref in self.graph.graph().edges(node_idx) {
            let edge_data = edge_ref.weight();
            let target_idx = edge_ref.target();

            // Apply filters
            if let Some(ref filter_type) = options.call_type_filter {
                if edge_data.call_type != *filter_type {
                    continue;
                }
            }

            if options.conditional_only && !edge_data.conditional {
                continue;
            }

            if let Some(ref config) = options.config_filter {
                if edge_data.config_guard.as_ref() != Some(config) {
                    continue;
                }
            }

            // Record edge
            if let (Some(from), Some(to)) = (
                self.graph.graph().node_weight(node_idx),
                self.graph.graph().node_weight(target_idx),
            ) {
                edges.push((from.name.clone(), to.name.clone()));
            }

            // Recurse
            self.dfs_recursive(
                target_idx,
                depth + 1,
                visited,
                visit_order,
                visited_symbols,
                edges,
                max_depth_reached,
                options,
            );
        }
    }

    /// Perform a topological traversal (only valid for DAGs)
    pub fn topological_sort(&self) -> Result<Vec<Symbol>> {
        use petgraph::algo::toposort;

        match toposort(self.graph.graph(), None) {
            Ok(sorted_indices) => {
                let mut result = Vec::new();
                for idx in sorted_indices {
                    if let Some(symbol) = self.graph.graph().node_weight(idx) {
                        result.push(symbol.clone());
                    }
                }
                Ok(result)
            }
            Err(_) => {
                // Graph has cycles, cannot perform topological sort
                anyhow::bail!("Cannot perform topological sort on graph with cycles")
            }
        }
    }

    /// Find all ancestors of a symbol (symbols that can reach it)
    pub fn find_ancestors(&self, target: &str, max_depth: Option<usize>) -> Vec<Symbol> {
        let Some(&target_idx) = self.graph.symbol_index().get(target) else {
            return Vec::new();
        };

        let mut ancestors = HashSet::new();
        let mut queue = VecDeque::new();
        queue.push_back((target_idx, 0));

        while let Some((node_idx, depth)) = queue.pop_front() {
            if let Some(max) = max_depth {
                if depth > max {
                    continue;
                }
            }

            // Look at incoming edges (who calls this node)
            for edge_ref in self
                .graph
                .graph()
                .edges_directed(node_idx, Direction::Incoming)
            {
                let source_idx = edge_ref.source();

                if !ancestors.contains(&source_idx) {
                    ancestors.insert(source_idx);
                    queue.push_back((source_idx, depth + 1));
                }
            }
        }

        // Convert to symbols
        ancestors
            .into_iter()
            .filter_map(|idx| self.graph.graph().node_weight(idx).cloned())
            .collect()
    }

    /// Find all descendants of a symbol (symbols it can reach)
    pub fn find_descendants(&self, source: &str, max_depth: Option<usize>) -> Vec<Symbol> {
        let Some(&source_idx) = self.graph.symbol_index().get(source) else {
            return Vec::new();
        };

        let mut descendants = HashSet::new();
        let mut queue = VecDeque::new();
        queue.push_back((source_idx, 0));

        while let Some((node_idx, depth)) = queue.pop_front() {
            if let Some(max) = max_depth {
                if depth > max {
                    continue;
                }
            }

            // Look at outgoing edges (what this node calls)
            for edge_ref in self
                .graph
                .graph()
                .edges_directed(node_idx, Direction::Outgoing)
            {
                let target_idx = edge_ref.target();

                if !descendants.contains(&target_idx) {
                    descendants.insert(target_idx);
                    queue.push_back((target_idx, depth + 1));
                }
            }
        }

        // Convert to symbols
        descendants
            .into_iter()
            .filter_map(|idx| self.graph.graph().node_weight(idx).cloned())
            .collect()
    }

    /// Perform a bidirectional search to find path between two nodes
    pub fn bidirectional_search(&self, start: &str, end: &str) -> Option<Vec<Symbol>> {
        let &start_idx = self.graph.symbol_index().get(start)?;
        let &end_idx = self.graph.symbol_index().get(end)?;

        // Forward search from start
        let mut forward_visited = HashSet::new();
        let mut forward_queue = VecDeque::new();
        let mut forward_parent = std::collections::HashMap::new();

        // Backward search from end
        let mut backward_visited = HashSet::new();
        let mut backward_queue = VecDeque::new();
        let mut backward_parent = std::collections::HashMap::new();

        forward_visited.insert(start_idx);
        forward_queue.push_back(start_idx);

        backward_visited.insert(end_idx);
        backward_queue.push_back(end_idx);

        while !forward_queue.is_empty() || !backward_queue.is_empty() {
            // Forward step
            if let Some(node) = forward_queue.pop_front() {
                for edge in self.graph.graph().edges(node) {
                    let target = edge.target();

                    if backward_visited.contains(&target) {
                        // Found meeting point
                        return self.reconstruct_bidirectional_path(
                            start_idx,
                            end_idx,
                            target,
                            &forward_parent,
                            &backward_parent,
                        );
                    }

                    if !forward_visited.contains(&target) {
                        forward_visited.insert(target);
                        forward_parent.insert(target, node);
                        forward_queue.push_back(target);
                    }
                }
            }

            // Backward step
            if let Some(node) = backward_queue.pop_front() {
                for edge in self.graph.graph().edges_directed(node, Direction::Incoming) {
                    let source = edge.source();

                    if forward_visited.contains(&source) {
                        // Found meeting point
                        // But first, record this connection
                        backward_parent.insert(source, node);
                        return self.reconstruct_bidirectional_path(
                            start_idx,
                            end_idx,
                            source,
                            &forward_parent,
                            &backward_parent,
                        );
                    }

                    if !backward_visited.contains(&source) {
                        backward_visited.insert(source);
                        backward_parent.insert(source, node);
                        backward_queue.push_back(source);
                    }
                }
            }
        }

        None
    }

    /// Helper to reconstruct path from bidirectional search
    fn reconstruct_bidirectional_path(
        &self,
        start: NodeIndex,
        end: NodeIndex,
        meeting: NodeIndex,
        forward_parent: &std::collections::HashMap<NodeIndex, NodeIndex>,
        backward_parent: &std::collections::HashMap<NodeIndex, NodeIndex>,
    ) -> Option<Vec<Symbol>> {
        let mut full_path = Vec::new();

        // Build forward path from start to meeting point
        let mut current = meeting;
        let mut forward_path = vec![current];

        while current != start {
            if let Some(&parent) = forward_parent.get(&current) {
                forward_path.push(parent);
                current = parent;
            } else {
                break;
            }
        }

        forward_path.reverse();
        full_path.extend(forward_path);

        // Build backward path from meeting point to end
        // Note: backward_parent maps node -> its parent in backward search
        // backward_parent[X] = Y means "X came from Y in backward search"
        // So to go from meeting to end, we follow the chain
        let mut backward_path = Vec::new();
        current = meeting;

        // Only add nodes after the meeting point
        while current != end {
            if let Some(&next) = backward_parent.get(&current) {
                backward_path.push(next);
                current = next;
            } else {
                // If we can't find a parent, check if meeting == end
                if meeting == end {
                    // No backward path needed
                    break;
                }
                // Otherwise, try to build path differently
                break;
            }
        }

        // Add backward path (excluding meeting point which is already in forward path)
        full_path.extend(backward_path);

        // Convert to symbols
        let mut result = Vec::new();
        for idx in full_path {
            if let Some(symbol) = self.graph.graph().node_weight(idx) {
                result.push(symbol.clone());
            }
        }

        Some(result)
    }

    /// Find strongly connected components and return them ordered by size
    pub fn find_components(&self) -> Vec<Vec<Symbol>> {
        use petgraph::algo::tarjan_scc;

        let sccs = tarjan_scc(self.graph.graph());
        let mut components = Vec::new();

        for scc in sccs {
            if !scc.is_empty() {
                let mut component = Vec::new();
                for idx in scc {
                    if let Some(symbol) = self.graph.graph().node_weight(idx) {
                        component.push(symbol.clone());
                    }
                }
                if !component.is_empty() {
                    components.push(component);
                }
            }
        }

        // Sort by size (largest first)
        components.sort_by_key(|c| std::cmp::Reverse(c.len()));
        components
    }

    /// Perform a level-order traversal (BFS by levels)
    pub fn level_order(&self, start: &str, max_depth: Option<usize>) -> Vec<Vec<Symbol>> {
        let Some(&start_idx) = self.graph.symbol_index().get(start) else {
            return Vec::new();
        };

        let mut levels = Vec::new();
        let mut visited = HashSet::new();
        let mut current_level = vec![start_idx];
        let mut depth = 0;

        while !current_level.is_empty() {
            if let Some(max) = max_depth {
                if depth > max {
                    break;
                }
            }

            let mut level_symbols = Vec::new();
            let mut next_level = Vec::new();

            for node_idx in current_level {
                if visited.contains(&node_idx) {
                    continue;
                }
                visited.insert(node_idx);

                if let Some(symbol) = self.graph.graph().node_weight(node_idx) {
                    level_symbols.push(symbol.clone());
                }

                for edge in self.graph.graph().edges(node_idx) {
                    let target = edge.target();
                    if !visited.contains(&target) {
                        next_level.push(target);
                    }
                }
            }

            if !level_symbols.is_empty() {
                levels.push(level_symbols);
            }

            current_level = next_level;
            depth += 1;
        }

        levels
    }
}

impl KernelGraph {
    /// Get the symbol index for traversal operations
    pub fn symbol_index(&self) -> &std::collections::HashMap<String, NodeIndex> {
        &self.symbol_index
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{CallEdge, SymbolType};

    fn create_test_graph() -> KernelGraph {
        let mut graph = KernelGraph::new();

        // Create a simple graph: A -> B -> C
        //                            -> D -> E
        let symbols = ["func_a", "func_b", "func_c", "func_d", "func_e"];

        for (i, name) in symbols.iter().enumerate() {
            let symbol = Symbol {
                name: name.to_string(),
                file_path: "test.c".to_string(),
                line_number: (i as u32 + 1) * 10,
                symbol_type: SymbolType::Function,
                signature: None,
                config_dependencies: vec![],
            };
            graph.add_symbol(symbol);
        }

        // Create edges
        let edges = [
            ("func_a", "func_b"),
            ("func_b", "func_c"),
            ("func_a", "func_d"),
            ("func_d", "func_e"),
        ];

        for (from, to) in edges.iter() {
            graph
                .add_call(
                    from,
                    to,
                    CallEdge {
                        call_type: CallType::Direct,
                        call_site_line: 100,
                        conditional: false,
                        config_guard: None,
                    },
                )
                .unwrap();
        }

        graph
    }

    #[test]
    fn test_bfs_traversal() {
        let graph = create_test_graph();
        let traversal = GraphTraversal::new(&graph);

        let result = traversal
            .bfs("func_a", TraversalOptions::default())
            .unwrap();

        assert_eq!(result.count, 5); // Should visit all 5 nodes
        assert_eq!(result.visited.len(), 5);
        assert_eq!(result.visit_order[0], "func_a");
        assert!(result.edges.len() >= 4); // At least 4 edges
    }

    #[test]
    fn test_dfs_traversal() {
        let graph = create_test_graph();
        let traversal = GraphTraversal::new(&graph);

        let result = traversal
            .dfs("func_a", TraversalOptions::default())
            .unwrap();

        assert_eq!(result.count, 5); // Should visit all 5 nodes
        assert_eq!(result.visited.len(), 5);
        assert_eq!(result.visit_order[0], "func_a");
    }

    #[test]
    fn test_traversal_with_depth_limit() {
        let graph = create_test_graph();
        let traversal = GraphTraversal::new(&graph);

        let options = TraversalOptions {
            max_depth: Some(1),
            ..Default::default()
        };

        let result = traversal.bfs("func_a", options).unwrap();

        // With depth 1, should visit A, B, and D only
        assert_eq!(result.count, 3);
        assert!(result.visit_order.contains(&"func_a".to_string()));
        assert!(result.visit_order.contains(&"func_b".to_string()));
        assert!(result.visit_order.contains(&"func_d".to_string()));
        assert!(!result.visit_order.contains(&"func_c".to_string()));
        assert!(!result.visit_order.contains(&"func_e".to_string()));
    }

    #[test]
    fn test_find_ancestors() {
        let graph = create_test_graph();
        let traversal = GraphTraversal::new(&graph);

        let ancestors = traversal.find_ancestors("func_c", None);

        // func_c's ancestors are func_b and func_a
        assert_eq!(ancestors.len(), 2);
        let names: Vec<String> = ancestors.iter().map(|s| s.name.clone()).collect();
        assert!(names.contains(&"func_a".to_string()) || names.contains(&"func_b".to_string()));
    }

    #[test]
    fn test_find_descendants() {
        let graph = create_test_graph();
        let traversal = GraphTraversal::new(&graph);

        let descendants = traversal.find_descendants("func_a", None);

        // func_a can reach all other functions
        assert_eq!(descendants.len(), 4);
    }

    #[test]
    fn test_level_order_traversal() {
        let graph = create_test_graph();
        let traversal = GraphTraversal::new(&graph);

        let levels = traversal.level_order("func_a", None);

        assert_eq!(levels.len(), 3); // Three levels: A, (B,D), (C,E)
        assert_eq!(levels[0].len(), 1); // Level 0: just A
        assert_eq!(levels[1].len(), 2); // Level 1: B and D
        assert_eq!(levels[2].len(), 2); // Level 2: C and E
    }

    #[test]
    fn test_topological_sort_with_dag() {
        // Create a DAG (no cycles)
        let graph = create_test_graph();
        let traversal = GraphTraversal::new(&graph);

        let result = traversal.topological_sort();
        assert!(result.is_ok());

        let sorted = result.unwrap();
        assert_eq!(sorted.len(), 5);

        // func_a should come before its dependencies
        let positions: std::collections::HashMap<String, usize> = sorted
            .iter()
            .enumerate()
            .map(|(i, s)| (s.name.clone(), i))
            .collect();

        assert!(positions["func_a"] < positions["func_b"]);
        assert!(positions["func_a"] < positions["func_d"]);
        assert!(positions["func_b"] < positions["func_c"]);
        assert!(positions["func_d"] < positions["func_e"]);
    }

    #[test]
    fn test_bidirectional_search() {
        let graph = create_test_graph();
        let traversal = GraphTraversal::new(&graph);

        // Find path from func_a to func_e
        let path = traversal.bidirectional_search("func_a", "func_e");
        assert!(path.is_some());

        let path = path.unwrap();

        assert_eq!(path[0].name, "func_a");
        assert_eq!(path[path.len() - 1].name, "func_e");
        // Path should be A -> D -> E (length 3)
        assert_eq!(path.len(), 3);
        assert_eq!(path[1].name, "func_d"); // Middle node should be D
    }

    #[test]
    fn test_find_components() {
        let mut graph = KernelGraph::new();

        // Create two disconnected components
        let symbols = ["comp1_a", "comp1_b", "comp2_a", "comp2_b"];

        for name in symbols.iter() {
            let symbol = Symbol {
                name: name.to_string(),
                file_path: "test.c".to_string(),
                line_number: 10,
                symbol_type: SymbolType::Function,
                signature: None,
                config_dependencies: vec![],
            };
            graph.add_symbol(symbol);
        }

        // Component 1: comp1_a <-> comp1_b (cycle)
        graph
            .add_call(
                "comp1_a",
                "comp1_b",
                CallEdge {
                    call_type: CallType::Direct,
                    call_site_line: 100,
                    conditional: false,
                    config_guard: None,
                },
            )
            .unwrap();

        graph
            .add_call(
                "comp1_b",
                "comp1_a",
                CallEdge {
                    call_type: CallType::Direct,
                    call_site_line: 200,
                    conditional: false,
                    config_guard: None,
                },
            )
            .unwrap();

        // Component 2: comp2_a -> comp2_b
        graph
            .add_call(
                "comp2_a",
                "comp2_b",
                CallEdge {
                    call_type: CallType::Direct,
                    call_site_line: 300,
                    conditional: false,
                    config_guard: None,
                },
            )
            .unwrap();

        let traversal = GraphTraversal::new(&graph);
        let components = traversal.find_components();

        // Should have at least 2 components
        assert!(components.len() >= 2);

        // First component should be the cycle (size 2)
        assert_eq!(components[0].len(), 2);
        let names: Vec<String> = components[0].iter().map(|s| s.name.clone()).collect();
        assert!(names.contains(&"comp1_a".to_string()));
        assert!(names.contains(&"comp1_b".to_string()));
    }

    #[test]
    fn test_traversal_with_filters() {
        let mut graph = KernelGraph::new();

        // Create graph with different edge types
        let symbols = ["func_a", "func_b", "func_c"];

        for name in symbols.iter() {
            let symbol = Symbol {
                name: name.to_string(),
                file_path: "test.c".to_string(),
                line_number: 10,
                symbol_type: SymbolType::Function,
                signature: None,
                config_dependencies: vec![],
            };
            graph.add_symbol(symbol);
        }

        // Direct call A -> B
        graph
            .add_call(
                "func_a",
                "func_b",
                CallEdge {
                    call_type: CallType::Direct,
                    call_site_line: 100,
                    conditional: false,
                    config_guard: None,
                },
            )
            .unwrap();

        // Indirect call A -> C
        graph
            .add_call(
                "func_a",
                "func_c",
                CallEdge {
                    call_type: CallType::Indirect,
                    call_site_line: 200,
                    conditional: false,
                    config_guard: None,
                },
            )
            .unwrap();

        let traversal = GraphTraversal::new(&graph);

        // Test filtering by call type (Direct only)
        let options = TraversalOptions {
            call_type_filter: Some(CallType::Direct),
            ..Default::default()
        };

        let result = traversal.bfs("func_a", options).unwrap();

        // Should visit A and B via direct call, but not C (indirect)
        assert_eq!(result.count, 2);
        assert!(result.visit_order.contains(&"func_a".to_string()));
        assert!(result.visit_order.contains(&"func_b".to_string()));
        assert!(!result.visit_order.contains(&"func_c".to_string()));

        // Check that only the direct edge was traversed
        assert_eq!(result.edges.len(), 1);
        assert_eq!(
            result.edges[0],
            ("func_a".to_string(), "func_b".to_string())
        );
    }
}
