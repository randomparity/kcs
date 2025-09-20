use anyhow::Result;
use indexmap::IndexMap;
use petgraph::graph::{EdgeIndex, NodeIndex};
use petgraph::visit::EdgeRef;
use petgraph::{Direction, Graph};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

pub mod builder;
pub mod call_edge;
pub mod call_site;
pub mod config;
pub mod cycles;
pub mod queries;
pub mod traversal;
pub mod types;

pub use call_edge::{CallEdge as CallEdgeModel, CallEdgeBuilder};
pub use call_site::{CallSite, CallSiteBuilder};
pub use cycles::{Cycle, CycleAnalysis, CycleDetector};
pub use traversal::{GraphTraversal, TraversalOptions, TraversalResult};
pub use types::{AnalysisScope, CallType as CallTypeEnum, ConfidenceLevel, PointerType};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Symbol {
    pub name: String,
    pub file_path: String,
    pub line_number: u32,
    pub symbol_type: SymbolType,
    pub signature: Option<String>,
    pub config_dependencies: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum SymbolType {
    Function,
    Variable,
    Macro,
    Type,
    Constant,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CallEdge {
    pub call_type: CallType,
    pub call_site_line: u32,
    pub conditional: bool,
    pub config_guard: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum CallType {
    Direct,
    Indirect,
    FunctionPointer,
    Macro,
}

pub type CallGraph = Graph<Symbol, CallEdge>;

#[derive(Debug, Clone)]
pub struct KernelGraph {
    graph: CallGraph,
    symbol_index: HashMap<String, NodeIndex>,
    config_symbols: IndexMap<String, Vec<NodeIndex>>,
}

impl KernelGraph {
    pub fn new() -> Self {
        Self {
            graph: Graph::new(),
            symbol_index: HashMap::new(),
            config_symbols: IndexMap::new(),
        }
    }

    pub fn add_symbol(&mut self, symbol: Symbol) -> NodeIndex {
        if let Some(&existing_idx) = self.symbol_index.get(&symbol.name) {
            return existing_idx;
        }

        let node_idx = self.graph.add_node(symbol.clone());
        self.symbol_index.insert(symbol.name.clone(), node_idx);

        // Index by config dependencies
        for config in &symbol.config_dependencies {
            self.config_symbols.entry(config.clone()).or_default().push(node_idx);
        }

        node_idx
    }

    pub fn add_call(&mut self, caller: &str, callee: &str, edge: CallEdge) -> Result<EdgeIndex> {
        let caller_idx = self
            .symbol_index
            .get(caller)
            .ok_or_else(|| anyhow::anyhow!("Caller symbol not found: {}", caller))?;
        let callee_idx = self
            .symbol_index
            .get(callee)
            .ok_or_else(|| anyhow::anyhow!("Callee symbol not found: {}", callee))?;

        Ok(self.graph.add_edge(*caller_idx, *callee_idx, edge))
    }

    pub fn get_symbol(&self, name: &str) -> Option<&Symbol> {
        self.symbol_index.get(name).and_then(|&idx| self.graph.node_weight(idx))
    }

    pub fn find_callers(&self, symbol_name: &str) -> Vec<&Symbol> {
        if let Some(&target_idx) = self.symbol_index.get(symbol_name) {
            self.graph
                .edges_directed(target_idx, Direction::Incoming)
                .filter_map(|edge| self.graph.node_weight(edge.source()))
                .collect()
        } else {
            Vec::new()
        }
    }

    pub fn find_callees(&self, symbol_name: &str) -> Vec<&Symbol> {
        if let Some(&source_idx) = self.symbol_index.get(symbol_name) {
            self.graph
                .edges_directed(source_idx, Direction::Outgoing)
                .filter_map(|edge| self.graph.node_weight(edge.target()))
                .collect()
        } else {
            Vec::new()
        }
    }

    pub fn get_call_path(&self, from: &str, to: &str) -> Option<Vec<&Symbol>> {
        use petgraph::algo::astar;

        let from_idx = *self.symbol_index.get(from)?;
        let to_idx = *self.symbol_index.get(to)?;

        // Use A* algorithm which returns the path
        let result = astar(
            &self.graph,
            from_idx,
            |node| node == to_idx,
            |_| 1, // edge cost
            |_| 0, // heuristic (0 makes it equivalent to Dijkstra)
        );

        if let Some((_, path)) = result {
            // Convert node indices to symbols
            let mut symbols = Vec::new();
            for node_idx in path {
                if let Some(symbol) = self.graph.node_weight(node_idx) {
                    symbols.push(symbol);
                }
            }
            if !symbols.is_empty() {
                Some(symbols)
            } else {
                None
            }
        } else {
            None
        }
    }

    /// Find all paths between two symbols up to a maximum length
    pub fn get_all_paths(&self, from: &str, to: &str, max_length: usize) -> Vec<Vec<&Symbol>> {
        use std::collections::VecDeque;

        let Some(&from_idx) = self.symbol_index.get(from) else {
            return Vec::new();
        };
        let Some(&to_idx) = self.symbol_index.get(to) else {
            return Vec::new();
        };

        let mut all_paths = Vec::new();
        let mut queue = VecDeque::new();

        // Start with the source node
        queue.push_back((from_idx, vec![from_idx], 0));

        while let Some((current, path, length)) = queue.pop_front() {
            if length >= max_length {
                continue;
            }

            for edge in self.graph.edges(current) {
                let next = edge.target();

                if next == to_idx {
                    // Found a path to the target
                    let mut full_path = path.clone();
                    full_path.push(next);

                    // Convert to symbols
                    let symbol_path: Vec<&Symbol> =
                        full_path.iter().filter_map(|&idx| self.graph.node_weight(idx)).collect();

                    if symbol_path.len() == full_path.len() {
                        all_paths.push(symbol_path);
                    }
                } else if !path.contains(&next) {
                    // Continue searching if not in current path (avoid cycles)
                    let mut new_path = path.clone();
                    new_path.push(next);
                    queue.push_back((next, new_path, length + 1));
                }
            }
        }

        all_paths
    }

    /// Get the shortest path length between two symbols
    pub fn get_path_length(&self, from: &str, to: &str) -> Option<usize> {
        use petgraph::algo::dijkstra;

        let from_idx = *self.symbol_index.get(from)?;
        let to_idx = *self.symbol_index.get(to)?;

        let paths = dijkstra(&self.graph, from_idx, Some(to_idx), |_| 1);

        paths.get(&to_idx).map(|&cost| cost as usize)
    }

    /// Check if there's a path between two symbols
    pub fn has_path(&self, from: &str, to: &str) -> bool {
        self.get_path_length(from, to).is_some()
    }

    /// Get all symbols reachable from a given symbol within a maximum depth
    pub fn get_reachable_symbols(&self, from: &str, max_depth: usize) -> Vec<&Symbol> {
        use petgraph::algo::dijkstra;

        let Some(&from_idx) = self.symbol_index.get(from) else {
            return Vec::new();
        };

        let paths = dijkstra(&self.graph, from_idx, None, |_| 1);

        let mut reachable = Vec::new();
        for (node_idx, cost) in paths {
            if cost <= max_depth as i32 {
                if let Some(symbol) = self.graph.node_weight(node_idx) {
                    reachable.push(symbol);
                }
            }
        }

        reachable
    }

    pub fn symbols_by_config(&self, config: &str) -> Option<&Vec<NodeIndex>> {
        self.config_symbols.get(config)
    }

    pub fn graph(&self) -> &CallGraph {
        &self.graph
    }

    pub fn symbol_count(&self) -> usize {
        self.graph.node_count()
    }

    pub fn call_count(&self) -> usize {
        self.graph.edge_count()
    }
}

impl Default for KernelGraph {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_graph_creation() {
        let mut graph = KernelGraph::new();

        let symbol1 = Symbol {
            name: "test_func".to_string(),
            file_path: "test.c".to_string(),
            line_number: 42,
            symbol_type: SymbolType::Function,
            signature: Some("int test_func(void)".to_string()),
            config_dependencies: vec!["CONFIG_TEST".to_string()],
        };

        let _idx = graph.add_symbol(symbol1);
        assert_eq!(graph.symbol_count(), 1);

        let retrieved = graph.get_symbol("test_func");
        assert!(retrieved.is_some());
        assert_eq!(retrieved.unwrap().line_number, 42);
    }

    #[test]
    fn test_call_relationships() {
        let mut graph = KernelGraph::new();

        let caller = Symbol {
            name: "caller_func".to_string(),
            file_path: "test.c".to_string(),
            line_number: 10,
            symbol_type: SymbolType::Function,
            signature: None,
            config_dependencies: vec![],
        };

        let callee = Symbol {
            name: "callee_func".to_string(),
            file_path: "test.c".to_string(),
            line_number: 20,
            symbol_type: SymbolType::Function,
            signature: None,
            config_dependencies: vec![],
        };

        graph.add_symbol(caller);
        graph.add_symbol(callee);

        let edge = CallEdge {
            call_type: CallType::Direct,
            call_site_line: 15,
            conditional: false,
            config_guard: None,
        };

        graph.add_call("caller_func", "callee_func", edge).unwrap();

        let callees = graph.find_callees("caller_func");
        assert_eq!(callees.len(), 1);
        assert_eq!(callees[0].name, "callee_func");

        let callers = graph.find_callers("callee_func");
        assert_eq!(callers.len(), 1);
        assert_eq!(callers[0].name, "caller_func");
    }

    #[test]
    fn test_path_reconstruction() {
        let mut graph = KernelGraph::new();

        // Create a chain: A -> B -> C -> D
        let symbols = ["func_a", "func_b", "func_c", "func_d"];

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

        // Create the chain
        let edges = [("func_a", "func_b"), ("func_b", "func_c"), ("func_c", "func_d")];

        for (caller, callee) in edges.iter() {
            let edge = CallEdge {
                call_type: CallType::Direct,
                call_site_line: 100,
                conditional: false,
                config_guard: None,
            };
            graph.add_call(caller, callee, edge).unwrap();
        }

        // Test path from A to D
        let path = graph.get_call_path("func_a", "func_d");
        assert!(path.is_some());

        let path = path.unwrap();
        assert_eq!(path.len(), 4);
        assert_eq!(path[0].name, "func_a");
        assert_eq!(path[1].name, "func_b");
        assert_eq!(path[2].name, "func_c");
        assert_eq!(path[3].name, "func_d");

        // Test path from B to D
        let path = graph.get_call_path("func_b", "func_d");
        assert!(path.is_some());
        let path = path.unwrap();
        assert_eq!(path.len(), 3);
        assert_eq!(path[0].name, "func_b");
        assert_eq!(path[1].name, "func_c");
        assert_eq!(path[2].name, "func_d");

        // Test non-existent path
        let path = graph.get_call_path("func_d", "func_a");
        assert!(path.is_none());
    }

    #[test]
    fn test_all_paths() {
        let mut graph = KernelGraph::new();

        // Create a diamond: A -> B -> D
        //                    A -> C -> D
        let symbols = ["func_a", "func_b", "func_c", "func_d"];

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

        // Create the diamond
        let edges = [
            ("func_a", "func_b"),
            ("func_a", "func_c"),
            ("func_b", "func_d"),
            ("func_c", "func_d"),
        ];

        for (caller, callee) in edges.iter() {
            let edge = CallEdge {
                call_type: CallType::Direct,
                call_site_line: 100,
                conditional: false,
                config_guard: None,
            };
            graph.add_call(caller, callee, edge).unwrap();
        }

        // Find all paths from A to D
        let paths = graph.get_all_paths("func_a", "func_d", 5);
        assert_eq!(paths.len(), 2);

        // Both paths should have 3 nodes
        for path in &paths {
            assert_eq!(path.len(), 3);
            assert_eq!(path[0].name, "func_a");
            assert_eq!(path[2].name, "func_d");
            // Middle node should be either B or C
            assert!(path[1].name == "func_b" || path[1].name == "func_c");
        }
    }

    #[test]
    fn test_path_length() {
        let mut graph = KernelGraph::new();

        // Create a chain with branch
        let symbols = ["func_a", "func_b", "func_c", "func_d"];

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

        // Create edges: A -> B -> C -> D
        //               A ---------> D (direct shortcut)
        let edges = [
            ("func_a", "func_b"),
            ("func_b", "func_c"),
            ("func_c", "func_d"),
            ("func_a", "func_d"), // Direct shortcut
        ];

        for (caller, callee) in edges.iter() {
            let edge = CallEdge {
                call_type: CallType::Direct,
                call_site_line: 100,
                conditional: false,
                config_guard: None,
            };
            graph.add_call(caller, callee, edge).unwrap();
        }

        // Shortest path from A to D should be 1 (direct)
        let length = graph.get_path_length("func_a", "func_d");
        assert_eq!(length, Some(1));

        // Path from A to C should be 2
        let length = graph.get_path_length("func_a", "func_c");
        assert_eq!(length, Some(2));

        // No path from D to A
        let length = graph.get_path_length("func_d", "func_a");
        assert_eq!(length, None);
    }

    #[test]
    fn test_has_path() {
        let mut graph = KernelGraph::new();

        // Create disconnected components
        let symbols = ["func_a", "func_b", "func_c", "func_d"];

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

        // Create two disconnected pairs: A -> B and C -> D
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

        graph
            .add_call(
                "func_c",
                "func_d",
                CallEdge {
                    call_type: CallType::Direct,
                    call_site_line: 200,
                    conditional: false,
                    config_guard: None,
                },
            )
            .unwrap();

        // Path exists within components
        assert!(graph.has_path("func_a", "func_b"));
        assert!(graph.has_path("func_c", "func_d"));

        // No path between components
        assert!(!graph.has_path("func_a", "func_c"));
        assert!(!graph.has_path("func_a", "func_d"));
        assert!(!graph.has_path("func_b", "func_c"));
        assert!(!graph.has_path("func_b", "func_d"));
    }

    #[test]
    fn test_reachable_symbols() {
        let mut graph = KernelGraph::new();

        // Create a tree structure
        let symbols = ["root", "child1", "child2", "grandchild1", "grandchild2"];

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

        // Create tree edges
        let edges = [
            ("root", "child1"),
            ("root", "child2"),
            ("child1", "grandchild1"),
            ("child2", "grandchild2"),
        ];

        for (caller, callee) in edges.iter() {
            let edge = CallEdge {
                call_type: CallType::Direct,
                call_site_line: 100,
                conditional: false,
                config_guard: None,
            };
            graph.add_call(caller, callee, edge).unwrap();
        }

        // Get reachable symbols from root with depth 1
        let reachable = graph.get_reachable_symbols("root", 1);
        assert_eq!(reachable.len(), 3); // root, child1, child2

        let names: Vec<String> = reachable.iter().map(|s| s.name.clone()).collect();
        assert!(names.contains(&"root".to_string()));
        assert!(names.contains(&"child1".to_string()));
        assert!(names.contains(&"child2".to_string()));

        // Get reachable symbols from root with depth 2
        let reachable = graph.get_reachable_symbols("root", 2);
        assert_eq!(reachable.len(), 5); // All symbols

        // Get reachable from child1
        let reachable = graph.get_reachable_symbols("child1", 1);
        assert_eq!(reachable.len(), 2); // child1 and grandchild1
    }

    #[test]
    fn test_empty_graph() {
        let graph = KernelGraph::new();
        assert_eq!(graph.symbol_count(), 0);
        assert_eq!(graph.call_count(), 0);
        assert!(graph.get_symbol("nonexistent").is_none());
        assert!(graph.find_callers("nonexistent").is_empty());
        assert!(graph.find_callees("nonexistent").is_empty());
    }

    #[test]
    fn test_symbol_duplicate_handling() {
        let mut graph = KernelGraph::new();

        let symbol1 = Symbol {
            name: "duplicate".to_string(),
            file_path: "file1.c".to_string(),
            line_number: 10,
            symbol_type: SymbolType::Function,
            signature: None,
            config_dependencies: vec![],
        };

        let symbol2 = Symbol {
            name: "duplicate".to_string(),
            file_path: "file2.c".to_string(),
            line_number: 20,
            symbol_type: SymbolType::Function,
            signature: None,
            config_dependencies: vec![],
        };

        let idx1 = graph.add_symbol(symbol1);
        let idx2 = graph.add_symbol(symbol2.clone());

        // Should reuse the same index for duplicate name
        assert_eq!(idx1, idx2);

        // Check that the symbol was NOT updated (first one wins)
        let retrieved = graph.get_symbol("duplicate").unwrap();
        assert_eq!(retrieved.file_path, "file1.c");
        assert_eq!(retrieved.line_number, 10);
    }

    #[test]
    fn test_invalid_edge_handling() {
        let mut graph = KernelGraph::new();

        let edge = CallEdge {
            call_type: CallType::Direct,
            call_site_line: 100,
            conditional: false,
            config_guard: None,
        };

        // Try to add edge between non-existent symbols
        let result = graph.add_call("nonexistent1", "nonexistent2", edge);
        assert!(result.is_err());
    }

    #[test]
    fn test_path_with_conditional_edges() {
        let mut graph = KernelGraph::new();

        let symbols = ["start", "middle", "end"];
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

        // Add conditional edge from start to middle
        let edge1 = CallEdge {
            call_type: CallType::Direct,
            call_site_line: 100,
            conditional: true,
            config_guard: Some("CONFIG_FEATURE".to_string()),
        };
        graph.add_call("start", "middle", edge1).unwrap();

        // Add unconditional edge from middle to end
        let edge2 = CallEdge {
            call_type: CallType::Direct,
            call_site_line: 200,
            conditional: false,
            config_guard: None,
        };
        graph.add_call("middle", "end", edge2).unwrap();

        // Path should still exist despite conditional edge
        assert!(graph.has_path("start", "end"));
        assert_eq!(graph.get_path_length("start", "end"), Some(2));
    }

    #[test]
    fn test_symbols_by_config() {
        let mut graph = KernelGraph::new();

        let symbol1 = Symbol {
            name: "func1".to_string(),
            file_path: "test.c".to_string(),
            line_number: 10,
            symbol_type: SymbolType::Function,
            signature: None,
            config_dependencies: vec!["CONFIG_A".to_string()],
        };

        let symbol2 = Symbol {
            name: "func2".to_string(),
            file_path: "test.c".to_string(),
            line_number: 20,
            symbol_type: SymbolType::Function,
            signature: None,
            config_dependencies: vec!["CONFIG_B".to_string()],
        };

        let symbol3 = Symbol {
            name: "func3".to_string(),
            file_path: "test.c".to_string(),
            line_number: 30,
            symbol_type: SymbolType::Function,
            signature: None,
            config_dependencies: vec!["CONFIG_A".to_string()],
        };

        graph.add_symbol(symbol1);
        graph.add_symbol(symbol2);
        graph.add_symbol(symbol3);

        // Check symbols grouped by config
        let config_a_symbols = graph.symbols_by_config("CONFIG_A");
        assert!(config_a_symbols.is_some());
        assert_eq!(config_a_symbols.unwrap().len(), 2);

        let config_b_symbols = graph.symbols_by_config("CONFIG_B");
        assert!(config_b_symbols.is_some());
        assert_eq!(config_b_symbols.unwrap().len(), 1);

        let config_c_symbols = graph.symbols_by_config("CONFIG_C");
        assert!(config_c_symbols.is_none());
    }

    #[test]
    fn test_complex_multi_path_scenario() {
        let mut graph = KernelGraph::new();

        // Create a diamond pattern: A -> B -> D
        //                           A -> C -> D
        let symbols = ["A", "B", "C", "D"];
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

        let edges = [("A", "B"), ("A", "C"), ("B", "D"), ("C", "D")];
        for (caller, callee) in edges.iter() {
            let edge = CallEdge {
                call_type: CallType::Direct,
                call_site_line: 100,
                conditional: false,
                config_guard: None,
            };
            graph.add_call(caller, callee, edge).unwrap();
        }

        // Should find 2 paths from A to D
        let paths = graph.get_all_paths("A", "D", 3);
        assert_eq!(paths.len(), 2);

        // Both paths should have length 2
        for path in paths {
            assert_eq!(path.len(), 3); // A + intermediate + D
        }

        // Path length should be the shortest
        assert_eq!(graph.get_path_length("A", "D"), Some(2));
    }
}
