//! Cycle detection algorithms for kernel call graphs
//!
//! This module provides efficient algorithms for detecting cycles in the kernel
//! call graph, which are important for:
//! - Identifying potential infinite recursion
//! - Understanding complex dependencies
//! - Detecting potential deadlock scenarios

use crate::{CallGraph, KernelGraph, Symbol};
use anyhow::Result;
use petgraph::algo::{is_cyclic_directed, tarjan_scc};
use petgraph::graph::NodeIndex;
use petgraph::visit::EdgeRef;
use petgraph::Direction;
use serde::{Deserialize, Serialize};
use std::collections::{HashSet, VecDeque};

/// Represents a cycle in the call graph
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Cycle {
    /// Symbols participating in the cycle
    pub symbols: Vec<Symbol>,
    /// The edges forming the cycle (as symbol name pairs)
    pub edges: Vec<(String, String)>,
    /// Whether the cycle involves indirect calls
    pub has_indirect_calls: bool,
    /// Config guards that might prevent the cycle
    pub config_guards: Vec<String>,
}

/// Cycle detection results
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CycleAnalysis {
    /// Whether the graph contains any cycles
    pub has_cycles: bool,
    /// All detected cycles
    pub cycles: Vec<Cycle>,
    /// Strongly connected components (SCCs) in the graph
    pub strongly_connected_components: Vec<Vec<String>>,
    /// Number of nodes involved in cycles
    pub nodes_in_cycles: usize,
}

/// Cycle detector for kernel call graphs
pub struct CycleDetector<'a> {
    graph: &'a KernelGraph,
}

impl<'a> CycleDetector<'a> {
    /// Create a new cycle detector for the given graph
    pub fn new(graph: &'a KernelGraph) -> Self {
        Self { graph }
    }

    /// Perform comprehensive cycle detection analysis
    pub fn analyze(&self) -> Result<CycleAnalysis> {
        let call_graph = self.graph.graph();

        // Quick check if graph has cycles
        let has_cycles = is_cyclic_directed(call_graph);

        if !has_cycles {
            return Ok(CycleAnalysis {
                has_cycles: false,
                cycles: Vec::new(),
                strongly_connected_components: Vec::new(),
                nodes_in_cycles: 0,
            });
        }

        // Find strongly connected components using Tarjan's algorithm
        let sccs = tarjan_scc(call_graph);

        // Convert SCCs to symbol names
        let mut scc_names = Vec::new();
        let mut nodes_in_cycles = 0;

        for scc in &sccs {
            if scc.len() > 1 {
                // SCC with more than one node is a cycle
                let names: Vec<String> = scc
                    .iter()
                    .filter_map(|&idx| call_graph.node_weight(idx))
                    .map(|symbol| symbol.name.clone())
                    .collect();

                nodes_in_cycles += names.len();
                scc_names.push(names);
            } else if scc.len() == 1 {
                // Check for self-loops
                let node_idx = scc[0];
                if self.has_self_loop(call_graph, node_idx) {
                    if let Some(symbol) = call_graph.node_weight(node_idx) {
                        nodes_in_cycles += 1;
                        scc_names.push(vec![symbol.name.clone()]);
                    }
                }
            }
        }

        // Extract individual cycles from SCCs
        let cycles = self.extract_cycles_from_sccs(&sccs)?;

        Ok(CycleAnalysis {
            has_cycles,
            cycles,
            strongly_connected_components: scc_names,
            nodes_in_cycles,
        })
    }

    /// Check if a node has a self-loop
    fn has_self_loop(&self, graph: &CallGraph, node: NodeIndex) -> bool {
        graph
            .edges_directed(node, Direction::Outgoing)
            .any(|edge| edge.target() == node)
    }

    /// Extract individual cycles from strongly connected components
    fn extract_cycles_from_sccs(&self, sccs: &[Vec<NodeIndex>]) -> Result<Vec<Cycle>> {
        let mut cycles = Vec::new();

        for scc in sccs {
            if scc.len() > 1 {
                // Multi-node SCC
                cycles.extend(self.find_cycles_in_scc(scc)?);
            } else if scc.len() == 1 {
                // Check for self-loop
                let node_idx = scc[0];
                if let Some(cycle) = self.extract_self_loop(node_idx)? {
                    cycles.push(cycle);
                }
            }
        }

        Ok(cycles)
    }

    /// Find specific cycles within a strongly connected component
    fn find_cycles_in_scc(&self, scc: &[NodeIndex]) -> Result<Vec<Cycle>> {
        let mut cycles = Vec::new();
        let graph = self.graph.graph();

        // For each node in the SCC, try to find a simple cycle starting from it
        for &start_node in scc {
            if let Some(cycle) = self.find_simple_cycle_from(start_node, scc) {
                // Check if this cycle is unique
                if !self.is_duplicate_cycle(&cycle, &cycles) {
                    cycles.push(cycle);
                }
            }
        }

        // If no simple cycles found, create one from the entire SCC
        if cycles.is_empty() && !scc.is_empty() {
            let mut symbols = Vec::new();
            let mut edges = Vec::new();
            let mut has_indirect_calls = false;
            let mut config_guards = HashSet::new();

            for &node_idx in scc {
                if let Some(symbol) = graph.node_weight(node_idx) {
                    symbols.push(symbol.clone());
                }
            }

            // Find edges within the SCC
            for &from_idx in scc {
                for edge_ref in graph.edges_directed(from_idx, Direction::Outgoing) {
                    let to_idx = edge_ref.target();
                    if scc.contains(&to_idx) {
                        if let (Some(from_sym), Some(to_sym)) =
                            (graph.node_weight(from_idx), graph.node_weight(to_idx))
                        {
                            edges.push((from_sym.name.clone(), to_sym.name.clone()));

                            let edge_data = edge_ref.weight();
                            if matches!(
                                edge_data.call_type,
                                crate::CallType::Indirect | crate::CallType::FunctionPointer
                            ) {
                                has_indirect_calls = true;
                            }
                            if let Some(guard) = &edge_data.config_guard {
                                config_guards.insert(guard.clone());
                            }
                        }
                    }
                }
            }

            cycles.push(Cycle {
                symbols,
                edges,
                has_indirect_calls,
                config_guards: config_guards.into_iter().collect(),
            });
        }

        Ok(cycles)
    }

    /// Find a simple cycle starting from a given node
    fn find_simple_cycle_from(&self, start: NodeIndex, scc: &[NodeIndex]) -> Option<Cycle> {
        let graph = self.graph.graph();
        let mut visited = HashSet::new();
        let mut path = Vec::new();
        let mut path_edges = Vec::new();

        // Use DFS to find a cycle
        if self.dfs_find_cycle(
            start,
            start,
            &mut visited,
            &mut path,
            &mut path_edges,
            scc,
            true,
        ) {
            // Convert path to cycle
            let symbols = path
                .iter()
                .filter_map(|&idx| graph.node_weight(idx).cloned())
                .collect();

            let mut has_indirect_calls = false;
            let mut config_guards = HashSet::new();

            for (from_idx, to_idx) in &path_edges {
                for edge_ref in graph.edges_connecting(*from_idx, *to_idx) {
                    let edge_data = edge_ref.weight();
                    if matches!(
                        edge_data.call_type,
                        crate::CallType::Indirect | crate::CallType::FunctionPointer
                    ) {
                        has_indirect_calls = true;
                    }
                    if let Some(guard) = &edge_data.config_guard {
                        config_guards.insert(guard.clone());
                    }
                }
            }

            let edges = path_edges
                .iter()
                .filter_map(|(from_idx, to_idx)| {
                    match (graph.node_weight(*from_idx), graph.node_weight(*to_idx)) {
                        (Some(from), Some(to)) => Some((from.name.clone(), to.name.clone())),
                        _ => None,
                    }
                })
                .collect();

            Some(Cycle {
                symbols,
                edges,
                has_indirect_calls,
                config_guards: config_guards.into_iter().collect(),
            })
        } else {
            None
        }
    }

    /// DFS helper to find cycles
    #[allow(clippy::too_many_arguments)]
    fn dfs_find_cycle(
        &self,
        current: NodeIndex,
        target: NodeIndex,
        visited: &mut HashSet<NodeIndex>,
        path: &mut Vec<NodeIndex>,
        path_edges: &mut Vec<(NodeIndex, NodeIndex)>,
        scc: &[NodeIndex],
        is_start: bool,
    ) -> bool {
        if !is_start && current == target {
            // Found a cycle back to the start
            return true;
        }

        if visited.contains(&current) {
            return false;
        }

        visited.insert(current);
        path.push(current);

        let graph = self.graph.graph();
        for edge_ref in graph.edges_directed(current, Direction::Outgoing) {
            let next = edge_ref.target();
            if scc.contains(&next) {
                path_edges.push((current, next));

                if self.dfs_find_cycle(next, target, visited, path, path_edges, scc, false) {
                    return true;
                }

                path_edges.pop();
            }
        }

        path.pop();
        visited.remove(&current);
        false
    }

    /// Extract a self-loop cycle
    fn extract_self_loop(&self, node_idx: NodeIndex) -> Result<Option<Cycle>> {
        let graph = self.graph.graph();

        // Check if node has self-loop
        let self_edges: Vec<_> = graph
            .edges_directed(node_idx, Direction::Outgoing)
            .filter(|edge| edge.target() == node_idx)
            .collect();

        if self_edges.is_empty() {
            return Ok(None);
        }

        let symbol = graph
            .node_weight(node_idx)
            .ok_or_else(|| anyhow::anyhow!("Node not found"))?;

        let mut has_indirect_calls = false;
        let mut config_guards = HashSet::new();

        for edge_ref in &self_edges {
            let edge_data = edge_ref.weight();
            if matches!(
                edge_data.call_type,
                crate::CallType::Indirect | crate::CallType::FunctionPointer
            ) {
                has_indirect_calls = true;
            }
            if let Some(guard) = &edge_data.config_guard {
                config_guards.insert(guard.clone());
            }
        }

        Ok(Some(Cycle {
            symbols: vec![symbol.clone()],
            edges: vec![(symbol.name.clone(), symbol.name.clone())],
            has_indirect_calls,
            config_guards: config_guards.into_iter().collect(),
        }))
    }

    /// Check if a cycle is a duplicate of existing cycles
    fn is_duplicate_cycle(&self, cycle: &Cycle, existing: &[Cycle]) -> bool {
        for existing_cycle in existing {
            // Check if they have the same set of symbols
            if cycle.symbols.len() == existing_cycle.symbols.len() {
                let cycle_names: HashSet<_> = cycle.symbols.iter().map(|s| &s.name).collect();
                let existing_names: HashSet<_> =
                    existing_cycle.symbols.iter().map(|s| &s.name).collect();

                if cycle_names == existing_names {
                    return true;
                }
            }
        }
        false
    }

    /// Find all simple paths that form cycles (limited depth for performance)
    pub fn find_all_cycles(&self, max_depth: usize) -> Result<Vec<Cycle>> {
        let mut all_cycles = Vec::new();
        let graph = self.graph.graph();

        // For each node, try to find cycles starting from it
        for node_idx in graph.node_indices() {
            let cycles = self.find_cycles_from_node(node_idx, max_depth)?;
            for cycle in cycles {
                if !self.is_duplicate_cycle(&cycle, &all_cycles) {
                    all_cycles.push(cycle);
                }
            }
        }

        Ok(all_cycles)
    }

    /// Find cycles starting from a specific node with depth limit
    fn find_cycles_from_node(&self, start: NodeIndex, max_depth: usize) -> Result<Vec<Cycle>> {
        let mut cycles = Vec::new();
        let graph = self.graph.graph();

        // BFS to find cycles
        let mut queue = VecDeque::new();
        queue.push_back((start, vec![start], vec![], 0));

        while let Some((current, path, edges, depth)) = queue.pop_front() {
            if depth >= max_depth {
                continue;
            }

            for edge_ref in graph.edges_directed(current, Direction::Outgoing) {
                let next = edge_ref.target();

                if next == start && path.len() > 1 {
                    // Found a cycle
                    let mut cycle_edges = edges.clone();
                    cycle_edges.push((current, next));

                    let symbols = path
                        .iter()
                        .filter_map(|&idx| graph.node_weight(idx).cloned())
                        .collect();

                    let edge_pairs = cycle_edges
                        .iter()
                        .filter_map(|(from_idx, to_idx)| {
                            match (graph.node_weight(*from_idx), graph.node_weight(*to_idx)) {
                                (Some(from), Some(to)) => {
                                    Some((from.name.clone(), to.name.clone()))
                                }
                                _ => None,
                            }
                        })
                        .collect();

                    cycles.push(Cycle {
                        symbols,
                        edges: edge_pairs,
                        has_indirect_calls: false, // Would need to check edge types
                        config_guards: Vec::new(), // Would need to collect guards
                    });
                } else if !path.contains(&next) {
                    // Continue searching
                    let mut new_path = path.clone();
                    new_path.push(next);
                    let mut new_edges = edges.clone();
                    new_edges.push((current, next));

                    queue.push_back((next, new_path, new_edges, depth + 1));
                }
            }
        }

        Ok(cycles)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{CallEdge, CallType, Symbol, SymbolType};

    #[test]
    fn test_no_cycles() {
        let mut graph = KernelGraph::new();

        // Create a simple DAG (no cycles)
        let a = Symbol {
            name: "func_a".to_string(),
            file_path: "test.c".to_string(),
            line_number: 10,
            symbol_type: SymbolType::Function,
            signature: None,
            config_dependencies: vec![],
        };

        let b = Symbol {
            name: "func_b".to_string(),
            file_path: "test.c".to_string(),
            line_number: 20,
            symbol_type: SymbolType::Function,
            signature: None,
            config_dependencies: vec![],
        };

        graph.add_symbol(a);
        graph.add_symbol(b);

        let edge = CallEdge {
            call_type: CallType::Direct,
            call_site_line: 15,
            conditional: false,
            config_guard: None,
        };

        graph.add_call("func_a", "func_b", edge).unwrap();

        let detector = CycleDetector::new(&graph);
        let analysis = detector.analyze().unwrap();

        assert!(!analysis.has_cycles);
        assert_eq!(analysis.cycles.len(), 0);
        assert_eq!(analysis.nodes_in_cycles, 0);
    }

    #[test]
    fn test_simple_cycle() {
        let mut graph = KernelGraph::new();

        // Create a simple cycle: A -> B -> A
        let a = Symbol {
            name: "func_a".to_string(),
            file_path: "test.c".to_string(),
            line_number: 10,
            symbol_type: SymbolType::Function,
            signature: None,
            config_dependencies: vec![],
        };

        let b = Symbol {
            name: "func_b".to_string(),
            file_path: "test.c".to_string(),
            line_number: 20,
            symbol_type: SymbolType::Function,
            signature: None,
            config_dependencies: vec![],
        };

        graph.add_symbol(a.clone());
        graph.add_symbol(b.clone());

        let edge1 = CallEdge {
            call_type: CallType::Direct,
            call_site_line: 15,
            conditional: false,
            config_guard: None,
        };

        let edge2 = CallEdge {
            call_type: CallType::Direct,
            call_site_line: 25,
            conditional: false,
            config_guard: None,
        };

        graph.add_call("func_a", "func_b", edge1).unwrap();
        graph.add_call("func_b", "func_a", edge2).unwrap();

        let detector = CycleDetector::new(&graph);
        let analysis = detector.analyze().unwrap();

        assert!(analysis.has_cycles);
        assert!(!analysis.cycles.is_empty());
        assert_eq!(analysis.nodes_in_cycles, 2);

        // Check that both functions are in the SCC
        assert_eq!(analysis.strongly_connected_components.len(), 1);
        let scc = &analysis.strongly_connected_components[0];
        assert!(scc.contains(&"func_a".to_string()));
        assert!(scc.contains(&"func_b".to_string()));
    }

    #[test]
    fn test_self_loop() {
        let mut graph = KernelGraph::new();

        // Create a self-loop: A -> A
        let a = Symbol {
            name: "recursive_func".to_string(),
            file_path: "test.c".to_string(),
            line_number: 10,
            symbol_type: SymbolType::Function,
            signature: None,
            config_dependencies: vec![],
        };

        graph.add_symbol(a.clone());

        let edge = CallEdge {
            call_type: CallType::Direct,
            call_site_line: 15,
            conditional: false,
            config_guard: None,
        };

        graph
            .add_call("recursive_func", "recursive_func", edge)
            .unwrap();

        let detector = CycleDetector::new(&graph);
        let analysis = detector.analyze().unwrap();

        assert!(analysis.has_cycles);
        assert!(!analysis.cycles.is_empty());
        assert_eq!(analysis.nodes_in_cycles, 1);

        // Check the cycle details
        let cycle = &analysis.cycles[0];
        assert_eq!(cycle.symbols.len(), 1);
        assert_eq!(cycle.symbols[0].name, "recursive_func");
        assert_eq!(cycle.edges.len(), 1);
        assert_eq!(
            cycle.edges[0],
            ("recursive_func".to_string(), "recursive_func".to_string())
        );
    }

    #[test]
    fn test_complex_cycle() {
        let mut graph = KernelGraph::new();

        // Create a more complex cycle: A -> B -> C -> A
        let symbols = ["func_a", "func_b", "func_c"];

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

        // Create the cycle
        let edges = vec![
            ("func_a", "func_b"),
            ("func_b", "func_c"),
            ("func_c", "func_a"),
        ];

        for (caller, callee) in edges {
            let edge = CallEdge {
                call_type: CallType::Direct,
                call_site_line: 100,
                conditional: false,
                config_guard: None,
            };
            graph.add_call(caller, callee, edge).unwrap();
        }

        let detector = CycleDetector::new(&graph);
        let analysis = detector.analyze().unwrap();

        assert!(analysis.has_cycles);
        assert_eq!(analysis.nodes_in_cycles, 3);
        assert_eq!(analysis.strongly_connected_components.len(), 1);
        assert_eq!(analysis.strongly_connected_components[0].len(), 3);
    }

    #[test]
    fn test_cycle_with_config_guards() {
        let mut graph = KernelGraph::new();

        // Create a cycle with config guards
        let a = Symbol {
            name: "func_a".to_string(),
            file_path: "test.c".to_string(),
            line_number: 10,
            symbol_type: SymbolType::Function,
            signature: None,
            config_dependencies: vec!["CONFIG_FEATURE_X".to_string()],
        };

        let b = Symbol {
            name: "func_b".to_string(),
            file_path: "test.c".to_string(),
            line_number: 20,
            symbol_type: SymbolType::Function,
            signature: None,
            config_dependencies: vec!["CONFIG_FEATURE_Y".to_string()],
        };

        graph.add_symbol(a);
        graph.add_symbol(b);

        let edge1 = CallEdge {
            call_type: CallType::Direct,
            call_site_line: 15,
            conditional: false,
            config_guard: Some("CONFIG_FEATURE_X".to_string()),
        };

        let edge2 = CallEdge {
            call_type: CallType::Indirect,
            call_site_line: 25,
            conditional: true,
            config_guard: Some("CONFIG_FEATURE_Y".to_string()),
        };

        graph.add_call("func_a", "func_b", edge1).unwrap();
        graph.add_call("func_b", "func_a", edge2).unwrap();

        let detector = CycleDetector::new(&graph);
        let analysis = detector.analyze().unwrap();

        assert!(analysis.has_cycles);

        // Find the cycle and check its properties
        let cycle = &analysis.cycles[0];
        assert!(cycle.has_indirect_calls);
        assert!(
            cycle
                .config_guards
                .contains(&"CONFIG_FEATURE_X".to_string())
                || cycle
                    .config_guards
                    .contains(&"CONFIG_FEATURE_Y".to_string())
        );
    }

    #[test]
    fn test_multiple_disconnected_cycles() {
        let mut graph = KernelGraph::new();

        // Create two separate cycles: A -> B -> A and C -> D -> C
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

        // First cycle
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
                "func_b",
                "func_a",
                CallEdge {
                    call_type: CallType::Direct,
                    call_site_line: 200,
                    conditional: false,
                    config_guard: None,
                },
            )
            .unwrap();

        // Second cycle
        graph
            .add_call(
                "func_c",
                "func_d",
                CallEdge {
                    call_type: CallType::Direct,
                    call_site_line: 300,
                    conditional: false,
                    config_guard: None,
                },
            )
            .unwrap();

        graph
            .add_call(
                "func_d",
                "func_c",
                CallEdge {
                    call_type: CallType::Direct,
                    call_site_line: 400,
                    conditional: false,
                    config_guard: None,
                },
            )
            .unwrap();

        let detector = CycleDetector::new(&graph);
        let analysis = detector.analyze().unwrap();

        assert!(analysis.has_cycles);
        assert_eq!(analysis.strongly_connected_components.len(), 2);
        assert_eq!(analysis.nodes_in_cycles, 4);
    }

    #[test]
    fn test_find_all_cycles_depth_limited() {
        let mut graph = KernelGraph::new();

        // Create a simple cycle for testing depth-limited search
        let symbols = ["func_a", "func_b"];

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

        graph
            .add_call(
                "func_a",
                "func_b",
                CallEdge {
                    call_type: CallType::Direct,
                    call_site_line: 15,
                    conditional: false,
                    config_guard: None,
                },
            )
            .unwrap();

        graph
            .add_call(
                "func_b",
                "func_a",
                CallEdge {
                    call_type: CallType::Direct,
                    call_site_line: 25,
                    conditional: false,
                    config_guard: None,
                },
            )
            .unwrap();

        let detector = CycleDetector::new(&graph);

        // Test with different depth limits
        let cycles_depth_2 = detector.find_all_cycles(2).unwrap();
        assert!(!cycles_depth_2.is_empty());

        let cycles_depth_1 = detector.find_all_cycles(1).unwrap();
        // With depth 1, we can't find the 2-node cycle
        assert_eq!(cycles_depth_1.len(), 0);
    }

    #[test]
    fn test_nested_cycles() {
        let mut graph = KernelGraph::new();

        // Create nested cycles: A -> B -> C -> A (outer)
        //                            B -> D -> B (inner)
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

        // Outer cycle
        let edges = vec![
            ("func_a", "func_b"),
            ("func_b", "func_c"),
            ("func_c", "func_a"),
            // Inner cycle
            ("func_b", "func_d"),
            ("func_d", "func_b"),
        ];

        for (caller, callee) in edges {
            let edge = CallEdge {
                call_type: CallType::Direct,
                call_site_line: 100,
                conditional: false,
                config_guard: None,
            };
            graph.add_call(caller, callee, edge).unwrap();
        }

        let detector = CycleDetector::new(&graph);
        let analysis = detector.analyze().unwrap();

        assert!(analysis.has_cycles);
        // All 4 functions are part of cycles
        assert_eq!(analysis.nodes_in_cycles, 4);
        // Should detect the strongly connected component
        assert!(!analysis.strongly_connected_components.is_empty());
    }

    #[test]
    fn test_cycle_with_function_pointers() {
        let mut graph = KernelGraph::new();

        let a = Symbol {
            name: "callback_a".to_string(),
            file_path: "test.c".to_string(),
            line_number: 10,
            symbol_type: SymbolType::Function,
            signature: None,
            config_dependencies: vec![],
        };

        let b = Symbol {
            name: "callback_b".to_string(),
            file_path: "test.c".to_string(),
            line_number: 20,
            symbol_type: SymbolType::Function,
            signature: None,
            config_dependencies: vec![],
        };

        graph.add_symbol(a);
        graph.add_symbol(b);

        // Create a cycle with function pointer calls
        let edge1 = CallEdge {
            call_type: CallType::FunctionPointer,
            call_site_line: 15,
            conditional: false,
            config_guard: None,
        };

        let edge2 = CallEdge {
            call_type: CallType::FunctionPointer,
            call_site_line: 25,
            conditional: false,
            config_guard: None,
        };

        graph.add_call("callback_a", "callback_b", edge1).unwrap();
        graph.add_call("callback_b", "callback_a", edge2).unwrap();

        let detector = CycleDetector::new(&graph);
        let analysis = detector.analyze().unwrap();

        assert!(analysis.has_cycles);
        // Check that indirect calls are detected
        let cycle = &analysis.cycles[0];
        assert!(cycle.has_indirect_calls);
    }

    #[test]
    fn test_large_cycle() {
        let mut graph = KernelGraph::new();

        // Create a large cycle with 10 functions
        let num_functions = 10;
        for i in 0..num_functions {
            let symbol = Symbol {
                name: format!("func_{}", i),
                file_path: "test.c".to_string(),
                line_number: (i as u32 + 1) * 10,
                symbol_type: SymbolType::Function,
                signature: None,
                config_dependencies: vec![],
            };
            graph.add_symbol(symbol);
        }

        // Create cycle: func_0 -> func_1 -> ... -> func_9 -> func_0
        for i in 0..num_functions {
            let next = (i + 1) % num_functions;
            let edge = CallEdge {
                call_type: CallType::Direct,
                call_site_line: 100 + i as u32,
                conditional: false,
                config_guard: None,
            };
            graph
                .add_call(&format!("func_{}", i), &format!("func_{}", next), edge)
                .unwrap();
        }

        let detector = CycleDetector::new(&graph);
        let analysis = detector.analyze().unwrap();

        assert!(analysis.has_cycles);
        assert_eq!(analysis.nodes_in_cycles, num_functions);
        // Should have one large SCC
        assert_eq!(analysis.strongly_connected_components.len(), 1);
        assert_eq!(
            analysis.strongly_connected_components[0].len(),
            num_functions
        );
    }

    #[test]
    fn test_conditional_cycle() {
        let mut graph = KernelGraph::new();

        let symbols = ["func_a", "func_b"];
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

        // Create a cycle where all edges are conditional
        let edge1 = CallEdge {
            call_type: CallType::Direct,
            call_site_line: 15,
            conditional: true,
            config_guard: Some("CONFIG_DEBUG".to_string()),
        };

        let edge2 = CallEdge {
            call_type: CallType::Direct,
            call_site_line: 25,
            conditional: true,
            config_guard: Some("CONFIG_VERBOSE".to_string()),
        };

        graph.add_call("func_a", "func_b", edge1).unwrap();
        graph.add_call("func_b", "func_a", edge2).unwrap();

        let detector = CycleDetector::new(&graph);
        let analysis = detector.analyze().unwrap();

        assert!(analysis.has_cycles);

        // Verify config guards are captured
        let cycle = &analysis.cycles[0];
        assert!(cycle.config_guards.contains(&"CONFIG_DEBUG".to_string()));
        assert!(cycle.config_guards.contains(&"CONFIG_VERBOSE".to_string()));
    }

    #[test]
    fn test_cycle_detection_with_external_edges() {
        let mut graph = KernelGraph::new();

        // Create cycle A -> B -> A with external node C -> A
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

        let edge = CallEdge {
            call_type: CallType::Direct,
            call_site_line: 100,
            conditional: false,
            config_guard: None,
        };

        // Create cycle
        graph.add_call("func_a", "func_b", edge.clone()).unwrap();
        graph.add_call("func_b", "func_a", edge.clone()).unwrap();
        // External edge
        graph.add_call("func_c", "func_a", edge).unwrap();

        let detector = CycleDetector::new(&graph);
        let analysis = detector.analyze().unwrap();

        assert!(analysis.has_cycles);
        // Only A and B should be in cycle
        assert_eq!(analysis.nodes_in_cycles, 2);

        // C should not be in any SCC
        for scc in &analysis.strongly_connected_components {
            assert!(!scc.contains(&"func_c".to_string()));
        }
    }
}
