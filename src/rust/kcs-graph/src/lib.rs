use anyhow::Result;
use indexmap::IndexMap;
use petgraph::graph::{EdgeIndex, NodeIndex};
use petgraph::visit::EdgeRef;
use petgraph::{Direction, Graph};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

pub mod builder;
pub mod config;
pub mod queries;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Symbol {
    pub name: String,
    pub file_path: String,
    pub line_number: u32,
    pub symbol_type: SymbolType,
    pub signature: Option<String>,
    pub config_dependencies: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
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

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum CallType {
    Direct,
    Indirect,
    FunctionPointer,
    Macro,
}

pub type CallGraph = Graph<Symbol, CallEdge>;

#[derive(Debug)]
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
            self.config_symbols
                .entry(config.clone())
                .or_default()
                .push(node_idx);
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
        self.symbol_index
            .get(name)
            .and_then(|&idx| self.graph.node_weight(idx))
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
        use petgraph::algo::dijkstra;

        let from_idx = self.symbol_index.get(from)?;
        let to_idx = self.symbol_index.get(to)?;

        let paths = dijkstra(&self.graph, *from_idx, Some(*to_idx), |_| 1);

        if paths.contains_key(to_idx) {
            // For now, just return direct path nodes
            // TODO: Implement actual path reconstruction
            Some(vec![
                self.graph.node_weight(*from_idx)?,
                self.graph.node_weight(*to_idx)?,
            ])
        } else {
            None
        }
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
}
