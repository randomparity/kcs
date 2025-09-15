use crate::{KernelGraph, Symbol};
use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet, VecDeque};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QueryResult {
    pub query_type: String,
    pub target_symbol: String,
    pub results: Vec<SymbolResult>,
    pub metadata: QueryMetadata,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SymbolResult {
    pub symbol: Symbol,
    pub call_chain: Option<Vec<String>>,
    pub config_active: bool,
    pub distance: Option<u32>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QueryMetadata {
    pub total_results: usize,
    pub depth_searched: u32,
    pub config_context: HashMap<String, bool>,
    pub execution_time_ms: u64,
}

pub struct QueryEngine<'a> {
    graph: &'a KernelGraph,
    config_context: HashMap<String, bool>,
}

impl<'a> QueryEngine<'a> {
    pub fn new(graph: &'a KernelGraph) -> Self {
        Self {
            graph,
            config_context: HashMap::new(),
        }
    }

    pub fn with_config_context(mut self, config: HashMap<String, bool>) -> Self {
        self.config_context = config;
        self
    }

    pub fn who_calls(&self, symbol_name: &str, max_depth: Option<u32>) -> Result<QueryResult> {
        let start_time = std::time::Instant::now();
        let max_depth = max_depth.unwrap_or(10);

        let mut results = Vec::new();
        let mut visited = HashSet::new();
        let mut queue = VecDeque::new();

        // Find the target symbol
        if let Some(_target_symbol) = self.graph.get_symbol(symbol_name) {
            queue.push_back((symbol_name.to_string(), vec![], 0));

            while let Some((current_symbol, call_chain, depth)) = queue.pop_front() {
                if depth >= max_depth || visited.contains(&current_symbol) {
                    continue;
                }

                visited.insert(current_symbol.clone());

                // Find all callers of current symbol
                let callers = self.graph.find_callers(&current_symbol);

                for caller in callers {
                    let mut new_call_chain = call_chain.clone();
                    new_call_chain.push(caller.name.clone());

                    let config_active = self.is_symbol_active(caller);

                    results.push(SymbolResult {
                        symbol: caller.clone(),
                        call_chain: if new_call_chain.is_empty() {
                            None
                        } else {
                            Some(new_call_chain.clone())
                        },
                        config_active,
                        distance: Some(depth + 1),
                    });

                    // Continue searching from this caller
                    if depth + 1 < max_depth {
                        queue.push_back((caller.name.clone(), new_call_chain, depth + 1));
                    }
                }
            }
        }

        let execution_time = start_time.elapsed().as_millis() as u64;

        let total_results = results.len();
        Ok(QueryResult {
            query_type: "who_calls".to_string(),
            target_symbol: symbol_name.to_string(),
            results,
            metadata: QueryMetadata {
                total_results,
                depth_searched: max_depth,
                config_context: self.config_context.clone(),
                execution_time_ms: execution_time,
            },
        })
    }

    pub fn list_dependencies(
        &self,
        symbol_name: &str,
        max_depth: Option<u32>,
    ) -> Result<QueryResult> {
        let start_time = std::time::Instant::now();
        let max_depth = max_depth.unwrap_or(10);

        let mut results = Vec::new();
        let mut visited = HashSet::new();
        let mut queue = VecDeque::new();

        if let Some(_target_symbol) = self.graph.get_symbol(symbol_name) {
            queue.push_back((symbol_name.to_string(), vec![], 0));

            while let Some((current_symbol, call_chain, depth)) = queue.pop_front() {
                if depth >= max_depth || visited.contains(&current_symbol) {
                    continue;
                }

                visited.insert(current_symbol.clone());

                // Find all callees of current symbol
                let callees = self.graph.find_callees(&current_symbol);

                for callee in callees {
                    let mut new_call_chain = call_chain.clone();
                    new_call_chain.push(callee.name.clone());

                    let config_active = self.is_symbol_active(callee);

                    results.push(SymbolResult {
                        symbol: callee.clone(),
                        call_chain: if new_call_chain.is_empty() {
                            None
                        } else {
                            Some(new_call_chain.clone())
                        },
                        config_active,
                        distance: Some(depth + 1),
                    });

                    // Continue searching from this callee
                    if depth + 1 < max_depth {
                        queue.push_back((callee.name.clone(), new_call_chain, depth + 1));
                    }
                }
            }
        }

        let execution_time = start_time.elapsed().as_millis() as u64;

        let total_results = results.len();
        Ok(QueryResult {
            query_type: "list_dependencies".to_string(),
            target_symbol: symbol_name.to_string(),
            results,
            metadata: QueryMetadata {
                total_results,
                depth_searched: max_depth,
                config_context: self.config_context.clone(),
                execution_time_ms: execution_time,
            },
        })
    }

    pub fn find_path(
        &self,
        from: &str,
        to: &str,
        max_depth: Option<u32>,
    ) -> Result<Option<Vec<String>>> {
        let max_depth = max_depth.unwrap_or(20);

        let mut queue = VecDeque::new();
        let mut visited = HashSet::new();
        let mut parent_map: HashMap<String, String> = HashMap::new();

        queue.push_back((from.to_string(), 0));
        visited.insert(from.to_string());

        while let Some((current, depth)) = queue.pop_front() {
            if current == to {
                // Reconstruct path
                let mut path = Vec::new();
                let mut current_node = to.to_string();

                while let Some(parent) = parent_map.get(&current_node) {
                    path.push(current_node.clone());
                    current_node = parent.clone();
                }
                path.push(from.to_string());
                path.reverse();

                return Ok(Some(path));
            }

            if depth >= max_depth {
                continue;
            }

            // Explore callees
            let callees = self.graph.find_callees(&current);
            for callee in callees {
                if !visited.contains(&callee.name) {
                    visited.insert(callee.name.clone());
                    parent_map.insert(callee.name.clone(), current.clone());
                    queue.push_back((callee.name.clone(), depth + 1));
                }
            }
        }

        Ok(None)
    }

    pub fn analyze_impact(
        &self,
        symbol_name: &str,
        change_type: ChangeType,
    ) -> Result<ImpactAnalysis> {
        let start_time = std::time::Instant::now();

        let mut affected_symbols = HashSet::new();
        let mut high_impact = Vec::new();
        let mut medium_impact = Vec::new();
        let mut low_impact = Vec::new();

        // Find all symbols that could be affected by changes to this symbol
        match change_type {
            ChangeType::SignatureChange => {
                // All direct callers are high impact
                let callers = self.graph.find_callers(symbol_name);
                for caller in callers {
                    affected_symbols.insert(caller.name.clone());
                    high_impact.push(caller.clone());
                }
            }
            ChangeType::BehaviorChange => {
                // Direct callers are medium impact, indirect are low
                let direct_callers = self.graph.find_callers(symbol_name);
                for caller in direct_callers {
                    affected_symbols.insert(caller.name.clone());
                    medium_impact.push(caller.clone());

                    // Indirect callers are low impact
                    let indirect_callers = self.graph.find_callers(&caller.name);
                    for indirect in indirect_callers {
                        if !affected_symbols.contains(&indirect.name) {
                            affected_symbols.insert(indirect.name.clone());
                            low_impact.push(indirect.clone());
                        }
                    }
                }
            }
            ChangeType::Deletion => {
                // All direct callers are high impact (broken)
                let callers = self.graph.find_callers(symbol_name);
                for caller in callers {
                    affected_symbols.insert(caller.name.clone());
                    high_impact.push(caller.clone());
                }
            }
        }

        let execution_time = start_time.elapsed().as_millis() as u64;

        Ok(ImpactAnalysis {
            target_symbol: symbol_name.to_string(),
            change_type,
            high_impact,
            medium_impact,
            low_impact,
            total_affected: affected_symbols.len(),
            execution_time_ms: execution_time,
        })
    }

    fn is_symbol_active(&self, symbol: &Symbol) -> bool {
        if symbol.config_dependencies.is_empty() {
            return true; // No config dependencies means always active
        }

        // Check if all config dependencies are satisfied
        symbol
            .config_dependencies
            .iter()
            .all(|config| self.config_context.get(config).copied().unwrap_or(false))
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ChangeType {
    SignatureChange,
    BehaviorChange,
    Deletion,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ImpactAnalysis {
    pub target_symbol: String,
    pub change_type: ChangeType,
    pub high_impact: Vec<Symbol>,
    pub medium_impact: Vec<Symbol>,
    pub low_impact: Vec<Symbol>,
    pub total_affected: usize,
    pub execution_time_ms: u64,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{CallEdge, CallType, SymbolType};

    fn create_test_graph() -> KernelGraph {
        let mut graph = KernelGraph::new();

        // Add some test symbols
        let symbols = vec![
            Symbol {
                name: "vfs_read".to_string(),
                file_path: "fs/read_write.c".to_string(),
                line_number: 450,
                symbol_type: SymbolType::Function,
                signature: None,
                config_dependencies: vec!["CONFIG_VFS".to_string()],
            },
            Symbol {
                name: "sys_read".to_string(),
                file_path: "fs/read_write.c".to_string(),
                line_number: 600,
                symbol_type: SymbolType::Function,
                signature: None,
                config_dependencies: vec![],
            },
            Symbol {
                name: "file_read".to_string(),
                file_path: "fs/file.c".to_string(),
                line_number: 100,
                symbol_type: SymbolType::Function,
                signature: None,
                config_dependencies: vec![],
            },
        ];

        for symbol in symbols {
            graph.add_symbol(symbol);
        }

        // Add call relationships
        let edge = CallEdge {
            call_type: CallType::Direct,
            call_site_line: 605,
            conditional: false,
            config_guard: None,
        };
        graph.add_call("sys_read", "vfs_read", edge).unwrap();

        let edge2 = CallEdge {
            call_type: CallType::Direct,
            call_site_line: 455,
            conditional: false,
            config_guard: None,
        };
        graph.add_call("vfs_read", "file_read", edge2).unwrap();

        graph
    }

    #[test]
    fn test_who_calls_query() -> Result<()> {
        let graph = create_test_graph();
        let engine = QueryEngine::new(&graph);

        let result = engine.who_calls("vfs_read", Some(5))?;

        assert_eq!(result.target_symbol, "vfs_read");
        assert_eq!(result.results.len(), 1);
        assert_eq!(result.results[0].symbol.name, "sys_read");

        Ok(())
    }

    #[test]
    fn test_list_dependencies_query() -> Result<()> {
        let graph = create_test_graph();
        let engine = QueryEngine::new(&graph);

        let result = engine.list_dependencies("vfs_read", Some(5))?;

        assert_eq!(result.target_symbol, "vfs_read");
        assert_eq!(result.results.len(), 1);
        assert_eq!(result.results[0].symbol.name, "file_read");

        Ok(())
    }
}
