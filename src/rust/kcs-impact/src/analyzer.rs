use crate::{AffectedSymbol, ImpactLevel};
use anyhow::Result;
use kcs_graph::{KernelGraph, Symbol};
use std::collections::{HashMap, HashSet};

pub struct AdvancedAnalyzer {
    graph: KernelGraph,
    config_context: HashMap<String, bool>,
    #[allow(dead_code)]
    analysis_cache: HashMap<String, Vec<AffectedSymbol>>,
}

impl AdvancedAnalyzer {
    pub fn new(graph: KernelGraph) -> Self {
        Self {
            graph,
            config_context: HashMap::new(),
            analysis_cache: HashMap::new(),
        }
    }

    pub fn graph(&self) -> &KernelGraph {
        &self.graph
    }

    pub fn with_config_context(mut self, config: HashMap<String, bool>) -> Self {
        self.config_context = config;
        self
    }

    pub fn analyze_function_signature_change(
        &self,
        function_name: &str,
    ) -> Result<Vec<AffectedSymbol>> {
        let mut affected = Vec::new();

        // Find all direct callers - these are critically affected
        let direct_callers = self.graph.find_callers(function_name);
        for caller in &direct_callers {
            affected.push(AffectedSymbol {
                symbol: (*caller).clone(),
                impact_level: ImpactLevel::Critical,
                impact_reason: format!("Direct caller of {} with signature change", function_name),
                call_chain_distance: 1,
                requires_recompilation: true,
                requires_testing: true,
            });
        }

        // Find function pointer assignments - also critical
        affected.extend(self.find_function_pointer_uses(function_name)?);

        // Find indirect callers - medium to low impact
        for caller in &direct_callers {
            let indirect = self.find_callers_recursive(&caller.name, 2, 4)?;
            affected.extend(indirect);
        }

        Ok(affected)
    }

    pub fn analyze_struct_change(
        &self,
        struct_name: &str,
        _field_changes: &[String],
    ) -> Result<Vec<AffectedSymbol>> {
        let mut affected = Vec::new();

        // Find all functions that use this struct
        // This is a simplified implementation - would need AST analysis for full accuracy
        if let Some(symbol) = self.graph.get_symbol(struct_name) {
            // Find potential users by looking at file
            let file_functions = self.find_symbols_in_file(&symbol.file_path);
            for func in file_functions {
                affected.push(AffectedSymbol {
                    symbol: func.clone(),
                    impact_level: ImpactLevel::High,
                    impact_reason: format!("Potentially uses struct {} in same file", struct_name),
                    call_chain_distance: 0,
                    requires_recompilation: true,
                    requires_testing: true,
                });
            }
        }

        Ok(affected)
    }

    pub fn analyze_macro_change(&self, macro_name: &str) -> Result<Vec<AffectedSymbol>> {
        let mut affected = Vec::new();

        // Macro changes are tricky - they can affect anything that includes the header
        // This is a simplified analysis
        if let Some(macro_symbol) = self.graph.get_symbol(macro_name) {
            if macro_symbol.file_path.ends_with(".h") {
                // Header file macro - potentially affects many files
                affected.push(AffectedSymbol {
                    symbol: macro_symbol.clone(),
                    impact_level: ImpactLevel::High,
                    impact_reason: "Macro definition change in header file".to_string(),
                    call_chain_distance: 0,
                    requires_recompilation: true,
                    requires_testing: true,
                });
            }
        }

        Ok(affected)
    }

    pub fn analyze_config_change(&self, config_option: &str) -> Result<Vec<AffectedSymbol>> {
        let mut affected = Vec::new();

        // Find all symbols that depend on this config option
        if let Some(dependent_symbols) = self.graph.symbols_by_config(config_option) {
            for &symbol_idx in dependent_symbols {
                if let Some(symbol) = self.graph.graph().node_weight(symbol_idx) {
                    affected.push(AffectedSymbol {
                        symbol: symbol.clone(),
                        impact_level: ImpactLevel::Medium,
                        impact_reason: format!("Depends on config option {}", config_option),
                        call_chain_distance: 0,
                        requires_recompilation: true,
                        requires_testing: true,
                    });
                }
            }
        }

        Ok(affected)
    }

    pub fn analyze_blast_radius(
        &self,
        symbol_name: &str,
        max_depth: u32,
    ) -> Result<HashMap<u32, Vec<Symbol>>> {
        let mut blast_radius = HashMap::new();
        let mut visited = HashSet::new();
        let mut current_level = vec![symbol_name.to_string()];

        for depth in 0..=max_depth {
            let mut next_level = Vec::new();
            let mut current_symbols = Vec::new();

            for symbol in &current_level {
                if visited.contains(symbol) {
                    continue;
                }
                visited.insert(symbol.clone());

                if let Some(sym) = self.graph.get_symbol(symbol) {
                    current_symbols.push(sym.clone());

                    // Find callers for next level
                    let callers = self.graph.find_callers(symbol);
                    for caller in callers {
                        if !visited.contains(&caller.name) {
                            next_level.push(caller.name.clone());
                        }
                    }
                }
            }

            if !current_symbols.is_empty() {
                blast_radius.insert(depth, current_symbols);
            }

            current_level = next_level;
            if current_level.is_empty() {
                break;
            }
        }

        Ok(blast_radius)
    }

    pub fn find_critical_path(&self, from: &str, to: &str) -> Result<Option<Vec<String>>> {
        // Use graph algorithms to find the shortest path that would be most critical
        // This is a simplified implementation
        Ok(self
            .graph
            .get_call_path(from, to)
            .map(|path| path.iter().map(|s| s.name.clone()).collect()))
    }

    fn find_function_pointer_uses(&self, _function_name: &str) -> Result<Vec<AffectedSymbol>> {
        let affected = Vec::new();

        // This would require more sophisticated analysis to find function pointer assignments
        // For now, return empty - would need AST-level analysis

        Ok(affected)
    }

    fn find_callers_recursive(
        &self,
        symbol_name: &str,
        current_depth: u32,
        max_depth: u32,
    ) -> Result<Vec<AffectedSymbol>> {
        let mut affected = Vec::new();

        if current_depth > max_depth {
            return Ok(affected);
        }

        let callers = self.graph.find_callers(symbol_name);
        for caller in callers {
            let impact_level = match current_depth {
                2 => ImpactLevel::Medium,
                3 => ImpactLevel::Low,
                _ => ImpactLevel::Low,
            };

            affected.push(AffectedSymbol {
                symbol: (*caller).clone(),
                impact_level,
                impact_reason: format!("Indirect caller at depth {}", current_depth),
                call_chain_distance: current_depth,
                requires_recompilation: current_depth <= 3,
                requires_testing: current_depth <= 4,
            });

            // Recurse
            let recursive =
                self.find_callers_recursive(&caller.name, current_depth + 1, max_depth)?;
            affected.extend(recursive);
        }

        Ok(affected)
    }

    fn find_symbols_in_file(&self, _file_path: &str) -> Vec<&Symbol> {
        // This would need to be implemented with a proper index
        // For now, just return empty
        Vec::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use kcs_graph::SymbolType;

    fn create_test_graph() -> KernelGraph {
        let mut graph = KernelGraph::new();

        let symbols = vec![
            Symbol {
                name: "caller".to_string(),
                file_path: "test.c".to_string(),
                line_number: 10,
                symbol_type: SymbolType::Function,
                signature: None,
                config_dependencies: vec![],
            },
            Symbol {
                name: "target".to_string(),
                file_path: "test.c".to_string(),
                line_number: 20,
                symbol_type: SymbolType::Function,
                signature: None,
                config_dependencies: vec![],
            },
        ];

        for symbol in symbols {
            graph.add_symbol(symbol);
        }

        // Add a call relationship
        use kcs_graph::{CallEdge, CallType};
        let edge = CallEdge {
            call_type: CallType::Direct,
            call_site_line: 15,
            conditional: false,
            config_guard: None,
        };
        graph.add_call("caller", "target", edge).unwrap();

        graph
    }

    #[test]
    fn test_function_signature_analysis() -> Result<()> {
        let graph = create_test_graph();
        let analyzer = AdvancedAnalyzer::new(graph);

        let affected = analyzer.analyze_function_signature_change("target")?;

        // Should find the caller
        assert_eq!(affected.len(), 1);
        assert_eq!(affected[0].symbol.name, "caller");
        assert!(matches!(affected[0].impact_level, ImpactLevel::Critical));

        Ok(())
    }

    #[test]
    fn test_blast_radius_analysis() -> Result<()> {
        let graph = create_test_graph();
        let analyzer = AdvancedAnalyzer::new(graph);

        let blast_radius = analyzer.analyze_blast_radius("target", 3)?;

        // Should have symbols at different depths
        assert!(blast_radius.contains_key(&0)); // target itself
        assert!(blast_radius.contains_key(&1)); // direct callers

        Ok(())
    }
}
