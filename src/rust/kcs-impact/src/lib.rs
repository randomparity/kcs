use anyhow::Result;
use kcs_graph::{KernelGraph, Symbol};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use std::path::Path;

pub mod analyzer;
pub mod diff_analyzer;
pub mod patch_parser;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ImpactAnalysis {
    pub target_changes: Vec<ChangePoint>,
    pub affected_symbols: Vec<AffectedSymbol>,
    pub risk_assessment: RiskAssessment,
    pub recommendations: Vec<String>,
    pub analysis_metadata: AnalysisMetadata,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChangePoint {
    pub file_path: String,
    pub line_number: u32,
    pub change_type: ChangeType,
    pub symbol_name: Option<String>,
    pub description: String,
    pub diff_context: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ChangeType {
    FunctionAdded,
    FunctionRemoved,
    FunctionModified,
    SignatureChanged,
    VariableAdded,
    VariableRemoved,
    VariableModified,
    MacroChanged,
    StructChanged,
    ConfigChanged,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AffectedSymbol {
    pub symbol: Symbol,
    pub impact_level: ImpactLevel,
    pub impact_reason: String,
    pub call_chain_distance: u32,
    pub requires_recompilation: bool,
    pub requires_testing: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ImpactLevel {
    Critical, // Breaks compilation or runtime
    High,     // Significant behavior change
    Medium,   // Minor behavior change
    Low,      // Cosmetic or distant change
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RiskAssessment {
    pub overall_risk: RiskLevel,
    pub compilation_risk: bool,
    pub runtime_risk: bool,
    pub abi_risk: bool,
    pub performance_risk: bool,
    pub security_risk: bool,
    pub affected_subsystems: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum RiskLevel {
    Low,
    Medium,
    High,
    Critical,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AnalysisMetadata {
    pub analysis_time: chrono::DateTime<chrono::Utc>,
    pub execution_time_ms: u64,
    pub total_changes: usize,
    pub total_affected_symbols: usize,
    pub graph_size: usize,
    pub config_context: HashMap<String, bool>,
}

pub struct ImpactAnalyzer {
    graph: KernelGraph,
    config_context: HashMap<String, bool>,
}

impl ImpactAnalyzer {
    pub fn new(graph: KernelGraph) -> Self {
        Self {
            graph,
            config_context: HashMap::new(),
        }
    }

    pub fn with_config_context(mut self, config: HashMap<String, bool>) -> Self {
        self.config_context = config;
        self
    }

    pub fn analyze_patch<P: AsRef<Path>>(&self, patch_path: P) -> Result<ImpactAnalysis> {
        let start_time = std::time::Instant::now();
        let analysis_time = chrono::Utc::now();

        // Parse the patch file to extract changes
        let patch_content = std::fs::read_to_string(&patch_path)?;
        let change_points = patch_parser::parse_patch(&patch_content)?;

        // Analyze impact for each change
        let mut affected_symbols = Vec::new();
        let mut all_affected = HashSet::new();

        for change in &change_points {
            let change_impact = self.analyze_change_point(change)?;
            for affected in change_impact {
                if all_affected.insert(affected.symbol.name.clone()) {
                    affected_symbols.push(affected);
                }
            }
        }

        // Assess overall risk
        let risk_assessment = self.assess_risk(&change_points, &affected_symbols);

        // Generate recommendations
        let recommendations =
            self.generate_recommendations(&change_points, &affected_symbols, &risk_assessment);

        let execution_time = start_time.elapsed().as_millis() as u64;

        let total_changes = change_points.len();
        let total_affected_symbols = affected_symbols.len();

        Ok(ImpactAnalysis {
            target_changes: change_points,
            affected_symbols,
            risk_assessment,
            recommendations,
            analysis_metadata: AnalysisMetadata {
                analysis_time,
                execution_time_ms: execution_time,
                total_changes,
                total_affected_symbols,
                graph_size: self.graph.symbol_count(),
                config_context: self.config_context.clone(),
            },
        })
    }

    pub fn analyze_symbol_change(
        &self,
        symbol_name: &str,
        change_type: ChangeType,
    ) -> Result<Vec<AffectedSymbol>> {
        let mut affected = Vec::new();

        // Find direct callers
        let callers = self.graph.find_callers(symbol_name);
        for caller in &callers {
            let impact_level = match change_type {
                ChangeType::SignatureChanged | ChangeType::FunctionRemoved => ImpactLevel::Critical,
                ChangeType::FunctionModified => ImpactLevel::High,
                _ => ImpactLevel::Medium,
            };

            affected.push(AffectedSymbol {
                symbol: (*caller).clone(),
                impact_level,
                impact_reason: format!("Direct caller of changed symbol {}", symbol_name),
                call_chain_distance: 1,
                requires_recompilation: true,
                requires_testing: true,
            });
        }

        // Find indirect dependencies based on call chain
        let mut visited = HashSet::new();
        visited.insert(symbol_name.to_string());

        for caller in &callers {
            self.find_indirect_impact(&caller.name, &mut affected, &mut visited, 2, 5);
        }

        Ok(affected)
    }

    fn analyze_change_point(&self, change: &ChangePoint) -> Result<Vec<AffectedSymbol>> {
        if let Some(symbol_name) = &change.symbol_name {
            self.analyze_symbol_change(symbol_name, change.change_type.clone())
        } else {
            // File-level change without specific symbol
            Ok(Vec::new())
        }
    }

    fn find_indirect_impact(
        &self,
        symbol_name: &str,
        affected: &mut Vec<AffectedSymbol>,
        visited: &mut HashSet<String>,
        distance: u32,
        max_distance: u32,
    ) {
        if distance > max_distance || visited.contains(symbol_name) {
            return;
        }

        visited.insert(symbol_name.to_string());

        let callers = self.graph.find_callers(symbol_name);
        for caller in callers {
            let impact_level = match distance {
                1..=2 => ImpactLevel::Medium,
                3..=4 => ImpactLevel::Low,
                _ => ImpactLevel::Low,
            };

            affected.push(AffectedSymbol {
                symbol: (*caller).clone(),
                impact_level,
                impact_reason: format!("Indirect dependency at distance {}", distance),
                call_chain_distance: distance,
                requires_recompilation: distance <= 2,
                requires_testing: distance <= 3,
            });

            // Recurse to find further dependencies
            self.find_indirect_impact(&caller.name, affected, visited, distance + 1, max_distance);
        }
    }

    fn assess_risk(&self, changes: &[ChangePoint], affected: &[AffectedSymbol]) -> RiskAssessment {
        let mut compilation_risk = false;
        let mut runtime_risk = false;
        let mut abi_risk = false;
        let mut performance_risk = false;
        let mut security_risk = false;
        let mut affected_subsystems = HashSet::new();

        // Analyze changes for risk factors
        for change in changes {
            match change.change_type {
                ChangeType::FunctionRemoved | ChangeType::SignatureChanged => {
                    compilation_risk = true;
                    abi_risk = true;
                }
                ChangeType::FunctionModified => {
                    runtime_risk = true;
                }
                ChangeType::StructChanged => {
                    abi_risk = true;
                    compilation_risk = true;
                }
                _ => {}
            }

            // Determine affected subsystem from file path
            if change.file_path.starts_with("security/") {
                security_risk = true;
                affected_subsystems.insert("security".to_string());
            } else if change.file_path.starts_with("mm/") {
                performance_risk = true;
                affected_subsystems.insert("memory".to_string());
            } else if change.file_path.starts_with("fs/") {
                affected_subsystems.insert("filesystem".to_string());
            } else if change.file_path.starts_with("net/") {
                affected_subsystems.insert("networking".to_string());
            } else if change.file_path.starts_with("drivers/") {
                affected_subsystems.insert("drivers".to_string());
            }
        }

        // Calculate overall risk based on impact levels
        let critical_count = affected
            .iter()
            .filter(|a| matches!(a.impact_level, ImpactLevel::Critical))
            .count();
        let high_count = affected
            .iter()
            .filter(|a| matches!(a.impact_level, ImpactLevel::High))
            .count();

        let overall_risk = if critical_count > 0 {
            RiskLevel::Critical
        } else if high_count > 5 {
            RiskLevel::High
        } else if high_count > 0 || affected.len() > 20 {
            RiskLevel::Medium
        } else {
            RiskLevel::Low
        };

        RiskAssessment {
            overall_risk,
            compilation_risk,
            runtime_risk,
            abi_risk,
            performance_risk,
            security_risk,
            affected_subsystems: affected_subsystems.into_iter().collect(),
        }
    }

    fn generate_recommendations(
        &self,
        _changes: &[ChangePoint],
        affected: &[AffectedSymbol],
        risk: &RiskAssessment,
    ) -> Vec<String> {
        let mut recommendations = Vec::new();

        if risk.compilation_risk {
            recommendations
                .push("Run full kernel build to check for compilation errors".to_string());
        }

        if risk.abi_risk {
            recommendations
                .push("Check for ABI compatibility issues with external modules".to_string());
        }

        if risk.security_risk {
            recommendations.push("Perform security review and testing".to_string());
        }

        if affected.iter().any(|a| a.requires_testing) {
            recommendations.push("Run regression tests for affected subsystems".to_string());
        }

        let critical_symbols: Vec<_> = affected
            .iter()
            .filter(|a| matches!(a.impact_level, ImpactLevel::Critical))
            .map(|a| &a.symbol.name)
            .collect();

        if !critical_symbols.is_empty() {
            recommendations.push(format!(
                "Critical symbols affected: {}. Thorough testing required.",
                critical_symbols
                    .iter()
                    .map(|s| s.as_str())
                    .collect::<Vec<_>>()
                    .join(", ")
            ));
        }

        match risk.overall_risk {
            RiskLevel::Critical => {
                recommendations
                    .push("CRITICAL: This change has high risk of breaking the kernel".to_string());
            }
            RiskLevel::High => {
                recommendations
                    .push("HIGH RISK: Extensive testing recommended before merge".to_string());
            }
            RiskLevel::Medium => {
                recommendations.push(
                    "Medium risk: Standard testing procedures should be sufficient".to_string(),
                );
            }
            RiskLevel::Low => {
                recommendations.push("Low risk: Minimal testing required".to_string());
            }
        }

        recommendations
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use kcs_graph::SymbolType;

    fn create_test_graph() -> KernelGraph {
        let mut graph = KernelGraph::new();

        let symbol1 = Symbol {
            name: "test_function".to_string(),
            file_path: "test.c".to_string(),
            line_number: 42,
            symbol_type: SymbolType::Function,
            signature: Some("int test_function(void)".to_string()),
            config_dependencies: vec![],
        };

        graph.add_symbol(symbol1);
        graph
    }

    #[test]
    fn test_impact_analyzer_creation() {
        let graph = create_test_graph();
        let analyzer = ImpactAnalyzer::new(graph);

        // Test that we can create an analyzer
        assert_eq!(analyzer.graph.symbol_count(), 1);
    }

    #[test]
    fn test_symbol_change_analysis() -> Result<()> {
        let graph = create_test_graph();
        let analyzer = ImpactAnalyzer::new(graph);

        let affected =
            analyzer.analyze_symbol_change("test_function", ChangeType::FunctionModified)?;

        // No callers in our test graph, so should be empty
        assert_eq!(affected.len(), 0);

        Ok(())
    }
}
