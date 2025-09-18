use anyhow::Result;
use kcs_graph::KernelGraph;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::Path;

pub mod drift_detector;
pub mod report_generator;
pub mod spec_parser;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DriftReport {
    pub specification: SpecificationInfo,
    pub implementation: ImplementationInfo,
    pub drift_findings: Vec<DriftFinding>,
    pub compliance_score: f64,
    pub recommendations: Vec<String>,
    pub metadata: ReportMetadata,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SpecificationInfo {
    pub source: String,
    pub version: String,
    pub timestamp: chrono::DateTime<chrono::Utc>,
    pub requirements: Vec<Requirement>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ImplementationInfo {
    pub kernel_version: String,
    pub commit_hash: String,
    pub config: HashMap<String, bool>,
    pub analyzed_symbols: usize,
    pub analysis_timestamp: chrono::DateTime<chrono::Utc>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Requirement {
    pub id: String,
    pub category: RequirementCategory,
    pub description: String,
    pub expected_symbols: Vec<String>,
    pub expected_behavior: String,
    pub mandatory: bool,
    pub config_dependencies: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum RequirementCategory {
    Syscall,
    API,
    ABI,
    Performance,
    Security,
    Compatibility,
    Feature,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DriftFinding {
    pub requirement_id: String,
    pub drift_type: DriftType,
    pub severity: DriftSeverity,
    pub description: String,
    pub expected: String,
    pub actual: String,
    pub affected_symbols: Vec<String>,
    pub file_locations: Vec<FileLocation>,
    pub remediation_suggestion: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub enum DriftType {
    MissingSymbol,
    UnexpectedSymbol,
    SignatureMismatch,
    BehaviorDifference,
    PerformanceRegression,
    SecurityViolation,
    ABIBreak,
    ConfigMismatch,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub enum DriftSeverity {
    Critical,
    High,
    Medium,
    Low,
    Info,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FileLocation {
    pub file_path: String,
    pub line_number: u32,
    pub column: Option<u32>,
    pub context: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReportMetadata {
    pub analysis_time: chrono::DateTime<chrono::Utc>,
    pub execution_time_ms: u64,
    pub total_requirements: usize,
    pub requirements_checked: usize,
    pub drift_count: usize,
    pub critical_drift_count: usize,
}

pub struct DriftAnalyzer {
    graph: KernelGraph,
    config_context: HashMap<String, bool>,
}

impl DriftAnalyzer {
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

    pub fn analyze_spec_drift<P: AsRef<Path>>(
        &self,
        spec_path: P,
        kernel_version: &str,
    ) -> Result<DriftReport> {
        let start_time = std::time::Instant::now();
        let analysis_time = chrono::Utc::now();

        // Parse the specification
        let spec_content = std::fs::read_to_string(&spec_path)?;
        let specification = spec_parser::parse_specification(&spec_content)?;

        // Create drift detector with the same graph and config
        let detector = drift_detector::DriftDetector::new(self.graph.clone())
            .with_config(self.config_context.clone())
            .with_depth_limit(10);

        // Analyze each requirement using the drift detector
        let mut drift_findings = Vec::new();
        let mut requirements_checked = 0;

        for requirement in &specification.requirements {
            if self.should_check_requirement(requirement) {
                requirements_checked += 1;

                // Use the drift detector for comprehensive analysis
                let mut findings = Vec::new();

                // Check for missing symbols
                findings.extend(
                    detector.detect_missing_symbols(&requirement.expected_symbols, &requirement.id),
                );

                // Check for signature mismatches
                for symbol in &requirement.expected_symbols {
                    if let Some(finding) = detector.detect_signature_mismatch(
                        symbol,
                        &requirement.expected_behavior,
                        &requirement.id,
                    ) {
                        findings.push(finding);
                    }
                }

                // Check for behavior differences
                findings.extend(detector.detect_behavior_differences(requirement)?);

                // Check for ABI breaks if applicable
                if requirement.category == RequirementCategory::ABI {
                    findings.extend(detector.detect_abi_breaks(requirement)?);
                }

                // Check for config mismatches
                findings.extend(detector.detect_config_mismatches(requirement));

                drift_findings.extend(findings);
            }
        }

        // Calculate compliance score
        let compliance_score =
            self.calculate_compliance_score(&specification.requirements, &drift_findings);

        // Generate recommendations
        let recommendations = self.generate_recommendations(&drift_findings);

        let execution_time = start_time.elapsed().as_millis() as u64;

        let critical_drift_count = drift_findings
            .iter()
            .filter(|f| matches!(f.severity, DriftSeverity::Critical))
            .count();

        let total_requirements = specification.requirements.len();
        let drift_count = drift_findings.len();

        Ok(DriftReport {
            specification: SpecificationInfo {
                source: spec_path.as_ref().to_string_lossy().to_string(),
                version: "1.0".to_string(), // TODO: Parse from spec
                timestamp: analysis_time,
                requirements: specification.requirements,
            },
            implementation: ImplementationInfo {
                kernel_version: kernel_version.to_string(),
                commit_hash: "unknown".to_string(), // TODO: Get from git
                config: self.config_context.clone(),
                analyzed_symbols: self.graph.symbol_count(),
                analysis_timestamp: analysis_time,
            },
            drift_findings,
            compliance_score,
            recommendations,
            metadata: ReportMetadata {
                analysis_time,
                execution_time_ms: execution_time,
                total_requirements,
                requirements_checked,
                drift_count,
                critical_drift_count,
            },
        })
    }

    pub fn check_symbol_conformance(
        &self,
        symbol_name: &str,
        expected_signature: &str,
    ) -> Result<Option<DriftFinding>> {
        if let Some(symbol) = self.graph.get_symbol(symbol_name) {
            if let Some(actual_signature) = &symbol.signature {
                if actual_signature != expected_signature {
                    return Ok(Some(DriftFinding {
                        requirement_id: format!("symbol_{}", symbol_name),
                        drift_type: DriftType::SignatureMismatch,
                        severity: DriftSeverity::High,
                        description: format!("Symbol '{}' signature mismatch", symbol_name),
                        expected: expected_signature.to_string(),
                        actual: actual_signature.clone(),
                        affected_symbols: vec![symbol_name.to_string()],
                        file_locations: vec![FileLocation {
                            file_path: symbol.file_path.clone(),
                            line_number: symbol.line_number,
                            column: None,
                            context: actual_signature.clone(),
                        }],
                        remediation_suggestion: format!(
                            "Update signature to match specification: {}",
                            expected_signature
                        ),
                    }));
                }
            }
        } else {
            return Ok(Some(DriftFinding {
                requirement_id: format!("symbol_{}", symbol_name),
                drift_type: DriftType::MissingSymbol,
                severity: DriftSeverity::Critical,
                description: format!("Required symbol '{}' not found", symbol_name),
                expected: format!(
                    "Symbol {} with signature {}",
                    symbol_name, expected_signature
                ),
                actual: "Symbol not found".to_string(),
                affected_symbols: vec![symbol_name.to_string()],
                file_locations: vec![],
                remediation_suggestion: format!("Implement required symbol: {}", symbol_name),
            }));
        }

        Ok(None)
    }

    /// Check conformance for a batch of API requirements
    pub fn check_api_conformance(
        &self,
        api_requirements: &[Requirement],
    ) -> Result<Vec<DriftFinding>> {
        let detector = drift_detector::DriftDetector::new(self.graph.clone())
            .with_config(self.config_context.clone());

        let mut findings = Vec::new();
        for requirement in api_requirements {
            if requirement.category == RequirementCategory::API {
                findings.extend(
                    detector.detect_missing_symbols(&requirement.expected_symbols, &requirement.id),
                );
            }
        }
        Ok(findings)
    }

    fn should_check_requirement(&self, requirement: &Requirement) -> bool {
        // Check if config dependencies are satisfied
        if !requirement.config_dependencies.is_empty() {
            for config in &requirement.config_dependencies {
                if !self.config_context.get(config).copied().unwrap_or(false) {
                    return false; // Skip if config dependency not met
                }
            }
        }
        true
    }

    /// Perform call chain analysis between two symbols
    pub fn analyze_call_chain(
        &self,
        from_symbol: &str,
        to_symbol: &str,
    ) -> Result<Option<Vec<String>>> {
        if let Some(path) = self.graph.get_call_path(from_symbol, to_symbol) {
            Ok(Some(path.iter().map(|s| s.name.clone()).collect()))
        } else {
            Ok(None)
        }
    }

    /// Get all symbols that depend on a specific configuration
    pub fn get_config_dependent_symbols(&self, config_name: &str) -> Vec<String> {
        if let Some(indices) = self.graph.symbols_by_config(config_name) {
            indices
                .iter()
                .filter_map(|idx| self.graph.graph().node_weight(*idx).map(|s| s.name.clone()))
                .collect()
        } else {
            Vec::new()
        }
    }

    fn calculate_compliance_score(
        &self,
        requirements: &[Requirement],
        findings: &[DriftFinding],
    ) -> f64 {
        if requirements.is_empty() {
            return 100.0;
        }

        let total_weight: f64 = requirements
            .iter()
            .map(|r| if r.mandatory { 2.0 } else { 1.0 })
            .sum();

        let violation_weight: f64 = findings
            .iter()
            .map(|f| match f.severity {
                DriftSeverity::Critical => 4.0,
                DriftSeverity::High => 3.0,
                DriftSeverity::Medium => 2.0,
                DriftSeverity::Low => 1.0,
                DriftSeverity::Info => 0.5,
            })
            .sum();

        let compliance = ((total_weight - violation_weight) / total_weight * 100.0).max(0.0);
        compliance.min(100.0)
    }

    fn generate_recommendations(&self, findings: &[DriftFinding]) -> Vec<String> {
        let mut recommendations = Vec::new();

        let critical_count = findings
            .iter()
            .filter(|f| matches!(f.severity, DriftSeverity::Critical))
            .count();
        let high_count = findings
            .iter()
            .filter(|f| matches!(f.severity, DriftSeverity::High))
            .count();

        if critical_count > 0 {
            recommendations.push(format!(
                "CRITICAL: {} critical specification violations found - immediate action required",
                critical_count
            ));
        }

        if high_count > 0 {
            recommendations.push(format!(
                "HIGH: {} high-severity violations found - review and fix recommended",
                high_count
            ));
        }

        // Group findings by type for recommendations
        let mut missing_symbols = 0;
        let mut signature_mismatches = 0;
        let mut abi_breaks = 0;

        for finding in findings {
            match finding.drift_type {
                DriftType::MissingSymbol => missing_symbols += 1,
                DriftType::SignatureMismatch => signature_mismatches += 1,
                DriftType::ABIBreak => abi_breaks += 1,
                _ => {}
            }
        }

        if missing_symbols > 0 {
            recommendations.push(format!(
                "Implement {} missing symbols to meet specification requirements",
                missing_symbols
            ));
        }

        if signature_mismatches > 0 {
            recommendations.push(format!(
                "Fix {} function signature mismatches",
                signature_mismatches
            ));
        }

        if abi_breaks > 0 {
            recommendations.push(format!("Address {} ABI compatibility issues", abi_breaks));
        }

        if recommendations.is_empty() {
            recommendations
                .push("Implementation appears to be compliant with specification".to_string());
        }

        recommendations
    }

    /// Analyze drift and generate a formatted report
    pub fn analyze_and_report<P: AsRef<Path>>(
        &self,
        spec_path: P,
        kernel_version: &str,
        output_format: report_generator::OutputFormat,
    ) -> Result<String> {
        // Perform drift analysis
        let drift_report = self.analyze_spec_drift(spec_path, kernel_version)?;

        // Generate formatted report
        let generator =
            report_generator::ReportGenerator::new(drift_report).with_format(output_format);
        generator.generate()
    }

    /// Analyze drift and save report to file
    pub fn analyze_and_save_report<P: AsRef<Path>, Q: AsRef<Path>>(
        &self,
        spec_path: P,
        kernel_version: &str,
        output_path: Q,
        output_format: report_generator::OutputFormat,
    ) -> Result<()> {
        // Perform analysis and generate report
        let report_content = self.analyze_and_report(spec_path, kernel_version, output_format)?;

        // Save to file
        std::fs::write(output_path, report_content)?;
        Ok(())
    }

    /// Run a comprehensive drift analysis with multiple specifications
    pub fn analyze_multiple_specs<P: AsRef<Path>>(
        &self,
        spec_paths: &[P],
        kernel_version: &str,
    ) -> Result<Vec<DriftReport>> {
        let mut reports = Vec::new();

        for spec_path in spec_paths {
            let report = self.analyze_spec_drift(spec_path, kernel_version)?;
            reports.push(report);
        }

        Ok(reports)
    }

    /// Perform incremental drift analysis comparing two kernel versions
    pub fn analyze_version_drift(
        &self,
        spec_path: &Path,
        old_graph: &KernelGraph,
        old_version: &str,
        new_version: &str,
    ) -> Result<(DriftReport, DriftReport)> {
        // Analyze old version
        let old_analyzer =
            DriftAnalyzer::new(old_graph.clone()).with_config_context(self.config_context.clone());
        let old_report = old_analyzer.analyze_spec_drift(spec_path, old_version)?;

        // Analyze new version (self)
        let new_report = self.analyze_spec_drift(spec_path, new_version)?;

        Ok((old_report, new_report))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use kcs_graph::{Symbol, SymbolType};

    fn create_test_graph() -> KernelGraph {
        let mut graph = KernelGraph::new();

        let symbol = Symbol {
            name: "sys_open".to_string(),
            file_path: "fs/open.c".to_string(),
            line_number: 100,
            symbol_type: SymbolType::Function,
            signature: Some(
                "long sys_open(const char *filename, int flags, umode_t mode)".to_string(),
            ),
            config_dependencies: vec![],
        };

        graph.add_symbol(symbol);
        graph
    }

    #[test]
    fn test_drift_analyzer_creation() {
        let graph = create_test_graph();
        let analyzer = DriftAnalyzer::new(graph);
        assert_eq!(analyzer.graph.symbol_count(), 1);
    }

    #[test]
    fn test_symbol_conformance_check() -> Result<()> {
        let graph = create_test_graph();
        let analyzer = DriftAnalyzer::new(graph);

        // Test matching signature
        let result = analyzer.check_symbol_conformance(
            "sys_open",
            "long sys_open(const char *filename, int flags, umode_t mode)",
        )?;
        assert!(result.is_none());

        // Test mismatched signature
        let result = analyzer
            .check_symbol_conformance("sys_open", "int sys_open(char *filename, int flags)")?;
        assert!(result.is_some());
        assert!(matches!(
            result.unwrap().drift_type,
            DriftType::SignatureMismatch
        ));

        // Test missing symbol
        let result = analyzer.check_symbol_conformance("sys_missing", "int sys_missing(void)")?;
        assert!(result.is_some());
        assert!(matches!(
            result.unwrap().drift_type,
            DriftType::MissingSymbol
        ));

        Ok(())
    }
}
