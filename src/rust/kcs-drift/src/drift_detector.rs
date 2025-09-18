use crate::{
    DriftFinding, DriftSeverity, DriftType, FileLocation, Requirement, RequirementCategory,
};
use anyhow::Result;
use kcs_graph::queries::QueryEngine;
use kcs_graph::{KernelGraph, SymbolType};
use std::collections::{HashMap, HashSet};

pub struct DriftDetector {
    graph: KernelGraph,
    config_context: HashMap<String, bool>,
    depth_limit: usize,
}

impl DriftDetector {
    pub fn new(graph: KernelGraph) -> Self {
        Self {
            graph,
            config_context: HashMap::new(),
            depth_limit: 10,
        }
    }

    pub fn with_config(mut self, config: HashMap<String, bool>) -> Self {
        self.config_context = config;
        self
    }

    pub fn with_depth_limit(mut self, limit: usize) -> Self {
        self.depth_limit = limit;
        self
    }

    pub fn detect_missing_symbols(
        &self,
        expected_symbols: &[String],
        requirement_id: &str,
    ) -> Vec<DriftFinding> {
        let mut findings = Vec::new();

        for symbol in expected_symbols {
            if self.graph.get_symbol(symbol).is_none() {
                findings.push(DriftFinding {
                    requirement_id: requirement_id.to_string(),
                    drift_type: DriftType::MissingSymbol,
                    severity: DriftSeverity::Critical,
                    description: format!(
                        "Required symbol '{}' not found in implementation",
                        symbol
                    ),
                    expected: symbol.clone(),
                    actual: "Not found".to_string(),
                    affected_symbols: vec![symbol.clone()],
                    file_locations: vec![],
                    remediation_suggestion: format!(
                        "Implement missing symbol '{}' as specified",
                        symbol
                    ),
                });
            }
        }

        findings
    }

    pub fn detect_unexpected_symbols(
        &self,
        _namespace: &str,
        _allowed_symbols: &HashSet<String>,
        _requirement_id: &str,
    ) -> Vec<DriftFinding> {
        // Note: Without a way to iterate all symbols, we can't detect unexpected ones
        // This would require enhancing kcs_graph with an iterator method
        // For now, return empty findings
        // TODO: Add symbol iteration to KernelGraph

        Vec::new()
    }

    pub fn detect_signature_mismatch(
        &self,
        symbol_name: &str,
        expected_signature: &str,
        requirement_id: &str,
    ) -> Option<DriftFinding> {
        if let Some(symbol) = self.graph.get_symbol(symbol_name) {
            if let Some(actual_signature) = &symbol.signature {
                if !self.signatures_match(actual_signature, expected_signature) {
                    return Some(DriftFinding {
                        requirement_id: requirement_id.to_string(),
                        drift_type: DriftType::SignatureMismatch,
                        severity: DriftSeverity::High,
                        description: format!("Symbol '{}' has incorrect signature", symbol_name),
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
                            "Update signature of '{}' to match specification",
                            symbol_name
                        ),
                    });
                }
            } else {
                // Symbol exists but has no signature information
                return Some(DriftFinding {
                    requirement_id: requirement_id.to_string(),
                    drift_type: DriftType::SignatureMismatch,
                    severity: DriftSeverity::Medium,
                    description: format!("Symbol '{}' missing signature information", symbol_name),
                    expected: expected_signature.to_string(),
                    actual: "No signature available".to_string(),
                    affected_symbols: vec![symbol_name.to_string()],
                    file_locations: vec![FileLocation {
                        file_path: symbol.file_path.clone(),
                        line_number: symbol.line_number,
                        column: None,
                        context: "Missing signature".to_string(),
                    }],
                    remediation_suggestion: format!(
                        "Add signature information for '{}'",
                        symbol_name
                    ),
                });
            }
        }

        None
    }

    pub fn detect_behavior_differences(
        &self,
        requirement: &Requirement,
    ) -> Result<Vec<DriftFinding>> {
        let mut findings = Vec::new();

        // Check for behavioral patterns based on requirement category
        match requirement.category {
            RequirementCategory::Syscall => {
                findings.extend(self.detect_syscall_behavior_drift(requirement)?);
            }
            RequirementCategory::Security => {
                findings.extend(self.detect_security_behavior_drift(requirement)?);
            }
            RequirementCategory::Performance => {
                findings.extend(self.detect_performance_drift(requirement)?);
            }
            _ => {
                // Generic behavior check
                findings.extend(self.detect_generic_behavior_drift(requirement)?);
            }
        }

        Ok(findings)
    }

    pub fn detect_abi_breaks(&self, requirement: &Requirement) -> Result<Vec<DriftFinding>> {
        let mut findings = Vec::new();

        for symbol_name in &requirement.expected_symbols {
            if let Some(symbol) = self.graph.get_symbol(symbol_name) {
                // Check if symbol is exported
                if symbol.symbol_type == SymbolType::Function {
                    // Check for ABI-breaking changes (simplified check)
                    if let Some(signature) = &symbol.signature {
                        if self.has_abi_breaking_change(signature, &requirement.expected_behavior) {
                            findings.push(DriftFinding {
                                requirement_id: requirement.id.clone(),
                                drift_type: DriftType::ABIBreak,
                                severity: DriftSeverity::Critical,
                                description: format!(
                                    "ABI-breaking change detected in '{}'",
                                    symbol_name
                                ),
                                expected: requirement.expected_behavior.clone(),
                                actual: signature.clone(),
                                affected_symbols: vec![symbol_name.clone()],
                                file_locations: vec![FileLocation {
                                    file_path: symbol.file_path.clone(),
                                    line_number: symbol.line_number,
                                    column: None,
                                    context: signature.clone(),
                                }],
                                remediation_suggestion: format!(
                                    "Restore ABI compatibility for '{}'",
                                    symbol_name
                                ),
                            });
                        }
                    }
                }
            } else {
                // Missing ABI symbol is also an ABI break
                findings.push(DriftFinding {
                    requirement_id: requirement.id.clone(),
                    drift_type: DriftType::ABIBreak,
                    severity: DriftSeverity::Critical,
                    description: format!("Required ABI symbol '{}' is missing", symbol_name),
                    expected: symbol_name.clone(),
                    actual: "Not found".to_string(),
                    affected_symbols: vec![symbol_name.clone()],
                    file_locations: vec![],
                    remediation_suggestion: format!(
                        "Implement missing ABI symbol '{}'",
                        symbol_name
                    ),
                });
            }
        }

        Ok(findings)
    }

    pub fn detect_config_mismatches(&self, requirement: &Requirement) -> Vec<DriftFinding> {
        let mut findings = Vec::new();

        for config_dep in &requirement.config_dependencies {
            let is_enabled = self
                .config_context
                .get(config_dep)
                .copied()
                .unwrap_or(false);

            if requirement.mandatory && !is_enabled {
                findings.push(DriftFinding {
                    requirement_id: requirement.id.clone(),
                    drift_type: DriftType::ConfigMismatch,
                    severity: DriftSeverity::High,
                    description: format!(
                        "Mandatory requirement '{}' requires config '{}' which is not enabled",
                        requirement.id, config_dep
                    ),
                    expected: format!("{} = enabled", config_dep),
                    actual: format!("{} = disabled", config_dep),
                    affected_symbols: requirement.expected_symbols.clone(),
                    file_locations: vec![],
                    remediation_suggestion: format!(
                        "Enable config option '{}' for requirement '{}'",
                        config_dep, requirement.id
                    ),
                });
            }
        }

        findings
    }

    pub fn detect_call_chain_drift(
        &self,
        entry_point: &str,
        expected_callees: &[String],
        requirement_id: &str,
    ) -> Result<Vec<DriftFinding>> {
        let mut findings = Vec::new();

        // Use QueryEngine to get actual callees
        let engine = QueryEngine::new(&self.graph).with_config_context(self.config_context.clone());
        let query_result = engine.list_dependencies(entry_point, Some(self.depth_limit as u32))?;

        let actual_set: HashSet<String> = query_result
            .results
            .into_iter()
            .map(|r| r.symbol.name)
            .collect();
        let expected_set: HashSet<String> = expected_callees.iter().cloned().collect();

        // Find missing expected callees
        for expected in expected_callees {
            if !actual_set.contains(expected) {
                findings.push(DriftFinding {
                    requirement_id: requirement_id.to_string(),
                    drift_type: DriftType::BehaviorDifference,
                    severity: DriftSeverity::Medium,
                    description: format!(
                        "Expected call from '{}' to '{}' not found",
                        entry_point, expected
                    ),
                    expected: format!("{} -> {}", entry_point, expected),
                    actual: "Call not present".to_string(),
                    affected_symbols: vec![entry_point.to_string(), expected.clone()],
                    file_locations: vec![],
                    remediation_suggestion: format!(
                        "Add required call from '{}' to '{}'",
                        entry_point, expected
                    ),
                });
            }
        }

        // Find unexpected callees (optional, may generate noise)
        for actual in &actual_set {
            if !expected_set.contains(actual) && !self.is_standard_library_function(actual) {
                findings.push(DriftFinding {
                    requirement_id: requirement_id.to_string(),
                    drift_type: DriftType::BehaviorDifference,
                    severity: DriftSeverity::Low,
                    description: format!("Unexpected call from '{}' to '{}'", entry_point, actual),
                    expected: "Not expected".to_string(),
                    actual: format!("{} -> {}", entry_point, actual),
                    affected_symbols: vec![entry_point.to_string(), actual.to_string()],
                    file_locations: vec![],
                    remediation_suggestion: format!(
                        "Review whether call from '{}' to '{}' is intended",
                        entry_point, actual
                    ),
                });
            }
        }

        Ok(findings)
    }

    // Helper methods

    fn signatures_match(&self, actual: &str, expected: &str) -> bool {
        // Normalize and compare signatures by removing all whitespace variations
        let normalize = |sig: &str| -> String {
            // Remove multiple spaces, normalize pointer notation, and spaces around parentheses
            let mut normalized = sig.to_string();

            // Normalize multiple spaces to single space
            while normalized.contains("  ") {
                normalized = normalized.replace("  ", " ");
            }

            // Normalize spaces around special characters
            normalized = normalized
                .replace(" *", "*")
                .replace("* ", "*")
                .replace(" (", "(")
                .replace("( ", "(")
                .replace(" )", ")")
                .replace(") ", ")")
                .replace(" ,", ",")
                .replace(", ", ",")
                .trim()
                .to_string();

            normalized
        };

        normalize(actual) == normalize(expected)
    }

    fn has_abi_breaking_change(&self, actual: &str, expected: &str) -> bool {
        // Simple ABI break detection - check for return type or parameter changes
        if actual == expected {
            return false;
        }

        // Extract function components for comparison
        let extract_components = |sig: &str| -> (String, Vec<String>) {
            if let Some(open_paren_pos) = sig.find('(') {
                let return_and_name = &sig[..open_paren_pos];
                let params_part = &sig[open_paren_pos..];

                // Extract parameters
                let params = params_part
                    .trim_start_matches('(')
                    .trim_end_matches(')')
                    .split(',')
                    .map(|p| p.trim().to_string())
                    .collect();

                (return_and_name.to_string(), params)
            } else {
                (sig.to_string(), vec![])
            }
        };

        let (actual_ret, actual_params) = extract_components(actual);
        let (expected_ret, expected_params) = extract_components(expected);

        // Different return types or different number of parameters = ABI break
        actual_ret != expected_ret || actual_params.len() != expected_params.len()
    }

    fn is_standard_library_function(&self, name: &str) -> bool {
        // Common kernel utility functions that are often called
        name.starts_with("__")
            || name.starts_with("kmalloc")
            || name.starts_with("kfree")
            || name.starts_with("printk")
            || name.starts_with("mutex_")
            || name.starts_with("spin_")
            || name.starts_with("atomic_")
            || name == "memcpy"
            || name == "memset"
            || name == "strcmp"
    }

    fn detect_syscall_behavior_drift(
        &self,
        requirement: &Requirement,
    ) -> Result<Vec<DriftFinding>> {
        let mut findings = Vec::new();

        for symbol_name in &requirement.expected_symbols {
            if symbol_name.starts_with("sys_") {
                // Check if syscall has proper error handling
                if let Some(symbol) = self.graph.get_symbol(symbol_name) {
                    // Simple check: syscalls should return long or int
                    if let Some(sig) = &symbol.signature {
                        if !sig.starts_with("long ") && !sig.starts_with("int ") {
                            findings.push(DriftFinding {
                                requirement_id: requirement.id.clone(),
                                drift_type: DriftType::BehaviorDifference,
                                severity: DriftSeverity::Medium,
                                description: format!(
                                    "Syscall '{}' has non-standard return type",
                                    symbol_name
                                ),
                                expected: "Should return long or int".to_string(),
                                actual: sig.clone(),
                                affected_symbols: vec![symbol_name.clone()],
                                file_locations: vec![FileLocation {
                                    file_path: symbol.file_path.clone(),
                                    line_number: symbol.line_number,
                                    column: None,
                                    context: sig.clone(),
                                }],
                                remediation_suggestion: format!(
                                    "Update syscall '{}' to use standard return type",
                                    symbol_name
                                ),
                            });
                        }
                    }
                }
            }
        }

        Ok(findings)
    }

    fn detect_security_behavior_drift(
        &self,
        requirement: &Requirement,
    ) -> Result<Vec<DriftFinding>> {
        let mut findings = Vec::new();

        for symbol_name in &requirement.expected_symbols {
            if symbol_name.contains("security") || symbol_name.contains("cap_") {
                // Check if security functions are actually called
                let engine =
                    QueryEngine::new(&self.graph).with_config_context(self.config_context.clone());
                let query_result = engine.who_calls(symbol_name, Some(1))?;

                if query_result.results.is_empty() && requirement.mandatory {
                    findings.push(DriftFinding {
                        requirement_id: requirement.id.clone(),
                        drift_type: DriftType::SecurityViolation,
                        severity: DriftSeverity::Critical,
                        description: format!(
                            "Security function '{}' exists but is never called",
                            symbol_name
                        ),
                        expected: "Security function should be invoked".to_string(),
                        actual: "Function not called".to_string(),
                        affected_symbols: vec![symbol_name.clone()],
                        file_locations: vec![],
                        remediation_suggestion: format!(
                            "Ensure security function '{}' is properly integrated",
                            symbol_name
                        ),
                    });
                }
            }
        }

        Ok(findings)
    }

    fn detect_performance_drift(&self, requirement: &Requirement) -> Result<Vec<DriftFinding>> {
        let mut findings = Vec::new();

        // Performance drift detection would require runtime metrics
        // For now, we can check for known performance anti-patterns
        for symbol_name in &requirement.expected_symbols {
            if let Some(_symbol) = self.graph.get_symbol(symbol_name) {
                // Check call depth for potential performance issues
                let engine =
                    QueryEngine::new(&self.graph).with_config_context(self.config_context.clone());
                let query_result =
                    engine.list_dependencies(symbol_name, Some(self.depth_limit as u32))?;

                if query_result.results.len() > 50 {
                    findings.push(DriftFinding {
                        requirement_id: requirement.id.clone(),
                        drift_type: DriftType::PerformanceRegression,
                        severity: DriftSeverity::Medium,
                        description: format!(
                            "Function '{}' has excessive call chain depth ({}+ callees)",
                            symbol_name,
                            query_result.results.len()
                        ),
                        expected: "Reasonable call chain depth".to_string(),
                        actual: format!("{} callees detected", query_result.results.len()),
                        affected_symbols: vec![symbol_name.clone()],
                        file_locations: vec![],
                        remediation_suggestion: format!(
                            "Review and optimize call chain for '{}'",
                            symbol_name
                        ),
                    });
                }
            }
        }

        Ok(findings)
    }

    fn detect_generic_behavior_drift(
        &self,
        requirement: &Requirement,
    ) -> Result<Vec<DriftFinding>> {
        let mut findings = Vec::new();

        // Generic behavior checks
        for symbol_name in &requirement.expected_symbols {
            if let Some(symbol) = self.graph.get_symbol(symbol_name) {
                // Check if function symbols have implementations
                if symbol.symbol_type == SymbolType::Function && symbol.signature.is_none() {
                    findings.push(DriftFinding {
                        requirement_id: requirement.id.clone(),
                        drift_type: DriftType::BehaviorDifference,
                        severity: DriftSeverity::Low,
                        description: format!(
                            "Symbol '{}' lacks implementation details",
                            symbol_name
                        ),
                        expected: "Complete implementation".to_string(),
                        actual: "Missing implementation details".to_string(),
                        affected_symbols: vec![symbol_name.clone()],
                        file_locations: vec![FileLocation {
                            file_path: symbol.file_path.clone(),
                            line_number: symbol.line_number,
                            column: None,
                            context: "Incomplete implementation".to_string(),
                        }],
                        remediation_suggestion: format!(
                            "Complete implementation for '{}'",
                            symbol_name
                        ),
                    });
                }
            }
        }

        Ok(findings)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use kcs_graph::Symbol;

    fn create_test_graph() -> KernelGraph {
        let mut graph = KernelGraph::new();

        // Add test symbols
        graph.add_symbol(Symbol {
            name: "sys_open".to_string(),
            file_path: "fs/open.c".to_string(),
            line_number: 100,
            symbol_type: SymbolType::Function,
            signature: Some(
                "long sys_open(const char *filename, int flags, umode_t mode)".to_string(),
            ),
            config_dependencies: vec![],
        });

        graph.add_symbol(Symbol {
            name: "vfs_read".to_string(),
            file_path: "fs/read_write.c".to_string(),
            line_number: 200,
            symbol_type: SymbolType::Function,
            signature: Some(
                "ssize_t vfs_read(struct file *file, char __user *buf, size_t count, loff_t *pos)"
                    .to_string(),
            ),
            config_dependencies: vec!["CONFIG_VFS".to_string()],
        });

        graph.add_symbol(Symbol {
            name: "security_file_open".to_string(),
            file_path: "security/security.c".to_string(),
            line_number: 300,
            symbol_type: SymbolType::Function,
            signature: Some("int security_file_open(struct file *file)".to_string()),
            config_dependencies: vec!["CONFIG_SECURITY".to_string()],
        });

        graph
    }

    #[test]
    fn test_detect_missing_symbols() {
        let graph = create_test_graph();
        let detector = DriftDetector::new(graph);

        let expected = vec!["sys_open".to_string(), "sys_missing".to_string()];
        let findings = detector.detect_missing_symbols(&expected, "TEST_001");

        assert_eq!(findings.len(), 1);
        assert_eq!(findings[0].affected_symbols[0], "sys_missing");
        assert!(matches!(findings[0].drift_type, DriftType::MissingSymbol));
    }

    #[test]
    fn test_detect_signature_mismatch() {
        let graph = create_test_graph();
        let detector = DriftDetector::new(graph);

        // Test matching signature
        let finding = detector.detect_signature_mismatch(
            "sys_open",
            "long sys_open(const char *filename, int flags, umode_t mode)",
            "TEST_001",
        );
        assert!(finding.is_none());

        // Test mismatched signature
        let finding = detector.detect_signature_mismatch(
            "sys_open",
            "int sys_open(const char *filename, int flags)",
            "TEST_002",
        );
        assert!(finding.is_some());
        assert!(matches!(
            finding.unwrap().drift_type,
            DriftType::SignatureMismatch
        ));
    }

    #[test]
    fn test_detect_config_mismatches() {
        let graph = create_test_graph();
        let mut config = HashMap::new();
        config.insert("CONFIG_VFS".to_string(), false);

        let detector = DriftDetector::new(graph).with_config(config);

        let requirement = Requirement {
            id: "REQ_001".to_string(),
            category: RequirementCategory::API,
            description: "VFS requirement".to_string(),
            expected_symbols: vec!["vfs_read".to_string()],
            expected_behavior: String::new(),
            mandatory: true,
            config_dependencies: vec!["CONFIG_VFS".to_string()],
        };

        let findings = detector.detect_config_mismatches(&requirement);
        assert_eq!(findings.len(), 1);
        assert!(matches!(findings[0].drift_type, DriftType::ConfigMismatch));
    }

    #[test]
    fn test_detect_abi_breaks() {
        let graph = create_test_graph();
        let detector = DriftDetector::new(graph);

        let requirement = Requirement {
            id: "ABI_001".to_string(),
            category: RequirementCategory::ABI,
            description: "ABI requirement".to_string(),
            expected_symbols: vec!["sys_open".to_string(), "sys_missing".to_string()],
            expected_behavior: "long sys_open(const char *filename, int flags, umode_t mode)"
                .to_string(),
            mandatory: true,
            config_dependencies: vec![],
        };

        let findings = detector.detect_abi_breaks(&requirement).unwrap();
        assert!(!findings.is_empty());

        // Should find sys_missing as an ABI break
        let missing_finding = findings
            .iter()
            .find(|f| f.affected_symbols.contains(&"sys_missing".to_string()))
            .expect("Should find missing symbol");
        assert!(matches!(missing_finding.drift_type, DriftType::ABIBreak));
    }

    #[test]
    fn test_signatures_match() {
        let graph = KernelGraph::new();
        let detector = DriftDetector::new(graph);

        // Test normalized signatures
        assert!(detector.signatures_match("int  func( void * ptr )", "int func(void*ptr)"));

        assert!(!detector.signatures_match("int func(void)", "void func(int)"));
    }

    #[test]
    fn test_has_abi_breaking_change() {
        let graph = KernelGraph::new();
        let detector = DriftDetector::new(graph);

        // Different return type = ABI break
        assert!(detector.has_abi_breaking_change("int func(void)", "void func(void)"));

        // Different parameter count = ABI break
        assert!(detector.has_abi_breaking_change("int func(int a, int b)", "int func(int a)"));

        // Same signature = no ABI break
        assert!(!detector.has_abi_breaking_change("int func(int a)", "int func(int a)"));
    }
}
