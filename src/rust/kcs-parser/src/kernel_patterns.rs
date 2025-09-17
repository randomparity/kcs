//! Kernel-specific pattern recognition
//!
//! This module provides pattern detection for Linux kernel-specific constructs
//! such as EXPORT_SYMBOL declarations, module parameters, boot parameters, etc.

use regex::Regex;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Type of kernel-specific pattern detected
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum PatternType {
    /// EXPORT_SYMBOL and its variants
    ExportSymbol,
    /// module_param and related macros
    ModuleParam,
    /// Boot parameter declarations (__setup, early_param, etc.)
    BootParam,
}

/// Represents a detected kernel pattern
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KernelPattern {
    /// Type of pattern detected
    pub pattern_type: PatternType,
    /// Name of the symbol/parameter
    pub name: String,
    /// Source file path
    pub file_path: String,
    /// Line number where pattern was found
    pub line_number: u32,
    /// Additional metadata specific to pattern type
    pub metadata: HashMap<String, String>,
}

/// Pattern detector for kernel-specific constructs
pub struct PatternDetector {
    export_symbol_regex: Regex,
    export_symbol_gpl_regex: Regex,
    export_symbol_ns_regex: Regex,
    export_symbol_ns_gpl_regex: Regex,
}

impl Default for PatternDetector {
    fn default() -> Self {
        Self::new()
    }
}

impl PatternDetector {
    /// Create a new pattern detector
    pub fn new() -> Self {
        // EXPORT_SYMBOL patterns
        // Matches: EXPORT_SYMBOL(symbol_name)
        let export_symbol_regex =
            Regex::new(r"(?m)\bEXPORT_SYMBOL\s*\(\s*([a-zA-Z_][a-zA-Z0-9_]*)\s*\)").unwrap();

        // Matches: EXPORT_SYMBOL_GPL(symbol_name)
        let export_symbol_gpl_regex =
            Regex::new(r"(?m)\bEXPORT_SYMBOL_GPL\s*\(\s*([a-zA-Z_][a-zA-Z0-9_]*)\s*\)").unwrap();

        // Matches: EXPORT_SYMBOL_NS(symbol_name, namespace)
        let export_symbol_ns_regex = Regex::new(
            r#"(?m)\bEXPORT_SYMBOL_NS\s*\(\s*([a-zA-Z_][a-zA-Z0-9_]*)\s*,\s*([^)]+)\s*\)"#,
        )
        .unwrap();

        // Matches: EXPORT_SYMBOL_NS_GPL(symbol_name, namespace)
        let export_symbol_ns_gpl_regex = Regex::new(
            r#"(?m)\bEXPORT_SYMBOL_NS_GPL\s*\(\s*([a-zA-Z_][a-zA-Z0-9_]*)\s*,\s*([^)]+)\s*\)"#,
        )
        .unwrap();

        Self {
            export_symbol_regex,
            export_symbol_gpl_regex,
            export_symbol_ns_regex,
            export_symbol_ns_gpl_regex,
        }
    }

    /// Detect EXPORT_SYMBOL patterns in source code
    pub fn detect_export_symbols(&self, content: &str, file_path: &str) -> Vec<KernelPattern> {
        let mut patterns = Vec::new();

        // Find all EXPORT_SYMBOL declarations
        for cap in self.export_symbol_regex.captures_iter(content) {
            if let Some(symbol_match) = cap.get(1) {
                let symbol_name = symbol_match.as_str().to_string();
                // Use the start of the full match (cap.get(0)) for line number
                let line_number = calculate_line_number(content, cap.get(0).unwrap().start());

                let mut metadata = HashMap::new();
                metadata.insert("export_type".to_string(), "EXPORT_SYMBOL".to_string());

                patterns.push(KernelPattern {
                    pattern_type: PatternType::ExportSymbol,
                    name: symbol_name,
                    file_path: file_path.to_string(),
                    line_number,
                    metadata,
                });
            }
        }

        // Find all EXPORT_SYMBOL_GPL declarations
        for cap in self.export_symbol_gpl_regex.captures_iter(content) {
            if let Some(symbol_match) = cap.get(1) {
                let symbol_name = symbol_match.as_str().to_string();
                // Use the start of the full match (cap.get(0)) for line number
                let line_number = calculate_line_number(content, cap.get(0).unwrap().start());

                let mut metadata = HashMap::new();
                metadata.insert("export_type".to_string(), "EXPORT_SYMBOL_GPL".to_string());

                patterns.push(KernelPattern {
                    pattern_type: PatternType::ExportSymbol,
                    name: symbol_name,
                    file_path: file_path.to_string(),
                    line_number,
                    metadata,
                });
            }
        }

        // Find all EXPORT_SYMBOL_NS declarations
        for cap in self.export_symbol_ns_regex.captures_iter(content) {
            if let (Some(symbol_match), Some(ns_match)) = (cap.get(1), cap.get(2)) {
                let symbol_name = symbol_match.as_str().to_string();
                let namespace = ns_match.as_str().trim().to_string();
                // Use the start of the full match (cap.get(0)) for line number
                let line_number = calculate_line_number(content, cap.get(0).unwrap().start());

                let mut metadata = HashMap::new();
                metadata.insert("export_type".to_string(), "EXPORT_SYMBOL_NS".to_string());
                metadata.insert("namespace".to_string(), namespace);

                patterns.push(KernelPattern {
                    pattern_type: PatternType::ExportSymbol,
                    name: symbol_name,
                    file_path: file_path.to_string(),
                    line_number,
                    metadata,
                });
            }
        }

        // Find all EXPORT_SYMBOL_NS_GPL declarations
        for cap in self.export_symbol_ns_gpl_regex.captures_iter(content) {
            if let (Some(symbol_match), Some(ns_match)) = (cap.get(1), cap.get(2)) {
                let symbol_name = symbol_match.as_str().to_string();
                let namespace = ns_match.as_str().trim().to_string();
                // Use the start of the full match (cap.get(0)) for line number
                let line_number = calculate_line_number(content, cap.get(0).unwrap().start());

                let mut metadata = HashMap::new();
                metadata.insert(
                    "export_type".to_string(),
                    "EXPORT_SYMBOL_NS_GPL".to_string(),
                );
                metadata.insert("namespace".to_string(), namespace);

                patterns.push(KernelPattern {
                    pattern_type: PatternType::ExportSymbol,
                    name: symbol_name,
                    file_path: file_path.to_string(),
                    line_number,
                    metadata,
                });
            }
        }

        patterns
    }

    /// Detect all kernel patterns in source code
    pub fn detect_patterns(&self, content: &str, file_path: &str) -> Vec<KernelPattern> {
        let mut patterns = Vec::new();

        // Detect export symbols
        patterns.extend(self.detect_export_symbols(content, file_path));

        // TODO: Add module_param detection (T011)
        // TODO: Add boot parameter detection (T012)

        patterns
    }
}

/// Calculate line number from byte offset in content
fn calculate_line_number(content: &str, byte_offset: usize) -> u32 {
    let up_to_offset = &content[..byte_offset.min(content.len())];
    up_to_offset.lines().count() as u32 + 1
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_export_symbol_detection() {
        let detector = PatternDetector::new();
        let content = r#"
void my_function(void) {
    // Function implementation
}
EXPORT_SYMBOL(my_function);

static int helper_func(int x) {
    return x * 2;
}
EXPORT_SYMBOL_GPL(helper_func);
"#;

        let patterns = detector.detect_export_symbols(content, "test.c");
        assert_eq!(patterns.len(), 2);

        assert_eq!(patterns[0].name, "my_function");
        assert_eq!(
            patterns[0].metadata.get("export_type"),
            Some(&"EXPORT_SYMBOL".to_string())
        );
        // Line numbers: content starts with "\n", so EXPORT_SYMBOL is on line 5
        assert_eq!(patterns[0].line_number, 5);

        assert_eq!(patterns[1].name, "helper_func");
        assert_eq!(
            patterns[1].metadata.get("export_type"),
            Some(&"EXPORT_SYMBOL_GPL".to_string())
        );
        assert_eq!(patterns[1].line_number, 10);
    }

    #[test]
    fn test_export_symbol_ns_detection() {
        let detector = PatternDetector::new();
        let content = r#"
int ns_function(void) {
    return 42;
}
EXPORT_SYMBOL_NS(ns_function, MY_NAMESPACE);

void gpl_ns_func(void) {
}
EXPORT_SYMBOL_NS_GPL(gpl_ns_func, ANOTHER_NS);
"#;

        let patterns = detector.detect_export_symbols(content, "test.c");
        assert_eq!(patterns.len(), 2);

        assert_eq!(patterns[0].name, "ns_function");
        assert_eq!(
            patterns[0].metadata.get("export_type"),
            Some(&"EXPORT_SYMBOL_NS".to_string())
        );
        assert_eq!(
            patterns[0].metadata.get("namespace"),
            Some(&"MY_NAMESPACE".to_string())
        );

        assert_eq!(patterns[1].name, "gpl_ns_func");
        assert_eq!(
            patterns[1].metadata.get("export_type"),
            Some(&"EXPORT_SYMBOL_NS_GPL".to_string())
        );
        assert_eq!(
            patterns[1].metadata.get("namespace"),
            Some(&"ANOTHER_NS".to_string())
        );
    }

    #[test]
    fn test_preprocessor_conditional_handling() {
        let detector = PatternDetector::new();
        let content = r#"
#ifdef CONFIG_SOMETHING
void conditional_func(void) {
}
EXPORT_SYMBOL(conditional_func);
#endif

#ifndef CONFIG_OTHER
int another_func(void) {
    return 0;
}
EXPORT_SYMBOL_GPL(another_func);
#endif
"#;

        let patterns = detector.detect_export_symbols(content, "test.c");
        // Should detect both even inside preprocessor conditionals
        assert_eq!(patterns.len(), 2);
        assert_eq!(patterns[0].name, "conditional_func");
        assert_eq!(patterns[1].name, "another_func");
    }

    #[test]
    fn test_line_number_calculation() {
        let content = "line1\nline2\nline3\n";
        assert_eq!(calculate_line_number(content, 0), 1);
        assert_eq!(calculate_line_number(content, 6), 2);
        assert_eq!(calculate_line_number(content, 12), 3);
    }

    #[test]
    fn test_complex_symbol_names() {
        let detector = PatternDetector::new();
        let content = r#"
EXPORT_SYMBOL(__some_function_123);
EXPORT_SYMBOL_GPL(_internal_helper);
EXPORT_SYMBOL(CamelCaseFunction);
"#;

        let patterns = detector.detect_export_symbols(content, "test.c");
        assert_eq!(patterns.len(), 3);

        // Create a set of names to check without ordering requirements
        let names: Vec<&str> = patterns.iter().map(|p| p.name.as_str()).collect();
        assert!(names.contains(&"__some_function_123"));
        assert!(names.contains(&"_internal_helper"));
        assert!(names.contains(&"CamelCaseFunction"));

        // Verify export types
        for pattern in &patterns {
            match pattern.name.as_str() {
                "__some_function_123" => assert_eq!(
                    pattern.metadata.get("export_type"),
                    Some(&"EXPORT_SYMBOL".to_string())
                ),
                "_internal_helper" => assert_eq!(
                    pattern.metadata.get("export_type"),
                    Some(&"EXPORT_SYMBOL_GPL".to_string())
                ),
                "CamelCaseFunction" => assert_eq!(
                    pattern.metadata.get("export_type"),
                    Some(&"EXPORT_SYMBOL".to_string())
                ),
                _ => panic!("Unexpected symbol name"),
            }
        }
    }
}
