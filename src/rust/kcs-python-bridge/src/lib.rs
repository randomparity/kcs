//! Python bridge for KCS Rust libraries
//!
//! Provides Python bindings for the Rust-based kernel analysis components
//! using PyO3 for high-performance integration.

use pyo3::prelude::*;
use pyo3::types::PyDict;
use std::collections::HashMap;

use kcs_parser::{Parser, ParserConfig, SymbolInfo, ParseResult};

/// Python wrapper for the Rust Parser
#[pyclass]
struct PyParser {
    parser: Parser,
}

/// Python representation of symbol information
#[pyclass]
#[derive(Clone)]
struct PySymbolInfo {
    #[pyo3(get)]
    name: String,
    #[pyo3(get)]
    kind: String,
    #[pyo3(get)]
    file_path: String,
    #[pyo3(get)]
    start_line: u32,
    #[pyo3(get)]
    end_line: u32,
    #[pyo3(get)]
    signature: Option<String>,
}

/// Python representation of parse results
#[pyclass]
struct PyParseResult {
    #[pyo3(get)]
    symbols: Vec<PySymbolInfo>,
    #[pyo3(get)]
    call_edges: Vec<(String, String)>,
    #[pyo3(get)]
    errors: Vec<String>,
}

#[pymethods]
impl PyParser {
    #[new]
    fn new(
        tree_sitter_enabled: bool,
        clang_enabled: bool,
        target_arch: Option<String>,
        kernel_version: Option<String>,
    ) -> PyResult<Self> {
        let config = ParserConfig {
            tree_sitter_enabled,
            clang_enabled,
            target_arch: target_arch.unwrap_or_else(|| "x86_64".to_string()),
            kernel_version: kernel_version.unwrap_or_else(|| "6.1".to_string()),
        };

        let parser = Parser::new(config)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("Failed to create parser: {}", e)))?;

        Ok(PyParser { parser })
    }

    /// Parse a single file and extract symbols
    fn parse_file(&mut self, file_path: &str, content: &str) -> PyResult<PyParseResult> {
        let result = self.parser
            .parse_file(file_path, content)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("Parse error: {}", e)))?;

        Ok(convert_parse_result(result))
    }

    /// Parse multiple files in batch for better performance
    fn parse_files(&mut self, files: HashMap<String, String>) -> PyResult<PyParseResult> {
        let result = self.parser
            .parse_files(files)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("Batch parse error: {}", e)))?;

        Ok(convert_parse_result(result))
    }

    /// Extract symbols from kernel directory
    fn parse_kernel_tree(&mut self, kernel_path: &str, config_name: &str) -> PyResult<PyParseResult> {
        let result = self.parser
            .parse_kernel_tree(kernel_path, config_name)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("Kernel tree parse error: {}", e)))?;

        Ok(convert_parse_result(result))
    }

    /// Set parser configuration
    fn configure(&mut self, py_config: &PyDict) -> PyResult<()> {
        let mut config = ParserConfig {
            tree_sitter_enabled: true,
            clang_enabled: false,
            target_arch: "x86_64".to_string(),
            kernel_version: "6.1".to_string(),
        };

        if let Some(tree_sitter) = py_config.get_item("tree_sitter_enabled")? {
            config.tree_sitter_enabled = tree_sitter.extract()?;
        }

        if let Some(clang) = py_config.get_item("clang_enabled")? {
            config.clang_enabled = clang.extract()?;
        }

        if let Some(arch) = py_config.get_item("target_arch")? {
            config.target_arch = arch.extract()?;
        }

        if let Some(version) = py_config.get_item("kernel_version")? {
            config.kernel_version = version.extract()?;
        }

        self.parser.reconfigure(config)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("Configuration error: {}", e)))?;

        Ok(())
    }
}

/// Convert Rust ParseResult to Python representation
fn convert_parse_result(result: ParseResult) -> PyParseResult {
    let symbols = result.symbols
        .into_iter()
        .map(|s| PySymbolInfo {
            name: s.name,
            kind: s.kind,
            file_path: s.file_path,
            start_line: s.start_line,
            end_line: s.end_line,
            signature: s.signature,
        })
        .collect();

    let call_edges = result.call_edges
        .into_iter()
        .map(|(caller, callee)| (caller, callee))
        .collect();

    PyParseResult {
        symbols,
        call_edges,
        errors: result.errors,
    }
}

/// Utility function to extract symbols from a single file
#[pyfunction]
fn parse_c_file(file_path: &str, content: &str, arch: Option<&str>) -> PyResult<PyParseResult> {
    let config = ParserConfig {
        tree_sitter_enabled: true,
        clang_enabled: false,
        target_arch: arch.unwrap_or("x86_64").to_string(),
        kernel_version: "6.1".to_string(),
    };

    let mut parser = Parser::new(config)
        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("Parser creation failed: {}", e)))?;

    let result = parser
        .parse_file(file_path, content)
        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("Parse failed: {}", e)))?;

    Ok(convert_parse_result(result))
}

/// Utility function to analyze kernel patterns in code
#[pyfunction]
fn analyze_kernel_patterns(content: &str) -> PyResult<Vec<String>> {
    // TODO: Implement actual kernel pattern analysis
    // This would detect EXPORT_SYMBOL, module_param, etc.

    let patterns = vec![
        "EXPORT_SYMBOL".to_string(),
        "module_param".to_string(),
        "MODULE_LICENSE".to_string(),
        "subsys_initcall".to_string(),
    ];

    let mut found_patterns = Vec::new();
    for pattern in patterns {
        if content.contains(&pattern) {
            found_patterns.push(pattern);
        }
    }

    Ok(found_patterns)
}

/// Python module definition
#[pymodule]
fn kcs_python_bridge(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<PyParser>()?;
    m.add_class::<PySymbolInfo>()?;
    m.add_class::<PyParseResult>()?;
    m.add_function(wrap_pyfunction!(parse_c_file, m)?)?;
    m.add_function(wrap_pyfunction!(analyze_kernel_patterns, m)?)?;
    Ok(())
}
