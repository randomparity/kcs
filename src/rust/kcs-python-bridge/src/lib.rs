//! Python bridge for KCS Rust libraries
//!
//! Provides Python bindings for the Rust-based kernel analysis components
//! using PyO3 for high-performance integration.

use pyo3::prelude::*;
use pyo3::types::PyDict;
use std::collections::HashMap;

use kcs_parser::{ParsedFile, Parser, ExtendedParserConfig, ParseResult};

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
        use_clang: Option<bool>,
        compile_commands_path: Option<String>,
        target_arch: Option<String>,
        config_name: Option<String>,
    ) -> PyResult<Self> {
        use std::path::PathBuf;

        let config = ExtendedParserConfig {
            use_clang: use_clang.unwrap_or(false),
            compile_commands_path: compile_commands_path.map(PathBuf::from),
            include_paths: Vec::new(),
            defines: HashMap::new(),
            arch: target_arch.unwrap_or_else(|| "x86_64".to_string()),
            config_name: config_name.unwrap_or_else(|| "defconfig".to_string()),
        };

        let parser = Parser::new(config).map_err(|e| {
            pyo3::exceptions::PyRuntimeError::new_err(format!("Failed to create parser: {}", e))
        })?;

        Ok(PyParser { parser })
    }

    /// Parse a single file and extract symbols
    fn parse_file(&mut self, file_path: &str) -> PyResult<PyParseResult> {
        let result = self.parser.parse_file(file_path).map_err(|e| {
            pyo3::exceptions::PyRuntimeError::new_err(format!("Parse error: {}", e))
        })?;

        Ok(convert_parsed_file(result))
    }

    /// Parse multiple files in batch for better performance
    fn parse_files(&mut self, files: HashMap<String, String>) -> PyResult<PyParseResult> {
        let result = self.parser.parse_files_content(files).map_err(|e| {
            pyo3::exceptions::PyRuntimeError::new_err(format!("Batch parse error: {}", e))
        })?;

        Ok(convert_parse_result(result))
    }

    /// Extract symbols from kernel directory
    fn parse_kernel_tree(
        &mut self,
        kernel_path: &str,
        config_name: &str,
    ) -> PyResult<PyParseResult> {
        let result = self
            .parser
            .parse_kernel_tree(kernel_path, config_name)
            .map_err(|e| {
                pyo3::exceptions::PyRuntimeError::new_err(format!("Kernel tree parse error: {}", e))
            })?;

        Ok(convert_parse_result(result))
    }

    /// Set parser configuration (deprecated - configuration is now set at construction time)
    fn configure(&mut self, _py_config: &PyDict) -> PyResult<()> {
        // Configuration is now immutable and set at parser creation
        // This method is kept for backward compatibility but does nothing
        Ok(())
    }
}

/// Convert Rust ParseResult to Python representation
fn convert_parsed_file(result: ParsedFile) -> PyParseResult {
    let symbols = result
        .symbols
        .into_iter()
        .map(|s| PySymbolInfo {
            name: s.name,
            kind: format!("{:?}", s.kind), // Convert enum to string
            file_path: result.path.to_string_lossy().to_string(),
            start_line: s.start_line,
            end_line: s.end_line,
            signature: Some(s.signature),
        })
        .collect();

    PyParseResult {
        symbols,
        call_edges: Vec::new(), // ParsedFile doesn't have call edges
        errors: Vec::new(),
    }
}

fn convert_parse_result(result: ParseResult) -> PyParseResult {
    let symbols = result
        .symbols
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

    let call_edges = result
        .call_edges
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
fn parse_c_file(_file_path: &str, content: &str, arch: Option<&str>) -> PyResult<PyParseResult> {
    use std::fs;

    let config = ExtendedParserConfig {
        use_clang: false,
        compile_commands_path: None,
        include_paths: Vec::new(),
        defines: HashMap::new(),
        arch: arch.unwrap_or("x86_64").to_string(),
        config_name: "defconfig".to_string(),
    };

    let mut parser = Parser::new(config).map_err(|e| {
        pyo3::exceptions::PyRuntimeError::new_err(format!("Parser creation failed: {}", e))
    })?;

    // Write content to a temporary file since the new API reads from file
    let temp_file = format!("/tmp/kcs_parse_{}.c", std::process::id());
    fs::write(&temp_file, content).map_err(|e| {
        pyo3::exceptions::PyRuntimeError::new_err(format!("Failed to write temp file: {}", e))
    })?;

    let result = parser
        .parse_file(&temp_file)
        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("Parse failed: {}", e)))?;

    // Clean up temp file
    let _ = fs::remove_file(&temp_file);

    Ok(convert_parsed_file(result))
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
