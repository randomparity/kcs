//! Python bridge for KCS Rust libraries
//!
//! Provides Python bindings for the Rust-based kernel analysis components
//! using PyO3 for high-performance integration.

use pyo3::prelude::*;
use pyo3::types::PyDict;
use std::collections::HashMap;

use kcs_extractor::{EntryType, ExtractionConfig, Extractor};
use kcs_parser::kernel_patterns::{PatternDetector, PatternType};
use kcs_parser::{ExtendedParserConfig, ParseResult, ParsedFile, Parser};

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

/// Python representation of an entry point
#[pyclass]
#[derive(Clone)]
struct PyEntryPoint {
    #[pyo3(get)]
    name: String,
    #[pyo3(get)]
    entry_type: String,
    #[pyo3(get)]
    file_path: String,
    #[pyo3(get)]
    line_number: u32,
    #[pyo3(get)]
    signature: String,
    #[pyo3(get)]
    description: Option<String>,
    #[pyo3(get)]
    metadata: Option<HashMap<String, String>>,
}

/// Python representation of a kernel pattern
#[pyclass]
#[derive(Clone)]
struct PyKernelPattern {
    #[pyo3(get)]
    pattern_type: String,
    #[pyo3(get)]
    name: String,
    #[pyo3(get)]
    file_path: String,
    #[pyo3(get)]
    line_number: u32,
    #[pyo3(get)]
    metadata: HashMap<String, String>,
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
        let config = ExtendedParserConfig {
            use_clang: use_clang.unwrap_or(false),
            compile_commands_path: compile_commands_path.map(std::path::PathBuf::from),
            include_paths: Vec::new(),
            defines: HashMap::new(),
            arch: target_arch.unwrap_or_else(|| "x86_64".to_string()),
            config_name: config_name.unwrap_or_else(|| "defconfig".to_string()),
            include_call_graphs: false,
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
        .map(|edge| (edge.caller.clone(), edge.callee.clone()))
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
        include_call_graphs: false,
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

/// Detect kernel patterns in source code
#[pyfunction]
fn detect_patterns(file_path: &str, content: &str) -> PyResult<Vec<PyKernelPattern>> {
    let detector = PatternDetector::new();
    let patterns = detector.detect_patterns(content, file_path);

    Ok(patterns
        .into_iter()
        .map(|p| PyKernelPattern {
            pattern_type: match p.pattern_type {
                PatternType::ExportSymbol => "export_symbol".to_string(),
                PatternType::ModuleParam => "module_param".to_string(),
                PatternType::BootParam => "boot_param".to_string(),
            },
            name: p.name,
            file_path: p.file_path,
            line_number: p.line_number,
            metadata: p.metadata,
        })
        .collect())
}

/// Extract entry points from a kernel directory
#[pyfunction]
fn extract_entry_points(
    kernel_dir: &str,
    include_syscalls: Option<bool>,
    include_ioctls: Option<bool>,
    include_file_ops: Option<bool>,
    include_sysfs: Option<bool>,
    include_procfs: Option<bool>,
    include_debugfs: Option<bool>,
    include_netlink: Option<bool>,
    include_interrupts: Option<bool>,
    include_modules: Option<bool>,
) -> PyResult<Vec<PyEntryPoint>> {
    let config = ExtractionConfig {
        include_syscalls: include_syscalls.unwrap_or(true),
        include_ioctls: include_ioctls.unwrap_or(true),
        include_file_ops: include_file_ops.unwrap_or(true),
        include_sysfs: include_sysfs.unwrap_or(true),
        include_procfs: include_procfs.unwrap_or(true),
        include_debugfs: include_debugfs.unwrap_or(true),
        include_netlink: include_netlink.unwrap_or(true),
        include_interrupts: include_interrupts.unwrap_or(true),
        include_modules: include_modules.unwrap_or(true),
    };

    let extractor = Extractor::new(config);
    let entry_points = extractor
        .extract_from_directory(kernel_dir)
        .map_err(|e| {
            pyo3::exceptions::PyRuntimeError::new_err(format!("Entry point extraction failed: {}", e))
        })?;

    Ok(entry_points
        .into_iter()
        .map(|ep| PyEntryPoint {
            name: ep.name,
            entry_type: match ep.entry_type {
                EntryType::Syscall => "syscall".to_string(),
                EntryType::Ioctl => "ioctl".to_string(),
                EntryType::FileOps => "file_ops".to_string(),
                EntryType::Sysfs => "sysfs".to_string(),
                EntryType::ProcFs => "procfs".to_string(),
                EntryType::DebugFs => "debugfs".to_string(),
                EntryType::Netlink => "netlink".to_string(),
                EntryType::Interrupt => "interrupt".to_string(),
                EntryType::ModuleInit => "module_init".to_string(),
                EntryType::ModuleExit => "module_exit".to_string(),
            },
            file_path: ep.file_path,
            line_number: ep.line_number,
            signature: ep.signature,
            description: ep.description,
            metadata: ep.metadata.map(|m| {
                m.into_iter()
                    .map(|(k, v)| (k, v.to_string()))
                    .collect()
            }),
        })
        .collect())
}

/// Python module definition
#[pymodule]
fn kcs_python_bridge(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<PyParser>()?;
    m.add_class::<PySymbolInfo>()?;
    m.add_class::<PyParseResult>()?;
    m.add_class::<PyEntryPoint>()?;
    m.add_class::<PyKernelPattern>()?;
    m.add_function(wrap_pyfunction!(parse_c_file, m)?)?;
    m.add_function(wrap_pyfunction!(detect_patterns, m)?)?;
    m.add_function(wrap_pyfunction!(extract_entry_points, m)?)?;
    Ok(())
}
