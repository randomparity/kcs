//! Python bridge for KCS Rust libraries
//!
//! Provides Python bindings for the Rust-based kernel analysis components
//! using PyO3 for high-performance integration.

use pyo3::prelude::*;
use std::collections::HashMap;

use kcs_extractor::{EntryType, ExtractionConfig, Extractor};
use kcs_parser::kernel_patterns::{PatternDetector, PatternType};
use kcs_parser::{ExtendedParserConfig, ParseResult, ParsedFile, Parser};
use kcs_serializer::{
    ChecksumCalculator, ChecksumConfig, ChunkInfo, ChunkWriter, ChunkWriterConfig,
    ManifestBuilder, ManifestBuilderConfig, ChunkManifest, ChunkMetadata, ChunkInput
};

/// Python wrapper for the Rust Parser
#[pyclass(unsendable)]
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

/// Python wrapper for ChunkWriter
#[pyclass(unsendable)]
struct PyChunkWriter {
    writer: ChunkWriter,
}

/// Python representation of chunk information
#[pyclass]
#[derive(Clone)]
struct PyChunkInfo {
    #[pyo3(get)]
    chunk_id: String,
    #[pyo3(get)]
    size_bytes: usize,
    #[pyo3(get)]
    checksum_sha256: String,
    #[pyo3(get)]
    format: String,
    #[pyo3(get)]
    compressed_size: usize,
    #[pyo3(get)]
    uncompressed_size: usize,
    #[pyo3(get)]
    compression_ratio: f64,
    #[pyo3(get)]
    item_count: usize,
}

/// Python wrapper for ManifestBuilder
#[pyclass(unsendable)]
struct PyManifestBuilder {
    builder: ManifestBuilder,
}

/// Python representation of chunk manifest
#[pyclass]
#[derive(Clone)]
struct PyChunkManifest {
    #[pyo3(get)]
    version: String,
    #[pyo3(get)]
    created: String,
    #[pyo3(get)]
    kernel_version: Option<String>,
    #[pyo3(get)]
    kernel_path: Option<String>,
    #[pyo3(get)]
    config: Option<String>,
    #[pyo3(get)]
    total_chunks: usize,
    #[pyo3(get)]
    total_size_bytes: u64,
    #[pyo3(get)]
    chunks: Vec<PyChunkMetadata>,
}

/// Python representation of chunk metadata
#[pyclass]
#[derive(Clone)]
struct PyChunkMetadata {
    #[pyo3(get)]
    id: String,
    #[pyo3(get)]
    sequence: usize,
    #[pyo3(get)]
    file: String,
    #[pyo3(get)]
    subsystem: String,
    #[pyo3(get)]
    size_bytes: u64,
    #[pyo3(get)]
    checksum_sha256: String,
    #[pyo3(get)]
    symbol_count: Option<usize>,
}

/// Python wrapper for ChecksumCalculator
#[pyclass(unsendable)]
struct PyChecksumCalculator {
    calculator: ChecksumCalculator,
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
    fn configure(&mut self, _py_config: &Bound<'_, pyo3::types::PyDict>) -> PyResult<()> {
        // Configuration is now immutable and set at parser creation
        // This method is kept for backward compatibility but does nothing
        Ok(())
    }
}

#[pymethods]
impl PyChunkWriter {
    #[new]
    fn new(
        max_chunk_size: Option<usize>,
        target_chunk_size: Option<usize>,
        auto_split: Option<bool>,
        output_directory: Option<String>,
        include_metadata: Option<bool>,
    ) -> PyResult<Self> {
        use std::path::PathBuf;

        let config = ChunkWriterConfig {
            max_chunk_size: max_chunk_size.unwrap_or(50 * 1024 * 1024), // 50MB default
            target_chunk_size: target_chunk_size.unwrap_or(45 * 1024 * 1024), // 45MB default
            auto_split: auto_split.unwrap_or(false),
            output_directory: output_directory.map(PathBuf::from),
            include_metadata: include_metadata.unwrap_or(false),
            ..Default::default()
        };

        let writer = ChunkWriter::new(config).map_err(|e| {
            pyo3::exceptions::PyRuntimeError::new_err(format!("Failed to create chunk writer: {}", e))
        })?;

        Ok(PyChunkWriter { writer })
    }

    /// Write raw chunk data
    fn write_chunk(&mut self, chunk_id: &str, data: &[u8]) -> PyResult<PyChunkInfo> {
        let chunk_info = self.writer.write_chunk(chunk_id, data).map_err(|e| {
            pyo3::exceptions::PyRuntimeError::new_err(format!("Failed to write chunk: {}", e))
        })?;

        Ok(convert_chunk_info(chunk_info))
    }

    /// Write JSON-serializable data as a chunk
    fn write_json_chunk(&mut self, chunk_id: &str, json_data: &str) -> PyResult<PyChunkInfo> {
        let chunk_info = self.writer.write_chunk(chunk_id, json_data.as_bytes()).map_err(|e| {
            pyo3::exceptions::PyRuntimeError::new_err(format!("Failed to write JSON chunk: {}", e))
        })?;

        Ok(convert_chunk_info(chunk_info))
    }

    /// Write chunk to file
    fn write_chunk_to_file(&mut self, chunk_id: &str, data: &[u8]) -> PyResult<String> {
        let file_info = self.writer.write_chunk_to_file(chunk_id, data).map_err(|e| {
            pyo3::exceptions::PyRuntimeError::new_err(format!("Failed to write chunk to file: {}", e))
        })?;

        Ok(file_info.file_path.to_string_lossy().to_string())
    }
}

#[pymethods]
impl PyManifestBuilder {
    #[new]
    fn new(
        version: Option<String>,
        kernel_version: Option<String>,
        kernel_path: Option<String>,
        config: Option<String>,
        output_directory: Option<String>,
        chunk_prefix: Option<String>,
    ) -> PyResult<Self> {
        use std::path::PathBuf;

        let builder_config = ManifestBuilderConfig {
            version: version.unwrap_or_else(|| "1.0.0".to_string()),
            kernel_version,
            kernel_path,
            config,
            output_directory: output_directory.map(PathBuf::from),
            chunk_prefix: chunk_prefix.unwrap_or_else(|| "kernel".to_string()),
            validate_schema: true,
            sort_chunks: true,
        };

        let builder = ManifestBuilder::new(builder_config).map_err(|e| {
            pyo3::exceptions::PyRuntimeError::new_err(format!("Failed to create manifest builder: {}", e))
        })?;

        Ok(PyManifestBuilder { builder })
    }

    /// Add a chunk to the manifest
    fn add_chunk(
        &mut self,
        file_path: String,
        subsystem: String,
        symbol_count: Option<usize>,
        entry_point_count: Option<usize>,
        file_count: Option<usize>,
    ) -> PyResult<String> {
        use std::path::PathBuf;

        let chunk_input = ChunkInput {
            file_path: PathBuf::from(file_path),
            subsystem,
            symbol_count: symbol_count.unwrap_or(0),
            entry_point_count: entry_point_count.unwrap_or(0),
            file_count: file_count.unwrap_or(0),
        };

        let chunk_id = self.builder.add_chunk(chunk_input).map_err(|e| {
            pyo3::exceptions::PyRuntimeError::new_err(format!("Failed to add chunk: {}", e))
        })?;

        Ok(chunk_id)
    }

    /// Build the final manifest
    fn build(&mut self) -> PyResult<PyChunkManifest> {
        let manifest = self.builder.build().map_err(|e| {
            pyo3::exceptions::PyRuntimeError::new_err(format!("Failed to build manifest: {}", e))
        })?;

        Ok(convert_chunk_manifest(manifest))
    }

    /// Build manifest and write to file
    fn build_and_write(&self, file_path: &str) -> PyResult<PyChunkManifest> {
        use std::path::Path;

        let manifest = self.builder.build_and_write(Path::new(file_path)).map_err(|e| {
            pyo3::exceptions::PyRuntimeError::new_err(format!("Failed to build and write manifest: {}", e))
        })?;

        Ok(convert_chunk_manifest(manifest))
    }
}

#[pymethods]
impl PyChecksumCalculator {
    #[new]
    fn new(
        verify_on_read: Option<bool>,
        buffer_size: Option<usize>,
        cache_checksums: Option<bool>,
    ) -> PyResult<Self> {
        let config = ChecksumConfig {
            verify_on_read: verify_on_read.unwrap_or(false),
            buffer_size: buffer_size.unwrap_or(64 * 1024), // 64KB default
            cache_checksums: cache_checksums.unwrap_or(true),
            ..Default::default()
        };

        let calculator = ChecksumCalculator::new(config).map_err(|e| {
            pyo3::exceptions::PyRuntimeError::new_err(format!("Failed to create checksum calculator: {}", e))
        })?;

        Ok(PyChecksumCalculator { calculator })
    }

    /// Calculate SHA256 checksum for data
    fn calculate_sha256(&mut self, data: &[u8]) -> PyResult<String> {
        self.calculator.calculate_sha256(data).map_err(|e| {
            pyo3::exceptions::PyRuntimeError::new_err(format!("Failed to calculate checksum: {}", e))
        })
    }

    /// Calculate SHA256 checksum for a file
    fn calculate_sha256_file(&mut self, file_path: &str) -> PyResult<String> {
        use std::path::Path;

        self.calculator.calculate_sha256_file(Path::new(file_path)).map_err(|e| {
            pyo3::exceptions::PyRuntimeError::new_err(format!("Failed to calculate file checksum: {}", e))
        })
    }

    /// Verify data against expected checksum
    fn verify_checksum(&mut self, data: &[u8], expected: &str) -> PyResult<bool> {
        self.calculator.verify_checksum(data, expected).map_err(|e| {
            pyo3::exceptions::PyRuntimeError::new_err(format!("Failed to verify checksum: {}", e))
        })
    }

    /// Verify file against expected checksum
    fn verify_file_checksum(&mut self, file_path: &str, expected: &str) -> PyResult<bool> {
        use std::path::Path;

        self.calculator.verify_file_checksum(Path::new(file_path), expected).map_err(|e| {
            pyo3::exceptions::PyRuntimeError::new_err(format!("Failed to verify file checksum: {}", e))
        })
    }

    /// Clear the checksum cache
    fn clear_cache(&mut self) {
        self.calculator.clear_cache();
    }

    /// Get the number of entries in the cache
    fn get_cache_size(&self) -> usize {
        self.calculator.get_cache_size()
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

/// Convert Rust ChunkInfo to Python representation
fn convert_chunk_info(info: ChunkInfo) -> PyChunkInfo {
    PyChunkInfo {
        chunk_id: info.chunk_id,
        size_bytes: info.size_bytes,
        checksum_sha256: info.checksum_sha256,
        format: info.format,
        compressed_size: info.compressed_size,
        uncompressed_size: info.uncompressed_size,
        compression_ratio: info.compression_ratio,
        item_count: info.item_count,
    }
}

/// Convert Rust ChunkManifest to Python representation
fn convert_chunk_manifest(manifest: ChunkManifest) -> PyChunkManifest {
    let chunks = manifest
        .chunks
        .into_iter()
        .map(convert_chunk_metadata)
        .collect();

    PyChunkManifest {
        version: manifest.version,
        created: manifest.created,
        kernel_version: manifest.kernel_version,
        kernel_path: manifest.kernel_path,
        config: manifest.config,
        total_chunks: manifest.total_chunks,
        total_size_bytes: manifest.total_size_bytes,
        chunks,
    }
}

/// Convert Rust ChunkMetadata to Python representation
fn convert_chunk_metadata(metadata: ChunkMetadata) -> PyChunkMetadata {
    PyChunkMetadata {
        id: metadata.id,
        sequence: metadata.sequence,
        file: metadata.file,
        subsystem: metadata.subsystem,
        size_bytes: metadata.size_bytes,
        checksum_sha256: metadata.checksum_sha256,
        symbol_count: metadata.symbol_count,
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

/// Extract and enhance symbols with Clang semantic analysis
#[pyfunction]
fn enhance_symbols_with_clang(
    file_path: &str,
    compile_commands_path: Option<&str>,
) -> PyResult<Vec<PySymbolInfo>> {
    use std::path::PathBuf;

    // Create a parser with Clang enabled
    let config = ExtendedParserConfig {
        use_clang: true,
        compile_commands_path: compile_commands_path.map(PathBuf::from),
        include_paths: Vec::new(),
        defines: HashMap::new(),
        arch: "x86_64".to_string(),
        config_name: "defconfig".to_string(),
        include_call_graphs: false,
    };

    let mut parser = Parser::new(config).map_err(|e| {
        pyo3::exceptions::PyRuntimeError::new_err(format!("Parser creation failed: {}", e))
    })?;

    let result = parser
        .parse_file(file_path)
        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("Parse error: {}", e)))?;

    Ok(result
        .symbols
        .into_iter()
        .map(|s| PySymbolInfo {
            name: s.name,
            kind: format!("{:?}", s.kind).to_lowercase(),
            file_path: file_path.to_string(),
            start_line: s.start_line,
            end_line: s.end_line,
            signature: Some(s.signature),
        })
        .collect())
}

/// Extract entry points from a kernel directory
#[pyfunction]
#[allow(clippy::too_many_arguments)]
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
    let entry_points = extractor.extract_from_directory(kernel_dir).map_err(|e| {
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
            metadata: ep
                .metadata
                .map(|m| m.into_iter().map(|(k, v)| (k, v.to_string())).collect()),
        })
        .collect())
}

/// Utility function to create and write a complete chunk with manifest
#[pyfunction]
fn write_chunk_with_manifest(
    chunk_id: &str,
    data: &[u8],
    output_dir: &str,
    subsystem: &str,
    kernel_version: Option<&str>,
    kernel_path: Option<&str>,
    config: Option<&str>,
) -> PyResult<String> {
    use std::path::PathBuf;

    // Create chunk writer
    let config_writer = ChunkWriterConfig {
        max_chunk_size: 50 * 1024 * 1024, // 50MB
        output_directory: Some(PathBuf::from(output_dir)),
        include_metadata: true,
        ..Default::default()
    };

    let mut writer = ChunkWriter::new(config_writer).map_err(|e| {
        pyo3::exceptions::PyRuntimeError::new_err(format!("Failed to create chunk writer: {}", e))
    })?;

    // Write chunk
    let chunk_info = writer.write_chunk(chunk_id, data).map_err(|e| {
        pyo3::exceptions::PyRuntimeError::new_err(format!("Failed to write chunk: {}", e))
    })?;

    // Create manifest builder
    let manifest_config = ManifestBuilderConfig {
        version: "1.0.0".to_string(),
        kernel_version: kernel_version.map(|s| s.to_string()),
        kernel_path: kernel_path.map(|s| s.to_string()),
        config: config.map(|s| s.to_string()),
        output_directory: Some(PathBuf::from(output_dir)),
        chunk_prefix: "kernel".to_string(),
        validate_schema: true,
        sort_chunks: true,
    };

    let mut manifest_builder = ManifestBuilder::new(manifest_config).map_err(|e| {
        pyo3::exceptions::PyRuntimeError::new_err(format!("Failed to create manifest builder: {}", e))
    })?;

    // Add chunk to manifest
    let chunk_input = ChunkInput {
        file_path: PathBuf::from(output_dir).join(format!("{}.json", chunk_id)),
        subsystem: subsystem.to_string(),
        symbol_count: chunk_info.item_count,
        entry_point_count: 0, // Could be provided as parameter
        file_count: 1,
    };

    let _chunk_id_result = manifest_builder.add_chunk(chunk_input).map_err(|e| {
        pyo3::exceptions::PyRuntimeError::new_err(format!("Failed to add chunk to manifest: {}", e))
    })?;

    // Write manifest
    let manifest_path = PathBuf::from(output_dir).join("manifest.json");
    let _manifest = manifest_builder.build_and_write(&manifest_path).map_err(|e| {
        pyo3::exceptions::PyRuntimeError::new_err(format!("Failed to write manifest: {}", e))
    })?;

    Ok(manifest_path.to_string_lossy().to_string())
}

/// Utility function to load and verify a chunk from file
#[pyfunction]
fn load_and_verify_chunk(file_path: &str, expected_checksum: &str) -> PyResult<Vec<u8>> {
    use std::fs;
    use std::path::Path;

    // Read file
    let data = fs::read(file_path).map_err(|e| {
        pyo3::exceptions::PyRuntimeError::new_err(format!("Failed to read chunk file: {}", e))
    })?;

    // Create checksum calculator
    let config = ChecksumConfig::default();
    let mut calculator = ChecksumCalculator::new(config).map_err(|e| {
        pyo3::exceptions::PyRuntimeError::new_err(format!("Failed to create checksum calculator: {}", e))
    })?;

    // Verify checksum
    let is_valid = calculator.verify_file_checksum(Path::new(file_path), expected_checksum).map_err(|e| {
        pyo3::exceptions::PyRuntimeError::new_err(format!("Failed to verify checksum: {}", e))
    })?;

    if !is_valid {
        return Err(pyo3::exceptions::PyRuntimeError::new_err(
            "Chunk checksum verification failed"
        ));
    }

    Ok(data)
}

/// Utility function to load manifest from file
#[pyfunction]
fn load_manifest(manifest_path: &str) -> PyResult<PyChunkManifest> {
    use std::fs;

    let manifest_data = fs::read_to_string(manifest_path).map_err(|e| {
        pyo3::exceptions::PyRuntimeError::new_err(format!("Failed to read manifest file: {}", e))
    })?;

    let manifest: ChunkManifest = serde_json::from_str(&manifest_data).map_err(|e| {
        pyo3::exceptions::PyRuntimeError::new_err(format!("Failed to parse manifest JSON: {}", e))
    })?;

    Ok(convert_chunk_manifest(manifest))
}

/// Python module definition
#[pymodule]
fn kcs_python_bridge(_py: Python, m: &Bound<'_, pyo3::types::PyModule>) -> PyResult<()> {
    // Parser classes
    m.add_class::<PyParser>()?;
    m.add_class::<PySymbolInfo>()?;
    m.add_class::<PyParseResult>()?;
    m.add_class::<PyEntryPoint>()?;
    m.add_class::<PyKernelPattern>()?;

    // Chunk coordination classes
    m.add_class::<PyChunkWriter>()?;
    m.add_class::<PyChunkInfo>()?;
    m.add_class::<PyManifestBuilder>()?;
    m.add_class::<PyChunkManifest>()?;
    m.add_class::<PyChunkMetadata>()?;
    m.add_class::<PyChecksumCalculator>()?;

    // Parser functions
    m.add_function(wrap_pyfunction!(parse_c_file, m)?)?;
    m.add_function(wrap_pyfunction!(detect_patterns, m)?)?;
    m.add_function(wrap_pyfunction!(extract_entry_points, m)?)?;
    m.add_function(wrap_pyfunction!(enhance_symbols_with_clang, m)?)?;

    // Chunk coordination functions
    m.add_function(wrap_pyfunction!(write_chunk_with_manifest, m)?)?;
    m.add_function(wrap_pyfunction!(load_and_verify_chunk, m)?)?;
    m.add_function(wrap_pyfunction!(load_manifest, m)?)?;

    Ok(())
}
