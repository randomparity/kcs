//! KCS Parser Library
//!
//! Fast kernel source code parsing using tree-sitter and clang integration.
//! Provides structured parsing of C code with kernel-specific patterns.

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::{Path, PathBuf};

/// Symbol information for Python bridge
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SymbolInfo {
    pub name: String,
    pub kind: String,
    pub file_path: String,
    pub start_line: u32,
    pub end_line: u32,
    pub signature: Option<String>,
}

/// Parse results for Python bridge
#[derive(Debug, Default)]
pub struct ParseResult {
    pub symbols: Vec<SymbolInfo>,
    pub call_edges: Vec<CallEdge>,
    pub errors: Vec<String>,
}

/// Configuration for Python bridge
#[derive(Debug, Clone)]
pub struct ParserConfig {
    pub tree_sitter_enabled: bool,
    pub clang_enabled: bool,
    pub target_arch: String,
    pub kernel_version: String,
}

impl Default for ParserConfig {
    fn default() -> Self {
        Self {
            tree_sitter_enabled: true,
            clang_enabled: false,
            target_arch: "x86_64".to_string(),
            kernel_version: "6.1".to_string(),
        }
    }
}

pub mod ast;
pub mod call_extractor;
pub mod clang_bridge;
pub mod config;
pub mod kernel_patterns;
pub mod tree_sitter_parser;
pub mod types;

// Re-export key types for public API
pub use types::{CallEdge, CallType};

/// Represents a parsed source file
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ParsedFile {
    pub path: PathBuf,
    pub sha: String,
    pub config: String,
    pub symbols: Vec<Symbol>,
    pub includes: Vec<String>,
    pub macros: Vec<Macro>,
}

/// Represents a symbol in the kernel (function, struct, variable, etc.)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Symbol {
    pub name: String,
    pub kind: SymbolKind,
    pub start_line: u32,
    pub end_line: u32,
    pub start_col: u32,
    pub end_col: u32,
    pub signature: String,
    pub visibility: Visibility,
    pub attributes: Vec<String>,
}

/// Types of symbols we can parse
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum SymbolKind {
    Function,
    Struct,
    Union,
    Enum,
    Typedef,
    Variable,
    Macro,
    Constant,
}

/// Symbol visibility
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum Visibility {
    Static,
    Global,
    Extern,
}

/// Macro definition
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Macro {
    pub name: String,
    pub definition: String,
    pub start_line: u32,
    pub parameters: Vec<String>,
}

/// Extended parser configuration
#[derive(Debug, Clone)]
pub struct ExtendedParserConfig {
    pub use_clang: bool,
    pub compile_commands_path: Option<PathBuf>,
    pub include_paths: Vec<PathBuf>,
    pub defines: HashMap<String, String>,
    pub arch: String,
    pub config_name: String,
}

impl Default for ExtendedParserConfig {
    fn default() -> Self {
        Self {
            use_clang: true,
            compile_commands_path: None,
            include_paths: vec![],
            defines: HashMap::new(),
            arch: "x86_64".to_string(),
            config_name: "defconfig".to_string(),
        }
    }
}

/// Main parser interface
pub struct Parser {
    config: ExtendedParserConfig,
    tree_sitter_parser: tree_sitter_parser::TreeSitterParser,
    clang_bridge: Option<clang_bridge::ClangBridge>,
}

impl Parser {
    /// Create a new parser with the given configuration
    pub fn new(config: ExtendedParserConfig) -> Result<Self> {
        let tree_sitter_parser = tree_sitter_parser::TreeSitterParser::new()
            .context("Failed to initialize tree-sitter parser")?;

        let clang_bridge = if config.use_clang {
            Some(
                clang_bridge::ClangBridge::new(&config)
                    .context("Failed to initialize clang bridge")?,
            )
        } else {
            None
        };

        Ok(Self {
            config,
            tree_sitter_parser,
            clang_bridge,
        })
    }

    /// Parse a single file
    pub fn parse_file<P: AsRef<Path>>(&mut self, file_path: P) -> Result<ParsedFile> {
        let file_path = file_path.as_ref();
        tracing::debug!("Parsing file: {}", file_path.display());

        // Read file content
        let content = std::fs::read_to_string(file_path)
            .with_context(|| format!("Failed to read file: {}", file_path.display()))?;

        // Calculate SHA for file version tracking
        let sha = calculate_sha(&content);

        // Parse with tree-sitter first (fast structural parsing)
        let tree_sitter_result = self
            .tree_sitter_parser
            .parse(&content)
            .context("Tree-sitter parsing failed")?;

        // Enhance with clang if available (semantic accuracy)
        let symbols = if let Some(ref mut clang_bridge) = self.clang_bridge {
            clang_bridge
                .enhance_symbols(file_path, &tree_sitter_result.symbols)
                .context("Clang enhancement failed")?
        } else {
            tree_sitter_result.symbols
        };

        Ok(ParsedFile {
            path: file_path.to_path_buf(),
            sha,
            config: format!("{}:{}", self.config.arch, self.config.config_name),
            symbols,
            includes: tree_sitter_result.includes,
            macros: tree_sitter_result.macros,
        })
    }

    /// Parse multiple files in parallel
    pub fn parse_files<P: AsRef<Path>>(&mut self, file_paths: &[P]) -> Result<Vec<ParsedFile>> {
        // Note: Simplified sequential implementation for now
        // Parallel processing with clang is complex due to state management
        let mut results = Vec::new();
        let total_files = file_paths.len();

        for (index, path) in file_paths.iter().enumerate() {
            let path_ref = path.as_ref();

            // Show progress every 500 files for large batches, 50 for medium, or 10 for small
            let should_show_progress = if total_files > 5000 {
                index % 500 == 0 || index == total_files - 1
            } else if total_files > 1000 {
                index % 100 == 0 || index == total_files - 1
            } else if total_files > 100 {
                index % 50 == 0 || index == total_files - 1
            } else {
                index % 10 == 0 || index == total_files - 1
            };

            if should_show_progress {
                let progress = ((index + 1) as f32 / total_files as f32 * 100.0) as u32;
                tracing::info!(
                    "📁 Progress: {}/{} files ({}%) - Currently: {}",
                    index + 1,
                    total_files,
                    progress,
                    path_ref.file_name().unwrap_or_default().to_string_lossy()
                );
            }

            match self.parse_file(path) {
                Ok(parsed_file) => results.push(parsed_file),
                Err(e) => {
                    tracing::warn!("Failed to parse {}: {}", path_ref.display(), e);
                    // Continue processing other files instead of failing completely
                }
            }
        }

        tracing::info!("Parsed {} files", results.len());
        Ok(results)
    }

    /// Parse files with content provided directly (for Python bridge)
    pub fn parse_file_content(&mut self, file_path: &str, content: &str) -> Result<ParseResult> {
        tracing::debug!("Parsing file content: {}", file_path);

        // Parse with tree-sitter
        let tree_sitter_result = self
            .tree_sitter_parser
            .parse(content)
            .context("Tree-sitter parsing failed")?;

        // Convert to bridge format
        let symbols = tree_sitter_result
            .symbols
            .into_iter()
            .map(|s| SymbolInfo {
                name: s.name,
                kind: format!("{:?}", s.kind).to_lowercase(),
                file_path: file_path.to_string(),
                start_line: s.start_line,
                end_line: s.end_line,
                signature: Some(s.signature),
            })
            .collect();

        // Extract call edges from AST using call extractor
        let call_extractor =
            call_extractor::CallExtractor::new().context("Failed to create call extractor")?;

        let call_result = call_extractor
            .extract_calls(&tree_sitter_result.tree, content, file_path)
            .context("Failed to extract call edges")?;

        let call_edges = call_result.call_edges;

        Ok(ParseResult {
            symbols,
            call_edges,
            errors: vec![],
        })
    }

    /// Parse multiple files with content (for Python bridge batch operations)
    pub fn parse_files_content(&mut self, files: HashMap<String, String>) -> Result<ParseResult> {
        let mut all_symbols = Vec::new();
        let mut all_call_edges = Vec::new();
        let mut all_errors = Vec::new();

        for (file_path, content) in files {
            match self.parse_file_content(&file_path, &content) {
                Ok(result) => {
                    all_symbols.extend(result.symbols);
                    all_call_edges.extend(result.call_edges);
                    all_errors.extend(result.errors);
                }
                Err(e) => {
                    all_errors.push(format!("Failed to parse {}: {}", file_path, e));
                }
            }
        }

        Ok(ParseResult {
            symbols: all_symbols,
            call_edges: all_call_edges,
            errors: all_errors,
        })
    }

    /// Parse kernel tree (for Python bridge)
    pub fn parse_kernel_tree(
        &mut self,
        kernel_path: &str,
        config_name: &str,
    ) -> Result<ParseResult> {
        tracing::info!(
            "Parsing kernel tree: {} with config: {}",
            kernel_path,
            config_name
        );

        // Update parser config
        self.config.config_name = config_name.to_string();

        // Parse directory
        let parsed_files = self.parse_directory(kernel_path)?;

        // Convert to bridge format
        let mut all_symbols = Vec::new();
        let all_call_edges = Vec::new();

        for parsed_file in parsed_files {
            for symbol in parsed_file.symbols {
                all_symbols.push(SymbolInfo {
                    name: symbol.name,
                    kind: format!("{:?}", symbol.kind).to_lowercase(),
                    file_path: parsed_file.path.to_string_lossy().to_string(),
                    start_line: symbol.start_line,
                    end_line: symbol.end_line,
                    signature: Some(symbol.signature),
                });
            }
        }

        Ok(ParseResult {
            symbols: all_symbols,
            call_edges: all_call_edges,
            errors: vec![],
        })
    }

    /// Reconfigure parser (for Python bridge)
    pub fn reconfigure(&mut self, config: ParserConfig) -> Result<()> {
        // Update basic config from bridge config
        self.config.arch = config.target_arch;
        self.config.config_name = config.kernel_version;

        // Reinitialize clang bridge if needed
        if config.clang_enabled && self.clang_bridge.is_none() {
            self.clang_bridge = Some(
                clang_bridge::ClangBridge::new(&self.config)
                    .context("Failed to initialize clang bridge")?,
            );
        } else if !config.clang_enabled {
            self.clang_bridge = None;
        }

        Ok(())
    }

    /// Parse an entire directory recursively
    pub fn parse_directory<P: AsRef<Path>>(&mut self, dir_path: P) -> Result<Vec<ParsedFile>> {
        let dir_path = dir_path.as_ref();
        tracing::info!("Parsing directory: {}", dir_path.display());

        let c_files = find_c_files(dir_path)?;
        tracing::info!("Found {} C files", c_files.len());

        self.parse_files(&c_files)
    }
}

/// Calculate SHA256 hash of content for version tracking
fn calculate_sha(content: &str) -> String {
    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash, Hasher};

    let mut hasher = DefaultHasher::new();
    content.hash(&mut hasher);
    format!("{:x}", hasher.finish())
}

/// Find all C source files in a directory
fn find_c_files<P: AsRef<Path>>(dir_path: P) -> Result<Vec<PathBuf>> {
    use walkdir::WalkDir;

    let mut c_files = Vec::new();

    for entry in WalkDir::new(dir_path)
        .follow_links(false)
        .into_iter()
        .filter_map(|e| e.ok())
    {
        let path = entry.path();
        if let Some(extension) = path.extension() {
            if matches!(extension.to_str(), Some("c") | Some("h") | Some("S")) {
                // Skip some common non-kernel files
                if !should_skip_file(path) {
                    c_files.push(path.to_path_buf());
                }
            }
        }
    }

    Ok(c_files)
}

/// Determine if a file should be skipped during parsing
fn should_skip_file(path: &Path) -> bool {
    let path_str = path.to_string_lossy();

    // Skip test files, build artifacts, etc.
    path_str.contains("/tools/")
        || path_str.contains("/scripts/")
        || path_str.contains("/Documentation/")
        || path_str.ends_with(".tmp")  // Only skip files that END with .tmp, not paths containing .tmp
        || path_str.contains(".mod.c")
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    #[test]
    fn test_parser_creation() {
        let config = ExtendedParserConfig::default();
        let parser = Parser::new(config);
        assert!(parser.is_ok());
    }

    #[test]
    fn test_find_c_files() {
        let temp_dir = tempdir().unwrap();
        let temp_path = temp_dir.path();

        // Create test files
        std::fs::write(temp_path.join("test.c"), "int main() {}").unwrap();
        std::fs::write(temp_path.join("test.h"), "#include <stdio.h>").unwrap();
        std::fs::write(temp_path.join("test.txt"), "not a c file").unwrap();

        let c_files = find_c_files(temp_path).unwrap();
        assert_eq!(c_files.len(), 2);
    }

    #[test]
    fn test_calculate_sha() {
        let content1 = "int main() { return 0; }";
        let content2 = "int main() { return 1; }";

        let sha1 = calculate_sha(content1);
        let sha2 = calculate_sha(content2);

        assert_ne!(sha1, sha2);
        assert!(!sha1.is_empty());
        assert!(!sha2.is_empty());
    }
}
