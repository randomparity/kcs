//! Clang integration for semantic analysis
//!
//! Enhances tree-sitter results with clang's semantic understanding

use anyhow::{Context, Result};
use std::collections::HashMap;
use std::path::{Path, PathBuf};

use crate::{ExtendedParserConfig, Symbol};

#[cfg(feature = "clang")]
use clang_sys::*;

#[cfg(feature = "clang")]
use std::ffi::CString;

/// Wrapper around Clang's index for semantic analysis
pub struct ClangBridge {
    config: ExtendedParserConfig,
    #[cfg(feature = "clang")]
    index: Option<CXIndex>,
    /// Cached translation units for files
    #[cfg(feature = "clang")]
    translation_units: HashMap<PathBuf, CXTranslationUnit>,
    /// Compilation database if available
    compilation_database: Option<CompilationDatabase>,
}

/// Represents compilation database information
#[derive(Debug, Clone)]
pub struct CompilationDatabase {
    /// Map from file path to compilation command
    commands: HashMap<PathBuf, CompileCommand>,
}

/// Single compilation command from compile_commands.json
#[derive(Debug, Clone)]
pub struct CompileCommand {
    pub directory: PathBuf,
    pub command: String,
    pub arguments: Vec<String>,
    pub file: PathBuf,
}

impl ClangBridge {
    /// Create a new ClangBridge with the given configuration
    pub fn new(config: &ExtendedParserConfig) -> Result<Self> {
        // Initialize clang index if the feature is enabled
        #[cfg(feature = "clang")]
        let index = if config.use_clang {
            // Safety: clang_createIndex is safe to call with valid arguments
            // excludeDeclarationsFromPCH = 0 (false), displayDiagnostics = 0 (false)
            let idx = unsafe { clang_createIndex(0, 0) };
            if idx.is_null() {
                anyhow::bail!("Failed to create Clang index");
            }
            Some(idx)
        } else {
            None
        };

        // Load compilation database if provided
        let compilation_database = if let Some(ref path) = config.compile_commands_path {
            Self::load_compilation_database(path)?
        } else {
            None
        };

        Ok(Self {
            config: config.clone(),
            #[cfg(feature = "clang")]
            index,
            #[cfg(feature = "clang")]
            translation_units: HashMap::new(),
            compilation_database,
        })
    }

    /// Load compilation database from compile_commands.json
    fn load_compilation_database(path: &Path) -> Result<Option<CompilationDatabase>> {
        if !path.exists() {
            return Ok(None);
        }

        let content = std::fs::read_to_string(path)
            .with_context(|| format!("Failed to read compile_commands.json at {:?}", path))?;

        let commands: Vec<serde_json::Value> = serde_json::from_str(&content)
            .with_context(|| format!("Failed to parse compile_commands.json at {:?}", path))?;

        let mut db = HashMap::new();

        for cmd in commands {
            let file = cmd.get("file").and_then(|f| f.as_str());
            let directory = cmd.get("directory").and_then(|d| d.as_str());
            let command = cmd.get("command").and_then(|c| c.as_str());

            if let (Some(file), Some(directory)) = (file, directory) {
                // Note: command can be None if arguments is provided instead
                let arguments = if let Some(args) = cmd.get("arguments") {
                    args.as_array()
                        .map(|arr| {
                            arr.iter()
                                .filter_map(|v| v.as_str().map(String::from))
                                .collect()
                        })
                        .unwrap_or_else(Vec::new)
                } else if let Some(command) = command {
                    Self::parse_command_line(command)
                } else {
                    Vec::new()
                };

                let file_path = PathBuf::from(file);
                let compile_cmd = CompileCommand {
                    directory: PathBuf::from(directory),
                    command: command.unwrap_or("").to_string(),
                    arguments,
                    file: file_path.clone(),
                };

                db.insert(file_path, compile_cmd);
            }
        }

        Ok(Some(CompilationDatabase { commands: db }))
    }

    /// Parse a command line string into arguments
    fn parse_command_line(command: &str) -> Vec<String> {
        // Simple parsing - could be improved with proper shell parsing
        command.split_whitespace().map(String::from).collect()
    }

    /// Get or create a translation unit for the given file
    #[cfg(feature = "clang")]
    fn get_translation_unit(&mut self, file_path: &Path) -> Result<CXTranslationUnit> {
        // Check if we already have a cached translation unit
        if let Some(tu) = self.translation_units.get(file_path) {
            return Ok(*tu);
        }

        let index = self
            .index
            .ok_or_else(|| anyhow::anyhow!("Clang index not initialized"))?;

        // Prepare compilation arguments
        let mut clang_args = Vec::new();

        // Add include paths
        for include_path in &self.config.include_paths {
            clang_args.push(format!("-I{}", include_path.display()));
        }

        // Add defines
        for (key, value) in &self.config.defines {
            if value.is_empty() {
                clang_args.push(format!("-D{}", key));
            } else {
                clang_args.push(format!("-D{}={}", key, value));
            }
        }

        // Add architecture-specific flags
        if !self.config.arch.is_empty() {
            clang_args.push(format!("-march={}", self.config.arch));
        }

        // Check if we have compilation database info for this file
        if let Some(ref db) = self.compilation_database {
            if let Some(cmd) = db.commands.get(file_path) {
                // Use the arguments from compilation database
                clang_args = cmd.arguments.clone();
                // Remove the compiler name and source file from arguments
                if !clang_args.is_empty() {
                    clang_args.remove(0); // Remove compiler
                }
                if let Some(pos) = clang_args
                    .iter()
                    .position(|arg| arg == file_path.to_str().unwrap_or(""))
                {
                    clang_args.remove(pos); // Remove source file
                }
            }
        }

        // Convert arguments to C strings
        let c_args: Vec<CString> = clang_args
            .iter()
            .filter_map(|arg| CString::new(arg.as_str()).ok())
            .collect();
        let c_arg_ptrs: Vec<*const std::os::raw::c_char> =
            c_args.iter().map(|arg| arg.as_ptr()).collect();

        // Convert file path to C string
        let c_file_path = CString::new(file_path.to_str().unwrap_or(""))
            .map_err(|_| anyhow::anyhow!("Invalid file path"))?;

        // Parse the file
        // Safety: clang_parseTranslationUnit is safe with valid inputs
        let tu = unsafe {
            clang_parseTranslationUnit(
                index,
                c_file_path.as_ptr(),
                c_arg_ptrs.as_ptr(),
                c_arg_ptrs.len() as i32,
                std::ptr::null_mut(),
                0,
                CXTranslationUnit_DetailedPreprocessingRecord | 512, /* CXTranslationUnit_KeepGoing */
            )
        };

        if tu.is_null() {
            anyhow::bail!("Failed to parse translation unit for {:?}", file_path);
        }

        // Cache the translation unit
        self.translation_units.insert(file_path.to_path_buf(), tu);

        Ok(tu)
    }

    /// Enhance symbols with Clang semantic information
    pub fn enhance_symbols(&mut self, file_path: &Path, symbols: &[Symbol]) -> Result<Vec<Symbol>> {
        #[cfg(not(feature = "clang"))]
        {
            // If clang feature is not enabled, just return the original symbols
            let _ = file_path; // Suppress unused warning
            return Ok(symbols.to_vec());
        }

        #[cfg(feature = "clang")]
        {
            if !self.config.use_clang || self.index.is_none() {
                // Clang not enabled or failed to initialize
                return Ok(symbols.to_vec());
            }

            // Get translation unit for the file
            let _tu = match self.get_translation_unit(file_path) {
                Ok(tu) => tu,
                Err(e) => {
                    eprintln!(
                        "Warning: Failed to get translation unit for {:?}: {}",
                        file_path, e
                    );
                    // Fall back to original symbols
                    return Ok(symbols.to_vec());
                }
            };

            // For now, just return the original symbols
            // Symbol enhancement will be implemented in T019 and T020
            Ok(symbols.to_vec())
        }
    }
}

impl Drop for ClangBridge {
    fn drop(&mut self) {
        #[cfg(feature = "clang")]
        {
            // Dispose of translation units
            for (_, tu) in self.translation_units.drain() {
                // Safety: clang_disposeTranslationUnit is safe with valid translation unit
                unsafe {
                    clang_disposeTranslationUnit(tu);
                }
            }

            // Dispose of the index
            if let Some(index) = self.index.take() {
                // Safety: clang_disposeIndex is safe with valid index
                unsafe {
                    clang_disposeIndex(index);
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs::File;
    use std::io::Write;
    use tempfile::tempdir;

    #[test]
    fn test_clang_bridge_creation() {
        let config = ExtendedParserConfig::default();
        let bridge = ClangBridge::new(&config);
        assert!(bridge.is_ok());
    }

    #[test]
    fn test_clang_bridge_without_clang() {
        let config = ExtendedParserConfig {
            use_clang: false,
            ..Default::default()
        };
        let bridge = ClangBridge::new(&config);
        assert!(bridge.is_ok());
    }

    #[test]
    fn test_load_compilation_database() -> Result<()> {
        let temp_dir = tempdir()?;
        let compile_commands_path = temp_dir.path().join("compile_commands.json");

        let mut file = File::create(&compile_commands_path)?;
        writeln!(
            file,
            r#"[
    {{
        "directory": "/home/user/kernel",
        "command": "gcc -c -I/usr/include -DDEBUG fs/ext4/inode.c",
        "file": "fs/ext4/inode.c"
    }},
    {{
        "directory": "/home/user/kernel",
        "arguments": ["gcc", "-c", "-I/usr/include", "-DRELEASE", "mm/page_alloc.c"],
        "file": "mm/page_alloc.c"
    }}
]"#
        )?;

        let db = ClangBridge::load_compilation_database(&compile_commands_path)?;
        assert!(db.is_some());

        let db = db.unwrap();
        assert_eq!(db.commands.len(), 2);

        let cmd1 = db.commands.get(&PathBuf::from("fs/ext4/inode.c"));
        assert!(cmd1.is_some());

        let cmd2 = db.commands.get(&PathBuf::from("mm/page_alloc.c"));
        assert!(cmd2.is_some());

        Ok(())
    }

    #[test]
    fn test_enhance_symbols_without_clang() -> Result<()> {
        let config = ExtendedParserConfig {
            use_clang: false,
            ..Default::default()
        };

        let mut bridge = ClangBridge::new(&config)?;

        let symbols = vec![Symbol {
            name: "test_func".to_string(),
            kind: crate::SymbolKind::Function,
            start_line: 1,
            end_line: 5,
            start_col: 0,
            end_col: 0,
            signature: "void test_func(void)".to_string(),
            visibility: crate::Visibility::Global,
            attributes: vec![],
        }];

        let enhanced = bridge.enhance_symbols(Path::new("test.c"), &symbols)?;
        assert_eq!(enhanced.len(), symbols.len());
        assert_eq!(enhanced[0].name, symbols[0].name);

        Ok(())
    }

    #[cfg(feature = "clang")]
    #[test]
    fn test_clang_bridge_with_clang() -> Result<()> {
        let config = ExtendedParserConfig {
            use_clang: true,
            ..Default::default()
        };

        // Create a temporary C file
        let temp_dir = tempdir()?;
        let test_file = temp_dir.path().join("test.c");
        let mut file = File::create(&test_file)?;
        writeln!(file, "int main() {{ return 0; }}")?;

        let mut bridge = ClangBridge::new(&config)?;

        let symbols = vec![Symbol {
            name: "main".to_string(),
            kind: crate::SymbolKind::Function,
            start_line: 1,
            end_line: 1,
            start_col: 0,
            end_col: 26,
            signature: "int main()".to_string(),
            visibility: crate::Visibility::Global,
            attributes: vec![],
        }];

        // Should not fail even with clang enabled
        let enhanced = bridge.enhance_symbols(&test_file, &symbols)?;
        assert_eq!(enhanced.len(), symbols.len());

        Ok(())
    }
}
