//! Clang integration for semantic analysis
//!
//! Enhances tree-sitter results with clang's semantic understanding

use anyhow::{Context, Result};
use std::collections::HashMap;
use std::path::{Path, PathBuf};

use crate::{ExtendedParserConfig, Symbol, SymbolKind};

#[cfg(feature = "clang")]
use clang_sys::*;

#[cfg(feature = "clang")]
use std::ffi::CString;

#[cfg(feature = "clang")]
use std::ffi::CStr;

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

    /// Extract detailed type information from a Clang cursor
    #[cfg(feature = "clang")]
    fn extract_type_info(cursor: CXCursor) -> HashMap<String, serde_json::Value> {
        let mut type_info = HashMap::new();

        // Get the type of the cursor
        let cursor_type = unsafe { clang_getCursorType(cursor) };

        // Get the return type for functions
        let cursor_kind = unsafe { clang_getCursorKind(cursor) };

        if cursor_kind == CXCursor_FunctionDecl || cursor_kind == CXCursor_CXXMethod {
            // Get return type
            let result_type = unsafe { clang_getResultType(cursor_type) };
            let result_type_spelling = unsafe {
                let spelling = clang_getTypeSpelling(result_type);
                Self::cxstring_to_string(spelling)
            };
            if !result_type_spelling.is_empty() {
                type_info.insert(
                    "return_type".to_string(),
                    serde_json::Value::String(result_type_spelling),
                );
            }

            // Get parameter types
            let num_args = unsafe { clang_getNumArgTypes(cursor_type) };
            if num_args >= 0 {
                let mut params = Vec::new();
                for i in 0..num_args {
                    let arg_type = unsafe { clang_getArgType(cursor_type, i as u32) };
                    let arg_type_spelling = unsafe {
                        let spelling = clang_getTypeSpelling(arg_type);
                        Self::cxstring_to_string(spelling)
                    };

                    // Try to get parameter name
                    let arg_cursor = unsafe { clang_Cursor_getArgument(cursor, i as u32) };
                    let arg_name = unsafe {
                        let spelling = clang_getCursorSpelling(arg_cursor);
                        Self::cxstring_to_string(spelling)
                    };

                    let mut param_info = serde_json::Map::new();
                    param_info.insert(
                        "type".to_string(),
                        serde_json::Value::String(arg_type_spelling),
                    );
                    if !arg_name.is_empty() {
                        param_info.insert("name".to_string(), serde_json::Value::String(arg_name));
                    }
                    params.push(serde_json::Value::Object(param_info));
                }
                if !params.is_empty() {
                    type_info.insert("parameters".to_string(), serde_json::Value::Array(params));
                }
            }

            // Check if function is variadic
            let is_variadic = unsafe { clang_isFunctionTypeVariadic(cursor_type) };
            if is_variadic != 0 {
                type_info.insert("is_variadic".to_string(), serde_json::Value::Bool(true));
            }
        }

        // Note: Storage class extraction removed due to compatibility issues
        // Could be re-added with proper clang-sys version detection

        // Get type spelling
        let type_spelling = unsafe {
            let spelling = clang_getTypeSpelling(cursor_type);
            Self::cxstring_to_string(spelling)
        };
        if !type_spelling.is_empty() {
            type_info.insert(
                "type_spelling".to_string(),
                serde_json::Value::String(type_spelling.clone()),
            );
        }

        // Check if it's a definition vs declaration
        let is_definition = unsafe { clang_isCursorDefinition(cursor) };
        if is_definition != 0 {
            type_info.insert("is_definition".to_string(), serde_json::Value::Bool(true));
        }

        // Get canonical type (resolved typedef)
        let canonical_type = unsafe { clang_getCanonicalType(cursor_type) };
        let canonical_spelling = unsafe {
            let spelling = clang_getTypeSpelling(canonical_type);
            Self::cxstring_to_string(spelling)
        };

        let type_kind = cursor_type.kind;
        if type_kind == CXType_Typedef && canonical_spelling != type_spelling {
            type_info.insert(
                "canonical_type".to_string(),
                serde_json::Value::String(canonical_spelling),
            );
        }

        type_info
    }

    /// Convert a CXString to a Rust String
    #[cfg(feature = "clang")]
    fn cxstring_to_string(cx_string: CXString) -> String {
        unsafe {
            let c_str = clang_getCString(cx_string);
            if c_str.is_null() {
                String::new()
            } else {
                let result = CStr::from_ptr(c_str).to_string_lossy().to_string();
                clang_disposeString(cx_string);
                result
            }
        }
    }

    /// Find a cursor at a specific location
    #[cfg(feature = "clang")]
    fn find_cursor_at_location(
        tu: CXTranslationUnit,
        line: u32,
        column: u32,
        file_path: &Path,
    ) -> Option<CXCursor> {
        // Get the file handle
        let c_file_path = CString::new(file_path.to_str()?).ok()?;
        let file = unsafe { clang_getFile(tu, c_file_path.as_ptr()) };
        if file.is_null() {
            return None;
        }

        // Get location at line and column
        let location = unsafe { clang_getLocation(tu, file, line, column) };

        // Get cursor at that location
        let cursor = unsafe { clang_getCursor(tu, location) };

        // Check if cursor is valid
        let is_invalid = unsafe { clang_isInvalid(clang_getCursorKind(cursor)) };
        if is_invalid != 0 {
            return None;
        }

        Some(cursor)
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
            let tu = match self.get_translation_unit(file_path) {
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

            // Enhance each symbol
            let mut enhanced_symbols = Vec::new();
            for symbol in symbols {
                let mut enhanced = symbol.clone();

                // Try to find the cursor for this symbol
                if let Some(cursor) = Self::find_cursor_at_location(
                    tu,
                    symbol.start_line,
                    symbol.start_col,
                    file_path,
                ) {
                    // Extract type information
                    let type_info = Self::extract_type_info(cursor);

                    if !type_info.is_empty() {
                        // Initialize metadata if needed
                        if enhanced.metadata.is_none() {
                            enhanced.metadata = Some(serde_json::Map::new());
                        }

                        // Add type information to metadata
                        if let Some(ref mut metadata) = enhanced.metadata {
                            for (key, value) in type_info {
                                metadata.insert(key, value);
                            }
                        }
                    }

                    // For functions, also try to enhance the signature
                    if matches!(symbol.kind, SymbolKind::Function) {
                        // Get the full type spelling which includes parameter names
                        let cursor_type = unsafe { clang_getCursorType(cursor) };
                        let type_spelling = unsafe {
                            let spelling = clang_getTypeSpelling(cursor_type);
                            Self::cxstring_to_string(spelling)
                        };

                        // If we got a better signature from Clang, use it
                        if !type_spelling.is_empty() && type_spelling != enhanced.signature {
                            if let Some(ref mut metadata) = enhanced.metadata {
                                metadata.insert(
                                    "enhanced_signature".to_string(),
                                    serde_json::Value::String(type_spelling),
                                );
                            }
                        }
                    }
                }

                enhanced_symbols.push(enhanced);
            }

            Ok(enhanced_symbols)
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
            metadata: None,
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
            metadata: None,
        }];

        // Should not fail even with clang enabled
        let enhanced = bridge.enhance_symbols(&test_file, &symbols)?;
        assert_eq!(enhanced.len(), symbols.len());

        Ok(())
    }

    #[cfg(feature = "clang")]
    #[test]
    fn test_symbol_type_extraction() -> Result<()> {
        let config = ExtendedParserConfig {
            use_clang: true,
            ..Default::default()
        };

        // Create a temporary C file with more complex types
        let temp_dir = tempdir()?;
        let test_file = temp_dir.path().join("test.c");
        let mut file = File::create(&test_file)?;
        writeln!(
            file,
            r#"
static int add(int a, int b) {{
    return a + b;
}}

int multiply(int x, int y) {{
    return x * y;
}}
"#
        )?;

        let mut bridge = ClangBridge::new(&config)?;

        let symbols = vec![
            Symbol {
                name: "add".to_string(),
                kind: crate::SymbolKind::Function,
                start_line: 2,
                end_line: 4,
                start_col: 0,
                end_col: 0,
                signature: "static int add(int a, int b)".to_string(),
                visibility: crate::Visibility::Static,
                attributes: vec![],
                metadata: None,
            },
            Symbol {
                name: "multiply".to_string(),
                kind: crate::SymbolKind::Function,
                start_line: 6,
                end_line: 8,
                start_col: 0,
                end_col: 0,
                signature: "int multiply(int x, int y)".to_string(),
                visibility: crate::Visibility::Global,
                attributes: vec![],
                metadata: None,
            },
        ];

        let enhanced = bridge.enhance_symbols(&test_file, &symbols)?;
        assert_eq!(enhanced.len(), symbols.len());

        // Check that metadata was added
        for symbol in &enhanced {
            if symbol.name == "add" {
                assert!(symbol.metadata.is_some());
                if let Some(ref metadata) = symbol.metadata {
                    // Should have return type and parameters
                    assert!(
                        metadata.contains_key("return_type")
                            || metadata.contains_key("type_spelling")
                    );
                }
            }
        }

        Ok(())
    }
}
