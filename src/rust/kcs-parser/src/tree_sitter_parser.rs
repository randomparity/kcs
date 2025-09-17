//! Tree-sitter based C parser for fast structural analysis

use anyhow::{Context, Result};
use tree_sitter::{Language, Parser, Query, QueryCursor, Tree};

use crate::{Macro, Symbol, SymbolKind, Visibility};

pub struct TreeSitterParser {
    parser: Parser,
    #[allow(dead_code)]
    language: Language,
    symbol_query: Query,
    include_query: Query,
    macro_query: Query,
}

pub struct TreeSitterResult {
    pub symbols: Vec<Symbol>,
    pub includes: Vec<String>,
    pub macros: Vec<Macro>,
    pub tree: Tree,
}

impl TreeSitterParser {
    pub fn new() -> Result<Self> {
        let language = tree_sitter_c::language();
        let mut parser = Parser::new();
        parser
            .set_language(language)
            .context("Failed to set C language for tree-sitter")?;

        // Query for symbols (functions, structs, etc.)
        let symbol_query = Query::new(
            language,
            r#"
            (function_definition
              declarator: (function_declarator
                declarator: (identifier) @function.name)
              body: (compound_statement)) @function.definition

            (declaration
              declarator: (function_declarator
                declarator: (identifier) @function.declaration)) @function.decl

            (struct_specifier
              name: (type_identifier) @struct.name) @struct.definition

            (union_specifier
              name: (type_identifier) @union.name) @union.definition

            (enum_specifier
              name: (type_identifier) @enum.name) @enum.definition

            (type_definition
              type: (type_identifier) @typedef.name) @typedef.definition

            (declaration
              declarator: (identifier) @variable.name) @variable.definition
            "#,
        )
        .context("Failed to create symbol query")?;

        // Query for includes
        let include_query = Query::new(
            language,
            r#"
            (preproc_include
              path: (string_literal) @include.path)
            (preproc_include
              path: (system_lib_string) @include.system)
            "#,
        )
        .context("Failed to create include query")?;

        // Query for macros
        let macro_query = Query::new(
            language,
            r#"
            (preproc_def
              name: (identifier) @macro.name
              value: (_)? @macro.value)

            (preproc_function_def
              name: (identifier) @macro.function.name
              parameters: (preproc_params) @macro.function.params
              value: (_)? @macro.function.value)
            "#,
        )
        .context("Failed to create macro query")?;

        Ok(Self {
            parser,
            language,
            symbol_query,
            include_query,
            macro_query,
        })
    }

    pub fn parse(&mut self, source: &str) -> Result<TreeSitterResult> {
        let tree = self
            .parser
            .parse(source, None)
            .context("Failed to parse source with tree-sitter")?;

        let symbols = self.extract_symbols(source, &tree)?;
        let includes = self.extract_includes(source, &tree)?;
        let macros = self.extract_macros(source, &tree)?;

        Ok(TreeSitterResult {
            symbols,
            includes,
            macros,
            tree,
        })
    }

    fn extract_symbols(&self, source: &str, tree: &Tree) -> Result<Vec<Symbol>> {
        let mut query_cursor = QueryCursor::new();
        let matches =
            query_cursor.captures(&self.symbol_query, tree.root_node(), source.as_bytes());

        let mut symbols = Vec::new();
        let source_bytes = source.as_bytes();

        for (match_, _) in matches {
            for capture in match_.captures {
                let node = capture.node;
                let capture_name = self
                    .symbol_query
                    .capture_names()
                    .get(capture.index as usize)
                    .unwrap();

                let text = node.utf8_text(source_bytes).unwrap_or("");
                let start_pos = node.start_position();
                let end_pos = node.end_position();

                let (symbol_kind, name) = match capture_name.as_str() {
                    "function.name" => (SymbolKind::Function, text.to_string()),
                    "struct.name" => (SymbolKind::Struct, text.to_string()),
                    "union.name" => (SymbolKind::Union, text.to_string()),
                    "enum.name" => (SymbolKind::Enum, text.to_string()),
                    "typedef.name" => (SymbolKind::Typedef, text.to_string()),
                    "variable.name" => (SymbolKind::Variable, text.to_string()),
                    _ => continue,
                };

                // Get the full signature/definition
                let signature = if let Some(parent) = node.parent() {
                    parent.utf8_text(source_bytes).unwrap_or(text).to_string()
                } else {
                    text.to_string()
                };

                // Determine visibility (simplified heuristic)
                let visibility = if signature.contains("static") {
                    Visibility::Static
                } else if signature.contains("extern") {
                    Visibility::Extern
                } else {
                    Visibility::Global
                };

                symbols.push(Symbol {
                    name,
                    kind: symbol_kind,
                    start_line: start_pos.row as u32 + 1,
                    end_line: end_pos.row as u32 + 1,
                    start_col: start_pos.column as u32,
                    end_col: end_pos.column as u32,
                    signature: signature.trim().to_string(),
                    visibility,
                    attributes: self.extract_attributes(&signature),
                    metadata: None,
                });
            }
        }

        Ok(symbols)
    }

    fn extract_includes(&self, source: &str, tree: &Tree) -> Result<Vec<String>> {
        let mut query_cursor = QueryCursor::new();
        let matches =
            query_cursor.captures(&self.include_query, tree.root_node(), source.as_bytes());

        let mut includes = Vec::new();
        let source_bytes = source.as_bytes();

        for (match_, _) in matches {
            for capture in match_.captures {
                let node = capture.node;
                let text = node.utf8_text(source_bytes).unwrap_or("");

                // Remove quotes from include path
                let clean_path = text.trim_matches('"').trim_matches('<').trim_matches('>');
                includes.push(clean_path.to_string());
            }
        }

        Ok(includes)
    }

    fn extract_macros(&self, source: &str, tree: &Tree) -> Result<Vec<Macro>> {
        let mut query_cursor = QueryCursor::new();
        let matches = query_cursor.matches(&self.macro_query, tree.root_node(), source.as_bytes());

        let mut macros = std::collections::HashMap::new();
        let source_bytes = source.as_bytes();

        for match_ in matches {
            let mut name = String::new();
            let mut definition = String::new();
            let mut parameters = Vec::new();
            let mut start_line = 0;

            // Process all captures for this match
            for capture in match_.captures {
                let node = capture.node;
                let capture_name = self
                    .macro_query
                    .capture_names()
                    .get(capture.index as usize)
                    .unwrap();
                let text = node.utf8_text(source_bytes).unwrap_or("");

                match capture_name.as_str() {
                    "macro.name" | "macro.function.name" => {
                        name = text.to_string();
                        start_line = node.start_position().row as u32 + 1;
                    }
                    "macro.value" | "macro.function.value" => {
                        definition = text.to_string();
                    }
                    "macro.function.params" => {
                        // Parse parameters (simplified)
                        parameters = text
                            .trim_matches('(')
                            .trim_matches(')')
                            .split(',')
                            .map(|s| s.trim().to_string())
                            .filter(|s| !s.is_empty())
                            .collect();
                    }
                    _ => {}
                }
            }

            if !name.is_empty() {
                // Use a composite key for deduplication (name + line)
                let key = (name.clone(), start_line);
                macros.insert(
                    key,
                    Macro {
                        name,
                        definition,
                        start_line,
                        parameters,
                    },
                );
            }
        }

        // Convert HashMap values to Vec
        Ok(macros.into_values().collect())
    }

    fn extract_attributes(&self, signature: &str) -> Vec<String> {
        let mut attributes = Vec::new();

        // Common kernel attributes
        let kernel_attributes = [
            "__init",
            "__exit",
            "__devinit",
            "__devexit",
            "__cpuinit",
            "__cpuexit",
            "__weak",
            "__always_inline",
            "__noinline",
            "__noreturn",
            "__pure",
            "__const",
            "__cold",
            "__hot",
            "EXPORT_SYMBOL",
            "EXPORT_SYMBOL_GPL",
            "asmlinkage",
            "fastcall",
            "__user",
            "__kernel",
            "__iomem",
            "__percpu",
        ];

        for attr in &kernel_attributes {
            if signature.contains(attr) {
                attributes.push(attr.to_string());
            }
        }

        attributes
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tree_sitter_parser_creation() {
        let parser = TreeSitterParser::new();
        assert!(parser.is_ok());
    }

    #[test]
    fn test_parse_simple_function() {
        let mut parser = TreeSitterParser::new().unwrap();
        let source = r#"
int main(void) {
    return 0;
}
"#;

        let result = parser.parse(source).unwrap();
        assert!(!result.symbols.is_empty());

        let main_func = result
            .symbols
            .iter()
            .find(|s| s.name == "main" && s.kind == SymbolKind::Function);
        assert!(main_func.is_some());
    }

    #[test]
    fn test_parse_struct() {
        let mut parser = TreeSitterParser::new().unwrap();
        let source = r#"
struct task_struct {
    int pid;
    char comm[16];
};
"#;

        let result = parser.parse(source).unwrap();
        let struct_symbol = result
            .symbols
            .iter()
            .find(|s| s.name == "task_struct" && s.kind == SymbolKind::Struct);
        assert!(struct_symbol.is_some());
    }

    #[test]
    fn test_parse_includes() {
        let mut parser = TreeSitterParser::new().unwrap();
        let source = r#"
#include <linux/kernel.h>
#include "local_header.h"
"#;

        let result = parser.parse(source).unwrap();
        assert_eq!(result.includes.len(), 2);
        assert!(result.includes.contains(&"linux/kernel.h".to_string()));
        assert!(result.includes.contains(&"local_header.h".to_string()));
    }

    #[test]
    fn test_parse_macros() {
        let mut parser = TreeSitterParser::new().unwrap();
        let source = r#"
#define MAX_PATH 4096
#define FUNC(x, y) ((x) + (y))
"#;

        let result = parser.parse(source).unwrap();
        assert_eq!(result.macros.len(), 2);

        let max_path_macro = result.macros.iter().find(|m| m.name == "MAX_PATH");
        assert!(max_path_macro.is_some());
        assert_eq!(max_path_macro.unwrap().definition, "4096");

        let func_macro = result.macros.iter().find(|m| m.name == "FUNC");
        assert!(func_macro.is_some());
        assert_eq!(func_macro.unwrap().parameters.len(), 2);
    }
}
