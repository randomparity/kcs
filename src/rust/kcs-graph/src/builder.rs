use crate::{CallEdge, CallType, KernelGraph, Symbol, SymbolType};
use anyhow::Result;
use serde_json::Value;
use std::collections::HashMap;
use std::path::Path;

pub struct GraphBuilder {
    graph: KernelGraph,
    config_context: HashMap<String, bool>,
}

impl GraphBuilder {
    pub fn new() -> Self {
        Self {
            graph: KernelGraph::new(),
            config_context: HashMap::new(),
        }
    }

    pub fn with_config_context(mut self, config: HashMap<String, bool>) -> Self {
        self.config_context = config;
        self
    }

    pub fn load_from_parser_output(&mut self, parser_output: &str) -> Result<()> {
        let data: Value = serde_json::from_str(parser_output)?;

        // Extract symbols from parser output
        if let Some(symbols) = data.get("symbols").and_then(|s| s.as_array()) {
            for symbol_data in symbols {
                if let Ok(symbol) = self.parse_symbol(symbol_data) {
                    self.graph.add_symbol(symbol);
                }
            }
        }

        // Extract call relationships
        if let Some(calls) = data.get("calls").and_then(|c| c.as_array()) {
            for call_data in calls {
                if let Ok((caller, callee, edge)) = self.parse_call(call_data) {
                    if let Err(e) = self.graph.add_call(&caller, &callee, edge) {
                        tracing::warn!("Failed to add call edge {}->{}: {}", caller, callee, e);
                    }
                }
            }
        }

        Ok(())
    }

    pub fn load_from_compilation_database<P: AsRef<Path>>(
        &mut self,
        compile_commands_path: P,
    ) -> Result<()> {
        let content = std::fs::read_to_string(compile_commands_path)?;
        let compile_db: Vec<Value> = serde_json::from_str(&content)?;

        for entry in compile_db {
            if let Some(file) = entry.get("file").and_then(|f| f.as_str()) {
                if let Some(command) = entry.get("command").and_then(|c| c.as_str()) {
                    self.extract_from_compile_unit(file, command)?;
                }
            }
        }

        Ok(())
    }

    pub fn finalize(self) -> KernelGraph {
        self.graph
    }

    fn parse_symbol(&self, symbol_data: &Value) -> Result<Symbol> {
        let name = symbol_data
            .get("name")
            .and_then(|n| n.as_str())
            .ok_or_else(|| anyhow::anyhow!("Missing symbol name"))?
            .to_string();

        let file_path = symbol_data
            .get("file")
            .and_then(|f| f.as_str())
            .unwrap_or("unknown")
            .to_string();

        let line_number = symbol_data
            .get("line")
            .and_then(|l| l.as_u64())
            .unwrap_or(0) as u32;

        let symbol_type = match symbol_data.get("type").and_then(|t| t.as_str()) {
            Some("function") => SymbolType::Function,
            Some("variable") => SymbolType::Variable,
            Some("macro") => SymbolType::Macro,
            Some("type") => SymbolType::Type,
            Some("constant") => SymbolType::Constant,
            _ => SymbolType::Function, // Default fallback
        };

        let signature = symbol_data
            .get("signature")
            .and_then(|s| s.as_str())
            .map(|s| s.to_string());

        let config_dependencies = symbol_data
            .get("config_deps")
            .and_then(|deps| deps.as_array())
            .map(|arr| {
                arr.iter()
                    .filter_map(|v| v.as_str())
                    .map(|s| s.to_string())
                    .collect()
            })
            .unwrap_or_default();

        Ok(Symbol {
            name,
            file_path,
            line_number,
            symbol_type,
            signature,
            config_dependencies,
        })
    }

    fn parse_call(&self, call_data: &Value) -> Result<(String, String, CallEdge)> {
        let caller = call_data
            .get("caller")
            .and_then(|c| c.as_str())
            .ok_or_else(|| anyhow::anyhow!("Missing caller"))?
            .to_string();

        let callee = call_data
            .get("callee")
            .and_then(|c| c.as_str())
            .ok_or_else(|| anyhow::anyhow!("Missing callee"))?
            .to_string();

        let call_type = match call_data.get("call_type").and_then(|t| t.as_str()) {
            Some("direct") => CallType::Direct,
            Some("indirect") => CallType::Indirect,
            Some("function_pointer") => CallType::FunctionPointer,
            Some("macro") => CallType::Macro,
            _ => CallType::Direct,
        };

        let call_site_line = call_data.get("line").and_then(|l| l.as_u64()).unwrap_or(0) as u32;

        let conditional = call_data
            .get("conditional")
            .and_then(|c| c.as_bool())
            .unwrap_or(false);

        let config_guard = call_data
            .get("config_guard")
            .and_then(|g| g.as_str())
            .map(|s| s.to_string());

        let edge = CallEdge {
            call_type,
            call_site_line,
            conditional,
            config_guard,
        };

        Ok((caller, callee, edge))
    }

    fn extract_from_compile_unit(&mut self, _file_path: &str, _command: &str) -> Result<()> {
        // This would integrate with the parser to extract symbols and calls
        // from a specific compilation unit
        // For now, this is a placeholder
        Ok(())
    }
}

impl Default for GraphBuilder {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_builder_creation() {
        let builder = GraphBuilder::new();
        let graph = builder.finalize();
        assert_eq!(graph.symbol_count(), 0);
    }

    #[test]
    fn test_parse_symbol_from_json() -> Result<()> {
        let builder = GraphBuilder::new();
        let symbol_json = serde_json::json!({
            "name": "vfs_read",
            "file": "fs/read_write.c",
            "line": 450,
            "type": "function",
            "signature": "ssize_t vfs_read(struct file *file, char __user *buf, size_t count, loff_t *pos)",
            "config_deps": ["CONFIG_VFS"]
        });

        let symbol = builder.parse_symbol(&symbol_json)?;
        assert_eq!(symbol.name, "vfs_read");
        assert_eq!(symbol.file_path, "fs/read_write.c");
        assert_eq!(symbol.line_number, 450);
        assert!(matches!(symbol.symbol_type, SymbolType::Function));
        assert_eq!(symbol.config_dependencies, vec!["CONFIG_VFS"]);

        Ok(())
    }
}
