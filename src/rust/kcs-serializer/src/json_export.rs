//! JSON Graph format export functionality
//!
//! This module implements serialization of kernel call graphs to the JSON Graph format,
//! which is a standardized way to represent graph structures in JSON.

use anyhow::Result;
use kcs_graph::{CallType, KernelGraph, SymbolType};
use petgraph::visit::EdgeRef;
use serde::{Deserialize, Serialize};
use serde_json::json;
use std::collections::HashMap;
use std::fs;
use std::io::Write;

use crate::{GraphExporter, GraphImporter, GraphMetadata};

/// JSON Graph format exporter
pub struct JsonGraphExporter {
    /// Whether to pretty-print the JSON output
    pretty: bool,
    /// Whether to include metadata
    include_metadata: bool,
}

impl JsonGraphExporter {
    /// Create a new JSON graph exporter
    pub fn new() -> Self {
        Self {
            pretty: true,
            include_metadata: true,
        }
    }

    /// Set whether to pretty-print the JSON output
    pub fn with_pretty(mut self, pretty: bool) -> Self {
        self.pretty = pretty;
        self
    }

    /// Set whether to include metadata
    pub fn with_metadata(mut self, include: bool) -> Self {
        self.include_metadata = include;
        self
    }
}

impl Default for JsonGraphExporter {
    fn default() -> Self {
        Self::new()
    }
}

/// JSON Graph format representation
#[derive(Debug, Serialize, Deserialize)]
pub struct JsonGraph {
    /// Graph metadata
    #[serde(skip_serializing_if = "Option::is_none")]
    pub metadata: Option<GraphMetadata>,
    /// Graph nodes
    pub nodes: Vec<JsonNode>,
    /// Graph edges
    pub edges: Vec<JsonEdge>,
}

/// Node in JSON Graph format
#[derive(Debug, Serialize, Deserialize)]
pub struct JsonNode {
    /// Unique identifier for the node
    pub id: String,
    /// Node label (function name)
    pub label: String,
    /// Additional node attributes
    #[serde(skip_serializing_if = "Option::is_none")]
    pub attributes: Option<serde_json::Value>,
}

/// Edge in JSON Graph format
#[derive(Debug, Serialize, Deserialize)]
pub struct JsonEdge {
    /// Source node ID
    pub source: String,
    /// Target node ID
    pub target: String,
    /// Edge label (call type)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub label: Option<String>,
    /// Additional edge attributes
    #[serde(skip_serializing_if = "Option::is_none")]
    pub attributes: Option<serde_json::Value>,
}

impl GraphExporter for JsonGraphExporter {
    fn export_to_string(&self, graph: &kcs_graph::KernelGraph) -> Result<String> {
        let json_graph = self.convert_to_json_graph(graph)?;

        if self.pretty {
            Ok(serde_json::to_string_pretty(&json_graph)?)
        } else {
            Ok(serde_json::to_string(&json_graph)?)
        }
    }

    fn export_to_file(&self, graph: &kcs_graph::KernelGraph, path: &str) -> Result<()> {
        let json_string = self.export_to_string(graph)?;
        let mut file = fs::File::create(path)?;
        file.write_all(json_string.as_bytes())?;
        Ok(())
    }
}

impl JsonGraphExporter {
    /// Convert a KernelGraph to JSON Graph format
    fn convert_to_json_graph(&self, graph: &KernelGraph) -> Result<JsonGraph> {
        let mut nodes = Vec::new();
        let mut edges = Vec::new();
        let mut node_id_map = HashMap::new();

        let call_graph = graph.graph();

        // Convert nodes
        for node_idx in call_graph.node_indices() {
            if let Some(symbol) = call_graph.node_weight(node_idx) {
                let node_id = format!("node_{}", node_idx.index());
                node_id_map.insert(node_idx, node_id.clone());

                let attributes = json!({
                    "file_path": symbol.file_path,
                    "line_number": symbol.line_number,
                    "symbol_type": format!("{:?}", symbol.symbol_type),
                    "signature": symbol.signature,
                    "config_dependencies": symbol.config_dependencies,
                });

                nodes.push(JsonNode {
                    id: node_id,
                    label: symbol.name.clone(),
                    attributes: Some(attributes),
                });
            }
        }

        // Convert edges
        for edge_ref in call_graph.edge_references() {
            let source_idx = edge_ref.source();
            let target_idx = edge_ref.target();
            let edge_data = edge_ref.weight();

            if let (Some(source_id), Some(target_id)) =
                (node_id_map.get(&source_idx), node_id_map.get(&target_idx))
            {
                let attributes = json!({
                    "call_site_line": edge_data.call_site_line,
                    "conditional": edge_data.conditional,
                    "config_guard": edge_data.config_guard,
                });

                edges.push(JsonEdge {
                    source: source_id.clone(),
                    target: target_id.clone(),
                    label: Some(format!("{:?}", edge_data.call_type)),
                    attributes: Some(attributes),
                });
            }
        }

        // Create metadata if requested
        let metadata = if self.include_metadata {
            Some(GraphMetadata {
                node_count: nodes.len(),
                edge_count: edges.len(),
                ..GraphMetadata::default()
            })
        } else {
            None
        };

        Ok(JsonGraph {
            metadata,
            nodes,
            edges,
        })
    }

    /// Convert a JSON Graph back to KernelGraph
    fn convert_from_json_graph(&self, json_graph: &JsonGraph) -> Result<KernelGraph> {
        let mut graph = KernelGraph::new();
        let mut id_to_name = HashMap::new();

        // Add nodes
        for node in &json_graph.nodes {
            let mut symbol = kcs_graph::Symbol {
                name: node.label.clone(),
                file_path: String::new(),
                line_number: 0,
                symbol_type: SymbolType::Function,
                signature: None,
                config_dependencies: Vec::new(),
            };

            // Extract attributes if present
            if let Some(attrs) = &node.attributes {
                if let Some(file_path) = attrs.get("file_path").and_then(|v| v.as_str()) {
                    symbol.file_path = file_path.to_string();
                }
                if let Some(line_number) = attrs.get("line_number").and_then(|v| v.as_u64()) {
                    symbol.line_number = line_number as u32;
                }
                if let Some(symbol_type_str) = attrs.get("symbol_type").and_then(|v| v.as_str()) {
                    symbol.symbol_type = match symbol_type_str {
                        "Variable" => SymbolType::Variable,
                        "Macro" => SymbolType::Macro,
                        "Type" => SymbolType::Type,
                        "Constant" => SymbolType::Constant,
                        _ => SymbolType::Function,
                    };
                }
                if let Some(sig) = attrs.get("signature").and_then(|v| v.as_str()) {
                    symbol.signature = Some(sig.to_string());
                }
                if let Some(deps) = attrs.get("config_dependencies").and_then(|v| v.as_array()) {
                    symbol.config_dependencies = deps
                        .iter()
                        .filter_map(|d| d.as_str().map(|s| s.to_string()))
                        .collect();
                }
            }

            graph.add_symbol(symbol.clone());
            id_to_name.insert(node.id.clone(), symbol.name);
        }

        // Add edges
        for edge in &json_graph.edges {
            if let (Some(source_name), Some(target_name)) =
                (id_to_name.get(&edge.source), id_to_name.get(&edge.target))
            {
                let mut call_edge = kcs_graph::CallEdge {
                    call_type: CallType::Direct,
                    call_site_line: 0,
                    conditional: false,
                    config_guard: None,
                };

                // Parse call type from label
                if let Some(label) = &edge.label {
                    call_edge.call_type = match label.as_str() {
                        "Indirect" => CallType::Indirect,
                        "FunctionPointer" => CallType::FunctionPointer,
                        "Macro" => CallType::Macro,
                        _ => CallType::Direct,
                    };
                }

                // Extract attributes if present
                if let Some(attrs) = &edge.attributes {
                    if let Some(line) = attrs.get("call_site_line").and_then(|v| v.as_u64()) {
                        call_edge.call_site_line = line as u32;
                    }
                    if let Some(cond) = attrs.get("conditional").and_then(|v| v.as_bool()) {
                        call_edge.conditional = cond;
                    }
                    if let Some(guard) = attrs.get("config_guard").and_then(|v| v.as_str()) {
                        call_edge.config_guard = Some(guard.to_string());
                    }
                }

                graph.add_call(source_name, target_name, call_edge)?;
            }
        }

        Ok(graph)
    }
}

impl GraphImporter for JsonGraphExporter {
    fn import_from_string(&self, data: &str) -> Result<kcs_graph::KernelGraph> {
        let json_graph: JsonGraph = serde_json::from_str(data)?;
        self.convert_from_json_graph(&json_graph)
    }

    fn import_from_file(&self, path: &str) -> Result<kcs_graph::KernelGraph> {
        let contents = fs::read_to_string(path)?;
        self.import_from_string(&contents)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use kcs_graph::{CallEdge, CallType, Symbol, SymbolType};

    fn create_test_graph() -> KernelGraph {
        let mut graph = KernelGraph::new();

        // Add symbols
        let symbol1 = Symbol {
            name: "test_func1".to_string(),
            file_path: "test.c".to_string(),
            line_number: 10,
            symbol_type: SymbolType::Function,
            signature: Some("int test_func1(void)".to_string()),
            config_dependencies: vec!["CONFIG_TEST".to_string()],
        };

        let symbol2 = Symbol {
            name: "test_func2".to_string(),
            file_path: "test.c".to_string(),
            line_number: 20,
            symbol_type: SymbolType::Function,
            signature: Some("void test_func2(int)".to_string()),
            config_dependencies: vec![],
        };

        graph.add_symbol(symbol1);
        graph.add_symbol(symbol2);

        // Add edge
        let edge = CallEdge {
            call_type: CallType::Direct,
            call_site_line: 15,
            conditional: true,
            config_guard: Some("CONFIG_FEATURE_X".to_string()),
        };

        graph.add_call("test_func1", "test_func2", edge).unwrap();

        graph
    }

    #[test]
    fn test_json_exporter_creation() {
        let exporter = JsonGraphExporter::new();
        assert!(exporter.pretty);
        assert!(exporter.include_metadata);
    }

    #[test]
    fn test_json_exporter_configuration() {
        let exporter = JsonGraphExporter::new()
            .with_pretty(false)
            .with_metadata(false);
        assert!(!exporter.pretty);
        assert!(!exporter.include_metadata);
    }

    #[test]
    fn test_export_to_json() {
        let graph = create_test_graph();
        let exporter = JsonGraphExporter::new();

        let json_string = exporter.export_to_string(&graph).unwrap();

        // Parse back to verify structure
        let json: serde_json::Value = serde_json::from_str(&json_string).unwrap();

        assert!(json.get("metadata").is_some());
        assert!(json.get("nodes").is_some());
        assert!(json.get("edges").is_some());

        let nodes = json.get("nodes").unwrap().as_array().unwrap();
        assert_eq!(nodes.len(), 2);

        let edges = json.get("edges").unwrap().as_array().unwrap();
        assert_eq!(edges.len(), 1);
    }

    #[test]
    fn test_export_without_metadata() {
        let graph = create_test_graph();
        let exporter = JsonGraphExporter::new().with_metadata(false);

        let json_string = exporter.export_to_string(&graph).unwrap();
        let json: serde_json::Value = serde_json::from_str(&json_string).unwrap();

        assert!(json.get("metadata").is_none());
        assert!(json.get("nodes").is_some());
        assert!(json.get("edges").is_some());
    }

    #[test]
    fn test_node_attributes() {
        let graph = create_test_graph();
        let exporter = JsonGraphExporter::new();

        let json_string = exporter.export_to_string(&graph).unwrap();
        let json: serde_json::Value = serde_json::from_str(&json_string).unwrap();

        let nodes = json.get("nodes").unwrap().as_array().unwrap();

        // Find test_func1 node
        let func1_node = nodes
            .iter()
            .find(|n| n.get("label").unwrap().as_str() == Some("test_func1"))
            .unwrap();

        let attrs = func1_node.get("attributes").unwrap();
        assert_eq!(attrs.get("file_path").unwrap().as_str(), Some("test.c"));
        assert_eq!(attrs.get("line_number").unwrap().as_u64(), Some(10));
        assert_eq!(attrs.get("symbol_type").unwrap().as_str(), Some("Function"));

        let config_deps = attrs
            .get("config_dependencies")
            .unwrap()
            .as_array()
            .unwrap();
        assert_eq!(config_deps.len(), 1);
        assert_eq!(config_deps[0].as_str(), Some("CONFIG_TEST"));
    }

    #[test]
    fn test_edge_attributes() {
        let graph = create_test_graph();
        let exporter = JsonGraphExporter::new();

        let json_string = exporter.export_to_string(&graph).unwrap();
        let json: serde_json::Value = serde_json::from_str(&json_string).unwrap();

        let edges = json.get("edges").unwrap().as_array().unwrap();
        assert_eq!(edges.len(), 1);

        let edge = &edges[0];
        assert_eq!(edge.get("label").unwrap().as_str(), Some("Direct"));

        let attrs = edge.get("attributes").unwrap();
        assert_eq!(attrs.get("call_site_line").unwrap().as_u64(), Some(15));
        assert_eq!(attrs.get("conditional").unwrap().as_bool(), Some(true));
        assert_eq!(
            attrs.get("config_guard").unwrap().as_str(),
            Some("CONFIG_FEATURE_X")
        );
    }

    #[test]
    fn test_round_trip_conversion() {
        let original_graph = create_test_graph();
        let exporter = JsonGraphExporter::new();

        // Export to JSON string
        let json_string = exporter.export_to_string(&original_graph).unwrap();

        // Import back
        let imported_graph = exporter.import_from_string(&json_string).unwrap();

        // Verify counts match
        assert_eq!(imported_graph.symbol_count(), original_graph.symbol_count());
        assert_eq!(imported_graph.call_count(), original_graph.call_count());

        // Verify symbols exist
        assert!(imported_graph.get_symbol("test_func1").is_some());
        assert!(imported_graph.get_symbol("test_func2").is_some());

        // Verify the call edge exists
        let callers = imported_graph.find_callers("test_func2");
        assert_eq!(callers.len(), 1);
        assert_eq!(callers[0].name, "test_func1");
    }

    #[test]
    fn test_empty_graph() {
        let graph = KernelGraph::new();
        let exporter = JsonGraphExporter::new();

        let json_string = exporter.export_to_string(&graph).unwrap();
        let json: serde_json::Value = serde_json::from_str(&json_string).unwrap();

        let nodes = json.get("nodes").unwrap().as_array().unwrap();
        assert_eq!(nodes.len(), 0);

        let edges = json.get("edges").unwrap().as_array().unwrap();
        assert_eq!(edges.len(), 0);

        // Verify metadata shows zero counts
        let metadata = json.get("metadata").unwrap();
        assert_eq!(metadata.get("node_count").unwrap().as_u64(), Some(0));
        assert_eq!(metadata.get("edge_count").unwrap().as_u64(), Some(0));
    }

    #[test]
    fn test_complex_graph() {
        let mut graph = KernelGraph::new();

        // Add various symbol types
        let symbols = vec![
            Symbol {
                name: "global_var".to_string(),
                file_path: "globals.c".to_string(),
                line_number: 5,
                symbol_type: SymbolType::Variable,
                signature: None,
                config_dependencies: vec![],
            },
            Symbol {
                name: "MACRO_FUNC".to_string(),
                file_path: "macros.h".to_string(),
                line_number: 100,
                symbol_type: SymbolType::Macro,
                signature: None,
                config_dependencies: vec!["CONFIG_MACRO".to_string()],
            },
            Symbol {
                name: "struct_type".to_string(),
                file_path: "types.h".to_string(),
                line_number: 50,
                symbol_type: SymbolType::Type,
                signature: None,
                config_dependencies: vec![],
            },
        ];

        for symbol in symbols {
            graph.add_symbol(symbol);
        }

        // Add different edge types
        graph.add_symbol(Symbol {
            name: "caller".to_string(),
            file_path: "test.c".to_string(),
            line_number: 1,
            symbol_type: SymbolType::Function,
            signature: None,
            config_dependencies: vec![],
        });

        graph
            .add_call(
                "caller",
                "global_var",
                CallEdge {
                    call_type: CallType::Direct,
                    call_site_line: 10,
                    conditional: false,
                    config_guard: None,
                },
            )
            .unwrap();

        graph
            .add_call(
                "caller",
                "MACRO_FUNC",
                CallEdge {
                    call_type: CallType::Macro,
                    call_site_line: 20,
                    conditional: true,
                    config_guard: Some("CONFIG_FEATURE".to_string()),
                },
            )
            .unwrap();

        let exporter = JsonGraphExporter::new();
        let json_string = exporter.export_to_string(&graph).unwrap();

        // Verify we can parse it back
        let _json: serde_json::Value = serde_json::from_str(&json_string).unwrap();

        // Import and verify
        let imported = exporter.import_from_string(&json_string).unwrap();
        assert_eq!(imported.symbol_count(), 4);
        assert_eq!(imported.call_count(), 2);
    }
}
