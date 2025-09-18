//! GraphML format export functionality
//!
//! This module implements serialization of kernel call graphs to the GraphML format,
//! which is an XML-based format for graph data interchange.

use anyhow::Result;
use kcs_graph::{CallEdge, CallType, KernelGraph, Symbol, SymbolType};
use petgraph::visit::EdgeRef;
use quick_xml::events::{BytesDecl, BytesEnd, BytesStart, BytesText, Event};
use quick_xml::reader::Reader;
use quick_xml::Writer;
use std::collections::HashMap;
use std::io::Cursor;

use crate::{GraphExporter, GraphImporter};

/// GraphML format exporter
pub struct GraphMLExporter {
    /// Whether to include extended attributes
    include_attributes: bool,
    /// XML indentation string
    indent: Option<String>,
}

impl GraphMLExporter {
    /// Create a new GraphML exporter
    pub fn new() -> Self {
        Self {
            include_attributes: true,
            indent: Some("  ".to_string()),
        }
    }

    /// Set whether to include extended attributes
    pub fn with_attributes(mut self, include: bool) -> Self {
        self.include_attributes = include;
        self
    }

    /// Set XML indentation (None for no indentation)
    pub fn with_indent(mut self, indent: Option<String>) -> Self {
        self.indent = indent;
        self
    }

    /// Create XML declaration
    fn write_xml_declaration<W: std::io::Write>(&self, writer: &mut Writer<W>) -> Result<()> {
        writer.write_event(Event::Decl(BytesDecl::new("1.0", Some("UTF-8"), None)))?;
        Ok(())
    }

    /// Create GraphML root element
    fn write_graphml_start<W: std::io::Write>(&self, writer: &mut Writer<W>) -> Result<()> {
        let mut elem = BytesStart::new("graphml");
        elem.push_attribute(("xmlns", "http://graphml.graphdrawing.org/xmlns"));
        elem.push_attribute(("xmlns:xsi", "http://www.w3.org/2001/XMLSchema-instance"));
        elem.push_attribute((
            "xsi:schemaLocation",
            "http://graphml.graphdrawing.org/xmlns http://graphml.graphdrawing.org/xmlns/1.0/graphml.xsd",
        ));
        writer.write_event(Event::Start(elem))?;
        Ok(())
    }

    /// Write key definitions for node and edge attributes
    fn write_key_definitions<W: std::io::Write>(&self, writer: &mut Writer<W>) -> Result<()> {
        // Define keys for node attributes
        self.write_key(writer, "d0", "node", "string", "file_path", None)?;
        self.write_key(writer, "d1", "node", "int", "line_number", None)?;
        self.write_key(writer, "d2", "node", "string", "symbol_type", None)?;
        self.write_key(writer, "d3", "node", "string", "signature", None)?;
        self.write_key(writer, "d4", "node", "string", "config_dependencies", None)?;

        // Define keys for edge attributes
        self.write_key(writer, "d5", "edge", "string", "call_type", None)?;
        self.write_key(writer, "d6", "edge", "int", "call_site_line", None)?;
        self.write_key(writer, "d7", "edge", "boolean", "conditional", None)?;
        self.write_key(writer, "d8", "edge", "string", "config_guard", None)?;

        Ok(())
    }

    /// Write a single key definition
    fn write_key<W: std::io::Write>(
        &self,
        writer: &mut Writer<W>,
        id: &str,
        for_type: &str,
        attr_type: &str,
        attr_name: &str,
        default: Option<&str>,
    ) -> Result<()> {
        let mut elem = BytesStart::new("key");
        elem.push_attribute(("id", id));
        elem.push_attribute(("for", for_type));
        elem.push_attribute(("attr.name", attr_name));
        elem.push_attribute(("attr.type", attr_type));

        if let Some(default_value) = default {
            writer.write_event(Event::Start(elem))?;
            writer.write_event(Event::Start(BytesStart::new("default")))?;
            writer.write_event(Event::Text(BytesText::from_escaped(default_value)))?;
            writer.write_event(Event::End(BytesEnd::new("default")))?;
            writer.write_event(Event::End(BytesEnd::new("key")))?;
        } else {
            writer.write_event(Event::Empty(elem))?;
        }

        Ok(())
    }

    /// Write graph element and its contents
    fn write_graph<W: std::io::Write>(
        &self,
        writer: &mut Writer<W>,
        graph: &KernelGraph,
    ) -> Result<()> {
        let mut elem = BytesStart::new("graph");
        elem.push_attribute(("id", "G"));
        elem.push_attribute(("edgedefault", "directed"));
        writer.write_event(Event::Start(elem))?;

        // Create node ID mapping
        let mut node_id_map = HashMap::new();
        let call_graph = graph.graph();

        // Write nodes
        for node_idx in call_graph.node_indices() {
            if let Some(symbol) = call_graph.node_weight(node_idx) {
                let node_id = format!("n{}", node_idx.index());
                node_id_map.insert(node_idx, node_id.clone());
                self.write_node(writer, &node_id, symbol)?;
            }
        }

        // Write edges
        let mut edge_count = 0;
        for edge_ref in call_graph.edge_references() {
            let source_idx = edge_ref.source();
            let target_idx = edge_ref.target();
            let edge_data = edge_ref.weight();

            if let (Some(source_id), Some(target_id)) =
                (node_id_map.get(&source_idx), node_id_map.get(&target_idx))
            {
                let edge_id = format!("e{}", edge_count);
                self.write_edge(writer, &edge_id, source_id, target_id, edge_data)?;
                edge_count += 1;
            }
        }

        writer.write_event(Event::End(BytesEnd::new("graph")))?;
        Ok(())
    }

    /// Write a single node element
    fn write_node<W: std::io::Write>(
        &self,
        writer: &mut Writer<W>,
        node_id: &str,
        symbol: &kcs_graph::Symbol,
    ) -> Result<()> {
        let mut elem = BytesStart::new("node");
        elem.push_attribute(("id", node_id));
        writer.write_event(Event::Start(elem))?;

        // Write node data
        if self.include_attributes {
            self.write_data(writer, "d0", &symbol.file_path)?;
            self.write_data(writer, "d1", &symbol.line_number.to_string())?;
            self.write_data(writer, "d2", &format!("{:?}", symbol.symbol_type))?;

            if let Some(ref signature) = symbol.signature {
                self.write_data(writer, "d3", signature)?;
            }

            if !symbol.config_dependencies.is_empty() {
                let config_deps = symbol.config_dependencies.join(",");
                self.write_data(writer, "d4", &config_deps)?;
            }
        }

        // Write node label as a data element (commonly used by graph viewers)
        let mut data_elem = BytesStart::new("data");
        data_elem.push_attribute(("key", "label"));
        writer.write_event(Event::Start(data_elem))?;
        writer.write_event(Event::Text(BytesText::new(&symbol.name)))?;
        writer.write_event(Event::End(BytesEnd::new("data")))?;

        writer.write_event(Event::End(BytesEnd::new("node")))?;
        Ok(())
    }

    /// Write a single edge element
    fn write_edge<W: std::io::Write>(
        &self,
        writer: &mut Writer<W>,
        edge_id: &str,
        source_id: &str,
        target_id: &str,
        edge_data: &kcs_graph::CallEdge,
    ) -> Result<()> {
        let mut elem = BytesStart::new("edge");
        elem.push_attribute(("id", edge_id));
        elem.push_attribute(("source", source_id));
        elem.push_attribute(("target", target_id));
        writer.write_event(Event::Start(elem))?;

        // Write edge data
        if self.include_attributes {
            self.write_data(writer, "d5", &format!("{:?}", edge_data.call_type))?;
            self.write_data(writer, "d6", &edge_data.call_site_line.to_string())?;
            self.write_data(writer, "d7", &edge_data.conditional.to_string())?;

            if let Some(ref config_guard) = edge_data.config_guard {
                self.write_data(writer, "d8", config_guard)?;
            }
        }

        writer.write_event(Event::End(BytesEnd::new("edge")))?;
        Ok(())
    }

    /// Write a data element
    fn write_data<W: std::io::Write>(
        &self,
        writer: &mut Writer<W>,
        key: &str,
        value: &str,
    ) -> Result<()> {
        let mut elem = BytesStart::new("data");
        elem.push_attribute(("key", key));
        writer.write_event(Event::Start(elem))?;
        writer.write_event(Event::Text(BytesText::new(value)))?;
        writer.write_event(Event::End(BytesEnd::new("data")))?;
        Ok(())
    }
}

impl Default for GraphMLExporter {
    fn default() -> Self {
        Self::new()
    }
}

impl GraphExporter for GraphMLExporter {
    fn export_to_string(&self, graph: &kcs_graph::KernelGraph) -> Result<String> {
        let mut cursor = Cursor::new(Vec::new());
        let mut writer = if self.indent.is_some() {
            Writer::new_with_indent(&mut cursor, b' ', 2)
        } else {
            Writer::new(&mut cursor)
        };

        // Write XML declaration
        self.write_xml_declaration(&mut writer)?;

        // Write GraphML root element
        self.write_graphml_start(&mut writer)?;

        // Write key definitions
        self.write_key_definitions(&mut writer)?;

        // Write the graph
        self.write_graph(&mut writer, graph)?;

        // Close GraphML root element
        writer.write_event(Event::End(BytesEnd::new("graphml")))?;

        // Convert to string
        let result = String::from_utf8(cursor.into_inner())?;
        Ok(result)
    }

    fn export_to_file(&self, graph: &kcs_graph::KernelGraph, path: &str) -> Result<()> {
        let xml_string = self.export_to_string(graph)?;
        std::fs::write(path, xml_string)?;
        Ok(())
    }
}

impl GraphMLExporter {
    /// Parse GraphML from XML string
    fn parse_graphml(&self, xml: &str) -> Result<KernelGraph> {
        let mut reader = Reader::from_str(xml);
        reader.config_mut().trim_text(true);

        let mut graph = KernelGraph::new();
        let mut node_id_to_name = HashMap::new();
        let mut key_definitions = HashMap::new();
        let mut current_node: Option<(String, HashMap<String, String>)> = None;
        let mut current_edge: Option<(String, String, String, HashMap<String, String>)> = None;
        let mut current_data_key: Option<String> = None;

        let mut buf = Vec::new();

        loop {
            match reader.read_event_into(&mut buf) {
                Ok(Event::Start(ref e)) => {
                    match e.name().as_ref() {
                        b"key" => {
                            // Parse key definition
                            let mut id = None;
                            let mut attr_name = None;
                            for attr in e.attributes() {
                                let attr = attr?;
                                match attr.key.as_ref() {
                                    b"id" => {
                                        id = Some(String::from_utf8_lossy(&attr.value).to_string())
                                    }
                                    b"attr.name" => {
                                        attr_name =
                                            Some(String::from_utf8_lossy(&attr.value).to_string())
                                    }
                                    _ => {}
                                }
                            }
                            if let (Some(id), Some(name)) = (id, attr_name) {
                                key_definitions.insert(id, name);
                            }
                        }
                        b"node" => {
                            // Start parsing a node
                            let mut node_id = None;
                            for attr in e.attributes() {
                                let attr = attr?;
                                if attr.key.as_ref() == b"id" {
                                    node_id =
                                        Some(String::from_utf8_lossy(&attr.value).to_string());
                                }
                            }
                            if let Some(id) = node_id {
                                current_node = Some((id, HashMap::new()));
                            }
                        }
                        b"edge" => {
                            // Start parsing an edge
                            let mut edge_id = None;
                            let mut source = None;
                            let mut target = None;
                            for attr in e.attributes() {
                                let attr = attr?;
                                match attr.key.as_ref() {
                                    b"id" => {
                                        edge_id =
                                            Some(String::from_utf8_lossy(&attr.value).to_string())
                                    }
                                    b"source" => {
                                        source =
                                            Some(String::from_utf8_lossy(&attr.value).to_string())
                                    }
                                    b"target" => {
                                        target =
                                            Some(String::from_utf8_lossy(&attr.value).to_string())
                                    }
                                    _ => {}
                                }
                            }
                            if let (Some(id), Some(src), Some(tgt)) = (edge_id, source, target) {
                                current_edge = Some((id, src, tgt, HashMap::new()));
                            }
                        }
                        b"data" => {
                            // Start parsing data element
                            for attr in e.attributes() {
                                let attr = attr?;
                                if attr.key.as_ref() == b"key" {
                                    current_data_key =
                                        Some(String::from_utf8_lossy(&attr.value).to_string());
                                }
                            }
                        }
                        _ => {}
                    }
                }
                Ok(Event::Text(e)) => {
                    // Handle text content of data elements
                    if let Some(ref key) = current_data_key {
                        // Unescape XML entities (&lt; -> <, &gt; -> >, &amp; -> &, etc.)
                        let value = match e.unescape() {
                            Ok(v) => v.to_string(),
                            Err(_) => String::from_utf8_lossy(e.as_ref()).to_string(),
                        };

                        if let Some((_, ref mut attrs)) = current_node {
                            attrs.insert(key.clone(), value);
                        } else if let Some((_, _, _, ref mut attrs)) = current_edge {
                            attrs.insert(key.clone(), value);
                        }
                    }
                }
                Ok(Event::End(ref e)) => {
                    match e.name().as_ref() {
                        b"node" => {
                            // Finish parsing node and add to graph
                            if let Some((node_id, attrs)) = current_node.take() {
                                let symbol =
                                    self.create_symbol_from_attrs(&attrs, &key_definitions);
                                graph.add_symbol(symbol.clone());
                                node_id_to_name.insert(node_id, symbol.name);
                            }
                        }
                        b"edge" => {
                            // Finish parsing edge and add to graph
                            if let Some((_, source_id, target_id, attrs)) = current_edge.take() {
                                if let (Some(source_name), Some(target_name)) = (
                                    node_id_to_name.get(&source_id),
                                    node_id_to_name.get(&target_id),
                                ) {
                                    let edge =
                                        self.create_edge_from_attrs(&attrs, &key_definitions);
                                    graph.add_call(source_name, target_name, edge)?;
                                }
                            }
                        }
                        b"data" => {
                            current_data_key = None;
                        }
                        _ => {}
                    }
                }
                Ok(Event::Eof) => break,
                Err(e) => anyhow::bail!("Error parsing GraphML: {:?}", e),
                _ => {}
            }
            buf.clear();
        }

        Ok(graph)
    }

    /// Create a Symbol from parsed attributes
    fn create_symbol_from_attrs(
        &self,
        attrs: &HashMap<String, String>,
        key_defs: &HashMap<String, String>,
    ) -> Symbol {
        let mut symbol = Symbol {
            name: attrs
                .get("label")
                .cloned()
                .unwrap_or_else(|| "unknown".to_string()),
            file_path: String::new(),
            line_number: 0,
            symbol_type: SymbolType::Function,
            signature: None,
            config_dependencies: Vec::new(),
        };

        // Map attributes by their key definitions
        for (key_id, key_name) in key_defs {
            if let Some(value) = attrs.get(key_id) {
                match key_name.as_str() {
                    "file_path" => symbol.file_path = value.clone(),
                    "line_number" => symbol.line_number = value.parse().unwrap_or(0),
                    "symbol_type" => {
                        symbol.symbol_type = match value.as_str() {
                            "Variable" => SymbolType::Variable,
                            "Macro" => SymbolType::Macro,
                            "Type" => SymbolType::Type,
                            "Constant" => SymbolType::Constant,
                            _ => SymbolType::Function,
                        };
                    }
                    "signature" => symbol.signature = Some(value.clone()),
                    "config_dependencies" => {
                        symbol.config_dependencies = value
                            .split(',')
                            .map(|s| s.trim().to_string())
                            .filter(|s| !s.is_empty())
                            .collect();
                    }
                    _ => {}
                }
            }
        }

        // Also check direct key values (d0, d1, etc.)
        if let Some(value) = attrs.get("d0") {
            symbol.file_path = value.clone();
        }
        if let Some(value) = attrs.get("d1") {
            symbol.line_number = value.parse().unwrap_or(0);
        }
        if let Some(value) = attrs.get("d2") {
            symbol.symbol_type = match value.as_str() {
                "Variable" => SymbolType::Variable,
                "Macro" => SymbolType::Macro,
                "Type" => SymbolType::Type,
                "Constant" => SymbolType::Constant,
                _ => SymbolType::Function,
            };
        }
        if let Some(value) = attrs.get("d3") {
            symbol.signature = Some(value.clone());
        }
        if let Some(value) = attrs.get("d4") {
            symbol.config_dependencies = value
                .split(',')
                .map(|s| s.trim().to_string())
                .filter(|s| !s.is_empty())
                .collect();
        }

        symbol
    }

    /// Create a CallEdge from parsed attributes
    fn create_edge_from_attrs(
        &self,
        attrs: &HashMap<String, String>,
        key_defs: &HashMap<String, String>,
    ) -> CallEdge {
        let mut edge = CallEdge {
            call_type: CallType::Direct,
            call_site_line: 0,
            conditional: false,
            config_guard: None,
        };

        // Map attributes by their key definitions
        for (key_id, key_name) in key_defs {
            if let Some(value) = attrs.get(key_id) {
                match key_name.as_str() {
                    "call_type" => {
                        edge.call_type = match value.as_str() {
                            "Indirect" => CallType::Indirect,
                            "FunctionPointer" => CallType::FunctionPointer,
                            "Macro" => CallType::Macro,
                            _ => CallType::Direct,
                        };
                    }
                    "call_site_line" => edge.call_site_line = value.parse().unwrap_or(0),
                    "conditional" => edge.conditional = value.parse().unwrap_or(false),
                    "config_guard" => edge.config_guard = Some(value.clone()),
                    _ => {}
                }
            }
        }

        // Also check direct key values (d5, d6, etc.)
        if let Some(value) = attrs.get("d5") {
            edge.call_type = match value.as_str() {
                "Indirect" => CallType::Indirect,
                "FunctionPointer" => CallType::FunctionPointer,
                "Macro" => CallType::Macro,
                _ => CallType::Direct,
            };
        }
        if let Some(value) = attrs.get("d6") {
            edge.call_site_line = value.parse().unwrap_or(0);
        }
        if let Some(value) = attrs.get("d7") {
            edge.conditional = value.parse().unwrap_or(false);
        }
        if let Some(value) = attrs.get("d8") {
            edge.config_guard = Some(value.clone());
        }

        edge
    }
}

impl GraphImporter for GraphMLExporter {
    fn import_from_string(&self, data: &str) -> Result<kcs_graph::KernelGraph> {
        self.parse_graphml(data)
    }

    fn import_from_file(&self, path: &str) -> Result<kcs_graph::KernelGraph> {
        let contents = std::fs::read_to_string(path)?;
        self.import_from_string(&contents)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use kcs_graph::{CallEdge, CallType, KernelGraph, Symbol, SymbolType};

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
    fn test_graphml_exporter_creation() {
        let exporter = GraphMLExporter::new();
        assert!(exporter.include_attributes);
        assert_eq!(exporter.indent, Some("  ".to_string()));
    }

    #[test]
    fn test_graphml_exporter_configuration() {
        let exporter = GraphMLExporter::new()
            .with_attributes(false)
            .with_indent(None);
        assert!(!exporter.include_attributes);
        assert!(exporter.indent.is_none());
    }

    #[test]
    fn test_write_xml_declaration() {
        use std::io::Cursor;

        let exporter = GraphMLExporter::new();
        let mut cursor = Cursor::new(Vec::new());
        let mut writer = Writer::new(&mut cursor);

        exporter.write_xml_declaration(&mut writer).unwrap();

        let result = String::from_utf8(cursor.into_inner()).unwrap();
        assert!(result.contains("<?xml version=\"1.0\" encoding=\"UTF-8\"?>"));
    }

    #[test]
    fn test_export_to_graphml() {
        let graph = create_test_graph();
        let exporter = GraphMLExporter::new();

        let xml_string = exporter.export_to_string(&graph).unwrap();

        // Verify basic structure
        assert!(xml_string.contains("<?xml version=\"1.0\" encoding=\"UTF-8\"?>"));
        assert!(xml_string.contains("<graphml"));
        assert!(xml_string.contains("xmlns=\"http://graphml.graphdrawing.org/xmlns\""));
        assert!(xml_string.contains("</graphml>"));

        // Verify key definitions
        assert!(xml_string.contains("attr.name=\"file_path\""));
        assert!(xml_string.contains("attr.name=\"line_number\""));
        assert!(xml_string.contains("attr.name=\"symbol_type\""));

        // Verify nodes
        assert!(xml_string.contains("<node"));
        assert!(xml_string.contains("<data key=\"label\">test_func1</data>"));
        assert!(xml_string.contains("<data key=\"label\">test_func2</data>"));

        // Verify edges
        assert!(xml_string.contains("<edge"));
        assert!(xml_string.contains("source="));
        assert!(xml_string.contains("target="));
    }

    #[test]
    fn test_export_without_attributes() {
        let graph = create_test_graph();
        let exporter = GraphMLExporter::new().with_attributes(false);

        let xml_string = exporter.export_to_string(&graph).unwrap();

        // Should still have nodes with labels
        assert!(xml_string.contains("<data key=\"label\">test_func1</data>"));

        // Should not have detailed attributes
        assert!(!xml_string.contains("<data key=\"d0\">"));
        assert!(!xml_string.contains("<data key=\"d1\">"));
    }

    #[test]
    fn test_export_without_indent() {
        let graph = create_test_graph();
        let exporter = GraphMLExporter::new().with_indent(None);

        let xml_string = exporter.export_to_string(&graph).unwrap();

        // Should be on one line (no newlines except potentially in the XML declaration)
        let line_count = xml_string.lines().count();
        assert!(line_count <= 2); // At most XML declaration and content
    }

    #[test]
    fn test_round_trip_conversion() {
        let original_graph = create_test_graph();
        let exporter = GraphMLExporter::new();

        // Export to GraphML string
        let xml_string = exporter.export_to_string(&original_graph).unwrap();

        // Import back
        let imported_graph = exporter.import_from_string(&xml_string).unwrap();

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
        let exporter = GraphMLExporter::new();

        let xml_string = exporter.export_to_string(&graph).unwrap();

        // Verify structure is valid but empty
        assert!(xml_string.contains("<graph"));
        assert!(xml_string.contains("</graph>"));

        // Should not contain any nodes or edges
        assert!(!xml_string.contains("<node"));
        assert!(!xml_string.contains("<edge"));
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
            Symbol {
                name: "caller".to_string(),
                file_path: "test.c".to_string(),
                line_number: 1,
                symbol_type: SymbolType::Function,
                signature: None,
                config_dependencies: vec![],
            },
        ];

        for symbol in symbols {
            graph.add_symbol(symbol);
        }

        // Add different edge types
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

        let exporter = GraphMLExporter::new();
        let xml_string = exporter.export_to_string(&graph).unwrap();

        // Verify we can parse it back
        let imported = exporter.import_from_string(&xml_string).unwrap();
        assert_eq!(imported.symbol_count(), 4);
        assert_eq!(imported.call_count(), 2);

        // Verify symbol types are preserved
        let global_var = imported.get_symbol("global_var").unwrap();
        assert_eq!(global_var.symbol_type, SymbolType::Variable);

        let macro_func = imported.get_symbol("MACRO_FUNC").unwrap();
        assert_eq!(macro_func.symbol_type, SymbolType::Macro);
    }

    #[test]
    fn test_special_characters_in_names() {
        let mut graph = KernelGraph::new();

        // Add symbols with special XML characters
        graph.add_symbol(Symbol {
            name: "func<T>".to_string(),
            file_path: "test&file.c".to_string(),
            line_number: 10,
            symbol_type: SymbolType::Function,
            signature: Some("void func<T>(T& val)".to_string()),
            config_dependencies: vec!["CONFIG_A>B".to_string()],
        });

        let exporter = GraphMLExporter::new();
        let xml_string = exporter.export_to_string(&graph).unwrap();

        // Verify XML is well-formed (no parsing errors)
        let imported = exporter.import_from_string(&xml_string).unwrap();
        assert_eq!(imported.symbol_count(), 1);

        // Verify special characters are preserved
        let symbol = imported.get_symbol("func<T>").unwrap();
        assert_eq!(symbol.file_path, "test&file.c");
    }

    #[test]
    fn test_export_to_file() {
        use tempfile::tempdir;

        let graph = create_test_graph();
        let exporter = GraphMLExporter::new();

        // Create a temporary directory
        let dir = tempdir().unwrap();
        let file_path = dir.path().join("test.graphml");

        // Export to file
        exporter
            .export_to_file(&graph, file_path.to_str().unwrap())
            .unwrap();

        // Verify file exists and can be read back
        assert!(file_path.exists());
        let imported = exporter
            .import_from_file(file_path.to_str().unwrap())
            .unwrap();
        assert_eq!(imported.symbol_count(), graph.symbol_count());
    }
}
