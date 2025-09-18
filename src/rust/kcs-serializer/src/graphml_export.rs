//! GraphML format export functionality
//!
//! This module implements serialization of kernel call graphs to the GraphML format,
//! which is an XML-based format for graph data interchange.

use anyhow::Result;
use quick_xml::events::{BytesDecl, BytesEnd, BytesStart, BytesText, Event};
use quick_xml::Writer;

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
    #[allow(dead_code)] // Will be used in T035
    fn write_xml_declaration<W: std::io::Write>(&self, writer: &mut Writer<W>) -> Result<()> {
        writer.write_event(Event::Decl(BytesDecl::new("1.0", Some("UTF-8"), None)))?;
        Ok(())
    }

    /// Create GraphML root element
    #[allow(dead_code)] // Will be used in T035
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
    #[allow(dead_code)] // Will be used in T035
    fn write_key_definitions<W: std::io::Write>(&self, writer: &mut Writer<W>) -> Result<()> {
        // Define keys for node attributes
        self.write_key(writer, "d0", "node", "string", "file_path", None)?;
        self.write_key(writer, "d1", "node", "int", "line_number", None)?;
        self.write_key(writer, "d2", "node", "string", "symbol_type", None)?;
        self.write_key(writer, "d3", "node", "string", "signature", None)?;

        // Define keys for edge attributes
        self.write_key(writer, "d4", "edge", "string", "call_type", None)?;
        self.write_key(writer, "d5", "edge", "int", "call_site_line", None)?;
        self.write_key(writer, "d6", "edge", "boolean", "conditional", None)?;
        self.write_key(writer, "d7", "edge", "string", "config_guard", None)?;

        Ok(())
    }

    /// Write a single key definition
    #[allow(dead_code)] // Will be used in T035
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
}

impl Default for GraphMLExporter {
    fn default() -> Self {
        Self::new()
    }
}

impl GraphExporter for GraphMLExporter {
    fn export_to_string(&self, _graph: &kcs_graph::KernelGraph) -> Result<String> {
        // TODO: Implement actual export logic in T035
        anyhow::bail!("GraphML export not yet implemented - will be completed in T035")
    }

    fn export_to_file(&self, graph: &kcs_graph::KernelGraph, path: &str) -> Result<()> {
        let xml_string = self.export_to_string(graph)?;
        std::fs::write(path, xml_string)?;
        Ok(())
    }
}

impl GraphImporter for GraphMLExporter {
    fn import_from_string(&self, _data: &str) -> Result<kcs_graph::KernelGraph> {
        // TODO: Implement actual import logic in T035
        anyhow::bail!("GraphML import not yet implemented - will be completed in T035")
    }

    fn import_from_file(&self, path: &str) -> Result<kcs_graph::KernelGraph> {
        let contents = std::fs::read_to_string(path)?;
        self.import_from_string(&contents)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

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
}
