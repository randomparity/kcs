//! JSON Graph format export functionality
//!
//! This module implements serialization of kernel call graphs to the JSON Graph format,
//! which is a standardized way to represent graph structures in JSON.

use anyhow::Result;
use serde::{Deserialize, Serialize};
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
    fn export_to_string(&self, _graph: &kcs_graph::KernelGraph) -> Result<String> {
        // TODO: Implement actual export logic in T034
        anyhow::bail!("JSON export not yet implemented - will be completed in T034")
    }

    fn export_to_file(&self, graph: &kcs_graph::KernelGraph, path: &str) -> Result<()> {
        let json_string = self.export_to_string(graph)?;
        let mut file = fs::File::create(path)?;
        file.write_all(json_string.as_bytes())?;
        Ok(())
    }
}

impl GraphImporter for JsonGraphExporter {
    fn import_from_string(&self, _data: &str) -> Result<kcs_graph::KernelGraph> {
        // TODO: Implement actual import logic in T034
        anyhow::bail!("JSON import not yet implemented - will be completed in T034")
    }

    fn import_from_file(&self, path: &str) -> Result<kcs_graph::KernelGraph> {
        let contents = fs::read_to_string(path)?;
        self.import_from_string(&contents)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

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
}
