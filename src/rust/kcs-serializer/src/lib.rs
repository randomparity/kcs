//! Kernel call graph serialization library
//!
//! This library provides serialization and deserialization capabilities for kernel
//! call graphs in various formats including JSON Graph format and GraphML.
//!
//! # Supported Formats
//!
//! - **JSON Graph Format**: A standardized format for representing graphs in JSON
//! - **GraphML**: An XML-based format for graph data interchange
//!
//! # Features
//!
//! - Chunking support for large graphs to manage memory usage
//! - Metadata preservation including config dependencies and call types
//! - Bidirectional conversion between KernelGraph and serialization formats
//! - Streaming serialization for memory-efficient processing

#![warn(missing_docs)]

use anyhow::Result;
use serde::{Deserialize, Serialize};

pub mod chunk_writer;
pub mod chunker;
pub mod graphml_export;
pub mod json_export;

#[cfg(test)]
mod chunk_writer_test;

#[cfg(test)]
mod manifest_test;

#[cfg(test)]
mod checksum_test;

// Re-export main types for convenience
pub use chunk_writer::{ChunkWriter, ChunkWriterConfig, ChunkWriterError, ChunkInfo, ChunkMetadata, FileInfo};
pub use chunker::{ChunkOptions, GraphChunk, GraphChunker};
pub use graphml_export::GraphMLExporter;
pub use json_export::JsonGraphExporter;

/// Common trait for graph exporters
pub trait GraphExporter {
    /// Export a kernel graph to a string representation
    fn export_to_string(&self, graph: &kcs_graph::KernelGraph) -> Result<String>;

    /// Export a kernel graph to a file
    fn export_to_file(&self, graph: &kcs_graph::KernelGraph, path: &str) -> Result<()>;
}

/// Common trait for graph importers
pub trait GraphImporter {
    /// Import a kernel graph from a string representation
    fn import_from_string(&self, data: &str) -> Result<kcs_graph::KernelGraph>;

    /// Import a kernel graph from a file
    fn import_from_file(&self, path: &str) -> Result<kcs_graph::KernelGraph>;
}

/// Metadata associated with a serialized graph
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GraphMetadata {
    /// Version of the serialization format
    pub format_version: String,
    /// Timestamp of when the graph was created
    pub created_at: String,
    /// Kernel version or identifier
    pub kernel_version: Option<String>,
    /// Configuration used for the kernel build
    pub kernel_config: Option<String>,
    /// Number of nodes in the graph
    pub node_count: usize,
    /// Number of edges in the graph
    pub edge_count: usize,
    /// Tool that created the graph
    pub created_by: String,
}

impl Default for GraphMetadata {
    fn default() -> Self {
        Self {
            format_version: "1.0.0".to_string(),
            created_at: chrono::Utc::now().to_rfc3339(),
            kernel_version: None,
            kernel_config: None,
            node_count: 0,
            edge_count: 0,
            created_by: "kcs-serializer".to_string(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_metadata_default() {
        let metadata = GraphMetadata::default();
        assert_eq!(metadata.format_version, "1.0.0");
        assert_eq!(metadata.created_by, "kcs-serializer");
        assert_eq!(metadata.node_count, 0);
        assert_eq!(metadata.edge_count, 0);
    }
}
