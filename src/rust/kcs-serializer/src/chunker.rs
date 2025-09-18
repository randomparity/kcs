//! Graph chunking support for large graphs
//!
//! This module provides functionality to split large kernel call graphs into
//! smaller chunks for memory-efficient processing and serialization.

use anyhow::Result;
use kcs_graph::KernelGraph;
use serde::{Deserialize, Serialize};

/// Options for controlling graph chunking behavior
#[derive(Debug, Clone)]
pub struct ChunkOptions {
    /// Maximum number of nodes per chunk
    pub max_nodes_per_chunk: usize,
    /// Maximum number of edges per chunk
    pub max_edges_per_chunk: usize,
    /// Whether to preserve connected components intact
    pub preserve_components: bool,
    /// Whether to include boundary edges (edges crossing chunk boundaries)
    pub include_boundary_edges: bool,
}

impl Default for ChunkOptions {
    fn default() -> Self {
        Self {
            max_nodes_per_chunk: 1000,
            max_edges_per_chunk: 5000,
            preserve_components: true,
            include_boundary_edges: true,
        }
    }
}

/// A chunk of a graph
#[derive(Debug, Serialize, Deserialize)]
pub struct GraphChunk {
    /// Chunk identifier
    pub id: usize,
    /// Total number of chunks
    pub total_chunks: usize,
    /// Node IDs included in this chunk
    pub node_ids: Vec<String>,
    /// Edge information for this chunk
    pub edges: Vec<ChunkEdge>,
    /// Boundary edges (edges to nodes in other chunks)
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub boundary_edges: Vec<BoundaryEdge>,
}

/// Edge within a chunk
#[derive(Debug, Serialize, Deserialize)]
pub struct ChunkEdge {
    /// Source node ID
    pub source: String,
    /// Target node ID
    pub target: String,
    /// Call type
    pub call_type: String,
}

/// Edge crossing chunk boundaries
#[derive(Debug, Serialize, Deserialize)]
pub struct BoundaryEdge {
    /// Source node ID
    pub source: String,
    /// Target node ID
    pub target: String,
    /// Target chunk ID
    pub target_chunk: usize,
    /// Call type
    pub call_type: String,
}

/// Graph chunker for splitting large graphs
pub struct GraphChunker {
    options: ChunkOptions,
}

impl GraphChunker {
    /// Create a new graph chunker with default options
    pub fn new() -> Self {
        Self {
            options: ChunkOptions::default(),
        }
    }

    /// Create a new graph chunker with custom options
    pub fn with_options(options: ChunkOptions) -> Self {
        Self { options }
    }

    /// Split a graph into chunks
    pub fn chunk_graph(&self, _graph: &KernelGraph) -> Result<Vec<GraphChunk>> {
        // TODO: Implement actual chunking logic in T036
        anyhow::bail!("Graph chunking not yet implemented - will be completed in T036")
    }

    /// Reassemble chunks into a complete graph
    pub fn reassemble_chunks(&self, _chunks: &[GraphChunk]) -> Result<KernelGraph> {
        // TODO: Implement actual reassembly logic in T036
        anyhow::bail!("Graph reassembly not yet implemented - will be completed in T036")
    }

    /// Estimate the number of chunks needed for a graph
    pub fn estimate_chunks(&self, graph: &KernelGraph) -> usize {
        let node_count = graph.symbol_count();
        let edge_count = graph.call_count();

        let chunks_by_nodes = node_count.div_ceil(self.options.max_nodes_per_chunk);
        let chunks_by_edges = edge_count.div_ceil(self.options.max_edges_per_chunk);

        chunks_by_nodes.max(chunks_by_edges).max(1)
    }
}

impl Default for GraphChunker {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_chunk_options_default() {
        let options = ChunkOptions::default();
        assert_eq!(options.max_nodes_per_chunk, 1000);
        assert_eq!(options.max_edges_per_chunk, 5000);
        assert!(options.preserve_components);
        assert!(options.include_boundary_edges);
    }

    #[test]
    fn test_graph_chunker_creation() {
        let chunker = GraphChunker::new();
        assert_eq!(chunker.options.max_nodes_per_chunk, 1000);

        let custom_options = ChunkOptions {
            max_nodes_per_chunk: 500,
            max_edges_per_chunk: 2500,
            preserve_components: false,
            include_boundary_edges: false,
        };
        let custom_chunker = GraphChunker::with_options(custom_options.clone());
        assert_eq!(custom_chunker.options.max_nodes_per_chunk, 500);
    }

    #[test]
    fn test_estimate_chunks() {
        let chunker = GraphChunker::new();
        let graph = KernelGraph::new();

        // Empty graph should still have 1 chunk
        assert_eq!(chunker.estimate_chunks(&graph), 1);

        // TODO: Add more comprehensive tests when KernelGraph supports querying counts
    }
}
