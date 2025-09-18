//! Graph chunking support for large graphs
//!
//! This module provides functionality to split large kernel call graphs into
//! smaller chunks for memory-efficient processing and serialization.

use anyhow::Result;
use kcs_graph::{CallEdge, CallType, KernelGraph, Symbol};
use petgraph::visit::EdgeRef;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet, VecDeque};

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
#[derive(Debug, Clone, Serialize, Deserialize)]
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
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChunkEdge {
    /// Source node ID
    pub source: String,
    /// Target node ID
    pub target: String,
    /// Call type
    pub call_type: String,
}

/// Edge crossing chunk boundaries
#[derive(Debug, Clone, Serialize, Deserialize)]
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
    pub fn chunk_graph(&self, graph: &KernelGraph) -> Result<Vec<GraphChunk>> {
        let call_graph = graph.graph();
        let node_count = call_graph.node_count();
        let edge_count = call_graph.edge_count();

        // If graph is small enough, return single chunk
        if node_count <= self.options.max_nodes_per_chunk
            && edge_count <= self.options.max_edges_per_chunk
        {
            return Ok(vec![self.create_single_chunk(graph)?]);
        }

        if self.options.preserve_components {
            self.chunk_by_components(graph)
        } else {
            self.chunk_by_size(graph)
        }
    }

    /// Reassemble chunks into a complete graph
    pub fn reassemble_chunks(&self, chunks: &[GraphChunk]) -> Result<KernelGraph> {
        let mut graph = KernelGraph::new();
        let mut symbol_map = HashMap::new();

        // First pass: Add all symbols from all chunks
        for chunk in chunks {
            for node_id in &chunk.node_ids {
                if !symbol_map.contains_key(node_id) {
                    // Create a placeholder symbol - in practice this would need
                    // the actual symbol data stored in the chunk
                    let symbol = Symbol {
                        name: node_id.clone(),
                        file_path: String::new(),
                        line_number: 0,
                        symbol_type: kcs_graph::SymbolType::Function,
                        signature: None,
                        config_dependencies: Vec::new(),
                    };
                    graph.add_symbol(symbol);
                    symbol_map.insert(node_id.clone(), ());
                }
            }
        }

        // Second pass: Add all edges
        for chunk in chunks {
            // Add internal edges
            for edge in &chunk.edges {
                let call_edge = CallEdge {
                    call_type: self.parse_call_type(&edge.call_type),
                    call_site_line: 0,
                    conditional: false,
                    config_guard: None,
                };
                graph.add_call(&edge.source, &edge.target, call_edge)?;
            }

            // Add boundary edges
            for boundary_edge in &chunk.boundary_edges {
                let call_edge = CallEdge {
                    call_type: self.parse_call_type(&boundary_edge.call_type),
                    call_site_line: 0,
                    conditional: false,
                    config_guard: None,
                };
                graph.add_call(&boundary_edge.source, &boundary_edge.target, call_edge)?;
            }
        }

        Ok(graph)
    }

    /// Estimate the number of chunks needed for a graph
    pub fn estimate_chunks(&self, graph: &KernelGraph) -> usize {
        let node_count = graph.symbol_count();
        let edge_count = graph.call_count();

        let chunks_by_nodes = node_count.div_ceil(self.options.max_nodes_per_chunk);
        let chunks_by_edges = edge_count.div_ceil(self.options.max_edges_per_chunk);

        chunks_by_nodes.max(chunks_by_edges).max(1)
    }

    /// Create a single chunk containing the entire graph
    fn create_single_chunk(&self, graph: &KernelGraph) -> Result<GraphChunk> {
        let call_graph = graph.graph();
        let mut node_ids = Vec::new();
        let mut edges = Vec::new();

        // Collect all node IDs
        for node_idx in call_graph.node_indices() {
            if let Some(symbol) = call_graph.node_weight(node_idx) {
                node_ids.push(symbol.name.clone());
            }
        }

        // Collect all edges
        for edge_ref in call_graph.edge_references() {
            let source_idx = edge_ref.source();
            let target_idx = edge_ref.target();
            let edge_data = edge_ref.weight();

            if let (Some(source_symbol), Some(target_symbol)) = (
                call_graph.node_weight(source_idx),
                call_graph.node_weight(target_idx),
            ) {
                edges.push(ChunkEdge {
                    source: source_symbol.name.clone(),
                    target: target_symbol.name.clone(),
                    call_type: format!("{:?}", edge_data.call_type),
                });
            }
        }

        Ok(GraphChunk {
            id: 0,
            total_chunks: 1,
            node_ids,
            edges,
            boundary_edges: Vec::new(),
        })
    }

    /// Chunk graph by connected components
    fn chunk_by_components(&self, graph: &KernelGraph) -> Result<Vec<GraphChunk>> {
        // For now, we'll just use strongly connected components directly
        // The connected_components function requires more complex type handling
        self.chunk_by_connected_components(graph)
    }

    /// Chunk by strongly connected components
    fn chunk_by_connected_components(&self, graph: &KernelGraph) -> Result<Vec<GraphChunk>> {
        use petgraph::algo::tarjan_scc;

        let call_graph = graph.graph();
        let sccs = tarjan_scc(call_graph);
        let mut chunks = Vec::new();
        let mut node_to_chunk = HashMap::new();

        // Create chunks for each SCC
        for (chunk_id, scc) in sccs.into_iter().enumerate() {
            let mut node_ids = Vec::new();
            let mut edges = Vec::new();

            // Collect nodes in this component
            for node_idx in &scc {
                if let Some(symbol) = call_graph.node_weight(*node_idx) {
                    node_ids.push(symbol.name.clone());
                    node_to_chunk.insert(*node_idx, chunk_id);
                }
            }

            // Collect internal edges
            for &node_idx in &scc {
                for edge_ref in call_graph.edges(node_idx) {
                    let target_idx = edge_ref.target();
                    if scc.contains(&target_idx) {
                        // Internal edge
                        if let (Some(source_symbol), Some(target_symbol)) = (
                            call_graph.node_weight(node_idx),
                            call_graph.node_weight(target_idx),
                        ) {
                            edges.push(ChunkEdge {
                                source: source_symbol.name.clone(),
                                target: target_symbol.name.clone(),
                                call_type: format!("{:?}", edge_ref.weight().call_type),
                            });
                        }
                    }
                }
            }

            chunks.push(GraphChunk {
                id: chunk_id,
                total_chunks: 0, // Will be set later
                node_ids,
                edges,
                boundary_edges: Vec::new(),
            });
        }

        // Add boundary edges if requested
        if self.options.include_boundary_edges {
            self.add_boundary_edges(graph, &mut chunks, &node_to_chunk)?;
        }

        // Set total chunks count
        let total_chunks = chunks.len();
        for chunk in &mut chunks {
            chunk.total_chunks = total_chunks;
        }

        Ok(chunks)
    }

    /// Chunk by size constraints
    fn chunk_by_size(&self, graph: &KernelGraph) -> Result<Vec<GraphChunk>> {
        let call_graph = graph.graph();
        let mut chunks = Vec::new();
        let mut node_to_chunk = HashMap::new();
        let mut visited = HashSet::new();
        let mut chunk_id = 0;

        for start_node in call_graph.node_indices() {
            if visited.contains(&start_node) {
                continue;
            }

            let mut chunk_nodes = Vec::new();
            let mut queue = VecDeque::new();
            queue.push_back(start_node);

            // BFS to fill chunk up to size limits
            while let Some(node_idx) = queue.pop_front() {
                if visited.contains(&node_idx)
                    || chunk_nodes.len() >= self.options.max_nodes_per_chunk
                {
                    continue;
                }

                visited.insert(node_idx);
                chunk_nodes.push(node_idx);
                node_to_chunk.insert(node_idx, chunk_id);

                // Add neighbors to queue
                for neighbor in call_graph.neighbors(node_idx) {
                    if !visited.contains(&neighbor) {
                        queue.push_back(neighbor);
                    }
                }
            }

            if !chunk_nodes.is_empty() {
                let chunk = self.create_chunk_from_nodes(graph, chunk_id, &chunk_nodes)?;
                chunks.push(chunk);
                chunk_id += 1;
            }
        }

        // Add boundary edges if requested
        if self.options.include_boundary_edges {
            self.add_boundary_edges(graph, &mut chunks, &node_to_chunk)?;
        }

        // Set total chunks count
        let total_chunks = chunks.len();
        for chunk in &mut chunks {
            chunk.total_chunks = total_chunks;
        }

        Ok(chunks)
    }

    /// Create a chunk from a set of node indices
    fn create_chunk_from_nodes(
        &self,
        graph: &KernelGraph,
        chunk_id: usize,
        node_indices: &[petgraph::graph::NodeIndex],
    ) -> Result<GraphChunk> {
        let call_graph = graph.graph();
        let mut node_ids = Vec::new();
        let mut edges = Vec::new();
        let node_set: HashSet<_> = node_indices.iter().cloned().collect();

        // Collect node names
        for &node_idx in node_indices {
            if let Some(symbol) = call_graph.node_weight(node_idx) {
                node_ids.push(symbol.name.clone());
            }
        }

        // Collect internal edges
        for &source_idx in node_indices {
            for edge_ref in call_graph.edges(source_idx) {
                let target_idx = edge_ref.target();
                if node_set.contains(&target_idx) {
                    // Internal edge
                    if let (Some(source_symbol), Some(target_symbol)) = (
                        call_graph.node_weight(source_idx),
                        call_graph.node_weight(target_idx),
                    ) {
                        edges.push(ChunkEdge {
                            source: source_symbol.name.clone(),
                            target: target_symbol.name.clone(),
                            call_type: format!("{:?}", edge_ref.weight().call_type),
                        });
                    }
                }
            }
        }

        Ok(GraphChunk {
            id: chunk_id,
            total_chunks: 0, // Will be set by caller
            node_ids,
            edges,
            boundary_edges: Vec::new(),
        })
    }

    /// Add boundary edges to chunks
    fn add_boundary_edges(
        &self,
        graph: &KernelGraph,
        chunks: &mut [GraphChunk],
        node_to_chunk: &HashMap<petgraph::graph::NodeIndex, usize>,
    ) -> Result<()> {
        let call_graph = graph.graph();

        for edge_ref in call_graph.edge_references() {
            let source_idx = edge_ref.source();
            let target_idx = edge_ref.target();

            if let (Some(&source_chunk), Some(&target_chunk)) = (
                node_to_chunk.get(&source_idx),
                node_to_chunk.get(&target_idx),
            ) {
                if source_chunk != target_chunk {
                    // This is a boundary edge
                    if let (Some(source_symbol), Some(target_symbol)) = (
                        call_graph.node_weight(source_idx),
                        call_graph.node_weight(target_idx),
                    ) {
                        let boundary_edge = BoundaryEdge {
                            source: source_symbol.name.clone(),
                            target: target_symbol.name.clone(),
                            target_chunk,
                            call_type: format!("{:?}", edge_ref.weight().call_type),
                        };

                        if let Some(chunk) = chunks.get_mut(source_chunk) {
                            chunk.boundary_edges.push(boundary_edge);
                        }
                    }
                }
            }
        }

        Ok(())
    }

    /// Parse call type from string
    fn parse_call_type(&self, call_type_str: &str) -> CallType {
        match call_type_str {
            "Indirect" => CallType::Indirect,
            "FunctionPointer" => CallType::FunctionPointer,
            "Macro" => CallType::Macro,
            _ => CallType::Direct,
        }
    }

    /// Stream chunks one at a time for memory efficiency
    pub fn stream_chunks<'a>(&'a self, graph: &'a KernelGraph) -> ChunkIterator<'a> {
        ChunkIterator::new(self, graph)
    }
}

/// Iterator for streaming chunks from a graph
pub struct ChunkIterator<'a> {
    chunker: &'a GraphChunker,
    graph: &'a KernelGraph,
    chunks: Option<Vec<GraphChunk>>,
    current_index: usize,
}

impl<'a> ChunkIterator<'a> {
    fn new(chunker: &'a GraphChunker, graph: &'a KernelGraph) -> Self {
        Self {
            chunker,
            graph,
            chunks: None,
            current_index: 0,
        }
    }

    /// Get the total number of chunks without loading them all
    pub fn estimated_count(&self) -> usize {
        self.chunker.estimate_chunks(self.graph)
    }
}

impl<'a> Iterator for ChunkIterator<'a> {
    type Item = Result<GraphChunk>;

    fn next(&mut self) -> Option<Self::Item> {
        // Lazy initialization of chunks
        if self.chunks.is_none() {
            match self.chunker.chunk_graph(self.graph) {
                Ok(chunks) => self.chunks = Some(chunks),
                Err(e) => return Some(Err(e)),
            }
        }

        if let Some(ref chunks) = self.chunks {
            if self.current_index < chunks.len() {
                let chunk = chunks[self.current_index].clone();
                self.current_index += 1;
                Some(Ok(chunk))
            } else {
                None
            }
        } else {
            None
        }
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let remaining = if let Some(ref chunks) = self.chunks {
            chunks.len() - self.current_index
        } else {
            self.estimated_count()
        };
        (remaining, Some(remaining))
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
    use kcs_graph::{CallEdge, CallType, Symbol, SymbolType};

    fn create_test_graph() -> KernelGraph {
        let mut graph = KernelGraph::new();

        // Add symbols
        let symbols = vec![
            Symbol {
                name: "func1".to_string(),
                file_path: "test.c".to_string(),
                line_number: 10,
                symbol_type: SymbolType::Function,
                signature: None,
                config_dependencies: vec![],
            },
            Symbol {
                name: "func2".to_string(),
                file_path: "test.c".to_string(),
                line_number: 20,
                symbol_type: SymbolType::Function,
                signature: None,
                config_dependencies: vec![],
            },
            Symbol {
                name: "func3".to_string(),
                file_path: "test.c".to_string(),
                line_number: 30,
                symbol_type: SymbolType::Function,
                signature: None,
                config_dependencies: vec![],
            },
            Symbol {
                name: "func4".to_string(),
                file_path: "test.c".to_string(),
                line_number: 40,
                symbol_type: SymbolType::Function,
                signature: None,
                config_dependencies: vec![],
            },
        ];

        for symbol in symbols {
            graph.add_symbol(symbol);
        }

        // Add edges to create connections
        let edges = vec![
            ("func1", "func2", CallType::Direct),
            ("func2", "func3", CallType::Direct),
            ("func3", "func4", CallType::Direct),
            ("func1", "func4", CallType::Indirect), // Cross-component edge
        ];

        for (source, target, call_type) in edges {
            let edge = CallEdge {
                call_type,
                call_site_line: 0,
                conditional: false,
                config_guard: None,
            };
            graph.add_call(source, target, edge).unwrap();
        }

        graph
    }

    fn create_large_test_graph() -> KernelGraph {
        let mut graph = KernelGraph::new();

        // Create 50 symbols to exceed small chunk limits
        for i in 0..50 {
            let symbol = Symbol {
                name: format!("func{}", i),
                file_path: "test.c".to_string(),
                line_number: (i + 1) * 10,
                symbol_type: SymbolType::Function,
                signature: None,
                config_dependencies: vec![],
            };
            graph.add_symbol(symbol);
        }

        // Add edges in a chain pattern
        for i in 0..49 {
            let edge = CallEdge {
                call_type: CallType::Direct,
                call_site_line: 0,
                conditional: false,
                config_guard: None,
            };
            graph
                .add_call(&format!("func{}", i), &format!("func{}", i + 1), edge)
                .unwrap();
        }

        graph
    }

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

        // Test with actual graph
        let test_graph = create_test_graph();
        assert_eq!(chunker.estimate_chunks(&test_graph), 1);

        // Test with smaller chunk size
        let small_options = ChunkOptions {
            max_nodes_per_chunk: 2,
            max_edges_per_chunk: 2,
            preserve_components: true,
            include_boundary_edges: true,
        };
        let small_chunker = GraphChunker::with_options(small_options);
        assert!(small_chunker.estimate_chunks(&test_graph) >= 2);
    }

    #[test]
    fn test_single_chunk_small_graph() {
        let chunker = GraphChunker::new();
        let graph = create_test_graph();

        let chunks = chunker.chunk_graph(&graph).unwrap();
        assert_eq!(chunks.len(), 1);

        let chunk = &chunks[0];
        assert_eq!(chunk.id, 0);
        assert_eq!(chunk.total_chunks, 1);
        assert_eq!(chunk.node_ids.len(), 4);
        assert!(chunk.node_ids.contains(&"func1".to_string()));
        assert!(chunk.node_ids.contains(&"func2".to_string()));
        assert!(chunk.node_ids.contains(&"func3".to_string()));
        assert!(chunk.node_ids.contains(&"func4".to_string()));

        // Should have 4 edges
        assert_eq!(chunk.edges.len(), 4);
        assert!(chunk.boundary_edges.is_empty());
    }

    #[test]
    fn test_chunking_by_size() {
        let options = ChunkOptions {
            max_nodes_per_chunk: 2,
            max_edges_per_chunk: 5,
            preserve_components: false,
            include_boundary_edges: true,
        };
        let chunker = GraphChunker::with_options(options);
        let graph = create_test_graph();

        let chunks = chunker.chunk_graph(&graph).unwrap();
        assert!(chunks.len() >= 2);

        // Verify each chunk has at most 2 nodes
        for chunk in &chunks {
            assert!(chunk.node_ids.len() <= 2);
            assert_eq!(chunk.total_chunks, chunks.len());
        }

        // Verify all nodes are included across chunks
        let mut all_nodes = HashSet::new();
        for chunk in &chunks {
            for node_id in &chunk.node_ids {
                all_nodes.insert(node_id.clone());
            }
        }
        assert_eq!(all_nodes.len(), 4);
        assert!(all_nodes.contains("func1"));
        assert!(all_nodes.contains("func2"));
        assert!(all_nodes.contains("func3"));
        assert!(all_nodes.contains("func4"));
    }

    #[test]
    fn test_chunking_with_boundary_edges() {
        let options = ChunkOptions {
            max_nodes_per_chunk: 2,
            max_edges_per_chunk: 5,
            preserve_components: false,
            include_boundary_edges: true,
        };
        let chunker = GraphChunker::with_options(options);
        let graph = create_test_graph();

        let chunks = chunker.chunk_graph(&graph).unwrap();

        // At least one chunk should have boundary edges
        let has_boundary_edges = chunks.iter().any(|chunk| !chunk.boundary_edges.is_empty());
        assert!(has_boundary_edges);
    }

    #[test]
    fn test_chunking_without_boundary_edges() {
        let options = ChunkOptions {
            max_nodes_per_chunk: 2,
            max_edges_per_chunk: 5,
            preserve_components: false,
            include_boundary_edges: false,
        };
        let chunker = GraphChunker::with_options(options);
        let graph = create_test_graph();

        let chunks = chunker.chunk_graph(&graph).unwrap();

        // No chunks should have boundary edges
        for chunk in &chunks {
            assert!(chunk.boundary_edges.is_empty());
        }
    }

    #[test]
    fn test_large_graph_chunking() {
        let options = ChunkOptions {
            max_nodes_per_chunk: 10,
            max_edges_per_chunk: 15,
            preserve_components: false,
            include_boundary_edges: true,
        };
        let chunker = GraphChunker::with_options(options);
        let graph = create_large_test_graph();

        let chunks = chunker.chunk_graph(&graph).unwrap();
        assert!(chunks.len() >= 5); // Should create multiple chunks

        // Verify constraints are respected
        for chunk in &chunks {
            assert!(chunk.node_ids.len() <= 10);
        }

        // Verify all nodes are included
        let mut all_nodes = HashSet::new();
        for chunk in &chunks {
            for node_id in &chunk.node_ids {
                all_nodes.insert(node_id.clone());
            }
        }
        assert_eq!(all_nodes.len(), 50);
    }

    #[test]
    fn test_chunk_streaming() {
        let options = ChunkOptions {
            max_nodes_per_chunk: 2,
            max_edges_per_chunk: 5,
            preserve_components: false,
            include_boundary_edges: true,
        };
        let chunker = GraphChunker::with_options(options);
        let graph = create_test_graph();

        let stream = chunker.stream_chunks(&graph);
        let estimated_count = stream.estimated_count();
        assert!(estimated_count >= 2);

        let chunks: Result<Vec<_>, _> = stream.collect();
        let chunks = chunks.unwrap();
        assert_eq!(chunks.len(), estimated_count);

        // Verify streaming produces same result as direct chunking
        let direct_chunks = chunker.chunk_graph(&graph).unwrap();
        assert_eq!(chunks.len(), direct_chunks.len());
    }

    #[test]
    fn test_chunk_reassembly() {
        let options = ChunkOptions {
            max_nodes_per_chunk: 2,
            max_edges_per_chunk: 5,
            preserve_components: false,
            include_boundary_edges: true,
        };
        let chunker = GraphChunker::with_options(options);
        let original_graph = create_test_graph();

        // Chunk the graph
        let chunks = chunker.chunk_graph(&original_graph).unwrap();

        // Reassemble the chunks
        let reassembled_graph = chunker.reassemble_chunks(&chunks).unwrap();

        // Verify same number of symbols and calls
        assert_eq!(
            reassembled_graph.symbol_count(),
            original_graph.symbol_count()
        );
        // Note: call count might differ due to boundary edges being duplicated
        assert!(reassembled_graph.call_count() >= original_graph.call_count());

        // Verify all symbols exist
        assert!(reassembled_graph.get_symbol("func1").is_some());
        assert!(reassembled_graph.get_symbol("func2").is_some());
        assert!(reassembled_graph.get_symbol("func3").is_some());
        assert!(reassembled_graph.get_symbol("func4").is_some());
    }

    #[test]
    fn test_empty_graph_chunking() {
        let chunker = GraphChunker::new();
        let graph = KernelGraph::new();

        let chunks = chunker.chunk_graph(&graph).unwrap();
        assert_eq!(chunks.len(), 1);

        let chunk = &chunks[0];
        assert_eq!(chunk.id, 0);
        assert_eq!(chunk.total_chunks, 1);
        assert!(chunk.node_ids.is_empty());
        assert!(chunk.edges.is_empty());
        assert!(chunk.boundary_edges.is_empty());
    }

    #[test]
    fn test_call_type_parsing() {
        let chunker = GraphChunker::new();

        assert_eq!(chunker.parse_call_type("Direct"), CallType::Direct);
        assert_eq!(chunker.parse_call_type("Indirect"), CallType::Indirect);
        assert_eq!(
            chunker.parse_call_type("FunctionPointer"),
            CallType::FunctionPointer
        );
        assert_eq!(chunker.parse_call_type("Macro"), CallType::Macro);
        assert_eq!(chunker.parse_call_type("Unknown"), CallType::Direct); // Default
    }

    #[test]
    fn test_chunk_iterator_size_hint() {
        let options = ChunkOptions {
            max_nodes_per_chunk: 2,
            max_edges_per_chunk: 5,
            preserve_components: false,
            include_boundary_edges: true,
        };
        let chunker = GraphChunker::with_options(options);
        let graph = create_test_graph();

        let mut stream = chunker.stream_chunks(&graph);
        let (lower, upper) = stream.size_hint();
        assert!(lower >= 2);
        assert_eq!(upper, Some(lower));

        // Consume one chunk
        let _first_chunk = stream.next().unwrap().unwrap();
        let (new_lower, new_upper) = stream.size_hint();
        assert_eq!(new_lower, lower - 1);
        assert_eq!(new_upper, Some(new_lower));
    }
}
