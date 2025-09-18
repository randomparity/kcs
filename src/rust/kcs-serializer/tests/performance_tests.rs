//! Performance and benchmarking tests for kcs-serializer
//!
//! These tests verify that serialization performance meets reasonable
//! expectations and help identify performance regressions.

use kcs_graph::{CallEdge, CallType, KernelGraph, Symbol, SymbolType};
use kcs_serializer::{
    ChunkOptions, GraphChunker, GraphExporter, GraphImporter, GraphMLExporter, JsonGraphExporter,
};
use std::time::{Duration, Instant};

/// Create a graph with specified number of nodes and edges for performance testing
fn create_performance_test_graph(node_count: usize, edge_density: f64) -> KernelGraph {
    let mut graph = KernelGraph::new();

    // Create nodes
    for i in 0..node_count {
        let symbol = Symbol {
            name: format!("perf_func_{}", i),
            file_path: format!("perf_file_{}.c", i / 100),
            line_number: (i % 1000 + 1) as u32,
            symbol_type: match i % 4 {
                0 => SymbolType::Function,
                1 => SymbolType::Variable,
                2 => SymbolType::Macro,
                _ => SymbolType::Type,
            },
            signature: if i % 3 == 0 {
                Some(format!("signature_for_func_{}", i))
            } else {
                None
            },
            config_dependencies: if i % 5 == 0 {
                vec![format!("CONFIG_{}", i)]
            } else {
                vec![]
            },
        };
        graph.add_symbol(symbol);
    }

    // Create edges based on density
    let edge_count = (node_count as f64 * edge_density) as usize;
    for i in 0..edge_count {
        let source_idx = i % node_count;
        let target_idx = (i + 1) % node_count;

        let edge = CallEdge {
            call_type: match i % 4 {
                0 => CallType::Direct,
                1 => CallType::Indirect,
                2 => CallType::FunctionPointer,
                _ => CallType::Macro,
            },
            call_site_line: (i % 1000 + 1) as u32,
            conditional: i % 7 == 0,
            config_guard: if i % 11 == 0 {
                Some(format!("CONFIG_GUARD_{}", i))
            } else {
                None
            },
        };

        graph
            .add_call(
                &format!("perf_func_{}", source_idx),
                &format!("perf_func_{}", target_idx),
                edge,
            )
            .unwrap();
    }

    graph
}

/// Measure the time taken by a function
fn measure_time<F, R>(f: F) -> (R, Duration)
where
    F: FnOnce() -> R,
{
    let start = Instant::now();
    let result = f();
    let duration = start.elapsed();
    (result, duration)
}

#[test]
fn test_json_serialization_performance() {
    let test_cases = vec![
        (100, 1.5, "small graph"),
        (500, 2.0, "medium graph"),
        (1000, 1.0, "large sparse graph"),
        (200, 5.0, "dense graph"),
    ];

    let json_exporter = JsonGraphExporter::new();

    for (node_count, edge_density, description) in test_cases {
        let graph = create_performance_test_graph(node_count, edge_density);

        println!(
            "Testing JSON serialization for {}: {} nodes, {} edges",
            description,
            graph.symbol_count(),
            graph.call_count()
        );

        // Test export performance
        let (json_string, export_time) =
            measure_time(|| json_exporter.export_to_string(&graph).unwrap());

        // Test import performance
        let (imported_graph, import_time) =
            measure_time(|| json_exporter.import_from_string(&json_string).unwrap());

        println!(
            "  Export: {:?}, Import: {:?}, Size: {} bytes",
            export_time,
            import_time,
            json_string.len()
        );

        // Performance assertions (generous limits for CI)
        assert!(
            export_time < Duration::from_millis(5000),
            "JSON export too slow for {}: {:?}",
            description,
            export_time
        );
        assert!(
            import_time < Duration::from_millis(5000),
            "JSON import too slow for {}: {:?}",
            description,
            import_time
        );

        // Verify correctness
        assert_eq!(imported_graph.symbol_count(), graph.symbol_count());
        assert_eq!(imported_graph.call_count(), graph.call_count());
    }
}

#[test]
fn test_graphml_serialization_performance() {
    let test_cases = vec![
        (100, 1.5, "small graph"),
        (500, 2.0, "medium graph"),
        (1000, 1.0, "large sparse graph"),
        (200, 5.0, "dense graph"),
    ];

    let graphml_exporter = GraphMLExporter::new();

    for (node_count, edge_density, description) in test_cases {
        let graph = create_performance_test_graph(node_count, edge_density);

        println!(
            "Testing GraphML serialization for {}: {} nodes, {} edges",
            description,
            graph.symbol_count(),
            graph.call_count()
        );

        // Test export performance
        let (graphml_string, export_time) =
            measure_time(|| graphml_exporter.export_to_string(&graph).unwrap());

        // Test import performance
        let (imported_graph, import_time) = measure_time(|| {
            graphml_exporter
                .import_from_string(&graphml_string)
                .unwrap()
        });

        println!(
            "  Export: {:?}, Import: {:?}, Size: {} bytes",
            export_time,
            import_time,
            graphml_string.len()
        );

        // Performance assertions (generous limits for CI)
        assert!(
            export_time < Duration::from_millis(10000),
            "GraphML export too slow for {}: {:?}",
            description,
            export_time
        );
        assert!(
            import_time < Duration::from_millis(10000),
            "GraphML import too slow for {}: {:?}",
            description,
            import_time
        );

        // Verify correctness
        assert_eq!(imported_graph.symbol_count(), graph.symbol_count());
        assert_eq!(imported_graph.call_count(), graph.call_count());
    }
}

#[test]
fn test_chunking_performance() {
    let graph = create_performance_test_graph(500, 2.0);

    let chunk_options = ChunkOptions {
        max_nodes_per_chunk: 50,
        max_edges_per_chunk: 100,
        preserve_components: false,
        include_boundary_edges: true,
    };
    let chunker = GraphChunker::with_options(chunk_options);

    println!(
        "Testing chunking performance: {} nodes, {} edges",
        graph.symbol_count(),
        graph.call_count()
    );

    // Test chunking performance
    let (chunks, chunk_time) = measure_time(|| chunker.chunk_graph(&graph).unwrap());

    // Test reassembly performance
    let (reassembled, reassemble_time) =
        measure_time(|| chunker.reassemble_chunks(&chunks).unwrap());

    println!(
        "  Chunking: {:?}, Reassembly: {:?}, Chunks: {}",
        chunk_time,
        reassemble_time,
        chunks.len()
    );

    // Performance assertions
    assert!(
        chunk_time < Duration::from_millis(2000),
        "Chunking too slow: {:?}",
        chunk_time
    );
    assert!(
        reassemble_time < Duration::from_millis(2000),
        "Reassembly too slow: {:?}",
        reassemble_time
    );

    // Verify correctness
    assert_eq!(reassembled.symbol_count(), graph.symbol_count());
    assert!(chunks.len() > 1, "Should create multiple chunks");
}

#[test]
fn test_streaming_performance() {
    let graph = create_performance_test_graph(300, 2.0);

    let chunk_options = ChunkOptions {
        max_nodes_per_chunk: 30,
        max_edges_per_chunk: 60,
        preserve_components: false,
        include_boundary_edges: true,
    };
    let chunker = GraphChunker::with_options(chunk_options);

    println!(
        "Testing streaming performance: {} nodes, {} edges",
        graph.symbol_count(),
        graph.call_count()
    );

    // Test streaming vs direct chunking performance
    let (direct_chunks, direct_time) = measure_time(|| chunker.chunk_graph(&graph).unwrap());

    let (streaming_chunks, streaming_time) = measure_time(|| {
        chunker
            .stream_chunks(&graph)
            .collect::<Result<Vec<_>, _>>()
            .unwrap()
    });

    println!(
        "  Direct: {:?}, Streaming: {:?}",
        direct_time, streaming_time
    );

    // Streaming might be slightly slower due to iterator overhead
    assert!(
        streaming_time < Duration::from_millis(3000),
        "Streaming too slow: {:?}",
        streaming_time
    );

    // Results should be identical
    assert_eq!(direct_chunks.len(), streaming_chunks.len());
}

#[test]
fn test_memory_efficiency() {
    // This test is more about ensuring we don't use excessive memory
    // We can't easily measure memory usage in a unit test, but we can
    // ensure operations complete for reasonably large graphs

    let large_graph = create_performance_test_graph(2000, 1.0);

    let json_exporter = JsonGraphExporter::new();
    let graphml_exporter = GraphMLExporter::new();

    println!(
        "Testing memory efficiency: {} nodes, {} edges",
        large_graph.symbol_count(),
        large_graph.call_count()
    );

    // These should complete without excessive memory usage
    let (json_string, json_time) =
        measure_time(|| json_exporter.export_to_string(&large_graph).unwrap());

    let (graphml_string, graphml_time) =
        measure_time(|| graphml_exporter.export_to_string(&large_graph).unwrap());

    println!("  JSON: {:?}, GraphML: {:?}", json_time, graphml_time);
    println!(
        "  JSON size: {} bytes, GraphML size: {} bytes",
        json_string.len(),
        graphml_string.len()
    );

    // Should complete in reasonable time
    assert!(json_time < Duration::from_secs(30));
    assert!(graphml_time < Duration::from_secs(30));

    // Sanity check on output size (shouldn't be excessive)
    assert!(json_string.len() < 100_000_000); // 100MB limit
    assert!(graphml_string.len() < 100_000_000); // 100MB limit
}

#[test]
fn test_format_size_comparison() {
    let test_graphs = vec![
        create_performance_test_graph(50, 2.0),
        create_performance_test_graph(100, 1.0),
        create_performance_test_graph(200, 3.0),
    ];

    let json_exporter = JsonGraphExporter::new();
    let graphml_exporter = GraphMLExporter::new();

    for (i, graph) in test_graphs.iter().enumerate() {
        let json_string = json_exporter.export_to_string(graph).unwrap();
        let graphml_string = graphml_exporter.export_to_string(graph).unwrap();

        println!(
            "Graph {}: {} nodes, {} edges",
            i,
            graph.symbol_count(),
            graph.call_count()
        );
        println!(
            "  JSON: {} bytes, GraphML: {} bytes",
            json_string.len(),
            graphml_string.len()
        );
        println!(
            "  Ratio: {:.2}",
            graphml_string.len() as f64 / json_string.len() as f64
        );

        // JSON is typically more compact than GraphML
        // but this can vary based on content
        assert!(!json_string.is_empty());
        assert!(!graphml_string.is_empty());
    }
}

#[test]
fn test_concurrent_performance() {
    use std::sync::Arc;
    use std::thread;

    let graph = Arc::new(create_performance_test_graph(200, 2.0));
    let json_exporter = Arc::new(JsonGraphExporter::new());

    println!("Testing concurrent serialization performance");

    let (results, concurrent_time) = measure_time(|| {
        let handles: Vec<_> = (0..4)
            .map(|_| {
                let graph = Arc::clone(&graph);
                let exporter = Arc::clone(&json_exporter);

                thread::spawn(move || exporter.export_to_string(&graph).unwrap())
            })
            .collect();

        handles
            .into_iter()
            .map(|h| h.join().unwrap())
            .collect::<Vec<_>>()
    });

    println!("  Concurrent (4 threads): {:?}", concurrent_time);

    // All results should be identical
    let first_result = &results[0];
    for result in &results[1..] {
        assert_eq!(result.len(), first_result.len());
    }

    // Should be faster than 4x sequential (due to parallelism)
    // but we'll just ensure it completes in reasonable time
    assert!(concurrent_time < Duration::from_secs(10));
}

#[test]
fn test_scalability_characteristics() {
    // Test how performance scales with graph size
    let sizes = vec![50, 100, 200, 400];
    let json_exporter = JsonGraphExporter::new();

    let mut times = Vec::new();

    for size in sizes {
        let graph = create_performance_test_graph(size, 1.5);
        let (_, export_time) = measure_time(|| json_exporter.export_to_string(&graph).unwrap());

        println!("Size {}: {:?}", size, export_time);
        times.push((size, export_time));
    }

    // Performance should scale reasonably (not exponentially)
    // Check that doubling size doesn't more than quadruple time
    for i in 1..times.len() {
        let (prev_size, prev_time) = times[i - 1];
        let (curr_size, curr_time) = times[i];

        let size_ratio = curr_size as f64 / prev_size as f64;
        let time_ratio = curr_time.as_nanos() as f64 / prev_time.as_nanos() as f64;

        println!(
            "Size ratio: {:.2}, Time ratio: {:.2}",
            size_ratio, time_ratio
        );

        // Time should scale no worse than O(n^2)
        assert!(
            time_ratio < size_ratio * size_ratio * 2.0,
            "Performance scaling too poor: {:.2}x size -> {:.2}x time",
            size_ratio,
            time_ratio
        );
    }
}
